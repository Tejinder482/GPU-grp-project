// barnes_hut.cu
// Barnes-Hut version of the simulation (the extension part).
//
// Every time step:
//   1. copy the current positions back to the host
//   2. build the octree on the CPU and flatten it (octree.h)
//   3. copy the flattened tree to the GPU
//   4. one GPU thread per particle walks the tree, gets its acceleration and
//      does the same velocity/position update as the direct code
//
// A far-away cell is replaced by one body at its centre of mass when
//   cell width / distance < theta
// theta = 0 opens every cell, which gives back the exact direct sum.
//
// Building the tree on the CPU keeps the code simple. A full GPU tree build
// (Morton codes + sorting) would remove the copies but is a lot more work.
//
// Build: nvcc -O3 -arch=native -o barnes_hut barnes_hut.cu
// Run:   ./barnes_hut input.txt output.txt [theta] [block_size]

#include <cuda_runtime.h>
#include <cmath>
#include "common.h"
#include "octree.h"

// Stackless walk over the flattened tree (see octree.h).
// nInter counts how many cells/bodies were actually used, just for statistics.
__device__ float3 treeAccel(float4 bi, const float4* __restrict__ com,
                            const float* __restrict__ width, const int* __restrict__ child,
                            const int* __restrict__ next, int numNodes, float theta2,
                            float eps, int& nInter)
{
    float3 a = make_float3(0.0f, 0.0f, 0.0f);
    int k = 0;
    while (k < numNodes) {
        float4 c = com[k];
        float rx = c.x - bi.x;
        float ry = c.y - bi.y;
        float rz = c.z - bi.z;
        float r2 = rx * rx + ry * ry + rz * rz;
        float w = width[k];

        // compare squares so we don't need a sqrt for the test
        if (child[k] < 0 || w * w < theta2 * r2) {
            // leaf, or far enough: use the centre of mass, same formula as before.
            // (the particle's own leaf gives r = 0, so it adds nothing)
            float d2 = r2 + eps;
            float inv = rsqrtf(d2);
            float s = c.w * inv * inv * inv;
            a.x += rx * s;
            a.y += ry * s;
            a.z += rz * s;
            nInter++;
            k = next[k];   // skip the whole subtree
        } else {
            k = child[k];  // too close, go inside the cell
        }
    }
    return a;
}

// update = 0 only computes accelerations (used once at the start for the
// accuracy check). inter can be null.
__global__ void bhStep(const int* __restrict__ order, float4* pos, float4* vel, float4* acc,
                       int* inter, const float4* __restrict__ com,
                       const float* __restrict__ width, const int* __restrict__ child,
                       const int* __restrict__ next, int n, int numNodes, float theta2,
                       float G, float eps, float dt, int update)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;

    // Threads take particles in tree order instead of input order. Then the 32
    // threads of a warp hold particles that are close together, they open
    // almost the same cells, and the warp diverges much less.
    int i = order[t];
    float4 bi = pos[i];

    int nInter = 0;
    float3 a = treeAccel(bi, com, width, child, next, numNodes, theta2, eps, nInter);
    a.x *= G;
    a.y *= G;
    a.z *= G;
    acc[i] = make_float4(a.x, a.y, a.z, 0.0f);
    if (inter) inter[i] = nInter;
    if (!update) return;

    float4 v = vel[i];
    v.x += a.x * dt;
    v.y += a.y * dt;
    v.z += a.z * dt;
    bi.x += v.x * dt;
    bi.y += v.y * dt;
    bi.z += v.z * dt;
    vel[i] = v;
    pos[i] = bi; // in place is fine here: during the walk threads read the tree, not pos
}

// device copy of the flattened tree, grows when the tree gets bigger
struct DeviceTree {
    float4* com = nullptr;
    float* width = nullptr;
    int* child = nullptr;
    int* next = nullptr;
    int capacity = 0;

    void upload(const FlatTree& t)
    {
        int m = t.numNodes();
        if (m > capacity) {
            release();
            capacity = m + m / 2;
            CUDA_CHECK(cudaMalloc(&com, capacity * sizeof(float4)));
            CUDA_CHECK(cudaMalloc(&width, capacity * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&child, capacity * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&next, capacity * sizeof(int)));
        }
        CUDA_CHECK(cudaMemcpy(com, t.com.data(), m * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(width, t.width.data(), m * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(child, t.child.data(), m * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(next, t.next.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    }

    void release()
    {
        cudaFree(com);
        cudaFree(width);
        cudaFree(child);
        cudaFree(next);
        com = nullptr; width = nullptr; child = nullptr; next = nullptr;
        capacity = 0;
    }
};

// Relative error of the Barnes-Hut accelerations against the exact direct sum,
// done in double on the CPU. The exact sum costs O(N) per particle, so only
// `samples` particles (spread evenly over the array) are checked.
static void forceError(const std::vector<float4>& pos, const std::vector<float4>& acc,
                       const SimParams& p, int samples, double& meanErr, double& maxErr)
{
    const int n = p.n;
    const int m = std::min(n, samples);
    meanErr = 0.0;
    maxErr = 0.0;
    for (int s = 0; s < m; s++) {
        int i = (int)((long long)s * n / m);
        double ax = 0.0, ay = 0.0, az = 0.0;
        for (int j = 0; j < n; j++) {
            double rx = (double)pos[j].x - pos[i].x;
            double ry = (double)pos[j].y - pos[i].y;
            double rz = (double)pos[j].z - pos[i].z;
            double d2 = rx * rx + ry * ry + rz * rz + p.eps;
            double inv3 = 1.0 / (d2 * std::sqrt(d2));
            ax += pos[j].w * rx * inv3;
            ay += pos[j].w * ry * inv3;
            az += pos[j].w * rz * inv3;
        }
        ax *= p.G;
        ay *= p.G;
        az *= p.G;
        double ex = acc[i].x - ax, ey = acc[i].y - ay, ez = acc[i].z - az;
        double ref = std::sqrt(ax * ax + ay * ay + az * az);
        double err = std::sqrt(ex * ex + ey * ey + ez * ez) / (ref > 0.0 ? ref : 1.0);
        meanErr += err;
        maxErr = std::max(maxErr, err);
    }
    meanErr /= m;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("usage: %s input.txt output.txt [theta] [block_size]\n", argv[0]);
        return 1;
    }
    const char* inFile = argv[1];
    const char* outFile = argv[2];
    float theta = (argc > 3) ? (float)atof(argv[3]) : 0.5f;
    int bs = (argc > 4) ? atoi(argv[4]) : 128;
    if (theta < 0.0f) {
        fprintf(stderr, "theta must be >= 0\n");
        return 1;
    }
    if (bs < 32 || bs > 1024 || bs % 32 != 0) {
        fprintf(stderr, "block size should be a multiple of 32, at most 1024\n");
        return 1;
    }

    SimParams p;
    std::vector<float4> pos, vel;
    if (!readInput(inFile, p, pos, vel)) return 1;
    if (p.eps <= 0.0f) {
        fprintf(stderr, "eps must be > 0\n");
        return 1;
    }

    const int n = p.n;
    const size_t bytes = (size_t)n * sizeof(float4);
    const int blocks = (n + bs - 1) / bs;
    const float theta2 = theta * theta;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("mode         : barnes-hut (theta %.2f, block size %d)\n", theta, bs);
    printf("GPU          : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("N = %d, steps = %d, dt = %g, G = %g, eps = %g\n", n, p.steps, p.dt, p.G, p.eps);

    float4 *dPos, *dVel, *dAcc;
    int *dOrder, *dInter;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));
    CUDA_CHECK(cudaMalloc(&dOrder, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dInter, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dPos, pos.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, vel.data(), bytes, cudaMemcpyHostToDevice));

    Octree tree;
    FlatTree flat;
    DeviceTree dTree;

    // --- accuracy check on the initial positions (also serves as warm-up) ---
    tree.build(pos);
    tree.flatten(flat);
    dTree.upload(flat);
    CUDA_CHECK(cudaMemcpy(dOrder, flat.order.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    bhStep<<<blocks, bs>>>(dOrder, dPos, dVel, dAcc, dInter, dTree.com, dTree.width,
                           dTree.child, dTree.next, n, flat.numNodes(), theta2,
                           p.G, p.eps, p.dt, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float4> acc(n);
    std::vector<int> inter(n);
    CUDA_CHECK(cudaMemcpy(acc.data(), dAcc, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(inter.data(), dInter, n * sizeof(int), cudaMemcpyDeviceToHost));
    double avgInter = 0.0;
    for (int i = 0; i < n; i++) avgInter += inter[i];
    avgInter /= n;

    double errMean, errMax;
    forceError(pos, acc, p, 1000, errMean, errMax);
    printf("tree         : %d nodes, depth %d, %.1f interactions per particle (direct: %d)\n",
           flat.numNodes(), tree.depth(), avgInter, n);
    printf("force error  : mean %.3e, max %.3e (relative, vs exact sum)\n", errMean, errMax);

    // --- timed time-step loop ---
    cudaEvent_t k0, k1;
    CUDA_CHECK(cudaEventCreate(&k0));
    CUDA_CHECK(cudaEventCreate(&k1));
    double buildMs = 0.0, kernelMs = 0.0;

    double t0 = nowMs();
    for (int s = 0; s < p.steps; s++) {
        CUDA_CHECK(cudaMemcpy(pos.data(), dPos, bytes, cudaMemcpyDeviceToHost));

        double b0 = nowMs();
        tree.build(pos, &flat.order); // insert in last step's tree order (see octree.h)
        tree.flatten(flat);
        buildMs += nowMs() - b0;

        dTree.upload(flat);
        CUDA_CHECK(cudaMemcpy(dOrder, flat.order.data(), n * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(k0));
        bhStep<<<blocks, bs>>>(dOrder, dPos, dVel, dAcc, nullptr, dTree.com, dTree.width,
                               dTree.child, dTree.next, n, flat.numNodes(), theta2,
                               p.G, p.eps, p.dt, 1);
        CUDA_CHECK(cudaEventRecord(k1));
        CUDA_CHECK(cudaEventSynchronize(k1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, k0, k1));
        kernelMs += ms;
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double loopMs = nowMs() - t0;

    CUDA_CHECK(cudaMemcpy(pos.data(), dPos, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel.data(), dVel, bytes, cudaMemcpyDeviceToHost));
    if (!writeOutput(outFile, p, pos, vel)) return 1;

    double otherMs = loopMs - buildMs - kernelMs; // copies + launch overhead
    double msPerStep = p.steps > 0 ? loopMs / p.steps : 0.0;
    printf("time-step loop : %.3f ms  (%.4f ms per step)\n", loopMs, msPerStep);
    printf("  tree build (CPU) : %.3f ms\n", buildMs);
    printf("  tree walk  (GPU) : %.3f ms\n", kernelMs);
    printf("  copies + other   : %.3f ms\n", otherMs);
    printf("output written to %s\n", outFile);
    printf("[summary] mode=bh n=%d steps=%d block=%d theta=%.3f time_ms=%.4f "
           "ms_per_step=%.5f build_ms=%.4f kernel_ms=%.4f other_ms=%.4f nodes=%d "
           "depth=%d avg_inter=%.2f err_mean=%.4e err_max=%.4e\n",
           n, p.steps, bs, theta, loopMs, msPerStep, buildMs, kernelMs, otherMs,
           flat.numNodes(), tree.depth(), avgInter, errMean, errMax);

    dTree.release();
    cudaEventDestroy(k0);
    cudaEventDestroy(k1);
    cudaFree(dPos);
    cudaFree(dVel);
    cudaFree(dAcc);
    cudaFree(dOrder);
    cudaFree(dInter);
    return 0;
}
