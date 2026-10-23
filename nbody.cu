// nbody.cu
// Direct O(N^2) gravitational N-body simulation.
//
// Modes:
//   naive  one GPU thread per particle, every thread reads all N positions
//          straight from global memory
//   tiled  same mapping, but a block loads the positions into shared memory
//          one tile at a time and all its threads reuse that tile
//   cpu    simple double precision loop on the host (reference + baseline)
//
// Build: nvcc -O3 -arch=native -o nbody nbody.cu
// Run:   ./nbody input.txt output.txt [naive|tiled|cpu] [block_size]

#include <cuda_runtime.h>
#include <cmath>
#include <string>
#include "common.h"

// ---------------------------------------------------------------------------
// GPU part
// ---------------------------------------------------------------------------

// Adds the pull of body j on body i to a.
// From the assignment: F_i = G m_i * sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps)^(3/2)
// We need F_i / m_i anyway, so m_i cancels and is never used. G is applied
// once at the end instead of N times inside the loop.
// For j == i the vector r is zero, so the term adds nothing (eps > 0 stops 0/0).
__device__ __forceinline__ float3 bodyBody(float4 bi, float4 bj, float3 a, float eps)
{
    float rx = bj.x - bi.x;
    float ry = bj.y - bi.y;
    float rz = bj.z - bi.z;
    float d2 = rx * rx + ry * ry + rz * rz + eps;
    float inv = rsqrtf(d2);          // 1/sqrt in one hardware instruction
    float s = bj.w * inv * inv * inv; // m_j / d2^(3/2)
    a.x += rx * s;
    a.y += ry * s;
    a.z += rz * s;
    return a;
}

// v(t+1) = v(t) + a dt,  r(t+1) = r(t) + v(t+1) dt   (as given in the assignment)
__device__ __forceinline__ void integrate(int i, float4 bi, float3 a, float4* posNew,
                                          float4* vel, float G, float dt)
{
    float4 v = vel[i];
    v.x += G * a.x * dt;
    v.y += G * a.y * dt;
    v.z += G * a.z * dt;
    bi.x += v.x * dt;
    bi.y += v.y * dt;
    bi.z += v.z * dt;
    vel[i] = v;
    posNew[i] = bi; // w (the mass) is carried over unchanged
}

// Positions are double buffered: every thread reads the old positions and
// writes its new position to the other buffer. Otherwise a fast thread could
// move its particle while a slower one is still computing forces from it.
// Each thread only touches its own velocity, so vel can be updated in place.

__global__ void stepNaive(const float4* posOld, float4* posNew, float4* vel,
                          int n, float G, float eps, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float4 bi = posOld[i];
    float3 a = make_float3(0.0f, 0.0f, 0.0f);
    for (int j = 0; j < n; j++)
        a = bodyBody(bi, posOld[j], a, eps);

    integrate(i, bi, a, posNew, vel, G, dt);
}

// TILE == blockDim.x. It is a template parameter so the inner loop has a
// compile-time trip count and can be unrolled.
template <int TILE>
__global__ void stepTiled(const float4* __restrict__ posOld, float4* __restrict__ posNew,
                          float4* __restrict__ vel, int n, float G, float eps, float dt)
{
    __shared__ float4 tile[TILE];

    int i = blockIdx.x * TILE + threadIdx.x;

    // Threads past the end of the array can't return early: they still have
    // to load their part of each tile and reach both __syncthreads().
    float4 bi = (i < n) ? posOld[i] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float3 a = make_float3(0.0f, 0.0f, 0.0f);

    for (int base = 0; base < n; base += TILE) {
        // every thread brings in one body, so one global read per thread per tile
        int j = base + threadIdx.x;
        // the last tile can be partial; pad it with zero-mass bodies (no force)
        tile[threadIdx.x] = (j < n) ? posOld[j] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        __syncthreads();

#pragma unroll 8
        for (int k = 0; k < TILE; k++)
            a = bodyBody(bi, tile[k], a, eps);

        __syncthreads(); // nobody may overwrite the tile while others still read it
    }

    if (i < n) integrate(i, bi, a, posNew, vel, G, dt);
}

static void launchStep(bool tiled, int bs, int blocks, const float4* in, float4* out,
                       float4* vel, const SimParams& p)
{
    if (!tiled) {
        stepNaive<<<blocks, bs>>>(in, out, vel, p.n, p.G, p.eps, p.dt);
        return;
    }
    switch (bs) {
        case 32:   stepTiled<32><<<blocks, 32>>>(in, out, vel, p.n, p.G, p.eps, p.dt); break;
        case 64:   stepTiled<64><<<blocks, 64>>>(in, out, vel, p.n, p.G, p.eps, p.dt); break;
        case 128:  stepTiled<128><<<blocks, 128>>>(in, out, vel, p.n, p.G, p.eps, p.dt); break;
        case 256:  stepTiled<256><<<blocks, 256>>>(in, out, vel, p.n, p.G, p.eps, p.dt); break;
        case 512:  stepTiled<512><<<blocks, 512>>>(in, out, vel, p.n, p.G, p.eps, p.dt); break;
        case 1024: stepTiled<1024><<<blocks, 1024>>>(in, out, vel, p.n, p.G, p.eps, p.dt); break;
    }
}

// Runs the whole simulation on the GPU.
// loopMs  = time of the time-step loop only (kernels)
// totalMs = loop + copying the data to the GPU and back
static void runGPU(bool tiled, int bs, const SimParams& p, std::vector<float4>& pos,
                   std::vector<float4>& vel, float& loopMs, float& totalMs)
{
    const int n = p.n;
    const size_t bytes = (size_t)n * sizeof(float4);
    const int blocks = (n + bs - 1) / bs;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU          : %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    float4 *dPos[2], *dVel;
    CUDA_CHECK(cudaMalloc(&dPos[0], bytes));
    CUDA_CHECK(cudaMalloc(&dPos[1], bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));

    // Warm-up launch on zeroed buffers, so the one-time cost of loading the
    // kernel (lazy module loading in CUDA 12) is not counted in the timing.
    CUDA_CHECK(cudaMemset(dPos[0], 0, bytes));
    CUDA_CHECK(cudaMemset(dVel, 0, bytes));
    launchStep(tiled, bs, blocks, dPos[0], dPos[1], dVel, p);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evLoop0, evLoop1, evEnd;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evLoop0));
    CUDA_CHECK(cudaEventCreate(&evLoop1));
    CUDA_CHECK(cudaEventCreate(&evEnd));

    CUDA_CHECK(cudaEventRecord(evStart));
    CUDA_CHECK(cudaMemcpy(dPos[0], pos.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, vel.data(), bytes, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(evLoop0));
    int cur = 0;
    for (int s = 0; s < p.steps; s++) {
        launchStep(tiled, bs, blocks, dPos[cur], dPos[1 - cur], dVel, p);
        cur = 1 - cur; // swap the position buffers
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(evLoop1));

    CUDA_CHECK(cudaMemcpy(pos.data(), dPos[cur], bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vel.data(), dVel, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(evEnd));
    CUDA_CHECK(cudaEventSynchronize(evEnd));

    CUDA_CHECK(cudaEventElapsedTime(&loopMs, evLoop0, evLoop1));
    CUDA_CHECK(cudaEventElapsedTime(&totalMs, evStart, evEnd));

    cudaEventDestroy(evStart);
    cudaEventDestroy(evLoop0);
    cudaEventDestroy(evLoop1);
    cudaEventDestroy(evEnd);
    cudaFree(dPos[0]);
    cudaFree(dPos[1]);
    cudaFree(dVel);
}

// ---------------------------------------------------------------------------
// CPU reference (double precision, single thread)
// ---------------------------------------------------------------------------

static void runCPU(const SimParams& p, std::vector<float4>& pos, std::vector<float4>& vel)
{
    const int n = p.n;
    std::vector<double> x(n), y(n), z(n), vx(n), vy(n), vz(n), m(n);
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        x[i] = pos[i].x;  y[i] = pos[i].y;  z[i] = pos[i].z;  m[i] = pos[i].w;
        vx[i] = vel[i].x; vy[i] = vel[i].y; vz[i] = vel[i].z;
    }

    for (int s = 0; s < p.steps; s++) {
        // all forces first, from the positions at time t
        for (int i = 0; i < n; i++) {
            double sx = 0.0, sy = 0.0, sz = 0.0;
            for (int j = 0; j < n; j++) {
                double rx = x[j] - x[i], ry = y[j] - y[i], rz = z[j] - z[i];
                double d2 = rx * rx + ry * ry + rz * rz + p.eps;
                double inv3 = 1.0 / (d2 * std::sqrt(d2));
                sx += m[j] * rx * inv3;
                sy += m[j] * ry * inv3;
                sz += m[j] * rz * inv3;
            }
            ax[i] = p.G * sx;
            ay[i] = p.G * sy;
            az[i] = p.G * sz;
        }
        // then move everyone
        for (int i = 0; i < n; i++) {
            vx[i] += ax[i] * p.dt;  vy[i] += ay[i] * p.dt;  vz[i] += az[i] * p.dt;
            x[i] += vx[i] * p.dt;   y[i] += vy[i] * p.dt;   z[i] += vz[i] * p.dt;
        }
    }

    for (int i = 0; i < n; i++) {
        pos[i] = make_float4((float)x[i], (float)y[i], (float)z[i], pos[i].w);
        vel[i] = make_float4((float)vx[i], (float)vy[i], (float)vz[i], 0.0f);
    }
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("usage: %s input.txt output.txt [naive|tiled|cpu] [block_size]\n", argv[0]);
        return 1;
    }
    const char* inFile = argv[1];
    const char* outFile = argv[2];
    std::string mode = (argc > 3) ? argv[3] : "tiled";
    int bs = (argc > 4) ? atoi(argv[4]) : 256;

    if (mode != "naive" && mode != "tiled" && mode != "cpu") {
        fprintf(stderr, "unknown mode '%s'\n", mode.c_str());
        return 1;
    }
    if (mode == "tiled" && bs != 32 && bs != 64 && bs != 128 && bs != 256 && bs != 512 && bs != 1024) {
        fprintf(stderr, "tiled mode needs block size 32, 64, 128, 256, 512 or 1024\n");
        return 1;
    }
    if (mode == "naive" && (bs < 32 || bs > 1024 || bs % 32 != 0)) {
        fprintf(stderr, "block size should be a multiple of 32, at most 1024\n");
        return 1;
    }

    SimParams p;
    std::vector<float4> pos, vel;
    if (!readInput(inFile, p, pos, vel)) return 1;
    if (p.eps <= 0.0f) {
        fprintf(stderr, "eps must be > 0 (the i == j term would be 0/0)\n");
        return 1;
    }

    printf("mode         : %s", mode.c_str());
    if (mode != "cpu") printf(" (block size %d)", bs);
    printf("\nN = %d, steps = %d, dt = %g, G = %g, eps = %g\n", p.n, p.steps, p.dt, p.G, p.eps);

    float loopMs = 0.0f, totalMs = 0.0f;
    if (mode == "cpu") {
        double t0 = nowMs();
        runCPU(p, pos, vel);
        loopMs = totalMs = (float)(nowMs() - t0);
        bs = 0;
    } else {
        runGPU(mode == "tiled", bs, p, pos, vel, loopMs, totalMs);
    }

    if (!writeOutput(outFile, p, pos, vel)) return 1;

    // 20 floating point operations per interaction is the usual count for this
    // kernel (rsqrt counted as 4), same as in GPU Gems 3, ch. 31
    double interactions = (double)p.n * p.n * p.steps;
    double msPerStep = p.steps > 0 ? loopMs / p.steps : 0.0;
    double gflops = loopMs > 0 ? 20.0 * interactions / (loopMs * 1e6) : 0.0;

    printf("time-step loop : %.3f ms  (%.4f ms per step)\n", loopMs, msPerStep);
    printf("incl. copies   : %.3f ms\n", totalMs);
    printf("throughput     : %.1f GFLOP/s, %.3e interactions/s\n", gflops,
           loopMs > 0 ? interactions / (loopMs * 1e-3) : 0.0);
    printf("output written to %s\n", outFile);

    // one line that is easy to parse from the notebook
    printf("[summary] mode=%s n=%d steps=%d block=%d time_ms=%.4f total_ms=%.4f "
           "ms_per_step=%.5f gflops=%.2f\n",
           mode.c_str(), p.n, p.steps, bs, loopMs, totalMs, msPerStep, gflops);
    return 0;
}
