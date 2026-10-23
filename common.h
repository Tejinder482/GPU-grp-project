// common.h
// Things shared by nbody.cu and barnes_hut.cu: reading/writing the particle
// files, CUDA error checking and a host timer.

#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <vector>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

struct SimParams {
    int   n;      // number of particles
    int   steps;  // number of time steps
    float dt;
    float G;
    float eps;    // softening, added to |r|^2 exactly like in the assignment formula
};

// Positions are stored as float4 (x, y, z, mass) and velocities as
// float4 (vx, vy, vz, unused). The 4th component keeps every load 16 bytes,
// which the GPU can do in one instruction.

inline double nowMs()
{
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Input file (plain text):
//   N steps dt G eps          <- first non-comment line
//   x y z vx vy vz m          <- then one line per particle
// Lines starting with '#' are skipped.
inline bool readInput(const char* path, SimParams& p,
                      std::vector<float4>& pos, std::vector<float4>& vel)
{
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "could not open %s\n", path);
        return false;
    }

    char line[512];
    bool haveHeader = false;
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        haveHeader = (sscanf(line, "%d %d %f %f %f",
                             &p.n, &p.steps, &p.dt, &p.G, &p.eps) == 5);
        break;
    }
    if (!haveHeader || p.n <= 0 || p.steps < 0) {
        fprintf(stderr, "bad header in %s (expected: N steps dt G eps)\n", path);
        fclose(f);
        return false;
    }

    pos.resize(p.n);
    vel.resize(p.n);
    int i = 0;
    while (i < p.n && fgets(line, sizeof(line), f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        float x, y, z, vx, vy, vz, m;
        if (sscanf(line, "%f %f %f %f %f %f %f", &x, &y, &z, &vx, &vy, &vz, &m) != 7) {
            fprintf(stderr, "could not parse particle %d in %s\n", i, path);
            fclose(f);
            return false;
        }
        pos[i] = make_float4(x, y, z, m);
        vel[i] = make_float4(vx, vy, vz, 0.0f);
        i++;
    }
    fclose(f);

    if (i != p.n) {
        fprintf(stderr, "header says %d particles but only %d were found\n", p.n, i);
        return false;
    }
    return true;
}

// Output uses the same format as the input, so a result can be fed back in
// to continue the simulation.
inline bool writeOutput(const char* path, const SimParams& p,
                        const std::vector<float4>& pos, const std::vector<float4>& vel)
{
    FILE* f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "could not write %s\n", path);
        return false;
    }
    fprintf(f, "# state after %d steps\n", p.steps);
    fprintf(f, "%d %d %g %g %g\n", p.n, p.steps, p.dt, p.G, p.eps);
    for (int i = 0; i < p.n; i++) {
        fprintf(f, "%.8e %.8e %.8e %.8e %.8e %.8e %.8e\n",
                pos[i].x, pos[i].y, pos[i].z, vel[i].x, vel[i].y, vel[i].z, pos[i].w);
    }
    fclose(f);
    return true;
}
