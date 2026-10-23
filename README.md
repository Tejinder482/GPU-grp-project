# N-body simulation with CUDA

Gravitational N-body simulation on the GPU:

- direct O(N²) method with two kernels: a naive one and a shared-memory tiled one
- a CPU (double precision) version used as reference and baseline
- Barnes-Hut approximation (octree built on the CPU, tree walk on the GPU) as the extension

## Files

| File | Contents |
|---|---|
| `nbody.cu` | direct O(N²) simulation: `naive`, `tiled` and `cpu` modes |
| `barnes_hut.cu` | Barnes-Hut simulation + force error check against the exact sum |
| `octree.h` | octree build and flattening used by `barnes_hut.cu` |
| `common.h` | input/output, error checking, timer |
| `gen_input.py` | generates random input files |
| `sample_input.txt` | small example input (512 particles, 100 steps) |
| `nbody_colab.ipynb` | builds everything, runs the tests and benchmarks, makes the graphs |
| `Makefile` | build |

## Build

```
make
```

or by hand:

```
nvcc -O3 -arch=native -o nbody nbody.cu
nvcc -O3 -arch=native -o barnes_hut barnes_hut.cu
```

If your nvcc doesn't know `-arch=native`, give the compute capability instead, e.g. `make ARCH=-arch=sm_75` for a T4.

## Input format

```
N steps dt G eps
x y z vx vy vz m        <- N lines, one per particle
```

Lines starting with `#` are ignored. `eps` is added to |r|² like in the formula from the assignment, so it has to be > 0.

Random input: `python3 gen_input.py 4096 in.txt --steps 100 --dt 1e-3 --G 1 --eps 1e-4`

## Run

```
./nbody      in.txt out.txt [naive|tiled|cpu] [block_size]     # default: tiled 256
./barnes_hut in.txt out.txt [theta] [block_size]               # default: 0.5 128
```

`out.txt` has the final positions and velocities in the same format as the input (so it can be used as input again).
The execution time is printed: time of the time-step loop, time including host/device copies, and GFLOP/s.
`barnes_hut` also prints the split between tree build, tree walk and copies, the average number of
interactions per particle, and the relative force error against the exact direct sum.

The last line of the output (`[summary] ...`) has everything as `key=value` pairs, the notebook reads that.

## Running on Colab

1. Open `nbody_colab.ipynb` in Colab, Runtime → Change runtime type → T4 GPU.
2. Run all, and upload `nbody_cuda.zip` (this folder, zipped) when asked.

It takes a few minutes. At the end `results.zip` is downloaded with the CSV files, the graphs and the sample input/output.
