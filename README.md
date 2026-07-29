# CUDA Parallel Reduction

A benchmark-driven implementation of sum reduction in CUDA. The project builds a two-pass GPU reduction from warp-level shuffles and shared-memory block reduction, validates it across awkward input sizes, and compares it with NVIDIA CUB.

## Highlights

- Warp-shuffle fan-in followed by shared-memory block reduction
- Fixed-grid, grid-stride first pass to keep the GPU occupied
- Second kernel to combine per-block partial sums
- Correct handling of non-power-of-two and sub-warp input sizes
- Automated correctness checks against a double-precision CPU reference
- Benchmarks against `cub::DeviceReduce::Sum`

## Benchmark snapshot

Measurements below are the median of 100 timed runs after three warmups on an NVIDIA GeForce RTX 3050 6 GB Laptop GPU. The tested launch configuration uses 256 threads per block and a grid capped at twice the number of streaming multiprocessors.

| Elements | Custom kernel | Effective bandwidth | CUB | Gap to CUB |
| ---: | ---: | ---: | ---: | ---: |
| 1M | 0.056 ms | 74.5 GB/s | 0.035 ms | 38% |
| 4M | 0.127 ms | 132.1 GB/s | 0.111 ms | 12% |
| 16M | 0.478 ms | 140.3 GB/s | 0.419 ms | 12% |
| 64M | 1.695 ms | 158.4 GB/s | 1.643 ms | 3% |

At large input sizes, both implementations become memory-bandwidth-bound and converge. The custom kernel reaches 94.3% of the GPU's stated 168 GB/s peak bandwidth at 64M elements.

Results are hardware- and toolchain-dependent. Re-run the benchmark on your own GPU before comparing implementations.

## Build and run

Requirements:

- CUDA-capable NVIDIA GPU
- NVIDIA CUDA Toolkit with `nvcc`
- GNU Make and a shell capable of running the generated executables

```bash
make run        # build and run the smoke test
make test       # run the correctness suite
make bench      # benchmark the custom kernel
make bench-cub  # benchmark CUB DeviceReduce::Sum
make clean
```

The correctness suite covers single-element, sub-warp, exact-warp, non-power-of-two, full-block, and multi-block inputs with a relative-error tolerance of `1e-5`.

## Implementation

The reduction is split into two stages:

1. Each block consumes a grid-stride slice of the input, reduces values within each warp, then combines the warp totals in shared memory.
2. A second launch reduces the per-block partial sums to one scalar.

This avoids relying on cross-block synchronization while keeping the intermediate buffer bounded by the number of active blocks.

## Repository layout

```text
.
├── Makefile
├── props.cu
└── reduction/
    ├── reduction.cu          # kernels and reduceSum launcher
    ├── main.cpp              # smoke test
    ├── test_reduction.cu     # correctness suite
    ├── bench_reduction.cu    # custom-kernel benchmark
    ├── bench_cub.cu          # CUB reference benchmark
    ├── include/
    │   ├── reduction.cuh
    │   └── cuda_check.cuh
    └── README.md             # detailed profiling notes
```

See [`reduction/README.md`](reduction/README.md) for the full benchmark tables, Nsight observations, and design notes.
