# Parallel Reduction (warp → block → grid)

## What it does
Two-pass parallel sum reduction over arbitrary-length float arrays.
Warp-shuffle fan-in → shared-memory block reduction → two-kernel grid reduction.

## Hardware
- GPU: NVIDIA GeForce RTX 3050 6GB Laptop GPU
- Peak memory bandwidth: 168 GB/s

## Performance
Median over 100 timed runs, 3 warmup runs discarded. Launch config: 2 × numSMs blocks, 256 threads/block.

| N        | Median (ms) | GB/s  | % of peak (168 GB/s) | CPU time (ms) | GPU speedup |
|----------|-------------|-------|----------------------|---------------|-------------|
| 1M       | 0.056       | 74.5  | 44.3%                | 0.57          | ~10×        |
| 4M       | 0.127       | 132.1 | 78.6%                | 1.97          | ~16×        |
| 16M      | 0.478       | 140.3 | 83.5%                | 8.52          | ~18×        |
| 64M      | 1.695       | 158.4 | 94.3%                | 34.52         | ~20×        |

Bandwidth scales toward peak as N grows — at small N the kernel is latency-bound (fixed launch overhead),
at large N it is fully memory-bandwidth-bound.

## Correctness
Automated test suite (`make test`) covers 7 cases:

| Case                        | Result |
|-----------------------------|--------|
| N=1 (single element)        | PASS   |
| N=31 (sub-warp, non-pow2)   | PASS   |
| N=32 (exactly one warp)     | PASS   |
| N=255 (sub-block, non-pow2) | PASS   |
| N=256 (exactly one block)   | PASS   |
| N=1024 (max single block)   | PASS   |
| N=1M (multi-block)          | PASS   |

Tolerance: `rel_err < 1e-5`. CPU reference computed in `double` to avoid FP bias.

## vs Reference
### CUB `DeviceReduce::Sum`

| N   | Custom median (ms) | Custom GB/s | CUB median (ms) | CUB GB/s | CUB % of peak | Gap  |
|-----|--------------------|-------------|-----------------|----------|---------------|------|
| 1M  | 0.056              | 74.5        | 0.035           | 120.5    | 71.7%         | −38% |
| 4M  | 0.127              | 132.1       | 0.111           | 150.5    | 89.6%         | −12% |
| 16M | 0.478              | 140.3       | 0.419           | 160.3    | 95.4%         | −12% |
| 64M | 1.695              | 158.4       | 1.643           | 163.3    | 97.2%         | −3%  |

At large N both implementations are fully bandwidth-bound and converge (−3% gap at 64M).
The gap is largest at small N (−38% at 1M) where CUB benefits from a single-pass kernel
that avoids the second-pass launch overhead and uses a more optimal thread/block config.

- `torch.sum`: correctness verified, rel_err < 1e-5

## Nsight evidence
- Regime: memory-bandwidth-bound
- DRAM throughput: 87.6% of peak at N=1M (Nsight) vs 64.0 GB/s (event timer) — event timer includes kernel launch overhead
- Dominant stall: Long Scoreboard (89% of warp cycles) — warps waiting on DRAM loads
- Compute utilization: 8% — confirms memory-bound diagnosis

## Key design choices
- Fixed grid (2 × numSMs blocks) with stride loop — saturates SMs without oversizing pass-2
- `MAX_WARPS=32` shared memory — supports up to 1024 threads/block
- Two-pass reduction — no cross-block sync primitive exists in CUDA; pass-1 writes per-block partial sums, pass-2 reduces them in a single block
- `numBlocks` clamped to `ceil(N / blockSize)` — prevents idle blocks from writing zeros into the output buffer on small inputs

## Files
| File | Purpose |
|------|---------|
| `reduction.cu` | Kernel implementations + `reduceSum` launcher |
| `include/reduction.cuh` | Public API declaration |
| `include/cuda_check.cuh` | `CUDA_CHECK` error-checking macro (shared across all files) |
| `main.cpp` | Smoke-test entry point (1M elements, pass/fail) |
| `test_reduction.cu` | Automated correctness harness (`make test`) |
| `bench_reduction.cu` | Benchmark harness, 100-run median, GB/s, CPU comparison (`make bench`) |
| `bench_cub.cu` | CUB `DeviceReduce::Sum` reference benchmark (`make bench-cub`) |

## Build
```bash
make run        # build + run smoke test
make test       # build + run correctness suite
make bench      # build + run custom kernel benchmark
make bench-cub  # build + run CUB DeviceReduce::Sum benchmark
make clean
```