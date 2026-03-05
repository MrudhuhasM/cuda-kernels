#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include "reduction.cuh"
#include "cuda_check.cuh"

static const float PEAK_BW_GBS = 168.0f;
static const int   WARMUP_RUNS = 3;
static const int   TIMED_RUNS  = 100;

// Returns elapsed time in ms; result returned to prevent dead-code elimination.
static double cpu_reduce(const float* arr, int N, double* out_ms) {
    auto t0 = std::chrono::high_resolution_clock::now();
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += static_cast<double>(arr[i]);
    auto t1 = std::chrono::high_resolution_clock::now();
    *out_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return s;
}

static void bench(int N, int numSMs) {
    const int blockSize = 256;
    int numBlocks = numSMs * 2;
    int maxBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > maxBlocks) numBlocks = maxBlocks;

    // Host buffer (all ones for a clean reference)
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  (size_t)N        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, (size_t)numBlocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup — reduceSum syncs internally, no extra sync needed
    for (int w = 0; w < WARMUP_RUNS; ++w)
        reduceSum(numBlocks, blockSize, d_input, d_output, N);

    // Timed runs — reduceSum syncs internally before ev_stop is recorded,
    // so cudaEventSynchronize is a no-op but kept for clarity
    std::vector<float> times(TIMED_RUNS);
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int r = 0; r < TIMED_RUNS; ++r) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        reduceSum(numBlocks, blockSize, d_input, d_output, N);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));
        CUDA_CHECK(cudaEventElapsedTime(&times[r], ev_start, ev_stop));
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    std::sort(times.begin(), times.end());
    float median_ms = times[TIMED_RUNS / 2];
    float mean_ms   = 0.0f;
    for (float t : times) mean_ms += t;
    mean_ms /= TIMED_RUNS;

    // Effective bandwidth: each element read once
    double bytes   = static_cast<double>(N) * sizeof(float);
    double gbs     = (bytes / 1.0e9) / (median_ms / 1.0e3);
    double pct     = 100.0 * gbs / PEAK_BW_GBS;

    // CPU baseline (single-threaded); cpu_res is printed to prevent dead-code elimination
    double cpu_ms  = 0.0;
    double cpu_res = cpu_reduce(h_input, N, &cpu_ms);

    printf("N=%-12d  median=%7.3f ms  mean=%7.3f ms  %6.1f GB/s  %5.1f%% peak  CPU=%7.3f ms  cpu_sum=%.0f\n",
           N, median_ms, mean_ms, gbs, pct, cpu_ms, cpu_res);

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Reduction Benchmark  |  GPU: %s  |  peak BW: %.0f GB/s ===\n",
           prop.name, PEAK_BW_GBS);
    printf("%-14s  %-18s  %-18s  %-12s  %-11s  %s\n",
           "N", "median", "mean", "GB/s", "% peak", "CPU time");
    printf("--------------------------------------------------------------------------------\n");

    // 1M, 4M, 16M, 64M elements
    const int sizes[] = {1 << 20, 1 << 22, 1 << 24, 1 << 26};
    for (int N : sizes)
        bench(N, prop.multiProcessorCount);

    return 0;
}
