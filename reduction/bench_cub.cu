#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cub/device/device_reduce.cuh>
#include "cuda_check.cuh"

static const float PEAK_BW_GBS = 168.0f;
static const int   WARMUP_RUNS = 3;
static const int   TIMED_RUNS  = 100;

static void bench_cub(int N) {
    // Host input: all ones
    float* h_input = new float[N];
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input,  (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, (size_t)N * sizeof(float), cudaMemcpyHostToDevice));

    // Query temp storage size
    void*  d_temp  = nullptr;
    size_t temp_bytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_output, N));
    CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

    // Warmup
    for (int w = 0; w < WARMUP_RUNS; ++w) {
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_output, N));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(TIMED_RUNS);
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    for (int r = 0; r < TIMED_RUNS; ++r) {
        CUDA_CHECK(cudaEventRecord(ev_start));
        CUDA_CHECK(cub::DeviceReduce::Sum(d_temp, temp_bytes, d_input, d_output, N));
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

    double bytes = static_cast<double>(N) * sizeof(float);
    double gbs   = (bytes / 1.0e9) / (median_ms / 1.0e3);
    double pct   = 100.0 * gbs / PEAK_BW_GBS;

    // Verify result
    float h_result = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    printf("N=%-12d  median=%7.3f ms  mean=%7.3f ms  %6.1f GB/s  %5.1f%% peak  result=%.0f\n",
           N, median_ms, mean_ms, gbs, pct, static_cast<double>(h_result));

    delete[] h_input;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_temp));
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== CUB DeviceReduce::Sum Benchmark  |  GPU: %s  |  peak BW: %.0f GB/s ===\n",
           prop.name, PEAK_BW_GBS);
    printf("%-14s  %-18s  %-18s  %-12s  %-11s  %s\n",
           "N", "median", "mean", "GB/s", "% peak", "result");
    printf("--------------------------------------------------------------------------------\n");

    const int sizes[] = {1 << 20, 1 << 22, 1 << 24, 1 << 26};
    for (int N : sizes)
        bench_cub(N);

    return 0;
}
