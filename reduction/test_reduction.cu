#include <cstdio>
#include <cmath>
#include "reduction.cuh"
#include "cuda_check.cuh"

static void fill_input(float* arr, int N) {
    for (int i = 0; i < N; ++i)
        arr[i] = static_cast<float>(i + 1);
}

static double cpu_sum(const float* arr, int N) {
    double s = 0.0;
    for (int i = 0; i < N; ++i)
        s += static_cast<double>(arr[i]);
    return s;
}

static void run_test(int N, int numsms) {
    const int blockSize = 256;
    int numBlocks = numsms * 2;
    int maxBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > maxBlocks) numBlocks = maxBlocks;

    float* h_input = new float[N];
    fill_input(h_input, N);
    double expected = cpu_sum(h_input, N);

    float *d_input, *d_output;
    // allocate d_output with size N — wasteful but safe since numBlocks <= N always
    cudaMalloc(&d_input,  (size_t)N * sizeof(float));
    cudaMalloc(&d_output, (size_t)N * sizeof(float));

    cudaMemcpy(d_input, h_input, (size_t)N * sizeof(float), cudaMemcpyHostToDevice);
    reduceSum(numBlocks, blockSize, d_input, d_output, N);
    cudaDeviceSynchronize();

    float h_result = 0.0f;
    cudaMemcpy(&h_result, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    double got     = static_cast<double>(h_result);
    double rel_err = std::fabs(got - expected) / (std::fabs(expected) + 1e-12);
    bool   pass    = rel_err < 1e-5;

    printf("[%s] N=%-10d  got=%.6f  expected=%.6f  rel_err=%.2e\n",
           pass ? "PASS" : "FAIL", N, got, expected, rel_err);

    delete[] h_input;
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("=== Reduction Correctness Tests ===\n");
    // edge cases: single element, warp boundary, sub-warp, block boundary, sub-block, max single block
    const int sizes[] = {1, 32, 31, 256, 255, 1024, 1 << 20};

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int numsms = prop.multiProcessorCount;

    for (int N : sizes)
        run_test(N, numsms);
    return 0;
}
