#include <iostream>
#include <cmath>
#include <chrono>
#include "reduction.cuh"
#include "cuda_check.cuh"


void fill_array(float *array, int size) {
    for (int i = 0; i < size; ++i) {
        array[i] = static_cast<float>(i + 1); 
    }
}

bool check_result(float *input, float *result, int size) {
    double expected = 0.0;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < size; ++i) {
        expected += static_cast<double>(input[i]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end - start;
    std::cout << "CPU reduction time: " << cpu_time.count() << " ms\n";
    double got = static_cast<double>(result[0]);
    double rel_err = std::abs(got - expected) / (std::abs(expected) + 1e-12);
    return rel_err < 1e-6; // relative tolerance
}

int main(){

    int N = 1 << 20; // 1M elements

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    int numsms = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    numsms = prop.multiProcessorCount;

    int blockSize = 256;
    int numBlocks = numsms * 2; // 2 blocks per SM for good occupancy
    if (numBlocks > (N + blockSize - 1) / blockSize) {
        numBlocks = (N + blockSize - 1) / blockSize;
    }

    
    float *h_input = new float[N];
    float *h_output = new float[numBlocks];

    float *d_input, *d_output;

    CUDA_CHECK(cudaMalloc(&d_input,  N        * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, numBlocks * sizeof(float)));

    fill_array(h_input, N);
    CUDA_CHECK(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));
    reduceSum(numBlocks, blockSize, d_input, d_output, N);  // syncs internally
    CUDA_CHECK(cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    if (check_result(h_input, h_output, N)) {
        std::cout << "Reduction successful! Result: " << h_output[0] << std::endl;
    } else {
        double expected = static_cast<double>(N) * (static_cast<double>(N) + 1.0) / 2.0;
        std::cerr << "Reduction failed! Expected: " << expected << ", Got: " << h_output[0] << std::endl;
    }
    

    delete[] h_input;
    delete[] h_output;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}