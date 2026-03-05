
#include <cuda_runtime.h>
#include "cuda_check.cuh"


__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


__device__ float blockReduceSum(float val) {
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    const int MAX_WARPS = 32; 
    __shared__ float warpSums[MAX_WARPS];

    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();
    val = (lane < numWarps) ? warpSums[lane] : 0.0f;
    if (warpId == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

__global__ void reduceSumKernel(const float* __restrict__ input, float* __restrict__ output, int size) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += gridSize) {
        sum += input[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sum;
    }
}

__global__ void reduceFinalKernel(float* __restrict__ data, int size) {
    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // should be within single block
    int gridSize = blockDim.x * gridDim.x;
    for (int i = idx; i < size; i += gridSize) {
        sum += data[i];
    }
    sum = blockReduceSum(sum);
    if (threadIdx.x == 0) {
        data[0] = sum;
    }
}

void reduceSum(const int numBlocks, const int blockSize, const float* input, float* output, int size) {
    reduceSumKernel<<<numBlocks, blockSize>>>(input, output, size);
    CUDA_CHECK(cudaGetLastError());
    if (numBlocks > 1) {
        reduceFinalKernel<<<1, blockSize>>>(output, numBlocks);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}