
#include <cuda_runtime.h>
#include <cstdio>


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

void reduceSum(const float* input, float* output, int size) {
    
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    int numsms = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    numsms = prop.multiProcessorCount;

    int blockSize = 256;
    int numBlocks = numsms * 2; // 2 blocks per SM for good occupancy
    if (numBlocks > (size + blockSize - 1) / blockSize) {
        numBlocks = (size + blockSize - 1) / blockSize;
    }

    cudaEvent_t startK1, stopK1, startK2, stopK2;
    cudaEventCreate(&startK1);
    cudaEventCreate(&stopK1);
    cudaEventCreate(&startK2);
    cudaEventCreate(&stopK2);

    
    const int warmupRuns = 3;
    for (int w = 0; w < warmupRuns; ++w) {
        reduceSumKernel<<<numBlocks, blockSize>>>(input, output, size);
        if (numBlocks > 1) {
            reduceFinalKernel<<<1, blockSize>>>(output, numBlocks);
        }
        cudaDeviceSynchronize();
    }

    
    cudaEventRecord(startK1);
    reduceSumKernel<<<numBlocks, blockSize>>>(input, output, size);
    cudaEventRecord(stopK1);
    cudaEventSynchronize(stopK1);

    float ms1 = 0.0f;
    cudaEventElapsedTime(&ms1, startK1, stopK1);
    printf("[Kernel 1 - reduceSumKernel]  blocks=%d  time=%.3f ms\n", numBlocks, ms1);

    
    if (numBlocks > 1) {
        cudaEventRecord(startK2);
        reduceFinalKernel<<<1, blockSize>>>(output, numBlocks);
        cudaEventRecord(stopK2);
        cudaEventSynchronize(stopK2);

        float ms2 = 0.0f;
        cudaEventElapsedTime(&ms2, startK2, stopK2);
        printf("[Kernel 2 - reduceFinalKernel] blocks=1   time=%.3f ms\n", ms2);
        printf("[Total kernel time]                         %.3f ms\n", ms1 + ms2);
    }

    cudaEventDestroy(startK1);
    cudaEventDestroy(stopK1);
    cudaEventDestroy(startK2);
    cudaEventDestroy(stopK2);
}