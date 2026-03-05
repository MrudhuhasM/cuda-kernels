#include <cuda_runtime.h>
#include <cstdio>


int main(){
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("No CUDA devices found.\n");
        return 1;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Total global memory: %zu bytes (%zu GB)\n", prop.totalGlobalMem, prop.totalGlobalMem / (1024 * 1024 * 1024));
        printf("  Max Grid size: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);        
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max warps per block: %d\n", prop.maxThreadsPerBlock / prop.warpSize);
        printf("  Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per block: %d\n", prop.regsPerBlock);
        printf("  Max threads per dimension: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Streaming multiprocessors (SMs): %d\n", prop.multiProcessorCount);
        printf("  Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
}
}