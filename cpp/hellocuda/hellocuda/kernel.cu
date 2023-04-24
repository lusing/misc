// System includes
#include <stdio.h>
#include <assert.h>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>

using namespace std;

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

__global__ void testKernel(int val) {
    printf("[%d, %d]:\t\tValue is:%d\n", blockIdx.y * gridDim.x + blockIdx.x,
        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x,
        val);
}

int main(int argc, char** argv) {
    int devID;
    cudaDeviceProp props;

    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);

    if (cudaError == cudaSuccess) {
        cout << "There are " << deviceCount << " cuda devices." << endl;
    }

    for (int i = 0; i < deviceCount; i++)
    {
        cudaError = cudaGetDeviceProperties(&props, i);

        if (cudaError == cudaSuccess) {
            cout << "Device Name： " << props.name << endl;
            cout << "Compute Capability version: " << props.major << "." << props.minor << endl;
            cout << "设备上可用的全局内存总量:(G字节)" << props.totalGlobalMem / 1024 / 1024 / 1024 << endl;
            cout << "时钟频率（以MHz为单位）:" << props.clockRate / 1000 << endl;
            cout << "设备上多处理器的数量:" << props.multiProcessorCount << endl;
            cout << "每个块的最大线程数:" << props.maxThreadsPerBlock <<endl;
            cout << "内存总线宽度(位)" << props.memoryBusWidth << endl;
            cout << "一个块的每个维度的最大尺寸:" << props.maxThreadsDim[0] << ","<< props.maxThreadsDim[1] << "," << props.maxThreadsDim[2] << endl;
            cout << "一个网格的每个维度的最大尺寸:" << props.maxGridSize[0] << "," << props.maxGridSize[1] << "," << props.maxGridSize[2] <<endl;
        }
    }

    // Kernel configuration, where a two-dimensional grid and
    // three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel << <dimGrid, dimBlock >> > (10);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
