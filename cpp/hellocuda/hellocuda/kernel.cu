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

    cudaDeviceProp deviceProp;
    int deviceCount;
    cudaError_t cudaError;
    cudaError = cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; i++)
    {
        cudaError = cudaGetDeviceProperties(&deviceProp, i);

        cout << "设备 " << i + 1 << " 的主要属性： " << endl;
        cout << "设备显卡型号： " << deviceProp.name << endl;
        cout << "设备全局内存总量（以MB为单位）： " << deviceProp.totalGlobalMem / 1024 / 1024 << endl;
        cout << "设备上一个线程块（Block）中可用的最大共享内存（以KB为单位）： " << deviceProp.sharedMemPerBlock / 1024 << endl;
        cout << "设备上一个线程块（Block）种可用的32位寄存器数量： " << deviceProp.regsPerBlock << endl;
        cout << "设备上一个线程块（Block）可包含的最大线程数量： " << deviceProp.maxThreadsPerBlock << endl;
        cout << "设备的计算功能集（Compute Capability）的版本号： " << deviceProp.major << "." << deviceProp.minor << endl;
        cout << "设备上多处理器的数量： " << deviceProp.multiProcessorCount << endl;
    }

    // Get GPU information
    //checkCudaErrors(cudaGetDevice(&devID));
    //checkCudaErrors(cudaGetDeviceProperties(&props, devID));
    //printf("Device %d: \"%s\" with Compute %d.%d capability\n", devID, props.name,
    //    props.major, props.minor);

    //printf("printf() is called. Output:\n\n");

    // Kernel configuration, where a two-dimensional grid and
    // three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 2);
    testKernel << <dimGrid, dimBlock >> > (10);
    cudaDeviceSynchronize();

    return EXIT_SUCCESS;
}
