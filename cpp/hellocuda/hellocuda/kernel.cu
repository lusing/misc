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

__global__ void testKernel(float val) {
    printf("[%d, %d]:\t\tValue is:%f\n", blockIdx.y * gridDim.x + blockIdx.x,
        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x,
        __sinf(val* threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
            threadIdx.x));
}

__global__ void sqrtKernel(float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = sqrtf(in[i]);
        printf("[%d, %d]:\t\tValue is:%f\n", blockIdx.x, threadIdx.x, out[i]);
    }
}

int main(int argc, char** argv) {
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
            //props.block
        }
    }

    // Kernel configuration, where a two-dimensional grid and
    // three-dimensional blocks are configured.
    dim3 dimGrid(2, 2);
    dim3 dimBlock(2, 2, 3);
    testKernel << <dimGrid, dimBlock >> > (0.5);
    cudaDeviceSynchronize();

    const int n = 1024;
    size_t size = n * sizeof(float);
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    float* d_in, * d_out;

    // Initialize input array
    for (int i = 0; i < n; ++i) {
        h_in[i] = (float)i;
    }

    // Allocate device memory
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    cout << blocksPerGrid << " " << threadsPerBlock << endl;
    sqrtKernel << <blocksPerGrid, threadsPerBlock >> > (d_in, d_out, n);

    // Copy output data to host
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    // Verify results
    for (int i = 0; i < n; ++i) {
        if (fabsf(h_out[i] - sqrtf(h_in[i])) > 1e-5) {
            printf("Error: h_out[%d] = %f, sqrtf(h_in[%d]) = %f\n", i, h_out[i], i, sqrtf(h_in[i]));
        }
    }

    printf("Success!\n");

    // Free memory
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return EXIT_SUCCESS;
}
