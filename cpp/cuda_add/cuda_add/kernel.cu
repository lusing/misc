
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>

__global__ void matrixAdd2Kernel(float* A, float* B, float* C, int N, int M) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < M) {
        C[row * M + col] = A[row * M + col] + B[row * M + col];
    }
}

void matrixAdd2(float* A, float* B, float* C, int N, int M) {
    int size = N * M * sizeof(float);
    float* d_A, * d_B, * d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_B, size);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_C, size);

    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);

    matrixAdd2Kernel << <gridDim, blockDim >> > (d_A, d_B, d_C, N, M);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main2() {
    int N = 1024;
    int M = 1024;

    float* A = new float[N * M];
    float* B = new float[N * M];
    float* C = new float[N * M];

    // Initialize A and B matrices
    for (int i = 0; i < N * M; i++) {
        A[i] = static_cast<float>(i);
        B[i] = static_cast<float>(i * 2);
    }

    matrixAdd2(A, B, C, N, M);

    // Print a portion of the result matrix for validation
    for (int i = 0; i < 10; i++) {
        std::cout << C[i] << " ";
    }
    std::cout << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}


// 矩阵加法的CUDA核函数
__global__ void matrixAdd10(int* A, int* B, int* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        C[row * width + col] = A[row * width + col] + B[row * width + col];
    }
}

int main() {
    // 矩阵维度
    int width = 4;

    // 分配CPU内存
    int* A, * B, * C;
    A = (int*)malloc(width * width * sizeof(int));
    B = (int*)malloc(width * width * sizeof(int));
    C = (int*)malloc(width * width * sizeof(int));

    // 初始化A和B矩阵
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            A[i * width + j] = i;
            B[i * width + j] = j;
        }
    }

    // 为GPU矩阵分配内存
    int* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, width * width * sizeof(int));
    cudaMalloc((void**)&d_B, width * width * sizeof(int));
    cudaMalloc((void**)&d_C, width * width * sizeof(int));

    // 将矩阵从CPU内存复制到GPU内存
    cudaMemcpy(d_A, A, width * width * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, width * width * sizeof(int), cudaMemcpyHostToDevice);

    // 配置CUDA核函数参数
    dim3 threads(width, width);
    dim3 grid(1, 1);
    matrixAdd10 <<<grid, threads >>> (d_A, d_B, d_C, width);

    // 等待CUDA核函数执行完毕
    cudaDeviceSynchronize();

    // 将结果从GPU内存复制到CPU内存
    cudaMemcpy(C, d_C, width * width * sizeof(int), cudaMemcpyDeviceToHost);

    // 验证结果
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            if (C[i * width + j] != i + j) {
                printf("错误!");
                return 0;
            }
        }
    }
    printf("矩阵加法成功!");

    // 释放CPU和GPU内存
    free(A); free(B); free(C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}
