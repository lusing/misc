// mataddavx.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <immintrin.h> // 包含 AVX 指令集头文件

void matrix_addition_avx(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j += 8) { // 每次处理 8 个元素（AVX 可以处理 256 位，即 8 个单精度浮点数）
            __m256 vecA = _mm256_loadu_ps(&A[i * size + j]);
            __m256 vecB = _mm256_loadu_ps(&B[i * size + j]);
            __m256 vecC = _mm256_add_ps(vecA, vecB);
            _mm256_storeu_ps(&C[i * size + j], vecC);
        }
    }
}

int main() {
    int size = 8; // 假设矩阵大小为 8x8
    float A[64] = { /* ... */ }; // 初始化矩阵 A
    float B[64] = { /* ... */ }; // 初始化矩阵 B
    float C[64] = { 0 }; // 结果矩阵 C

    matrix_addition_avx(A, B, C, size);

    // 输出结果
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << C[i * size + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
