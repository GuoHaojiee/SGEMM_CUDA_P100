#pragma once
/*
 * 内核 1：朴素 SGEMM（Naive）
 *
 * 优化思路：
 *   最简单的实现，每个线程负责计算输出矩阵 C 中的一个元素。
 *   直接从全局内存读取 A 和 B，完全没有数据复用。
 *
 * P100 适配说明：
 *   参数保持不变（32×32 线程块）。
 *   受限于全局内存带宽和访存不规则，性能极低，作为基准参考。
 *   在 P100 上瓶颈为 Memory Bound（HBM2 带宽利用率低）。
 *
 * Matrix sizes: MxK * KxN = MxN
 */

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // if statement is necessary to make things work under tile quantization
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}