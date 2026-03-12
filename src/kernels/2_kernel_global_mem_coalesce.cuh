#pragma once
/*
 * 内核 2：全局内存合并访问（Global Memory Coalescing）
 *
 * 优化思路：
 *   将线程索引重新映射为 1D，使相邻线程访问连续的内存地址，
 *   从而触发硬件的内存合并（coalescing），减少内存事务数量。
 *
 * P100 适配说明：
 *   参数保持不变（BLOCKSIZE=32）。
 *   P100 的 HBM2 内存控制器对合并访问友好，此内核能有效利用带宽，
 *   但仍属于 Memory Bound 阶段。
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha,
                                          const float *A, const float *B,
                                          float beta, float *C) {
  const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  // if statement is necessary to make things work under tile quantization
  if (cRow < M && cCol < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[cRow * K + i] * B[i * N + cCol];
    }
    C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
  }
}