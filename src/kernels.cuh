#pragma once

// 本项目针对 NVIDIA P100 (sm_60, Pascal 架构) 复现并逐步优化 SGEMM
// 保留内核 1~6、9、10，删除了银行冲突解决方案（7、8）和双缓冲内核（11、12）

#include "kernels/1_naive.cuh"
#include "kernels/2_kernel_global_mem_coalesce.cuh"
#include "kernels/3_kernel_shared_mem_blocking.cuh"
#include "kernels/4_kernel_1D_blocktiling.cuh"
#include "kernels/5_kernel_2D_blocktiling.cuh"
#include "kernels/6_kernel_vectorize.cuh"
#include "kernels/9_kernel_autotuned.cuh"
#include "kernels/10_kernel_warptiling.cuh"
