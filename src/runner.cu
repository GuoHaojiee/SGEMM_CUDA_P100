#include "kernels.cuh"
#include "runner.cuh"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>

float get_sec() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return (1e6 * time.tv_sec + time.tv_usec);
}

float cpu_elapsed_time(float &beg, float &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

// 打印 P100 GPU 关键参数，包括共享内存、寄存器、L2 缓存等信息
void CudaDeviceInfo() {
  int deviceId;
  cudaGetDevice(&deviceId);

  cudaDeviceProp props;
  memset(&props, 0, sizeof(props));
  cudaGetDeviceProperties(&props, deviceId);

  printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    l2CacheSize: %dKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
         deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
         props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
         props.regsPerBlock, props.regsPerMultiprocessor,
         props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
         props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
         props.l2CacheSize / 1024,
         props.multiProcessorCount, props.warpSize);
  // P100 关键规格参考：
  //   - 16GB HBM2，理论带宽 ~720 GB/s
  //   - 10.6 TFLOPS FP32（单精度）
  //   - 每 SM 共享内存 48KB，寄存器 65536 个
  //   - 56 个 SM，Warp size 32
  //   - sm_60，不支持 __ldg 以外的 cache hint（无 L2 prefetch 指令）
};

void randomize_matrix(float *mat, int N) {
  // 使用 gettimeofday 而非 srand(time(NULL))，精度更高，避免重复随机数
  struct timeval time;
  memset(&time, 0, sizeof(time));
  gettimeofday(&time, NULL);
  srand(time.tv_usec);
  for (int i = 0; i < N; i++) {
    float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
    tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
    mat[i] = tmp;
  }
}

void range_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = i;
  }
}

void zero_init_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    mat[i] = 0.0;
  }
}

void copy_matrix(const float *src, float *dest, int N) {
  int i;
  for (i = 0; src + i && dest + i && i < N; i++)
    *(dest + i) = *(src + i);
  if (i != N)
    printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs) {
  int i;
  fs << std::setprecision(2)
     << std::fixed;
  fs << "[";
  for (i = 0; i < M * N; i++) {
    if ((i + 1) % N == 0)
      fs << std::setw(5) << A[i];
    else
      fs << std::setw(5) << A[i] << ", ";
    if ((i + 1) % N == 0) {
      if (i + 1 < M * N)
        fs << ";\n";
    }
  }
  fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N; i++) {
    diff = std::fabs(matRef[i] - matOut[i]);
    if (isnan(diff) || diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return false;
    }
  }
  return true;
}

int div_ceil(int numerator, int denominator) {
  std::div_t res = std::div(numerator, denominator);
  return res.rem ? (res.quot + 1) : res.quot;
}

// cuBLAS FP32 基准（列主序，通过转置等价实现行主序 A*B）
void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha,
                   float *A, float *B, float beta, float *C) {
  /* CUDA 10.2 兼容：computeType 用 CUDA_R_32F，algo 用 CUBLAS_GEMM_DEFAULT */
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F,
               N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUDA_R_32F,
               CUBLAS_GEMM_DEFAULT);
}

// ─────────────────────────────────────────────
// 内核 1：Naive（朴素实现）
// 每个线程计算 C 矩阵的一个元素，无任何优化
// ─────────────────────────────────────────────
void run_sgemm_naive(int M, int N, int K, float alpha, float *A, float *B,
                     float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32, 32);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// ─────────────────────────────────────────────
// 内核 2：全局内存合并访问
// 1D 线程布局，提升 DRAM 访问合并效率
// ─────────────────────────────────────────────
void run_sgemm_coalesce(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  sgemm_global_mem_coalesce<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// ─────────────────────────────────────────────
// 内核 3：共享内存分块缓存（SMEM blocking）
// 将 A、B 的子块加载到共享内存，减少全局内存访问
// ─────────────────────────────────────────────
void run_sgemm_shared_mem_block(int M, int N, int K, float alpha, float *A,
                                float *B, float beta, float *C) {
  dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
  dim3 blockDim(32 * 32);
  // 将 L1 cache 全部划给共享内存
  cudaFuncSetAttribute(sgemm_shared_mem_block<32>,
                       cudaFuncAttributePreferredSharedMemoryCarveout,
                       cudaSharedmemCarveoutMaxShared);
  sgemm_shared_mem_block<32>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// ─────────────────────────────────────────────
// 内核 4：1D 块级 tiling（BM=64, BN=64, BK=8, TM=8）
// 每个线程负责 TM 个输出元素，提升寄存器利用率
// ─────────────────────────────────────────────
void runSgemm1DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;
  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  sgemm1DBlocktiling<BM, BN, BK, TM>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// ─────────────────────────────────────────────
// 内核 5：2D 块级 tiling（BM=128, BN=128, BK=8, TM=8, TN=8）
// 每线程计算 TM×TN=64 个元素，大幅提升算术强度
// ─────────────────────────────────────────────
void runSgemm2DBlocktiling(int M, int N, int K, float alpha, float *A, float *B,
                           float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemm2DBlocktiling<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

// ─────────────────────────────────────────────
// 内核 6：向量化内存访问（float4）
// 在内核 5 基础上，加载时使用 float4 向量化指令，并转置 A 优化访存
// ─────────────────────────────────────────────
void runSgemmVectorize(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  const uint BK = 8;
  const uint TM = 8;
  const uint TN = 8;
  if (M >= 128 and N >= 128) {
    const uint BM = 128;
    const uint BN = 128;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  } else {
    const uint BM = 64;
    const uint BN = 64;
    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));
    sgemmVectorize<BM, BN, BK, TM, TN>
        <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
  }
}

// ─────────────────────────────────────────────
// 内核 9：自动调优参数（P100 适配版）
// 综合前序所有优化，针对 P100 调整 tiling 参数：
//   BM=128, BN=128, BK=8（P100 HBM2 高带宽，BK 适当减小减少 SMEM 压力）
//   TM=8, TN=8（每线程 64 个输出元素）
//   共享内存占用：(128×8 + 8×128)×4 = 8KB，远低于 P100 的 48KB 限制
// P100 适配说明：相比 A6000 将 BK 从 16 减小到 8，
//   减少了单次 SMEM 占用，有助于提升 SM 占用率
// ─────────────────────────────────────────────
void runSgemmAutotuned(int M, int N, int K, float alpha, float *A, float *B,
                       float beta, float *C) {
  // P100 调优参数
  const uint K9_BK = 8;
  const uint K9_TM = 8;
  const uint K9_TN = 8;
  const uint K9_BM = 128;
  const uint K9_BN = 128;
  dim3 blockDim(K9_NUM_THREADS);
  /* static_assert 已移除（需要 C++11）；参数正确性已手工验证 */

  dim3 gridDim(CEIL_DIV(N, K9_BN), CEIL_DIV(M, K9_BM));
  sgemmAutotuned<K9_BM, K9_BN, K9_BK, K9_TM, K9_TN>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// ─────────────────────────────────────────────
// 内核 10：Warp-level tiling（P100 适配版）
// 在线程块 tiling 的基础上，将工作进一步分配给 Warp：
//   BM=64, BN=64, BK=16 —— 块级分块
//   WM=32, WN=32, WNITER=2 —— Warp 级分块
//   TM=4, TN=4 —— 每线程寄存器分块
//   NUM_THREADS=128（4 个 Warp）
// P100 适配说明：相比 A6000 缩小了块级分块（128→64），
//   减少每块共享内存使用（8KB vs 32KB），
//   适合 P100 的 Warp 调度策略（56 SM，每 SM 64 个 Warp 上限）
// ─────────────────────────────────────────────
void runSgemmWarptiling(int M, int N, int K, float alpha, float *A, float *B,
                        float beta, float *C) {
  // P100 调优参数
  const uint K10_NUM_THREADS = 128;
  const uint K10_BN = 64;
  const uint K10_BM = 64;
  const uint K10_BK = 16;
  const uint K10_WN = 32;
  const uint K10_WM = 32;
  const uint K10_WNITER = 2;
  const uint K10_TN = 4;
  const uint K10_TM = 4;
  dim3 blockDim(K10_NUM_THREADS);

  /* constexpr → const，兼容 C++03；static_assert 已移除（需要 C++11） */
  const uint NUM_WARPS = K10_NUM_THREADS / 32;
  const uint K10_WMITER =
      (K10_WM * K10_WN) / (32 * K10_TM * K10_TN * K10_WNITER);

  dim3 gridDim(CEIL_DIV(N, K10_BN), CEIL_DIV(M, K10_BM));
  sgemmWarptiling<K10_BM, K10_BN, K10_BK, K10_WM, K10_WN, K10_WNITER, K10_TM,
                  K10_TN, K10_NUM_THREADS>
      <<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
}

// 内核分发：根据 kernel_num 调用对应的 SGEMM 实现
// 有效内核：0（cuBLAS）、1、2、3、4、5、6、9、10
void run_kernel(int kernel_num, int M, int N, int K, float alpha, float *A,
                float *B, float beta, float *C, cublasHandle_t handle) {
  switch (kernel_num) {
  case 0:
    runCublasFP32(handle, M, N, K, alpha, A, B, beta, C);
    break;
  case 1:
    run_sgemm_naive(M, N, K, alpha, A, B, beta, C);
    break;
  case 2:
    run_sgemm_coalesce(M, N, K, alpha, A, B, beta, C);
    break;
  case 3:
    run_sgemm_shared_mem_block(M, N, K, alpha, A, B, beta, C);
    break;
  case 4:
    runSgemm1DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 5:
    runSgemm2DBlocktiling(M, N, K, alpha, A, B, beta, C);
    break;
  case 6:
    runSgemmVectorize(M, N, K, alpha, A, B, beta, C);
    break;
  case 9:
    runSgemmAutotuned(M, N, K, alpha, A, B, beta, C);
    break;
  case 10:
    runSgemmWarptiling(M, N, K, alpha, A, B, beta, C);
    break;
  default:
    printf("[ERROR] Unknown kernel number %d. Valid: 0-6, 9, 10\n", kernel_num);
    exit(EXIT_FAILURE);
  }
}
