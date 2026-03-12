#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <runner.cuh>
#include <vector>

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

/* 检查内核编号是否合法（0=cuBLAS，1~6，9，10） */
static bool is_valid_kernel(int k) {
  return k == 0 || (k >= 1 && k <= 6) || k == 9 || k == 10;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: sgemm <kernel_num>  (0=cuBLAS, 1-6, 9, 10)"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  /* atoi 替代 std::stoi，兼容 C++03 */
  int kernel_num = atoi(argv[1]);
  if (!is_valid_kernel(kernel_num)) {
    std::cerr << "Invalid kernel number. Valid: 0 (cuBLAS), 1-6, 9, 10"
              << std::endl;
    exit(EXIT_FAILURE);
  }

  /* 读取目标 GPU 编号 */
  int deviceIdx = 0;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

  /* 打印 GPU 信息（P100 关键参数） */
  CudaDeviceInfo();

  cublasHandle_t handle;
  if (cublasCreate(&handle)) {
    std::cerr << "Create cublas handle error." << std::endl;
    exit(EXIT_FAILURE);
  }

  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  /* 测试矩阵规模：用 push_back 替代初始化列表，兼容 C++03 */
  std::vector<int> SIZE;
  SIZE.push_back(128);
  SIZE.push_back(256);
  SIZE.push_back(512);
  SIZE.push_back(1024);
  SIZE.push_back(2048);
  SIZE.push_back(4096);

  long m, n, k, max_size;
  max_size = SIZE[SIZE.size() - 1];
  std::cout << "Max size: " << max_size << std::endl;

  float alpha = 0.5, beta = 3.0; /* GEMM 参数：C = alpha*A*B + beta*C */

  /* NULL 替代 nullptr，兼容 C++03 */
  float *A     = NULL, *B     = NULL, *C     = NULL, *C_ref = NULL;
  float *dA    = NULL, *dB    = NULL, *dC    = NULL, *dC_ref = NULL;

  A     = (float *)malloc(sizeof(float) * max_size * max_size);
  B     = (float *)malloc(sizeof(float) * max_size * max_size);
  C     = (float *)malloc(sizeof(float) * max_size * max_size);
  C_ref = (float *)malloc(sizeof(float) * max_size * max_size);

  randomize_matrix(A, max_size * max_size);
  randomize_matrix(B, max_size * max_size);
  randomize_matrix(C, max_size * max_size);

  cudaCheck(cudaMalloc((void **)&dA,    sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dB,    sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC,    sizeof(float) * max_size * max_size));
  cudaCheck(cudaMalloc((void **)&dC_ref, sizeof(float) * max_size * max_size));

  cudaCheck(cudaMemcpy(dA, A, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dB, B, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                       cudaMemcpyHostToDevice));

  int repeat_times = 50;

  /* 下标循环替代 range-based for，兼容 C++03 */
  for (int si = 0; si < (int)SIZE.size(); si++) {
    int size = SIZE[si];
    m = n = k = size;

    std::cout << "dimensions(m=n=k) " << m
              << ", alpha: " << alpha << ", beta: " << beta << std::endl;

    /* 验证正确性（与 cuBLAS 结果对比），同时消除 cold-start 误差 */
    if (kernel_num != 0) {
      run_kernel(0, m, n, k, alpha, dA, dB, beta, dC_ref, handle); /* cuBLAS */
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
      cudaCheck(cudaDeviceSynchronize());
      cudaCheck(cudaGetLastError());
      cudaMemcpy(C,     dC,     sizeof(float) * m * n, cudaMemcpyDeviceToHost);
      cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

      if (!verify_matrix(C_ref, C, m * n)) {
        std::cout << "Failed to pass the correctness verification against "
                     "NVIDIA cuBLAS." << std::endl;
        if (m <= 128) {
          std::cout << " Logging faulty output into " << errLogFile << "\n";
          std::ofstream fs;
          fs.open(errLogFile.c_str());
          fs << "A:\n";     print_matrix(A,     m, n, fs);
          fs << "B:\n";     print_matrix(B,     m, n, fs);
          fs << "C:\n";     print_matrix(C,     m, n, fs);
          fs << "Should:\n"; print_matrix(C_ref, m, n, fs);
        }
        exit(EXIT_FAILURE);
      }
    }

    /* 计时循环 */
    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      run_kernel(kernel_num, m, n, k, alpha, dA, dB, beta, dC, handle);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; /* 毫秒 → 秒 */

    long flops = 2 * m * n * k;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. size: (%ld).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time,
        m);
    fflush(stdout);

    /* 恢复 dC = dC_ref，供下一轮使用 */
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));
  }

  free(A); free(B); free(C); free(C_ref);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dC_ref);
  cublasDestroy(handle);

  return 0;
}
