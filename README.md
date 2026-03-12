# P100 SGEMM 从零实现与逐步优化

本项目是在 **NVIDIA Tesla P100（sm\_60, Pascal 架构）** 上从零手写并逐步优化单精度矩阵乘法（SGEMM）的个人复现项目。
参考自 [siboehm/SGEMM\_CUDA](https://github.com/siboehm/SGEMM_CUDA)，针对 P100 的硬件特性重新调整了 tiling 参数，并删除了在 P100 上不适用的内核（双缓冲 K11/K12、银行冲突解决 K7/K8）。

---

## 硬件环境

| 规格 | 参数 |
|---|---|
| GPU | NVIDIA Tesla P100 SXM2 16GB |
| 架构 | Pascal（sm\_60） |
| 显存 | 16 GB HBM2 |
| 显存带宽 | ~720 GB/s |
| FP32 算力 | 10.6 TFLOPS |
| SM 数量 | 56 |
| 每 SM 共享内存 | 48 KB |
| 每 SM 寄存器 | 65536 |
| Warp Size | 32 |
| 特殊说明 | sm\_60 不支持 `__ldg` 以外的 cache hint，无 L2 prefetch 指令 |

---

## 各内核优化思路

### 内核 1：朴素实现（Naive）
每个线程计算输出矩阵 C 中的一个元素，直接读取全局内存中的 A 和 B，没有任何数据复用。
属于纯 Memory Bound 场景，性能极低，作为优化基准。

### 内核 2：全局内存合并访问（GMEM Coalescing）
将 2D 线程索引重映射为 1D，使相邻线程访问连续的内存地址，触发硬件内存合并（coalescing）。
P100 的 HBM2 内存控制器对合并访问友好，此步可将有效带宽提升数倍。

### 内核 3：共享内存分块缓存（SMEM Blocking）
将 A 和 B 的 32×32 子块加载到共享内存（SRAM）中缓存，利用数据局部性实现 K 方向的复用。
共享内存读取延迟远低于全局内存，此内核开始向 Compute Bound 过渡。

### 内核 4：1D 块级寄存器 Tiling（1D Block Tiling）
每个线程计算 C 的一列中连续的 TM=8 个元素，通过寄存器数组缓存中间结果。
寄存器复用减少了共享内存读取次数，算术强度进一步提升。
使用参数：BM=64，BN=64，BK=8，TM=8。

### 内核 5：2D 块级寄存器 Tiling（2D Block Tiling）
将 tiling 扩展到 M 和 N 两个维度，每线程计算 TM×TN=64 个输出元素。
使用 regM、regN 两个寄存器数组分别缓存 A 和 B 的片段，进一步减少 SMEM 访问。
使用参数：BM=128，BN=128，BK=8，TM=8，TN=8。

### 内核 6：向量化内存访问（Vectorized float4）
在内核 5 基础上，使用 `float4` 指令（128bit）批量加载 A 和 B，减少内存事务数。
加载 A 时同步完成转置存储，使后续按列读取时地址连续，充分利用 P100 的 HBM2 带宽。
使用参数与内核 5 相同：BM=128，BN=128，BK=8，TM=8，TN=8。

### 内核 9：自动调优参数（Autotuned，P100 版）
综合使用 SMEM 分块、2D 寄存器 tiling、float4 向量化，通过模板参数实现编译期调优。
针对 P100 将 BK 从 A6000 的 16 调整为 8，降低每块共享内存占用（8 KB），提升 SM 占用率。
使用参数：BM=128，BN=128，**BK=8**，TM=8，TN=8，256 线程/块。

### 内核 10：Warp-level Tiling（P100 版）
在线程块 tiling 的基础上，将工作细分给 Warp，每个 Warp 独立处理一个 WM×WN 子矩阵。
通过 `loadFromGmem` 和 `processFromSmem` 两阶段解耦，使加载和计算逻辑更清晰。
针对 P100 缩小块级分块（BM/BN 从 128 改为 64），适配 P100 的 Warp 调度策略。
使用参数：BM=64，BN=64，BK=16，WM=32，WN=32，WNITER=2，TM=4，TN=4，128 线程/块。

---

## Roofline 分析

P100 的 Roofline 分界点（Ridge Point）约为：

```
算力 / 带宽 = 10,600 GFLOPS / 720 GB/s ≈ 14.7 FLOP/Byte
```

对 SGEMM，算术强度为 `2*M*N*K / (bytes_read + bytes_write)`。典型的 Memory Bound → Compute Bound 转变节点：

| 内核 | 算术强度（估算） | 瓶颈 |
|---|---|---|
| K1 Naive | ~0.5 FLOP/B | Memory Bound |
| K2 GMEM Coalesce | ~1 FLOP/B | Memory Bound |
| K3 SMEM Block | ~8 FLOP/B | 接近转变点 |
| K4 1D Tiling | ~16 FLOP/B | Compute Bound |
| K5~K10 | >16 FLOP/B | Compute Bound |

从内核 3 到内核 4 是典型的 Memory Bound → Compute Bound 转变节点。K5 以后的内核差异主要体现在指令级效率（向量化、Warp 调度）上。

---

## 编译与运行

### 环境要求

- CUDA Toolkit 11.x 或以上
- CMake ≥ 3.19
- 支持 sm\_60 的 GPU（P100）

### 编译

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### 运行单个内核

```bash
# 运行内核 K（有效值：0=cuBLAS, 1, 2, 3, 4, 5, 6, 9, 10）
./build/sgemm <K>

# 示例：运行内核 10（Warptiling）
./build/sgemm 10
```

### 批量性能测试

```bash
# 自动编译 + 运行所有内核 + 生成性能图表
python3 benchmark.py
```

图表保存为 `benchmark_result.png`。

---

## Benchmark 结果

> 运行 `python3 benchmark.py` 后生成。

![benchmark_result](benchmark_result.png)

---

## 与 siboehm 原项目的主要差异

| 方面 | 原项目（A6000, sm\_86） | 本项目（P100, sm\_60） |
|---|---|---|
| 目标 GPU | NVIDIA RTX A6000 | NVIDIA Tesla P100 |
| Compute Capability | sm\_86 (Ampere) | sm\_60 (Pascal) |
| CMakeLists 架构 | `CUDA_COMPUTE_CAPABILITY 86` | `CUDA_COMPUTE_CAPABILITY 60` |
| 保留内核 | K1~K12 | K1~K6, K9, K10 |
| 删除内核 | — | K7/K8（银行冲突）、K11/K12（双缓冲） |
| K9 BK 参数 | BK=16 | **BK=8**（减小 SMEM 压力） |
| K10 块级分块 | BM=BN=128 | **BM=BN=64**（适配 P100 Warp 调度） |
| K10 Warp 分块 | WM=WN=64, WNITER=4 | **WM=WN=32, WNITER=2** |
| K10 TM | TM=8 | **TM=4** |
| 删除双缓冲原因 | 支持 sm\_86 的异步拷贝 | P100 sm\_60 不支持 `cuda::memcpy_async` |
| 语言 / README | 英文 | 中文 |
