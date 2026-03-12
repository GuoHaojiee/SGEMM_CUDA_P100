#!/usr/bin/env python3
"""
benchmark.py —— P100 SGEMM 内核性能基准测试脚本

功能：
  1. 自动编译项目（cmake + make）
  2. 依次运行内核 0（cuBLAS）、1~6、9、10，各测 3 种矩阵规模
  3. 解析输出中的 GFLOPS 数据
  4. 用 matplotlib 画出性能对比折线图（x 轴：矩阵规模，y 轴：GFLOPS）
  5. 保存为 benchmark_result.png
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt

# ── 配置 ──────────────────────────────────────────────────────────────────────

# 项目根目录（脚本所在目录）
PROJECT_ROOT = Path(__file__).parent.resolve()
BUILD_DIR = PROJECT_ROOT / "build"

# 测试的矩阵规模
SIZES = [1024, 2048, 4096]

# 测试的内核编号及名称（0 = cuBLAS 基准）
KERNELS = {
    0:  "cuBLAS (baseline)",
    1:  "K1: Naive",
    2:  "K2: GMEM Coalesce",
    3:  "K3: SMEM Block",
    4:  "K4: 1D Tiling",
    5:  "K5: 2D Tiling",
    6:  "K6: Vectorize",
    9:  "K9: Autotuned",
    10: "K10: Warptiling",
}

# ── 编译 ──────────────────────────────────────────────────────────────────────

def build_project():
    """使用 cmake + make 编译项目，失败则退出。"""
    print("=" * 60)
    print("编译项目 ...")
    BUILD_DIR.mkdir(exist_ok=True)

    # cmake 配置
    cmake_cmd = ["cmake", "-DCMAKE_BUILD_TYPE=Release", str(PROJECT_ROOT)]
    ret = subprocess.run(cmake_cmd, cwd=BUILD_DIR, capture_output=False)
    if ret.returncode != 0:
        print("[ERROR] cmake 配置失败", file=sys.stderr)
        sys.exit(1)

    # make 编译
    make_cmd = ["make", "-j", str(os.cpu_count() or 4)]
    ret = subprocess.run(make_cmd, cwd=BUILD_DIR, capture_output=False)
    if ret.returncode != 0:
        print("[ERROR] make 编译失败", file=sys.stderr)
        sys.exit(1)

    print("编译完成。")
    print("=" * 60)


# ── 运行内核 ──────────────────────────────────────────────────────────────────

def run_kernel(kernel_num: int, size: int) -> float | None:
    """
    运行指定内核，解析并返回该规模下的 GFLOPS。
    返回 None 表示解析失败。
    """
    sgemm_bin = BUILD_DIR / "sgemm"
    if not sgemm_bin.exists():
        print(f"[ERROR] 找不到可执行文件：{sgemm_bin}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    # 让程序只跑到目标 size（sgemm.cu 会遍历所有 SIZE，我们只取目标行）
    result = subprocess.run(
        [str(sgemm_bin), str(kernel_num)],
        capture_output=True,
        text=True,
        env=env,
    )

    if result.returncode != 0:
        print(f"[WARN] 内核 {kernel_num} 运行失败（size={size}）：{result.stderr.strip()}")
        return None

    # 从输出中找对应 size 的 GFLOPS 行
    # 输出格式：Average elapsed time: (x.xxxxxx) s, performance: (xxxx.x) GFLOPS. size: (N).
    pattern = re.compile(
        r"performance:\s*\(\s*([\d.]+)\s*\)\s*GFLOPS\.\s*size:\s*\(\s*"
        + str(size)
        + r"\s*\)"
    )
    for line in result.stdout.splitlines():
        m = pattern.search(line)
        if m:
            return float(m.group(1))

    print(f"[WARN] 内核 {kernel_num} 在 size={size} 时未找到 GFLOPS 数据。")
    return None


# ── 基准测试主流程 ─────────────────────────────────────────────────────────────

def run_benchmarks() -> dict[int, list[float | None]]:
    """
    对所有内核和规模执行测试，返回 {kernel_num: [gflops_1024, gflops_2048, gflops_4096]}。
    """
    results: dict[int, list[float | None]] = {}

    for knum in KERNELS:
        print(f"\n>>> 测试内核 {knum} ({KERNELS[knum]})")
        gflops_list = []
        for size in SIZES:
            print(f"    size={size} ...", end=" ", flush=True)
            gflops = run_kernel(knum, size)
            print(f"{gflops:.1f} GFLOPS" if gflops else "N/A")
            gflops_list.append(gflops)
        results[knum] = gflops_list

    return results


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def plot_results(results: dict[int, list[float | None]]):
    """绘制性能折线图并保存为 benchmark_result.png。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # cuBLAS 基准单独用虚线加粗绘制
    cublas_data = results.get(0)
    if cublas_data:
        valid = [(s, g) for s, g in zip(SIZES, cublas_data) if g is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "k--", linewidth=2.5, marker="D",
                    markersize=8, label="cuBLAS (baseline)", zorder=10)

    # 颜色循环（跳过黑色留给 cuBLAS）
    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    color_idx = 0

    for knum, name in KERNELS.items():
        if knum == 0:
            continue  # cuBLAS 已单独绘制
        data = results.get(knum, [])
        valid = [(s, g) for s, g in zip(SIZES, data) if g is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o", markersize=7,
                color=colors[color_idx % len(colors)], label=name, linewidth=1.8)
        color_idx += 1

    ax.set_xlabel("矩阵规模 N（方阵 N×N）", fontsize=13)
    ax.set_ylabel("性能 (GFLOPS)", fontsize=13)
    ax.set_title("NVIDIA P100 (sm_60) SGEMM 内核性能对比", fontsize=15)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)

    output_path = PROJECT_ROOT / "benchmark_result.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n图表已保存至：{output_path}")


# ── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build_project()
    results = run_benchmarks()
    plot_results(results)

    # 打印汇总表格
    print("\n" + "=" * 60)
    print(f"{'内核':<25}" + "".join(f"{s:>12}" for s in SIZES))
    print("-" * 60)
    for knum, name in KERNELS.items():
        data = results.get(knum, [None] * len(SIZES))
        row = f"{name:<25}"
        for g in data:
            row += f"{g:>11.1f}" if g else f"{'N/A':>11}"
        print(row)
    print("=" * 60)
