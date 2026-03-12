#!/usr/bin/env python3
"""
benchmark.py ── NVIDIA P100 (sm_60) SGEMM 性能基准测试

以 cuBLAS（内核 0）为 100% 基准，对比各手写内核的性能。

输出：
  benchmark_results/<n>_output.txt  每个内核的原始程序输出
  benchmark_result.png              性能对比图（双子图）

用法：
  python3 benchmark.py              # 编译 + 运行 + 绘图
  python3 benchmark.py --no-build   # 跳过编译（已有可执行文件时）
  python3 benchmark.py --plot-only  # 只重新绘图（读取已有 txt 结果）
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # 无桌面环境（SSH 登录 P100 服务器）时必须
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── 全局配置 ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.resolve()
BUILD_DIR    = PROJECT_ROOT / "build"
RESULTS_DIR  = PROJECT_ROOT / "benchmark_results"

# 内核编号 → 显示名称
KERNELS: dict[int, str] = {
    0:  "cuBLAS",
    1:  "K1: Naive",
    2:  "K2: GMEM Coalesce",
    3:  "K3: SMEM Block",
    4:  "K4: 1D Tiling",
    5:  "K5: 2D Tiling",
    6:  "K6: Vectorize",
    9:  "K9: Autotuned",
    10: "K10: Warptiling",
}

# 解析程序输出中的 GFLOPS 行
_PATTERN = re.compile(
    r"performance:\s*\(\s*([\d.]+)\s*\)\s*GFLOPS\.\s*size:\s*\(\s*(\d+)\s*\)"
)

# ── 编译 ──────────────────────────────────────────────────────────────────────

def build_project() -> None:
    print("=" * 62)
    print("  编译项目（cmake + make）")
    print("=" * 62)
    BUILD_DIR.mkdir(exist_ok=True)

    r = subprocess.run(
        ["cmake", "-DCMAKE_BUILD_TYPE=Release", str(PROJECT_ROOT)],
        cwd=BUILD_DIR,
    )
    if r.returncode != 0:
        sys.exit("[ERROR] cmake 配置失败")

    r = subprocess.run(
        ["make", "-j", str(os.cpu_count() or 4), "--no-print-directory"],
        cwd=BUILD_DIR,
    )
    if r.returncode != 0:
        sys.exit("[ERROR] make 编译失败")
    print("编译完成。\n")


# ── 运行内核 ──────────────────────────────────────────────────────────────────

def run_kernel(kernel_num: int) -> dict[int, float]:
    """
    运行 ./build/sgemm <kernel_num>，将输出保存到 txt 文件，
    返回 {size: gflops}。
    """
    sgemm = BUILD_DIR / "sgemm"
    if not sgemm.exists():
        sys.exit(f"[ERROR] 找不到可执行文件：{sgemm}")

    name = KERNELS[kernel_num]
    print(f"  运行 内核 {kernel_num:2d} ({name}) ...", end=" ", flush=True)

    result = subprocess.run(
        [str(sgemm), str(kernel_num)],
        capture_output=True, text=True,
    )

    # 保存原始输出，方便事后复查
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"{kernel_num}_output.txt"
    out_file.write_text(result.stdout + result.stderr)

    if result.returncode != 0:
        print(f"[FAIL] 退出码 {result.returncode}")
        return {}

    # 解析 GFLOPS 行
    data: dict[int, float] = {}
    for line in result.stdout.splitlines():
        m = _PATTERN.search(line)
        if m:
            gflops = float(m.group(1))
            size   = int(m.group(2))
            data[size] = gflops

    if data:
        sizes_str = "  ".join(f"{s}→{g:.0f}" for s, g in sorted(data.items()))
        print(f"OK  [{sizes_str} GFLOPS]")
    else:
        print("[WARN] 未解析到 GFLOPS 数据")
    return data


# ── 解析已有 txt（--plot-only 模式）────────────────────────────────────────────

def load_from_txt() -> dict[int, dict[int, float]]:
    all_data: dict[int, dict[int, float]] = {}
    for knum in KERNELS:
        f = RESULTS_DIR / f"{knum}_output.txt"
        if not f.exists():
            print(f"  [WARN] 找不到 {f.name}，跳过内核 {knum}")
            continue
        data: dict[int, float] = {}
        for line in f.read_text().splitlines():
            m = _PATTERN.search(line)
            if m:
                data[int(m.group(2))] = float(m.group(1))
        if data:
            all_data[knum] = data
    return all_data


# ── 绘图 ──────────────────────────────────────────────────────────────────────

# 颜色方案（cuBLAS 用黑色，其余用 tab10）
_COLORS = {
    0:  "black",
    1:  "#d62728",   # 红
    2:  "#ff7f0e",   # 橙
    3:  "#bcbd22",   # 黄绿
    4:  "#2ca02c",   # 绿
    5:  "#17becf",   # 青
    6:  "#1f77b4",   # 蓝
    9:  "#9467bd",   # 紫
    10: "#e377c2",   # 粉
}


def plot_results(all_data: dict[int, dict[int, float]]) -> None:
    """
    双子图：
      左图 — 绝对 GFLOPS 折线图（cuBLAS 粗虚线作参考）
      右图 — 相对 cuBLAS 百分比水平条形图（固定 size=最大值）
    """
    if not all_data:
        print("[ERROR] 没有数据，无法绘图")
        return

    # cuBLAS 基准
    cublas_data: dict[int, float] = all_data.get(0, {})
    if not cublas_data:
        print("[WARN] 缺少 cuBLAS 数据，无法计算相对百分比")

    # 取所有内核都覆盖的 size 列表（升序）
    all_sizes: list[int] = sorted(
        set.intersection(*[set(d.keys()) for d in all_data.values()])
        if all_data else set()
    )
    bar_size = max(all_sizes) if all_sizes else 4096  # 条形图用最大规模

    # ── 布局 ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.38)
    ax_line = fig.add_subplot(gs[0])
    ax_bar  = fig.add_subplot(gs[1])

    # ── 左图：绝对 GFLOPS 折线图 ─────────────────────────────────────────────
    for knum, name in KERNELS.items():
        if knum not in all_data:
            continue
        data  = all_data[knum]
        sizes = sorted(data.keys())
        gf    = [data[s] for s in sizes]
        color = _COLORS.get(knum, "gray")
        if knum == 0:
            # cuBLAS：粗虚线
            ax_line.plot(sizes, gf, color=color, linestyle="--",
                         linewidth=2.5, marker="D", markersize=7,
                         label=name, zorder=10)
        else:
            ax_line.plot(sizes, gf, color=color, linestyle="-",
                         linewidth=1.8, marker="o", markersize=6,
                         label=name)

    ax_line.set_xlabel("矩阵规模 N（方阵 N×N）", fontsize=12)
    ax_line.set_ylabel("性能 (GFLOPS)", fontsize=12)
    ax_line.set_title("各内核绝对性能（P100 sm_60）", fontsize=13)
    ax_line.set_xticks(all_sizes)
    ax_line.set_xticklabels([str(s) for s in all_sizes], rotation=30)
    ax_line.legend(fontsize=9, loc="upper left")
    ax_line.grid(True, linestyle="--", alpha=0.45)
    ax_line.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.1f}k" if x >= 1000 else f"{x:.0f}"
    ))

    # ── 右图：相对 cuBLAS 百分比条形图 ─────────────────────────────────────
    cublas_gf = cublas_data.get(bar_size) if cublas_data else None

    bar_kernels = [k for k in KERNELS if k != 0 and k in all_data
                   and bar_size in all_data[k]]
    bar_names   = [KERNELS[k] for k in bar_kernels]
    bar_colors  = [_COLORS.get(k, "gray") for k in bar_kernels]

    if cublas_gf:
        bar_pcts = [all_data[k][bar_size] / cublas_gf * 100
                    for k in bar_kernels]
        xlabel = f"相对 cuBLAS 性能（%）\ncuBLAS = {cublas_gf:.0f} GFLOPS @ N={bar_size}"
    else:
        bar_pcts = [all_data[k][bar_size] for k in bar_kernels]
        xlabel = f"GFLOPS @ N={bar_size}"

    # 从小到大排序
    sorted_pairs = sorted(zip(bar_pcts, bar_names, bar_colors), key=lambda x: x[0])
    bar_pcts_s, bar_names_s, bar_colors_s = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    y_pos = range(len(bar_names_s))
    bars  = ax_bar.barh(y_pos, bar_pcts_s, color=bar_colors_s,
                        edgecolor="white", height=0.65)

    # 在条形末端显示百分比数值
    for bar, pct in zip(bars, bar_pcts_s):
        ax_bar.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", ha="left", fontsize=9,
        )

    # cuBLAS 100% 参考线
    if cublas_gf:
        ax_bar.axvline(100, color="black", linestyle="--",
                       linewidth=1.8, label="cuBLAS = 100%")
        ax_bar.legend(fontsize=9)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(bar_names_s, fontsize=10)
    ax_bar.set_xlabel(xlabel, fontsize=11)
    ax_bar.set_title(f"相对 cuBLAS 性能（N={bar_size}）", fontsize=13)
    ax_bar.set_xlim(0, max(bar_pcts_s or [100]) * 1.18 if bar_pcts_s else 110)
    ax_bar.grid(True, axis="x", linestyle="--", alpha=0.4)

    fig.suptitle("NVIDIA P100 (sm_60)  SGEMM 内核性能对比",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "benchmark_result.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n图表已保存：{out_path}")


# ── 汇总表格 ──────────────────────────────────────────────────────────────────

def print_table(all_data: dict[int, dict[int, float]]) -> None:
    cublas = all_data.get(0, {})
    all_sizes = sorted(set().union(*[d.keys() for d in all_data.values()]))

    print("\n" + "=" * 72)
    header = f"{'内核':<22}" + "".join(f"{s:>10}" for s in all_sizes)
    if cublas:
        header += f"  {'% cuBLAS':>10}"
    print(header)
    print("-" * 72)

    bar_size = max(all_sizes) if all_sizes else 4096
    for knum, name in KERNELS.items():
        if knum not in all_data:
            continue
        data = all_data[knum]
        row  = f"{name:<22}"
        for s in all_sizes:
            g = data.get(s)
            row += f"{g:>10.0f}" if g else f"{'N/A':>10}"
        if cublas and cublas.get(bar_size) and data.get(bar_size):
            pct = data[bar_size] / cublas[bar_size] * 100
            row += f"  {pct:>9.1f}%"
        print(row)

    print("=" * 72)


# ── 入口 ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = set(sys.argv[1:])
    plot_only  = "--plot-only" in args
    no_build   = "--no-build"  in args or plot_only

    if not no_build:
        build_project()

    if plot_only:
        print("读取已有结果文件 ...")
        all_data = load_from_txt()
    else:
        print("=" * 62)
        print("  运行各内核（内核 0 = cuBLAS 基准）")
        print("=" * 62)
        all_data: dict[int, dict[int, float]] = {}
        for knum in KERNELS:
            data = run_kernel(knum)
            if data:
                all_data[knum] = data

    print_table(all_data)
    plot_results(all_data)


if __name__ == "__main__":
    main()
