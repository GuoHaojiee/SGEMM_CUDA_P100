import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 原始数据（直接从 results.log 整理） ──────────────────────────────
sizes = [128, 256, 512, 1024, 2048, 4096]

data = {
    "K0: cuBLAS":        [418.7,  1762.4, 3904.5, 4189.2, 5323.9, 6226.3],
    "K1: Naive":         [24.7,   51.1,   84.8,   93.6,   86.0,   71.0  ],
    "K2: GMEM Coalesce": [105.0,  222.0,  329.1,  388.9,  369.5,  372.8 ],
    "K3: SMEM Cache":    [333.6,  852.2,  1521.2, 1632.4, 1669.5, 1689.4],
    "K4: 1D Tiling":     [144.7,  614.6,  1681.3, 2477.8, 2828.3, 2881.5],
    "K5: 2D Tiling":     [50.1,   221.0,  991.1,  2502.2, 3978.8, 4839.3],
    "K6: Vectorize":     [73.2,   300.5,  1257.8, 3256.0, 4973.0, 5586.7],
    "K9: Autotuned":     [80.8,   292.1,  1217.2, 3209.2, 4598.8, 5468.0],
    "K10: Warptiling":   [203.4,  918.6,  2631.9, 5182.5, 5525.2, 5701.4],
}

# P100 理论峰值 FP32 = 10.6 TFLOPS = 10600 GFLOPS
P100_PEAK = 10600.0

# ── 配色：cuBLAS 用黑色虚线，其余用渐变色 ────────────────────────────
colors = {
    "K0: cuBLAS":        "black",
    "K1: Naive":         "#d62728",
    "K2: GMEM Coalesce": "#ff7f0e",
    "K3: SMEM Cache":    "#bcbd22",
    "K4: 1D Tiling":     "#2ca02c",
    "K5: 2D Tiling":     "#17becf",
    "K6: Vectorize":     "#1f77b4",
    "K9: Autotuned":     "#9467bd",
    "K10: Warptiling":   "#e377c2",
}

linestyles = {k: ("--" if k == "K0: cuBLAS" else "-") for k in data}
linewidths = {k: (2.5 if k == "K0: cuBLAS" else 1.8) for k in data}

# ── 图1：绝对 GFLOPS 折线图 ──────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("SGEMM Kernel Performance on Tesla P100-SXM2-16GB",
             fontsize=14, fontweight="bold", y=1.01)

ax = axes[0]
for name, gflops in data.items():
    ax.plot(sizes, gflops,
            label=name,
            color=colors[name],
            linestyle=linestyles[name],
            linewidth=linewidths[name],
            marker="o", markersize=5)

# P100 峰值线
ax.axhline(P100_PEAK, color="gray", linestyle=":", linewidth=1.2,
           label=f"P100 Peak ({P100_PEAK/1000:.1f} TFLOPS)")

ax.set_xscale("log", base=2)
ax.set_xlabel("Matrix Size (M=N=K)", fontsize=11)
ax.set_ylabel("Performance (GFLOPS)", fontsize=11)
ax.set_title("Absolute Performance", fontsize=12)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax.set_xticks(sizes)
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, P100_PEAK * 1.05)

# ── 图2：相对 cuBLAS 百分比折线图 ────────────────────────────────────
ax2 = axes[1]
cublas = data["K0: cuBLAS"]

for name, gflops in data.items():
    if name == "K0: cuBLAS":
        continue
    pct = [g / c * 100 for g, c in zip(gflops, cublas)]
    ax2.plot(sizes, pct,
             label=name,
             color=colors[name],
             linestyle=linestyles[name],
             linewidth=linewidths[name],
             marker="o", markersize=5)

ax2.axhline(100, color="black", linestyle="--", linewidth=2.0, label="K0: cuBLAS (100%)")
ax2.set_xscale("log", base=2)
ax2.set_xlabel("Matrix Size (M=N=K)", fontsize=11)
ax2.set_ylabel("% of cuBLAS Performance", fontsize=11)
ax2.set_title("Relative to cuBLAS", fontsize=12)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
ax2.set_xticks(sizes)
ax2.legend(fontsize=8, loc="lower right")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 120)

plt.tight_layout()
plt.savefig("benchmark_result.png", dpi=150, bbox_inches="tight")
print("图片已保存为 benchmark_result.png")

# ── 打印汇总表 ────────────────────────────────────────────────────────
print("\n========== 4096x4096 性能汇总 ==========")
print(f"{'内核':<22} {'GFLOPS':>10} {'占cuBLAS%':>10}")
print("-" * 44)
cublas_4096 = data["K0: cuBLAS"][-1]
for name, gflops in data.items():
    pct = gflops[-1] / cublas_4096 * 100
    print(f"{name:<22} {gflops[-1]:>10.1f} {pct:>9.1f}%")
print(f"\nP100 理论峰值: {P100_PEAK} GFLOPS")
print(f"K10 占理论峰值: {data['K10: Warptiling'][-1]/P100_PEAK*100:.1f}%")
