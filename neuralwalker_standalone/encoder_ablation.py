"""
Encoder ablation study — NeurIPS-style bar chart with error bars.
Saves to encoder_ablation.pdf (vector, embeds fonts).

Usage:
    python plot_encoder_ablation.py
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator

# ── Reproduce-able font embedding ──────────────────────────────────────────
matplotlib.rcParams.update({
    "pdf.fonttype": 42,        # TrueType → embedded in PDF
    "ps.fonttype":  42,
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":    9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── Data ────────────────────────────────────────────────────────────────────
encoders = ["Conv", "Mamba", "Trans."]

# rows: Val (top), Test (bottom) — cols: MRR, Hits@1
rows = {
    "Validation": {
        "MRR":    dict(means=[0.3000, 0.2703, 0.2403], stds=[0.1163, 0.0712, 0.0503]),
        "Hits@1": dict(means=[0.2546, 0.1899, 0.1773], stds=[0.1251, 0.0748, 0.0614]),
    },
    "Test": {
        "MRR":    dict(means=[0.2971, 0.3348, 0.3127], stds=[0.1623, 0.0926, 0.1470]),
        "Hits@1": dict(means=[0.2000, 0.2236, 0.1924], stds=[0.1648, 0.1137, 0.1984]),
    },
}
subplot_labels = [["(a)", "(b)"], ["(c)", "(d)"]]

# Wong colour-blind-safe palette (blue, green, orange)
COLORS      = ["#0072B2", "#009E73", "#E69F00"]
EDGE_COLORS = ["#005080", "#006B4E", "#9E6D00"]

# ── Layout ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(
    2, 2,
    figsize=(4.5, 4.0),
    sharey=False,
)
fig.subplots_adjust(wspace=0.38, hspace=0.50, left=0.12, right=0.98, top=0.88, bottom=0.12)

n = len(encoders)
x = np.arange(n)
bar_w = 0.52

for r, (split, metrics) in enumerate(rows.items()):
    for c, (metric, d) in enumerate(metrics.items()):
        ax = axes[r, c]
        means = np.array(d["means"])
        stds  = np.array(d["stds"])

        ax.bar(
            x, means,
            width=bar_w,
            color=COLORS,
            edgecolor=EDGE_COLORS,
            linewidth=0.6,
            zorder=3,
        )

        # Error bars (±1σ)
        ax.errorbar(
            x, means,
            yerr=stds,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=3,
            capthick=0.8,
            zorder=4,
        )

        # Subtle horizontal grid
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax.grid(axis="y", which="major", linewidth=0.3, color="#cccccc", zorder=0)
        ax.grid(axis="y", which="minor", linewidth=0.15, color="#eeeeee", zorder=0)
        ax.set_axisbelow(True)

        ax.set_xticks(x)
        ax.set_xticklabels(encoders, rotation=0)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(0, 0.56)
        ax.yaxis.set_major_locator(MultipleLocator(0.1))

        # y-label only on left column
        if c == 0:
            ax.set_ylabel(f"{split}\nScore")

        ax.set_title(f"{split} {metric}", pad=4, fontsize=9)
        ax.text(
            0.04, 0.97, subplot_labels[r][c],
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=8, style="italic",
        )

# ── Shared legend ───────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(facecolor=COLORS[i], edgecolor=EDGE_COLORS[i],
                   linewidth=0.6, label=encoders[i])
    for i in range(n)
]
fig.legend(
    handles=patches,
    loc="upper center",
    ncol=3,
    frameon=False,
    fontsize=8,
    bbox_to_anchor=(0.52, 1.02),
    handlelength=1.0,
    handleheight=0.8,
    columnspacing=1.0,
)

# ── Save ────────────────────────────────────────────────────────────────────
out = "encoder_ablation.pdf"
fig.savefig(out, format="pdf", bbox_inches="tight", dpi=300)
print(f"Saved → {out}")

# Also save PNG for quick preview
fig.savefig("encoder_ablation.png", dpi=300, bbox_inches="tight")
print("Saved → encoder_ablation.png")

plt.show()