"""
Ablation: num_walks vs Test MRR on Cora (walk_len=16, recurrent_steps=1)
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ----- Data (Cora, ExpA, wl=16, rs=1) -----
num_walks = [1, 2, 4, 8, 16]
test_mrr_mean = [0.36, 0.46, 0.48, 0.52, 0.34]
test_mrr_std  = [0.01, 0.01, 0.03, 0.01, 0.02]

# ----- Plot -----
fig, ax = plt.subplots(figsize=(5, 3.5))

x = np.arange(len(num_walks))

bars = ax.bar(
    x, test_mrr_mean,
    yerr=test_mrr_std,
    capsize=4,
    width=0.55,
    color='#4C72B0',
    edgecolor='white',
    linewidth=0.6,
    error_kw=dict(elinewidth=1.2, ecolor='#333333', capthick=1.2),
    zorder=3,
)

# Annotate each bar with mean ± std
for rect, mean, std in zip(bars, test_mrr_mean, test_mrr_std):
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        mean + std + 0.005,
        f'{mean:.2f}±{std:.2f}',
        ha='center', va='bottom', fontsize=7.5, color='#222222',
    )

ax.set_xticks(x)
ax.set_xticklabels([str(n) for n in num_walks])
ax.set_xlabel('Number of Walks', fontsize=11)
ax.set_ylabel('Test MRR', fontsize=11)
ax.set_title('Cora — Num Walks Ablation\n(walk length = 16, recurrent steps = 1)', fontsize=10)

ax.set_ylim(0, max(test_mrr_mean) + max(test_mrr_std) + 0.08)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.6, zorder=0)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()

out = 'data_analysis/ablation_numwalks_cora.pdf'
plt.savefig(out, bbox_inches='tight')
print(f'Saved: {out}')
plt.show()
