import pandas as pd
import matplotlib.pyplot as plt


def plot_ecc_neurips_style(
    csv_path="result.csv",
    out_prefix="ecc_plot",
    figsize=(4.2, 2.8),
    font_size=12,
):
    df = pd.read_csv(csv_path)

    agg = (
        df.groupby(["graph", "method", "R"])
        .agg(
            mean_ECC=("mean_ECC", "mean"),
            std_ECC=("mean_ECC", "std"),
        )
        .reset_index()
    )

    methods = ["simple_rw", "budget_rw", "hrw"]

    labels = {
        "simple_rw": "RW",
        "budget_rw": "Budget-RW",
        "hrw": "HRW",
    }

    # Color style inspired by your example:
    # deep blue, medium blue, deep green; HRW highlighted by darkest green.
    colors = {
        "simple_rw": "#0B4FA3",    # deep blue
        "budget_rw": "#5DA5DA",    # soft blue
        "hrw": "#006D2C",          # deep green, highlight
    }

    markers = {
        "simple_rw": "D",
        "budget_rw": "D",
        "hrw": "D",
    }

    linestyles = {
        "simple_rw": "--",
        "budget_rw": "-",
        "hrw": "-",
    }

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": font_size,
        "axes.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    graphs = agg["graph"].unique()

    for graph in graphs:
        fig, ax = plt.subplots(figsize=figsize)

        sub = agg[agg["graph"] == graph]

        for m in methods:
            d = sub[sub["method"] == m].sort_values("R")
            if len(d) == 0:
                continue

            x = d["R"].to_numpy()
            y = d["mean_ECC"].to_numpy()
            yerr = d["std_ECC"].fillna(0).to_numpy()

            lw = 2.6 if m == "hrw" else 2.0
            alpha = 1.0 if m == "hrw" else 0.75

            ax.plot(
                x,
                y,
                label=labels[m],
                color=colors[m],
                marker=markers[m],
                linestyle=linestyles[m],
                linewidth=lw,
                markersize=6,
                alpha=alpha,
            )

            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color=colors[m],
                alpha=0.12,
                linewidth=0,
            )

        ax.set_xlabel("Recurrent step $R$", fontsize=font_size + 2)
        ax.set_ylabel("Mean ECC", fontsize=font_size + 2)
        ax.set_title(graph, fontsize=font_size + 1)

        ax.grid(True, linestyle="--", linewidth=1.0, alpha=0.45)

        ax.tick_params(
            axis="both",
            which="major",
            labelsize=font_size,
            length=5,
            width=1.0,
        )
        ax.tick_params(
            axis="both",
            which="minor",
            length=3,
            width=0.8,
        )
        ax.minorticks_on()

        ax.legend(frameon=False, fontsize=font_size - 1)

        plt.tight_layout()
        fig.savefig(f"{out_prefix}_{graph}.pdf")
        fig.savefig(f"{out_prefix}_{graph}.png", dpi=300)
        plt.close(fig)

    print("Saved plots:")
    for graph in graphs:
        print(f"{out_prefix}_{graph}.pdf")
        print(f"{out_prefix}_{graph}.png")


plot_ecc_neurips_style("result.csv")