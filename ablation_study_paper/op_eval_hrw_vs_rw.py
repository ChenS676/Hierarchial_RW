# eval_hrw_vs_rw.py

import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse

# ============================================================
# Graph construction
# ============================================================

def make_synthetic_bottleneck_graph(
    n1=80,
    n2=80,
    p_in=0.12,
    num_bridges=2,
    seed=0,
):
    rng = np.random.default_rng(seed)

    G1 = nx.erdos_renyi_graph(n1, p_in, seed=seed)
    G2 = nx.erdos_renyi_graph(n2, p_in, seed=seed + 1)
    G2 = nx.relabel_nodes(G2, {i: i + n1 for i in G2.nodes()})

    G = nx.disjoint_union(G1, G2)

    bridges = []
    for _ in range(num_bridges):
        u = int(rng.integers(0, n1))
        v = int(rng.integers(n1, n1 + n2))
        G.add_edge(u, v)
        bridges.append(tuple(sorted((u, v))))

    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()
    G = nx.convert_node_labels_to_integers(G)

    return G, set(bridges)


def load_real_graph():
    G = nx.karate_club_graph()
    G = nx.convert_node_labels_to_integers(G)
    return G


def detect_bottlenecks(G, top_ratio=0.08):
    edge_score = nx.edge_betweenness_centrality(G)
    k = max(1, int(len(edge_score) * top_ratio))
    top_edges = sorted(edge_score, key=edge_score.get, reverse=True)[:k]
    return set(tuple(sorted(e)) for e in top_edges)


# ============================================================
# Random walk methods
# ============================================================

def one_step_rw(G, start, walk_len, rng):
    path = [start]
    cur = start

    for _ in range(walk_len):
        nbrs = list(G.neighbors(cur))
        if len(nbrs) == 0:
            break

        cur = rng.choice(nbrs)
        path.append(cur)

    return path


def simple_rw_trajectories(G, source, M=8, walk_len=6, rng=None):
    return [
        one_step_rw(G, source, walk_len, rng)
        for _ in range(M)
    ]


def budget_matched_rw_trajectories(G, source, R=1, M=8, walk_len=6, rng=None):
    """
    Budget-matched simple random walk.

    HRW uses approximately (R + 1) rounds of expansion.
    This baseline uses the same number of walk segments,
    but every segment restarts from the original source.
    """
    total_walks = (R + 1) * M

    return [
        one_step_rw(G, source, walk_len, rng)
        for _ in range(total_walks)
    ]


def hrw_trajectories(
    G,
    source,
    R=1,
    M=8,
    walk_len=6,
    rng=None,
    merge_terminal=True,
):
    """
    Hierarchical random walk.

    R = 0:
        ordinary random walks from the source.

    R > 0:
        after each recurrent step, the terminal nodes become
        the frontier seeds for the next expansion.
    """
    all_paths = []
    frontier = [source]

    for _ in range(R + 1):
        next_frontier = []

        for seed in frontier:
            for _ in range(M):
                path = one_step_rw(G, seed, walk_len, rng)
                all_paths.append(path)
                next_frontier.append(path[-1])

        if merge_terminal:
            next_frontier = list(set(next_frontier))

        frontier = next_frontier

    return all_paths


# ============================================================
# Metrics
# ============================================================

def earliest_arrival_times(paths, source):
    eat = {source: 0}

    for path in paths:
        for t, node in enumerate(path):
            if node not in eat:
                eat[node] = t
            else:
                eat[node] = min(eat[node], t)

    return eat


def bottleneck_cross_coverage(paths, bottleneck_edges):
    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            e = tuple(sorted((u, v)))
            if e in bottleneck_edges:
                return 1.0

    return 0.0


def evaluate_method(
    G,
    bottleneck_edges,
    method,
    R,
    M=8,
    walk_len=6,
    pair_samples=5000,
    seed=0,
):
    rng = random.Random(seed)
    nodes = list(G.nodes())

    source_to_eat = {}
    source_cross_flags = []

    for source in nodes:
        if method == "simple_rw":
            paths = simple_rw_trajectories(
                G,
                source,
                M=M,
                walk_len=walk_len,
                rng=rng,
            )

        elif method == "budget_rw":
            paths = budget_matched_rw_trajectories(
                G,
                source,
                R=R,
                M=M,
                walk_len=walk_len,
                rng=rng,
            )

        elif method == "hrw":
            paths = hrw_trajectories(
                G,
                source,
                R=R,
                M=M,
                walk_len=walk_len,
                rng=rng,
                merge_terminal=True,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        source_to_eat[source] = earliest_arrival_times(paths, source)
        source_cross_flags.append(
            bottleneck_cross_coverage(paths, bottleneck_edges)
        )

    missing_penalty = (R + 1) * walk_len + 1

    eat_values = []
    ecc_values = []

    for _ in range(pair_samples):
        u, v = rng.sample(nodes, 2)

        eat_uv = source_to_eat[u].get(v, missing_penalty)
        eat_vu = source_to_eat[v].get(u, missing_penalty)

        eat_values.append(eat_uv)
        ecc_values.append(eat_uv + eat_vu)

    return {
        "R": R,
        "method": method,
        "mean_EAT": float(np.mean(eat_values)),
        "mean_ECC": float(np.mean(ecc_values)),
        "bottleneck_cross_coverage": float(np.mean(source_cross_flags)),
    }


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    G,
    bottleneck_edges,
    graph_name,
    R_values=(0, 1, 2, 3, 4),
    methods=("simple_rw", "budget_rw", "hrw"),
    M=8,
    walk_len=6,
    pair_samples=5000,
    repeats=5,
):
    rows = []

    for method in methods:
        for R in R_values:
            for repeat in range(repeats):
                result = evaluate_method(
                    G=G,
                    bottleneck_edges=bottleneck_edges,
                    method=method,
                    R=R,
                    M=M,
                    walk_len=walk_len,
                    pair_samples=pair_samples,
                    seed=1000 + repeat,
                )

                result["graph"] = graph_name
                result["repeat"] = repeat
                result["num_nodes"] = G.number_of_nodes()
                result["num_edges"] = G.number_of_edges()
                result["num_bottleneck_edges"] = len(bottleneck_edges)
                result["M"] = M
                result["walk_len"] = walk_len
                result["pair_samples"] = pair_samples

                rows.append(result)

    return rows


def save_result_csv(output_path="result.csv"):
    synthetic_G, synthetic_bottlenecks = make_synthetic_bottleneck_graph(
        n1=80,
        n2=80,
        p_in=0.12,
        num_bridges=2,
        seed=0,
    )

    real_G = load_real_graph()
    real_bottlenecks = detect_bottlenecks(real_G, top_ratio=0.08)

    all_rows = []

    all_rows += run_experiment(
        synthetic_G,
        synthetic_bottlenecks,
        graph_name="synthetic",
        R_values=[0, 1, 2, 3, 4],
        methods=["simple_rw", "budget_rw", "hrw"],
        M=8,
        walk_len=6,
        pair_samples=5000,
        repeats=5,
    )

    all_rows += run_experiment(
        real_G,
        real_bottlenecks,
        graph_name="real_world",
        R_values=[0, 1, 2, 3, 4],
        methods=["simple_rw", "budget_rw", "hrw"],
        M=8,
        walk_len=6,
        pair_samples=5000,
        repeats=5,
    )

    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)

    print(f"Saved raw result to {output_path}")
    return df


# ============================================================
# Visualization from result.csv
# ============================================================

def load_and_aggregate_result(csv_path="result.csv"):
    df = pd.read_csv(csv_path)

    agg = (
        df.groupby(["graph", "method", "R"])
        .agg(
            mean_EAT=("mean_EAT", "mean"),
            std_EAT=("mean_EAT", "std"),
            mean_ECC=("mean_ECC", "mean"),
            std_ECC=("mean_ECC", "std"),
            coverage=("bottleneck_cross_coverage", "mean"),
            std_coverage=("bottleneck_cross_coverage", "std"),
        )
        .reset_index()
    )

    agg.to_csv("result_aggregated.csv", index=False)
    print("Saved aggregated result to result_aggregated.csv")

    return agg

# ============================================================
# Visualization from result.csv
# ============================================================

def load_and_aggregate_result(csv_path="result.csv"):
    df = pd.read_csv(csv_path)

    agg = (
        df.groupby(["graph", "method", "R"])
        .agg(
            mean_EAT=("mean_EAT", "mean"),
            std_EAT=("mean_EAT", "std"),
            mean_ECC=("mean_ECC", "mean"),
            std_ECC=("mean_ECC", "std"),
            coverage=("bottleneck_cross_coverage", "mean"),
            std_coverage=("bottleneck_cross_coverage", "std"),
        )
        .reset_index()
    )

    agg.to_csv("result_aggregated.csv", index=False)
    return agg


def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    })


def plot_one_metric_one_graph(
    agg,
    graph,
    metric,
    ylabel,
    out_path,
):
    method_order = ["simple_rw", "budget_rw", "hrw"]

    method_names = {
        "simple_rw": "RW",
        "budget_rw": "Budget-RW",
        "hrw": "HRW",
    }

    method_styles = {
        "simple_rw": {
            "marker": "o",
            "linestyle": ":",
            "color": "#0B4FA3",
        },
        "budget_rw": {
            "marker": "s",
            "linestyle": "--",
            "color": "#5DA5DA",
        },
        "hrw": {
            "marker": "^",
            "linestyle": "-",
            "color": "#006D2C",
        },
    }

    std_map = {
        "coverage": "std_coverage",
        "mean_EAT": "std_EAT",
        "mean_ECC": "std_ECC",
    }

    graph_titles = {
        "synthetic": "Synthetic graph",
        "real_world": "Real-world graph",
    }

    fig, ax = plt.subplots(figsize=(3.6, 2.6))

    sub_graph = agg[agg["graph"] == graph]

    for method in method_order:
        sub = sub_graph[sub_graph["method"] == method].sort_values("R")

        if len(sub) == 0:
            continue

        x = sub["R"].to_numpy()
        y = sub[metric].to_numpy()
        yerr = sub[std_map[metric]].fillna(0).to_numpy()

        style = method_styles[method]

        lw = 2.4 if method == "hrw" else 1.8
        alpha = 1.0 if method == "hrw" else 0.8

        ax.plot(
            x,
            y,
            label=method_names[method],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=lw,
            markersize=5,
            alpha=alpha,
        )

        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=style["color"],
            alpha=0.14,
            linewidth=0,
        )

    if metric == "coverage":
        # Adaptive y-axis for bottleneck/coverage
        y_vals = sub_graph["coverage"].to_numpy()

        ymin = max(0.0, y_vals.min() - 0.05)
        ymax = min(1.0, y_vals.max() + 0.05)

        # If range is too small, enforce minimum spread for readability
        if ymax - ymin < 0.1:
            center = 0.5 * (ymax + ymin)
            ymin = max(0.0, center - 0.05)
            ymax = min(1.0, center + 0.05)

        ax.set_ylim(ymin, ymax)

    ax.set_title(graph_titles.get(graph, graph))
    ax.set_xlabel("Recurrent step $R$")
    ax.set_ylabel(ylabel)

    if metric == "coverage":
        ax.set_ylim(-0.03, 1.03)

    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4)
    ax.legend(frameon=False, loc="best")

    plt.tight_layout()
    fig.savefig(out_path)
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=300)
    print(f"Saved plot to {out_path} and {out_path.replace('.pdf', '.png')}")
    plt.close(fig)


def plot_selected_metrics_separately(
    csv_path="result.csv",
    out_prefix="hrw_plot",
    metrics=None,
):
    set_plot_style()
    agg = load_and_aggregate_result(csv_path)

    if metrics is None:
        metrics = ["coverage", "eat", "ecc"]

    metric_info = {
        "coverage": ("coverage", "Coverage ↑"),
        "bottleneck": ("coverage", "Bottleneck-crossing probability ↑"),
        "eat": ("mean_EAT", "Mean EAT ↓"),
        "ecc": ("mean_ECC", "Mean ECC ↓"),
    }

    graph_order = ["synthetic", "real_world"]

    for graph in graph_order:
        for metric_key in metrics:
            metric, ylabel = metric_info[metric_key]

            out_path = f"{out_prefix}_{graph}_{metric_key}.pdf"

            plot_one_metric_one_graph(
                agg=agg,
                graph=graph,
                metric=metric,
                ylabel=ylabel,
                out_path=out_path,
            )

            print(f"Saved {out_path}")
            print(f"Saved {out_path.replace('.pdf', '.png')}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run", action="store_true")
    parser.add_argument("--csv", type=str, default="result.csv")
    parser.add_argument("--out_prefix", type=str, default="hrw_plot")

    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--bottleneck", action="store_true")
    parser.add_argument("--mean_eat", action="store_true")
    parser.add_argument("--mean_ecc", action="store_true")
    parser.add_argument("--all", action="store_true")

    args = parser.parse_args()

    if args.run:
        save_result_csv(args.csv)

    if args.all:
        metrics = ["coverage", "bottleneck", "eat", "ecc"]
    else:
        metrics = []
        if args.coverage:
            metrics.append("coverage")
        if args.bottleneck:
            metrics.append("bottleneck")
        if args.mean_eat:
            metrics.append("eat")
        if args.mean_ecc:
            metrics.append("ecc")

    if len(metrics) == 0:
        metrics = ["coverage", "eat", "ecc"]

    plot_selected_metrics_separately(
        csv_path=args.csv,
        out_prefix=args.out_prefix,
        metrics=metrics,
    )
# # Run experiment and plot default metrics
# uv run eval_hrw_eat_bottleneck.py --run

# # Plot only bottleneck-crossing probability
# uv run eval_hrw_eat_bottleneck.py --bottleneck

# # Plot mean EAT and mean ECC
# uv run eval_hrw_eat_bottleneck.py --mean_eat --mean_ecc

# # Plot all
# uv run eval_hrw_eat_bottleneck.py --all

# # Use existing CSV only
# uv run eval_hrw_eat_bottleneck.py --csv result.csv --mean_ecc