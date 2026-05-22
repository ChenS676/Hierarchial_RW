import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import torch


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


def one_step_non_backtracking_rw(G, start, walk_len, rng):
    path = [start]
    prev = None
    cur = start

    for _ in range(walk_len):
        nbrs = list(G.neighbors(cur))
        if len(nbrs) == 0:
            break

        if prev is not None and len(nbrs) > 1:
            candidates = [x for x in nbrs if x != prev]
        else:
            candidates = nbrs

        nxt = rng.choice(candidates)
        path.append(nxt)

        prev = cur
        cur = nxt

    return path


def one_step_node2vec_rw(G, start, walk_len, rng, p=1.0, q=0.5):
    path = [start]

    if walk_len == 0:
        return path

    cur = start
    nbrs = list(G.neighbors(cur))

    if len(nbrs) == 0:
        return path

    nxt = rng.choice(nbrs)
    path.append(nxt)

    prev = cur
    cur = nxt

    for _ in range(1, walk_len):
        nbrs = list(G.neighbors(cur))
        if len(nbrs) == 0:
            break

        weights = []

        for x in nbrs:
            if x == prev:
                weight = 1.0 / p
            elif G.has_edge(x, prev):
                weight = 1.0
            else:
                weight = 1.0 / q

            weights.append(weight)

        nxt = rng.choices(nbrs, weights=weights, k=1)[0]
        path.append(nxt)

        prev = cur
        cur = nxt

    return path


# ============================================================
# Trajectory generators
# ============================================================

def simple_rw_trajectories(G, source, M=8, walk_len=6, rng=None):
    return [one_step_rw(G, source, walk_len, rng) for _ in range(M)]


def budget_matched_rw_trajectories(G, source, R=1, M=8, walk_len=6, rng=None):
    total_walks = (R + 1) * M
    return [one_step_rw(G, source, walk_len, rng) for _ in range(total_walks)]


def budget_matched_node2vec_trajectories(
    G,
    source,
    R=1,
    M=8,
    walk_len=6,
    rng=None,
    p=1.0,
    q=0.5,
):
    total_walks = (R + 1) * M

    return [
        one_step_node2vec_rw(
            G,
            source,
            walk_len,
            rng,
            p=p,
            q=q,
        )
        for _ in range(total_walks)
    ]


def budget_matched_non_backtracking_trajectories(
    G,
    source,
    R=1,
    M=8,
    walk_len=6,
    rng=None,
):
    total_walks = (R + 1) * M

    return [
        one_step_non_backtracking_rw(G, source, walk_len, rng)
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


def paths_to_tensor(paths):
    if len(paths) == 0:
        return torch.empty((0, 0), dtype=torch.long)

    max_len = max(len(p) for p in paths)
    padded = []

    for p in paths:
        if len(p) < max_len:
            p = p + [p[-1]] * (max_len - len(p))
        padded.append(p)

    return torch.tensor(padded, dtype=torch.long)


def node_coverage_ratio(num_nodes, paths):
    if len(paths) == 0:
        return 0.0

    walks = paths_to_tensor(paths)

    if walks.numel() == 0:
        return 0.0

    visited = torch.unique(walks.reshape(-1))
    return float(visited.numel()) / float(num_nodes)


def edge_coverage_ratio(G, paths):
    if len(paths) == 0 or G.number_of_edges() == 0:
        return 0.0

    visited_edges = set()

    for path in paths:
        for u, v in zip(path[:-1], path[1:]):
            if u != v:
                visited_edges.add(tuple(sorted((u, v))))

    return float(len(visited_edges)) / float(G.number_of_edges())


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
    source_node_coverages = []
    source_edge_coverages = []

    for source in nodes:
        if method == "simple_rw":
            paths = simple_rw_trajectories(G, source, M, walk_len, rng)

        elif method == "budget_rw":
            paths = budget_matched_rw_trajectories(G, source, R, M, walk_len, rng)

        elif method == "node2vec":
            paths = budget_matched_node2vec_trajectories(
                G,
                source,
                R,
                M,
                walk_len,
                rng,
                p=1.0,
                q=0.5,
            )

        elif method == "non_backtracking_rw":
            paths = budget_matched_non_backtracking_trajectories(
                G,
                source,
                R,
                M,
                walk_len,
                rng,
            )

        elif method == "hrw":
            paths = hrw_trajectories(
                G,
                source,
                R,
                M,
                walk_len,
                rng,
                merge_terminal=True,
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        source_to_eat[source] = earliest_arrival_times(paths, source)

        source_cross_flags.append(
            bottleneck_cross_coverage(paths, bottleneck_edges)
        )

        source_node_coverages.append(
            node_coverage_ratio(G.number_of_nodes(), paths)
        )

        source_edge_coverages.append(
            edge_coverage_ratio(G, paths)
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
        "node_coverage": float(np.mean(source_node_coverages)),
        "edge_coverage": float(np.mean(source_edge_coverages)),
    }


# ============================================================
# Experiment
# ============================================================

def run_experiment(
    G,
    bottleneck_edges,
    graph_name,
    R_values=(0, 1, 2, 3, 4),
    methods=(
        "simple_rw",
        "budget_rw",
        "node2vec",
        "non_backtracking_rw",
        "hrw",
    ),
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

    methods = [
        "simple_rw",
        "budget_rw",
        "node2vec",
        "non_backtracking_rw",
        "hrw",
    ]

    all_rows = []

    all_rows += run_experiment(
        synthetic_G,
        synthetic_bottlenecks,
        graph_name="synthetic",
        R_values=[0, 1, 2, 3, 4],
        methods=methods,
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
        methods=methods,
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
# Aggregation
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
            bottleneck_cross_coverage=("bottleneck_cross_coverage", "mean"),
            std_bottleneck_cross_coverage=("bottleneck_cross_coverage", "std"),
            node_coverage=("node_coverage", "mean"),
            std_node_coverage=("node_coverage", "std"),
            edge_coverage=("edge_coverage", "mean"),
            std_edge_coverage=("edge_coverage", "std"),
        )
        .reset_index()
    )

    agg.to_csv("result_aggregated.csv", index=False)
    print("Saved aggregated result to result_aggregated.csv")

    return agg


# ============================================================
# Visualization
# ============================================================

def set_plot_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.titlesize": 12,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def plot_metric_panel(
    ax,
    data,
    graph,
    metric,
    ylabel,
    title,
    method_order,
    method_names,
    method_styles,
):
    sub_graph = data[data["graph"] == graph]

    std_map = {
        "mean_EAT": "std_EAT",
        "mean_ECC": "std_ECC",
        "bottleneck_cross_coverage": "std_bottleneck_cross_coverage",
        "node_coverage": "std_node_coverage",
        "edge_coverage": "std_edge_coverage",
    }

    for method in method_order:
        sub = sub_graph[sub_graph["method"] == method].sort_values("R")

        if len(sub) == 0:
            continue

        x = sub["R"].to_numpy()
        y = sub[metric].to_numpy()
        yerr = sub[std_map[metric]].fillna(0).to_numpy()

        style = method_styles[method]

        lw = 2.6 if method == "hrw" else 1.8
        alpha = 1.0 if method == "hrw" else 0.8

        ax.plot(
            x,
            y,
            label=method_names[method],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=lw,
            markersize=5.5,
            alpha=alpha,
        )

        ax.fill_between(
            x,
            y - yerr,
            y + yerr,
            color=style["color"],
            alpha=0.12,
            linewidth=0,
        )

    ax.set_title(title)
    ax.set_xlabel("Recurrent step $R$")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)
    ax.minorticks_on()

    if metric in {
        "bottleneck_cross_coverage",
        "node_coverage",
        "edge_coverage",
    }:
        y_vals = sub_graph[metric].to_numpy()
        ymin = max(0.0, np.nanmin(y_vals) - 0.05)
        ymax = min(1.0, np.nanmax(y_vals) + 0.05)

        if ymax - ymin < 0.12:
            center = 0.5 * (ymax + ymin)
            ymin = max(0.0, center - 0.06)
            ymax = min(1.0, center + 0.06)

        ax.set_ylim(ymin, ymax)


def plot_from_result_csv(csv_path="result.csv", out_prefix=""):
    set_plot_style()
    agg = load_and_aggregate_result(csv_path)

    method_order = [
        "simple_rw",
        "budget_rw",
        "node2vec",
        "non_backtracking_rw",
        "hrw",
    ]

    method_names = {
        "simple_rw": "RW",
        "budget_rw": "Budget-RW",
        "node2vec": "Node2Vec",
        "non_backtracking_rw": "NB-RW",
        "hrw": "HRW",
    }

    method_styles = {
        "simple_rw": {"marker": "o", "linestyle": ":", "color": "#85B7F4"},
        "budget_rw": {"marker": "s", "linestyle": "--", "color": "#50A8E8"},
        "node2vec": {"marker": "D", "linestyle": "-.", "color": "#6CBDE5"},
        "non_backtracking_rw": {"marker": "v", "linestyle": "--", "color": "#367F99"},
        "hrw": {"marker": "^", "linestyle": "--", "color": "#3F91EF"},
    }

    graph_order = ["synthetic", "real_world"]

    graph_titles = {
        "synthetic": "Synthetic graph",
        "real_world": "Real-world graph",
    }

    metrics = [
        ("bottleneck_cross_coverage", "Bottleneck Cross↑", "bottleneck"),
        ("node_coverage", "Node coverage ↑", "node_coverage"),
        ("edge_coverage", "Edge coverage ↑", "edge_coverage"),
        ("mean_EAT", "Mean EAT ↓", "eat"),
        ("mean_ECC", "Mean ECC ↓", "ecc"),
    ]

    for graph in graph_order:
        for metric, ylabel, tag in metrics:
            fig, ax = plt.subplots(figsize=(3.6, 2.6))

            plot_metric_panel(
                ax=ax,
                data=agg,
                graph=graph,
                metric=metric,
                ylabel=ylabel,
                title=graph_titles[graph],
                method_order=method_order,
                method_names=method_names,
                method_styles=method_styles,
            )

            ax.legend(frameon=False, fontsize=7, loc="best")
            plt.tight_layout()

            #fig.savefig(f"{out_prefix}_{graph}_{tag}.pdf", dpi=800)
            fig.savefig(f"{out_prefix}_{graph}_{tag}.png", dpi=800)
            plt.close(fig)

    print("Saved separate metric figures.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    save_result_csv("result.csv")
    plot_from_result_csv("result.csv")