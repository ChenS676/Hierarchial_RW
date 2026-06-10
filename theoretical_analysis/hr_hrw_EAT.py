#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt

from torch_cluster import random_walk as cluster_random_walk


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_undirected_edge_index(G: nx.Graph, device="cpu"):
    edges = []
    for u, v in G.edges():
        edges.append((u, v))
        edges.append((v, u))
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def add_self_loops_for_isolates(edge_index: torch.Tensor, num_nodes: int):
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    deg.scatter_add_(0, edge_index[0], torch.ones(edge_index.size(1), dtype=torch.long, device=edge_index.device))
    isolates = torch.where(deg == 0)[0]
    if isolates.numel() == 0:
        return edge_index
    loops = torch.stack([isolates, isolates], dim=0)
    return torch.cat([edge_index, loops], dim=1)


def anonymous_walk_code(walk: torch.Tensor):
    seen = {}
    code = []
    next_id = 0
    for x in walk.tolist():
        if x not in seen:
            seen[x] = next_id
            next_id += 1
        code.append(seen[x])
    return tuple(code)


def deduplicate_walks_by_anonymous_code(walks: torch.Tensor):
    keep = []
    seen_codes = set()
    for i, w in enumerate(walks):
        code = anonymous_walk_code(w)
        if code not in seen_codes:
            seen_codes.add(code)
            keep.append(i)
    if not keep:
        return walks[:0]
    return walks[torch.tensor(keep, dtype=torch.long, device=walks.device)]


@dataclass
class HRWResult:
    walks_by_level: list
    all_walks: torch.Tensor
    starts_by_level: list
    budget_transitions: int


@dataclass
class RWResult:
    walks: torch.Tensor
    starts: torch.Tensor
    budget_transitions: int


@torch.no_grad()
def hierarchical_random_walk_fast(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    R: int,
    L: int,
    M: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
    merge_terminal: bool = True,
    dedup_anonymous: bool = False,
):
    row, col = edge_index[0], edge_index[1]
    frontier = start_nodes.long()
    walks_by_level = []
    starts_by_level = []

    for _ in range(R + 1):
        starts = frontier.repeat_interleave(M)
        walks = cluster_random_walk(
            row=row,
            col=col,
            start=starts,
            walk_length=L,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )

        if dedup_anonymous:
            walks = deduplicate_walks_by_anonymous_code(walks)
            starts = walks[:, 0]

        walks_by_level.append(walks)
        starts_by_level.append(starts)

        terminals = walks[:, -1]
        frontier = torch.unique(terminals) if merge_terminal else terminals

    all_walks = torch.cat(walks_by_level, dim=0) if walks_by_level else torch.empty((0, L + 1), dtype=torch.long, device=edge_index.device)
    budget_transitions = int(all_walks.size(0) * L)
    return HRWResult(walks_by_level, all_walks, starts_by_level, budget_transitions)


@torch.no_grad()
def standard_random_walk_budget_matched(
    edge_index: torch.Tensor,
    start_nodes: torch.Tensor,
    total_budget_transitions: int,
    long_walk_length: int,
    p: float = 1.0,
    q: float = 1.0,
    num_nodes: int | None = None,
):
    row, col = edge_index[0], edge_index[1]
    T = int(long_walk_length)
    K = total_budget_transitions // T
    rem = total_budget_transitions % T

    walks = []
    starts_out = []

    if K > 0:
        starts_full = start_nodes.repeat(math.ceil(K / len(start_nodes)))[:K]
        full_walks = cluster_random_walk(
            row=row,
            col=col,
            start=starts_full,
            walk_length=T,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )
        walks.append(full_walks)
        starts_out.append(starts_full)

    if rem > 0:
        start_rem = start_nodes[(K % len(start_nodes)):(K % len(start_nodes)) + 1]
        rem_walk = cluster_random_walk(
            row=row,
            col=col,
            start=start_rem,
            walk_length=rem,
            p=p,
            q=q,
            coalesced=False,
            num_nodes=num_nodes,
        )
        walks.append(rem_walk)
        starts_out.append(start_rem)

    if not walks:
        out = torch.empty((0, T + 1), dtype=torch.long, device=edge_index.device)
        starts = torch.empty((0,), dtype=torch.long, device=edge_index.device)
    else:
        max_len = max(w.size(1) for w in walks)
        padded = []
        for w in walks:
            if w.size(1) < max_len:
                pad_val = w[:, -1:].repeat(1, max_len - w.size(1))
                w = torch.cat([w, pad_val], dim=1)
            padded.append(w)
        out = torch.cat(padded, dim=0)
        starts = torch.cat(starts_out, dim=0)

    return RWResult(out, starts, total_budget_transitions)


def coverage_ratio(num_nodes: int, walks: torch.Tensor):
    if walks.numel() == 0:
        return 0.0
    visited = torch.unique(walks.reshape(-1))
    return float(visited.numel()) / float(num_nodes)


def visited_nodes(walks: torch.Tensor):
    if walks.numel() == 0:
        return set()
    return set(torch.unique(walks.reshape(-1)).cpu().tolist())


def earliest_access_time_from_walks(walks: torch.Tensor, target: int):
    best = float("inf")
    for w in walks:
        hits = (w == target).nonzero(as_tuple=False)
        if hits.numel() > 0: 
            best = min(best, int(hits[0].item()))
    return best


def grouped_walks_by_source(walks: torch.Tensor, starts: torch.Tensor):
    groups = defaultdict(list)
    for s, w in zip(starts.tolist(), walks):
        groups[s].append(w)
    for k in groups:
        groups[k] = torch.stack(groups[k], dim=0)
    return groups


def cut_crossing_probability(walks: torch.Tensor, S_cut: set[int]):
    if walks.numel() == 0:
        return 0.0
    cnt = 0
    for w in walks.tolist():
        crossed = False
        for a, b in zip(w[:-1], w[1:]):
            if (a in S_cut) != (b in S_cut):
                crossed = True
                break
        cnt += int(crossed)
    return cnt / walks.size(0)


def build_barbell_graph(left_size=20, bridge_len=2):
    G = nx.barbell_graph(left_size, bridge_len)
    S_cut = set(range(left_size))
    return G, S_cut


def build_sbm_graph(sizes=(50, 50), p_in=0.12, p_out=0.01, seed=0):
    probs = [[p_in, p_out], [p_out, p_in]]
    G = nx.stochastic_block_model(list(sizes), probs, seed=seed)
    G = nx.convert_node_labels_to_integers(G)
    S_cut = set(range(sizes[0]))
    return G, S_cut


def build_lollipop_graph(clique_size=20, path_len=30):
    G = nx.lollipop_graph(clique_size, path_len)
    S_cut = set(range(clique_size))  # clique side
    return G, S_cut


def build_grid_graph(m=10, n=10):
    G = nx.grid_2d_graph(m, n)
    G = nx.convert_node_labels_to_integers(G)
    left_half = set()
    for idx in range(G.number_of_nodes()):
        row = idx // n
        col = idx % n
        if col < n // 2:
            left_half.add(idx)
    return G, left_half


def build_random_regular_graph(num_nodes=100, degree=4, seed=0):
    G = nx.random_regular_graph(degree, num_nodes, seed=seed)
    S_cut = set(range(num_nodes // 2))
    return G, S_cut


def build_er_graph(num_nodes=100, p=0.05, seed=0):
    G = nx.erdos_renyi_graph(num_nodes, p, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
    S_cut = set(range(G.number_of_nodes() // 2))
    return G, S_cut


def build_caveman_graph(num_cliques=5, clique_size=20):
    G = nx.connected_caveman_graph(num_cliques, clique_size)
    S_cut = set(range(clique_size))  # first community
    return G, S_cut


def build_watts_strogatz_graph(num_nodes=100, k=6, p=0.05, seed=0):
    G = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        G = nx.convert_node_labels_to_integers(G)
    S_cut = set(range(G.number_of_nodes() // 2))
    return G, S_cut


@torch.no_grad()
def run_experiment(
    G: nx.Graph,
    S_cut: set[int],
    R=3,
    L=4,
    M=2,
    p=1.0,
    q=1.0,
    num_seed_nodes=20,
    num_pairs=20,
    merge_terminal=True,
    dedup_anonymous=False,
    device="cpu",
):
    num_nodes = G.number_of_nodes()
    edge_index = to_undirected_edge_index(G, device=device)
    edge_index = add_self_loops_for_isolates(edge_index, num_nodes)

    left = sorted(list(S_cut))
    right = sorted(list(set(G.nodes()) - S_cut))

    num_seed_nodes = min(num_seed_nodes, len(left))
    num_pairs = min(num_pairs, num_seed_nodes, len(right))

    seeds = torch.tensor(random.sample(left, num_seed_nodes), dtype=torch.long, device=device)
    targets = random.sample(right, num_pairs)

    hrw = hierarchical_random_walk_fast(
        edge_index=edge_index,
        start_nodes=seeds,
        R=R,
        L=L,
        M=M,
        p=p,
        q=q,
        num_nodes=num_nodes,
        merge_terminal=merge_terminal,
        dedup_anonymous=dedup_anonymous,
    )

    total_budget = hrw.budget_transitions
    long_walk_length = (R + 1) * L

    rw = standard_random_walk_budget_matched(
        edge_index=edge_index,
        start_nodes=seeds,
        total_budget_transitions=total_budget,
        long_walk_length=long_walk_length,
        p=p,
        q=q,
        num_nodes=num_nodes,
    )


    hrw_cov = coverage_ratio(num_nodes, hrw.all_walks)
    rw_cov = coverage_ratio(num_nodes, rw.walks)

    hrw_cut = cut_crossing_probability(hrw.all_walks, S_cut)
    rw_cut = cut_crossing_probability(rw.walks, S_cut)

    hrw_by_src = grouped_walks_by_source(hrw.all_walks, torch.cat(hrw.starts_by_level, dim=0))
    rw_by_src = grouped_walks_by_source(rw.walks, rw.starts)

    hrw_eats, rw_eats = [], []
    for src, tgt in zip(seeds[:num_pairs].tolist(), targets):
        if src in hrw_by_src:
            hrw_eats.append(earliest_access_time_from_walks(hrw_by_src[src], tgt))
        if src in rw_by_src:
            rw_eats.append(earliest_access_time_from_walks(rw_by_src[src], tgt))

    def finite_mean(xs):
        xs = [x for x in xs if np.isfinite(x)]
        return float(np.mean(xs)) if xs else float("inf")

    results = {
        "R": R,
        "L": L,
        "M": M,
        "p": p,
        "q": q, 
        "num_nodes": num_nodes,
        "num_edges": G.number_of_edges(),
        "hrw_total_walks": int(hrw.all_walks.size(0)),
        "rw_total_walks": int(rw.walks.size(0)),
        "hrw_coverage": hrw_cov,
        "rw_coverage": rw_cov,
        "hrw_cut_crossing_prob": hrw_cut,
        "rw_cut_crossing_prob": rw_cut,
        "hrw_mean_eat": finite_mean(hrw_eats),
        "rw_mean_eat": finite_mean(rw_eats),
        "hrw_visited": visited_nodes(hrw.all_walks),
        "rw_visited": visited_nodes(rw.walks)
    }
    return results


def visualize_results(G, S_cut, results, title_prefix=""):
    pos = nx.spring_layout(G, seed=0)

    hrw_visited = results["hrw_visited"]
    rw_visited = results["rw_visited"]

    fig = plt.figure(figsize=(14, 10))

    # 1. Coverage ratio
    ax1 = plt.subplot(2, 2, 1)
    ax1.bar(["RW", "HRW"], [results["rw_coverage"], results["hrw_coverage"]])
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Coverage Ratio")
    ax1.set_ylabel("Fraction of visited nodes")

    # 2. Cut crossing rate
    ax2 = plt.subplot(2, 2, 2)
    ax2.bar(["RW", "HRW"], [results["rw_cut_crossing_prob"], results["hrw_cut_crossing_prob"]])
    ax2.set_ylim(0, 1.0)
    ax2.set_title("Cut-Crossing Rate")
    ax2.set_ylabel("Fraction of walks crossing cut")

    # 3. Mean EAT
    ax3 = plt.subplot(2, 2, 3)
    eat_vals = [results["rw_mean_eat"], results["hrw_mean_eat"]]
    ax3.bar(["RW", "HRW"], eat_vals)
    ax3.set_title("Mean Earliest Access Time (EAT)")
    ax3.set_ylabel("Steps")

    # 4. Combined graph visualization
    ax4 = plt.subplot(2, 2, 4)
    node_colors = []
    for v in G.nodes():
        if v in hrw_visited and v in rw_visited:
            node_colors.append("tab:purple")       # visited by both
        elif v in hrw_visited:
            node_colors.append("tab:blue")         # HRW only
        elif v in rw_visited:
            node_colors.append("tab:orange")       # RW only
        elif v in S_cut:
            node_colors.append("lightblue")
        else:
            node_colors.append("lightgray")

    nx.draw(G, pos, node_size=80, node_color=node_colors, with_labels=False, ax=ax4)
    ax4.set_title("Visited Nodes: RW (orange), HRW (blue), Both (purple)")

    plt.suptitle(f"{title_prefix}Budget-Matched RW vs HRW", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"rq1/{title_prefix}rw_vs_hrw.png")
    plt.show()


def main(graph_name: str, R: int):
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument("--graph", type=str, default="barbell",
    #                     choices=["barbell", "sbm", "lollipop", "grid",
    #                              "random_regular", "er", "caveman", "watts_strogatz"])
    # parser.add_argument("--R", type=int, default=3)
    parser.add_argument("--L", type=int, default=8)
    parser.add_argument("--M", type=int, default=3)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--num_seed_nodes", type=int, default=20)
    parser.add_argument("--num_pairs", type=int, default=20)
    parser.add_argument("--merge_terminal", action="store_true")
    parser.add_argument("--dedup_anonymous", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    if graph_name == "barbell":
        G, S_cut = build_barbell_graph(left_size=20, bridge_len=2)
    elif graph_name == "lollipop":
        G, S_cut = build_lollipop_graph(clique_size=20, path_len=30)
    elif graph_name == "grid":
        G, S_cut = build_grid_graph(m=10, n=10)
    elif graph_name == "random_regular":
        G, S_cut = build_random_regular_graph(num_nodes=100, degree=4, seed=args.seed)
    elif graph_name == "er":
        G, S_cut = build_er_graph(num_nodes=100, p=0.05, seed=args.seed)
    elif graph_name == "caveman":
        G, S_cut = build_caveman_graph(num_cliques=2, clique_size=20)
    elif graph_name == "watts_strogatz":
        G, S_cut = build_watts_strogatz_graph(num_nodes=100, k=6, p=0.05, seed=args.seed)
    else:
        G, S_cut = build_sbm_graph(sizes=(50, 50), p_in=0.12, p_out=0.01, seed=args.seed)

    results = run_experiment(
        G=G,
        S_cut=S_cut,
        R=R,
        L=args.L,
        M=args.M,
        p=args.p,
        q=args.q,
        num_seed_nodes=args.num_seed_nodes,
        num_pairs=args.num_pairs,
        merge_terminal=args.merge_terminal,
        dedup_anonymous=args.dedup_anonymous,
        device=args.device,
    )

    print("\n=== Budget-Matched HRW vs RW ===")
    for k, v in results.items():
        if k not in {"hrw_visited", "rw_visited", "seeds", "targets"}:
            print(f"{k}: {v}")

    visualize_results(G, S_cut, results, title_prefix=f"{graph_name}: ")

    return results 

if __name__ == "__main__":
    result_full = dict()
    for graph_name in ["barbell",
                       "sbm",
                       "lollipop",
                       "grid",
                       "random_regular",
                       "er",
                       "caveman",
                       "watts_strogatz"]:
        result_full[graph_name] = dict()          # initialise inner dict first
        for R in [1, 2, 3, 4, 5, 6]:
            print(f"\n\n=== Running experiment on {graph_name} graph ===")
            result_full[graph_name][R] = main(graph_name, R)

        # save result into csv with good formatting
        import pandas as pd
        df = pd.DataFrame.from_dict(result_full, orient="index")
        df.to_csv("hrw_vs_rw_results.csv")

