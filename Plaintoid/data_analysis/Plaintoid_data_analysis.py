"""
node_visit_counter.py
=====================
Standalone analysis script: how frequently is each node sampled
across all mini-batches of a link-prediction training set?

The script loads a graph, builds the same train/val/test edge splits
used by the HeART pipeline, iterates over training mini-batches, and
counts how many times each node appears as an endpoint node.

No model, no optimizer, no W&B — analysis only.

Usage
-----
    python node_visit_counter.py --data_name Cora --data_root ./data/Cora
    python node_visit_counter.py --data_name Cora --global_batch_size 64 \\
        --neg_sample_ratio 2 --save_fig cora_dashboard.pdf

Figures are always saved to disk (never plt.show()).
Default output path: node_visit_dashboard.pdf
"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch_geometric.transforms as T
import torch_geometric.data.data
import torch_geometric.data.storage

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, degree, remove_self_loops, to_undirected


# =============================================================================
# Argument parsing
# =============================================================================

def get_config() -> dict:
    parser = argparse.ArgumentParser(
        description="Node visit frequency analysis for link-prediction datasets"
    )
    parser.add_argument("--data_name",       type=str,   default="PubMed")
    parser.add_argument("--val_split_ratio", type=float, default=0.15)
    parser.add_argument("--test_split_ratio",type=float, default=0.05)
    parser.add_argument("--use_fixed_splits",action="store_true",
                        help="Load pre-saved splits from --split_dir")
    parser.add_argument("--global_batch_size",type=int,  default=256)
    parser.add_argument("--neg_sample_ratio",type=int,   default=1,
                        help="Negative edges per positive edge (mirrors training)")
    parser.add_argument("--seed",            type=int,   default=2025)
    parser.add_argument("--save_fig",        type=str,   default=None,
                        help="Path to save figures (e.g. cora_dashboard.pdf). "
                             "Defaults to node_visit_dashboard.pdf if omitted.")
    args = parser.parse_args()
    return vars(args)


# =============================================================================
# Data loading  (mirrors setup_data from the training pipeline)
# =============================================================================

def _check_edge_leakage(
    train_ei: torch.Tensor,
    val_pos: torch.Tensor,
    test_pos: torch.Tensor,
    num_nodes: int,
) -> None:
    """Print pairwise overlap rates between train / val / test positive edge sets.

    Edges are canonicalized to (min, max) so (u,v) and (v,u) are treated
    as the same undirected edge before comparison.
    """
    def canonical_keys(ei):
        lo = torch.minimum(ei[0], ei[1]).long()
        hi = torch.maximum(ei[0], ei[1]).long()
        return (lo * num_nodes + hi).unique()

    def count_intersect(a, b):
        combined = torch.cat([a, b]).sort().values
        return int((combined[:-1] == combined[1:]).sum())

    tr = canonical_keys(train_ei)
    va = canonical_keys(val_pos)
    te = canonical_keys(test_pos)

    tv = count_intersect(tr, va)
    tt = count_intersect(tr, te)
    vt = count_intersect(va, te)

    print(f"  Edge overlap  (train={tr.numel():,}  val={va.numel():,}  test={te.numel():,}):")
    print(f"    train ∩ val  : {tv:,}  ({100*tv/max(va.numel(),1):.2f}% of val)")
    print(f"    train ∩ test : {tt:,}  ({100*tt/max(te.numel(),1):.2f}% of test)")
    print(f"    val   ∩ test : {vt:,}  ({100*vt/max(te.numel(),1):.2f}% of test)")
    if tv > 0 or tt > 0 or vt > 0:
        print("    *** WARNING: data leakage detected ***")
    else:
        print("    Clean — no overlap found.")


def _rescue_isolated(
    train_ei: torch.Tensor,
    val_pos: torch.Tensor,
    test_pos: torch.Tensor,
    num_nodes: int,
) -> tuple:
    """For each node that has no training edge, borrow one positive val/test edge.

    Returns the augmented training edge_index and the number of rescued nodes.
    """
    deg = torch.zeros(num_nodes, dtype=torch.long)
    deg.index_add_(0, train_ei[0].cpu(), torch.ones(train_ei.size(1), dtype=torch.long))
    deg.index_add_(0, train_ei[1].cpu(), torch.ones(train_ei.size(1), dtype=torch.long))

    isolated = (deg == 0).nonzero(as_tuple=True)[0]
    if isolated.numel() == 0:
        return train_ei, 0

    rescued = []
    for node in isolated:
        nid = node.item()
        for pool in (val_pos, test_pos):
            mask = (pool[0] == nid) | (pool[1] == nid)
            if mask.any():
                idx = int(mask.nonzero(as_tuple=True)[0][0])
                rescued.append(pool[:, idx : idx + 1].cpu())
                break

    if not rescued:
        return train_ei, 0

    extra = torch.cat(rescued, dim=1)
    extra_both = torch.cat([extra, extra.flip(0)], dim=1).to(train_ei.device)
    return torch.cat([train_ei, extra_both], dim=1), len(rescued)


def load_data(config: dict, device: torch.device):
    """Load graph and build train/val/test edge splits.

    Returns
    -------
    num_nodes : int
    full_train_pos_edge_index : LongTensor [2, E_train]
    """
    if config["data_name"] in ("Cora", "PubMed", "CiteSeer"):
        dataset = Planetoid(root=f"./data/{config['data_name']}", name=config['data_name'])
        data = dataset[0].to(device)
    else:
        raise ValueError(
            f"Unknown dataset '{config['data_name']}'. "
            "Add your own loader here."
        )

    # --- Edge cleanup ---
    if data.is_directed():
        data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    n_isolated = int((deg == 0).sum())
    print(
        f"[{config['data_name']}] nodes={data.num_nodes}  "
        f"edges={data.edge_index.size(1)}  "
        f"avg_deg={deg.mean().item():.2f}  "
        f"max_deg={int(deg.max())}  "
        f"isolated={n_isolated}"
    )

    # --- Remove isolated nodes (mirrors training pipeline) ---
    nodes_before = data.num_nodes
    data = T.RemoveIsolatedNodes()(data)
    if data.num_nodes < nodes_before:
        print(f"RemoveIsolatedNodes: {nodes_before} → {data.num_nodes} nodes "
              f"(removed {nodes_before - data.num_nodes})")

    # --- Splits ---
    transform = T.RandomLinkSplit(
        num_val=config["val_split_ratio"],
        num_test=config["test_split_ratio"],
        is_undirected=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(data)
    full_train_pos_edge_index = train_data.edge_index.to(device)

    val_pos  = val_data.edge_label_index[:,  val_data.edge_label  == 1].to(device)
    test_pos = test_data.edge_label_index[:, test_data.edge_label == 1].to(device)
    _check_edge_leakage(full_train_pos_edge_index, val_pos, test_pos, data.num_nodes)
    full_train_pos_edge_index, n_rescued = _rescue_isolated(
        full_train_pos_edge_index, val_pos, test_pos, data.num_nodes
    )
    if n_rescued:
        print(f"  Guaranteed-coverage: rescued {n_rescued} node(s) by borrowing one val/test edge")

    print(f"Training positive edges: {full_train_pos_edge_index.size(1)}")
    return int(data.num_nodes), full_train_pos_edge_index


# =============================================================================
# Negative sampling  (mirrors sample_negative_edges from training pipeline)
# =============================================================================

def sample_negative_edges(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    device: torch.device,
) -> torch.Tensor:
    from torch_geometric.utils import negative_sampling
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method="sparse",
    ).to(device)


# =============================================================================
# NodeVisitCounter
# =============================================================================

class NodeVisitCounter:
    """Count how frequently each node is sampled across training mini-batches.

    In the HeART pipeline every mini-batch selects the unique endpoint nodes
    from positive and negative edges::

        all_nodes = torch.cat([
            batch_pos_edges[0], batch_pos_edges[1],
            batch_neg_edges[0], batch_neg_edges[1],
        ]).unique()

    This class records, for each node id, how many mini-batches it appeared
    in.  A count of 0 means the node received no gradient signal during
    training.

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.
    """

    def __init__(self, num_nodes: int) -> None:
        self.num_nodes = num_nodes
        self.counts = torch.zeros(num_nodes, dtype=torch.long)
        self.total_batches = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------

    def update(self, sampled_nodes: torch.Tensor) -> None:
        """Record one mini-batch of sampled endpoint nodes.

        Parameters
        ----------
        sampled_nodes : torch.Tensor, shape (K,)
            Unique node ids present in this batch.  May live on any device.
        """
        ids = sampled_nodes.cpu()
        self.counts.index_add_(
            0, ids, torch.ones(ids.size(0), dtype=torch.long)
        )
        self.total_batches += 1

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return scalar statistics over per-node visit counts.

        Returns
        -------
        dict
            total_batches   : int   — number of update() calls
            never_sampled   : int   — nodes with count == 0
            pct_never       : float — % of all nodes never sampled
            mean_count      : float — mean visits (sampled nodes only)
            median_count    : float — median visits (sampled nodes only)
            min_nonzero     : int   — minimum visits among sampled nodes
            max_count       : int   — maximum visits across all nodes
            p10             : float — 10th percentile (sampled nodes)
            p90             : float — 90th percentile (sampled nodes)
        """
        c = self.counts
        never   = int((c == 0).sum())
        sampled = c[c > 0].float()
        return {
            "total_batches" : self.total_batches,
            "never_sampled" : never,
            "pct_never"     : 100.0 * never / self.num_nodes,
            "mean_count"    : float(sampled.mean())        if sampled.numel() else 0.0,
            "median_count"  : float(sampled.median())      if sampled.numel() else 0.0,
            "min_nonzero"   : int(sampled.min())           if sampled.numel() else 0,
            "max_count"     : int(c.max()),
            "p10"           : float(sampled.quantile(0.10)) if sampled.numel() else 0.0,
            "p90"           : float(sampled.quantile(0.90)) if sampled.numel() else 0.0,
        }

    def never_sampled_nodes(self) -> torch.Tensor:
        """Return all node ids that were never sampled."""
        return torch.where(self.counts == 0)[0]

    def least_visited(self, k: int = 20) -> torch.Tensor:
        """Return the k node ids with the lowest non-zero visit count.

        Parameters
        ----------
        k : int
            Number of node ids to return, sorted ascending by count.
        """
        nonzero_mask = self.counts > 0
        ids   = torch.where(nonzero_mask)[0]
        vals  = self.counts[ids]
        order = vals.argsort()
        return ids[order[:k]]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a human-readable frequency summary to stdout."""
        s = self.summary()
        print("=" * 54)
        print(f"  Node sampling frequency  —  {s['total_batches']} batches")
        print(f"  {self.num_nodes} total nodes")
        print("=" * 54)
        print(f"  Never sampled  : {s['never_sampled']:>8,d}  ({s['pct_never']:.1f}%)")
        print(f"  Min (non-zero) : {s['min_nonzero']:>8,d}")
        print(f"  P10            : {s['p10']:>11.1f}")
        print(f"  Median         : {s['median_count']:>11.1f}")
        print(f"  Mean           : {s['mean_count']:>11.1f}")
        print(f"  P90            : {s['p90']:>11.1f}")
        print(f"  Max            : {s['max_count']:>8,d}")
        print("=" * 54)

    def plot_histogram(
        self,
        save_path: Optional[str] = None,
        bins: int = 60,
        log_y: bool = True,
    ) -> None:
        """Plot the per-node visit-count distribution (histogram + CDF).

        Parameters
        ----------
        save_path : str, optional
            Save figure to this path.  Defaults to node_visit_histogram.pdf.
        bins : int
            Number of histogram bins.
        log_y : bool
            Use log scale on the histogram y-axis (recommended — the
            distribution is typically very right-skewed).
        """
        c = self.counts.numpy()
        s = self.summary()

        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
        fig.subplots_adjust(wspace=0.35)

        # Left: full distribution including zeros
        ax = axes[0]
        ax.hist(c, bins=bins, color="#378ADD", edgecolor="none")
        if log_y:
            ax.set_yscale("log")
        ax.axvline(
            s["mean_count"], color="#E24B4A", linewidth=1.0,
            linestyle="--", label=f"mean={s['mean_count']:.1f}"
        )
        ax.set_xlabel("visit count", fontsize=9)
        ax.set_ylabel("# nodes (log)" if log_y else "# nodes", fontsize=9)
        ax.set_title("All nodes", fontsize=9, fontweight="500")
        ax.legend(fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # Right: CDF over sampled nodes only
        ax2 = axes[1]
        sampled = np.sort(c[c > 0])
        cdf = np.arange(1, len(sampled) + 1) / len(sampled)
        ax2.plot(sampled, cdf, color="#378ADD", linewidth=1.2)
        ax2.axvline(s["p10"], color="#888780", linewidth=0.8,
                    linestyle=":", label="P10")
        ax2.axvline(s["p90"], color="#888780", linewidth=0.8,
                    linestyle="--", label="P90")
        ax2.set_xlabel("visit count", fontsize=9)
        ax2.set_ylabel("CDF", fontsize=9)
        ax2.set_title("Sampled nodes only", fontsize=9, fontweight="500")
        ax2.legend(fontsize=8)
        ax2.tick_params(labelsize=8)
        ax2.spines[["top", "right"]].set_visible(False)

        fig.suptitle(
            f"Node visit frequency  |  "
            f"{s['pct_never']:.1f}% never sampled  |  "
            f"{s['total_batches']} batches",
            fontsize=9,
        )

        out = save_path or "node_visit_histogram.pdf"
        fig.savefig(out, bbox_inches="tight", dpi=200)
        print(f"Saved: {out}")
        plt.close(fig)


# =============================================================================
# ADD YOUR ANALYSIS FUNCTIONS BELOW THIS LINE
# =============================================================================
#
# Each function should accept `counter: NodeVisitCounter` as its first
# argument, plus any extra parameters you need.  Add a call to it in main().
#
# =============================================================================

def plot_degree_vs_frequency(
    counter: NodeVisitCounter,
    train_edge_index: torch.Tensor,
    save_path: Optional[str] = None,
    sample_size: int = 2000,
    seed: int = 0,
) -> None:
    """Scatter plot of node degree vs visit count.

    Shows whether low-degree nodes are systematically under-sampled relative
    to high-degree hub nodes.  Both axes use log scale to handle the wide
    dynamic range typical of power-law degree distributions.

    Parameters
    ----------
    counter : NodeVisitCounter
        Populated counter (call update() for all batches first).
    train_edge_index : torch.Tensor, shape (2, E)
        Training edge index used to compute per-node degree.
    save_path : str, optional
        Save figure to this path.  Defaults to node_degree_vs_frequency.pdf.
    sample_size : int
        Max nodes to plot (random subsample if graph is large).
    seed : int
        RNG seed for reproducible subsampling.
    """
    num_nodes = counter.num_nodes
    src = train_edge_index[0].cpu()
    dst = train_edge_index[1].cpu()
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.index_add_(0, src, torch.ones_like(src))
    degrees.index_add_(0, dst, torch.ones_like(dst))

    rng = np.random.default_rng(seed)
    idx = rng.choice(num_nodes, size=min(sample_size, num_nodes), replace=False)
    d = degrees[idx].numpy()
    c = counter.counts[idx].numpy()

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.scatter(d + 0.5, c + 0.5, s=8, alpha=0.4, color="#378ADD",
               linewidths=0, rasterized=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("node degree (log)", fontsize=9)
    ax.set_ylabel("visit count (log)", fontsize=9)
    ax.set_title("Degree vs visit count", fontsize=9, fontweight="500")
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = save_path or "node_degree_vs_frequency.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)


def plot_full_dashboard(
    counter: NodeVisitCounter,
    train_edge_index: torch.Tensor,
    data_name: str = "",
    save_path: Optional[str] = None,
    scatter_sample: int = 2000,
    seed: int = 0,
) -> None:
    """Three-panel dashboard: histogram + CDF + degree-vs-frequency scatter.

    Produces the same layout shown in the interactive preview:
      - Panel 1: log-scale histogram of visit counts (all nodes)
      - Panel 2: CDF of visit counts (sampled nodes only) with P10/P90 markers
      - Panel 3: degree vs visit count scatter (log-log)

    Parameters
    ----------
    counter : NodeVisitCounter
        Populated counter (call update() for all batches first).
    train_edge_index : torch.Tensor, shape (2, E)
        Training edge index used to compute per-node degree.
    save_path : str, optional
        Save figure to this path.  Defaults to node_visit_dashboard.pdf.
    scatter_sample : int
        Max nodes to show in the scatter panel.
    seed : int
        RNG seed for reproducible subsampling of the scatter panel.
    """
    s = counter.summary()
    c = counter.counts.numpy()

    # --- degree for scatter ---
    num_nodes = counter.num_nodes
    src = train_edge_index[0].cpu()
    dst = train_edge_index[1].cpu()
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    degrees.index_add_(0, src, torch.ones_like(src))
    degrees.index_add_(0, dst, torch.ones_like(dst))

    rng = np.random.default_rng(seed)
    idx = rng.choice(num_nodes, size=min(scatter_sample, num_nodes), replace=False)
    sc_deg = degrees[idx].numpy()
    sc_cnt = counter.counts[idx].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.8))
    fig.subplots_adjust(wspace=0.38)

    # Panel 1: histogram
    ax = axes[0]
    ax.hist(c, bins=50, color="#378ADD", edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(s["mean_count"], color="#E24B4A", linewidth=1.0,
               linestyle="--", label=f"mean={s['mean_count']:.1f}")
    ax.set_xlabel("visit count", fontsize=9)
    ax.set_ylabel("# nodes (log)", fontsize=9)
    ax.set_title("All nodes", fontsize=9, fontweight="500")
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: CDF
    ax2 = axes[1]
    sampled = np.sort(c[c > 0])
    cdf = np.arange(1, len(sampled) + 1) / len(sampled)
    ax2.plot(sampled, cdf, color="#378ADD", linewidth=1.2)
    ax2.axvline(s["p10"], color="#888780", linewidth=0.8,
                linestyle=":", label=f"P10={s['p10']:.0f}")
    ax2.axvline(s["p90"], color="#888780", linewidth=0.8,
                linestyle="--", label=f"P90={s['p90']:.0f}")
    ax2.set_xlabel("visit count", fontsize=9)
    ax2.set_ylabel("CDF", fontsize=9)
    ax2.set_title("Sampled nodes only", fontsize=9, fontweight="500")
    ax2.legend(fontsize=8)
    ax2.tick_params(labelsize=8)
    ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3: degree vs visit count
    ax3 = axes[2]
    ax3.scatter(sc_deg + 0.5, sc_cnt + 0.5, s=8, alpha=0.4,
                color="#378ADD", linewidths=0, rasterized=True)
    # ax3.set_xscale("log")
    # ax3.set_yscale("log")
    ax3.set_xlabel("node degree (log)", fontsize=9)
    ax3.set_ylabel("visit count (log)", fontsize=9)
    ax3.set_title("Degree vs visit count", fontsize=9, fontweight="500")
    ax3.tick_params(labelsize=8)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Node visit frequency  |  {s['pct_never']:.1f}% never sampled  |  "
        f"{s['total_batches']} batches  |  {num_nodes} nodes",
        fontsize=9,
    )

    out = save_path or f"{data_name}_node_visit_dashboard.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()

    torch.manual_seed(config["seed"])

    num_nodes, full_train_pos_edge_index = load_data(config, device)

    counter = NodeVisitCounter(num_nodes=num_nodes)

    batch_size = config["global_batch_size"]
    loader = DataLoader(
        TensorDataset(full_train_pos_edge_index.t()),
        batch_size=batch_size,
        shuffle=True,
    )

    print(f"\nCounting node visits over {math.ceil(full_train_pos_edge_index.size(1) / batch_size)} batches ...")

    for (batch_pos_t,) in loader:
        batch_pos_edges = batch_pos_t.t().to(device)

        batch_neg_edges = sample_negative_edges(
            pos_edge_index=full_train_pos_edge_index,
            num_nodes=num_nodes,
            num_neg_samples=batch_pos_edges.size(1) * config["neg_sample_ratio"],
            device=device,
        )

        all_nodes = torch.cat([
            batch_pos_edges[0], batch_pos_edges[1],
            batch_neg_edges[0], batch_neg_edges[1],
        ]).unique()

        counter.update(all_nodes)

    counter.report()

    never = counter.never_sampled_nodes()
    if never.numel() == 0:
        print(">> All nodes were sampled at least once during training.")
    else:
        print(f">> WARNING: {never.numel():,} / {num_nodes:,} nodes "
              f"({100.0 * never.numel() / num_nodes:.1f}%) were NEVER sampled during training.")
        src = full_train_pos_edge_index[0].cpu()
        dst = full_train_pos_edge_index[1].cpu()
        train_deg = torch.zeros(num_nodes, dtype=torch.long)
        train_deg.index_add_(0, src, torch.ones(src.size(0), dtype=torch.long))
        train_deg.index_add_(0, dst, torch.ones(dst.size(0), dtype=torch.long))
        n_isolated = int((train_deg[never] == 0).sum())
        print(f"   {n_isolated:,} have no training edges (isolated after train/val/test split)")
        if never.numel() - n_isolated > 0:
            print(f"   {never.numel() - n_isolated:,} have training edges but were randomly not hit")

    # --- plots ---
    # Three-panel dashboard (histogram + CDF + degree scatter)
    dashboard_path = config["save_fig"]  # e.g. "cora_dashboard.pdf"
    plot_full_dashboard(
        counter,
        full_train_pos_edge_index,
        data_name=config["data_name"],
        save_path=dashboard_path,
    )

    # ------------------------------------------------------------------
    # Call your additional analysis functions here, e.g.:
    #   plot_degree_vs_frequency(counter, full_train_pos_edge_index,
    #                            save_path="cora_degree_scatter.pdf")
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()