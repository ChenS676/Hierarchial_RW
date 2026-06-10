"""
node_visit_counter.py
=====================
Standalone analysis script: how frequently is each node sampled
across all mini-batches of a link-prediction training set?

The script loads a graph, builds the same train/val/test edge splits
used by the HeART pipeline, iterates over training mini-batches, and
counts how many times each node appears as an endpoint node.

No model, no optimizer, no W&B — analysis only.

Supported datasets
------------------
  Planetoid : Cora, CiteSeer, PubMed
  OGB       : ogbl-collab, ogbl-ddi, ogbl-citation2, ogbl-ppa, ...

Usage
-----
    python node_visit_counter.py --data_name PubMed --data_root ./data/PubMed
    python node_visit_counter.py --data_name ogbl-collab --data_root ./data/ogbl-collab
    python node_visit_counter.py --data_name ogbl-collab --global_batch_size 1024 \\
        --neg_sample_ratio 1 --save_fig ogbl_collab_dashboard.pdf

Figures are always saved to disk (never plt.show()).
Default output path: <data_name>_node_visit_dashboard.pdf
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
import torch_geometric.data.data
import torch_geometric.data.storage
import torch_geometric.transforms as T
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, degree, remove_self_loops, to_undirected

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])
from ogb.linkproppred import PygLinkPropPredDataset

_original_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage
])

# =============================================================================
# Argument parsing
# =============================================================================

def get_config() -> dict:
    parser = argparse.ArgumentParser(
        description="Node visit frequency analysis for link-prediction datasets"
    )
    parser.add_argument("--data_name",       type=str,   default="ogbl-citation2")
#    parser.add_argument("--data_root",       type=str,   default="./data/ogbl-citation2")
    parser.add_argument("--val_split_ratio", type=float, default=0.15)
    parser.add_argument("--test_split_ratio",type=float, default=0.05)
    parser.add_argument("--use_fixed_splits",action="store_true",
                        help="Load pre-saved splits from --split_dir")
    parser.add_argument("--split_dir",       type=str,   default="./data/ogbl-citation2/fixed_splits")
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


def load_data(config: dict, device: torch.device):
    """Load graph and build train/val/test edge splits.

    Returns
    -------
    num_nodes : int
    full_train_pos_edge_index : LongTensor [2, E_train]
    ei : LongTensor [2, E_graph]  — full-graph edge_index for neg-sample masking
    neg_method : str              — 'dense' or 'sparse' for negative_sampling()
    """
    data_name = config["data_name"]
    neg_method = "sparse"

    if data_name == "ogbl-ddi":
        # ddi has no node features (embeddings are learned) and the graph fills
        # ~66% of the possible edge space, so dense negative sampling is
        # required.  Load via ToSparseTensor so adj_t holds the full graph
        # topology (data.edge_index is absent after this transform).
        dataset = PygLinkPropPredDataset(name=data_name, root=config["data_root"],
                                         transform=T.ToSparseTensor())
        data = dataset[0].to(device)
        split_edge = dataset.get_edge_split()
        full_train_pos_edge_index = split_edge["train"]["edge"].t().to(device)
        # adj_t stores edges as (dst, src); flip to conventional (src, dst)
        row, col, _ = data.adj_t.coo()
        ei = torch.stack([col, row], dim=0)
        data.num_nodes = data.adj_t.size(0)
        is_undirected = True
        neg_method = "dense"
        val_pos  = split_edge["valid"]["edge"].t().to(device)
        test_pos = split_edge["test"]["edge"].t().to(device)

    elif data_name.startswith("ogbl-"):
        # citation2 is directed; collab/ppa are undirected
        is_undirected = data_name != "ogbl-citation2"

        dataset = PygLinkPropPredDataset(name=data_name, root=config["data_root"])
        data = dataset[0].to(device)
        split_edge = dataset.get_edge_split()
        train_split = split_edge["train"]

        # ogbl-citation2 stores edges as separate source/target tensors
        if "source_node" in train_split:
            src = train_split["source_node"].to(device)
            dst = train_split["target_node"].to(device)
            full_train_pos_edge_index = torch.stack([src, dst], dim=0)
        else:
            full_train_pos_edge_index = train_split["edge"].t().to(device)

        # ogbl-collab: benchmark convention adds validation edges to the graph
        # adjacency (val edges are temporally before the test cutoff)
        if data_name == "ogbl-collab" and "valid" in split_edge:
            val_ei = split_edge["valid"]["edge"].t().to(device)
            ei = torch.cat([full_train_pos_edge_index, val_ei], dim=1)
        elif data.edge_index is not None:
            ei = data.edge_index
        else:
            ei = full_train_pos_edge_index

        if data.num_nodes is None:
            data.num_nodes = int(
                max(full_train_pos_edge_index.max().item(),
                    ei.max().item())
            ) + 1

        if "source_node" in split_edge.get("valid", {}):
            val_pos  = torch.stack([split_edge["valid"]["source_node"].to(device),
                                    split_edge["valid"]["target_node"].to(device)], dim=0)
            test_pos = torch.stack([split_edge["test"]["source_node"].to(device),
                                    split_edge["test"]["target_node"].to(device)], dim=0)
        else:
            val_pos  = split_edge["valid"]["edge"].t().to(device)
            test_pos = split_edge["test"]["edge"].t().to(device)

    elif data_name in ("Cora", "PubMed", "CiteSeer"):
        is_undirected = True
        dataset = Planetoid(root=config["data_root"], name=data_name)
        data = dataset[0].to(device)
        ei = data.edge_index

        if config["use_fixed_splits"]:
            import os
            split_path = os.path.join(
                config["split_dir"], f"{data_name}_fixed_split.pt"
            )
            sd = torch.load(split_path, map_location=device, weights_only=False)
            full_train_pos_edge_index = sd["train"]["edge_index"].to(device)
            val_pos  = sd["val"]["edge_label_index"][:,  sd["val"]["edge_label"]  == 1].to(device)
            test_pos = sd["test"]["edge_label_index"][:, sd["test"]["edge_label"] == 1].to(device)
        else:
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

    else:
        raise ValueError(
            f"Unknown dataset '{data_name}'. "
            "Use a Planetoid name (Cora/CiteSeer/PubMed) or an OGB name (ogbl-*)."
        )

    # --- Leakage check ---
    _check_edge_leakage(full_train_pos_edge_index, val_pos, test_pos, data.num_nodes)

    # --- Edge cleanup for graph stats ---
    if is_undirected:
        ei = to_undirected(ei)
    ei, _ = coalesce(ei, None, num_nodes=data.num_nodes)
    ei, _ = remove_self_loops(ei)

    deg = degree(ei[1], data.num_nodes, dtype=torch.float)
    x_shape = tuple(data.x.shape) if data.x is not None else None
    print(
        f"[{data_name}] nodes={data.num_nodes}  "
        f"edges={ei.size(1)}  "
        f"avg_deg={deg.mean().item():.2f}  "
        f"max_deg={int(deg.max())}  "
        f"isolated={(deg == 0).sum().item()}  "
        f"x={x_shape}"
    )

    print(f"Training positive edges: {full_train_pos_edge_index.size(1)}")
    return data.num_nodes, full_train_pos_edge_index, ei, neg_method


# =============================================================================
# Negative sampling  (mirrors sample_negative_edges from training pipeline)
# =============================================================================

def sample_negative_edges(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    device: torch.device,
    method: str = "sparse",
) -> torch.Tensor:
    from torch_geometric.utils import negative_sampling
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method=method,
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
        self.counts += torch.bincount(sampled_nodes.cpu(), minlength=self.num_nodes)
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
            "percentage_never"     : 100.0 * never / self.num_nodes,
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
        print(f"  Never sampled  : {s['never_sampled']:>8,d}  ({s['percentage_never']:.1f}%)")
        print(f"  Min (non-zero) : {s['min_nonzero']:>8,d}")
        print(f"  P10            : {s['p10']:>11.1f}")
        print(f"  Median         : {s['median_count']:>11.1f}")
        print(f"  Mean           : {s['mean_count']:>11.1f}")
        print(f"  P90            : {s['p90']:>11.1f}")
        print(f"  Max            : {s['max_count']:>8,d}")
        print("=" * 54)



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

    prefix = f"{data_name}  |  " if data_name else ""
    fig.suptitle(
        f"{prefix}Node visit frequency  |  {s['percentage_never']:.1f}% never sampled  |  "
        f"{s['total_batches']} batches  |  {num_nodes} nodes",
        fontsize=9,
    )

    out = save_path or "node_visit_dashboard.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print(f"Saved: {out}")
    plt.close(fig)


# =============================================================================
# main
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    config['data_root'] = config.get('data_root') or f"./data/{config['data_name']}"
    print(f"PyTorch  : {torch.__version__}")
    print(f"Loading  : {config['data_name']} from {config['data_root']} ...")

    torch.manual_seed(config["seed"])
    num_nodes, full_train_pos_edge_index, ei, neg_method = load_data(config, device)
    counter = NodeVisitCounter(num_nodes=num_nodes)
    batch_size = config["global_batch_size"]
    neg_ratio  = config["neg_sample_ratio"]
    total_pos  = full_train_pos_edge_index.size(1)

    # Pre-sample ALL negatives once — avoids rebuilding the sparse adjacency
    # matrix on every batch (was the dominant bottleneck for large datasets).
    # ei is the full-graph edge_index used as the exclusion mask; for ogbl-ddi
    # this is the complete adj_t (not just training edges) with method='dense'.
    print(f"\nPre-sampling {total_pos * neg_ratio:,} negative edges ...")
    all_neg_edge_index = sample_negative_edges(
        pos_edge_index=ei,
        num_nodes=num_nodes,
        num_neg_samples=total_pos * neg_ratio,
        device=device,
        method=neg_method,
    )

    pos_loader = DataLoader(
        TensorDataset(full_train_pos_edge_index.t()),
        batch_size=batch_size,
        shuffle=True,
    )
    
    neg_loader = DataLoader(
        TensorDataset(all_neg_edge_index.t()),
        batch_size=batch_size * neg_ratio,
        shuffle=False,
    )

    num_batches = math.ceil(total_pos / batch_size)
    print(f"Counting node visits over {num_batches} batches ...")
    for (batch_pos_t,), (batch_neg_t,) in zip(pos_loader, neg_loader):
        batch_pos_edges = batch_pos_t.t().to(device)
        batch_neg_edges = batch_neg_t.t().to(device)
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
    name = config["data_name"]
    dashboard_path = config["save_fig"] or f"{name}_node_visit_dashboard.pdf"
    plot_full_dashboard(
        counter,
        full_train_pos_edge_index,
        data_name=name,
        save_path=dashboard_path,
    )


if __name__ == "__main__":
    main()

# python obgl_analysis.py --data_name ogbl-ddi    --data_root ./data/ogbl-ddi    --save_fig ogbl-ddi_node_visit_dashboard.pdf 
# python obgl_analysis.py --data_name ogbl-collab --data_root ./data/ogbl-collab --save_fig ogbl-collab_node_visit_dashboard.pdf 
# python obgl_analysis.py --data_name ogbl-ppa    --data_root ./data/ogbl-ddi    --save_fig ogbl-ppa_node_visit_dashboard.pdf 
