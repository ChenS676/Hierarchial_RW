import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import graphbench
from tqdm import tqdm
import wandb
import math
import argparse
import dataclasses
from einops import rearrange
from typing import Optional, Tuple, List
from torch_cluster import random_walk as cluster_random_walk
import random
import numpy as np
from muon import SingleDeviceMuonWithAuxAdam
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add
import time
import torch.compiler

# ==========================================
# 0. UTILS
# ==========================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification — handles the severe class imbalance
    in MST tasks (MST edges are a small minority of all edges).
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs   = inputs.view(-1)
        targets  = targets.view(-1)
        bce      = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt       = torch.exp(-bce)
        f_loss   = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean': return f_loss.mean()
        if self.reduction == 'sum':  return f_loss.sum()
        return f_loss


class AddUndirectedContext(object):
    """
    Builds an undirected message-passing graph alongside the original
    directed edge_index/edge_attr, and computes a simple degree feature.

    For MST, the original directed edge_attr (edge weights) is left
    completely untouched — only mp_edge_index / mp_edge_attr are added.
    """
    def __call__(self, data):
        mp_edge_index, mp_edge_attr = to_undirected(
            data.edge_index,
            data.edge_attr,
            num_nodes=data.num_nodes,
            reduce="mean"
        )
        data.mp_edge_index = mp_edge_index
        data.mp_edge_attr  = mp_edge_attr

        row    = mp_edge_index[0]
        degree = scatter_add(
            torch.ones(mp_edge_index.size(1), device=data.edge_index.device),
            row, dim=0, dim_size=data.num_nodes
        )
        # Concatenate degree with any existing node features
        if data.x is not None and data.x.dim() > 1:
            data.x = torch.cat([data.x.float(), degree.unsqueeze(-1)], dim=-1)
        else:
            data.x = degree.unsqueeze(-1)
        return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")


def parse_metrics(metrics):
    """Returns (acc, f1) from graphbench evaluator output regardless of format."""
    if isinstance(metrics, dict):
        f1  = metrics.get('f1', metrics.get('F1', 0.0))
        acc = metrics.get('accuracy', metrics.get('acc', f1))
        return acc, f1
    if isinstance(metrics, (list, tuple)):
        acc = metrics[0] if len(metrics) > 0 else 0.0
        f1  = metrics[1] if len(metrics) > 1 else acc
        return acc, f1
    return metrics, metrics


def build_run_name(config: dict) -> str:
    """
    Constructs a human-readable W&B run name encoding key hyperparameters.
    Example: mst_easy_h256_L1_H16_w8x8_rs1_rwse16_focal_s2025
    """
    pe_tag   = f"_{config['pe_type']}{config['pe_dim']}" if config["use_pe"] else "_nope"
    loss_tag = "_focal" if config.get("use_focal_loss", True) else "_bce"
    return (
        f"{config['dataset_name']}"
        f"_h{config['hidden_dim']}"
        f"_L{config['layers']}"
        f"_H{config['num_heads']}"
        f"_w{config['num_walks']}x{config['walk_length']}"
        f"_rs{config['recurrent_steps']}"
        f"{pe_tag}"
        f"{loss_tag}"
        f"_s{config['seed']}"
    )


# ==========================================
# 1. ARCHITECTURE COMPONENTS
# ==========================================

@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim, self.base, self.max_seq_len = dim, base, max_seq_len
        self.rope_init()

    def reset_parameters(self): self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2)[: self.dim // 2].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        idx   = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        cache = torch.stack([
            torch.cos(torch.einsum("i,j->ij", idx, self.theta)),
            torch.sin(torch.einsum("i,j->ij", idx, self.theta)),
        ], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        rope_cache = self.cache[:x.size(1)] if input_pos is None else self.cache[input_pos]
        xshaped    = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack([
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ], -1).flatten(3)
        return x_out.type_as(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, seq_len, n_layer,
                 attn_dropout_p=0.0, ffn_dropout_p=0.0, resid_dropout_p=0.0,
                 drop_path_p=0.0, config=MupConfig()):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim      = hidden_dim
        self.num_heads       = num_heads
        self.resid_dropout_p = resid_dropout_p
        self.ffn_dropout_p   = ffn_dropout_p
        self.drop_path_p     = drop_path_p

        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight  = nn.Parameter(torch.ones(hidden_dim))
        self.qkv  = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=(hidden_dim // num_heads), max_seq_len=seq_len)

        s = config.init_std
        m = config.mup_width_multiplier
        torch.nn.init.normal_(self.up.weight,   std=s / math.sqrt(2 * n_layer * m))
        torch.nn.init.normal_(self.gate.weight, std=s / math.sqrt(2 * n_layer * m))
        torch.nn.init.normal_(self.down.weight, std=s / math.sqrt(m))
        torch.nn.init.normal_(self.qkv.weight,  std=s / math.sqrt(m))
        torch.nn.init.normal_(self.o.weight,    std=s / math.sqrt(m))

    def forward(self, x, offset=None):
        attnx   = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        q, k, v = self.qkv(attnx).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d') for t in (q, k, v)]
        o_walks  = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / k.shape[-1])
        attn_out = self.o(rearrange(o_walks, 'n h t d -> n t (h d)', h=self.num_heads))
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)

        ffnx    = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(ffnx)) * self.gate(ffnx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        return x + self._drop_path(ffn_out)

    def _drop_path(self, x):
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_path_p
        shape     = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep_prob) * torch.floor(keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device))


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads,
                 num_walks, seq_len, attn_dropout_p=0.0, ffn_dropout_p=0.0,
                 resid_dropout_p=0.0, drop_path_p=0.0, config=MupConfig()):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads, num_walks * seq_len,
                n_layer=num_layers, attn_dropout_p=attn_dropout_p,
                ffn_dropout_p=ffn_dropout_p, resid_dropout_p=resid_dropout_p,
                drop_path_p=drop_path_p, config=config
            ) for _ in range(num_layers)
        ])
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, anon_indices, source_nodes=None):
        _, ctx_len, _ = x.shape
        for depth, idx in enumerate(reversed(anon_indices)):
            for layer in self.layers:
                x = layer(x, idx)
            if depth < len(anon_indices) - 1:
                x = F.rms_norm(x[:, -1, :], [self.hidden_dim], eps=1e-5) * self.norm_weight
                x = rearrange(x, '(n t) z -> n t z', t=ctx_len)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return F.normalize(x[:, -1, :], dim=-1)


# ==========================================
# 2. RANDOM WALK UTILITIES
# ==========================================

@torch.no_grad()
def anonymize_rws(rws, rev_walks=True):
    if rev_walks:
        rws = torch.flip(rws, dims=[-1])
    s, _  = torch.sort(rws, dim=-1)
    su    = torch.searchsorted(s, rws)
    c     = torch.full_like(s, fill_value=s.shape[-1])
    rw_i  = torch.arange(rws.shape[-1], device=rws.device)[None, :].expand_as(s)
    first = c.scatter_reduce_(-1, su, rw_i, reduce="amin")
    ret   = first.gather(-1, su)
    if rev_walks:
        ret = torch.flip(ret, dims=[-1])
    return ret


def get_walk_edge_attrs(edge_index, edge_attr, walks, num_nodes):
    device              = edge_index.device
    row, col            = edge_index
    edge_hashes         = row.to(torch.long) * num_nodes + col.to(torch.long)
    sorted_hashes, perm = torch.sort(edge_hashes)
    src = walks[:, :, :-1].flatten().to(torch.long)
    dst = walks[:, :, 1:].flatten().to(torch.long)
    idx = perm[torch.searchsorted(sorted_hashes, src * num_nodes + dst)]
    b, w, l  = walks.shape
    edge_dim = edge_attr.size(-1)
    return torch.cat([
        torch.zeros((b, w, 1, edge_dim), device=device, dtype=edge_attr.dtype),
        edge_attr[idx].view(b, w, l - 1, edge_dim)
    ], dim=2)


@torch.compiler.disable
def get_random_walk_batch(
    edge_index: torch.Tensor,
    x: torch.Tensor,
    start_nodes: torch.Tensor,
    walk_length: int,
    num_walks: int,
    num_nodes: int,
    recurrent_steps: int = 1,
    p: float = 1.0,
    q: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    with torch.no_grad():
        row, col        = edge_index
        current_sources = start_nodes
        rws_list        = []
        for _ in range(recurrent_steps):
            sources_repeated = current_sources.repeat_interleave(num_walks)
            walks = cluster_random_walk(
                row, col, sources_repeated, walk_length - 1, p=p, q=q, num_nodes=num_nodes
            )
            rws = torch.flip(walks.view(current_sources.size(0), num_walks, walk_length).flatten(1, 2), dims=[-1])
            rws_list.append(rws)
            if recurrent_steps > 1:
                current_sources = rws.reshape(-1)
        anon_indices = [anonymize_rws(r, rev_walks=True) for r in rws_list]
    return x[rws_list[-1]], anon_indices, rws_list


# ==========================================
# 3. DECODER
# ==========================================

class EdgeDecoderWithFeatures(nn.Module):
    """
    Predicts edge labels from (src ⊙ dst) concatenated with the edge embedding.
    For MST, the edge weight embedding is a key signal.
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        self.lin1    = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2    = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(hidden_dim)

    def forward(self, h_src, h_dst, h_edge):
        x = self.lin1(torch.cat([h_src * h_dst, h_edge], dim=-1))
        x = F.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        return self.lin2(x)


# ==========================================
# 4. FULL MODEL
# ==========================================

class RWTransformerForEdgeClassification(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, config, mup_config=MupConfig()):
        super().__init__()
        self.hidden_dim      = config["hidden_dim"]
        self.walk_length     = config.get("walk_length", 8)
        self.num_walks       = config.get("num_walks", 8)
        self.recurrent_steps = config.get("recurrent_steps", 1)
        self.encoding_dim    = config.get("encoding_dim", 256)

        self.node_encoder = nn.Linear(node_in_dim, self.encoding_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, self.encoding_dim)
        self.input_norm   = nn.LayerNorm(self.encoding_dim)

        if config["use_pe"]:
            if config["pe_type"] == "rwse":
                self.rwse_encoder = nn.Linear(config["pe_dim"], self.encoding_dim)
            elif config["pe_type"] == "lap":
                self.lap_pe_encoder = nn.Linear(config["pe_dim"], self.encoding_dim)

        s = config['mup_init_std']
        torch.nn.init.normal_(self.node_encoder.weight, std=s)
        torch.nn.init.normal_(self.edge_encoder.weight, std=s)
        if config["use_pe"]:
            enc = getattr(self, f"{config['pe_type']}_encoder", None)
            if enc is not None:
                torch.nn.init.normal_(enc.weight, std=s)

        self.transformer = Transformer(
            emb_dim=self.encoding_dim,
            num_layers=config["layers"],
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.hidden_dim * 4,
            num_heads=config.get("num_heads", 8),
            num_walks=self.num_walks,
            seq_len=self.walk_length,
            attn_dropout_p=config["dropout"],
            ffn_dropout_p=config["dropout"],
            config=mup_config
        )
        self.classifier = EdgeDecoderWithFeatures(self.hidden_dim, dropout=config["dropout"])

    def forward(self, x, edge_index, edge_attr, mp_edge_index=None, mp_edge_attr=None,
                use_pe=False, pe_type=None, lap_pe=None, rwse=None):
        if x.dim() == 1:         x         = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr  = edge_attr.unsqueeze(-1)

        x_emb = self.input_norm(self.node_encoder(x))
        if use_pe:
            if pe_type == "lap" and lap_pe is not None:
                x_emb = x_emb + self.input_norm(self.lap_pe_encoder(torch.abs(lap_pe)))
            if pe_type == "rwse" and rwse is not None:
                x_emb = x_emb + self.input_norm(self.rwse_encoder(rwse))

        # Edge attr for the prediction head (uses original directed edges + weights)
        edge_attr_emb = self.input_norm(self.edge_encoder(edge_attr))

        # Walk graph (undirected for better exploration)
        walk_ei = mp_edge_index if mp_edge_index is not None else edge_index
        walk_ea = (mp_edge_attr if mp_edge_attr is not None else edge_attr)
        if walk_ea.dim() == 1: walk_ea = walk_ea.unsqueeze(-1)
        walk_ea_emb = self.input_norm(self.edge_encoder(walk_ea))

        unique_nodes = torch.arange(x.size(0), device=x.device)
        batch_feats, anon_idx, raw_walks = get_random_walk_batch(
            edge_index=walk_ei, x=x_emb, start_nodes=unique_nodes,
            walk_length=self.walk_length, num_walks=self.num_walks,
            num_nodes=x.size(0), recurrent_steps=self.recurrent_steps
        )
        walks_view  = raw_walks[-1].view(x.size(0), self.num_walks, self.walk_length)
        walk_efeats = get_walk_edge_attrs(walk_ei, walk_ea_emb, walks_view, x.size(0))
        batch_feats = batch_feats + walk_efeats.flatten(1, 2)

        node_emb = self.transformer(batch_feats, anon_idx)
        src, dst = edge_index
        return self.classifier(node_emb[src], node_emb[dst], edge_attr_emb)


# ==========================================
# 5. DATA PROCESSING — MST-AWARE
# ==========================================

class FixGraphBenchDataMST(object):
    """
    Cleans up GraphBench data for the MST task.

    Key difference from the generic version: edge weights (edge_attr) are
    PRESERVED because the model must learn which edges are cheapest.
    The original script overwrote edge_attr with ones — that discards the
    only signal that distinguishes MST edges from non-MST edges.
    """
    def __call__(self, data):
        real_num_nodes = int(data.x.size(0) - data.x.sum().item())

        if hasattr(data, 'num_nodes'): del data.num_nodes
        data.num_nodes = real_num_nodes

        # Resize node features if necessary
        data.x = torch.zeros(real_num_nodes, dtype=torch.float)

        if data.edge_index is not None:
            row, col = data.edge_index
            mask     = (row < real_num_nodes) & (col < real_num_nodes)
            data.edge_index = data.edge_index[:, mask]

            # Preserve ground-truth labels
            if data.y is not None and data.y.size(0) == row.size(0):
                data.y = data.y[mask]

            # ── CRITICAL FOR MST ──────────────────────────────────────────
            # Keep original edge weights; normalise to [0, 1] so the encoder
            # sees a consistent range regardless of graph scale.
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                ea = data.edge_attr[mask]
                if ea.dim() == 1: ea = ea.unsqueeze(-1)
                # Per-graph min-max normalisation
                ea_min = ea.min()
                ea_max = ea.max()
                if (ea_max - ea_min).abs() > 1e-6:
                    ea = (ea - ea_min) / (ea_max - ea_min)
                data.edge_attr = ea
            else:
                # Fallback: uniform weights (no weight info available)
                n_edges = data.edge_index.size(1)
                data.edge_attr = torch.ones(n_edges, 1, dtype=torch.float)
            # ─────────────────────────────────────────────────────────────

            if hasattr(data, 'num_edges'): del data.num_edges
        return data


# ==========================================
# 6. HELPERS
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_forward(model, data, config):
    return model(
        data.x.float(),
        data.edge_index,
        data.edge_attr.float(),
        mp_edge_index=getattr(data, 'mp_edge_index', None),
        mp_edge_attr=getattr(data, 'mp_edge_attr', None),
        use_pe=config["use_pe"],
        pe_type=config["pe_type"],
        lap_pe=getattr(data, 'lap_pe', None),
        rwse=getattr(data, 'rwse', None),
    )


def trapezoidal_lr_schedule(step, max_lr, min_lr, warmup, cool, total):
    if step <= warmup:
        return (step / warmup) * (max_lr - min_lr) + min_lr
    elif step <= total - cool:
        return max_lr
    else:
        return ((total - step) / cool) * (max_lr - min_lr) + min_lr


def compute_f1_metrics(y_true, y_pred):
    """Returns (precision, recall, f1) from binary tensors."""
    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()
    precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
    recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ==========================================
# 7. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="HRW Transformer — MST Edge Classification")

    # Dataset
    parser.add_argument("--dataset_name",       type=str,   default="mst_easy",
                        help="mst_easy | mst_hard | bridges_easy | ...")
    parser.add_argument("--data_root",           type=str,   default="./data_graphbench")
    parser.add_argument("--eval_metric_class",   type=str,   default="algoreas_classification")
    # Training
    parser.add_argument("--seed",                type=int,   default=2025)
    parser.add_argument("--epochs",              type=int,   default=10)
    parser.add_argument("--batch_size",          type=int,   default=256)
    parser.add_argument("--test_batch_size",     type=int,   default=32)
    parser.add_argument("--train_subset_ratio",  type=float, default=0.5)
    # Architecture
    parser.add_argument("--hidden_dim",          type=int,   default=128)
    parser.add_argument("--encoding_dim",        type=int,   default=128)
    parser.add_argument("--layers",              type=int,   default=3)
    parser.add_argument("--num_heads",           type=int,   default=16)
    parser.add_argument("--dropout",             type=float, default=0.1)
    # Optimisation
    parser.add_argument("--muon_min_lr",         type=float, default=1e-4)
    parser.add_argument("--muon_max_lr",         type=float, default=1e-3)
    parser.add_argument("--adam_max_lr",         type=float, default=5e-5)
    parser.add_argument("--mlp_lr",              type=float, default=1e-3)
    parser.add_argument("--grad_clip_norm",      type=float, default=0.5)
    # Random walk / PE
    parser.add_argument("--walk_length",         type=int,   default=32)
    parser.add_argument("--num_walks",           type=int,   default=8)
    parser.add_argument("--recurrent_steps",     type=int,   default=1)
    parser.add_argument("--use_pe",              type=bool,  default=False)
    parser.add_argument("--pe_type",             type=str,   default="rwse", choices=["lap", "rwse"])
    parser.add_argument("--pe_dim",              type=int,   default=16)
    # muP
    parser.add_argument("--mup_init_std",        type=float, default=0.01)
    parser.add_argument("--mup_width_multiplier",type=float, default=2.0)
    # Loss
    parser.add_argument("--use_focal_loss",      type=bool,  default=True,
                        help="Use FocalLoss (recommended for MST imbalance) vs BCEWithLogitsLoss")
    parser.add_argument("--focal_alpha",         type=float, default=0.25)
    parser.add_argument("--focal_gamma",         type=float, default=2.0)

    args   = parser.parse_args()
    config = vars(args)

    import pprint
    pprint.pp(config)

    set_seed(config["seed"])
    mup_config = MupConfig(
        init_std=config['mup_init_std'],
        mup_width_multiplier=config['mup_width_multiplier']
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    transforms_list = [FixGraphBenchDataMST(), AddUndirectedContext()]
    if config["use_pe"]:
        if config["pe_type"] == "lap":
            transforms_list.append(T.AddLaplacianEigenvectorPE(
                k=config["pe_dim"], attr_name='lap_pe', is_undirected=True))
        if config["pe_type"] == "rwse":
            transforms_list.append(T.AddRandomWalkPE(
                walk_length=config["pe_dim"], attr_name='rwse'))

    dataset = graphbench.Loader(
        root=config["data_root"],
        dataset_names=config["dataset_name"],
        transform=T.Compose(transforms_list)
    ).load()

    try:
        train_dataset = dataset[0]['train']
        val_dataset   = dataset[0]['valid']
        test_dataset  = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = val_dataset = test_dataset = dataset

    print(f"Split sizes → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    # Inspect a peek batch to determine input dims
    _peek       = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"Input dims → Node: {node_in_dim}  Edge: {edge_in_dim}")

    # Quick sanity-check: print label balance for MST
    y_sample = _peek.y.float()
    print(f"Label balance (peek batch) → pos={y_sample.mean():.3f}  "
          f"neg={1 - y_sample.mean():.3f}  total={y_sample.numel()}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = RWTransformerForEdgeClassification(
        node_in_dim, edge_in_dim, config, mup_config
    ).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # ── Scheduling bookkeeping ───────────────────────────────────────────────
    num_train_total   = len(train_dataset)
    window_size       = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch = math.ceil(window_size / config["batch_size"])
    total_steps       = batches_per_epoch * config["epochs"]
    warmup_steps      = total_steps // 10
    cool_steps        = int(total_steps * 0.1)

    print(f"\nTotal train samples  : {num_train_total:,}")
    print(f"Window / epoch       : {window_size:,}  ({config['train_subset_ratio']*100:.1f}%)")
    print(f"Batches / epoch      : {batches_per_epoch}")
    print(f"Total steps          : {total_steps}  (warmup={warmup_steps}, cool={cool_steps})")

    # ── Optimizer ───────────────────────────────────────────────────────────
    classifier_params_set = set(model.classifier.parameters())
    classifier_params     = list(model.classifier.parameters())
    body_params           = [p for p in model.parameters() if p not in classifier_params_set]
    hidden_weights        = [p for p in body_params if p.ndim >= 2 and p.requires_grad]
    hidden_gains_biases   = [p for p in body_params if p.ndim < 2  and p.requires_grad]
    if config['recurrent_steps'] <= 1:
        hidden_gains_biases = [p for p in hidden_gains_biases
                               if p is not model.transformer.norm_weight]

    param_groups = [
        {'params': hidden_weights,      'use_muon': True,  'lr': config['muon_max_lr'],
         'weight_decay': 0.01},
        {'params': hidden_gains_biases, 'use_muon': False, 'lr': config['adam_max_lr'],
         'betas': (0.9, 0.95), 'weight_decay': 0.1},
        {'params': classifier_params,   'use_muon': False, 'lr': config['mlp_lr'],
         'betas': (0.9, 0.95), 'weight_decay': 0.1},
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    print("\n" + "=" * 52)
    print("PARAMETER GROUP BREAKDOWN")
    print("=" * 52)
    labels = ["Group A  Body Matrices → Muon",
              "Group B  Body Vectors  → Adam",
              "Group C  Classifier    → Adam"]
    calc_total = 0
    for i, g in enumerate(optimizer.param_groups):
        cnt = sum(p.numel() for p in g['params'])
        calc_total += cnt
        print(f"  {labels[i]:<32} : {cnt:12,d}")
    print("-" * 52)
    print(f"  {'Sum':<32} : {calc_total:12,d}")
    print(f"  {'Total (reference)':<32} : {total_params:12,d}")
    status = "SUCCESS" if calc_total == total_params else f"FAILED (diff={abs(total_params-calc_total):,})"
    print(f"  Verification: {status}")
    print("=" * 52 + "\n")

    # ── Loss ────────────────────────────────────────────────────────────────
    if config["use_focal_loss"]:
        criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
        print(f"Loss: FocalLoss(alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss")

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ── W&B ─────────────────────────────────────────────────────────────────
    run_name = build_run_name(config)
    print(f"W&B run name: {run_name}\n")
    wandb.init(
        entity="graph-diffusion-model-link-prediction",
        project=f"graphbench_transformer_{config['dataset_name']}",
        name=run_name,
        config=config,
    )

    # ── Train epoch ──────────────────────────────────────────────────────────
    def train_epoch(epoch, pbar, train_loader):
        model.train()
        total_loss    = 0.0
        global_offset = (epoch - 1) * batches_per_epoch

        for batch_idx, data in enumerate(train_loader):
            data        = data.to(device)
            global_step = global_offset + batch_idx

            for pg in optimizer.param_groups:
                if pg.get('use_muon', False):
                    pg['lr'] = trapezoidal_lr_schedule(
                        global_step, config['muon_max_lr'], config['muon_min_lr'],
                        warmup_steps, cool_steps, total_steps
                    )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out  = run_forward(model, data, config)
                loss = criterion(out.squeeze(), data.y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
            optimizer.step()
            total_loss += loss.item()

            current_muon_lr = next(pg['lr'] for pg in optimizer.param_groups if pg.get('use_muon'))
            pbar.update(1)
            pbar.set_postfix({
                'epoch': epoch,
                'loss':  f'{loss.item():.4f}',
                'lr':    f'{current_muon_lr:.5f}',
            })

        return total_loss / len(train_loader)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(loader, split_name):
        model.eval()
        y_true_list, y_pred_list = [], []

        for data in loader:
            data  = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = run_forward(model, data, config)
            pred = (torch.sigmoid(out) > 0.5).long()
            y_true_list.append(data.y.cpu())
            y_pred_list.append(pred.cpu())

        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        # Graphbench evaluator score (acc + graphbench F1)
        gb_metrics  = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        gb_acc, gb_f1 = parse_metrics(gb_metrics)

        # Our own Precision / Recall / F1 (always available)
        precision, recall, f1 = compute_f1_metrics(y_true, y_pred)

        print(f"  [{split_name:4s}] F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
              f"gb_acc={gb_acc:.4f}  gb_f1={gb_f1:.4f}")

        # W&B — grouped by split, only F1-family metrics
        wandb.log({
            f"F1/{split_name}":        f1,
            f"Precision/{split_name}": precision,
            f"Recall/{split_name}":    recall,
            f"GB_F1/{split_name}":     gb_f1,
            f"GB_Acc/{split_name}":    gb_acc,
        })
        return f1, precision, recall, gb_acc, gb_f1

    # ── Training loop ────────────────────────────────────────────────────────
    print("Starting Training...\n")
    best_val_f1    = -float('inf')
    best_test_f1   = -float('inf')
    best_val_epoch = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_wall = time.time()

    with tqdm(total=total_steps) as pbar:
        for epoch in range(1, config["epochs"] + 1):
            start_idx    = ((epoch - 1) * window_size) % num_train_total
            indices      = [(start_idx + i) % num_train_total for i in range(window_size)]
            train_loader = DataLoader(
                torch.utils.data.Subset(train_dataset, indices),
                batch_size=config["batch_size"],
                shuffle=True, num_workers=4, pin_memory=True,
            )
            print(f"\nEpoch {epoch}  window_start={start_idx}  "
                  f"samples={len(indices):,}  batches={len(train_loader)}")

            avg_loss = train_epoch(epoch, pbar, train_loader)

            model.eval()
            val_f1,  val_p,  val_r,  val_acc,  val_gbf1  = evaluate(val_loader,  "Val")
            test_f1, test_p, test_r, test_acc, test_gbf1 = evaluate(test_loader, "Test")

            if val_f1 > best_val_f1:
                best_val_f1    = val_f1
                best_val_epoch = epoch
                print(f"  >> New Best Val F1 : {best_val_f1:.4f}  (epoch {best_val_epoch})")

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                print(f"  >> New Best Test F1: {best_test_f1:.4f}")

            # Single consolidated epoch log
            wandb.log({
                "epoch":          epoch,
                "Loss/train":     avg_loss,
                "F1/Val":         val_f1,
                "F1/Test":        test_f1,
                "Best/Val_F1":    best_val_f1,
                "Best/Test_F1":   best_test_f1,
                "Best/val_epoch": best_val_epoch,
            })

    print("\nTraining complete.")
    peak_mem   = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"Peak CUDA memory : {peak_mem:.2f} GiB")
    print(f"Total time       : {total_time:.2f} s")

    wandb.log({
        "System/peak_cuda_memory_gb": peak_mem,
        "System/total_runtime_sec":   total_time,
        "System/total_parameters":    total_params,
        "Best/Val_F1":                best_val_f1,
        "Best/Test_F1":               best_test_f1,
        "Best/val_epoch":             best_val_epoch,
    })
    wandb.finish()


if __name__ == "__main__":
    main()