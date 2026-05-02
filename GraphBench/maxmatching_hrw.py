import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import graphbench
from tqdm import tqdm
import wandb
import os
import math
import argparse
import dataclasses
from einops import rearrange
from typing import Optional, Tuple, List
from torch_sparse import SparseTensor
from torch_cluster import random_walk as cluster_random_walk
import random
import numpy as np
from muon import SingleDeviceMuonWithAuxAdam
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_max, scatter_mean, scatter_add
import time
import torch.compiler
import csv
from graphbench.helpers.utils import set_seed

# ==========================================
# 0. UTILS
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs   = inputs.view(-1)
        targets  = targets.view(-1)
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt       = torch.exp(-BCE_loss)
        F_loss   = (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean': return torch.mean(F_loss)
        if self.reduction == 'sum':  return torch.sum(F_loss)
        return F_loss


class AddUndirectedContext(object):
    def __call__(self, data):
        mp_edge_index, mp_edge_attr = to_undirected(
            data.edge_index,
            data.edge_attr,
            num_nodes=data.num_nodes,
            reduce="mean"
        )
        data.mp_edge_index = mp_edge_index
        data.mp_edge_attr  = mp_edge_attr

        row, col     = mp_edge_index
        weights      = mp_edge_attr
        degree       = scatter_add(torch.ones_like(weights), row, dim=0, dim_size=data.num_nodes)
        data.x       = torch.cat([degree], dim=-1)
        return data


# ==========================================
# WANDB HELPERS
# ==========================================

# Fixed project name — all runs land in the same W&B project for easy comparison
WANDB_PROJECT  = "max_matching_bench"
WANDB_ENTITY   = "graph-diffusion-model-link-prediction"


def build_run_name(config: dict) -> str:
    """
    Every run name starts with the model identifier (HRW) followed by
    the key hyperparameters that distinguish runs from one another.

    Format: HRW_<dataset>_h<dim>_L<layers>_H<heads>_w<walks>x<len>
                _rs<recurrent>_<pe-tag>_<nwpe-tag>_s<seed>

    Examples
    --------
    HRW_bipartite_matching_easy_h256_L1_H8_w8x8_rs1_nope_nwpe0_s2025
    HRW_bipartite_matching_easy_h256_L2_H8_w8x8_rs1_rwse16_nwpe7_s42
    """
    pe_tag   = f"_{config['pe_type']}{config['pe_dim']}" if config.get("use_pe")   else "_nope"
    nwpe_tag = f"_nwpe{config['nw_pe_window']}"          if config.get("use_nw_pe") else "_nwpe0"
    return (
        f"HRW"
        f"_{config['dataset_name']}"
        f"_h{config['hidden_dim']}"
        f"_L{config['layers']}"
        f"_H{config['num_heads']}"
        f"_w{config['num_walks']}x{config['walk_length']}"
        f"_rs{config['recurrent_steps']}"
        f"{pe_tag}"
        f"{nwpe_tag}"
        f"_s{config['seed']}"
    )


def compute_prf1(y_true: torch.Tensor, y_pred: torch.Tensor):
    """Returns (precision, recall, f1) from flat binary tensors."""
    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()
    precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
    recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ==========================================
# 1. ARCHITECTURE COMPONENTS
# ==========================================

def compute_neural_walker_pe(walks, edge_index, num_nodes, window_size=8):
    device  = walks.device
    B, W, L = walks.shape

    id_feats = []
    for k in range(1, window_size + 1):
        is_same = (walks[:, :, k:] == walks[:, :, :-k]).float()
        padding = torch.zeros((B, W, k), device=device)
        id_feats.append(torch.cat([padding, is_same], dim=2).unsqueeze(-1))
    identity_encoding = torch.cat(id_feats, dim=-1)

    row, col      = edge_index
    edge_hashes   = row * num_nodes + col
    sorted_hashes, _ = torch.sort(torch.unique(edge_hashes))

    adj_feats = []
    for k in range(1, window_size + 1):
        query_hashes  = walks[:, :, k:] * num_nodes + walks[:, :, :-k]
        idx_in_sorted = torch.searchsorted(sorted_hashes, query_hashes).clamp(
            max=sorted_hashes.size(0) - 1
        )
        is_connected = (sorted_hashes[idx_in_sorted] == query_hashes).float()
        padding      = torch.zeros((B, W, k), device=device)
        adj_feats.append(torch.cat([padding, is_connected], dim=2).unsqueeze(-1))
    adjacency_encoding = torch.cat(adj_feats, dim=-1)

    return torch.cat([identity_encoding, adjacency_encoding], dim=-1)


class RWSE(nn.Module):
    def __init__(self, rwse_steps, embed_dim):
        super().__init__()
        self.embed_dim  = embed_dim
        self.rwse_steps = rwse_steps
        self.encoder    = nn.Sequential(
            nn.Linear(rwse_steps, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, is_training):
        return self.encoder(data.rwse[:, : self.rwse_steps]), None


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

    def forward(self, x: torch.Tensor, *, input_pos=None) -> torch.Tensor:
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
        self.attn_dropout_p  = attn_dropout_p
        self.ffn_dropout_p   = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.drop_path_p     = drop_path_p
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight  = nn.Parameter(torch.ones(hidden_dim))
        self.qkv  = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o    = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=(hidden_dim // num_heads), max_seq_len=seq_len)

        std_ff = config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier)
        std_w  = config.init_std / math.sqrt(config.mup_width_multiplier)
        for w in (self.up.weight, self.gate.weight):
            nn.init.normal_(w, 0.0, std_ff)
        for w in (self.down.weight, self.qkv.weight, self.o.weight):
            nn.init.normal_(w, 0.0, std_w)

    def forward(self, x, offset=None):
        attnx    = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        q, k, v  = self.qkv(attnx).chunk(3, dim=-1)
        q, k, v  = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q        = self.rope(q, input_pos=offset)
        k        = self.rope(k, input_pos=offset)
        q, k, v  = [rearrange(t, 'n t h d -> n h t d', h=self.num_heads) for t in (q, k, v)]
        o_walks  = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / k.shape[-1])
        o_walks  = rearrange(o_walks, 'n h t d -> n t (h d)', h=self.num_heads)
        attn_out = self.o(o_walks)
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)
        ffnx    = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(ffnx)) * self.gate(ffnx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        x = x + self._drop_path(ffn_out)
        return x

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep_prob     = 1.0 - self.drop_path_p
        shape         = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep_prob) * torch.floor(random_tensor)


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads,
                 num_walks, seq_len, attn_dropout_p=0.0, ffn_dropout_p=0.0,
                 resid_dropout_p=0.0, drop_path_p=0.0, config: MupConfig = MupConfig()):
        super().__init__()
        self.mup_cfg    = config
        self.num_heads  = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim   = hidden_dim // num_heads
        self.layers     = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads, num_walks * seq_len,
                n_layer=num_layers, attn_dropout_p=attn_dropout_p,
                ffn_dropout_p=ffn_dropout_p, resid_dropout_p=resid_dropout_p,
                drop_path_p=drop_path_p, config=config
            ) for _ in range(num_layers)
        ])
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x, anon_indices, source_nodes=None):
        batch_size, ctx_len, _ = x.shape
        for depth, idx in enumerate(reversed(anon_indices)):
            for l in self.layers:
                x = l(x, idx)
            if depth < len(anon_indices) - 1:
                x = x[:, -1, :]
                x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
                x = rearrange(x, '(n t) z -> n t z', t=ctx_len)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        x = F.normalize(x[:, -1, :], dim=-1)
        return x


# ==========================================
# 2. RANDOM WALK UTILITIES
# ==========================================

@torch.no_grad()
def anonymize_rws(rws, rev_walks=True):
    if rev_walks:
        rws = torch.flip(rws, dims=[-1])
    s, si = torch.sort(rws, dim=-1)
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
    sorted_hashes, perm = torch.sort(row.long() * num_nodes + col.long())
    src = walks[:, :, :-1].flatten().long()
    dst = walks[:, :, 1:].flatten().long()

    # Bounds-safe searchsorted
    query_hashes = src * num_nodes + dst
    pos   = torch.searchsorted(sorted_hashes, query_hashes).clamp(max=sorted_hashes.size(0) - 1)
    valid = sorted_hashes[pos] == query_hashes
    idx   = torch.where(valid, perm[pos], torch.zeros_like(pos))

    b, w, l  = walks.shape
    edge_dim = edge_attr.size(-1)
    attrs    = edge_attr[idx].view(b, w, l - 1, edge_dim)
    attrs    = attrs * valid.view(b, w, l - 1, 1).float()
    return torch.cat([
        torch.zeros((b, w, 1, edge_dim), device=device, dtype=edge_attr.dtype),
        attrs
    ], dim=2)


@torch.compiler.disable
def get_random_walk_batch(
    edge_index, x, start_nodes, walk_length, num_walks,
    num_nodes, recurrent_steps=1, p=1.0, q=1.0
):
    with torch.no_grad():
        row, col        = edge_index
        current_sources = start_nodes
        rws_list        = []
        for _ in range(recurrent_steps):
            sources_repeated = current_sources.repeat_interleave(num_walks)
            walks = cluster_random_walk(
                row, col, sources_repeated, walk_length - 1, p=p, q=q, num_nodes=num_nodes
            )
            walks = walks.view(current_sources.size(0), num_walks, walk_length)
            rws   = torch.flip(walks.flatten(1, 2), dims=[-1])
            rws_list.append(rws)
            if recurrent_steps > 1:
                current_sources = rws.reshape(-1)
        anon_indices_list = [anonymize_rws(r, rev_walks=True) for r in rws_list]
    return x[rws_list[-1]], anon_indices_list, rws_list


class EdgeDecoderWithFeatures(nn.Module):
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
        return self.lin2(x)


# ==========================================
# 3. WRAPPER MODEL
# ==========================================

class RWTransformerForEdgeClassification(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, config, mup_config=MupConfig()):
        super().__init__()
        self.hidden_dim      = config["hidden_dim"]
        self.walk_length     = config.get("walk_length", 8)
        self.num_walks       = config.get("num_walks", 8)
        self.recurrent_steps = config.get("recurrent_steps", 1)
        self.nw_pe_window    = config.get("nw_pe_window", 5)
        self.use_pe          = config.get("use_pe", False)
        self.use_nw_pe       = config.get("use_nw_pe", False)

        if self.use_pe:
            self.pe_type = config.get("pe_type", None)
            self.pe_dim  = config.get("pe_dim", None)

        if self.use_nw_pe:
            self.nw_pe_encoder = nn.Linear(2 * self.nw_pe_window, self.hidden_dim)
            nn.init.normal_(self.nw_pe_encoder.weight, 0.0, config['mup_init_std'])

        self.node_encoder = nn.Linear(node_in_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, self.hidden_dim)
        self.input_norm   = nn.LayerNorm(self.hidden_dim)

        if self.use_pe:
            if self.pe_type == "rwse":
                self.rwse_encoder = RWSE(rwse_steps=self.pe_dim, embed_dim=self.hidden_dim)
            elif self.pe_type == "lap":
                self.lap_pe_encoder = nn.Linear(self.pe_dim, self.hidden_dim)
                nn.init.normal_(self.lap_pe_encoder.weight, 0.0, config['mup_init_std'])

        nn.init.normal_(self.edge_encoder.weight, 0.0, config['mup_init_std'])
        nn.init.normal_(self.node_encoder.weight, 0.0, config['mup_init_std'])

        self.transformer = Transformer(
            emb_dim=self.hidden_dim,
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
        self.classifier = EdgeDecoderWithFeatures(
            hidden_dim=self.hidden_dim, dropout=config["dropout"]
        )

    def forward(self, x, edge_index, edge_attr, mp_edge_index=None, mp_edge_attr=None,
                use_pe=False, pe_type=None, lap_pe=None, rwse=None):
        if x.dim() == 1:         x         = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr  = edge_attr.unsqueeze(-1)

        x_emb = self.input_norm(self.node_encoder(x))

        if self.use_pe:
            if pe_type == "lap" and lap_pe is not None:
                x_emb = x_emb + self.input_norm(self.lap_pe_encoder(torch.abs(lap_pe)))
            if pe_type == "rwse" and rwse is not None:
                x_emb = x_emb + self.input_norm(
                    self.rwse_encoder.encoder(rwse[:, : self.rwse_encoder.rwse_steps])
                )

        edge_attr_emb = self.input_norm(self.edge_encoder(edge_attr))

        walk_ei = mp_edge_index if mp_edge_index is not None else edge_index
        walk_ea = mp_edge_attr  if mp_edge_attr  is not None else edge_attr
        if walk_ea.dim() == 1: walk_ea = walk_ea.unsqueeze(-1)
        walk_ea_emb = self.input_norm(self.edge_encoder(walk_ea))

        unique_nodes = torch.arange(x.size(0), device=x.device)
        batch_feats, anon_idx, raw_walks = get_random_walk_batch(
            edge_index=walk_ei, x=x_emb,
            start_nodes=unique_nodes,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            num_nodes=x.size(0),
            recurrent_steps=self.recurrent_steps
        )

        walks_view      = raw_walks[-1].view(x.size(0), self.num_walks, self.walk_length)
        walk_edge_feats = get_walk_edge_attrs(walk_ei, walk_ea_emb, walks_view, x.size(0))
        batch_feats     = batch_feats + walk_edge_feats.flatten(1, 2)

        if self.use_nw_pe:
            nw_pe       = compute_neural_walker_pe(walks_view, walk_ei, x.size(0), self.nw_pe_window)
            batch_feats = batch_feats + self.nw_pe_encoder(nw_pe.flatten(1, 2))

        node_embeddings = self.transformer(batch_feats, anon_idx)

        src, dst = edge_index
        return self.classifier(node_embeddings[src], node_embeddings[dst], edge_attr_emb)


# ==========================================
# 4. DATA PROCESSING
# ==========================================

class FixGraphBenchData(object):
    def __call__(self, data):
        if data.x is None: return data
        real_num_nodes = data.x.size(0)
        if hasattr(data, 'num_nodes'): del data.num_nodes
        data.num_nodes = real_num_nodes
        if data.edge_index is not None:
            row, col = data.edge_index
            mask     = (row < real_num_nodes) & (col < real_num_nodes)
            data.edge_index = data.edge_index[:, mask]
            if data.edge_attr is not None and data.edge_attr.size(0) == mask.size(0):
                data.edge_attr = data.edge_attr[mask]
            if data.y is not None and data.y.size(0) == row.size(0):
                data.y = data.y[mask]
            if hasattr(data, 'num_edges'): del data.num_edges
        return data


# ==========================================
# 5. HELPERS
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


def parse_metrics(metrics):
    if isinstance(metrics, dict):
        f1  = metrics.get('f1', metrics.get('F1', 0.0))
        acc = metrics.get('accuracy', metrics.get('acc', f1))
        return acc, f1
    if isinstance(metrics, (list, tuple)):
        acc = metrics[0] if len(metrics) > 0 else 0.0
        f1  = metrics[1] if len(metrics) > 1 else acc
        return acc, f1
    return metrics, metrics


def trapezoidal_lr_schedule(global_step, max_lr, min_lr, warmup, cool, total_steps):
    if global_step <= warmup:
        return (global_step / warmup) * (max_lr - min_lr) + min_lr
    elif global_step <= total_steps - cool:
        return max_lr
    else:
        scale = (total_steps - global_step) / cool
        return scale * (max_lr - min_lr) + min_lr


# ==========================================
# 6. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name",         type=str,   default="bipartite_matching_easy")
    parser.add_argument("--data_root",            type=str,   default="./data_graphbench")
    parser.add_argument("--seed",                 type=int,   default=2025)
    parser.add_argument("--epochs",               type=int,   default=10)
    parser.add_argument("--batch_size",           type=int,   default=64)
    parser.add_argument("--test_batch_size",      type=int,   default=32)
    parser.add_argument("--hidden_dim",           type=int,   default=256)
    parser.add_argument("--layers",               type=int,   default=1)
    parser.add_argument("--num_heads",            type=int,   default=8)
    parser.add_argument("--dropout",              type=float, default=0.1)
    parser.add_argument("--muon_min_lr",          type=float, default=1e-4)
    parser.add_argument("--muon_max_lr",          type=float, default=1e-3)
    parser.add_argument("--adam_max_lr",          type=float, default=1e-4)
    parser.add_argument("--mlp_lr",               type=float, default=1e-4)
    parser.add_argument("--grad_clip_norm",       type=float, default=0.5)
    parser.add_argument("--train_subset_ratio",   type=float, default=0.1)
    parser.add_argument("--walk_length",          type=int,   default=16)
    parser.add_argument("--num_walks",            type=int,   default=8)
    parser.add_argument("--recurrent_steps",      type=int,   default=1)
    parser.add_argument("--use_pe",               type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--pe_type",              type=str,   default="rwse", choices=["lap", "rwse"])
    parser.add_argument("--pe_dim",               type=int,   default=16)
    parser.add_argument("--nw_pe_window",         type=int,   default=7)
    parser.add_argument("--use_nw_pe",            type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--mup_init_std",         type=float, default=0.01)
    parser.add_argument("--mup_width_multiplier", type=float, default=2.0)
    parser.add_argument("--eval_metric_class",    type=str,   default='algoreas_classification')

    args   = parser.parse_args()
    config = vars(args)
    import pprint
    pprint.pp(config)

    set_seed(config["seed"])
    mup_config = MupConfig(init_std=config['mup_init_std'],
                           mup_width_multiplier=config['mup_width_multiplier'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    class PEOnUndirected:
        def __init__(self, transform, use_edge_weights=False):
            self._transform       = transform
            self.use_edge_weights = use_edge_weights

        def __call__(self, data):
            orig_edge_index = data.edge_index
            data.edge_index = data.mp_edge_index
            data = self._transform(data)
            data.edge_index = orig_edge_index
            return data

    transforms_list = [FixGraphBenchData(), AddUndirectedContext()]
    if config["use_pe"]:
        if config["pe_type"] == "lap":
            transforms_list.append(PEOnUndirected(
                T.AddLaplacianEigenvectorPE(k=config["pe_dim"], attr_name='lap_pe', is_undirected=True)
            ))
        else:
            transforms_list.append(PEOnUndirected(
                T.AddRandomWalkPE(walk_length=config["pe_dim"], attr_name='rwse'),
                use_edge_weights=True
            ))

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

    print(f"Sizes → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    _peek       = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"Input dims → node: {node_in_dim}  edge: {edge_in_dim}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = RWTransformerForEdgeClassification(
        node_in_dim, edge_in_dim, config, mup_config
    ).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # ── Schedule bookkeeping ────────────────────────────────────────────────
    num_train_total    = len(train_dataset)
    eval_sample_window = max(1, int(num_train_total * 0.1))
    eval_every_steps   = max(1, round(eval_sample_window / config["batch_size"]))
    window_size        = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch  = math.ceil(window_size / config["batch_size"])
    total_steps        = batches_per_epoch * config["epochs"]
    warmup_steps       = total_steps // 10
    cool_steps         = int(total_steps * 0.1)

    print(f"\nTotal training samples  : {num_train_total:,}")
    print(f"Window size per epoch   : {window_size:,}")
    print(f"Batches per epoch       : {batches_per_epoch}")
    print(f"Total optimiser steps   : {total_steps}")
    print(f"Warmup / cooldown steps : {warmup_steps} / {cool_steps}")
    print(f"Eval every N steps      : {eval_every_steps}\n")

    # ── Optimizer ───────────────────────────────────────────────────────────
    classifier_params_set = set(model.classifier.parameters())
    classifier_params     = list(model.classifier.parameters())
    body_params           = [p for p in model.parameters() if p not in classifier_params_set]
    hidden_weights        = [p for p in body_params if p.ndim >= 2 and p.requires_grad]
    hidden_gains_biases   = [p for p in body_params if p.ndim < 2  and p.requires_grad]
    if config['recurrent_steps'] <= 1:
        norm_w = getattr(getattr(model, 'transformer', model), 'norm_weight', None)
        if norm_w is not None:
            hidden_gains_biases = [p for p in hidden_gains_biases if p is not norm_w]

    param_groups = [
        {'params': hidden_weights,      'use_muon': True,  'lr': config['muon_max_lr'], 'weight_decay': 0.01},
        {'params': hidden_gains_biases, 'use_muon': False, 'lr': config['adam_max_lr'], 'betas': (0.9, 0.95), 'weight_decay': 0.0},
        {'params': classifier_params,   'use_muon': False, 'lr': config['mlp_lr'],      'betas': (0.9, 0.95), 'weight_decay': 0.0},
    ]
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    print("=" * 52)
    print("PARAMETER GROUP BREAKDOWN")
    print("=" * 52)
    labels     = ["Group A  Body Matrices → Muon",
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
    ok = calc_total == total_params
    print(f"  Verification: {'SUCCESS' if ok else f'FAILED (diff={abs(total_params-calc_total):,})'}")
    print("=" * 52 + "\n")

    criterion = FocalLoss(gamma=2)

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ── W&B INIT ────────────────────────────────────────────────────────────
    run_name = build_run_name(config)
    print(f"W&B project : {WANDB_PROJECT}")
    print(f"W&B run     : {run_name}\n")
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,          # ← fixed project name
        name=run_name,                  # ← model-prefixed run name
        config=config,
    )

    checkpoint_dir  = os.path.join(config["dataset_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    # ── Evaluate helper ──────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(loader, split_name: str):
        """
        Returns (acc, f1, precision, recall) and logs to W&B under
        grouped F1/ Precision/ Recall/ panels.
        Caller must set model.eval() beforehand.
        """
        assert not model.training, "Call model.eval() before evaluating!"
        y_true_list, y_pred_list = [], []

        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = run_forward(model, data, config)
            pred = (torch.sigmoid(out) > 0.5).long()
            y_true_list.append(data.y.cpu())
            y_pred_list.append(pred.cpu())

        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        # GraphBench evaluator (acc + gb_f1)
        gb_metrics    = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        gb_acc, gb_f1 = parse_metrics(gb_metrics)

        # Our own P/R/F1 (always available)
        precision, recall, f1 = compute_prf1(y_true, y_pred)

        print(f"  [{split_name:4s}] F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
              f"gb_acc={gb_acc:.4f}  gb_f1={gb_f1:.4f}")

        # W&B — grouped by split so panels auto-cluster
        wandb.log({
            f"F1/{split_name}":        f1,
            f"Precision/{split_name}": precision,
            f"Recall/{split_name}":    recall,
            f"GB_F1/{split_name}":     gb_f1,
            f"GB_Acc/{split_name}":    gb_acc,
        })
        return gb_acc, gb_f1, precision, recall, f1

    # ── Pre-training baseline (step 0) ───────────────────────────────────────
    print("Running pre-training evaluation (step 0)...")
    model.eval()
    _, _, _, _, val_f1_0  = evaluate(val_loader,  "Val")
    _, _, _, _, test_f1_0 = evaluate(test_loader, "Test")
    wandb.log({"global_step": 0, "F1/Val": val_f1_0, "F1/Test": test_f1_0})

    # ── Training loop ────────────────────────────────────────────────────────
    print("\nStarting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_wall       = time.time()
    global_step      = 0
    running_loss_sum = 0.0
    running_loss_cnt = 0

    best_val_f1    = -float('inf')
    best_test_f1   = -float('inf')
    best_val_step  = 0
    best_test_step = 0

    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(1, config["epochs"] + 1):
            start_idx    = ((epoch - 1) * window_size) % num_train_total
            indices      = [(start_idx + i) % num_train_total for i in range(window_size)]
            train_loader = DataLoader(
                torch.utils.data.Subset(train_dataset, indices),
                batch_size=config["batch_size"],
                shuffle=True, num_workers=4, pin_memory=True,
            )
            print(f'\nEpoch {epoch} — window_start={start_idx}  '
                  f'samples={len(indices):,}  batches={len(train_loader)}')

            model.train()

            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)

                for pg in optimizer.param_groups:
                    if pg.get('use_muon', False):
                        pg['lr'] = trapezoidal_lr_schedule(
                            global_step, config['muon_max_lr'], config['muon_min_lr'],
                            warmup_steps, cool_steps, total_steps,
                        )

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out  = run_forward(model, data, config)
                    loss = criterion(out.squeeze(), data.y.float())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
                optimizer.step()

                global_step      += 1
                running_loss_sum += loss.item()
                running_loss_cnt += 1

                current_muon_lr = next(
                    (pg['lr'] for pg in optimizer.param_groups if pg.get('use_muon')),
                    config['muon_max_lr']
                )

                # Per-step log: only loss + lr — no eval metrics here
                wandb.log({
                    "Loss/train_step": loss.item(),
                    "LR/muon":         current_muon_lr,
                    "global_step":     global_step,
                })

                pbar.update(1)
                pbar.set_postfix({
                    'ep':   epoch,
                    'loss': f'{loss.item():.4f}',
                    'lr':   f'{current_muon_lr:.5f}',
                })

                # ── STEP-BASED EVAL ─────────────────────────────────────
                if global_step % eval_every_steps == 0:
                    avg_train_loss   = running_loss_sum / running_loss_cnt
                    running_loss_sum = 0.0
                    running_loss_cnt = 0

                    model.eval()
                    _, _, _, _, val_f1  = evaluate(val_loader,  "Val")
                    _, _, _, _, test_f1 = evaluate(test_loader, "Test")

                    if val_f1 > best_val_f1:
                        best_val_f1   = val_f1
                        best_val_step = global_step
                        print(f"  >> New Best Val F1 : {best_val_f1:.4f}  (step {best_val_step})")

                    if test_f1 > best_test_f1:
                        best_test_f1   = test_f1
                        best_test_step = global_step
                        torch.save({
                            'global_step':     global_step,
                            'epoch':           epoch,
                            'model_state':     model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'best_test_f1':    best_test_f1,
                            'config':          config,
                        }, best_model_path)
                        print(f"  >> New Best Test F1: {best_test_f1:.4f}  "
                              f"(step {best_test_step}) — saved to {best_model_path}")

                    # Epoch-level consolidated log
                    wandb.log({
                        "global_step":      global_step,
                        "epoch":            epoch,
                        "Loss/train_avg":   avg_train_loss,
                        "F1/Val":           val_f1,
                        "F1/Test":          test_f1,
                        "Best/Val_F1":      best_val_f1,
                        "Best/Test_F1":     best_test_f1,
                        "Best/val_step":    best_val_step,
                        "Best/test_step":   best_test_step,
                    })

                    model.train()

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\nRunning final evaluation...")
    model.eval()
    _, _, _, _, final_val_f1  = evaluate(val_loader,  "Val")
    _, _, _, _, final_test_f1 = evaluate(test_loader, "Test")

    peak_mem   = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"\nTraining complete.")
    print(f"Peak CUDA memory : {peak_mem:.2f} GiB")
    print(f"Total time       : {total_time:.2f} s")
    print(f"Best Val  F1     : {best_val_f1:.4f}  (step {best_val_step})")
    print(f"Best Test F1     : {best_test_f1:.4f}  (step {best_test_step})")

    wandb.log({
        "System/peak_cuda_memory_gb": peak_mem,
        "System/total_runtime_sec":   total_time,
        "System/total_parameters":    total_params,
        "Best/Val_F1":                best_val_f1,
        "Best/Test_F1":               best_test_f1,
        "Best/val_step":              best_val_step,
        "Best/test_step":             best_test_step,
        "Final/Val_F1":               final_val_f1,
        "Final/Test_F1":              final_test_f1,
    })

    # ── CSV log ──────────────────────────────────────────────────────────────
    os.makedirs(config["dataset_name"], exist_ok=True)
    csv_path = f"{config['dataset_name']}/results_{WANDB_PROJECT}.csv"
    row = {
        "run_name":           run_name,
        "dataset":            config["dataset_name"],
        "seed":               config["seed"],
        "hidden_dim":         config["hidden_dim"],
        "layers":             config["layers"],
        "num_heads":          config["num_heads"],
        "num_walks":          config["num_walks"],
        "walk_length":        config["walk_length"],
        "recurrent_steps":    config["recurrent_steps"],
        "use_pe":             config["use_pe"],
        "pe_type":            config["pe_type"],
        "pe_dim":             config["pe_dim"],
        "use_nw_pe":          config["use_nw_pe"],
        "nw_pe_window":       config["nw_pe_window"],
        "batch_size":         config["batch_size"],
        "muon_max_lr":        config["muon_max_lr"],
        "adam_max_lr":        config["adam_max_lr"],
        "mlp_lr":             config["mlp_lr"],
        "epochs":             config["epochs"],
        "dropout":            config["dropout"],
        "train_subset_ratio": config["train_subset_ratio"],
        "eval_every_steps":   eval_every_steps,
        "best_val_f1":        round(best_val_f1,   4),
        "best_test_f1":       round(best_test_f1,  4),
        "best_val_step":      best_val_step,
        "final_val_f1":       round(final_val_f1,  4),
        "final_test_f1":      round(final_test_f1, 4),
        "total_params":       total_params,
        "runtime_sec":        round(total_time, 1),
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"Results appended to {csv_path}")

    wandb.finish()


if __name__ == "__main__":
    main()