from types import SimpleNamespace
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
import csv
import matplotlib.pyplot as plt
from torch_geometric.utils import to_undirected, is_undirected, to_networkx, remove_self_loops
from torch_scatter import scatter_max, scatter_mean, scatter_add
import time
from graphbench.helpers.utils import set_seed
import networkx as nx
import torch.compiler
from scipy.sparse import csr_matrix

_original_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only


class AddUndirectedContext(object):
    def __call__(self, data):
        if is_undirected(data.edge_index):
            mp_edge_index = data.edge_index
        else:
            mp_edge_index = to_undirected(
                data.edge_index, num_nodes=data.num_nodes, reduce="mean"
            )
        data.mp_edge_index = mp_edge_index

        n = data.num_nodes
        data.x            = torch.ones(n, 1, dtype=torch.float)
        data.mp_edge_attr  = torch.ones(mp_edge_index.size(1), 1, dtype=torch.float)
        data.edge_attr     = torch.ones(data.edge_index.size(1), 1, dtype=torch.float)
        return data


# ==========================================
# POSITIONAL ENCODINGS
# ==========================================
def compute_neural_walker_pe(walks, edge_index, num_nodes, window_size=8):
    """
    Computes Identity and Adjacency encodings for random walks.
    """
    device = walks.device
    B, W, L = walks.shape
    
    # ---------------------------------------------------------
    # 1. Identity Encoding: w_i == w_{i-k}
    # ---------------------------------------------------------
    idx = torch.arange(L, device=device)
    
    id_feats = []

    for k in range(1, window_size + 1):
        current_step = walks[:, :, k:]      # Steps [k, k+1, ..., L-1]
        past_step    = walks[:, :, :-k]     # Steps [0, 1, ..., L-1-k]
        
        is_same = (current_step == past_step).float()
        
        padding = torch.zeros((B, W, k), device=device)
        is_same_padded = torch.cat([padding, is_same], dim=2) # (B, W, L)
        
        id_feats.append(is_same_padded.unsqueeze(-1))

    identity_encoding = torch.cat(id_feats, dim=-1)
    # ---------------------------------------------------------
    # 2. Adjacency Encoding: (w_i, w_{i-k}) in E
    # ---------------------------------------------------------
    row, col = edge_index
    edge_hashes = row * num_nodes + col
    sorted_hashes, _ = torch.sort(torch.unique(edge_hashes)) 
    
    adj_feats = []
    
    for k in range(2, window_size + 1):
        current_step = walks[:, :, k:]
        past_step    = walks[:, :, :-k]
        
        query_hashes = current_step * num_nodes + past_step
        
        idx_in_sorted = torch.searchsorted(sorted_hashes, query_hashes)
        idx_in_sorted = idx_in_sorted.clamp(max=sorted_hashes.size(0) - 1)
        found_hashes = sorted_hashes[idx_in_sorted]
        
        is_connected = (found_hashes == query_hashes).float()
        
        padding = torch.zeros((B, W, k), device=device)
        is_connected_padded = torch.cat([padding, is_connected], dim=2)
        
        adj_feats.append(is_connected_padded.unsqueeze(-1))  
          
    adjacency_encoding = torch.cat(adj_feats, dim=-1)
    return torch.cat([identity_encoding, adjacency_encoding], dim=-1)


def compute_laplacian_eigen(edge_index, num_nodes, max_freq,
                            normalized=False, normalize=False, large_graph=False):
    A = torch.zeros((num_nodes, num_nodes))
    A[edge_index[0], edge_index[1]] = 1
    if normalized:
        D12 = torch.diag(A.sum(1).clip(1) ** -0.5)
        L   = torch.eye(A.size(0)) - D12 @ A @ D12
    else:
        L = torch.diag(A.sum(1)) - A
    eigvals, eigvecs = torch.linalg.eigh(L)
    idx = (torch.argsort(eigvals)[:max_freq] if not large_graph else
           torch.cat([torch.argsort(eigvals)[:max_freq//2],
                      torch.argsort(eigvals, descending=True)[:max_freq//2]]))
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]
    eigvals = torch.real(eigvals).clamp_min(0)
    if normalize:
        eigvecs = eigvecs / eigvecs.norm(p=2, dim=0, keepdim=True).clamp_min(1e-12)
    if num_nodes < max_freq:
        eigvals = F.pad(eigvals, (0, max_freq - num_nodes), value=float("nan"))
        eigvecs = F.pad(eigvecs, (0, max_freq - num_nodes), value=float("nan"))
    return eigvals.unsqueeze(0).repeat(num_nodes, 1), eigvecs


class LPE(nn.Module):
    def __init__(self, embed_dim, lpe_num_eigvals, position_aware, lpe_bias, lpe_inner_dim):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(2, embed_dim, bias=False),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if position_aware:
            self.eps = nn.Parameter(1e-12 * torch.arange(lpe_inner_dim)[None])
        self.position_aware  = position_aware
        self.lpe_num_eigvals = lpe_num_eigvals

    def forward(self, eigvecs, eigvals, is_training):
        eigvecs = eigvecs[:, : self.lpe_num_eigvals]
        eigvals = eigvals[:, : self.lpe_num_eigvals]
        if is_training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5]  = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)
        if self.position_aware:
            eigvals = eigvals + self.eps[:, : self.lpe_num_eigvals]
        x = torch.stack((eigvecs, eigvals), 2)
        empty_mask = torch.isnan(x)
        x[empty_mask] = 0
        eigen_embed = self.phi(x)
        lpe_pe = self.rho(eigen_embed.sum(1))
        return lpe_pe, None


class CustomLaplacianPE:
    def __init__(self, max_freq, normalized=False, normalize=False, large_graph=False):
        self.max_freq    = max_freq
        self.normalized  = normalized
        self.normalize   = normalize
        self.large_graph = large_graph

    def __call__(self, data):
        eigvals, eigvecs = compute_laplacian_eigen(
            data.edge_index, data.num_nodes, self.max_freq,
            self.normalized, self.normalize, self.large_graph
        )
        data.eigvecs = eigvecs
        data.eigvals = eigvals
        return data


class RWSE(nn.Module):
    def __init__(self, rwse_steps, embed_dim):
        super().__init__()
        self.embed_dim  = embed_dim
        self.rwse_steps = rwse_steps
        self.encoder = nn.Sequential(
            nn.Linear(rwse_steps, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, data, is_training):
        return self.encoder(data.rwse[:, : self.rwse_steps]), None


# ==========================================
# ARCHITECTURE
# ==========================================

@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000):
        super().__init__()
        self.dim, self.base, self.max_seq_len = dim, base, max_seq_len
        self.rope_init()

    def reset_parameters(self): self.rope_init()

    def rope_init(self):
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[: self.dim // 2].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len=4096):
        idx   = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        cache = torch.stack([torch.cos(torch.einsum("i,j->ij", idx, self.theta)),
                             torch.sin(torch.einsum("i,j->ij", idx, self.theta))], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x, *, input_pos=None):
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
                 attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0,
                 resid_dropout_p: float = 0.0, drop_path_p: float = 0.0,
                 config=MupConfig()):
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

        torch.nn.init.normal_(self.up.weight.data,   mean=0.0, std=config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier))
        torch.nn.init.normal_(self.gate.weight.data, mean=0.0, std=config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier))
        torch.nn.init.normal_(self.down.weight.data, mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
        torch.nn.init.normal_(self.qkv.weight.data,  mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))
        torch.nn.init.normal_(self.o.weight.data,    mean=0.0, std=config.init_std / math.sqrt(config.mup_width_multiplier))

    def forward(self, x, offset=None):
        attnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        qkv   = self.qkv(attnx)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d', h=self.num_heads) for t in (q, k, v)]
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
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads,
                 num_walks, seq_len, attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0,
                 resid_dropout_p: float = 0.0, drop_path_p: float = 0.0,
                 config: MupConfig = MupConfig()):
        super().__init__()
        self.mup_cfg    = config
        self.num_heads  = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim   = hidden_dim // num_heads
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
        batch_size, ctx_len, _ = x.shape
        for depth, idx in enumerate(reversed(anon_indices)):
            for l in self.layers:
                x = l(x, idx)
            if depth < len(anon_indices) - 1:
                x = x[:, -1, :]
                x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
                x = rearrange(x, '(n t) z -> n t z', t=ctx_len)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return x[:, -1, :]


# ==========================================
# RANDOM WALK UTILITIES
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
    src, dst            = walks[:, :, :-1].flatten().long(), walks[:, :, 1:].flatten().long()
    idx                 = perm[torch.searchsorted(sorted_hashes, src * num_nodes + dst)]
    b, w, l             = walks.shape
    edge_dim            = edge_attr.size(-1)
    out = torch.cat([
        torch.zeros((b, w, 1, edge_dim), device=device, dtype=edge_attr.dtype),
        edge_attr[idx].view(b, w, l - 1, edge_dim)
    ], dim=2)
    return out


@torch.compiler.disable
def node2vec_random_walk(edge_index, start_nodes, walk_length, num_walks, num_nodes, p=1.0, q=1.0):
    if p == 1.0 and q == 1.0:
        starts = start_nodes.repeat_interleave(num_walks)
        walks  = cluster_random_walk(edge_index[0], edge_index[1], starts,
                                     walk_length - 1, p=1.0, q=1.0, num_nodes=num_nodes)
        return walks.view(start_nodes.size(0), num_walks, walk_length)

    device   = start_nodes.device
    row, col = edge_index
    inv_p, inv_q = 1.0 / p, 1.0 / q

    sort_idx = torch.argsort(row)
    row_s, col_s = row[sort_idx], col[sort_idx]
    deg = torch.zeros(num_nodes, dtype=torch.long, device=device)
    deg.scatter_add_(0, row_s, torch.ones_like(row_s))
    ptr = torch.zeros(num_nodes + 1, dtype=torch.long, device=device)
    ptr[1:] = deg.cumsum(0)

    edge_hashes, _ = torch.sort(row.long() * num_nodes + col.long())

    N     = start_nodes.size(0)
    total = N * num_walks
    max_deg = int(deg.max().item())

    cur  = start_nodes.repeat_interleave(num_walks)
    all_steps = [cur.clone()]

    deg_cur = deg[cur].clamp(min=1)
    offsets = (torch.rand(total, device=device) * deg_cur.float()).long()
    nxt     = col_s[(ptr[cur] + offsets).clamp(max=col_s.size(0) - 1)]
    all_steps.append(nxt)
    prev, cur = cur, nxt

    off2d = torch.arange(max_deg, device=device).unsqueeze(0)

    for _ in range(walk_length - 2):
        starts_cur = ptr[cur]
        ends_cur   = ptr[cur + 1]
        deg_cur    = ends_cur - starts_cur

        idx       = (starts_cur.unsqueeze(1) + off2d).clamp(max=col_s.size(0) - 1)
        valid     = off2d < deg_cur.unsqueeze(1)
        neighbors = col_s[idx]
        neighbors[~valid] = -1

        is_return = (neighbors == prev.unsqueeze(1))
        query    = prev.unsqueeze(1).long() * num_nodes + neighbors.long().clamp(min=0)
        si       = torch.searchsorted(edge_hashes, query.flatten()).clamp(max=edge_hashes.size(0) - 1)
        is_adj   = (edge_hashes[si] == query.flatten()).view_as(neighbors)
        is_adj[~valid] = False

        weights             = torch.full_like(neighbors, inv_q, dtype=torch.float)
        weights[is_adj]     = 1.0
        weights[is_return]  = inv_p
        weights[~valid]     = 0.0
        weights             = weights / weights.sum(1, keepdim=True).clamp(min=1e-12)

        chosen = torch.multinomial(weights, num_samples=1).squeeze(1)
        nxt    = neighbors[torch.arange(total, device=device), chosen]
        nxt[deg_cur == 0] = cur[deg_cur == 0]

        all_steps.append(nxt)
        prev, cur = cur, nxt

    walks = torch.stack(all_steps, dim=1)
    return walks.view(N, num_walks, walk_length)


@torch.compiler.disable
def get_random_walk_batch(edge_index, x, start_nodes, walk_length, num_walks,
                          num_nodes, recurrent_steps=1, p=1.0, q=1.0):
    with torch.no_grad():
        current_sources = start_nodes
        rws_list        = []
        for _ in range(recurrent_steps):
            walks = node2vec_random_walk(
                edge_index, current_sources,
                walk_length, num_walks, num_nodes, p=p, q=q
            )
            rws = torch.flip(walks.flatten(1, 2), dims=[-1])
            rws_list.append(rws)
            if recurrent_steps > 1:
                current_sources = rws.reshape(-1)
        anon_indices_list = [anonymize_rws(r, rev_walks=True) for r in rws_list]
    return x[rws_list[-1]], anon_indices_list, rws_list


# ==========================================
# MODEL
# ==========================================

class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes=2, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        return self.lin2(self.norm(F.gelu(self.lin1(h))))


class RWTransformerForNodeClassification(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, config, mup_config=MupConfig()):
        super().__init__()
        self.hidden_dim      = config["hidden_dim"]
        self.walk_length     = config.get("walk_length", 8)
        self.num_walks       = config.get("num_walks", 8)
        self.recurrent_steps = config.get("recurrent_steps", 1)
        self.node2vec_p      = config.get("node2vec_p", 1.0)
        self.node2vec_q      = config.get("node2vec_q", 1.0)
        self.nw_pe_window    = config.get("nw_pe_window", 5)
        self.ffn_multiplier  = config.get("ffn_multiplier", 4.0)
        self.drop_path_p     = config.get("drop_path_p", 0.0)
        self.use_nw_pe       = config.get("use_nw_pe", False)

        self.node_encoder = nn.Linear(node_in_dim, self.hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, self.hidden_dim)
        self.input_norm   = nn.LayerNorm(self.hidden_dim)

        if self.use_nw_pe:
            self.nw_pe_encoder = nn.Linear(2 * self.nw_pe_window, self.hidden_dim)
            nn.init.normal_(self.nw_pe_encoder.weight, mean=0.0, std=config['mup_init_std'])

        if config["use_pe"]:
            if config["pe_type"] == "rwse":
                self.rwse_encoder = RWSE(rwse_steps=config["pe_dim"], embed_dim=self.hidden_dim)
            elif config["pe_type"] == "lap":
                self.lpe = LPE(
                    embed_dim=self.hidden_dim,
                    lpe_num_eigvals=config["pe_dim"],
                    position_aware=True,
                    lpe_bias=True,
                    lpe_inner_dim=config["pe_dim"],
                )

        nn.init.normal_(self.edge_encoder.weight, mean=0.0, std=config['mup_init_std'])
        nn.init.normal_(self.node_encoder.weight, mean=0.0, std=config['mup_init_std'])

        self.transformer = Transformer(
            emb_dim=self.hidden_dim,
            num_layers=config["layers"],
            hidden_dim=self.hidden_dim,
            intermediate_dim=int(self.hidden_dim * self.ffn_multiplier),
            num_heads=config.get("num_heads", 8),
            num_walks=self.num_walks,
            seq_len=self.walk_length,
            attn_dropout_p=config["dropout"],
            ffn_dropout_p=config["dropout"],
            config=mup_config
        )
        self.classifier = NodeDecoder(hidden_dim=self.hidden_dim, dropout=config["dropout"])

    def forward(self, x, edge_index, edge_attr, mp_edge_index=None, mp_edge_attr=None,
                use_pe=False, pe_type=None, eigvecs=None, eigvals=None, rwse=None):
        if x.dim() == 1:         x        = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(-1)

        x_emb = self.input_norm(self.node_encoder(x.float()))

        if use_pe:
            if pe_type == "lap" and eigvecs is not None and eigvals is not None:
                lpe_out, _ = self.lpe(eigvecs, eigvals, self.training)
                x_emb = x_emb + lpe_out
            if pe_type == "rwse" and rwse is not None:
                rwse_out, _ = self.rwse_encoder(
                    SimpleNamespace(rwse=rwse.float()), self.training
                )
                x_emb = x_emb + rwse_out

        walk_ei     = mp_edge_index if mp_edge_index is not None else edge_index
        walk_ea     = mp_edge_attr  if mp_edge_attr  is not None else edge_attr
        if walk_ea.dim() == 1: walk_ea = walk_ea.unsqueeze(-1)
        walk_ea_emb = self.input_norm(self.edge_encoder(walk_ea.float()))

        batch_feats, anon_idx, raw_walks = get_random_walk_batch(
            edge_index=walk_ei, x=x_emb,
            start_nodes=torch.arange(x.size(0), device=x.device),
            walk_length=self.walk_length, num_walks=self.num_walks,
            num_nodes=x.size(0), recurrent_steps=self.recurrent_steps,
            p=self.node2vec_p, q=self.node2vec_q
        )
        walks_view = raw_walks[-1].view(x.size(0), self.num_walks, self.walk_length)
        walk_edge_attrs = get_walk_edge_attrs(walk_ei, walk_ea_emb, walks_view, x.size(0))
        batch_feats = batch_feats + walk_edge_attrs.flatten(1, 2)

        if self.use_nw_pe:
            nw_pe = compute_neural_walker_pe(walks_view, walk_ei, x.size(0), self.nw_pe_window)
            batch_feats = batch_feats + self.nw_pe_encoder(nw_pe.flatten(1, 2))

        return self.classifier(self.transformer(batch_feats, anon_idx))


# ==========================================
# HELPERS
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_forward(model, data, config):
    return model(
        data.x.float(), data.edge_index, data.edge_attr.float(),
        mp_edge_index=getattr(data, 'mp_edge_index', None),
        mp_edge_attr=getattr(data, 'mp_edge_attr', None),
        use_pe=config["use_pe"], pe_type=config["pe_type"],
        eigvecs=getattr(data, 'eigvecs', None),
        eigvals=getattr(data, 'eigvals', None),
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

# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()

    # Dataset & General
    parser.add_argument("--dataset_name",       type=str,   default="maxclique_easy")
    parser.add_argument("--data_root",          type=str,   default="./data_graphbench")
    parser.add_argument("--seed",               type=int,   default=2025)
    parser.add_argument("--epochs",             type=int,   default=10)
    parser.add_argument("--batch_size",         type=int,   default=256)
    parser.add_argument("--test_batch_size",    type=int,   default=32)

    # Architecture
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--layers",             type=int,   default=1)
    parser.add_argument("--num_heads",          type=int,   default=8)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--ffn_multiplier",     type=float, default=4.0)

    # Optimization
    parser.add_argument("--adam_max_lr",        type=float, default=1e-4)
    parser.add_argument("--weight_decay",       type=float, default=0.0)
    parser.add_argument("--grad_clip_norm",     type=float, default=0.5)
    parser.add_argument("--drop_path_p",        type=float, default=0.0)
    parser.add_argument("--train_subset_ratio", type=float, default=0.1)

    # Random Walk / PE
    parser.add_argument("--walk_length",        type=int,   default=8)
    parser.add_argument("--num_walks",          type=int,   default=10)
    parser.add_argument("--node2vec_p",         type=float, default=1.0)
    parser.add_argument("--node2vec_q",         type=float, default=1.0)
    parser.add_argument("--recurrent_steps",    type=int,   default=1)
    parser.add_argument("--use_pe",             type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--pe_type",            type=str,   default="rwse", choices=["lap", "rwse"])
    parser.add_argument("--pe_dim",             type=int,   default=16)
    parser.add_argument("--nw_pe_window",       type=int,   default=7)
    parser.add_argument("--use_nw_pe",          type=lambda x: x.lower() == 'true', default=False)

    # muP & Evaluation
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

    # ==========================================
    # TRANSFORM PIPELINE
    # ==========================================
    transforms_list = [AddUndirectedContext()]
    if config["use_pe"]:
        if config["pe_type"] == "lap":
            transforms_list.append(CustomLaplacianPE(max_freq=config["pe_dim"],
                                                      normalized=True, normalize=True))
        elif config["pe_type"] == "rwse":
            transforms_list.append(T.AddRandomWalkPE(walk_length=config["pe_dim"],
                                                      attr_name='rwse'))

    loader  = graphbench.Loader(
        root=config["data_root"],
        dataset_names=config["dataset_name"],
        transform=T.Compose(transforms_list)
    )
    dataset = loader.load()

    try:
        train_dataset = dataset[0]['train']
        val_dataset   = dataset[0]['valid']
        test_dataset  = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = val_dataset = test_dataset = dataset

    print(f"Sizes -> Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    _peek       = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"\nInput dims  →  node: {node_in_dim}  edge: {edge_in_dim}")

    model        = RWTransformerForNodeClassification(node_in_dim, edge_in_dim,
                                                      config, mup_config).to(device)
    # if hasattr(torch, "compile"):
    #     model = torch.compile(model)

    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    num_train_total    = len(train_dataset)
    eval_sample_window = max(1, int(num_train_total * 0.1))
    eval_every_steps   = max(1, round(eval_sample_window / config["batch_size"]))
    window_size        = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch  = math.ceil(window_size / config["batch_size"])
    total_steps        = batches_per_epoch * config["epochs"]

    print(f"\nDataset                  : {config['dataset_name']}")
    print(f"Total training samples   : {num_train_total:,}")
    print(f"Window size per epoch    : {window_size:,}  ({config['train_subset_ratio']*100:.1f}% of train)")
    print(f"Batches per epoch        : {batches_per_epoch}")
    print(f"Total optimiser steps    : {total_steps}")
    print(f"Eval every N steps       : {eval_every_steps}  (~1% of train set)\n")

    all_params = list(model.parameters())
    if config['recurrent_steps'] <= 1:
        all_params = [
            p for p in all_params
            if hasattr(model, 'transformer') and p is not model.transformer.norm_weight
        ]

    optimizer = torch.optim.AdamW(
        all_params, lr=config['adam_max_lr'], weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ==========================================
    # EVALUATION HELPER
    # Callers are responsible for all wandb logging.
    # Returns: (acc, f1, precision, recall, jaccard)
    # ==========================================
    @torch.no_grad()
    def evaluate(loader, split_name="val"):
        assert not model.training, "Call model.eval() before evaluating!"
        y_true, y_pred = [], []
        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, data, config)
            pred = logits.argmax(dim=-1, keepdim=True).long()
            y_true.append(data.y.cpu())
            y_pred.append(pred.cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        TP        = ((y_pred == 1) & (y_true == 1)).sum().float()
        FP        = ((y_pred == 1) & (y_true == 0)).sum().float()
        FN        = ((y_pred == 0) & (y_true == 1)).sum().float()
        precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
        recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
        common    = ((y_pred == 1) & (y_true == 1)).sum().item()
        union     = ((y_pred == 1) | (y_true == 1)).sum().item()
        jaccard   = common / union if union > 0 else 0.0

        metrics  = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        acc, f1  = parse_metrics(metrics)

        print(f"  [{split_name}] f1={f1:.4f}  precision={precision:.4f}  "
              f"recall={recall:.4f}  jaccard={jaccard:.4f}  "
              f"TP={int(TP)}  FP={int(FP)}  FN={int(FN)}")
        return acc, f1, precision, recall, jaccard

    # ==========================================
    # WANDB
    # ==========================================
    run_name = (
        f"HRW_BS{config['batch_size']}_HD{config['hidden_dim']}_"
        f"NW{config['num_walks']}_WL{config['walk_length']}_"
        f"P{config['node2vec_p']}_Q{config['node2vec_q']}_"
        f"LR{config['adam_max_lr']}_L{config['layers']}_"
        f"PE-{config['pe_type'] if config['use_pe'] else 'none'}_"
        f"NWPE{config['use_nw_pe']}"
    )
    wandb.init(
        entity="graph-diffusion-model-link-prediction",
        project=f"aniket_maxclique_{config['dataset_name']}",
        name=run_name, config=config
    )

    checkpoint_dir  = os.path.join(config["dataset_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{run_name}.pt")

    # ==========================================
    # PRE-TRAINING EVALUATION (step 0)
    # ==========================================
    print("Running pre-training evaluation (step 0)...")
    model.eval()
    val_acc_0,  val_f1_0,  val_pre_0,  val_rec_0,  val_jac_0  = evaluate(val_loader,  split_name="val")
    test_acc_0, test_f1_0, test_pre_0, test_rec_0, test_jac_0 = evaluate(test_loader, split_name="test")
    wandb.log({
        "global_step":    0,
        "val_acc":        val_acc_0,   "val_f1":        val_f1_0,
        "val_precision":  val_pre_0,   "val_recall":    val_rec_0,   "val_jaccard":  val_jac_0,
        "test_acc":       test_acc_0,  "test_f1":       test_f1_0,
        "test_precision": test_pre_0,  "test_recall":   test_rec_0,  "test_jaccard": test_jac_0,
    })

    # ==========================================
    # TRAINING LOOP
    # ==========================================
    print("\nStarting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_wall       = time.time()
    global_step      = 0
    running_loss_sum = 0.0
    running_loss_cnt = 0

    best_val_f1, best_test_f1 = -float('inf'), -float('inf')
    best_val_step, best_test_step = 0, 0

    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(1, config["epochs"] + 1):

            start_idx    = ((epoch - 1) * window_size) % num_train_total
            indices      = [(start_idx + i) % num_train_total for i in range(window_size)]
            train_loader = DataLoader(
                torch.utils.data.Subset(train_dataset, indices),
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

            print(f'\nEpoch {epoch} — window start={start_idx}, '
                  f'samples={len(indices):,}, batches={len(train_loader)}')

            model.train()

            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad(set_to_none=True)
                start_time_batch = time.time()

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = run_forward(model, data, config)
                    loss   = criterion(logits, data.y.long())

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
                optimizer.step()

                global_step      += 1
                running_loss_sum += loss.item()
                running_loss_cnt += 1

                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({"train_loss": loss.item(), "lr": current_lr, "global_step": global_step})

                if torch.cuda.is_available():
                    free, total_mem = torch.cuda.mem_get_info(device)
                    used_vram = (total_mem - free) / total_mem
                else:
                    used_vram = 0.0

                pbar.update(1)
                pbar.set_postfix({
                    'epoch': epoch,
                    'loss':  f'{loss.item():.4f}',
                    'mem':   f'{used_vram:.2f}',
                    'time':  f'{time.time() - start_time_batch:.2f}s',
                })

                # ---- STEP-BASED EVALUATION ----
                if global_step % eval_every_steps == 0:
                    print('Beginning Evaluation...')
                    avg_train_loss   = running_loss_sum / running_loss_cnt
                    running_loss_sum = 0.0
                    running_loss_cnt = 0

                    model.eval()
                    val_acc,  val_f1,  val_pre,  val_rec,  val_jac  = evaluate(val_loader,  split_name="val")
                    test_acc, test_f1, test_pre, test_rec, test_jac = evaluate(test_loader, split_name="test")

                    wandb.log({
                        "global_step":    global_step,
                        "epoch":          epoch,
                        "avg_train_loss": avg_train_loss,
                        "val_acc":        val_acc,        "val_f1":        val_f1,
                        "val_precision":  val_pre,        "val_recall":    val_rec,   "val_jaccard":  val_jac,
                        "test_acc":       test_acc,       "test_f1":       test_f1,
                        "test_precision": test_pre,       "test_recall":   test_rec,  "test_jaccard": test_jac,
                        "best_val_f1":    best_val_f1,    "best_test_f1":  best_test_f1,
                    })

                    print(
                        f"\n[Step {global_step} | Epoch {epoch}] "
                        f"avg_train_loss={avg_train_loss:.4f} | "
                        f"val_f1={val_f1:.4f}  test_f1={test_f1:.4f}"
                    )

                    if val_f1 > best_val_f1:
                        best_val_f1   = val_f1
                        best_val_step = global_step
                        print(f"  >> New Best Val F1: {best_val_f1:.4f}  (step {best_val_step})")

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
                              f"(step {best_test_step}) — checkpoint saved to {best_model_path}")

                    model.train()

    # ==========================================
    # FINAL EVALUATION
    # ==========================================
    print("\nRunning final evaluation...")
    model.eval()
    final_val_acc,  final_val_f1,  *_ = evaluate(val_loader,  split_name="val")
    final_test_acc, final_test_f1, *_ = evaluate(test_loader, split_name="test")

    print("\nTraining complete.")
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    if torch.cuda.is_available():
        print(f"Peak CUDA memory : {peak_mem:.2f} GiB")
    print(f"Total time       : {total_time:.2f} s")

    wandb.log({
        "peak_cuda_memory_gb": peak_mem,
        "total_runtime_sec":   total_time,
        "total_parameters":    total_params,
        "final_val_acc":       final_val_acc,
        "final_val_f1":        final_val_f1,
        "final_test_acc":      final_test_acc,
        "final_test_f1":       final_test_f1,
        "best_val_f1":         best_val_f1,
        "best_test_f1":        best_test_f1,
        "best_val_step":       best_val_step,
    })

    # ==========================================
    # CSV LOG
    # ==========================================
    os.makedirs(config['dataset_name'], exist_ok=True)
    csv_path = f"{config['dataset_name']}/hyperparam_sweep_results_new_v3.csv"
    row = {
        "dataset":            config["dataset_name"],
        "seed":               config["seed"],
        "batch_size":         config["batch_size"],
        "adam_max_lr":        config["adam_max_lr"],
        "epochs":             config["epochs"],
        "dropout":            config["dropout"],
        "hidden_dim":         config["hidden_dim"],
        "walk_length":        config["walk_length"],
        "num_walks":          config["num_walks"],
        "layers":             config["layers"],
        "train_subset_ratio": config["train_subset_ratio"],
        "eval_every_steps":   eval_every_steps,
        "best_val_f1":        round(best_val_f1,    4),
        "best_test_f1":       round(best_test_f1,   4),
        "best_val_step":      best_val_step,
        "final_val_f1":       round(final_val_f1,   4),
        "final_test_f1":      round(final_test_f1,  4),
        "use_pe":             config["use_pe"],
        "pe_dim":             config["pe_dim"],
        "pe_type":            config["pe_type"],
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"Logged to {csv_path}")

    wandb.finish()


if __name__ == "__main__":
    main()