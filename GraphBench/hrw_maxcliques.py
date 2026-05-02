from types import SimpleNamespace
from pandas import core
from sympy import evaluate
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
from torch_geometric.utils import degree
import networkx as nx
import torch.compiler

_original_torch_load = torch.load
def torch_load_no_weights_only(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = torch_load_no_weights_only
from scipy.sparse import csr_matrix


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
    Compute NeuralWalker-style positional encodings for a batch of random walks.

    For each walk token at position ``t``, two families of binary features are
    computed by scanning backwards over a context window of size ``window_size``:

    **Identity features** (``window_size`` channels):
        ``id_feats[k][t] = 1``  iff  ``walks[t] == walks[t-k]``,
        i.e. the current node was already visited ``k`` steps ago.
        Captures node revisit patterns and encodes structural periodicity
        within the walk (analogous to identity encoding in CRaWL).

    **Adjacency features** (``window_size`` channels):
        ``adj_feats[k][t] = 1``  iff  edge ``(walks[t-k], walks[t])`` exists
        in the graph, regardless of whether that edge was actually traversed.
        Captures off-walk structural connectivity between walk-visited nodes,
        enabling detection of triangles, chords, and higher-order motifs that
        pure identity encoding misses.

    Both families are zero-padded at the start of each walk so that token ``t``
    always has exactly ``window_size`` identity channels and ``window_size``
    adjacency channels, giving a total of ``2 * window_size`` features per token.

    Args:
        walks       (Tensor): Shape ``[B, W, L]`` — batched random walks where
                              ``B`` is batch size, ``W`` is walks per node, and
                              ``L`` is walk length.
        edge_index  (Tensor): Shape ``[2, E]`` — graph connectivity as
                              (source, destination) node index pairs.
        num_nodes   (int):    Total number of nodes in the graph. Used to
                              build unique edge hashes ``src * num_nodes + dst``
                              for O(log E) adjacency lookup via ``searchsorted``.
        window_size (int):    Number of look-back steps to consider.
                              Produces ``window_size`` identity and
                              ``window_size`` adjacency channels. Default ``8``.

    Returns:
        Tensor: Shape ``[B, W, L, 2 * window_size]``.
                The last dimension is ``[id_k=1, …, id_k=W, adj_k=1, …, adj_k=W]``.
                All values are ``0.0`` or ``1.0`` (binary float).
    """
    device = walks.device
    B, W, L = walks.shape
    id_feats, adj_feats = [], []
    for k in range(1, window_size + 1):
        is_same = (walks[:, :, k:] == walks[:, :, :-k]).float()
        id_feats.append(
            torch.cat([torch.zeros((B, W, k), device=device), is_same], dim=2).unsqueeze(-1)
        )
    row, col = edge_index
    edge_hashes = row * num_nodes + col
    sorted_hashes, _ = torch.sort(torch.unique(edge_hashes))
    for k in range(1, window_size + 1):
        query_hashes = walks[:, :, k:] * num_nodes + walks[:, :, :-k]
        idx = torch.searchsorted(sorted_hashes, query_hashes).clamp(max=sorted_hashes.size(0) - 1)
        is_connected = (sorted_hashes[idx] == query_hashes).float()
        adj_feats.append(
            torch.cat([torch.zeros((B, W, k), device=device), is_connected], dim=2).unsqueeze(-1)
        )
    return torch.cat([torch.cat(id_feats, dim=-1), torch.cat(adj_feats, dim=-1)], dim=-1)


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
            nn.Linear(2, embed_dim, bias=False), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
        )
        self.rho = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        if position_aware:
            self.eps = nn.Parameter(1e-12 * torch.arange(lpe_inner_dim)[None])
        self.position_aware  = position_aware
        self.lpe_num_eigvals = lpe_num_eigvals

    def forward(self, eigvecs, eigvals, is_training):
        eigvecs = eigvecs[:, :self.lpe_num_eigvals]
        eigvals = eigvals[:, :self.lpe_num_eigvals]
        if is_training:
            sign_flip = torch.rand(eigvecs.size(1), device=eigvecs.device)
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip <  0.5] = -1.0
            eigvecs = eigvecs * sign_flip.unsqueeze(0)
        if self.position_aware:
            eigvals = eigvals + self.eps[:, :self.lpe_num_eigvals]
        x = torch.stack((eigvecs, eigvals), 2)
        x[torch.isnan(x)] = 0
        return self.rho(self.phi(x).sum(1)), None


class CustomLaplacianPE:
    """On-the-fly LapPE — slow. Use --use_cached_lap_pe when possible."""
    def __init__(self, max_freq, normalized=False, normalize=False, large_graph=False):
        self.max_freq = max_freq; self.normalized = normalized
        self.normalize = normalize; self.large_graph = large_graph

    def __call__(self, data):
        eigvals, eigvecs = compute_laplacian_eigen(
            data.edge_index, data.num_nodes, self.max_freq,
            self.normalized, self.normalize, self.large_graph
        )
        data.eigvecs = eigvecs; data.eigvals = eigvals
        return data


class CachedLapPE:
    """
    Fast LapPE: load pre-computed eigenvectors from disk (zero CPU cost per step).
    Generate the cache with: python precompute_lap_pe.py
    """
    def __init__(self, cache_file: str):
        print(f"  Loading LapPE cache: {cache_file}")
        saved = torch.load(cache_file, weights_only=False)
        self.eigvecs = saved["eigvecs"]
        self.eigvals = saved["eigvals"]
        self._idx    = 0

    def reset(self):
        self._idx = 0

    def __call__(self, data):
        if self._idx >= len(self.eigvecs):
            raise IndexError(
                f"CachedLapPE ran out of entries at idx={self._idx} "
                f"(cache has {len(self.eigvecs)} entries). Call .reset() before reuse."
            )
        data.eigvecs = self.eigvecs[self._idx]
        data.eigvals = self.eigvals[self._idx]
        self._idx   += 1
        return data


class RWSE(nn.Module):
    def __init__(self, rwse_steps, embed_dim):
        super().__init__()
        self.rwse_steps = rwse_steps
        self.encoder = nn.Sequential(
            nn.Linear(rwse_steps, 2 * embed_dim), nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, rwse, is_training):
        return self.encoder(rwse[:, :self.rwse_steps])


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
        theta = 1.0 / (self.base ** (torch.arange(0, self.dim, 2)[:self.dim // 2].float() / self.dim))
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len=4096):
        idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        cache = torch.stack([
            torch.cos(torch.einsum("i,j->ij", idx, self.theta)),
            torch.sin(torch.einsum("i,j->ij", idx, self.theta))
        ], dim=-1)
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
                 attn_dropout_p=0.0, ffn_dropout_p=0.0, resid_dropout_p=0.0,
                 drop_path_p=0.0, config=MupConfig()):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim; self.num_heads = num_heads
        self.attn_dropout_p = attn_dropout_p; self.ffn_dropout_p = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p; self.drop_path_p = drop_path_p
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight  = nn.Parameter(torch.ones(hidden_dim))
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o   = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rope = RotaryPositionalEmbeddings(dim=(hidden_dim // num_heads), max_seq_len=seq_len)
        s, m = config.init_std, config.mup_width_multiplier
        nn.init.normal_(self.up.weight,   std=s / math.sqrt(2 * n_layer * m))
        nn.init.normal_(self.gate.weight, std=s / math.sqrt(2 * n_layer * m))
        nn.init.normal_(self.down.weight, std=s / math.sqrt(m))
        nn.init.normal_(self.qkv.weight,  std=s / math.sqrt(m))
        nn.init.normal_(self.o.weight,    std=s / math.sqrt(m))

    def _drop_path(self, x):
        if self.drop_path_p <= 0.0 or not self.training: return x
        keep = 1.0 - self.drop_path_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep) * torch.floor(keep + torch.rand(shape, dtype=x.dtype, device=x.device))

    def forward(self, x, offset=None):
        ax = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        q, k, v = self.qkv(ax).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset); k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d') for t in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / k.shape[-1])
        o = rearrange(o, 'n h t d -> n t (h d)', h=self.num_heads)
        attn_out = self.o(o)
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)
        fx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(fx)) * self.gate(fx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        return x + self._drop_path(ffn_out)


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads,
                 num_walks, seq_len, attn_dropout_p=0.0, ffn_dropout_p=0.0,
                 resid_dropout_p=0.0, drop_path_p=0.0, config=MupConfig()):
        super().__init__()
        self.hidden_dim = hidden_dim; self.num_heads = num_heads
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads, num_walks * seq_len,
                n_layer=num_layers, attn_dropout_p=attn_dropout_p,
                ffn_dropout_p=ffn_dropout_p, resid_dropout_p=resid_dropout_p,
                drop_path_p=drop_path_p, config=config
            ) for _ in range(num_layers)
        ])
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, 
        x, 
        anon_indices, 
        source_nodes=None):
        # x: [N, num_walks * walk_length, encoding_dim]
        # anon_indices: list of [N, num_walks * walk_length] anonymised walk indices
        # For recurrent_steps > 1, anon_indices has multiple entries.
        # Each step: run transformer layers, then if not the last step,
        # take the last token of each walk group and re-expand for the next level.
        N, ctx_len, D = x.shape
        for depth, idx in enumerate(reversed(anon_indices)):
            for l in self.layers:
                x = l(x, idx)
            if depth < len(anon_indices) - 1:
                # Extract last token: [N, D], then treat each token as a new
                # "source node" whose walk features are the next level's tokens.
                # x has shape [N, ctx_len, D] → take last token → [N, D]
                # Then re-expand so shape is [N_prev, ctx_len, D] for the outer level.
                # N at this level = N_prev * ctx_len (walk nodes from prev level)
                x_last = F.rms_norm(x[:, -1, :], [self.hidden_dim], eps=1e-5) * self.norm_weight
                # x_last: [N, D] — N nodes at this recurrent level
                # reshape back to [N_prev, ctx_len, D] where N_prev = N // ctx_len
                x = x_last.view(-1, ctx_len, self.hidden_dim)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return x[:, -1, :]


# ==========================================
# RANDOM WALK UTILITIES
# ==========================================

@torch.no_grad()
def anonymize_rws(rws, rev_walks=True):
    if rev_walks: rws = torch.flip(rws, dims=[-1])
    s, _  = torch.sort(rws, dim=-1)
    su    = torch.searchsorted(s, rws)
    c     = torch.full_like(s, fill_value=s.shape[-1])
    rw_i  = torch.arange(rws.shape[-1], device=rws.device)[None, :].expand_as(s)
    first = c.scatter_reduce_(-1, su, rw_i, reduce="amin")
    ret   = first.gather(-1, su)
    if rev_walks: ret = torch.flip(ret, dims=[-1])
    return ret


def get_walk_edge_attrs(edge_index, edge_attr, walks, num_nodes):
    device = edge_index.device
    row, col = edge_index
    sorted_hashes, perm = torch.sort(row.long() * num_nodes + col.long())
    src = walks[:, :, :-1].flatten().long()
    dst = walks[:, :, 1:].flatten().long()
    query = src * num_nodes + dst
    # Bug fix: clamp searchsorted result and verify match before indexing perm.
    # Walks that land on padding/isolated nodes produce hashes not in sorted_hashes;
    # without this check perm[idx] silently returns a wrong edge attribute.
    raw_idx   = torch.searchsorted(sorted_hashes, query).clamp(max=sorted_hashes.size(0) - 1)
    matched   = sorted_hashes[raw_idx] == query          # [E] bool mask
    safe_idx  = torch.where(matched, perm[raw_idx], torch.zeros_like(raw_idx))
    b, w, l   = walks.shape
    edge_dim  = edge_attr.size(-1)
    step_attrs = torch.where(
        matched.unsqueeze(-1),
        edge_attr[safe_idx],
        torch.zeros(matched.size(0), edge_dim, device=device, dtype=edge_attr.dtype)
    ).view(b, w, l - 1, edge_dim)
    return torch.cat([
        torch.zeros((b, w, 1, edge_dim), device=device, dtype=edge_attr.dtype),
        step_attrs,
    ], dim=2)


@torch.compiler.disable
def get_random_walk_batch(edge_index, x, start_nodes, walk_length, num_walks,
                          num_nodes, recurrent_steps=1, p=1.0, q=1.0):
    with torch.no_grad():
        row, col        = edge_index
        current_sources = start_nodes
        rws_list        = []
        for _ in range(recurrent_steps):
            walks = cluster_random_walk(
                row, col,
                current_sources.repeat_interleave(num_walks),
                walk_length - 1, p=p, q=q, num_nodes=num_nodes
            ).view(current_sources.size(0), num_walks, walk_length)
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
        self.encoding_dim    = config.get("encoding_dim", 384)
        self.node2vec_p      = config.get("node2vec_p", 1.0)
        self.node2vec_q      = config.get("node2vec_q", 1.0)
        self.nw_pe_window    = config.get("nw_pe_window", 5)
        self.ffn_multiplier  = config.get("ffn_multiplier", 4.0)

        self.node_encoder = nn.Linear(node_in_dim, self.encoding_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, self.encoding_dim)
        self.input_norm   = nn.LayerNorm(self.encoding_dim)

        self.use_nw_pe = config.get("use_nw_pe", False)
        if self.use_nw_pe:
            self.nw_pe_encoder = nn.Linear(2 * self.nw_pe_window, self.encoding_dim)
            nn.init.normal_(self.nw_pe_encoder.weight, std=config['mup_init_std'])

        self.use_lap_pe = config.get("use_lap_pe", False)
        if self.use_lap_pe:
            self.lpe = LPE(
                embed_dim=self.encoding_dim,
                lpe_num_eigvals=config["lap_pe_dim"],
                position_aware=True,
                lpe_bias=True,
                lpe_inner_dim=config["lap_pe_dim"],
            )

        self.use_rwse = config.get("use_rwse", False)
        if self.use_rwse:
            self.rwse_encoder = RWSE(
                rwse_steps=config["rwse_dim"],
                embed_dim=self.encoding_dim,
            )

        nn.init.normal_(self.edge_encoder.weight, std=config['mup_init_std'])
        nn.init.normal_(self.node_encoder.weight, std=config['mup_init_std'])

        self.transformer = Transformer(
            emb_dim=self.encoding_dim, num_layers=config["layers"],
            hidden_dim=self.hidden_dim,
            intermediate_dim=int(self.hidden_dim * self.ffn_multiplier),
            num_heads=config.get("num_heads", 8), num_walks=self.num_walks,
            seq_len=self.walk_length, attn_dropout_p=config["dropout"],
            ffn_dropout_p=config["dropout"], config=mup_config
        )
        self.classifier = NodeDecoder(hidden_dim=self.hidden_dim, dropout=config["dropout"])

    def forward(self, x, edge_index, edge_attr,
                mp_edge_index=None, mp_edge_attr=None,
                eigvecs=None, eigvals=None, rwse=None):
        if x.dim() == 1: x = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr = edge_attr.unsqueeze(-1)

        walk_ei = mp_edge_index if mp_edge_index is not None else edge_index
        deg = degree(walk_ei[1], num_nodes=x.size(0), dtype=torch.float).unsqueeze(-1)  # [N, 1]
        # Log-scale to avoid large values on high-degree nodes
        x = torch.log1p(deg)

        x_emb = self.input_norm(self.node_encoder(x.float()))

        if self.use_lap_pe and eigvecs is not None and eigvals is not None:
            lpe_out, _ = self.lpe(eigvecs, eigvals, self.training)
            x_emb = x_emb + lpe_out

        if self.use_rwse and rwse is not None:
            x_emb = x_emb + self.rwse_encoder(rwse.float(), self.training)

        walk_ei = mp_edge_index if mp_edge_index is not None else edge_index
        walk_ea = mp_edge_attr  if mp_edge_attr  is not None else edge_attr
        if walk_ea.dim() == 1: walk_ea = walk_ea.unsqueeze(-1)
        walk_ea_emb = self.input_norm(self.edge_encoder(walk_ea.float()))

        batch_feats, anon_idx, raw_walks = get_random_walk_batch(
            edge_index=walk_ei, x=x_emb,
            start_nodes=torch.arange(x.size(0), device=x.device),
            walk_length=self.walk_length, num_walks=self.num_walks,
            num_nodes=x.size(0), recurrent_steps=self.recurrent_steps,
            p=self.node2vec_p, q=self.node2vec_q
        )
        walks_view  = raw_walks[-1].view(x.size(0), self.num_walks, self.walk_length)
        batch_feats = batch_feats + get_walk_edge_attrs(
            walk_ei, walk_ea_emb, walks_view, x.size(0)
        ).flatten(1, 2)

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


def global_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5


# ==========================================
# MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name",         type=str,   default="maxclique_easy")
    parser.add_argument("--data_root",            type=str,   default="./data_graphbench")
    parser.add_argument("--seed",                 type=int,   default=2025)
    parser.add_argument("--epochs",               type=int,   default=300)
    # FIX 5: increased default batch size for A100
    parser.add_argument("--batch_size",           type=int,   default=512)
    parser.add_argument("--test_batch_size",      type=int,   default=128)
    parser.add_argument("--hidden_dim",           type=int,   default=256)
    parser.add_argument("--encoding_dim",         type=int,   default=256)
    parser.add_argument("--layers",               type=int,   default=1)
    parser.add_argument("--num_heads",            type=int,   default=8)
    parser.add_argument("--dropout",              type=float, default=0.1)
    parser.add_argument("--ffn_multiplier",       type=float, default=4.0)
    # FIX 5b: lr scaled linearly with batch size (linear scaling rule)
    # original: bs=256 lr=1e-4  →  bs=512 lr=2e-4
    parser.add_argument("--adam_max_lr",          type=float, default=2e-4)
    parser.add_argument("--weight_decay",         type=float, default=0.0)
    parser.add_argument("--grad_clip_norm",       type=float, default=0.5)
    parser.add_argument("--train_subset_ratio",   type=float, default=0.2)
    parser.add_argument("--walk_length",          type=int,   default=8)
    parser.add_argument("--num_walks",            type=int,   default=10)
    parser.add_argument("--node2vec_p",           type=float, default=1.0)
    parser.add_argument("--node2vec_q",           type=float, default=1.0)
    parser.add_argument("--recurrent_steps",      type=int,   default=1)
    parser.add_argument("--use_lap_pe",           type=lambda x: x.lower()=='true', default=False)
    parser.add_argument("--lap_pe_dim",           type=int,   default=16)
    # FIX 1: cache flag
    parser.add_argument("--use_cached_lap_pe",    type=lambda x: x.lower()=='true', default=False,
                        help="Load pre-computed LapPE from disk (run precompute_lap_pe.py first)")
    parser.add_argument("--use_rwse",             type=lambda x: x.lower()=='true', default=False)
    parser.add_argument("--rwse_dim",             type=int,   default=16)
    parser.add_argument("--nw_pe_window",         type=int,   default=7)
    parser.add_argument("--use_nw_pe",            type=lambda x: x.lower()=='true', default=False)
    parser.add_argument("--mup_init_std",         type=float, default=0.01)
    parser.add_argument("--mup_width_multiplier", type=float, default=2.0)
    parser.add_argument("--eval_metric_class",    type=str,   default='algoreas_classification')
    parser.add_argument("--wandb_project",        type=str,   default='bench_maxclique')
    parser.add_argument("--log_every",            type=int,   default=10)

    args   = parser.parse_args()
    config = vars(args)

    import pprint
    pprint.pp(config)

    set_seed(config["seed"])
    mup_config = MupConfig(init_std=config['mup_init_std'],
                           mup_width_multiplier=config['mup_width_multiplier'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ────────────────────────────────────────────────────────
    transforms_list = [AddUndirectedContext()]
    # Only add on-the-fly LapPE if not using cache
    if config["use_lap_pe"] and not config["use_cached_lap_pe"]:
        transforms_list.append(
            CustomLaplacianPE(max_freq=config["lap_pe_dim"], normalized=True, normalize=True)
        )
    if config["use_rwse"]:
        transforms_list.append(
            T.AddRandomWalkPE(walk_length=config["rwse_dim"], attr_name='rwse')
        )

    loader  = graphbench.Loader(root=config["data_root"],
                                dataset_names=config["dataset_name"],
                                transform=T.Compose(transforms_list))
    dataset = loader.load()

    try:
        train_dataset = dataset[0]['train']
        val_dataset   = dataset[0]['valid']
        test_dataset  = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = val_dataset = test_dataset = dataset

    # FIX 1: attach cached LapPE after loading — zero cost per training step
    if config["use_lap_pe"] and config["use_cached_lap_pe"]:
        def _attach(ds, split):
            cf = os.path.join(
                config["data_root"],
                f"{config['dataset_name']}_{split}_lapPE{config['lap_pe_dim']}.pt"
            )
            if os.path.exists(cf):
                lap_tf = CachedLapPE(cf)
            else:
                print(f"  WARNING: cache not found at {cf}, falling back to on-the-fly.")
                lap_tf = CustomLaplacianPE(config["lap_pe_dim"])
            return [lap_tf(ds[i]) for i in range(len(ds))]

        print("Attaching LapPE from cache ...")
        t0 = time.time()
        train_dataset = _attach(train_dataset, "train")
        val_dataset   = _attach(val_dataset,   "valid")
        test_dataset  = _attach(test_dataset,  "test")
        print(f"LapPE cache attached in {time.time()-t0:.1f}s")

    print(f"Sizes → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    dl_workers = max(5, min(8, int(os.environ.get("SLURM_CPUS_PER_TASK", 1)) - 1))
    print(f"DataLoader workers: {dl_workers}")
    dl_kwargs = dict(
        num_workers        = dl_workers,
        pin_memory         = torch.cuda.is_available() and dl_workers > 0,
        persistent_workers = dl_workers > 0,
        **({"prefetch_factor": 2} if dl_workers > 0 else {}),
    )

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"],
                             shuffle=False, **dl_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"],
                             shuffle=False, **dl_kwargs)

    _peek       = next(iter(DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"Input dims → node: {node_in_dim}  edge: {edge_in_dim}")

    model = RWTransformerForNodeClassification(
        node_in_dim, edge_in_dim, config, mup_config
    ).to(device)
    total_params = count_parameters(model)
    print(f"Parameters: {total_params:,}")

    num_train_total   = len(train_dataset)
    window_size       = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch = math.ceil(window_size / config["batch_size"])
    total_steps       = batches_per_epoch * config["epochs"]
    eval_every_steps  = max(1, round(max(1, int(num_train_total * 0.1)) / config["batch_size"]))

    print(f"Dataset: {config['dataset_name']} | train={num_train_total:,} "
          f"| window={window_size:,} | steps/epoch={batches_per_epoch} "
          f"| total={total_steps} | eval_every={eval_every_steps}")

    all_params = list(model.parameters())
    if config['recurrent_steps'] <= 1:
        all_params = [p for p in all_params
                      if hasattr(model, 'transformer') and p is not model.transformer.norm_weight]
    optimizer = torch.optim.AdamW(all_params, lr=config['adam_max_lr'],
                                  weight_decay=config['weight_decay'])

    # FIX 4: class-weighted loss for clique imbalance
    all_labels = torch.cat([train_dataset[i].y for i in range(len(train_dataset))])
    n_total    = all_labels.numel()
    n_pos      = (all_labels == 1).sum().item()
    n_neg      = n_total - n_pos
    w_pos      = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
    w_neg      = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
    print(f"Class weights → neg: {w_neg:.3f}  pos: {w_pos:.3f}  "
          f"(clique ratio: {n_pos/n_total*100:.1f}%)")
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([w_neg, w_pos], dtype=torch.float, device=device)
    )

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    pe_tag = []
    if config["use_lap_pe"]: pe_tag.append(f"lap{config['lap_pe_dim']}")
    if config["use_rwse"]:   pe_tag.append(f"rwse{config['rwse_dim']}")
    pe_str   = "+".join(pe_tag) if pe_tag else "none"
    run_name = (
        f"HRW_{config['dataset_name']}_"
        f"PE-{pe_str}_"
        f"HRW_BS{config['batch_size']}_HD{config['hidden_dim']}_"
        f"NW{config['num_walks']}_WL{config['walk_length']}_"
        f"LR{config['adam_max_lr']}_L{config['layers']}"
    )
    wandb.init(
        entity="graph-diffusion-model-link-prediction",
        project=config["wandb_project"],
        name=run_name, config=config
    )
    wandb.summary['total_params'] = total_params

    checkpoint_dir  = os.path.join(config["dataset_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"best_model_{run_name}.pt")

    @torch.no_grad()
    def evaluate(loader, split_name="Val"):
        assert not model.training
        y_true, y_pred = [], []
        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, data, config)
            y_true.append(data.y.cpu())
            y_pred.append(logits.argmax(-1, keepdim=True).long().cpu())

        y_true = torch.cat(y_true); y_pred = torch.cat(y_pred)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        TP = ((y_pred == 1) & (y_true == 1)).sum().float()
        FP = ((y_pred == 1) & (y_true == 0)).sum().float()
        FN = ((y_pred == 0) & (y_true == 1)).sum().float()
        precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
        recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
        common    = ((y_pred == 1) & (y_true == 1)).sum().item()
        union     = ((y_pred == 1) | (y_true == 1)).sum().item()
        jaccard   = common / union if union > 0 else 0.0
        metrics   = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        acc, f1   = parse_metrics(metrics)
        return dict(f1=f1, acc=acc, precision=precision, recall=recall, jaccard=jaccard,
                    TP=int(TP), FP=int(FP), FN=int(FN))

    model.eval()
    v0 = evaluate(val_loader, "Val"); t0 = evaluate(test_loader, "Test")
    print(f"[step 0] val_f1={v0['f1']:.4f}  test_f1={t0['f1']:.4f}")
    wandb.log({"step": 0, **{f"val/{k}": v for k, v in v0.items()},
                           **{f"test/{k}": v for k, v in t0.items()}})

    print("\nStarting training...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    start_wall       = time.time()
    global_step      = 0
    running_loss_sum = 0.0
    running_loss_cnt = 0
    best_val_f1, best_test_f1     = -float('inf'), -float('inf')
    best_val_step, best_test_step = 0, 0
    vram = 0.0

    for epoch in range(1, config["epochs"] + 1):
        start_idx = ((epoch - 1) * window_size) % num_train_total
        indices   = [(start_idx + i) % num_train_total for i in range(window_size)]
        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, indices),
            batch_size=config["batch_size"], shuffle=True,
            **dl_kwargs   # FIX 3: dynamic workers
        )
        model.train()
        epoch_t0 = time.time()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, data, config)
                loss   = criterion(logits, data.y.long())

            loss.backward()
            gnorm = global_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
            optimizer.step()

            global_step      += 1
            running_loss_sum += loss.item()
            running_loss_cnt += 1

            # FIX 2: mem_get_info only every 50 steps — avoids GPU sync each step
            # if global_step % 50 == 0 and torch.cuda.is_available():
            #     free, total_mem = torch.cuda.mem_get_info(device)
            #     vram = (total_mem - free) / total_mem

            if global_step % config["log_every"] == 0:
                wandb.log({
                    "step": global_step, "epoch": epoch,
                    "train/loss": loss.item(), "train/grad_norm": gnorm,
                    "train/lr": optimizer.param_groups[0]['lr'],
                    # "train/vram": vram,
                }, step=global_step)

            print(
                f"\rEp {epoch}/{config['epochs']} "
                f"[{batch_idx+1}/{len(train_loader)}] "
                f"step={global_step} loss={loss.item():.4f} "
                f"gnorm={gnorm:.3f} " # mem={vram:.2f}
                f"t={time.time()-epoch_t0:.0f}s",
                end="", flush=True
            )

            if global_step % eval_every_steps == 0:
                avg_loss = running_loss_sum / running_loss_cnt
                running_loss_sum = 0.0; running_loss_cnt = 0

                model.eval()
                val_res  = evaluate(val_loader,  "Val")
                test_res = evaluate(test_loader, "Test")
                model.train()

                wandb.log({
                    "step": global_step, "epoch": epoch,
                    "train/avg_loss": avg_loss,
                    "best/val_f1": best_val_f1, "best/test_f1": best_test_f1,
                    **{f"val/{k}":  v for k, v in val_res.items()},
                    **{f"test/{k}": v for k, v in test_res.items()},
                }, step=global_step)

                print(
                    f"\n  [eval step={global_step}] "
                    f"val_f1={val_res['f1']:.4f} "
                    f"test_f1={test_res['f1']:.4f} "
                    f"avg_loss={avg_loss:.4f}"
                )

                if val_res['f1'] > best_val_f1:
                    best_val_f1 = val_res['f1']; best_val_step = global_step
                    wandb.summary['best_val_f1']  = best_val_f1
                    wandb.summary['best_val_step'] = best_val_step
                    print(f"  ★ new best val  F1={best_val_f1:.4f}  (step {best_val_step})")

                if test_res['f1'] > best_test_f1:
                    best_test_f1 = test_res['f1']; best_test_step = global_step
                    wandb.summary['best_test_f1']  = best_test_f1
                    wandb.summary['best_test_step'] = best_test_step
                    print(f"  ★ new best test F1={best_test_f1:.4f}  (step {best_test_step})")

        print(
            f"\n[epoch {epoch}/{config['epochs']} done] "
            f"wall={time.time()-start_wall:.0f}s  "
            f"best_val={best_val_f1:.4f}  best_test={best_test_f1:.4f}"
        )

    print("\nFinal evaluation...")
    model.eval()
    final_val  = evaluate(val_loader,  "Val")
    final_test = evaluate(test_loader, "Test")
    peak_mem   = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"Final → val_f1={final_val['f1']:.4f}  test_f1={final_test['f1']:.4f}")
    print(f"Peak VRAM: {peak_mem:.2f} GiB  |  Total time: {total_time:.0f}s")

    wandb.log({
        "step": global_step,
        "final/val_f1": final_val['f1'], "final/test_f1": final_test['f1'],
        "final/peak_vram_gib": peak_mem, "final/runtime_sec": total_time,
    }, step=global_step)
    wandb.summary.update({
        'best_val_f1': best_val_f1, 'best_test_f1': best_test_f1,
        'best_val_step': best_val_step, 'final_val_f1': final_val['f1'],
        'final_test_f1': final_test['f1'], 'total_params': total_params,
    })

    os.makedirs(config['dataset_name'], exist_ok=True)
    csv_path = f"{config['dataset_name']}/hyperparam_sweep_results.csv"
    row = {
        "dataset": config["dataset_name"], "seed": config["seed"],
        "batch_size": config["batch_size"], "adam_max_lr": config["adam_max_lr"],
        "epochs": config["epochs"], "dropout": config["dropout"],
        "hidden_dim": config["hidden_dim"], "walk_length": config["walk_length"],
        "num_walks": config["num_walks"], "layers": config["layers"],
        "train_subset_ratio": config["train_subset_ratio"],
        "eval_every_steps": eval_every_steps,
        "best_val_f1":   round(best_val_f1,        4),
        "best_test_f1":  round(best_test_f1,       4),
        "best_val_step": best_val_step,
        "final_val_f1":  round(final_val['f1'],    4),
        "final_test_f1": round(final_test['f1'],   4),
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: w.writeheader()
        w.writerow(row)
    print(f"Logged to {csv_path}")
    wandb.finish()


if __name__ == "__main__":
    main()

# best uv run hrw_maxcliques.py --dataset_name maxclique_easy --walk_length 8 --num_walks 10 