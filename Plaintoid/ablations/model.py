"""
heart_model.py
--------------
Self-contained model definition for HeART (Heterogeneous Anonymized Random-walk Transformer)
for link prediction on graphs.

Contains:
  - MupConfig                    : μP initialization config dataclass
  - RotaryPositionalEmbeddings   : RoPE for sequence positional encoding
  - TransformerLayer             : Single Transformer block (RMSNorm, RoPE, SwiGLU, drop-path)
  - Transformer                  : Full encoder stacking TransformerLayers
  - LinkPredictorMLP             : MLP-based link scoring head
  - anonymize_rws                : Walk anonymization (replaces node IDs with relative first-occurrence indices)
  - get_random_walk_batch        : Samples, anonymizes, and fetches features for random walk batches
  - binary_cross_entropy_loss    : BCE loss over positive/negative edge scores
"""

import math
import dataclasses
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch_sparse import SparseTensor
from torch_cluster import random_walk as cluster_random_walk


# ---------------------------------------------------------------------------
# μP Configuration
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings (RoPE)
# ---------------------------------------------------------------------------
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10_000) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)


# ---------------------------------------------------------------------------
# Transformer Layer
# ---------------------------------------------------------------------------

class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, num_heads, seq_len, n_layer,
                 attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0,
                 resid_dropout_p: float = 0.0, drop_path_p: float = 0.0,
                 config=MupConfig()):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dropout_p = attn_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.drop_path_p = drop_path_p

        # Attention projections
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o   = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # FFN (SwiGLU)
        self.up   = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)

        # Learnable RMSNorm scales
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight  = nn.Parameter(torch.ones(hidden_dim))

        # RoPE
        self.rope = RotaryPositionalEmbeddings(
            dim=(hidden_dim // num_heads), max_seq_len=seq_len
        )

        # μP-style initialization
        std_base  = config.init_std / math.sqrt(config.mup_width_multiplier)
        std_resid = config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier)
        nn.init.normal_(self.up.weight,   mean=0.0, std=std_resid)
        nn.init.normal_(self.gate.weight, mean=0.0, std=std_resid)
        nn.init.normal_(self.down.weight, mean=0.0, std=std_base)
        nn.init.normal_(self.qkv.weight,  mean=0.0, std=std_base)
        nn.init.normal_(self.o.weight,    mean=0.0, std=std_base)

    def forward(self, x, offset=None):
        # --- Self-Attention sublayer ---
        attnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        qkv = self.qkv(attnx)
        q, k, v = qkv.chunk(3, dim=-1)

        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d', h=self.num_heads) for t in (q, k, v)]

        o_walks = F.scaled_dot_product_attention(
            q, k, v, is_causal=False, scale=1.0 / k.shape[-1]
        )
        o_walks = rearrange(o_walks, 'n h t d -> n t (h d)', h=self.num_heads)
        attn_out = self.o(o_walks)
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)

        # --- FFN sublayer (SwiGLU) ---
        ffnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(ffnx)) * self.gate(ffnx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        x = x + self._drop_path(ffn_out)
        return x

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic depth (drop path) regularization."""
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_path_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor

# ---------------------------------------------------------------------------
# Random Walk Sampling & Anonymization
# ---------------------------------------------------------------------------

@torch.no_grad()
def anonymize_rws(rws: torch.Tensor, rev_walks: bool = True) -> torch.Tensor:
    """
    Replace absolute node IDs in walks with relative first-occurrence indices.

    Each node is assigned the position in the walk where it first appeared,
    making the representation permutation-invariant w.r.t. global node IDs.

    Args:
        rws       : (N, walk_length) — raw node-ID walks
        rev_walks : if True, temporarily flip walks before anonymizing
                    (input walks are assumed to be in reversed Target->Source order)

    Returns:
        Anonymized indices of the same shape as `rws`.
    """
    if rev_walks:
        rws = torch.flip(rws, dims=[-1])
    s, si = torch.sort(rws, dim=-1)
    su = torch.searchsorted(s, rws)
    c = torch.full_like(s, fill_value=s.shape[-1])
    rw_i = torch.arange(rws.shape[-1], device=rws.device)[None, :].expand_as(s)
    first = c.scatter_reduce_(-1, su, rw_i, reduce="amin")
    ret = first.gather(-1, su)
    if rev_walks:
        ret = torch.flip(ret, dims=[-1])
    return ret


@torch.no_grad()
def get_random_walk_batch(
    adj: SparseTensor,
    x: torch.Tensor,
    start_nodes: torch.Tensor,
    walk_length: int,
    num_walks: int,
    recurrent_steps: int = 1,
    p: float = 1.0,
    q: float = 1.0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Sample random walks from `start_nodes`, anonymize them, and fetch node features.

    Each source node gets `num_walks` Node2Vec-style walks of length `walk_length`.
    Walks are reversed (Target->Source) before anonymization, per HeART convention.

    With `recurrent_steps > 1`, the walk nodes from step k become the source nodes
    for step k+1, expanding the receptive field hierarchically.

    Args:
        adj             : SparseTensor adjacency (COO format)
        x               : (num_nodes, emb_dim) node feature matrix
        start_nodes     : (B,) source node indices
        walk_length     : number of nodes per walk (including start)
        num_walks       : number of walks per source node
        recurrent_steps : depth of hierarchical walk recurrence
        p               : Node2Vec return parameter
        q               : Node2Vec in-out parameter

    Returns:
        batch_features    : (B, num_walks * walk_length, emb_dim)
        anon_indices_list : list of (B, num_walks * walk_length) anonymized index tensors,
                           one per recurrent step
    """
    row, col, _ = adj.coo()
    row = row.to(x.device)
    col = col.to(x.device)

    current_sources = start_nodes
    rws_list = []

    for _ in range(recurrent_steps):
        num_sources = current_sources.size(0)

        # Repeat each source node num_walks times -> (num_sources * num_walks,)
        sources_repeated = current_sources.repeat_interleave(num_walks)

        # Sample walks: cluster_random_walk returns (num_sources * num_walks, walk_length)
        walks = cluster_random_walk(
            row, col, sources_repeated,
            walk_length - 1,   # steps; cluster_random_walk prepends the start node
            p=p, q=q,
            num_nodes=adj.size(0)
        )

        # Reshape -> (num_sources, num_walks, walk_length) -> (num_sources, num_walks * walk_length)
        walks = walks.view(num_sources, num_walks, walk_length)
        rws = walks.flatten(1, 2)

        # Reverse walks to Target->Source order (HeART convention)
        rws = torch.flip(rws, dims=[-1])
        rws_list.append(rws)

        # Next recurrent step sources = all nodes visited in current walks
        if recurrent_steps > 1:
            current_sources = rws.reshape(-1)
            import pdb; pdb.set_trace()

    # Anonymize each step's walks independently
    anon_indices_list = [anonymize_rws(rws, rev_walks=True) for rws in rws_list]

    # Fetch features from the final recurrent step's walks
    final_raw_indices = rws_list[-1]
    batch_features = x[final_raw_indices]
    return batch_features, anon_indices_list


class Transformer(nn.Module):
    def __init__(self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads,
                 num_walks, seq_len, attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0,
                 resid_dropout_p: float = 0.0, drop_path_p: float = 0.0,
                 config: MupConfig = MupConfig()):
        super().__init__()
        self.mup_cfg = config
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads

        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads,
                num_walks * seq_len, n_layer=num_layers,
                attn_dropout_p=attn_dropout_p, ffn_dropout_p=ffn_dropout_p,
                resid_dropout_p=resid_dropout_p, drop_path_p=drop_path_p,
                config=config
            )
            for _ in range(num_layers)
        ])

        # Input projection
        self.emb = nn.Linear(emb_dim, hidden_dim, bias=False)
        # Final RMSNorm scale
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))

        nn.init.normal_(self.emb.weight, mean=0.0, std=config.init_std)

    def forward(self, x, anon_indices, source_nodes=None):
        """
        Args:
            x            : (batch_size, ctx_len, emb_dim)  — walk node features
            anon_indices : list of anonymized position tensors, one per recurrent step
            source_nodes : unused, kept for API compatibility

        Returns:
            node_emb : (batch_size, hidden_dim) — L2-normalised embedding of each source node
        """
        batch_size, ctx_len, _ = x.shape
        x = self.emb(x)

        for depth, idx in enumerate(reversed(anon_indices)):
            for layer in self.layers:
                x = layer(x, idx)
            # Between recurrent steps: pool to last token, norm, then reshape
            if depth < len(anon_indices) - 1:
                x = x[:, -1, :]
                x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
                # n = source nodes, t = num_walks * walk_length, z = hidden_dim
                x = rearrange(x, '(n t) z -> n t z', t=ctx_len)

        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        # Return L2-normalised embedding of the last (source) token
        x = F.normalize(x[:, -1, :], dim=-1)
        return x


# ---------------------------------------------------------------------------
# MLP Link Predictor
# ---------------------------------------------------------------------------

class LinkPredictorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, dropout=0.0):
        super().__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_dim, out_dim))
        else:
            self.lins.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.lins.append(nn.Linear(hidden_dim, out_dim))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, h1, h2):
        """Score a pair of node embeddings via element-wise product + MLP."""
        x = h1 * h2
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


# ---------------------------------------------------------------------------
# Loss Function
# ---------------------------------------------------------------------------

def binary_cross_entropy_loss(pos_scores, neg_scores):
    """
    BCE loss over positive and negative edge scores.

    Args:
        pos_scores : (B,)      — scores for positive edges
        neg_scores : (B, K)    — scores for K negative edges per positive

    Returns:
        scalar loss
    """
    pos_loss = -torch.log(pos_scores + 1e-15).mean()
    neg_loss = -torch.log(1 - neg_scores + 1e-15).mean()
    return pos_loss + neg_loss


