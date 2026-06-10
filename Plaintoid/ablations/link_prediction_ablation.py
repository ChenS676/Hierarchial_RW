"""
HeART Link Prediction with Recurrent Random Walks.

Script layout:
  1. Imports
  2. Config / argument parsing
  3. Model definitions  (MupConfig, RoPE, TransformerLayer, Transformer, LinkPredictorMLP)
  4. Graph utilities    (load_graph_arxiv23, downsample_edges, sample_negative_edges,
                         get_random_walk_batch, anonymize_rws)
  5. Training utilities (binary_cross_entropy_loss, trapezoidal_lr_schedule, get_metric_score)
  6. Evaluation         (test_edge, evaluate_link_prediction, evaluate_and_log)
  7. Setup functions    (setup_data, setup_model_and_optimizer, setup_wandb)
  8. main()
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os
os.environ["WANDB_MODE"] = "online"

import argparse
import dataclasses
import math
import pickle
import time
from itertools import chain
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.transforms as T
import tqdm.auto as tqdm
import wandb
from einops import rearrange
from muon import SingleDeviceMuonWithAuxAdam
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader, TensorDataset
from torch_cluster import random_walk as cluster_random_walk
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import coalesce, degree, remove_self_loops, to_undirected
from torch_sparse import SparseTensor

from evalutors import evaluate_auc, evaluate_hits, evaluate_mrr

torch.serialization.add_safe_globals([
    torch_geometric.data.data.DataEdgeAttr,
    torch_geometric.data.data.DataTensorAttr,
    torch_geometric.data.storage.GlobalStorage,
])


# =============================================================================
# 2. Config / argument parsing
# =============================================================================

def str_to_bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ('true', '1', 't', 'yes', 'y'):
        return True
    if val.lower() in ('false', '0', 'f', 'no', 'n'):
        return False
    raise argparse.ArgumentTypeError(f'Boolean value expected, got {val}')


def get_config() -> dict:
    parser = argparse.ArgumentParser(description="HeART Link Prediction")

    run = parser.add_argument_group('Run')
    run.add_argument('--seeds', type=int, nargs='+', default=[2025])
    run.add_argument('--num_epochs', type=int, default=60)
    run.add_argument('--global_batch_size', type=int, default=256)
    run.add_argument('--patience', type=int, default=20)
    run.add_argument('--train_edge_downsample_ratio', type=float, default=1.0)
    run.add_argument('--eval_metric', type=str, default='MRR')

    data = parser.add_argument_group('Data')
    data.add_argument('--data_root', type=str, default='data/Cora')
    data.add_argument('--data_name', type=str, default='Cora')
    data.add_argument('--val_split_ratio', type=float, default=0.15)
    data.add_argument('--test_split_ratio', type=float, default=0.05)
    data.add_argument('--deepwalk_pkl_path', type=str, default=None)
    data.add_argument('--use_fixed_splits', type=str_to_bool, default=False)
    data.add_argument('--split_dir', type=str, default='data/Cora/fixed_splits')
    data.add_argument('--use_laplacian_pe', type=str_to_bool, default=False)
    data.add_argument('--laplacian_pe_path', type=str, default='data/Cora/laplacian_pe.pt')
    data.add_argument('--laplacian_edge_subsampling_ratio', type=float, default=1.0)

    arch = parser.add_argument_group('Model Architecture')
    arch.add_argument('--num_layers', type=int, default=1)
    arch.add_argument('--hidden_dim', type=int, default=128)
    arch.add_argument('--intermediate_dim_multiplier', type=int, default=4)
    arch.add_argument('--num_heads', type=int, default=16)
    arch.add_argument('--recurrent_steps', type=int, default=1)
    arch.add_argument('--mup_init_std', type=float, default=0.01)
    arch.add_argument('--mup_width_multiplier', type=float, default=2.0)

    walk = parser.add_argument_group('Random Walk')
    walk.add_argument('--walk_length', type=int, default=8)
    walk.add_argument('--num_walks', type=int, default=8)
    walk.add_argument('--node2vec_p', type=float, default=1.0)
    walk.add_argument('--node2vec_q', type=float, default=1.0)

    opt = parser.add_argument_group('Optimizer')
    opt.add_argument('--muon_min_lr', type=float, default=1e-4)
    opt.add_argument('--muon_max_lr', type=float, default=1e-3)
    opt.add_argument('--adam_max_lr', type=float, default=1e-4)
    opt.add_argument('--adam_min_lr', type=float, default=0.0)
    opt.add_argument('--grad_clip_norm', type=float, default=0.1)

    mlp = parser.add_argument_group('MLP Link Predictor')
    mlp.add_argument('--use_mlp', type=str_to_bool, default=True)
    mlp.add_argument('--mlp_num_layers', type=int, default=3)
    mlp.add_argument('--mlp_lr', type=float, default=1e-3)
    mlp.add_argument('--mlp_dropout', type=float, default=0.1)

    reg = parser.add_argument_group('Regularization')
    reg.add_argument('--attn_dropout', type=float, default=0.1)
    reg.add_argument('--ffn_dropout', type=float, default=0.1)
    reg.add_argument('--resid_dropout', type=float, default=0.1)
    reg.add_argument('--drop_path', type=float, default=0.05)

    misc = parser.add_argument_group('Misc')
    misc.add_argument('--neg_sample_ratio', type=int, default=1)
    misc.add_argument('--hits_k', type=int, nargs='+', default=[1, 10, 50, 100])
    misc.add_argument('--use_deepwalk_embeds', type=str_to_bool, default=False)

    wb = parser.add_argument_group('Weights & Biases')
    wb.add_argument('--wb_entity', type=str, default='graph-diffusion-model-link-prediction')
    wb.add_argument('--wb_project', type=str, default='ani-cwue-link-prediction-final')
    wb.add_argument('--run_tag', type=str, default='')
    wb.add_argument('--exp_name', type=str, default='')

    args = parser.parse_args()
    config = vars(args)
    config['intermediate_dim'] = config['hidden_dim'] * config['intermediate_dim_multiplier']
    del config['intermediate_dim_multiplier']
    return config


# =============================================================================
# 3. Model definitions
# =============================================================================

@dataclasses.dataclass(frozen=True)
class MupConfig:
    init_std: float = 0.01
    mup_width_multiplier: float = 2.0


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
            self.base ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(max_seq_len, dtype=self.theta.dtype, device=self.theta.device)
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        return x_out.flatten(3).type_as(x)


class TransformerLayer(nn.Module):
    def __init__(
        self, hidden_dim, intermediate_dim, num_heads, seq_len, n_layer,
        attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0, drop_path_p: float = 0.0,
        config: MupConfig = MupConfig(),
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.attn_dropout_p = attn_dropout_p
        self.ffn_dropout_p = ffn_dropout_p
        self.resid_dropout_p = resid_dropout_p
        self.drop_path_p = drop_path_p

        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self.o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.up = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.input_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.attn_norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.rope = RotaryPositionalEmbeddings(dim=(hidden_dim // num_heads), max_seq_len=seq_len)

        std_inner = config.init_std / math.sqrt(2 * n_layer * config.mup_width_multiplier)
        std_outer = config.init_std / math.sqrt(config.mup_width_multiplier)
        for w in (self.up, self.gate):
            nn.init.normal_(w.weight, mean=0.0, std=std_inner)
        for w in (self.down, self.qkv, self.o):
            nn.init.normal_(w.weight, mean=0.0, std=std_outer)

    def forward(self, x, offset=None):
        attnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.attn_norm_weight
        q, k, v = self.qkv(attnx).chunk(3, dim=-1)
        q, k, v = [rearrange(t, 'n t (h d) -> n t h d', h=self.num_heads) for t in (q, k, v)]
        q = self.rope(q, input_pos=offset)
        k = self.rope(k, input_pos=offset)
        q, k, v = [rearrange(t, 'n t h d -> n h t d') for t in (q, k, v)]
        o_walks = F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=1.0 / k.shape[-1])
        attn_out = self.o(rearrange(o_walks, 'n h t d -> n t (h d)', h=self.num_heads))
        if self.resid_dropout_p > 0:
            attn_out = F.dropout(attn_out, p=self.resid_dropout_p, training=self.training)
        x = x + self._drop_path(attn_out)

        ffnx = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.input_norm_weight
        ffn_out = self.down(F.silu(self.up(ffnx)) * self.gate(ffnx))
        if self.ffn_dropout_p > 0:
            ffn_out = F.dropout(ffn_out, p=self.ffn_dropout_p, training=self.training)
        return x + self._drop_path(ffn_out)

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_path_p <= 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_path_p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        binary_tensor = torch.floor(keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device))
        return x.div(keep_prob) * binary_tensor


class Transformer(nn.Module):
    def __init__(
        self, emb_dim, num_layers, hidden_dim, intermediate_dim, num_heads, num_walks, seq_len,
        attn_dropout_p: float = 0.0, ffn_dropout_p: float = 0.0,
        resid_dropout_p: float = 0.0, drop_path_p: float = 0.0,
        config: MupConfig = MupConfig(),
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.emb = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.norm_weight = nn.Parameter(torch.ones(hidden_dim))
        self.layers = nn.ModuleList([
            TransformerLayer(
                hidden_dim, intermediate_dim, num_heads, num_walks * seq_len,
                n_layer=num_layers, attn_dropout_p=attn_dropout_p,
                ffn_dropout_p=ffn_dropout_p, resid_dropout_p=resid_dropout_p,
                drop_path_p=drop_path_p, config=config,
            )
            for _ in range(num_layers)
        ])
        nn.init.normal_(self.emb.weight, mean=0.0, std=config.init_std)

    def forward(self, x, anon_indices, source_nodes=None):
        _, ctx_len, _ = x.shape
        x = self.emb(x)
        for depth, idx in enumerate(reversed(anon_indices)):
            for layer in self.layers:
                x = layer(x, idx)
            if depth < len(anon_indices) - 1:
                x = F.rms_norm(x[:, -1, :], [self.hidden_dim], eps=1e-5) * self.norm_weight
                x = rearrange(x, '(n t) z -> n t z', t=ctx_len)
        x = F.rms_norm(x, [self.hidden_dim], eps=1e-5) * self.norm_weight
        return F.normalize(x[:, -1, :], dim=-1)


class LinkPredictorMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=1, num_layers=3, dropout=0.0):
        super().__init__()
        if num_layers == 1:
            self.lins = nn.ModuleList([nn.Linear(in_dim, out_dim)])
        else:
            self.lins = nn.ModuleList(
                [nn.Linear(in_dim, hidden_dim)]
                + [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)]
                + [nn.Linear(hidden_dim, out_dim)]
            )
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, h1, h2):
        x = h1 * h2
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return torch.sigmoid(self.lins[-1](x))


# =============================================================================
# 4. Graph / walk utilities
# =============================================================================

def load_graph_arxiv23(data_root) -> Data:
    return torch.load(data_root + 'arxiv_2023/graph.pt', weights_only=False)


def downsample_edges(edge_index, ratio=0.5, seed=42):
    torch.manual_seed(seed)
    num_edges = edge_index.size(1)
    perm = torch.randperm(num_edges)[:int(num_edges * ratio)]
    return edge_index[:, perm]


def sample_negative_edges(pos_edge_index, num_nodes, num_neg_samples, device):
    from torch_geometric.utils import negative_sampling
    return negative_sampling(
        edge_index=pos_edge_index,
        num_nodes=num_nodes,
        num_neg_samples=num_neg_samples,
        method='sparse',
    ).to(device)


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
    row, col, _ = adj.coo()
    row, col = row.to(x.device), col.to(x.device)
    current_sources = start_nodes
    rws_list = []
    for _ in range(recurrent_steps):
        num_sources = current_sources.size(0)
        sources_repeated = current_sources.repeat_interleave(num_walks)
        torch.seed()  
        walks = cluster_random_walk(
            row, col, sources_repeated, walk_length - 1,
            p=p, q=q, num_nodes=adj.size(0),
        )
        walks = walks.view(num_sources, num_walks, walk_length)
        rws = torch.flip(walks.flatten(1, 2), dims=[-1])
        rws_list.append(rws)
        if recurrent_steps > 1:
            current_sources = rws.reshape(-1)
    anon_indices_list = [anonymize_rws(rws, rev_walks=True) for rws in rws_list]
    batch_features = x[rws_list[-1]]
    return batch_features, anon_indices_list


@torch.no_grad()
def anonymize_rws(rws, rev_walks=True):
    if rev_walks:
        rws = torch.flip(rws, dims=[-1])
    s, _ = torch.sort(rws, dim=-1)
    su = torch.searchsorted(s, rws)
    c = torch.full_like(s, fill_value=s.shape[-1])
    rw_i = torch.arange(rws.shape[-1], device=rws.device)[None, :].expand_as(s)
    first = c.scatter_reduce_(-1, su, rw_i, reduce="amin")
    ret = first.gather(-1, su)
    if rev_walks:
        ret = torch.flip(ret, dims=[-1])
    return ret


# =============================================================================
# 5. Training utilities
# =============================================================================

def binary_cross_entropy_loss(pos_scores, neg_scores):
    return -torch.log(pos_scores + 1e-15).mean() - torch.log(1 - neg_scores + 1e-15).mean()


def trapezoidal_lr_schedule(global_batch_idx, max_lr, min_lr, warmup, cool, total_batches):
    if global_batch_idx <= warmup:
        return (global_batch_idx / warmup) * (max_lr - min_lr) + min_lr
    if global_batch_idx <= total_batches - cool:
        return max_lr
    return ((total_batches - global_batch_idx) / cool) * (max_lr - min_lr) + min_lr


def get_metric_score(evaluator_hit, evaluator_mrr, pos_val_pred, neg_val_pred, k_list: list):
    result = {}
    for K, v in evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list).items():
        result[K] = v
    result['MRR'] = evaluate_mrr(
        evaluator_mrr, pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1)
    )['MRR']
    val_pred = torch.cat([pos_val_pred, neg_val_pred])
    val_true = torch.cat([
        torch.ones(pos_val_pred.size(0), dtype=int),
        torch.zeros(neg_val_pred.size(0), dtype=int),
    ])
    auc_result = evaluate_auc(val_pred, val_true)
    result['AUC'] = auc_result['AUC']
    result['AP'] = auc_result['AP']
    return result


# =============================================================================
# 6. Evaluation
# =============================================================================

@torch.no_grad()
def test_edge(model, link_predictor, adj, X, config, edge_index, batch_size):
    all_scores = []
    for perm in DataLoader(range(edge_index.t().size(0)), batch_size=batch_size):
        batch_edge_index = edge_index.t()[perm].t()
        nodes = batch_edge_index.unique()
        batch, anon_indices = get_random_walk_batch(
            adj, X, nodes,
            walk_length=config['walk_length'],
            num_walks=config['num_walks'],
            recurrent_steps=config['recurrent_steps'],
            p=config['node2vec_p'],
            q=config['node2vec_q'],
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            embeddings = model(batch, anon_indices)
            node_to_idx = {n.item(): i for i, n in enumerate(nodes)}
            u = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[0]]]
            v = embeddings[[node_to_idx[n.item()] for n in batch_edge_index[1]]]
            if config['use_mlp']:
                scores = link_predictor(u, v)
            else:
                scores = torch.sigmoid((u * v).sum(dim=-1))
        all_scores.append(scores.cpu())
    return torch.cat(all_scores, dim=0).float()


@torch.no_grad()
def evaluate_link_prediction(
    model, link_predictor, pos_edge_index, neg_edge_index,
    adj, X, config, evaluator_hit, evaluator_mrr, device, eval_batch_size=512,
):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()
    pos_scores = test_edge(model, link_predictor, adj, X, config, pos_edge_index, eval_batch_size)
    neg_scores = test_edge(model, link_predictor, adj, X, config, neg_edge_index, eval_batch_size)
    return get_metric_score(
        evaluator_hit, evaluator_mrr,
        pos_val_pred=torch.flatten(pos_scores),
        neg_val_pred=torch.flatten(neg_scores),
        k_list=config.get('hits_k', [1, 10, 50, 100]),
    )


@torch.no_grad()
def evaluate_and_log(
    model, link_predictor, adj, X, config,
    evaluator_hit, evaluator_mrr, device,
    train_pos_edge_index, train_neg_edge_index,
    val_pos_edge_index, val_neg_edge_index,
    test_pos_edge_index, test_neg_edge_index,
    epoch, best_val_eval_metric, best_val_metrics,
    best_test_metrics, best_val_epoch,
    epochs_without_improvement, BEST_MODEL_PATH,
    global_batch_idx=None,
):
    model.eval()
    if config['use_mlp']:
        link_predictor.eval()

    eval_bs = config['global_batch_size']
    train_results = evaluate_link_prediction(
        model, link_predictor, train_pos_edge_index, train_neg_edge_index,
        adj, X, config, evaluator_hit, evaluator_mrr, device, eval_bs,
    )
    val_results = evaluate_link_prediction(
        model, link_predictor, val_pos_edge_index, val_neg_edge_index,
        adj, X, config, evaluator_hit, evaluator_mrr, device, eval_bs,
    )
    test_results = evaluate_link_prediction(
        model, link_predictor, test_pos_edge_index, test_neg_edge_index,
        adj, X, config, evaluator_hit, evaluator_mrr, device, eval_bs,
    )

    # Log per-epoch metrics
    wandb.log(
        {**{f'train/{k}': v for k, v in train_results.items()},
         **{f'val/{k}': v for k, v in val_results.items()},
         **{f'test/{k}': v for k, v in test_results.items()}},
        step=global_batch_idx,
    )

    # Console output — written through tqdm so the progress bar isn't corrupted
    def fmt(d): return ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
    tqdm.tqdm.write(f"Ep {epoch + 1:>4} | Train [{fmt(train_results)}] | Val [{fmt(val_results)}] | Test [{fmt(test_results)}]")

    # Update running best for every metric
    for k, v in val_results.items():
        best_val_metrics[k] = max(v, best_val_metrics.get(k, 0.0))
    for k, v in test_results.items():
        best_test_metrics[k] = max(v, best_test_metrics.get(k, 0.0))

    # Keep wandb summary in sync with current best values
    if wandb.run is not None:
        wandb.run.summary.update(
            {f'best_val_{k}': best_val_metrics[k] for k in val_results}
            | {f'best_test_{k}': best_test_metrics[k] for k in test_results}
        )

    # Model checkpoint & early stopping (based on primary eval_metric)
    val_metric = val_results[config['eval_metric']]
    early_stop = False

    if val_metric > best_val_eval_metric:
        best_val_eval_metric = val_metric
        best_val_epoch = epoch + 1
        if wandb.run is not None:
            wandb.run.summary['best_val_epoch'] = best_val_epoch

        save_dict = {'model_state_dict': model.state_dict(), 'config': config}
        if config['use_mlp']:
            save_dict['link_predictor_state_dict'] = link_predictor.state_dict()
        torch.save(save_dict, BEST_MODEL_PATH)
        tqdm.tqdm.write(f"  ✓ Best model @ epoch {epoch + 1} | val {config['eval_metric']}: {val_metric:.4f}")
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= config['patience']:
        tqdm.tqdm.write(f"  Early stopping after {epoch + 1} epochs ({config['patience']} evals without improvement).")
        early_stop = True

    return (
        best_val_eval_metric, best_val_metrics, best_test_metrics,
        best_val_epoch, epochs_without_improvement, early_stop,
    )


# =============================================================================
# 7. Setup functions
# =============================================================================

def setup_data(config: dict, device: torch.device):
    """Load graph, apply feature augmentations, build train/val/test splits and adjacency matrix."""

    # --- Load base graph ---
    if config['data_name'] in ['Cora', 'PubMed', 'CiteSeer']:
        dataset = Planetoid(root=config['data_root'], name=config['data_name'])
        data = dataset[0].to(device)
    elif config['data_name'].startswith('TAPE'):
        data = load_graph_arxiv23(config['data_root']).to(device)
    else:
        raise ValueError(f"Unknown dataset: {config['data_name']}")

    # --- Optional: DeepWalk embedding concatenation ---
    if config['use_deepwalk_embeds']:
        path = config['deepwalk_pkl_path']
        if not os.path.exists(path):
            raise FileNotFoundError(f"DeepWalk file not found: {path}")
        with open(path, 'rb') as f:
            dw = pickle.load(f)['data'].to(device)
        data.x = torch.cat([data.x, dw], dim=1)
        print(f"DeepWalk concatenated → feat dim: {data.x.shape[1]}")

    # --- Edge cleanup ---
    if data.is_directed():
        print("Directed graph → converting to undirected.")
        data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    # --- Graph statistics (single line) ---
    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    print(f"[{config['data_name']}] nodes={data.num_nodes} edges={data.edge_index.size(1)} "
          f"avg_deg={deg.mean().item():.2f} max_deg={int(deg.max())} isolated={int((deg==0).sum())}")

    # --- Optional: remove isolated nodes (TAPE datasets) ---
    nodes_before = data.num_nodes
    if config['data_name'].startswith('TAPE'):
        data = T.RemoveIsolatedNodes()(data)
    print(f"Isolated-node removal: {nodes_before} → {data.num_nodes} nodes")

    # --- Optional: Laplacian PE concatenation ---
    if config['use_laplacian_pe']:
        path = config['laplacian_pe_path']
        if not os.path.exists(path):
            raise FileNotFoundError(f"Laplacian PE file not found: {path}")
        lap_pe = torch.load(path, map_location=device, weights_only=False)
        if lap_pe.size(0) != data.x.size(0):
            print(f"WARNING: PE size {lap_pe.size(0)} != graph size {data.x.size(0)} — truncating.")
            lap_pe = lap_pe[:data.x.size(0)]
        data.x = torch.cat([data.x, lap_pe], dim=1)
        print(f"Laplacian PE concatenated → feat dim: {data.x.shape[1]}")

    # --- Build splits ---
    if config['use_fixed_splits']:
        split_path = os.path.join(config['split_dir'], f"{config['data_name']}_fixed_split.pt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Fixed split not found: {split_path}")
        sd = torch.load(split_path, map_location=device, weights_only=False)

        def make_data(key):
            return Data(
                x=data.x,
                edge_index=sd[key]['edge_index'].to(device),
                edge_label_index=sd[key]['edge_label_index'].to(device),
                edge_label=sd[key]['edge_label'].to(device),
            )
        train_data = make_data('train')
        val_data   = make_data('val')
        test_data  = make_data('test')
        full_train_pos_edge_index = train_data.edge_index
        print(f"Fixed split loaded — train edges: {full_train_pos_edge_index.size(1)}")
    else:
        print("Generating random splits on the fly...")
        transform = T.RandomLinkSplit(
            num_val=config['val_split_ratio'],
            num_test=config['test_split_ratio'],
            is_undirected=True,
            add_negative_train_samples=False,
        )
        train_data, val_data, test_data = transform(data)
        full_train_pos_edge_index = train_data.edge_index.to(device)

    def pos_neg(d):
        return (
            d.edge_label_index[:, d.edge_label == 1].to(device),
            d.edge_label_index[:, d.edge_label == 0].to(device),
        )
    val_pos_edge_index,  val_neg_edge_index  = pos_neg(val_data)
    test_pos_edge_index, test_neg_edge_index = pos_neg(test_data)

    # --- Adjacency matrix for random walks ---
    edge_weight = torch.ones(full_train_pos_edge_index.size(1), device=device)
    adj = SparseTensor.from_edge_index(
        full_train_pos_edge_index, edge_weight, [data.num_nodes, data.num_nodes]
    )

    return (
        data, train_data,
        full_train_pos_edge_index,
        val_pos_edge_index, val_neg_edge_index,
        test_pos_edge_index, test_neg_edge_index,
        adj,
    )


def setup_model_and_optimizer(config: dict, device: torch.device):
    """Instantiate Transformer + optional MLP predictor + Muon optimizer."""
    torch.manual_seed(config['seed'])
    mup_cfg = MupConfig(init_std=config['mup_init_std'], mup_width_multiplier=config['mup_width_multiplier'])

    model = Transformer(
        emb_dim=config['emb_dim'],
        num_layers=config['num_layers'],
        hidden_dim=config['hidden_dim'],
        intermediate_dim=config['intermediate_dim'],
        num_heads=config['num_heads'],
        seq_len=config['walk_length'],
        num_walks=config['num_walks'],
        attn_dropout_p=config['attn_dropout'],
        ffn_dropout_p=config['ffn_dropout'],
        resid_dropout_p=config['resid_dropout'],
        drop_path_p=config['drop_path'],
        config=mup_cfg,
    ).to(device)
    print(f"Transformer params: {sum(p.numel() for p in model.parameters()):,}")

    link_predictor = None
    if config['use_mlp']:
        link_predictor = LinkPredictorMLP(
            in_dim=config['hidden_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['mlp_num_layers'],
            dropout=config['mlp_dropout'],
        ).to(device)
        print("Link predictor: MLP")
    else:
        print("Link predictor: dot-product")

    # Param groups: Muon for weight matrices, Adam for gains/biases + MLP
    hidden_weights = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]
    hidden_gains_biases = [p for p in model.parameters() if p.ndim < 2 and p.requires_grad]
    if config['recurrent_steps'] <= 1:
        hidden_gains_biases = [p for p in hidden_gains_biases if p is not model.norm_weight]

    param_groups = [
        dict(params=hidden_weights,     use_muon=True,  lr=0.04,              weight_decay=0.01),
        dict(params=hidden_gains_biases, use_muon=False, lr=5e-5, betas=(0.9, 0.95), weight_decay=0.0),
    ]
    if config['use_mlp']:
        param_groups.append(
            dict(params=link_predictor.parameters(), use_muon=False,
                 lr=config['mlp_lr'], betas=(0.9, 0.95), weight_decay=0.0)
        )
    optimizer = SingleDeviceMuonWithAuxAdam(param_groups)

    return model, link_predictor, optimizer


def setup_wandb(config: dict) -> str:
    """Initialise W&B run and return the run ID."""
    if config['use_deepwalk_embeds']:
        pe_tag = 'pos_encoding_deepwalk'
    elif config['use_laplacian_pe']:
        pe_tag = 'pos_encoding_laplacian'
    else:
        pe_tag = 'pos_encoding_False'

    tag_prefix = f"{config['exp_name']}_" if config['exp_name'] else ''
    run_name = (
        f"{tag_prefix}{pe_tag}"
        f"_recurrent_steps_{config['recurrent_steps']}"
        f"_bs_{config['global_batch_size']}"
        f"_muon-max-lr_{config['muon_max_lr']}"
        f"_adam_max_lr_{config['adam_max_lr']}"
        f"_nwalks_{config['num_walks']}"
        f"_wl_{config['walk_length']}"
        f"_seed_{config['seed']}"
    )
    if config['train_edge_downsample_ratio'] < 1.0:
        run_name += f"_edge_dws_{config['train_edge_downsample_ratio']}"

    predictor_tag = 'MLP' if config['use_mlp'] else 'Dot Product'
    project = (
        f"{config['data_name']}_{config['run_tag']}"
        if config['run_tag']
        else f"{config['data_name']}_rw-cwue_latest_dec_{config['seed']}"
    )
    wandb.init(
        entity=config['wb_entity'],
        project=project,
        group=f"{config['data_name']} Random Walk Link Prediction {predictor_tag}",
        name=run_name,
        config=config,
    )
    return wandb.run.id


def save_code_to_wandb(run_id: str, config: dict):
    """Upload the training script and config YAML to W&B."""
    import yaml

    artifact = wandb.Artifact(
        name=f"code_{run_id}",
        type="code",
        metadata=config,
    )

    artifact.add_file(os.path.abspath(__file__), name="training_script.py")

    config_path = f"/tmp/config_{run_id}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    artifact.add_file(config_path, name="config.yaml")

    wandb.log_artifact(artifact)
    print(f"Code artifact uploaded to W&B.")


# =============================================================================
# 8. run_one_seed() + main()
# =============================================================================

def run_one_seed(config: dict, device: torch.device):
    """Run full training for a single seed. Returns (run_id, best_val_metrics, best_test_metrics, best_val_epoch)."""

    # ---------- Data ----------
    (
        data, train_data,
        full_train_pos_edge_index,
        val_pos_edge_index,  val_neg_edge_index,
        test_pos_edge_index, test_neg_edge_index,
        adj,
    ) = setup_data(config, device)

    config['emb_dim'] = data.x.size(1)
    print(f"emb_dim: {config['emb_dim']}")

    # ---------- W&B ----------
    run_id = setup_wandb(config)
    save_code_to_wandb(run_id, config)
    save_dir = os.path.join(config['data_name'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(save_dir, f"best_model_{run_id}.pth")

    # ---------- Model & optimizer ----------
    model, link_predictor, optimizer = setup_model_and_optimizer(config, device)

    # ---------- Training bookkeeping ----------
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    X = train_data.x.to(device).bfloat16()

    global_batch_size = config['global_batch_size']
    full_samples_per_epoch = full_train_pos_edge_index.size(1)
    downsampling = config['train_edge_downsample_ratio'] < 1.0

    if downsampling:
        samples_per_epoch = int(full_samples_per_epoch * config['train_edge_downsample_ratio'])
        print(f"Edge downsampling: {samples_per_epoch}/{full_samples_per_epoch} edges/epoch.")
        train_loader = None  # rebuilt each epoch
    else:
        samples_per_epoch = full_samples_per_epoch
        print(f"All {samples_per_epoch} positive edges per epoch.")
        train_loader = DataLoader(
            TensorDataset(full_train_pos_edge_index.t()),
            batch_size=global_batch_size, shuffle=True,
        )

    batches_per_epoch = math.ceil(samples_per_epoch / global_batch_size)
    total_batches = batches_per_epoch * config['num_epochs']
    warmup = total_batches // 10
    cool  = total_batches - warmup

    # Gradient accumulation (only needed for high-recurrence + small batch)
    accumulation_steps = (256 // global_batch_size) if config['recurrent_steps'] > 1 else 1
    print(f"bs={global_batch_size} | batches/ep={batches_per_epoch} | total={total_batches} | grad_accum={accumulation_steps}")

    best_val_eval_metric   = 0.0
    best_val_epoch         = 0
    epochs_without_improvement = 0
    best_val_metrics  = {}
    best_test_metrics = {}
    epochs_eval_steps = 5
    running_loss      = 0.0
    global_batch_idx  = 0

    # Snapshot initial weights for per-step parameter drift tracking
    initial_param_snapshot = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    prev_l2_param_drift = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm.tqdm(total=total_batches)

    # ==========================================================================
    # Training loop
    # ==========================================================================
    for epoch in range(config['num_epochs']):
        model.train()
        if config['use_mlp']:
            link_predictor.train()

        # Per-epoch downsampled loader
        if downsampling:
            epoch_train_pos = downsample_edges(
                full_train_pos_edge_index,
                ratio=config['train_edge_downsample_ratio'],
                seed=config['seed'] + epoch,
            )
            train_loader = DataLoader(
                TensorDataset(epoch_train_pos.t()),
                batch_size=global_batch_size, shuffle=True,
            )
        else:
            epoch_train_pos = full_train_pos_edge_index

        # Sample train edges for evaluation (matched to val-neg size)
        torch.manual_seed(config['seed'] + epoch)
        n_eval_train = min(val_neg_edge_index.size(1), epoch_train_pos.size(1))
        perm = torch.randperm(epoch_train_pos.size(1), device=device)[:n_eval_train]
        eval_train_pos_edge_index = epoch_train_pos[:, perm]

        # ------ Batch loop ------
        for batch_idx, (batch_pos_t,) in enumerate(train_loader):
            global_batch_idx = batch_idx + epoch * batches_per_epoch

            # LR schedule (Muon only)
            muon_lr = trapezoidal_lr_schedule(
                global_batch_idx, config['muon_max_lr'], config['muon_min_lr'],
                warmup, cool, total_batches,
            )
            for pg in optimizer.param_groups:
                if pg.get('use_muon', False):
                    pg['lr'] = muon_lr

            # Negative sampling
            batch_pos_edges = batch_pos_t.t().to(device)
            local_bs = batch_pos_edges.size(1)
            batch_neg_edges = sample_negative_edges(
                full_train_pos_edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=int(local_bs * config['neg_sample_ratio']),
                device=device,
            )

            t0 = time.time()
            all_nodes = torch.cat([
                batch_pos_edges[0], batch_pos_edges[1],
                batch_neg_edges[0], batch_neg_edges[1],
            ]).unique()

            # Random walk embeddings
            batch, anon_indices = get_random_walk_batch(
                adj, X, all_nodes,
                walk_length=config['walk_length'],
                num_walks=config['num_walks'],
                recurrent_steps=config['recurrent_steps'],
                p=config['node2vec_p'],
                q=config['node2vec_q'],
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                embeddings = model(batch, anon_indices)
                node_to_idx = {n.item(): i for i, n in enumerate(all_nodes)}
                pos_u = embeddings[[node_to_idx[u.item()] for u in batch_pos_edges[0]]]
                pos_v = embeddings[[node_to_idx[v.item()] for v in batch_pos_edges[1]]]
                neg_u = embeddings[[node_to_idx[u.item()] for u in batch_neg_edges[0]]]
                neg_v = embeddings[[node_to_idx[v.item()] for v in batch_neg_edges[1]]]

                if config['use_mlp']:
                    pos_scores = link_predictor(pos_u, pos_v)
                    neg_scores = link_predictor(neg_u, neg_v).view(-1, config['neg_sample_ratio'])
                else:
                    pos_scores = torch.sigmoid((pos_u * pos_v).sum(dim=-1))
                    neg_scores = torch.sigmoid((neg_u * neg_v).sum(dim=-1)).view(-1, config['neg_sample_ratio'])

                loss = binary_cross_entropy_loss(pos_scores, neg_scores)
                running_loss += loss.item()
                loss = loss / accumulation_steps

            loss.backward()

            # Optimizer step (respects gradient accumulation)
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                mlp_grad_norm = 0.0
                if config['use_mlp']:
                    with torch.no_grad():
                        for p in link_predictor.parameters():
                            if p.grad is not None:
                                mlp_grad_norm += p.grad.data.norm(2).item() ** 2
                        mlp_grad_norm **= 0.5

                all_params = (
                    chain(model.parameters(), link_predictor.parameters())
                    if config['use_mlp'] else model.parameters()
                )
                torch.nn.utils.clip_grad_norm_(all_params, float(config['grad_clip_norm']))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # L2 norm of parameter drift from initialisation
                with torch.no_grad():
                    l2_param_drift = torch.sqrt(sum(
                        (param - initial_param_snapshot[name]).norm(2) ** 2
                        for name, param in model.named_parameters()
                        if name in initial_param_snapshot
                    )).item()
                l2_param_drift_delta = l2_param_drift - prev_l2_param_drift
                prev_l2_param_drift  = l2_param_drift

                avg_loss = running_loss / accumulation_steps
                wandb.log(
                    {"train/loss": avg_loss, "train/grad_norm": mlp_grad_norm, "train/lr": muon_lr,
                     "model/L2_grad": l2_param_drift, "model/L2_grad_delta": l2_param_drift_delta},
                    step=global_batch_idx,
                )
                running_loss = 0.0

            free, total = torch.cuda.mem_get_info(device)
            pbar.set_description(
                f"loss: {loss.item() * accumulation_steps:.4f}, "
                f"mem: {(total - free) / total:.2f}, "
                f"t: {time.time() - t0:.2f}s"
            )
            pbar.update(1)

        if 'loss' in locals() and torch.isnan(loss):
            break

        # ------ Evaluation ------
        is_eval_epoch = (
            epoch == 0
            or (epoch + 1) % epochs_eval_steps == 0
            or epoch == config['num_epochs'] - 1
        )
        if is_eval_epoch:
            (
                best_val_eval_metric, best_val_metrics, best_test_metrics,
                best_val_epoch, epochs_without_improvement, early_stop,
            ) = evaluate_and_log(
                model=model, link_predictor=link_predictor,
                adj=adj, X=X, config=config,
                evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr, device=device,
                train_pos_edge_index=eval_train_pos_edge_index,
                train_neg_edge_index=val_neg_edge_index,
                val_pos_edge_index=val_pos_edge_index,
                val_neg_edge_index=val_neg_edge_index,
                test_pos_edge_index=test_pos_edge_index,
                test_neg_edge_index=test_neg_edge_index,
                epoch=epoch,
                best_val_eval_metric=best_val_eval_metric,
                best_val_metrics=best_val_metrics,
                best_test_metrics=best_test_metrics,
                best_val_epoch=best_val_epoch,
                epochs_without_improvement=epochs_without_improvement,
                BEST_MODEL_PATH=BEST_MODEL_PATH,
                global_batch_idx=global_batch_idx,
            )
            if early_stop:
                break

    pbar.close()

    # ---------- Post-training ----------
    if torch.cuda.is_available():
        peak_gib    = torch.cuda.max_memory_allocated() / (1024**3)
        current_gib = torch.cuda.memory_allocated()    / (1024**3)
        print(f"Peak vRAM: {peak_gib:.3f} GiB | Current: {current_gib:.3f} GiB")

    wandb.run.summary.update(
        {f'best_val_{k}': v  for k, v in best_val_metrics.items()}
        | {f'best_test_{k}': v for k, v in best_test_metrics.items()}
        | {'best_val_epoch': best_val_epoch}
    )

    def fmt(d): return ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
    print(f"Best val {config['eval_metric']}: {best_val_eval_metric:.4f} @ epoch {best_val_epoch}")
    print(f"Best val  metrics: [{fmt(best_val_metrics)}]")
    print(f"Best test metrics: [{fmt(best_test_metrics)}]")

    wandb.finish()
    return run_id, best_val_metrics, best_test_metrics, best_val_epoch


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    config = get_config()
    seeds = config.pop('seeds')

    all_val_metrics  = []
    all_test_metrics = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running seed {seed}  ({seeds.index(seed)+1}/{len(seeds)})")
        print(f"{'='*60}")
        seed_config = {**config, 'seed': seed}

        run_id, val_m, test_m, best_ep = run_one_seed(seed_config, device)
        all_val_metrics.append(val_m)
        all_test_metrics.append(test_m)

        # Per-seed CSV
        os.makedirs(seed_config['data_name'], exist_ok=True)
        run_row = {
            'run_id': run_id,
            'exp_name': seed_config['exp_name'],
            'run_tag': seed_config['run_tag'],
            'seed': seed,
            'walk_length': seed_config['walk_length'],
            'num_walks': seed_config['num_walks'],
            'node2vec_p': seed_config['node2vec_p'],
            'node2vec_q': seed_config['node2vec_q'],
            'recurrent_steps': seed_config['recurrent_steps'],
            'best_val_epoch': best_ep,
            **{f'best_val_{k}': v for k, v in val_m.items()},
            **{f'best_test_{k}': v for k, v in test_m.items()},
        }
        run_csv = os.path.join(
            seed_config['data_name'],
            f"result_{seed_config['exp_name'] or 'run'}_{seed_config['data_name']}_seed{seed}_{run_id}.csv",
        )
        pd.DataFrame([run_row]).to_csv(run_csv, index=False)
        print(f"Per-seed CSV saved: {run_csv}")

    # Summary CSV: mean ± std across all seeds
    os.makedirs(config['data_name'], exist_ok=True)
    summary_row = {
        'exp_name': config['exp_name'],
        'run_tag': config['run_tag'],
        'seeds': str(seeds),
        'num_seeds': len(seeds),
        'walk_length': config['walk_length'],
        'num_walks': config['num_walks'],
        'node2vec_p': config['node2vec_p'],
        'node2vec_q': config['node2vec_q'],
        'recurrent_steps': config['recurrent_steps'],
    }
    for k in all_val_metrics[0]:
        vals = [m[k] for m in all_val_metrics]
        summary_row[f'best_val_{k}_mean'] = float(np.mean(vals))
        summary_row[f'best_val_{k}_std']  = float(np.std(vals))
    for k in all_test_metrics[0]:
        vals = [m[k] for m in all_test_metrics]
        summary_row[f'best_test_{k}_mean'] = float(np.mean(vals))
        summary_row[f'best_test_{k}_std']  = float(np.std(vals))

    seeds_str = '_'.join(map(str, seeds))
    summary_csv = os.path.join(
        config['data_name'],
        f"summary_{config['exp_name'] or 'run'}_{config['data_name']}_seeds{seeds_str}.csv",
    )
    pd.DataFrame([summary_row]).to_csv(summary_csv, index=False)
    print(f"\nSummary CSV saved: {summary_csv}")

    # Print aggregated results
    def fmt_mean_std(prefix, metrics_list):
        parts = []
        for k in metrics_list[0]:
            vals = [m[k] for m in metrics_list]
            parts.append(f"{k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
        print(f"{prefix}: [{', '.join(parts)}]")

    print(f"\nAggregated over {len(seeds)} seeds:")
    fmt_mean_std("Val ", all_val_metrics)
    fmt_mean_std("Test", all_test_metrics)


if __name__ == "__main__":
    main()
