#!/usr/bin/env python
"""
run_demo.py — NeuralWalkerEncoder link prediction on Cora.

Evaluation pipeline is an exact port of Cora_clean.py:
  get_metric_score / test_edge / evaluate_link_prediction / evaluate_and_log
  all use the same signatures and logic; the only change is that test_edge
  runs one full-graph NeuralWalker forward instead of per-batch walk sampling.

    cd /path/to/Plaintoid
    python ../neuralwalker_standalone/run_demo.py [--args]

Dependencies: torch, torch_geometric, torch_scatter, einops, numpy, pandas,
              wandb, ogb, tqdm, sklearn
"""

# =============================================================================
# 1. Imports
# =============================================================================
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import wandb BEFORE adding Plaintoid/ to sys.path.
# Plaintoid/ contains a wandb/ run-log directory; Python 3 namespace packages
# would shadow the real wandb package if that directory is on the path first.
# wandb is optional — falls back to a no-op stub if not installed.
try:
    import wandb
    _WANDB_AVAILABLE = True
except ModuleNotFoundError:
    _WANDB_AVAILABLE = False

    class _WandbStub:
        """No-op stub so the rest of the script runs without wandb installed."""
        run = None
        class _Run:
            summary = {}
            def update(self, d): pass
        def init(self, **kwargs):
            print("[wandb] not installed — logging disabled.")
            self.run = self._Run()
            return self.run
        def log(self, *a, **kw): pass
        def finish(self, *a, **kw): pass
        def Artifact(self, **kw): return self
        def add_file(self, *a, **kw): pass
        def log_artifact(self, *a, **kw): pass

    wandb = _WandbStub()

# Locate evalutors.py (lives in Plaintoid/); works whether the script is run
# from Plaintoid/, neuralwalker_standalone/, or the repo root.
_here = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.getcwd(),
    os.path.join(_here, '..', 'Plaintoid'),
    _here,
]:
    _candidate = os.path.abspath(_candidate)
    if os.path.isfile(os.path.join(_candidate, 'evalutors.py')):
        sys.path.insert(0, _candidate)
        break

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import tqdm.auto as tqdm
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import (
    coalesce, degree, negative_sampling, remove_self_loops, to_undirected,
)

from evalutors import evaluate_auc, evaluate_hits, evaluate_mrr
from neuralwalker_lp import NeuralWalkerEncoder, RandomWalkSampler


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


def get_config():
    parser = argparse.ArgumentParser(description='NeuralWalker Link Prediction')

    run = parser.add_argument_group('Run')
    run.add_argument('--seed',             type=int,   default=2025)
    run.add_argument('--num_epochs',       type=int,   default=300)
    run.add_argument('--global_batch_size',type=int,   default=256)
    run.add_argument('--patience',         type=int,   default=20)
    run.add_argument('--eval_metric',      type=str,   default='MRR')

    data = parser.add_argument_group('Data')
    data.add_argument('--data_root',        type=str,   default='data/Cora')
    data.add_argument('--data_name',        type=str,   default='Cora')
    data.add_argument('--val_split_ratio',  type=float, default=0.15)
    data.add_argument('--test_split_ratio', type=float, default=0.05)

    arch = parser.add_argument_group('Architecture')
    arch.add_argument('--hidden_dim',     type=int,   default=128)
    arch.add_argument('--num_layers',     type=int,   default=1)
    arch.add_argument('--walk_length',    type=int,   default=8)
    arch.add_argument('--window_size',    type=int,   default=4)
    arch.add_argument('--dropout',        type=float, default=0.1)
    arch.add_argument('--mlp_num_layers', type=int,   default=3)
    arch.add_argument('--mlp_dropout',    type=float, default=0.1)

    opt = parser.add_argument_group('Optimizer')
    opt.add_argument('--lr',             type=float, default=1e-3)
    opt.add_argument('--min_lr',         type=float, default=0.0)
    opt.add_argument('--grad_clip_norm', type=float, default=0.1)

    misc = parser.add_argument_group('Misc')
    misc.add_argument('--neg_sample_ratio', type=int,  default=1)
    misc.add_argument('--hits_k', type=int, nargs='+', default=[1, 10, 50, 100])

    wb = parser.add_argument_group('W&B')
    wb.add_argument('--wb_entity',  type=str, default='')
    wb.add_argument('--wb_project', type=str, default='neuralwalker-cora-link-pred')
    wb.add_argument('--exp_name',   type=str, default='')
    wb.add_argument('--run_tag',    type=str, default='')

    args = parser.parse_args()
    return vars(args)


# =============================================================================
# 3. Model definitions
# =============================================================================

class LinkPredictorMLP(nn.Module):
    """Exact copy of Cora_clean.py LinkPredictorMLP — outputs sigmoid scores."""
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
        return torch.sigmoid(self.lins[-1](x))   # [E, 1], scores in [0, 1]


# =============================================================================
# 4. Training utilities  (exact copies from Cora_clean.py)
# =============================================================================

def binary_cross_entropy_loss(pos_scores, neg_scores):
    return (- torch.log(pos_scores + 1e-15).mean()
            - torch.log(1 - neg_scores + 1e-15).mean())


def trapezoidal_lr_schedule(global_batch_idx, max_lr, min_lr, warmup, cool, total_batches):
    if global_batch_idx <= warmup:
        return (global_batch_idx / warmup) * (max_lr - min_lr) + min_lr
    if global_batch_idx <= total_batches - cool:
        return max_lr
    return ((total_batches - global_batch_idx) / cool) * (max_lr - min_lr) + min_lr


def reset_batch(d, rx):
    """Restore raw node features and clear all mutable inter-pass state."""
    d.x = rx.clone()
    # WalkEncoder writes batch.edge_attr each forward; resetting prevents the
    # stale computation graph from leaking into the next forward pass.
    d.edge_attr = None
    if hasattr(d, 'virtual_node') and d.virtual_node is not None:
        d.virtual_node = None
    return d


# =============================================================================
# 5. get_metric_score  (exact copy from Cora_clean.py)
# =============================================================================

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
    result['AP']  = auc_result['AP']
    return result


# =============================================================================
# 6. Evaluation  (ported from Cora_clean.py; test_edge adapted for NeuralWalker)
# =============================================================================

@torch.no_grad()
def test_edge(encoder, decoder, walk_data, raw_x, config, edge_index, batch_size, device):
    """
    NeuralWalker adaptation of Cora_clean's test_edge.

    Cora_clean samples walks on-the-fly per mini-batch of nodes.
    Here we run one full-graph NeuralWalker forward (all N nodes at once),
    then batch only the MLP decoder — identical result, lower overhead.
    """
    encoder.eval()
    decoder.eval()

    reset_batch(walk_data, raw_x)
    h = encoder(walk_data).float()    # [N, hidden_size]
    reset_batch(walk_data, raw_x)     # clear stale edge_attr for next call

    src, dst = edge_index[0], edge_index[1]
    all_scores = []
    for perm in DataLoader(range(edge_index.size(1)), batch_size=batch_size):
        perm   = perm.to(device)
        scores = decoder(h[src[perm]], h[dst[perm]])
        all_scores.append(scores.cpu())
    return torch.cat(all_scores, dim=0).float()


@torch.no_grad()
def evaluate_link_prediction(
    encoder, decoder, walk_data, raw_x, config,
    pos_edge_index, neg_edge_index,
    evaluator_hit, evaluator_mrr, device, eval_batch_size=512,
):
    encoder.eval()
    decoder.eval()
    pos_scores = test_edge(encoder, decoder, walk_data, raw_x, config,
                           pos_edge_index, eval_batch_size, device)
    neg_scores = test_edge(encoder, decoder, walk_data, raw_x, config,
                           neg_edge_index, eval_batch_size, device)
    return get_metric_score(
        evaluator_hit, evaluator_mrr,
        pos_val_pred=torch.flatten(pos_scores),
        neg_val_pred=torch.flatten(neg_scores),
        k_list=config.get('hits_k', [1, 10, 50, 100]),
    )


@torch.no_grad()
def evaluate_and_log(
    encoder, decoder, walk_data, raw_x, config,
    evaluator_hit, evaluator_mrr, device,
    train_pos_edge_index, train_neg_edge_index,
    val_pos_edge_index,   val_neg_edge_index,
    test_pos_edge_index,  test_neg_edge_index,
    epoch, best_val_eval_metric, best_val_metrics, best_test_metrics,
    best_val_epoch, epochs_without_improvement, BEST_MODEL_PATH,
    global_batch_idx=None,
):
    encoder.eval()
    decoder.eval()

    eval_bs = config['global_batch_size']

    def ev(pos, neg):
        return evaluate_link_prediction(
            encoder, decoder, walk_data, raw_x, config,
            pos, neg, evaluator_hit, evaluator_mrr, device, eval_bs,
        )

    train_results = ev(train_pos_edge_index, train_neg_edge_index)
    val_results   = ev(val_pos_edge_index,   val_neg_edge_index)
    test_results  = ev(test_pos_edge_index,  test_neg_edge_index)

    wandb.log(
        {**{f'train/{k}': v for k, v in train_results.items()},
         **{f'val/{k}':   v for k, v in val_results.items()},
         **{f'test/{k}':  v for k, v in test_results.items()}},
        step=global_batch_idx,
    )

    for k, v in val_results.items():
        best_val_metrics[k]  = max(v, best_val_metrics.get(k,  0.0))
    for k, v in test_results.items():
        best_test_metrics[k] = max(v, best_test_metrics.get(k, 0.0))

    if wandb.run is not None:
        wandb.run.summary.update(
            {f'best_val_{k}':  best_val_metrics[k]  for k in val_results}
          | {f'best_test_{k}': best_test_metrics[k] for k in test_results}
        )

    val_metric = val_results[config['eval_metric']]
    early_stop = False

    if val_metric > best_val_eval_metric:
        best_val_eval_metric = val_metric
        best_val_epoch = epoch + 1
        if wandb.run is not None:
            wandb.run.summary['best_val_epoch'] = best_val_epoch
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'config': config,
        }, BEST_MODEL_PATH)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= config['patience']:
        early_stop = True

    return (
        best_val_eval_metric, best_val_metrics, best_test_metrics,
        best_val_epoch, epochs_without_improvement, early_stop,
    )


# =============================================================================
# 7. Setup functions
# =============================================================================

def setup_data(config, device):
    if config['data_name'] in ('Cora', 'PubMed', 'CiteSeer'):
        dataset = Planetoid(root=config['data_root'], name=config['data_name'])
        data = dataset[0].to(device)
    else:
        raise ValueError(f"Unknown dataset: {config['data_name']}")

    if data.is_directed():
        print("Directed graph → converting to undirected.")
        data.edge_index = to_undirected(data.edge_index)
    data.edge_index, _ = coalesce(data.edge_index, None, num_nodes=data.num_nodes)
    data.edge_index, _ = remove_self_loops(data.edge_index)

    deg = degree(data.edge_index[1], data.num_nodes, dtype=torch.float)
    print(
        f"[{config['data_name']}] nodes={data.num_nodes}  "
        f"edges={data.edge_index.size(1)}  avg_deg={deg.mean().item():.2f}  "
        f"max_deg={int(deg.max())}  feat_dim={data.x.size(1)}"
    )

    transform = T.RandomLinkSplit(
        num_val=config['val_split_ratio'],
        num_test=config['test_split_ratio'],
        is_undirected=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(data)
    train_pos = train_data.edge_index.to(device)

    def pos_neg(d):
        return (
            d.edge_label_index[:, d.edge_label == 1].to(device),
            d.edge_label_index[:, d.edge_label == 0].to(device),
        )

    val_pos,  val_neg  = pos_neg(val_data)
    test_pos, test_neg = pos_neg(test_data)

    print(
        f"Edges:  train_pos={train_pos.size(1)}  "
        f"val_pos={val_pos.size(1)}  val_neg={val_neg.size(1)}  "
        f"test_pos={test_pos.size(1)}  test_neg={test_neg.size(1)}"
    )
    return data, train_pos, val_pos, val_neg, test_pos, test_neg


def setup_walks(data, train_pos, config, device):
    """Pre-compute random walks on the training graph (offline, done once)."""
    walk_input = Data(
        x=data.x.cpu(),
        edge_index=train_pos.cpu(),
        num_nodes=data.num_nodes,
    )
    sampler   = RandomWalkSampler(length=config['walk_length'], window_size=config['window_size'])
    walk_data = sampler(walk_input)
    walk_data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    walk_data = walk_data.to(device)

    raw_x = data.x.to(device).clone()
    walk_data.x = raw_x.clone()

    print(
        f"Walks:  walk_node_idx {tuple(walk_data.walk_node_idx.shape)}  "
        f"walk_node_id_encoding {tuple(walk_data.walk_node_id_encoding.shape)}"
    )
    return walk_data, raw_x


def setup_model_and_optimizer(config, in_node_dim, device):
    torch.manual_seed(config['seed'])

    encoder = NeuralWalkerEncoder(
        in_node_dim=in_node_dim,
        node_embed=False,
        in_edge_dim=None,
        hidden_size=config['hidden_dim'],
        num_layers=config['num_layers'],
        walk_encoder='conv',
        walk_length=config['walk_length'],
        window_size=config['window_size'],
        dropout=config['dropout'],
    ).to(device)

    decoder = LinkPredictorMLP(
        in_dim=config['hidden_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['mlp_num_layers'],
        dropout=config['mlp_dropout'],
    ).to(device)

    print(f"Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config['lr'],
    )
    return encoder, decoder, optimizer


def setup_wandb(config, in_node_dim):
    tag_prefix = f"{config['exp_name']}_" if config['exp_name'] else ''
    run_name = (
        f"{tag_prefix}nw"
        f"_wl{config['walk_length']}_ws{config['window_size']}"
        f"_h{config['hidden_dim']}_l{config['num_layers']}"
        f"_lr{config['lr']}_seed{config['seed']}"
    )
    project = (
        f"{config['data_name']}_{config['run_tag']}"
        if config['run_tag']
        else config['wb_project']
    )
    init_kwargs = dict(
        project=project,
        name=run_name,
        config={**config, 'in_node_dim': in_node_dim},
    )
    if config['wb_entity']:
        init_kwargs['entity'] = config['wb_entity']
    wandb.init(**init_kwargs)
    return wandb.run.id


# =============================================================================
# 8. main()
# =============================================================================

def main():
    config = get_config()
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------- Data ----------
    data, train_pos, val_pos, val_neg, test_pos, test_neg = setup_data(config, device)

    # ---------- Walks (offline, once on training graph) ----------
    walk_data, raw_x = setup_walks(data, train_pos, config, device)

    # ---------- W&B ----------
    run_id = setup_wandb(config, data.x.size(1))
    save_dir = os.path.join(config['data_name'], 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(save_dir, f'best_model_{run_id}.pth')

    # ---------- Model & optimizer ----------
    encoder, decoder, optimizer = setup_model_and_optimizer(config, data.x.size(1), device)

    # ---------- OGB evaluators (same as Cora_clean.py) ----------
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')

    # ---------- Training bookkeeping (mirrors Cora_clean.py) ----------
    best_val_eval_metric       = 0.0
    best_val_epoch             = 0
    epochs_without_improvement = 0
    best_val_metrics           = {}
    best_test_metrics          = {}
    epochs_eval_steps          = 5
    global_batch_idx           = 0

    # LR schedule: trapezoidal across epochs (one step per epoch)
    total_steps = config['num_epochs']
    warmup = total_steps // 10
    cool   = total_steps - warmup

    # Sub-sample train positives for eval (same size as val_neg, as in Cora_clean)
    n_eval_train = min(val_neg.size(1), train_pos.size(1))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm.tqdm(total=total_steps)

    # ==========================================================================
    # Training loop
    # ==========================================================================
    for epoch in range(config['num_epochs']):
        encoder.train()
        decoder.train()

        # Trapezoidal LR schedule (mirrors Cora_clean.py)
        lr = trapezoidal_lr_schedule(
            global_batch_idx, config['lr'], config['min_lr'],
            warmup, cool, total_steps,
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        # Full-graph encoder forward (NeuralWalker processes all N nodes at once)
        reset_batch(walk_data, raw_x)
        h = encoder(walk_data).float()   # [N, hidden_size]

        # Negative sampling (same as Cora_clean.py)
        neg = negative_sampling(
            train_pos,
            num_nodes=data.num_nodes,
            num_neg_samples=train_pos.size(1) * config['neg_sample_ratio'],
        )

        pos_scores = decoder(h[train_pos[0]], h[train_pos[1]])
        neg_scores = decoder(h[neg[0]],       h[neg[1]])
        if config['neg_sample_ratio'] > 1:
            neg_scores = neg_scores.view(-1, config['neg_sample_ratio'])

        loss = binary_cross_entropy_loss(pos_scores, neg_scores)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            config['grad_clip_norm'],
        )
        optimizer.step()

        wandb.log(
            {'train/loss': loss.item(), 'train/lr': lr},
            step=global_batch_idx,
        )
        global_batch_idx += 1

        pbar.set_description(
            f"loss: {loss.item():.4f}, t: {time.time()-t0:.2f}s"
        )
        pbar.update(1)

        if torch.isnan(loss):
            break

        # Sub-sampled train eval edges (matched to val_neg size, as in Cora_clean)
        torch.manual_seed(config['seed'] + epoch)
        perm = torch.randperm(train_pos.size(1), device=device)[:n_eval_train]
        eval_train_pos = train_pos[:, perm]

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
                encoder=encoder, decoder=decoder,
                walk_data=walk_data, raw_x=raw_x, config=config,
                evaluator_hit=evaluator_hit, evaluator_mrr=evaluator_mrr, device=device,
                train_pos_edge_index=eval_train_pos,
                train_neg_edge_index=val_neg,   # fixed negatives for train eval (Cora_clean style)
                val_pos_edge_index=val_pos,
                val_neg_edge_index=val_neg,
                test_pos_edge_index=test_pos,
                test_neg_edge_index=test_neg,
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

    # ==========================================================================
    # Post-training: memory report, final W&B summary, CSV results
    # ==========================================================================
    if torch.cuda.is_available():
        peak_gib    = torch.cuda.max_memory_allocated() / (1024 ** 3)
        current_gib = torch.cuda.memory_allocated()    / (1024 ** 3)
        print(f"Peak vRAM: {peak_gib:.3f} GiB | Current: {current_gib:.3f} GiB")

    if wandb.run is not None:
        wandb.run.summary.update(
            {f'best_val_{k}':  v for k, v in best_val_metrics.items()}
          | {f'best_test_{k}': v for k, v in best_test_metrics.items()}
          | {'best_val_epoch': best_val_epoch}
        )

    def fmt(d): return ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
    print(f"Best val {config['eval_metric']}: {best_val_eval_metric:.4f} @ epoch {best_val_epoch}")
    print(f"Best val  metrics: [{fmt(best_val_metrics)}]")
    print(f"Best test metrics: [{fmt(best_test_metrics)}]")

    # Per-run CSV
    os.makedirs(config['data_name'], exist_ok=True)
    run_csv = os.path.join(
        config['data_name'],
        f"NeuralWalker_{config['data_name']}_LinkPred_{run_id}.csv",
    )
    pd.DataFrame([{
        'model_data_seed': f"NeuralWalker_{config['data_name']}_LinkPred_{config['seed']}",
        'best_val_epoch':  best_val_epoch,
        **{f'best_val_{k}':  v for k, v in best_val_metrics.items()},
        **{f'best_test_{k}': v for k, v in best_test_metrics.items()},
    }]).to_csv(run_csv, index=False)

    # Aggregated experiment CSV (append mode, mirrors Cora_clean.py)
    exp_csv = os.path.join(config['data_name'], 'experiment_results_neuralwalker.csv')
    exp_row = {
        'seed': config['seed'], 'hidden_dim': config['hidden_dim'],
        'num_layers': config['num_layers'], 'walk_length': config['walk_length'],
        'window_size': config['window_size'], 'lr': config['lr'],
        'grad_clip_norm': config['grad_clip_norm'],
        'neg_sample_ratio': config['neg_sample_ratio'],
        'patience': config['patience'], 'best_val_epoch': best_val_epoch,
        **{f'metrics(best_test_{k})': v for k, v in best_test_metrics.items()},
        **{f'metrics(best_val_{k})':  v for k, v in best_val_metrics.items()},
    }
    write_header = not os.path.exists(exp_csv)
    pd.DataFrame([exp_row]).to_csv(exp_csv, mode='a', header=write_header, index=False)

    print(f"CSV saved: {run_csv}")
    wandb.finish()


if __name__ == '__main__':
    main()
