#!/usr/bin/env python
"""
run_ddi_my.py — NeuralWalkerEncoder link prediction on ogbl-ddi.

Same training pipeline as run_demo.py (one full-graph encoder forward per epoch,
trapezoidal LR, grad clip, patience early stopping). Only data loading and
evaluation are replaced for ogbl-ddi (OGB split, Hits@K metric).
"""

# =============================================================================
# 1. Imports
# =============================================================================
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

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import (
    coalesce, negative_sampling, remove_self_loops, to_undirected,
)

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
    parser = argparse.ArgumentParser(description='NeuralWalker ogbl-ddi Link Prediction')

    run = parser.add_argument_group('Run')
    run.add_argument('--seed',             type=int,   default=2025)
    run.add_argument('--num_epochs',       type=int,   default=200)
    run.add_argument('--global_batch_size',type=int,   default=65536)
    run.add_argument('--patience',         type=int,   default=50)
    run.add_argument('--eval_metric',      type=str,   default='Hits@20')

    data = parser.add_argument_group('Data')
    data.add_argument('--data_root', type=str, default='data/ogbl_ddi')

    arch = parser.add_argument_group('Architecture')
    arch.add_argument('--hidden_dim',     type=int,   default=128)
    arch.add_argument('--num_layers',     type=int,   default=2)
    arch.add_argument('--walk_length',    type=int,   default=16)
    arch.add_argument('--window_size',    type=int,   default=4)
    arch.add_argument('--sample_rate',    type=float, default=0.5)
    arch.add_argument('--dropout',        type=float, default=0.3)
    arch.add_argument('--mlp_num_layers', type=int,   default=3)
    arch.add_argument('--mlp_dropout',    type=float, default=0.1)
    arch.add_argument('--walk_encoder',   type=str,   default='conv',
                      choices=['conv', 'mamba', 's4', 'transformer'])
    arch.add_argument('--num_heads',      type=int,   default=4)
    arch.add_argument('--mlp_ratio',      type=int,   default=1)
    arch.add_argument('--d_state',        type=int,   default=16)
    arch.add_argument('--d_conv',         type=int,   default=9)
    arch.add_argument('--expand',         type=int,   default=2)

    opt = parser.add_argument_group('Optimizer')
    opt.add_argument('--lr',             type=float, default=5e-3)
    opt.add_argument('--grad_clip_norm', type=float, default=1.0)

    misc = parser.add_argument_group('Misc')
    misc.add_argument('--hits_k', type=int, nargs='+', default=[10, 20, 30, 50])

    wb = parser.add_argument_group('W&B')
    wb.add_argument('--wb_entity',  type=str, default='')
    wb.add_argument('--wb_project', type=str, default='neuralwalker-ddi')
    wb.add_argument('--exp_name',   type=str, default='')

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

def reset_batch(d, rx, raw_edge_attr=None):
    """Restore raw node features and clear all mutable inter-pass state."""
    d.x = rx.clone()
    d.edge_attr = raw_edge_attr
    if hasattr(d, 'virtual_node') and d.virtual_node is not None:
        d.virtual_node = None
    return d


# =============================================================================
# 5. Training (adapted from ogbl_ddi_gnn.py train())
# =============================================================================

def _grad_norm(params):
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return 0.0
    return torch.stack([g.norm() for g in grads]).norm().item()


def train_one_epoch(encoder, decoder, walk_data, raw_x, raw_edge_attr,
                    train_pos, num_nodes, optimizer, batch_size, grad_clip, device):
    encoder.train()
    decoder.train()

    total_loss = total_examples = 0
    last_grad_stats = {}
    for perm in DataLoader(range(train_pos.size(1)), batch_size, shuffle=True):
        optimizer.zero_grad()

        reset_batch(walk_data, raw_x, raw_edge_attr)
        h = encoder(walk_data).float()   # [N, hidden_dim]

        perm = perm.to(device)
        edge = train_pos[:, perm]

        pos_out = decoder(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        neg_edge = negative_sampling(
            train_pos, num_nodes=num_nodes,
            num_neg_samples=perm.size(0), method='dense',
        )
        neg_out = decoder(h[neg_edge[0]], h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        # Capture grad norms BEFORE clipping to diagnose each component
        walk_params = list(encoder.blocks[0].walk_encoder.parameters())
        gin_params  = list(encoder.blocks[0].mp_layer.parameters())
        feat_params = list(encoder.feature_encoder.parameters())
        last_grad_stats = {
            'grad/walk_encoder': _grad_norm(walk_params),
            'grad/gin':          _grad_norm(gin_params),
            'grad/feat_encoder': _grad_norm(feat_params),
            'grad/decoder':      _grad_norm(list(decoder.parameters())),
        }

        torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)

        optimizer.step()

        total_loss    += loss.item() * perm.size(0)
        total_examples += perm.size(0)

    return total_loss / total_examples, last_grad_stats


# =============================================================================
# 6. get_metric_score  — ogbl-ddi: Hits@K via OGB Evaluator
# =============================================================================

def get_metric_score(evaluator, pos_pred, neg_pred, k_list):
    result = {}
    for K in k_list:
        evaluator.K = K
        result[f'Hits@{K}'] = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
    return result


# =============================================================================
# 6. Evaluation
# =============================================================================

@torch.no_grad()
def compute_embeddings(encoder, walk_data, raw_x, raw_edge_attr):
    """Single encoder forward pass — call once, reuse h for all splits."""
    reset_batch(walk_data, raw_x, raw_edge_attr)
    h = encoder(walk_data).float()
    reset_batch(walk_data, raw_x, raw_edge_attr)
    return h


@torch.no_grad()
def score_from_h(decoder, h, edge_index, batch_size, device):
    src, dst = edge_index[0], edge_index[1]
    all_scores = []
    for perm in DataLoader(range(edge_index.size(1)), batch_size=batch_size):
        perm = perm.to(device)
        all_scores.append(decoder(h[src[perm]], h[dst[perm]]).cpu())
    return torch.cat(all_scores, dim=0).float().flatten()


@torch.no_grad()
def evaluate_and_log(
    encoder, decoder, walk_data, raw_x, raw_edge_attr, config,
    evaluator, device,
    train_pos_edge_index,
    val_pos_edge_index,  val_neg_edge_index,
    test_pos_edge_index, test_neg_edge_index,
    epoch, best_val_eval_metric, best_val_metrics, best_test_metrics,
    best_val_epoch, epochs_without_improvement, BEST_MODEL_PATH,
    global_batch_idx=None,
):
    encoder.eval()
    decoder.eval()
    eval_bs = config['global_batch_size']
    k_list  = config['hits_k']

    # Single encoder forward — reuse h for all splits
    h = compute_embeddings(encoder, walk_data, raw_x, raw_edge_attr)

    def ev(pos, neg):
        pos_pred = score_from_h(decoder, h, pos, eval_bs, device)
        neg_pred = score_from_h(decoder, h, neg, eval_bs, device)
        return get_metric_score(evaluator, pos_pred, neg_pred, k_list), pos_pred, neg_pred

    val_results,  val_pos_pred,  val_neg_pred  = ev(val_pos_edge_index,  val_neg_edge_index)
    test_results, test_pos_pred, test_neg_pred = ev(test_pos_edge_index, test_neg_edge_index)

    # Score distributions — diagnostic for discriminability
    # Sample train pos to check for overfitting
    n_sample = min(val_pos_edge_index.size(1), train_pos_edge_index.size(1))
    idx = torch.randperm(train_pos_edge_index.size(1))[:n_sample]
    train_pos_sample = train_pos_edge_index[:, idx]
    train_pos_pred = score_from_h(decoder, h, train_pos_sample, eval_bs, device)
    train_neg_pred = val_neg_pred  # reuse val negatives for train overfitting check

    diag = {
        'diag/val_pos_mean':   val_pos_pred.mean().item(),
        'diag/val_neg_mean':   val_neg_pred.mean().item(),
        'diag/val_score_gap':  (val_pos_pred.mean() - val_neg_pred.mean()).item(),
        'diag/test_pos_mean':  test_pos_pred.mean().item(),
        'diag/test_neg_mean':  test_neg_pred.mean().item(),
        'diag/train_pos_mean': train_pos_pred.mean().item(),
        'diag/h_norm_mean':    h.norm(dim=1).mean().item(),
    }

    wandb.log(
        {**{f'val/{k}':  v for k, v in val_results.items()},
         **{f'test/{k}': v for k, v in test_results.items()},
         **diag},
        step=global_batch_idx,
    )

    val_metric = val_results[config['eval_metric']]
    early_stop = False

    if val_metric > best_val_eval_metric:
        best_val_eval_metric = val_metric
        best_val_epoch = epoch + 1
        best_val_metrics  = dict(val_results)
        best_test_metrics = dict(test_results)
        if wandb.run is not None:
            wandb.run.summary.update(
                {f'best_val_{k}':  best_val_metrics[k]  for k in val_results}
              | {f'best_test_{k}': best_test_metrics[k] for k in test_results}
              | {'best_val_epoch': best_val_epoch}
            )
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'config': config,
        }, BEST_MODEL_PATH)
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    fmt_r = lambda d: ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
    print(f"  [ep {epoch+1}] val: [{fmt_r(val_results)}]  test: [{fmt_r(test_results)}]")
    print(f"           diag: val_pos={val_pos_pred.mean():.3f} val_neg={val_neg_pred.mean():.3f} "
          f"train_pos={train_pos_pred.mean():.3f} h_norm={h.norm(dim=1).mean():.3f}")

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
    dataset   = PygLinkPropPredDataset(name='ogbl-ddi', root=config['data_root'])
    data      = dataset[0]
    split     = dataset.get_edge_split()
    num_nodes = data.num_nodes

    train_pos = split['train']['edge'].t().contiguous().to(device)
    val_pos   = split['valid']['edge'].t().contiguous().to(device)
    val_neg   = split['valid']['edge_neg'].t().contiguous().to(device)
    test_pos  = split['test']['edge'].t().contiguous().to(device)
    test_neg  = split['test']['edge_neg'].t().contiguous().to(device)

    # undirected training graph for walk pre-computation
    train_undirected = to_undirected(train_pos, num_nodes=num_nodes)
    train_undirected, _ = coalesce(train_undirected, None, num_nodes=num_nodes)
    train_undirected, _ = remove_self_loops(train_undirected)

    print(
        f"[ogbl-ddi] nodes={num_nodes}  train_edges={train_pos.size(1)}  "
        f"val_pos={val_pos.size(1)}  val_neg={val_neg.size(1)}  "
        f"test_pos={test_pos.size(1)}  test_neg={test_neg.size(1)}"
    )
    return num_nodes, train_undirected, train_pos, val_pos, val_neg, test_pos, test_neg


def setup_walks(num_nodes, train_undirected, config, device, seed=None):
    """Sample random walks on the training graph. Call each epoch with a new seed."""
    if seed is not None:
        torch.manual_seed(seed)
    node_idx_cpu = torch.arange(num_nodes, dtype=torch.long)
    walk_input = Data(
        x=node_idx_cpu,
        edge_index=train_undirected.cpu(),
        num_nodes=num_nodes,
    )
    sampler = RandomWalkSampler(
        length=config['walk_length'],
        window_size=config['window_size'],
        sample_rate=config['sample_rate'],
    )
    walk_data = sampler(walk_input)
    walk_data.batch = torch.zeros(num_nodes, dtype=torch.long)
    walk_data = walk_data.to(device)

    raw_x = node_idx_cpu.to(device)
    walk_data.x = raw_x.clone()

    raw_edge_attr = torch.ones(walk_data.edge_index.size(1), 1, device=device)
    walk_data.edge_attr = raw_edge_attr

    return walk_data, raw_x, raw_edge_attr


def setup_model_and_optimizer(config, num_nodes, device):
    torch.manual_seed(config['seed'])

    encoder = NeuralWalkerEncoder(
        in_node_dim=num_nodes,   # vocabulary size for internal Embedding
        node_embed=True,         # no input features → use nn.Embedding
        in_edge_dim=1,
        edge_embed=False,        # continuous ones → use nn.Linear, not nn.Embedding
        hidden_size=config['hidden_dim'],
        num_layers=config['num_layers'],
        walk_encoder=config['walk_encoder'],
        walk_length=config['walk_length'],
        window_size=config['window_size'],
        dropout=config['dropout'],
        vn_norm_type='layernorm',
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        d_state=config['d_state'],
        d_conv=config['d_conv'],
        expand=config['expand'],
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


def setup_wandb(config, num_nodes):
    tag_prefix = f"{config['exp_name']}_" if config['exp_name'] else ''
    run_name = (
        f"{tag_prefix}nw-ddi_{config['walk_encoder']}"
        f"_wl{config['walk_length']}_ws{config['window_size']}"
        f"_h{config['hidden_dim']}_l{config['num_layers']}"
        f"_lr{config['lr']}_seed{config['seed']}"
    )
    init_kwargs = dict(
        project=config['wb_project'],
        name=run_name,
        config={**config, 'num_nodes': num_nodes},
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
    num_nodes, train_undirected, train_pos, val_pos, val_neg, test_pos, test_neg = \
        setup_data(config, device)

    # ---------- Walks (sampled fresh each epoch) ----------
    walk_data, raw_x, raw_edge_attr = setup_walks(num_nodes, train_undirected, config, device, seed=config['seed'])
    print(
        f"Walks:  walk_node_idx {tuple(walk_data.walk_node_idx.shape)}  "
        f"walk_node_id_encoding {tuple(walk_data.walk_node_id_encoding.shape)}"
    )

    # ---------- W&B ----------
    run_id = setup_wandb(config, num_nodes)
    save_dir = os.path.join('ogbl-ddi', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(save_dir, f'best_model_{run_id}.pth')

    # ---------- Model & optimizer ----------
    encoder, decoder, optimizer = setup_model_and_optimizer(config, num_nodes, device)

    # ---------- OGB evaluator ----------
    evaluator = Evaluator(name='ogbl-ddi')

    # ---------- Training bookkeeping ----------
    best_val_eval_metric       = 0.0
    best_val_epoch             = 0
    epochs_without_improvement = 0
    best_val_metrics           = {}
    best_test_metrics          = {}
    epochs_eval_steps          = 5

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm.tqdm(total=config['num_epochs'])

    # ==========================================================================
    # Training loop  (mini-batch over all training edges, encoder called per batch)
    # ==========================================================================
    for epoch in range(config['num_epochs']):
        t0 = time.time()

        walk_data, raw_x, raw_edge_attr = setup_walks(
            num_nodes, train_undirected, config, device,
            seed=config['seed'] + epoch,
        )

        avg_loss, grad_stats = train_one_epoch(
            encoder, decoder, walk_data, raw_x, raw_edge_attr,
            train_pos, num_nodes, optimizer,
            batch_size=config['global_batch_size'],
            grad_clip=config['grad_clip_norm'],
            device=device,
        )

        wandb.log({'train/loss': avg_loss, **grad_stats}, step=epoch + 1)
        pbar.set_description(
            f"loss: {avg_loss:.4f}  "
            f"g_walk:{grad_stats.get('grad/walk_encoder',0):.3f} "
            f"g_gin:{grad_stats.get('grad/gin',0):.3f} "
            f"g_feat:{grad_stats.get('grad/feat_encoder',0):.3f}  "
            f"t:{time.time()-t0:.1f}s"
        )
        pbar.update(1)

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
                walk_data=walk_data, raw_x=raw_x, raw_edge_attr=raw_edge_attr, config=config,
                evaluator=evaluator, device=device,
                train_pos_edge_index=train_pos,
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
                global_batch_idx=epoch + 1,
            )
            if early_stop:
                print(f"Early stop @ epoch {epoch+1}")
                break

    pbar.close()

    # ==========================================================================
    # Results
    # ==========================================================================
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak vRAM: {peak_gib:.3f} GiB")

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

    os.makedirs('ogbl-ddi', exist_ok=True)
    run_csv = os.path.join('ogbl-ddi', f'NeuralWalker_DDI_{run_id}.csv')
    pd.DataFrame([{
        'run_id': run_id, 'seed': config['seed'],
        'walk_encoder': config['walk_encoder'],
        'best_val_epoch': best_val_epoch,
        **{f'best_val_{k}':  v for k, v in best_val_metrics.items()},
        **{f'best_test_{k}': v for k, v in best_test_metrics.items()},
    }]).to_csv(run_csv, index=False)
    print(f"CSV saved: {run_csv}")

    if wandb.run is not None:
        print(f"W&B run: https://wandb.ai/{wandb.run.entity}/{wandb.run.project}/runs/{wandb.run.id}")
    wandb.finish()


if __name__ == '__main__':
    main()