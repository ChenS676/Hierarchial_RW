#!/usr/bin/env python
"""
run_ddi.py — NeuralWalkerEncoder link prediction on ogbl-ddi.

Training pipeline mirrors run_demo.py exactly:
  - One full-graph encoder forward per epoch
  - Sub-sample training edges each epoch (--train_subsample)
  - One backward per epoch
  - Trapezoidal LR, grad clip, patience-based early stopping

ogbl-ddi specifics:
  - No node features: internal nn.Embedding indexed by node id
  - OGB pre-defined val/test negatives
  - Metric: Hits@K via OGB Evaluator
"""

# =============================================================================
# 1. Imports
# =============================================================================
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import wandb
    _WANDB_AVAILABLE = True
except ModuleNotFoundError:
    _WANDB_AVAILABLE = False

    class _WandbStub:
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
from torch_geometric.utils import coalesce, remove_self_loops, to_undirected

from neuralwalker_lp import NeuralWalkerEncoder, RandomWalkSampler


# =============================================================================
# 2. Config
# =============================================================================

def get_config():
    parser = argparse.ArgumentParser(description='NeuralWalker ogbl-ddi')

    run = parser.add_argument_group('Run')
    run.add_argument('--seed',              type=int,   default=2025)
    run.add_argument('--num_epochs',        type=int,   default=500)
    run.add_argument('--patience',          type=int,   default=50)
    run.add_argument('--eval_steps',        type=int,   default=5)
    run.add_argument('--eval_metric',       type=str,   default='Hits@20')
    run.add_argument('--global_batch_size', type=int,   default=65536,
                     help='batch size used only during MLP scoring in eval')
    run.add_argument('--train_subsample',   type=int,   default=200000,
                     help='number of training edges sampled per epoch')

    data = parser.add_argument_group('Data')
    data.add_argument('--data_root', type=str, default='data/ogbl_ddi')

    arch = parser.add_argument_group('Architecture')
    arch.add_argument('--hidden_dim',     type=int,   default=128)
    arch.add_argument('--num_layers',     type=int,   default=2)
    arch.add_argument('--walk_length',    type=int,   default=16)
    arch.add_argument('--window_size',    type=int,   default=8)
    arch.add_argument('--sample_rate',    type=float, default=1.0)
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
    opt.add_argument('--lr',              type=float, default=0.005)
    opt.add_argument('--min_lr',          type=float, default=0.0)
    opt.add_argument('--grad_clip_norm',  type=float, default=1.0)
    opt.add_argument('--neg_sample_ratio',type=int,   default=1)

    wb = parser.add_argument_group('W&B')
    wb.add_argument('--wb_project', type=str, default='neuralwalker-ddi')
    wb.add_argument('--wb_entity',  type=str, default='')
    wb.add_argument('--exp_name',   type=str, default='')

    args = parser.parse_args()
    return vars(args)


# =============================================================================
# 3. Model
# =============================================================================

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

    def forward(self, h1, h2):
        x = h1 * h2
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return torch.sigmoid(self.lins[-1](x))


# =============================================================================
# 4. Utilities
# =============================================================================

def binary_cross_entropy_loss(pos_scores, neg_scores):
    return (- torch.log(pos_scores + 1e-15).mean()
            - torch.log(1 - neg_scores + 1e-15).mean())


def trapezoidal_lr_schedule(step, max_lr, min_lr, warmup, cool, total):
    if warmup == 0 or step >= warmup:
        if step <= total - cool:
            return max_lr
        return ((total - step) / cool) * (max_lr - min_lr) + min_lr
    return (step / warmup) * (max_lr - min_lr) + min_lr


def reset_walk(walk_data, node_idx):
    """Restore integer node indices and clear stale edge_attr after encoder forward."""
    walk_data.x = node_idx.clone()
    walk_data.edge_attr = None
    if hasattr(walk_data, 'virtual_node'):
        walk_data.virtual_node = None
    return walk_data


# =============================================================================
# 5. Evaluation
# =============================================================================

@torch.no_grad()
def eval_hits(encoder, decoder, walk_data, node_idx, pos_edge, neg_edge,
              evaluator, k_list, device, batch_size):
    encoder.eval()
    decoder.eval()

    reset_walk(walk_data, node_idx)
    h = encoder(walk_data).float()   # [N, hidden_dim]
    reset_walk(walk_data, node_idx)

    def score_edges(edge):
        preds = []
        for perm in DataLoader(range(edge.size(1)), batch_size=batch_size):
            perm = perm.to(device)
            s = decoder(h[edge[0, perm]], h[edge[1, perm]])
            preds.append(s.squeeze().cpu())
        return torch.cat(preds)

    pos_pred = score_edges(pos_edge)
    neg_pred = score_edges(neg_edge)

    results = {}
    for K in k_list:
        evaluator.K = K
        results[f'Hits@{K}'] = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
    return results


# =============================================================================
# 6. Setup
# =============================================================================

def setup_data(config, device):
    dataset = PygLinkPropPredDataset(name='ogbl-ddi', root=config['data_root'])
    data    = dataset[0]
    split   = dataset.get_edge_split()

    num_nodes = data.num_nodes

    train_pos = split['train']['edge'].t().contiguous().to(device)
    val_pos   = split['valid']['edge'].t().contiguous().to(device)
    val_neg   = split['valid']['edge_neg'].t().contiguous().to(device)
    test_pos  = split['test']['edge'].t().contiguous().to(device)
    test_neg  = split['test']['edge_neg'].t().contiguous().to(device)

    # Undirected training graph for walk pre-computation
    train_undirected = to_undirected(train_pos, num_nodes=num_nodes)
    train_undirected, _ = coalesce(train_undirected, None, num_nodes=num_nodes)
    train_undirected, _ = remove_self_loops(train_undirected)

    print(
        f"[ogbl-ddi] nodes={num_nodes}  train_edges={train_pos.size(1)}  "
        f"val_pos={val_pos.size(1)}  val_neg={val_neg.size(1)}  "
        f"test_pos={test_pos.size(1)}  test_neg={test_neg.size(1)}"
    )
    return num_nodes, train_undirected, train_pos, val_pos, val_neg, test_pos, test_neg


def setup_walks(num_nodes, train_undirected, config, device):
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

    node_idx = node_idx_cpu.to(device)
    walk_data.x = node_idx.clone()

    print(
        f"Walks:  walk_node_idx {tuple(walk_data.walk_node_idx.shape)}  "
        f"walk_node_id_encoding {tuple(walk_data.walk_node_id_encoding.shape)}"
    )
    return walk_data, node_idx


def setup_model(config, num_nodes, device):
    torch.manual_seed(config['seed'])

    encoder = NeuralWalkerEncoder(
        in_node_dim=num_nodes,
        node_embed=True,
        in_edge_dim=None,
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
        update_edges=False,
        local_mp_type='gin_no_edge',
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


# =============================================================================
# 7. main()
# =============================================================================

def main():
    config = get_config()
    torch.manual_seed(config['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    num_nodes, train_undirected, train_pos, val_pos, val_neg, test_pos, test_neg = \
        setup_data(config, device)

    # ── Walks (offline, pre-computed once on training graph) ──────────────────
    walk_data, node_idx = setup_walks(num_nodes, train_undirected, config, device)

    # ── W&B ───────────────────────────────────────────────────────────────────
    run_name = (
        f"nw-ddi_{config['walk_encoder']}"
        f"_wl{config['walk_length']}_h{config['hidden_dim']}"
        f"_l{config['num_layers']}_lr{config['lr']}_seed{config['seed']}"
    )
    wandb.init(
        project=config['wb_project'],
        name=run_name,
        config=config,
        entity=config['wb_entity'] or None,
    )
    run_id = wandb.run.id if wandb.run else 'local'

    save_dir = os.path.join('ogbl-ddi', 'checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    BEST_MODEL_PATH = os.path.join(save_dir, f'best_model_{run_id}.pth')

    # ── Model ─────────────────────────────────────────────────────────────────
    encoder, decoder, optimizer = setup_model(config, num_nodes, device)

    # ── OGB evaluator ─────────────────────────────────────────────────────────
    evaluator = Evaluator(name='ogbl-ddi')
    k_list    = [10, 20, 30, 50]

    # ── Training state ────────────────────────────────────────────────────────
    best_val_metric   = 0.0
    best_val_epoch    = 0
    best_val_metrics  = {}
    best_test_metrics = {}
    no_improve        = 0
    global_step       = 0

    # LR schedule: trapezoidal across epochs (one step per epoch, like run_demo.py)
    total_steps = config['num_epochs']
    warmup      = total_steps // 10
    cool        = total_steps - warmup

    train_size     = train_pos.size(1)
    train_subsample = min(config['train_subsample'], train_size)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    pbar = tqdm.tqdm(total=total_steps, desc='epochs')

    # ==========================================================================
    # Training loop  (one encoder forward per epoch, mirrors run_demo.py)
    # ==========================================================================
    for epoch in range(config['num_epochs']):
        encoder.train()
        decoder.train()

        lr = trapezoidal_lr_schedule(
            global_step, config['lr'], config['min_lr'],
            warmup, cool, total_steps,
        )
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        # Sub-sample training edges for this epoch
        torch.manual_seed(config['seed'] + epoch)
        perm = torch.randperm(train_size, device=device)[:train_subsample]
        batch_pos = train_pos[:, perm]          # [2, train_subsample]

        # ONE full-graph encoder forward (N=4267 nodes; fast)
        reset_walk(walk_data, node_idx)
        h = encoder(walk_data).float()          # [N, hidden_dim]

        # Random negative destinations
        neg_dst = torch.randint(
            0, num_nodes,
            (train_subsample * config['neg_sample_ratio'],),
            device=device,
        )
        neg_src = batch_pos[0].repeat(config['neg_sample_ratio'])

        pos_scores = decoder(h[batch_pos[0]], h[batch_pos[1]])
        neg_scores = decoder(h[neg_src], h[neg_dst])
        if config['neg_sample_ratio'] > 1:
            neg_scores = neg_scores.view(-1, config['neg_sample_ratio'])

        loss = binary_cross_entropy_loss(pos_scores, neg_scores)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            config['grad_clip_norm'],
        )
        optimizer.step()
        global_step += 1

        wandb.log({'train/loss': loss.item(), 'train/lr': lr}, step=global_step)
        pbar.set_description(f"loss: {loss.item():.4f}, t: {time.time()-t0:.2f}s")
        pbar.update(1)

        if torch.isnan(loss):
            print("NaN loss — stopping.")
            break

        # ── Evaluation ────────────────────────────────────────────────────────
        is_eval = (
            epoch == 0
            or (epoch + 1) % config['eval_steps'] == 0
            or epoch == config['num_epochs'] - 1
        )
        if not is_eval:
            continue

        val_results  = eval_hits(encoder, decoder, walk_data, node_idx,
                                 val_pos,  val_neg,  evaluator, k_list,
                                 device, config['global_batch_size'])
        test_results = eval_hits(encoder, decoder, walk_data, node_idx,
                                 test_pos, test_neg, evaluator, k_list,
                                 device, config['global_batch_size'])

        wandb.log(
            {**{f'val/{k}':  v for k, v in val_results.items()},
             **{f'test/{k}': v for k, v in test_results.items()}},
            step=global_step,
        )

        val_primary = val_results[config['eval_metric']]
        if val_primary > best_val_metric:
            best_val_metric   = val_primary
            best_val_epoch    = epoch + 1
            best_val_metrics  = dict(val_results)
            best_test_metrics = dict(test_results)
            if wandb.run is not None:
                wandb.run.summary.update(
                    {f'best_val_{k}':  v for k, v in val_results.items()}
                  | {f'best_test_{k}': v for k, v in test_results.items()}
                  | {'best_val_epoch': best_val_epoch}
                )
            torch.save({
                'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'config': config,
            }, BEST_MODEL_PATH)
            no_improve = 0
        else:
            no_improve += 1

        fmt_r = lambda d: ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
        print(
            f"[ep {epoch+1}] val: [{fmt_r(val_results)}]  "
            f"test: [{fmt_r(test_results)}]  "
            f"no_improve={no_improve}"
        )

        if no_improve >= config['patience']:
            print(f"Early stop @ epoch {epoch+1}")
            break

    pbar.close()

    # ==========================================================================
    # Results
    # ==========================================================================
    if torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"Peak vRAM: {peak_gib:.3f} GiB")

    def fmt(d): return ', '.join(f'{k}: {v:.4f}' for k, v in d.items())
    print(f"Best val {config['eval_metric']}: {best_val_metric:.4f} @ epoch {best_val_epoch}")
    print(f"Best val  metrics: [{fmt(best_val_metrics)}]")
    print(f"Best test metrics: [{fmt(best_test_metrics)}]")

    if wandb.run is not None:
        wandb.run.summary.update(
            {f'best_val_{k}':  v for k, v in best_val_metrics.items()}
          | {f'best_test_{k}': v for k, v in best_test_metrics.items()}
          | {'best_val_epoch': best_val_epoch}
        )

    # Per-run CSV
    os.makedirs('ogbl-ddi', exist_ok=True)
    run_csv = os.path.join('ogbl-ddi', f'NeuralWalker_DDI_{run_id}.csv')
    pd.DataFrame([{
        'run_id':        run_id,
        'seed':          config['seed'],
        'walk_encoder':  config['walk_encoder'],
        'best_val_epoch': best_val_epoch,
        **{f'best_val_{k}':  v for k, v in best_val_metrics.items()},
        **{f'best_test_{k}': v for k, v in best_test_metrics.items()},
    }]).to_csv(run_csv, index=False)
    print(f"CSV saved: {run_csv}")

    wandb.finish()


if __name__ == '__main__':
    main()
