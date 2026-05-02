import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import graphbench
from tqdm import tqdm
import wandb
import os
import math
import argparse
from typing import Optional, Tuple, List
import random
import numpy as np
from torch.optim import AdamW
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add
import time
import csv
from graphbench.helpers.utils import set_seed

# ==========================================
# 0. UTILS  (identical to HRW pipeline)
# ==========================================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super().__init__()
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
            data.edge_index, data.edge_attr,
            num_nodes=data.num_nodes, reduce="mean"
        )
        data.mp_edge_index = mp_edge_index
        data.mp_edge_attr  = mp_edge_attr
        row    = mp_edge_index[0]
        degree = scatter_add(torch.ones_like(mp_edge_attr), row, dim=0, dim_size=data.num_nodes)
        data.x = torch.cat([degree], dim=-1)
        return data


# ==========================================
# WANDB HELPERS  (identical project, GT prefix)
# ==========================================

WANDB_PROJECT = "max_matching_bench"
WANDB_ENTITY  = "graph-diffusion-model-link-prediction"


def build_run_name(config: dict) -> str:
    """
    GT_<dataset>_h<dim>_L<layers>_H<heads>_<pe-tag>_s<seed>
    Example: GT_bipartite_matching_easy_h256_L4_H8_nope_s2025
    """
    pe_tag = f"_{config['pe_type']}{config['pe_dim']}" if config.get("use_pe") else "_nope"
    return (
        f"GT"
        f"_{config['dataset_name']}"
        f"_h{config['hidden_dim']}"
        f"_L{config['layers']}"
        f"_H{config['num_heads']}"
        f"{pe_tag}"
        f"_s{config['seed']}"
    )


def compute_prf1(y_true, y_pred):
    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()
    precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
    recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# ==========================================
# 1. MODEL — Graph Transformer
# ==========================================

class GTLayer(nn.Module):
    """
    Single Graph Transformer layer using PyG TransformerConv.
    TransformerConv implements the attention-based message passing from
    "Masked Label Prediction: Unified Message Passing Model for
    Semi-Supervised Classification" (Shi et al., 2020), which is
    edge-attribute aware.

    Pre-norm + residual for both the attention sub-layer and the FFN.
    """
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0,
                 edge_dim: int = None):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim if edge_dim is not None else hidden_dim,
            concat=True,          # output: num_heads × (hidden_dim // num_heads) = hidden_dim
            beta=True,            # learnable skip connection inside attention
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # Attention sub-layer (pre-norm + residual)
        x = x + self.conv(self.norm1(x), edge_index, edge_attr=edge_attr)
        # FFN sub-layer (pre-norm + residual)
        x = x + self.ffn(self.norm2(x))
        return x


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
        x = self.dropout(x)
        return self.lin2(x)


class GTForEdgeClassification(nn.Module):
    """
    Graph Transformer edge classifier.
    Message passing on undirected graph (mp_edge_index / mp_edge_attr);
    prediction uses original directed edge features.
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, config: dict):
        super().__init__()
        dim       = config["hidden_dim"]
        num_heads = config.get("num_heads", 8)
        dropout   = config["dropout"]
        self.use_pe = config.get("use_pe", False)

        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        if self.use_pe:
            pe_dim = config.get("pe_dim", 16)
            self.pe_encoder = nn.Sequential(
                nn.Linear(pe_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
            )

        self.layers = nn.ModuleList([
            GTLayer(dim, num_heads, dropout, edge_dim=dim)
            for _ in range(config["layers"])
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.classifier = EdgeDecoderWithFeatures(dim, dropout)

    def forward(self, x, edge_index, edge_attr,
                mp_edge_index=None, mp_edge_attr=None,
                use_pe=False, pe_type=None, lap_pe=None, rwse=None):
        if x.dim() == 1:         x         = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr  = edge_attr.unsqueeze(-1)

        mp_ei = mp_edge_index if mp_edge_index is not None else edge_index
        mp_ea = mp_edge_attr  if mp_edge_attr  is not None else edge_attr
        if mp_ea.dim() == 1: mp_ea = mp_ea.unsqueeze(-1)

        h         = self.node_encoder(x)
        h_mp_edge = self.edge_encoder(mp_ea)
        h_edge    = self.edge_encoder(edge_attr)

        if use_pe:
            if pe_type == "rwse" and rwse is not None:
                h = h + self.pe_encoder(rwse if rwse.dim() > 1 else rwse.unsqueeze(-1))
            elif pe_type == "lap" and lap_pe is not None:
                h = h + self.pe_encoder(torch.abs(lap_pe))

        for layer in self.layers:
            h = layer(h, mp_ei, h_mp_edge)
        h = self.final_norm(h)

        src, dst = edge_index
        return self.classifier(h[src], h[dst], h_edge)


# ==========================================
# 2. DATA PROCESSING  (identical to HRW)
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
# 3. HELPERS  (identical to HRW)
# ==========================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_forward(model, data, config):
    return model(
        data.x.float(), data.edge_index, data.edge_attr.float(),
        mp_edge_index=getattr(data, 'mp_edge_index', None),
        mp_edge_attr=getattr(data, 'mp_edge_attr', None),
        use_pe=config["use_pe"], pe_type=config["pe_type"],
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
        return ((total_steps - global_step) / cool) * (max_lr - min_lr) + min_lr


# ==========================================
# 4. MAIN  (identical training pipeline)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="GT — Bipartite Matching Edge Classification")

    parser.add_argument("--dataset_name",         type=str,   default="bipartite_matching_easy")
    parser.add_argument("--data_root",            type=str,   default="./data_graphbench")
    parser.add_argument("--seed",                 type=int,   default=2025)
    parser.add_argument("--epochs",               type=int,   default=10)
    parser.add_argument("--batch_size",           type=int,   default=64)
    parser.add_argument("--test_batch_size",      type=int,   default=32)
    parser.add_argument("--hidden_dim",           type=int,   default=256)
    parser.add_argument("--layers",               type=int,   default=4)
    parser.add_argument("--num_heads",            type=int,   default=8)
    parser.add_argument("--dropout",              type=float, default=0.1)
    parser.add_argument("--lr",                   type=float, default=3e-4)
    parser.add_argument("--weight_decay",         type=float, default=0.1)
    parser.add_argument("--grad_clip_norm",       type=float, default=0.5)
    parser.add_argument("--train_subset_ratio",   type=float, default=0.1)
    parser.add_argument("--use_pe",               type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--pe_type",              type=str,   default="rwse", choices=["lap", "rwse"])
    parser.add_argument("--pe_dim",               type=int,   default=16)
    parser.add_argument("--eval_metric_class",    type=str,   default='algoreas_classification')

    args   = parser.parse_args()
    config = vars(args)
    import pprint; pprint.pp(config)

    set_seed(config["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    class PEOnUndirected:
        def __init__(self, transform):
            self._transform = transform
        def __call__(self, data):
            orig = data.edge_index
            data.edge_index = data.mp_edge_index
            data = self._transform(data)
            data.edge_index = orig
            return data

    transforms_list = [FixGraphBenchData(), AddUndirectedContext()]
    if config["use_pe"]:
        if config["pe_type"] == "lap":
            transforms_list.append(PEOnUndirected(
                T.AddLaplacianEigenvectorPE(k=config["pe_dim"], attr_name='lap_pe', is_undirected=True)
            ))
        else:
            transforms_list.append(PEOnUndirected(
                T.AddRandomWalkPE(walk_length=config["pe_dim"], attr_name='rwse')
            ))

    dataset = graphbench.Loader(
        root=config["data_root"], dataset_names=config["dataset_name"],
        transform=T.Compose(transforms_list)
    ).load()

    try:
        train_dataset = dataset[0]['train']
        val_dataset   = dataset[0]['valid']
        test_dataset  = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = val_dataset = test_dataset = dataset

    print(f"Sizes → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    _peek       = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"Input dims → node: {node_in_dim}  edge: {edge_in_dim}")

    # ── Model ───────────────────────────────────────────────────────────────
    model        = GTForEdgeClassification(node_in_dim, edge_in_dim, config).to(device)
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

    print(f"Total steps: {total_steps}  warmup: {warmup_steps}  cool: {cool_steps}  eval_every: {eval_every_steps}")

    # ── Optimizer ───────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=config["lr"],
                      weight_decay=config["weight_decay"], betas=(0.9, 0.999))
    print(f"Optimizer: AdamW  lr={config['lr']}  wd={config['weight_decay']}")

    criterion = FocalLoss(gamma=2)

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ── W&B ─────────────────────────────────────────────────────────────────
    run_name = build_run_name(config)
    print(f"W&B project : {WANDB_PROJECT}")
    print(f"W&B run     : {run_name}\n")
    wandb.init(entity=WANDB_ENTITY, project=WANDB_PROJECT, name=run_name, config=config)

    checkpoint_dir  = os.path.join(config["dataset_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    # ── Evaluate helper ──────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(loader, split_name: str):
        assert not model.training
        y_true_list, y_pred_list = [], []
        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = run_forward(model, data, config)
            y_true_list.append(data.y.cpu())
            y_pred_list.append((torch.sigmoid(out) > 0.5).long().cpu())

        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        gb_metrics    = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        gb_acc, gb_f1 = parse_metrics(gb_metrics)
        precision, recall, f1 = compute_prf1(y_true, y_pred)

        print(f"  [{split_name:4s}] F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
              f"gb_acc={gb_acc:.4f}  gb_f1={gb_f1:.4f}")
        wandb.log({
            f"F1/{split_name}":        f1,
            f"Precision/{split_name}": precision,
            f"Recall/{split_name}":    recall,
            f"GB_F1/{split_name}":     gb_f1,
            f"GB_Acc/{split_name}":    gb_acc,
        })
        return gb_acc, gb_f1, precision, recall, f1

    # ── Pre-training baseline ────────────────────────────────────────────────
    print("Pre-training evaluation (step 0)...")
    model.eval()
    _, _, _, _, val_f1_0  = evaluate(val_loader,  "Val")
    _, _, _, _, test_f1_0 = evaluate(test_loader, "Test")
    wandb.log({"global_step": 0, "F1/Val": val_f1_0, "F1/Test": test_f1_0})

    # ── Training loop ────────────────────────────────────────────────────────
    print("\nStarting training...")
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    start_wall       = time.time()
    global_step      = 0
    running_loss_sum = 0.0
    running_loss_cnt = 0
    best_val_f1      = -float('inf')
    best_test_f1     = -float('inf')
    best_val_step    = 0
    best_test_step   = 0

    with tqdm(total=total_steps, desc="Training") as pbar:
        for epoch in range(1, config["epochs"] + 1):
            start_idx    = ((epoch - 1) * window_size) % num_train_total
            indices      = [(start_idx + i) % num_train_total for i in range(window_size)]
            train_loader = DataLoader(
                torch.utils.data.Subset(train_dataset, indices),
                batch_size=config["batch_size"], shuffle=True, num_workers=4, pin_memory=True,
            )
            print(f'\nEpoch {epoch} — window_start={start_idx}  '
                  f'samples={len(indices):,}  batches={len(train_loader)}')
            model.train()

            for batch_idx, data in enumerate(train_loader):
                data = data.to(device)

                lr = trapezoidal_lr_schedule(global_step, config['lr'], config['lr'] * 0.1,
                                             warmup_steps, cool_steps, total_steps)
                for pg in optimizer.param_groups: pg['lr'] = lr

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out  = run_forward(model, data, config)
                    loss = criterion(out.squeeze(), data.y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()

                global_step      += 1
                running_loss_sum += loss.item()
                running_loss_cnt += 1

                wandb.log({"Loss/train_step": loss.item(), "LR/adamw": lr, "global_step": global_step})
                pbar.update(1)
                pbar.set_postfix({'ep': epoch, 'loss': f'{loss.item():.4f}', 'lr': f'{lr:.5f}'})

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
                        torch.save({'global_step': global_step, 'epoch': epoch,
                                    'model_state': model.state_dict(),
                                    'best_test_f1': best_test_f1, 'config': config}, best_model_path)
                        print(f"  >> New Best Test F1: {best_test_f1:.4f}  (step {best_test_step})")

                    wandb.log({
                        "global_step": global_step, "epoch": epoch,
                        "Loss/train_avg": avg_train_loss,
                        "F1/Val": val_f1,           "F1/Test": test_f1,
                        "Best/Val_F1": best_val_f1, "Best/Test_F1": best_test_f1,
                        "Best/val_step": best_val_step, "Best/test_step": best_test_step,
                    })
                    model.train()

    # ── Final eval & logging ─────────────────────────────────────────────────
    print("\nFinal evaluation...")
    model.eval()
    _, _, _, _, final_val_f1  = evaluate(val_loader,  "Val")
    _, _, _, _, final_test_f1 = evaluate(test_loader, "Test")

    peak_mem   = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"Peak CUDA memory : {peak_mem:.2f} GiB  |  Total time: {total_time:.2f} s")
    print(f"Best Val F1: {best_val_f1:.4f}  Best Test F1: {best_test_f1:.4f}")

    wandb.log({
        "System/peak_cuda_memory_gb": peak_mem,   "System/total_runtime_sec": total_time,
        "System/total_parameters":    total_params,
        "Best/Val_F1":  best_val_f1,  "Best/Test_F1":  best_test_f1,
        "Best/val_step": best_val_step, "Best/test_step": best_test_step,
        "Final/Val_F1": final_val_f1, "Final/Test_F1": final_test_f1,
    })

    os.makedirs(config["dataset_name"], exist_ok=True)
    csv_path = f"{config['dataset_name']}/results_{WANDB_PROJECT}.csv"
    row = {
        "run_name": run_name, "dataset": config["dataset_name"], "seed": config["seed"],
        "hidden_dim": config["hidden_dim"], "layers": config["layers"], "num_heads": config["num_heads"],
        "use_pe": config["use_pe"], "pe_type": config["pe_type"], "pe_dim": config["pe_dim"],
        "batch_size": config["batch_size"], "lr": config["lr"],
        "epochs": config["epochs"], "dropout": config["dropout"],
        "train_subset_ratio": config["train_subset_ratio"],
        "eval_every_steps": eval_every_steps,
        "best_val_f1": round(best_val_f1, 4),   "best_test_f1": round(best_test_f1, 4),
        "best_val_step": best_val_step,
        "final_val_f1": round(final_val_f1, 4), "final_test_f1": round(final_test_f1, 4),
        "total_params": total_params,            "runtime_sec": round(total_time, 1),
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists: w.writeheader()
        w.writerow(row)
    print(f"Results appended to {csv_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
