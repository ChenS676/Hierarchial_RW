from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import graphbench
from tqdm import tqdm
import wandb
import os
import math
import argparse
import random
import numpy as np
import csv
import time
from torch_geometric.utils import to_undirected, is_undirected
from torch_scatter import scatter_add
from graphbench.helpers.utils import set_seed

# ==========================================
# 0. UTILS  (identical to HRW script)
# ==========================================

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
        data.x           = torch.ones(n, 1, dtype=torch.float)
        data.mp_edge_attr = torch.ones(mp_edge_index.size(1), 1, dtype=torch.float)
        data.edge_attr    = torch.ones(data.edge_index.size(1), 1, dtype=torch.float)
        return data


# ==========================================
# WANDB HELPERS
# ==========================================

WANDB_ENTITY = "graph-diffusion-model-link-prediction"


def build_run_name(config: dict) -> str:
    """
    GIN_<dataset>_h<dim>_L<layers>_<pe-tag>_s<seed>
    Example: GIN_maxclique_easy_h256_L6_nope_s2025
    """
    pe_tag = f"_{config['pe_type']}{config['pe_dim']}" if config.get("use_pe") else "_nope"
    return (
        f"GIN"
        f"_{config['dataset_name']}"
        f"_h{config['hidden_dim']}"
        f"_L{config['layers']}"
        f"{pe_tag}"
        f"_s{config['seed']}"
    )


def compute_prf1_jaccard(y_true, y_pred):
    TP      = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP      = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN      = ((y_pred == 0) & (y_true == 1)).sum().float()
    pre     = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
    rec     = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
    f1      = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0
    common  = ((y_pred == 1) & (y_true == 1)).sum().item()
    union   = ((y_pred == 1) | (y_true == 1)).sum().item()
    jaccard = common / union if union > 0 else 0.0
    return pre, rec, f1, jaccard


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
# 1. MODEL — GIN
# ==========================================

class ModernGINLayer(nn.Module):
    """GINEConv + pre-norm residual FFN."""
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv  = GINEConv(gin_mlp, train_eps=True)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = x + self.conv(self.norm1(x), edge_index, edge_attr=edge_attr)
        x = x + self.ffn(self.norm2(x))
        return x


class NodeDecoder(nn.Module):
    def __init__(self, hidden_dim, num_classes=2, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, num_classes)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h):
        return self.lin2(self.norm(F.gelu(self.lin1(h))))


class GINForNodeClassification(nn.Module):
    """
    GIN node classifier for max-clique.
    Message passing on the undirected graph (mp_edge_index).
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, config: dict):
        super().__init__()
        dim     = config["hidden_dim"]
        dropout = config["dropout"]
        self.use_pe = config.get("use_pe", False)
        self.pe_type = config.get("pe_type", "rwse")

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

        self.layers     = nn.ModuleList([ModernGINLayer(dim, dropout) for _ in range(config["layers"])])
        self.final_norm = nn.LayerNorm(dim)
        self.classifier = NodeDecoder(dim, num_classes=2, dropout=dropout)

    def forward(self, x, edge_index, edge_attr,
                mp_edge_index=None, mp_edge_attr=None,
                use_pe=False, pe_type=None,
                eigvecs=None, eigvals=None, rwse=None):
        if x.dim() == 1:         x         = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr  = edge_attr.unsqueeze(-1)

        mp_ei = mp_edge_index if mp_edge_index is not None else edge_index
        mp_ea = mp_edge_attr  if mp_edge_attr  is not None else edge_attr
        if mp_ea.dim() == 1: mp_ea = mp_ea.unsqueeze(-1)

        h         = self.node_encoder(x.float())
        h_mp_edge = self.edge_encoder(mp_ea.float())

        if use_pe:
            if pe_type == "rwse" and rwse is not None:
                h = h + self.pe_encoder(rwse.float())
            elif pe_type == "lap" and eigvecs is not None:
                h = h + self.pe_encoder(eigvecs.float())

        for layer in self.layers:
            h = layer(h, mp_ei, h_mp_edge)

        return self.classifier(self.final_norm(h))


# ==========================================
# 2. HELPERS  (identical to HRW)
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


# ==========================================
# 3. MAIN  (identical training pipeline to HRW)
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="GIN — Max Clique Node Classification")

    parser.add_argument("--dataset_name",       type=str,   default="maxclique_easy")
    parser.add_argument("--data_root",          type=str,   default="./data_graphbench")
    parser.add_argument("--seed",               type=int,   default=2025)
    parser.add_argument("--epochs",             type=int,   default=10)
    parser.add_argument("--batch_size",         type=int,   default=256)
    parser.add_argument("--test_batch_size",    type=int,   default=32)
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--layers",             type=int,   default=6)
    parser.add_argument("--dropout",            type=float, default=0.1)
    parser.add_argument("--adam_max_lr",        type=float, default=1e-4)
    parser.add_argument("--weight_decay",       type=float, default=0.0)
    parser.add_argument("--grad_clip_norm",     type=float, default=0.5)
    parser.add_argument("--train_subset_ratio", type=float, default=0.1)
    parser.add_argument("--use_pe",             type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument("--pe_type",            type=str,   default="rwse", choices=["lap", "rwse"])
    parser.add_argument("--pe_dim",             type=int,   default=16)
    parser.add_argument("--eval_metric_class",  type=str,   default='algoreas_classification')

    args   = parser.parse_args()
    config = vars(args)
    import pprint; pprint.pp(config)

    set_seed(config["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    transforms_list = [AddUndirectedContext()]
    if config["use_pe"]:
        if config["pe_type"] == "rwse":
            transforms_list.append(T.AddRandomWalkPE(walk_length=config["pe_dim"], attr_name='rwse'))

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
    model        = GINForNodeClassification(node_in_dim, edge_in_dim, config).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # ── Schedule bookkeeping (identical to HRW) ──────────────────────────────
    num_train_total    = len(train_dataset)
    eval_sample_window = max(1, int(num_train_total * 0.1))
    eval_every_steps   = max(1, round(eval_sample_window / config["batch_size"]))
    window_size        = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch  = math.ceil(window_size / config["batch_size"])
    total_steps        = batches_per_epoch * config["epochs"]

    print(f"Total steps: {total_steps}  eval_every: {eval_every_steps}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['adam_max_lr'], weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss()

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ── W&B ─────────────────────────────────────────────────────────────────
    run_name = build_run_name(config)
    project  = f"aniket_maxclique_{config['dataset_name']}"
    print(f"W&B project: {project}  run: {run_name}")
    wandb.init(entity=WANDB_ENTITY, project=project, name=run_name, config=config)

    checkpoint_dir  = os.path.join(config["dataset_name"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, f"{run_name}_best.pt")

    # ── Evaluate helper (identical structure to HRW) ─────────────────────────
    @torch.no_grad()
    def evaluate(loader, split_name="val"):
        assert not model.training
        y_true, y_pred = [], []
        for data in loader:
            data = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, data, config)
            y_true.append(data.y.cpu())
            y_pred.append(logits.argmax(dim=-1, keepdim=True).long().cpu())

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        pre, rec, f1, jaccard = compute_prf1_jaccard(y_true, y_pred)
        metrics  = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        acc, _   = parse_metrics(metrics)

        print(f"  [{split_name}] f1={f1:.4f}  precision={pre:.4f}  "
              f"recall={rec:.4f}  jaccard={jaccard:.4f}")

        wandb.log({
            f"F1/{split_name}":        f1,
            f"Precision/{split_name}": pre,
            f"Recall/{split_name}":    rec,
            f"Jaccard/{split_name}":   jaccard,
            f"Acc/{split_name}":       acc,
        })
        return acc, f1, pre, rec, jaccard

    # ── Pre-training baseline ────────────────────────────────────────────────
    print("Pre-training evaluation (step 0)...")
    model.eval()
    evaluate(val_loader,  "Val")
    evaluate(test_loader, "Test")
    wandb.log({"global_step": 0})

    # ── Training loop (identical to HRW) ────────────────────────────────────
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
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = run_forward(model, data, config)
                    loss   = criterion(logits, data.y.long())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
                optimizer.step()

                global_step      += 1
                running_loss_sum += loss.item()
                running_loss_cnt += 1

                wandb.log({"Loss/train_step": loss.item(),
                           "LR/adamw": optimizer.param_groups[0]['lr'],
                           "global_step": global_step})
                pbar.update(1)
                pbar.set_postfix({'ep': epoch, 'loss': f'{loss.item():.4f}'})

                if global_step % eval_every_steps == 0:
                    avg_loss         = running_loss_sum / running_loss_cnt
                    running_loss_sum = 0.0
                    running_loss_cnt = 0

                    model.eval()
                    _, val_f1,  *_ = evaluate(val_loader,  "Val")
                    _, test_f1, *_ = evaluate(test_loader, "Test")

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
                        "Loss/train_avg": avg_loss,
                        "F1/Val": val_f1,           "F1/Test": test_f1,
                        "Best/Val_F1": best_val_f1, "Best/Test_F1": best_test_f1,
                        "Best/val_step": best_val_step, "Best/test_step": best_test_step,
                    })
                    model.train()

    # ── Final eval ───────────────────────────────────────────────────────────
    print("\nFinal evaluation...")
    model.eval()
    _, final_val_f1,  *_ = evaluate(val_loader,  "Val")
    _, final_test_f1, *_ = evaluate(test_loader, "Test")

    peak_mem   = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"Peak CUDA memory: {peak_mem:.2f} GiB  |  Total time: {total_time:.2f}s")
    print(f"Best Val F1: {best_val_f1:.4f}  Best Test F1: {best_test_f1:.4f}")

    wandb.log({
        "System/peak_cuda_memory_gb": peak_mem, "System/total_runtime_sec": total_time,
        "System/total_parameters":    total_params,
        "Best/Val_F1": best_val_f1,   "Best/Test_F1": best_test_f1,
        "Final/Val_F1": final_val_f1, "Final/Test_F1": final_test_f1,
    })

    os.makedirs(config["dataset_name"], exist_ok=True)
    csv_path = f"{config['dataset_name']}/hyperparam_sweep_results_new_v3.csv"
    row = {
        "model": "GIN", "dataset": config["dataset_name"], "seed": config["seed"],
        "hidden_dim": config["hidden_dim"], "layers": config["layers"],
        "adam_max_lr": config["adam_max_lr"], "batch_size": config["batch_size"],
        "epochs": config["epochs"], "dropout": config["dropout"],
        "train_subset_ratio": config["train_subset_ratio"],
        "eval_every_steps": eval_every_steps,
        "use_pe": config["use_pe"], "pe_type": config["pe_type"], "pe_dim": config["pe_dim"],
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
    print(f"Logged to {csv_path}")
    wandb.finish()


if __name__ == "__main__":
    main()
