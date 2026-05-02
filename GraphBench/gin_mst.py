import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import graphbench
from tqdm import tqdm
import wandb
import math
import argparse
import dataclasses
from typing import Optional, Tuple, List
import random
import numpy as np
from torch.optim import AdamW
from torch_geometric.utils import to_undirected
from torch_scatter import scatter_add
import time

# ==========================================
# 0. UTILS
# ==========================================

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification — handles the severe class imbalance
    in MST tasks (MST edges are a small minority of all edges).
    """
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs   = inputs.view(-1)
        targets  = targets.view(-1)
        bce      = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt       = torch.exp(-bce)
        f_loss   = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == 'mean': return f_loss.mean()
        if self.reduction == 'sum':  return f_loss.sum()
        return f_loss


class AddUndirectedContext(object):
    """
    Builds an undirected message-passing graph alongside the original
    directed edge_index/edge_attr, and computes a simple degree feature.

    For MST, the original directed edge_attr (edge weights) is left
    completely untouched — only mp_edge_index / mp_edge_attr are added.
    """
    def __call__(self, data):
        mp_edge_index, mp_edge_attr = to_undirected(
            data.edge_index,
            data.edge_attr,
            num_nodes=data.num_nodes,
            reduce="mean"
        )
        data.mp_edge_index = mp_edge_index
        data.mp_edge_attr  = mp_edge_attr

        row    = mp_edge_index[0]
        degree = scatter_add(
            torch.ones(mp_edge_index.size(1), device=data.edge_index.device),
            row, dim=0, dim_size=data.num_nodes
        )
        if data.x is not None and data.x.dim() > 1:
            data.x = torch.cat([data.x.float(), degree.unsqueeze(-1)], dim=-1)
        else:
            data.x = degree.unsqueeze(-1)
        return data


class FixGraphBenchDataMST(object):
    """
    Cleans up GraphBench data for the MST task.

    Key difference from the generic version: edge weights (edge_attr) are
    PRESERVED because the model must learn which edges are cheapest.
    The original script overwrote edge_attr with ones — that discards the
    only signal that distinguishes MST edges from non-MST edges.
    """
    def __call__(self, data):
        real_num_nodes = int(data.x.size(0) - data.x.sum().item())

        if hasattr(data, 'num_nodes'): del data.num_nodes
        data.num_nodes = real_num_nodes

        data.x = torch.zeros(real_num_nodes, dtype=torch.float)

        if data.edge_index is not None:
            row, col = data.edge_index
            mask     = (row < real_num_nodes) & (col < real_num_nodes)
            data.edge_index = data.edge_index[:, mask]

            if data.y is not None and data.y.size(0) == row.size(0):
                data.y = data.y[mask]

            # ── CRITICAL FOR MST ──────────────────────────────────────────
            # Keep original edge weights; normalise to [0, 1] so the encoder
            # sees a consistent range regardless of graph scale.
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                ea = data.edge_attr[mask]
                if ea.dim() == 1: ea = ea.unsqueeze(-1)
                ea_min, ea_max = ea.min(), ea.max()
                if (ea_max - ea_min).abs() > 1e-6:
                    ea = (ea - ea_min) / (ea_max - ea_min)
                data.edge_attr = ea
            else:
                data.edge_attr = torch.ones(data.edge_index.size(1), 1, dtype=torch.float)
            # ─────────────────────────────────────────────────────────────

            if hasattr(data, 'num_edges'): del data.num_edges
        return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")


def parse_metrics(metrics):
    """Returns (acc, f1) from graphbench evaluator output regardless of format."""
    if isinstance(metrics, dict):
        f1  = metrics.get('f1', metrics.get('F1', 0.0))
        acc = metrics.get('accuracy', metrics.get('acc', f1))
        return acc, f1
    if isinstance(metrics, (list, tuple)):
        acc = metrics[0] if len(metrics) > 0 else 0.0
        f1  = metrics[1] if len(metrics) > 1 else acc
        return acc, f1
    return metrics, metrics


def build_run_name(config: dict) -> str:
    """
    Constructs a human-readable W&B run name encoding key hyperparameters.
    Example: GIN_mst_easy_h256_L6_nope_focal_s2025
    """
    pe_tag   = f"_{config['pe_type']}{config['pe_dim']}" if config["use_pe"] else "_nope"
    loss_tag = "_focal" if config.get("use_focal_loss", True) else "_bce"
    return (
        f"GIN"
        f"_{config['dataset_name']}"
        f"_h{config['hidden_dim']}"
        f"_L{config['layers']}"
        f"{pe_tag}"
        f"{loss_tag}"
        f"_s{config['seed']}"
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_f1_metrics(y_true, y_pred):
    """Returns (precision, recall, f1) from binary tensors."""
    TP = ((y_pred == 1) & (y_true == 1)).sum().float()
    FP = ((y_pred == 1) & (y_true == 0)).sum().float()
    FN = ((y_pred == 0) & (y_true == 1)).sum().float()
    precision = (TP / (TP + FP)).item() if (TP + FP) > 0 else 0.0
    recall    = (TP / (TP + FN)).item() if (TP + FN) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def trapezoidal_lr_schedule(step, max_lr, min_lr, warmup, cool, total):
    if step <= warmup:
        return (step / warmup) * (max_lr - min_lr) + min_lr
    elif step <= total - cool:
        return max_lr
    else:
        return ((total - step) / cool) * (max_lr - min_lr) + min_lr


# ==========================================
# 1. MODEL — GIN
# ==========================================

class ModernGINLayer(nn.Module):
    """
    Single GIN layer with:
      - GINEConv (edge-attribute-aware message passing) for structural mixing
      - SwiGLU-style FFN for channel mixing
      - Pre-norm residual connections (LayerNorm before each sub-layer)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv  = GINEConv(gin_mlp, train_eps=True)
        self.ffn   = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
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


class EdgeDecoderWithFeatures(nn.Module):
    """
    Predicts edge labels from (h_src ⊙ h_dst) concatenated with h_edge.
    For MST the edge weight embedding in h_edge is the primary signal.
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
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


class GINForEdgeClassification(nn.Module):
    """
    GIN-based edge classifier for MST.

    Message passing runs on the UNDIRECTED graph (mp_edge_index / mp_edge_attr)
    for full neighbourhood coverage; the final prediction uses the original
    directed edge features so edge weights flow into the decoder directly.
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, config: dict):
        super().__init__()
        dim     = config["hidden_dim"]
        dropout = config["dropout"]
        self.use_pe = config.get("use_pe", False)

        # ── Encoders ────────────────────────────────────────────────────
        self.node_encoder = nn.Sequential(
            nn.Linear(node_in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_in_dim, dim), nn.GELU(), nn.Linear(dim, dim)
        )

        # ── Optional PE encoder ─────────────────────────────────────────
        if self.use_pe:
            pe_dim = config.get("pe_dim", 16)
            self.pe_encoder = nn.Sequential(
                nn.Linear(pe_dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
            )

        # ── GIN layers ──────────────────────────────────────────────────
        self.layers     = nn.ModuleList([ModernGINLayer(dim, dropout) for _ in range(config["layers"])])
        self.final_norm = nn.LayerNorm(dim)

        # ── Decoder ─────────────────────────────────────────────────────
        self.classifier = EdgeDecoderWithFeatures(dim, dropout)

    def forward(self, x, edge_index, edge_attr,
                mp_edge_index=None, mp_edge_attr=None,
                use_pe=False, pe_type=None, lap_pe=None, rwse=None):
        if x.dim() == 1:         x         = x.unsqueeze(-1)
        if edge_attr.dim() == 1: edge_attr  = edge_attr.unsqueeze(-1)

        # Use undirected graph for message passing if available
        mp_ei = mp_edge_index if mp_edge_index is not None else edge_index
        mp_ea = mp_edge_attr  if mp_edge_attr  is not None else edge_attr
        if mp_ea.dim() == 1: mp_ea = mp_ea.unsqueeze(-1)

        # Encode nodes
        h = self.node_encoder(x)

        # Optional positional encoding
        if use_pe:
            if pe_type == "rwse" and rwse is not None:
                if rwse.dim() == 1: rwse = rwse.unsqueeze(-1)
                h = h + self.pe_encoder(rwse)
            elif pe_type == "lap" and lap_pe is not None:
                h = h + self.pe_encoder(torch.abs(lap_pe))

        # Encode edges for message passing
        h_mp_edge = self.edge_encoder(mp_ea)

        # Message passing on undirected graph
        for layer in self.layers:
            h = layer(h, mp_ei, h_mp_edge)
        h = self.final_norm(h)

        # Decode: use original directed edge features for prediction
        h_edge = self.edge_encoder(edge_attr)
        src, dst = edge_index
        return self.classifier(h[src], h[dst], h_edge)


# ==========================================
# 2. DATA PROCESSING — identical to HRW version
# ==========================================

def run_forward(model, data, config):
    """Single unified forward call — identical signature to HRW version."""
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


# ==========================================
# 3. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="GIN — MST Edge Classification")

    # Dataset
    parser.add_argument("--dataset_name",        type=str,   default="mst_easy",
                        help="mst_easy | mst_hard | bridges_easy | ...")
    parser.add_argument("--data_root",            type=str,   default="./data_graphbench")
    parser.add_argument("--eval_metric_class",    type=str,   default="algoreas_classification")
    # Training
    parser.add_argument("--seed",                 type=int,   default=2025)
    parser.add_argument("--epochs",               type=int,   default=10)
    parser.add_argument("--batch_size",           type=int,   default=256)
    parser.add_argument("--test_batch_size",      type=int,   default=32)
    parser.add_argument("--train_subset_ratio",   type=float, default=0.1)
    # Architecture
    parser.add_argument("--hidden_dim",           type=int,   default=256)
    parser.add_argument("--layers",               type=int,   default=6)
    parser.add_argument("--dropout",              type=float, default=0.1)
    # Optimisation — kept identical to HRW for fair comparison
    parser.add_argument("--lr",                   type=float, default=3e-4,
                        help="AdamW learning rate.")
    parser.add_argument("--weight_decay",         type=float, default=0.1)
    parser.add_argument("--grad_clip_norm",       type=float, default=0.5)
    # PE
    parser.add_argument("--use_pe",               type=bool,  default=False)
    parser.add_argument("--pe_type",              type=str,   default="rwse",
                        choices=["lap", "rwse"])
    parser.add_argument("--pe_dim",               type=int,   default=16)
    # Loss
    parser.add_argument("--use_focal_loss",       type=bool,  default=True,
                        help="FocalLoss (recommended for MST imbalance) vs BCEWithLogitsLoss.")
    parser.add_argument("--focal_alpha",          type=float, default=0.25)
    parser.add_argument("--focal_gamma",          type=float, default=2.0)

    args   = parser.parse_args()
    config = vars(args)

    import pprint
    pprint.pp(config)

    set_seed(config["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Transforms ──────────────────────────────────────────────────────────
    transforms_list = [FixGraphBenchDataMST(), AddUndirectedContext()]
    if config["use_pe"]:
        if config["pe_type"] == "lap":
            transforms_list.append(T.AddLaplacianEigenvectorPE(
                k=config["pe_dim"], attr_name='lap_pe', is_undirected=True))
        if config["pe_type"] == "rwse":
            transforms_list.append(T.AddRandomWalkPE(
                walk_length=config["pe_dim"], attr_name='rwse'))

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

    print(f"Split sizes → Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    val_loader  = DataLoader(val_dataset,  batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"],
                             shuffle=False, num_workers=4, pin_memory=True)

    # Inspect a peek batch to determine input dims
    _peek       = next(iter(DataLoader(train_dataset, batch_size=4, shuffle=False)))
    node_in_dim = 1 if _peek.x.dim() == 1 else _peek.x.size(1)
    edge_in_dim = 1 if _peek.edge_attr.dim() == 1 else _peek.edge_attr.size(1)
    print(f"Input dims → Node: {node_in_dim}  Edge: {edge_in_dim}")

    y_sample = _peek.y.float()
    print(f"Label balance (peek batch) → pos={y_sample.mean():.3f}  "
          f"neg={1 - y_sample.mean():.3f}  total={y_sample.numel()}")

    # ── Model ───────────────────────────────────────────────────────────────
    model        = GINForEdgeClassification(node_in_dim, edge_in_dim, config).to(device)
    total_params = count_parameters(model)
    print(f"Model parameters: {total_params:,}")

    # ── Scheduling bookkeeping — identical to HRW ───────────────────────────
    num_train_total   = len(train_dataset)
    window_size       = int(num_train_total * config["train_subset_ratio"])
    batches_per_epoch = math.ceil(window_size / config["batch_size"])
    total_steps       = batches_per_epoch * config["epochs"]
    warmup_steps      = total_steps // 10
    cool_steps        = int(total_steps * 0.1)

    print(f"\nTotal train samples  : {num_train_total:,}")
    print(f"Window / epoch       : {window_size:,}  ({config['train_subset_ratio']*100:.1f}%)")
    print(f"Batches / epoch      : {batches_per_epoch}")
    print(f"Total steps          : {total_steps}  (warmup={warmup_steps}, cool={cool_steps})")

    # ── Optimizer — AdamW with trapezoidal LR schedule ──────────────────────
    # GIN uses a single AdamW (no Muon needed — no matrix-heavy attention layers).
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.999),
    )

    print(f"\nOptimizer : AdamW  lr={config['lr']}  wd={config['weight_decay']}")
    print(f"Total parameters: {total_params:,}\n")

    # ── Loss ────────────────────────────────────────────────────────────────
    if config["use_focal_loss"]:
        criterion = FocalLoss(alpha=config["focal_alpha"], gamma=config["focal_gamma"])
        print(f"Loss: FocalLoss(alpha={config['focal_alpha']}, gamma={config['focal_gamma']})")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss")

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except Exception:
        evaluator = None

    # ── W&B — identical structure to HRW ────────────────────────────────────
    run_name = build_run_name(config)
    print(f"W&B run name: {run_name}\n")
    wandb.init(
        entity="graph-diffusion-model-link-prediction",
        project=f"graphbench_transformer_{config['dataset_name']}",
        name=run_name,
        config=config,
    )

    # ── Train epoch ──────────────────────────────────────────────────────────
    def train_epoch(epoch, pbar, train_loader):
        model.train()
        total_loss    = 0.0
        global_offset = (epoch - 1) * batches_per_epoch

        for batch_idx, data in enumerate(train_loader):
            data        = data.to(device)
            global_step = global_offset + batch_idx

            # Trapezoidal LR schedule applied to AdamW
            lr = trapezoidal_lr_schedule(
                global_step, config['lr'], config['lr'] * 0.1,
                warmup_steps, cool_steps, total_steps
            )
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out  = run_forward(model, data, config)
                loss = criterion(out.squeeze(), data.y.float())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config["grad_clip_norm"])
            optimizer.step()
            total_loss += loss.item()

            pbar.update(1)
            pbar.set_postfix({
                'epoch': epoch,
                'loss':  f'{loss.item():.4f}',
                'lr':    f'{lr:.5f}',
            })

        return total_loss / len(train_loader)

    # ── Evaluate — identical to HRW version ──────────────────────────────────
    @torch.no_grad()
    def evaluate(loader, split_name):
        model.eval()
        y_true_list, y_pred_list = [], []

        for data in loader:
            data  = data.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = run_forward(model, data, config)
            pred = (torch.sigmoid(out) > 0.5).long()
            y_true_list.append(data.y.cpu())
            y_pred_list.append(pred.cpu())

        y_true = torch.cat(y_true_list)
        y_pred = torch.cat(y_pred_list)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)

        gb_metrics    = evaluator.evaluate(y_true, y_pred) if evaluator else 0.0
        gb_acc, gb_f1 = parse_metrics(gb_metrics)
        precision, recall, f1 = compute_f1_metrics(y_true, y_pred)

        print(f"  [{split_name:4s}] F1={f1:.4f}  P={precision:.4f}  R={recall:.4f}  "
              f"gb_acc={gb_acc:.4f}  gb_f1={gb_f1:.4f}")

        # W&B — grouped by split, only F1-family metrics (identical to HRW)
        wandb.log({
            f"F1/{split_name}":        f1,
            f"Precision/{split_name}": precision,
            f"Recall/{split_name}":    recall,
            f"GB_F1/{split_name}":     gb_f1,
            f"GB_Acc/{split_name}":    gb_acc,
        })
        return f1, precision, recall, gb_acc, gb_f1

    # ── Training loop — identical to HRW ────────────────────────────────────
    print("Starting Training...\n")
    best_val_f1    = -float('inf')
    best_test_f1   = -float('inf')
    best_val_epoch = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start_wall = time.time()

    with tqdm(total=total_steps) as pbar:
        for epoch in range(1, config["epochs"] + 1):
            start_idx    = ((epoch - 1) * window_size) % num_train_total
            indices      = [(start_idx + i) % num_train_total for i in range(window_size)]
            train_loader = DataLoader(
                torch.utils.data.Subset(train_dataset, indices),
                batch_size=config["batch_size"],
                shuffle=True, num_workers=4, pin_memory=True,
            )
            print(f"\nEpoch {epoch}  window_start={start_idx}  "
                  f"samples={len(indices):,}  batches={len(train_loader)}")

            avg_loss = train_epoch(epoch, pbar, train_loader)

            model.eval()
            val_f1,  val_p,  val_r,  val_acc,  val_gbf1  = evaluate(val_loader,  "Val")
            test_f1, test_p, test_r, test_acc, test_gbf1 = evaluate(test_loader, "Test")

            if val_f1 > best_val_f1:
                best_val_f1    = val_f1
                best_val_epoch = epoch
                print(f"  >> New Best Val F1 : {best_val_f1:.4f}  (epoch {best_val_epoch})")

            if test_f1 > best_test_f1:
                best_test_f1 = test_f1
                print(f"  >> New Best Test F1: {best_test_f1:.4f}")

            # Single consolidated epoch log — identical keys to HRW
            wandb.log({
                "epoch":          epoch,
                "Loss/train":     avg_loss,
                "F1/Val":         val_f1,
                "F1/Test":        test_f1,
                "Best/Val_F1":    best_val_f1,
                "Best/Test_F1":   best_test_f1,
                "Best/val_epoch": best_val_epoch,
            })

    print("\nTraining complete.")
    peak_mem   = torch.cuda.max_memory_allocated() / 1024 ** 3 if torch.cuda.is_available() else 0.0
    total_time = time.time() - start_wall
    print(f"Peak CUDA memory : {peak_mem:.2f} GiB")
    print(f"Total time       : {total_time:.2f} s")
    print(f"Best Val F1      : {best_val_f1:.4f} (epoch {best_val_epoch})")
    print(f"Best Test F1     : {best_test_f1:.4f}")

    wandb.log({
        "System/peak_cuda_memory_gb": peak_mem,
        "System/total_runtime_sec":   total_time,
        "System/total_parameters":    total_params,
        "Best/Val_F1":                best_val_f1,
        "Best/Test_F1":               best_test_f1,
        "Best/val_epoch":             best_val_epoch,
    })
    wandb.finish()


if __name__ == "__main__":
    main()