import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_scatter import scatter
from torch.nn.init import uniform_

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.utils import negative_sampling

from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
import wandb

import warnings
warnings.filterwarnings("ignore")

# ── FIX: PyTorch 2.6 changed weights_only default to True ────────────────────
# OGB datasets save PyG objects which are rejected by the new default.
# Patch torch.load globally so all downstream calls use weights_only=False.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat
# ─────────────────────────────────────────────────────────────────────────────


# ==========================================
# 0. INLINE LOGGER  (replaces logger.py)
# ==========================================

class Logger:
    """
    Tracks (train, valid, test) Hits@K across runs and epochs.
    Best checkpoint is selected by validation score.
    """
    def __init__(self, runs: int):
        self.runs    = runs
        self.results = [[] for _ in range(runs)]

    def add_result(self, run: int, result: tuple):
        assert len(result) == 3, "Expected (train, valid, test)"
        self.results[run].append(result)

    def best_result(self, run: int):
        """Returns (train, valid, test) at the epoch with best valid score."""
        res    = torch.tensor(self.results[run])   # [epochs, 3]
        argmax = res[:, 1].argmax().item()
        return res[argmax].tolist()

    def print_statistics(self, run: int = None):
        if run is not None:
            trn, val, tst = self.best_result(run)
            print(f"  Run {run+1:02d}: "
                  f"Train {100*trn:.2f}%  "
                  f"Valid {100*val:.2f}%  "
                  f"Test  {100*tst:.2f}%")
        else:
            best = torch.tensor([self.best_result(r) for r in range(self.runs)]) * 100
            print(f"{'='*52}")
            for i, split in enumerate(['Train', 'Valid', 'Test ']):
                v = best[:, i]
                print(f"  {split}: {v.mean():.2f} ± {v.std():.2f}%")
            print(f"{'='*52}")


def build_run_name(args) -> str:
    """Descriptive W&B run name encoding model and key hyperparams."""
    return (
        f"{args.model}"
        f"_h{args.hidden_channels if args.model != 'lpformer' else args.dim}"
        f"_L{args.num_layers if args.model != 'lpformer' else args.trans_layers}"
        f"_lr{args.lr}"
        f"_ep{args.epochs}"
    )


# ==========================================
# 1. SHARED MLP BLOCK
# ==========================================

class MLP(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim, out_dim, dropout=0.0):
        super().__init__()
        self.lins  = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            self.lins.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                self.norms.append(nn.LayerNorm(dims[i + 1]))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lins[-1](x)


# ==========================================
# 2. GCN / SAGE BASELINES
# ==========================================

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj_t)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, adj_t)


class LinkPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return torch.sigmoid(self.lins[-1](x))


# ==========================================
# 3. LPFORMER COMPONENTS
# ==========================================

class LinkTransformerLayer(nn.Module):
    """Single LPFormer attention layer."""
    def __init__(self, dim, train_args, out_dim=None, node_dim=None):
        super().__init__()
        self.num_heads = train_args['num_heads']
        self.head_dim  = dim // self.num_heads
        self.scale     = self.head_dim ** -0.5
        in_dim  = node_dim if node_dim is not None else dim * 2
        out_dim = out_dim  if out_dim  is not None else dim

        self.q_lin  = nn.Linear(in_dim, self.num_heads * self.head_dim)
        self.k_lin  = nn.Linear(dim,    self.num_heads * self.head_dim)
        self.v_lin  = nn.Linear(dim,    self.num_heads * self.head_dim)
        self.o_lin  = nn.Linear(self.num_heads * self.head_dim, out_dim)
        self.pe_lin = nn.Linear(dim,    self.num_heads * self.head_dim)
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, node_mask, pairwise_feats, X_node, pes,
                extra_mask=None, return_weights=False):
        batch_idx = node_mask[0].long()
        node_idx  = node_mask[1].long()
        BS        = pairwise_feats.size(0)
        H, d      = self.num_heads, self.head_dim

        q = self.q_lin(pairwise_feats)
        k = self.k_lin(X_node[node_idx]) + self.pe_lin(pes)
        v = self.v_lin(X_node[node_idx])

        attn     = (q[batch_idx] * k).view(-1, H, d).sum(-1) * self.scale
        attn_max = scatter(attn, batch_idx, dim=0, dim_size=BS, reduce="max")[batch_idx]
        attn_exp = torch.exp(attn - attn_max)
        attn_sum = scatter(attn_exp, batch_idx, dim=0, dim_size=BS, reduce="sum")[batch_idx] + 1e-12
        attn_w   = attn_exp / attn_sum

        out = scatter(attn_w.unsqueeze(-1) * v.view(-1, H, d),
                      batch_idx, dim=0, dim_size=BS, reduce="sum").view(BS, H * d)
        out = self.norm(
            self.o_lin(out) + pairwise_feats[:, :self.o_lin.out_features]
            if pairwise_feats.size(-1) == self.o_lin.out_features
            else self.o_lin(out)
        )
        return out, attn_w if return_weights else None


class NodeEncoder(nn.Module):
    """Simple GNN node encoder used inside LPFormer."""
    def __init__(self, data, train_args, device="cuda"):
        super().__init__()
        dim        = train_args['dim']
        dropout    = train_args.get('dropout', 0.0)
        num_layers = train_args.get('gnn_layers', 2)
        in_dim     = data['x'].shape[1] if data['x'] is not None else dim

        self.convs = nn.ModuleList([SAGEConv(in_dim, dim)])
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(dim, dim))
        self.dropout = dropout

    def forward(self, x, adj, test_set=False):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, adj)), p=self.dropout, training=self.training)
        return self.convs[-1](x, adj)


class LinkTransformer(nn.Module):
    """LPFormer — adapted for ogbl-ddi (no node features: uses learnable embeddings)."""

    def __init__(self, train_args, data, device="cuda"):
        super().__init__()
        self.data       = data
        self.device     = device
        self.dim        = train_args['dim']
        self.att_drop   = train_args.get('att_drop', 0.0)
        self.num_layers = train_args['trans_layers']
        self.num_nodes  = data['num_nodes']
        self.mask       = "cn"   # CN-only for ogbl-ddi

        self.emb          = nn.Embedding(self.num_nodes, self.dim)
        nn.init.xavier_uniform_(self.emb.weight)
        self.gnn_norm     = nn.LayerNorm(self.dim)
        self.node_encoder = NodeEncoder(data, train_args, device=device)

        self.att_layers = nn.ModuleList()
        att_inner_dim   = self.dim * 2 if self.num_layers > 1 else self.dim
        self.att_layers.append(LinkTransformerLayer(self.dim, train_args, out_dim=att_inner_dim))
        for _ in range(self.num_layers - 2):
            self.att_layers.append(LinkTransformerLayer(self.dim, train_args, node_dim=self.dim))
        if self.num_layers > 1:
            self.att_layers.append(
                LinkTransformerLayer(self.dim, train_args, out_dim=self.dim, node_dim=self.dim)
            )

        self.elementwise_lin = MLP(2, self.dim, self.dim, self.dim)
        self.ppr_encoder_cn  = MLP(2, 2, self.dim, self.dim)
        pairwise_dim         = self.dim * train_args['num_heads'] + 1   # +1 for CN count
        self.pairwise_lin    = MLP(2, pairwise_dim, pairwise_dim, self.dim)
        self.predictor       = MLP(2, self.dim * 2, self.dim, 1,
                                   dropout=train_args.get('dropout', 0.0))

    def get_node_emb(self):
        return self.gnn_norm(self.node_encoder(self.emb.weight, self.data['adj_t']))

    def forward(self, batch, return_weights=False):
        batch      = batch.to(self.device)
        X_node     = self.get_node_emb()
        elem_feats = self.elementwise_lin(X_node[batch[0]] * X_node[batch[1]])
        pair_feats = self._calc_pairwise(batch, X_node, return_weights)
        return torch.sigmoid(self.predictor(torch.cat([elem_feats, pair_feats], dim=-1)))

    def _calc_pairwise(self, batch, X_node, return_weights=False):
        BS      = batch.size(1)
        adj     = self.data['adj_t']
        pair_adj = (torch.index_select(adj, 0, batch[0]) *
                    torch.index_select(adj, 0, batch[1])).coalesce()
        node_ix  = pair_adj.indices()
        src_ppr  = pair_adj.values().clamp(0, 1)

        if node_ix.size(1) == 0:
            return torch.zeros(BS, self.dim, device=self.device)

        pes = (self.ppr_encoder_cn(torch.stack([src_ppr, src_ppr], dim=-1)) * 2)
        pairwise_feats = torch.cat([X_node[batch[0]], X_node[batch[1]]], dim=-1)

        if self.training and self.att_drop > 0:
            keep    = math.ceil(node_ix.size(1) * (1 - self.att_drop))
            sel     = torch.randperm(node_ix.size(1), device=self.device)[:keep]
            node_ix, pes = node_ix[:, sel], pes[sel]

        for layer in self.att_layers:
            pairwise_feats, _ = layer(node_ix, pairwise_feats, X_node, pes,
                                      return_weights=return_weights)

        ones    = torch.ones(node_ix.size(1), device=self.device)
        num_cns = scatter(ones, node_ix[0].long(), dim=0,
                          dim_size=BS, reduce="sum").unsqueeze(-1)
        return self.pairwise_lin(torch.cat([pairwise_feats, num_cns], dim=-1))


# ==========================================
# 4. TRAIN / TEST
# ==========================================

def train_baseline(model, predictor, x, adj_t, split_edge, optimizer, batch_size):
    row, col, _ = adj_t.coo()
    edge_index  = torch.stack([col, row], dim=0)
    model.train(); predictor.train()
    pos_train_edge = split_edge['train']['edge'].to(x.device)
    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        h        = model(x, adj_t)
        edge     = pos_train_edge[perm].t()
        pos_out  = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_edge = negative_sampling(edge_index, num_nodes=x.size(0),
                                     num_neg_samples=perm.size(0), method='dense')
        neg_out  = predictor(h[neg_edge[0]], h[neg_edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss     = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        total_loss     += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)

    return total_loss / total_examples


def train_lpformer(lpformer, split_edge, optimizer, batch_size, device):
    lpformer.train()
    pos_train_edge = split_edge['train']['edge'].to(device)
    total_loss = total_examples = 0

    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        edge     = pos_train_edge[perm].t()
        pos_out  = lpformer(edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        ns       = edge.size(1)
        neg_out  = lpformer(torch.stack([
            torch.randint(0, lpformer.num_nodes, (ns,), device=device),
            torch.randint(0, lpformer.num_nodes, (ns,), device=device),
        ]))
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        loss     = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lpformer.parameters(), 1.0)
        optimizer.step()
        total_loss     += loss.item() * pos_out.size(0)
        total_examples += pos_out.size(0)

    return total_loss / total_examples


@torch.no_grad()
def test_baseline(model, predictor, x, adj_t, split_edge, evaluator, batch_size):
    model.eval(); predictor.eval()
    h = model(x, adj_t)

    def score(edges):
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            e = edges[perm].t()
            preds.append(predictor(h[e[0]], h[e[1]]).squeeze().cpu())
        return torch.cat(preds)

    pt = score(split_edge['eval_train']['edge'].to(x.device))
    pv = score(split_edge['valid']['edge'].to(x.device))
    nv = score(split_edge['valid']['edge_neg'].to(x.device))
    ps = score(split_edge['test']['edge'].to(x.device))
    ns = score(split_edge['test']['edge_neg'].to(x.device))

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        results[f'Hits@{K}'] = (
            evaluator.eval({'y_pred_pos': pt, 'y_pred_neg': nv})[f'hits@{K}'],
            evaluator.eval({'y_pred_pos': pv, 'y_pred_neg': nv})[f'hits@{K}'],
            evaluator.eval({'y_pred_pos': ps, 'y_pred_neg': ns})[f'hits@{K}'],
        )
    return results


@torch.no_grad()
def test_lpformer(lpformer, split_edge, evaluator, batch_size, device):
    lpformer.eval()

    def score(edges):
        preds = []
        for perm in DataLoader(range(edges.size(0)), batch_size):
            e = edges[perm].t().to(device)
            preds.append(lpformer(e).squeeze().cpu())
        return torch.cat(preds)

    pt = score(split_edge['eval_train']['edge'])
    pv = score(split_edge['valid']['edge'])
    nv = score(split_edge['valid']['edge_neg'])
    ps = score(split_edge['test']['edge'])
    ns = score(split_edge['test']['edge_neg'])

    results = {}
    for K in [10, 20, 30]:
        evaluator.K = K
        results[f'Hits@{K}'] = (
            evaluator.eval({'y_pred_pos': pt, 'y_pred_neg': nv})[f'hits@{K}'],
            evaluator.eval({'y_pred_pos': pv, 'y_pred_neg': nv})[f'hits@{K}'],
            evaluator.eval({'y_pred_pos': ps, 'y_pred_neg': ns})[f'hits@{K}'],
        )
    return results


# ==========================================
# 5. MAIN
# ==========================================

def main():
    parser = argparse.ArgumentParser(description='OGBL-DDI — GCN / SAGE / LPFormer')
    parser.add_argument('--device',          type=int,   default=0)
    parser.add_argument('--log_steps',       type=int,   default=1)
    parser.add_argument('--model',           type=str,   default='gcn',
                        choices=['gcn', 'sage', 'lpformer'])
    parser.add_argument('--use_sage',        action='store_true')
    parser.add_argument('--num_layers',      type=int,   default=2)
    parser.add_argument('--hidden_channels', type=int,   default=256)
    parser.add_argument('--dropout',         type=float, default=0.5)
    parser.add_argument('--batch_size',      type=int,   default=64 * 1024)
    parser.add_argument('--lr',              type=float, default=0.005)
    parser.add_argument('--epochs',          type=int,   default=200)
    parser.add_argument('--eval_steps',      type=int,   default=5)
    parser.add_argument('--runs',            type=int,   default=10)
    # LPFormer-specific
    parser.add_argument('--dim',             type=int,   default=256)
    parser.add_argument('--trans_layers',    type=int,   default=2)
    parser.add_argument('--num_heads',       type=int,   default=4)
    parser.add_argument('--gnn_layers',      type=int,   default=2)
    parser.add_argument('--thresh_cn',       type=float, default=0.0)
    parser.add_argument('--thresh_1hop',     type=float, default=1.0)
    parser.add_argument('--thresh_non1hop',  type=float, default=1.0)
    parser.add_argument('--att_drop',        type=float, default=0.0)
    # W&B
    parser.add_argument('--wandb_project',   type=str,   default='ogbl-ddi')
    parser.add_argument('--wandb_entity',    type=str,   default='')
    args = parser.parse_args()
    print(args)

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    dataset    = PygLinkPropPredDataset(name='ogbl-ddi', transform=T.ToSparseTensor())
    data_pyg   = dataset[0]
    adj_t      = data_pyg.adj_t.to(device)
    split_edge = dataset.get_edge_split()

    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    evaluator = Evaluator(name='ogbl-ddi')
    # Inline logger — no external logger.py needed
    loggers = {k: Logger(args.runs) for k in ['Hits@10', 'Hits@20', 'Hits@30']}

    # ── W&B init (single run covering all seeds) ─────────────────────────────
    run_name = build_run_name(args)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        config=vars(args),
    )

    # ── LPFormer ─────────────────────────────────────────────────────────────
    if args.model == 'lpformer':
        data_dict = {
            'x':         None,
            'adj_t':     adj_t,
            'num_nodes': data_pyg.num_nodes,
        }
        train_args = {
            'dim':            args.dim,
            'trans_layers':   args.trans_layers,
            'num_heads':      args.num_heads,
            'gnn_layers':     args.gnn_layers,
            'thresh_cn':      args.thresh_cn,
            'thresh_1hop':    args.thresh_1hop,
            'thresh_non1hop': args.thresh_non1hop,
            'att_drop':       args.att_drop,
            'dropout':        args.dropout,
        }

        for run in range(args.runs):
            lpformer  = LinkTransformer(train_args, data_dict, device=device).to(device)
            optimizer = torch.optim.Adam(lpformer.parameters(), lr=args.lr)

            for epoch in range(1, args.epochs + 1):
                loss = train_lpformer(lpformer, split_edge, optimizer,
                                      args.batch_size, device)

                if epoch % args.eval_steps == 0:
                    results = test_lpformer(lpformer, split_edge, evaluator,
                                            args.batch_size, device)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                    if epoch % args.log_steps == 0:
                        log_dict = {'epoch': epoch, 'run': run, 'train/loss': loss}
                        for key, (trn, val, tst) in results.items():
                            log_dict[f'{key}/train'] = 100 * trn
                            log_dict[f'{key}/valid'] = 100 * val
                            log_dict[f'{key}/test']  = 100 * tst
                            print(f'{key} | Run {run+1:02d} Ep {epoch:03d} '
                                  f'Loss {loss:.4f} '
                                  f'Train {100*trn:.2f}% '
                                  f'Valid {100*val:.2f}% '
                                  f'Test  {100*tst:.2f}%')
                        wandb.log(log_dict)
                        print('---')

            # Per-run best result → W&B summary
            for key in loggers:
                trn, val, tst = loggers[key].best_result(run)
                wandb.log({
                    f'{key}/best_train_run{run}': 100 * trn,
                    f'{key}/best_valid_run{run}': 100 * val,
                    f'{key}/best_test_run{run}':  100 * tst,
                    'run': run,
                })
                print(key); loggers[key].print_statistics(run)

    # ── GCN / SAGE ───────────────────────────────────────────────────────────
    else:
        emb      = nn.Embedding(data_pyg.num_nodes, args.hidden_channels).to(device)
        ModelCls = SAGE if (args.model == 'sage' or args.use_sage) else GCN

        for run in range(args.runs):
            nn.init.xavier_uniform_(emb.weight)
            model = ModelCls(args.hidden_channels, args.hidden_channels,
                             args.hidden_channels, args.num_layers,
                             args.dropout).to(device)
            model.reset_parameters()
            predictor = LinkPredictor(args.hidden_channels, args.hidden_channels,
                                      1, args.num_layers, args.dropout).to(device)
            predictor.reset_parameters()
            optimizer = torch.optim.Adam(
                list(model.parameters()) + list(emb.parameters()) +
                list(predictor.parameters()), lr=args.lr
            )

            for epoch in range(1, args.epochs + 1):
                loss = train_baseline(model, predictor, emb.weight, adj_t,
                                      split_edge, optimizer, args.batch_size)

                if epoch % args.eval_steps == 0:
                    results = test_baseline(model, predictor, emb.weight, adj_t,
                                            split_edge, evaluator, args.batch_size)
                    for key, result in results.items():
                        loggers[key].add_result(run, result)

                    if epoch % args.log_steps == 0:
                        log_dict = {'epoch': epoch, 'run': run, 'train/loss': loss}
                        for key, (trn, val, tst) in results.items():
                            log_dict[f'{key}/train'] = 100 * trn
                            log_dict[f'{key}/valid'] = 100 * val
                            log_dict[f'{key}/test']  = 100 * tst
                            print(f'{key} | Run {run+1:02d} Ep {epoch:02d} '
                                  f'Loss {loss:.4f} '
                                  f'Train {100*trn:.2f}% '
                                  f'Valid {100*val:.2f}% '
                                  f'Test  {100*tst:.2f}%')
                        wandb.log(log_dict)
                        print('---')

            # Per-run best result → W&B summary
            for key in loggers:
                trn, val, tst = loggers[key].best_result(run)
                wandb.log({
                    f'{key}/best_train_run{run}': 100 * trn,
                    f'{key}/best_valid_run{run}': 100 * val,
                    f'{key}/best_test_run{run}':  100 * tst,
                    'run': run,
                })
                print(key); loggers[key].print_statistics(run)

    # ── Final summary across all runs → W&B summary ──────────────────────────
    print("\n===== FINAL RESULTS =====")
    for key in loggers:
        best = torch.tensor([loggers[key].best_result(r)
                             for r in range(args.runs)]) * 100   # [runs, 3]
        wandb.summary[f'{key}/final_train_mean'] = best[:, 0].mean().item()
        wandb.summary[f'{key}/final_valid_mean'] = best[:, 1].mean().item()
        wandb.summary[f'{key}/final_test_mean']  = best[:, 2].mean().item()
        wandb.summary[f'{key}/final_train_std']  = best[:, 0].std().item()
        wandb.summary[f'{key}/final_valid_std']  = best[:, 1].std().item()
        wandb.summary[f'{key}/final_test_std']   = best[:, 2].std().item()
        print(key); loggers[key].print_statistics()

    wandb.finish()


if __name__ == "__main__":
    main()