import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, LayerNorm
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import graphbench
from tqdm import tqdm
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.utils import to_undirected

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        """
        Focal Loss for binary classification to handle class imbalance.
        
        Args:
            alpha (float): Weighting factor in range (0,1) to balance positive/negative examples.
            gamma (float): Focusing parameter for modulating loss. Higher gamma increases focus on hard examples.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of raw logits (not probabilities) with shape (Batch,) or (Batch, 1)
            targets: Tensor of binary labels {0, 1} with shape (Batch,) or (Batch, 1)
        
        Returns:
            loss: Scalar tensor if reduction='mean' or 'sum', otherwise tensor of shape (Batch,)
        """
        # Flatten for consistency
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate standard Binary Cross Entropy
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate probabilities
        pt = torch.exp(-BCE_loss) 
        
        
        
        # Apply Focal Loss formula: alpha * (1-pt)^gamma * BCE
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
    
class AddUndirectedContext(object):
    def __call__(self, data):
        # 1. Generate the Undirected Graph for Message Passing
        # This duplicates edges (u->v AND v->u) and attributes
        mp_edge_index, mp_edge_attr = to_undirected(
            data.edge_index, 
            data.edge_attr, 
            num_nodes=data.num_nodes,
            reduce="mean"  # If duplicate edges exist, average their attrs
        )
        
        # 2. Store them as NEW attributes
        # We leave data.edge_index, data.edge_attr, and data.y completely ALONE.
        data.mp_edge_index = mp_edge_index
        data.mp_edge_attr = mp_edge_attr
        
        return data


def log_distribution_plot(model, loader, device, epoch):
    model.eval()
    all_probs = []
    all_labels = []

    # 1. Get Predictions
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            rwse = data.random_walk_pe if hasattr(data, 'random_walk_pe') else None
            
            #out = model(data.x.float(), data.edge_index, data.edge_attr.float(), rwse=rwse)
            out = model(
                x=data.x.float(), 
                mp_edge_index=data.mp_edge_index, 
                mp_edge_attr=data.mp_edge_attr.float(),
                target_edge_index=data.edge_index,
                target_edge_attr=data.edge_attr.float(),
                rwse=rwse
            )
            probs = torch.sigmoid(out).cpu().numpy().flatten()
            labels = data.y.cpu().numpy().flatten()
            
            all_probs.extend(probs)
            all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 2. Setup Axes
    # Class 0 (Negative) -> Mapped to Negative X-Axis (e.g., -1)
    # Class 1 (Positive) -> Mapped to Positive X-Axis (e.g., +1)
    neg_mask = (all_labels == 0)
    pos_mask = (all_labels == 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Add "Jitter" (random noise) to x-axis so points don't pile up
    # This helps you see the DENSITY of points
    x_neg = np.random.normal(loc=-1.0, scale=0.1, size=neg_mask.sum())
    x_pos = np.random.normal(loc=1.0, scale=0.1, size=pos_mask.sum())

    # 3. Plot
    # Alpha=0.3 makes them transparent, so dark areas = high density
    ax.scatter(x_neg, all_probs[neg_mask], alpha=0.3, color='red', s=10, label='Ground Truth: 0')
    ax.scatter(x_pos, all_probs[pos_mask], alpha=0.3, color='blue', s=10, label='Ground Truth: 1')

    # 4. Formatting
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Decision Boundary')
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(-2, 2)
    ax.set_xticks([-1, 1])
    ax.set_xticklabels(['Negative Axis\n(Class 0)', 'Positive Axis\n(Class 1)'])
    ax.set_ylabel("Predicted Probability")
    ax.set_title(f"Separation Dynamics - Epoch {epoch}")
    ax.legend(loc='lower center')
    
    # 5. Log to WandB
    # This appends the image to the run history
    wandb.log({"separation_plot": wandb.Image(fig)}, commit=False)
    
    # Optional: Save locally if you want to make a GIF later
    # os.makedirs("plots", exist_ok=True)
    # plt.savefig(f"plots/epoch_{epoch}.png")
    
    plt.close(fig)

class ModernGINLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        
        # 1. Structural Mixing (GINE)
        # [ADAPTATION] Replaced nn.Dropout with a proper MLP projection.
        # Standard GIN requires an MLP here to approximate the function.
        gin_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.conv = GINEConv(gin_mlp, train_eps=True)
        
        # 2. Channel Mixing (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), # Expand for FFN
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim), # Project back
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # Block 1: Aggregation
        x_in = x
        x_norm = self.norm1(x)
        h = self.conv(x_norm, edge_index, edge_attr=edge_attr)
        x = x_in + h  # Residual
        
        # Block 2: FFN
        x_in = x
        x_norm = self.norm2(x)
        h = self.ffn(x_norm)
        x = x_in + h  # Residual
        return x

class EdgeDecoderWithFeatures(nn.Module):
    """
    [ADAPTATION] Improved Decoder.
    Combines Node Interaction (Hadamard) with explicit Edge Features.
    """
    def __init__(self, hidden_dim, dropout=0.0):
        super().__init__()
        # Input: (Node_Interaction) + (Edge_Embedding)
        # Node_Interaction = h_src * h_dst (dim)
        # Edge_Embedding = h_edge (dim)
        # Total Input Dim = 2 * hidden_dim
        self.input_dim = hidden_dim * 2
        
        self.lin1 = nn.Linear(self.input_dim, hidden_dim) 
        self.lin2 = nn.Linear(hidden_dim, 1)          
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_src, h_dst, h_edge):
        # 1. Node Interaction (Hadamard Product)
        node_interact = h_src * h_dst
        
        # 2. Fuse with Explicit Edge Embedding
        # This ensures the model uses edge-specific features for the final prediction
        x = torch.cat([node_interact, h_edge], dim=-1)
        
        # 3. MLP Prediction
        x = self.lin1(x)
        x = F.gelu(x)
        x = self.norm(x)
        x = self.dropout(x)
        x = self.lin2(x)
        
        return x

class GINforEdgeClassification(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, config):
        super().__init__()
        
        dim = config["hidden_dim"]
        pe_internal_dim = config.get("pe_hidden_dim", dim * 2) 
        dropout = config["dropout"]
        self.use_pe = config.get("use_pe", False)

        # 1. Encoders (Input -> dim)
        self.node_emb = nn.Sequential(
            nn.Linear(node_in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        self.edge_emb = nn.Sequential(
            nn.Linear(edge_in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        
        # 2. PE Encoder
        if self.use_pe:
            self.pe_emb = nn.Sequential(
                nn.Linear(16, pe_internal_dim),
                nn.GELU(),
                nn.Linear(pe_internal_dim, dim)
            )

        # 3. GIN Processor
        self.layers = nn.ModuleList()
        for _ in range(config["layers"]):
            self.layers.append(ModernGINLayer(dim, dropout))

        self.final_norm = nn.LayerNorm(dim)

        # 4. Decoder [ADAPTATION]
        self.edge_decoder = EdgeDecoderWithFeatures(dim, dropout)

    def forward(self, x, mp_edge_index, mp_edge_attr, target_edge_index, target_edge_attr, rwse=None):
        if x.dim() == 1: x = x.unsqueeze(-1)
        #if target_edge_attr.dim() == 1: target_edge_attr = target_edge_attr.unsqueeze(-1)
        if mp_edge_attr.dim() == 1: mp_edge_attr = mp_edge_attr.unsqueeze(-1)
        h_mp_edges = self.edge_emb(mp_edge_attr)    
        # --- Encode ---
        h_nodes = self.node_emb(x)
        h_edges = self.edge_emb(target_edge_attr)
        # --- Add PE ---
        if self.use_pe and rwse is not None:
            if rwse.dim() == 1: rwse = rwse.unsqueeze(-1)
            h_nodes = h_nodes + self.pe_emb(rwse)

        # --- Process (Message Passing) ---
        for layer in self.layers:
            # We pass h_edges to every layer for conditioning
            h_nodes = layer(h_nodes, mp_edge_index, h_mp_edges)

        h_nodes = self.final_norm(h_nodes)

        # --- Decode ---
        src_idx, dst_idx = target_edge_index
        h_src = h_nodes[src_idx]
        h_dst = h_nodes[dst_idx]
        # [ADAPTATION] Pass edge embeddings to decoder
        out = self.edge_decoder(h_src, h_dst, h_edges)
        
        return out


class FixGraphBenchData(object):
    def __call__(self, data):
        
        real_num_nodes = data.x.size(0) - data.x.sum().item()
        if hasattr(data, 'num_nodes'): del data.num_nodes
        data.num_nodes = real_num_nodes
        
        if data.x.size(0) != real_num_nodes:
            # Resize x to match real_num_nodes if needed
            data.x = torch.zeros(int(real_num_nodes), dtype=torch.float)
        
        if data.edge_index is not None:
            row, col = data.edge_index
            mask = (row < real_num_nodes) & (col < real_num_nodes)
            data.edge_index = data.edge_index[:, mask]
            
            # 1. Update data.y first so we have the correct reference shape
            if data.y is not None and data.y.size(0) == row.size(0):
                data.y = data.y[mask]

            # 2. Set edge_attr to be the same shape as y, with all values 1.0
            if hasattr(data, 'num_edges'): del data.num_edges 
        return data



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_label_distribution(dataset, split_name):
    pos_count = 0
    neg_count = 0
    
    for data in dataset:
        # Sum all 1s and 0s in the labels tensor
        pos_count += (data.y == 1).sum().item()
        neg_count += (data.y == 0).sum().item()
        
    total = pos_count + neg_count
    
    # Avoid division by zero if dataset is empty
    pos_ratio = (pos_count / total * 100) if total > 0 else 0
    
    print(f"{split_name:<10} | Total: {total:<8} | Pos (1): {pos_count:<8} | Neg (0): {neg_count:<8} | Pos Ratio: {pos_ratio:.2f}%")
    

def main():
    config = {
        "dataset_name": "mst_easy",
         # Matched to hidden_dim
        "hidden_dim": 384,       # [cite: 1357]
        "pe_hidden_dim": 768,    # Explicitly setting internal PE dim if desired [cite: 1357]
        "layers": 6,             # [cite: 1357]
        "dropout": 0.1,          # [cite: 1357]
        "batch_size": 256,       # [cite: 1357]
        "lr": 3e-4,              # [cite: 1357]
        "epochs": 10*10,            
        "eval_metric_class": 'algoreas_classification',
        "use_pe": False,
        "train_subset_ratio": 0.1,            
    }

    wandb.init(entity="graph-diffusion-model-link-prediction", project=f"graphbench_{config['dataset_name']}", config=config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #Transforms
    transforms_list = [FixGraphBenchData()]
    transforms_list.append(AddUndirectedContext())
    if config["use_pe"]:
        print("Using Random Walk Positional Encodings (RWSE)...")
        transforms_list.append(T.AddRandomWalkPE(walk_length=16, attr_name='random_walk_pe'))
    
    transform = T.Compose(transforms_list)
    
    # Load Data
    loader = graphbench.Loader(
        root='./data_graphbench', 
        dataset_names=config["dataset_name"],
        transform=transform
    )
    dataset = loader.load()
    
    try:
        train_dataset = dataset[0]['train']
        val_dataset = dataset[0]['valid']
        test_dataset = dataset[0]['test']
    except (TypeError, KeyError):
        train_dataset = dataset
        val_dataset = dataset
        test_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    sample_batch = next(iter(train_loader))
    import pdb;pdb.set_trace()
    node_in_dim = 1 if sample_batch.x.dim() == 1 else sample_batch.x.size(1)
    edge_in_dim = 1 if sample_batch.edge_attr.dim() == 1 else sample_batch.edge_attr.size(1)
    #import pdb; pdb.set_trace()
    # Initialize Model
    model = GINforEdgeClassification(node_in_dim, edge_in_dim, config).to(device)
    total_params = count_parameters(model)
    print(f"\n{'='*40}")
    print(f"Model Config: dim={config['hidden_dim']}, layers={config['layers']}")
    print(f"Total Parameters: {total_params:,}")
    # 1. Unpack source and destination nodes
    num_train_total = len(train_dataset)
    window_size = int(num_train_total * config["train_subset_ratio"])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=0.1, betas=(0.9, 0.999))
    criterion = torch.nn.BCEWithLogitsLoss() #Use HRW Loss Contrastive Loss
    #criterion = FocalLoss(alpha=0.25, gamma=2.0)
    wandb.watch(model, log="all", log_freq=10)

    try:
        evaluator = graphbench.Evaluator(config["eval_metric_class"])
    except:
        evaluator = None

    def train(epoch, pbar):
        model.train()
        total_loss = 0
        for data in train_loader:

            data = data.to(device)
            optimizer.zero_grad()
            
            rwse = data.random_walk_pe if config["use_pe"] and hasattr(data, 'random_walk_pe') else None

            #out = model(data.x.float(), data.edge_index, data.edge_attr.float(), rwse=rwse)
            logits = model(data.x.float(), data.edge_index, data.edge_attr.float(), rwse=rwse)
            
            # FocalLoss expects logits, not probabilities
            targets = data.y.float()
            
            loss = criterion(logits.squeeze(), targets)
            #loss = criterion(out.squeeze(), data.y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            total_loss += loss.item()
            
            # wandb.log({"train_batch_loss": loss.item()})
            # pbar.update(1)
            # pbar.set_postfix({'epoch': epoch, 'loss': f'{loss.item():.4f}'})
            
            # Log to WandB
            wandb.log({
                "train_batch_loss": loss.item()
            })
            #wandb.log({"train_batch_loss": loss.item()})
            pbar.update(1)
            pbar.set_postfix({'epoch': epoch, 'loss': f'{loss.item():.4f}'})
        return total_loss / len(train_loader)

    @torch.no_grad()
    def get_eval_metrics(metrics, evaluator, log_dict, eval_set_name="val"):
        if isinstance(metrics, (list, tuple)):
            if len(metrics) >= 1:
                log_dict[f"{eval_set_name}_{evaluator.metric[0]}"] = metrics[0]
            if len(metrics) >= 2:
                log_dict[f"{eval_set_name}_{evaluator.metric[1]}"] = metrics[1]
        else:
            log_dict[f"{eval_set_name}_{evaluator.metric}"] = metrics
        return log_dict
    
    @torch.no_grad()
    def test(loader):
        model.eval()
        y_true = []
        y_pred = []
        for data in loader:
            data = data.to(device)
            rwse = data.random_walk_pe if config["use_pe"] and hasattr(data, 'random_walk_pe') else None
            #out = model(data.x.float(), data.edge_index, data.edge_attr.float(), rwse=rwse)
            out = model(
                x=data.x.float(), 
                mp_edge_index=data.mp_edge_index, 
                mp_edge_attr=data.mp_edge_attr.float(),
                target_edge_index=data.edge_index,
                target_edge_attr=data.edge_attr.float(),
                rwse=rwse
            )
            pred = (torch.sigmoid(out) > 0.5).long()
            y_true.append(data.y.cpu())
            y_pred.append(pred.cpu())
        
        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
        if y_true.dim() == 1: y_true = y_true.unsqueeze(1)
        if y_pred.dim() == 1: y_pred = y_pred.unsqueeze(1)
    
        if evaluator:
            try:
                # GraphBench evaluator expects (y_true, y_pred)
                return evaluator.evaluate(y_true, y_pred)
            except Exception as e:
                print(f"Evaluator error: {e}")
                return 0.0
        else:
            return 0.0
    
    save_dir = f"{config['dataset_name']}/checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = os.path.join(save_dir, "best_model.pt")
    best_val_score = -float('inf')
    best_test_score = -float('inf')
    best_val_epoch = 0
    
    print("\nStarting Training...")
    total_steps = len(train_loader) * config["epochs"]
    with tqdm(total=total_steps, desc="Total Progress", unit="step") as pbar:
        
        for epoch in range(1, config["epochs"] + 1):
            model.train()
            
            # --- SHIFTING WINDOW LOGIC ---
            start_idx = ((epoch - 1) * window_size) % num_train_total
            # Handle wrap-around for the subset indices
            indices = [(start_idx + i) % num_train_total for i in range(window_size)]
            epoch_subset = torch.utils.data.Subset(train_dataset, indices)
            train_loader = DataLoader(epoch_subset, batch_size=config["batch_size"], shuffle=True)
            
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for data in pbar:
                data = data.to(device)
                optimizer.zero_grad()
                rwse = data.random_walk_pe if hasattr(data, 'random_walk_pe') else None
                
                # out = model(data.x.float(), data.edge_index, data.edge_attr.float(), rwse=rwse)
                
                # loss = criterion(out.view(-1), data.y.float())
                #logits = model(data.x.float(), data.edge_index, data.edge_attr.float(), rwse=rwse)
                logits = model(
                    x=data.x.float(), 
                    mp_edge_index=data.mp_edge_index, 
                    mp_edge_attr=data.mp_edge_attr.float(),
                    target_edge_index=data.edge_index,
                    target_edge_attr=data.edge_attr.float(),
                    rwse=rwse
                )
                 
                # FocalLoss expects logits, not probabilities
                targets = data.y.float()
                
                loss = criterion(logits.squeeze(), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}'
                })
                wandb.log({
                    "train_batch_loss": loss.item()
                })


            # Evaluation
            val_score = test(DataLoader(val_dataset, batch_size=config["batch_size"]))
            test_score = test(DataLoader(test_dataset, batch_size=config["batch_size"]))
            
            print(f"Epoch {epoch} Done. Loss: {total_loss/len(train_loader):.4f} | Val: {val_score} | Test: {test_score}")
            wandb.log({"epoch": epoch, "loss": total_loss/len(train_loader), "val_acc": val_score[0], "val_f1": val_score[1], "test_acc": test_score[0], "test_f1": test_score[1]})
            if best_val_score < val_score[1]:
                best_val_score = val_score[1]
                #best_test_score = test_score[1]
                best_val_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
            if best_test_score < test_score[1]:
                best_test_score = test_score[1]
                print(f"New Best Model Saved at Epoch {epoch} with Val Score: {best_val_score:.4f} and Test Score: {best_test_score:.4f}")
    print(f"\nTraining Complete. Best Validation Score: {best_val_score:.4f} at epoch {best_val_epoch}. \nBest Test Score: {best_test_score:.4f}.")
    wandb.finish()


if __name__ == "__main__":
    main()