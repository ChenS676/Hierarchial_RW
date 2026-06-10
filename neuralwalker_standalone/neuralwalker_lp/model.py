import torch
from torch import nn
import torch_geometric.nn as gnn
from .modules.feature_encoder import FeatureEncoder
from .modules.neuralwalker_layer import NeuralWalkerLayer


class NeuralWalkerEncoder(nn.Module):
    """NeuralWalker backbone — outputs node embeddings of shape [N, hidden_size].

    Designed for link prediction: no task-specific head is attached. Pass the
    returned node embeddings to your own decoder (dot-product, MLP, etc.).

    Required batch attributes (produced by RandomWalkSampler):
        x                     – node features
        edge_index            – [2, E]
        edge_attr             – [E, *] (optional)
        walk_node_idx         – [num_walks, walk_length]  long
        walk_edge_idx         – [num_walks, walk_length]  long
        walk_node_mask        – bool mask for padded walk steps
        walk_edge_mask        – bool mask for padded walk steps
        walk_node_id_encoding – [num_walks, walk_length, window_size]  float
        walk_node_adj_encoding– [num_walks, walk_length, window_size-1] float
        batch                 – graph membership vector (for batched graphs)

    Args:
        in_node_dim   : int, or 'atom' (OGB AtomEncoder), or nn.Module
        hidden_size   : hidden dimension (default 64)
        num_layers    : number of NeuralWalkerLayer blocks (default 2)
        walk_encoder  : sequence layer type — 'conv' | 'mamba' | 's4' | 'transformer'
        in_edge_dim   : int, 'bond', or None (no edge features)
        node_embed    : if True uses nn.Embedding, else nn.Linear for node features
        edge_embed    : if True uses nn.Embedding, else nn.Linear for edge features
        walk_length   : length of random walks used during preprocessing (default 50)
        window_size   : walk PE window size (default 8)
        **kwargs      : forwarded to NeuralWalkerLayer (d_state, d_conv, expand, etc.)
    """

    def __init__(
        self,
        in_node_dim,
        hidden_size=64,
        num_layers=2,
        walk_encoder='conv',
        in_edge_dim=None,
        node_embed=True,
        edge_embed=True,
        walk_length=50,
        window_size=8,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.walk_encoder_type = walk_encoder

        self.feature_encoder = FeatureEncoder(
            hidden_size=hidden_size,
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            node_embed=node_embed,
            edge_embed=edge_embed,
        )

        global_mp_type = kwargs.get('global_mp_type', 'vn')
        self.blocks = nn.ModuleList([
            NeuralWalkerLayer(
                hidden_size=hidden_size,
                sequence_layer_type=walk_encoder,
                d_state=kwargs.get('d_state', 16),
                d_conv=kwargs.get('d_conv', 9),
                expand=kwargs.get('expand', 2),
                mlp_ratio=kwargs.get('mlp_ratio', 2),
                use_encoder_norm=kwargs.get('use_encoder_norm', True),
                proj_mlp_ratio=kwargs.get('proj_mlp_ratio', 1),
                walk_length=walk_length,
                use_positional_encoding=kwargs.get('use_positional_encoding', True),
                pos_embed=kwargs.get('walk_pos_embed', False),
                window_size=window_size,
                bidirection=kwargs.get('bidirection', True),
                layer_idx=i,
                local_gnn_type=kwargs.get('local_mp_type', 'gin'),
                global_model_type=None if global_mp_type == 'vn' and i == num_layers - 1 else global_mp_type,
                num_heads=kwargs.get('num_heads', 4),
                dropout=dropout,
                attn_dropout=kwargs.get('attn_dropout', 0.0),
                vn_norm_first=kwargs.get('vn_norm_first', True),
                vn_norm_type=kwargs.get('vn_norm_type', 'batchnorm'),
                vn_pooling=kwargs.get('vn_pooling', 'sum'),
            ) for i in range(num_layers)
        ])

        self.node_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

        self.hidden_size = hidden_size

    def forward(self, batch):
        """Returns node embeddings of shape [N, hidden_size]."""
        batch.walk_pe = torch.cat(
            [batch.walk_node_id_encoding, batch.walk_node_adj_encoding], dim=-1
        )
        batch = self.feature_encoder(batch)
        for block in self.blocks:
            batch = block(batch)
        return self.node_out(batch.x)

    def get_params(self):
        """Returns parameter groups (needed when walk_encoder='s4')."""
        if self.walk_encoder_type == "s4":
            all_parameters = list(self.parameters())
            param_groups = [{"params": [p for p in all_parameters if not hasattr(p, "_optim")]}]
            hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
            hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
            for hp in hps:
                params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
                param_groups.append({"params": params, **hp})
            return param_groups
        return self.parameters()
