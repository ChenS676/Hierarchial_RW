import torch
from torch_geometric.data import Data
from .wrapper import sample_random_walks


class WalkData(Data):
    """Data wrapper that correctly increments walk indices when batching."""

    def __init__(self, data=None):
        if data is None:
            data_dict = {}
        else:
            data_dict = {key: item for key, item in data}
        super().__init__(**data_dict)

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'walk_node_idx':
            return self.num_nodes
        if key == 'walk_edge_idx':
            return self.edge_index.shape[1]
        return super(WalkData, self).__inc__(key, value, *args, **kwargs)


class RandomWalkSampler:
    """Transform that samples random walks and attaches them to a Data object.

    Use as a pre_transform or in-line transform. The output is a WalkData
    object with these extra attributes:
        walk_node_idx          – [num_walks, walk_length]  long
        walk_edge_idx          – [num_walks, walk_length]  long
        walk_node_mask         – bool, True where padding
        walk_edge_mask         – bool, True where padding
        walk_node_id_encoding  – [num_walks, walk_length, window_size]    float
        walk_node_adj_encoding – [num_walks, walk_length, window_size-1]  float

    Args:
        length      : walk length (default 50)
        sample_rate : fraction of nodes to use as walk roots (default 1.0)
        backtracking: allow the walk to immediately revisit the previous node
        strict      : stop walk at dead ends instead of backtracking
        pad_idx     : padding sentinel value (default -1)
        window_size : size of the local positional-encoding window (default 8)
    """

    def __init__(
        self,
        length=50,
        sample_rate=1.,
        backtracking=False,
        strict=False,
        pad_idx=-1,
        window_size=8,
        **kwargs
    ):
        self.length = length
        self.sample_rate = sample_rate
        self.backtracking = backtracking
        self.strict = strict
        self.pad_idx = pad_idx
        self.window_size = window_size

    def __call__(self, data):
        if not data.is_coalesced():
            data = data.coalesce()
        walk_node_index, walk_edge_index, walk_node_id_encoding, walk_node_adj_encoding = sample_random_walks(
            data.edge_index,
            data.num_nodes,
            self.length,
            self.sample_rate,
            self.backtracking,
            self.strict,
            self.window_size,
            self.pad_idx,
        )

        data.walk_node_idx = torch.from_numpy(walk_node_index)
        data.walk_edge_idx = torch.from_numpy(walk_edge_index)
        data.walk_node_mask = data.walk_node_idx == self.pad_idx
        data.walk_edge_mask = data.walk_edge_idx == self.pad_idx

        data.walk_node_id_encoding = torch.from_numpy(walk_node_id_encoding).float()
        data.walk_node_adj_encoding = torch.from_numpy(walk_node_adj_encoding).float()
        return WalkData(data)
