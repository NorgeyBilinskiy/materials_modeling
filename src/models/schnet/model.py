"""
SchNet-like model without torch-cluster dependency.
Re-implemented using standard GCN layers to avoid radius_graph at runtime.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class SchNet(nn.Module):
    """Lightweight SchNet replacement using GCN layers."""

    def __init__(
        self,
        hidden_channels: int = 64,
        num_filters: int = 64,
        num_interactions: int = 3,
        num_gaussians: int = 50,
        cutoff: float = 10.0,
        dropout: float = 0.2
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.num_layers = max(1, num_interactions)
        self.dropout_layer = nn.Dropout(dropout)

        # Node embedding from atomic number
        self.node_embedding = nn.Embedding(100, hidden_channels)

        # Stacked GCN layers
        self.conv_layers = nn.ModuleList(
            [GCNConv(hidden_channels, hidden_channels) for _ in range(self.num_layers)]
        )

        # Readout and regression head
        self.pool = global_mean_pool
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.out = nn.Linear(hidden_channels // 4, 1)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embed atomic numbers
        x = self.node_embedding(x.squeeze(-1))

        # Message passing
        for conv in self.conv_layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)

        # Global pooling
        x = self.pool(x, batch)

        # MLP head
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        x = self.out(x)

        return x.squeeze(-1)

def create_schnet_model(
    hidden_channels: int = 64,
    num_filters: int = 64,
    num_interactions: int = 3,
    num_gaussians: int = 50,
    cutoff: float = 10.0,
    dropout: float = 0.2
) -> SchNet:
    """
    Create a SchNet-like model with default parameters.
    """
    return SchNet(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
        dropout=dropout
    )
