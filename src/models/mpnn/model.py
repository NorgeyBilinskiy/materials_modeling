"""
MPNN (Message Passing Neural Network) model implementation.
Simplified version for crystal property prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class MPNNConv(MessagePassing):
    """
    Message Passing Neural Network convolution layer.
    """

    def __init__(self, node_channels, edge_channels, hidden_channels):
        super().__init__(aggr="add")

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.hidden_channels = hidden_channels

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * node_channels + edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_channels + hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Create message from source node, target node, and edge features
        message_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.message_mlp(message_input)

    def update(self, aggr_out, x):
        # Update node features
        update_input = torch.cat([x, aggr_out], dim=-1)
        return self.update_mlp(update_input)


class MPNN(nn.Module):
    """
    Message Passing Neural Network for crystal property prediction.
    """

    def __init__(
        self,
        num_node_features: int = 1,
        num_edge_features: int = 1,
        hidden_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
        num_classes: int = 1,
    ):
        super().__init__()

        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout

        # Node embedding layer
        self.node_embedding = nn.Embedding(100, hidden_channels)

        # Edge embedding layer
        self.edge_embedding = nn.Linear(num_edge_features, hidden_channels)

        # MPNN layers
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(
                MPNNConv(hidden_channels, hidden_channels, hidden_channels)
            )

        # Global pooling
        self.pool = global_mean_pool

        # Output layers
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc3 = nn.Linear(hidden_channels // 4, num_classes)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the MPNN model.
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Node embedding
        x = self.node_embedding(x.squeeze(-1))

        # Edge embedding
        edge_attr = self.edge_embedding(edge_attr)

        # MPNN layers
        for conv in self.conv_layers:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.dropout_layer(x)

        # Global pooling
        x = self.pool(x, batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)
        x = self.fc3(x)

        return x.squeeze(-1)


def create_mpnn_model(
    num_node_features: int = 1,
    num_edge_features: int = 1,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
) -> MPNN:
    """
    Create a MPNN model with default parameters.
    """
    return MPNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=1,
    )
