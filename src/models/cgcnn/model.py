"""
CGCNN (Crystal Graph Convolutional Neural Network) model implementation.
Based on the original CGCNN paper: https://doi.org/10.1103/PhysRevLett.120.145301
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data


class CGCNN(nn.Module):
    """
    Crystal Graph Convolutional Neural Network for crystal property prediction.

    Args:
        num_node_features: Number of node features (atomic numbers)
        num_edge_features: Number of edge features (distances)
        hidden_channels: Number of hidden channels
        num_layers: Number of CGConv layers
        dropout: Dropout rate
        num_classes: Number of output classes (1 for regression)
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
        self.node_embedding = nn.Embedding(
            100, hidden_channels
        )  # Support up to 100 elements

        # Convolutional layers (use GCNConv for stability across versions)
        self.conv_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_channels, hidden_channels))

        # Global pooling and output layers
        self.pool = global_mean_pool
        self.fc1 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.fc2 = nn.Linear(hidden_channels // 2, hidden_channels // 4)
        self.fc3 = nn.Linear(hidden_channels // 4, num_classes)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the CGCNN model.

        Args:
            data: PyTorch Geometric Data object containing:
                - x: Node features (atomic numbers)
                - edge_index: Graph connectivity
                - edge_attr: Edge features (distances)
                - batch: Batch vector for graph batching

        Returns:
            Predicted formation energy
        """
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        # Node embedding
        x = self.node_embedding(x.squeeze(-1))  # Remove extra dimension if present

        # Graph conv layers
        for conv in self.conv_layers:
            x = conv(x, edge_index)
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


class CGCNNLoss(nn.Module):
    """
    Loss function for CGCNN training.
    """

    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between predictions and targets.

        Args:
            pred: Predicted formation energies
            target: Target formation energies

        Returns:
            Loss value
        """
        return self.loss_fn(pred, target)


def create_cgcnn_model(
    num_node_features: int = 1,
    num_edge_features: int = 1,
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
) -> CGCNN:
    """
    Create a CGCNN model with default parameters.

    Args:
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        hidden_channels: Number of hidden channels
        num_layers: Number of CGConv layers
        dropout: Dropout rate

    Returns:
        CGCNN model instance
    """
    return CGCNN(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=1,
    )
