"""
SchNet (Schrödinger Network) model implementation.
Simplified version for NaCl formation energy prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SchNet as PyGSchNet
from torch_geometric.data import Data

class SchNet(nn.Module):
    """
    Simplified SchNet model for crystal property prediction.
    """
    
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
        
        # Use PyTorch Geometric's SchNet implementation
        self.schnet = PyGSchNet(
            hidden_channels=hidden_channels,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff
        )
        
        # Additional output layer for regression
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the SchNet model.
        """
        # Use PyG SchNet for feature extraction
        x = self.schnet(data.x, data.pos, data.batch)
        
        # Output layer for regression
        x = self.output_layer(x)
        
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
    Create a SchNet model with default parameters.
    """
    return SchNet(
        hidden_channels=hidden_channels,
        num_filters=num_filters,
        num_interactions=num_interactions,
        num_gaussians=num_gaussians,
        cutoff=cutoff,
        dropout=dropout
    )
