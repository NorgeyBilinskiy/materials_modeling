"""
Data preprocessing module for crystal structures.
Converts crystal structures to graph representations for ML models.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher

logger = logging.getLogger(__name__)

class CrystalGraphDataset(Dataset):
    """
    Dataset class for crystal structures converted to graphs.
    """
    
    def __init__(self, data_path: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data directory
            transform: Optional transform to apply
        """
        super().__init__(transform=transform)
        self.data_path = data_path
        self.structures = []
        self.targets = []
        self.load_data()
    
    def load_data(self):
        """Load crystal structures and targets from data directory."""
        # Load training data
        training_file = os.path.join(self.data_path, "training_data.json")
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                data = json.load(f)
            
            for item in data:
                # Create structure from lattice parameter
                structure = self.create_nacl_structure(item["lattice_parameter"])
                self.structures.append(structure)
                self.targets.append(item["formation_energy"])
        
        logger.info(f"Loaded {len(self.structures)} crystal structures")
    
    def create_nacl_structure(self, lattice_parameter: float) -> Structure:
        """
        Create NaCl structure with given lattice parameter.
        
        Args:
            lattice_parameter: Lattice parameter in Angstroms
            
        Returns:
            pymatgen Structure object
        """
        from pymatgen.core import Lattice
        
        lattice = Lattice.cubic(lattice_parameter)
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        return structure
    
    def len(self):
        return len(self.structures)
    
    def get(self, idx):
        """Get a single crystal graph."""
        structure = self.structures[idx]
        target = self.targets[idx]
        
        # Convert structure to graph
        graph = self.structure_to_graph(structure)
        graph.y = torch.tensor([target], dtype=torch.float)
        
        return graph
    
    def structure_to_graph(self, structure: Structure, cutoff: float = 8.0) -> Data:
        """
        Convert crystal structure to graph representation.
        
        Args:
            structure: pymatgen Structure object
            cutoff: Cutoff distance for neighbor search
            
        Returns:
            PyTorch Geometric Data object
        """
        # Get atomic numbers and positions
        atomic_numbers = [site.specie.Z for site in structure]
        positions = structure.cart_coords
        
        # Create node features (atomic numbers)
        node_features = torch.tensor(atomic_numbers, dtype=torch.long)
        
        # Find neighbors within cutoff distance
        neighbors = structure.get_all_neighbors(cutoff, include_index=True)
        
        # Create edge indices
        edge_indices = []
        edge_attrs = []
        
        for i, site_neighbors in enumerate(neighbors):
            for neighbor, distance, _, _ in site_neighbors:
                # Find neighbor index
                neighbor_idx = None
                for j, site in enumerate(structure):
                    if site == neighbor:
                        neighbor_idx = j
                        break
                
                if neighbor_idx is not None:
                    edge_indices.append([i, neighbor_idx])
                    edge_indices.append([neighbor_idx, i])  # Undirected graph
                    
                    # Edge attributes (distance)
                    edge_attrs.append(distance)
                    edge_attrs.append(distance)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).unsqueeze(-1)
        
        # Create graph data
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(positions, dtype=torch.float)
        )
        
        return graph

def preprocess_data(data_path: str = "data/") -> None:
    """
    Preprocess crystal structure data for machine learning.
    
    Args:
        data_path: Path to the data directory
    """
    logger.info("Starting data preprocessing...")
    
    # Create processed data directory
    processed_path = os.path.join(data_path, "processed")
    os.makedirs(processed_path, exist_ok=True)
    
    # Load training data
    training_file = os.path.join(data_path, "training_data.json")
    if not os.path.exists(training_file):
        logger.error("Training data not found. Run download first.")
        return
    
    with open(training_file, 'r') as f:
        training_data = json.load(f)
    
    # Create dataset
    dataset = CrystalGraphDataset(data_path)
    
    # Save processed dataset
    torch.save(dataset, os.path.join(processed_path, "crystal_graphs.pt"))
    
    # Create train/val/test split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    # Save splits
    torch.save(train_dataset, os.path.join(processed_path, "train.pt"))
    torch.save(val_dataset, os.path.join(processed_path, "val.pt"))
    torch.save(test_dataset, os.path.join(processed_path, "test.pt"))
    
    # Save metadata
    metadata = {
        "total_samples": len(dataset),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "node_features": 1,  # Atomic numbers
        "edge_features": 1,  # Distances
        "target_mean": np.mean([d.y.item() for d in dataset]),
        "target_std": np.std([d.y.item() for d in dataset])
    }
    
    with open(os.path.join(processed_path, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Preprocessing completed. Created {len(dataset)} graph samples.")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

def get_element_embeddings() -> Dict[int, List[float]]:
    """
    Get element embeddings for atomic numbers.
    
    Returns:
        Dictionary mapping atomic numbers to embeddings
    """
    # Simple one-hot encoding for common elements
    elements = {
        1: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # H
        3: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # Li
        6: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # C
        7: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # N
        8: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # O
        9: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # F
        11: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], # Na
        12: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], # Mg
        15: [0, 0, 0, 0, 0, 0, 0, 0, 1, 0], # P
        16: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], # S
        17: [0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0], # Cl
    }
    
    return elements

def create_graph_features(structure: Structure) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create graph features from crystal structure.
    
    Args:
        structure: pymatgen Structure object
        
    Returns:
        Tuple of (node_features, edge_index, edge_features)
    """
    # Get atomic numbers
    atomic_numbers = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)
    
    # Get positions
    positions = torch.tensor(structure.cart_coords, dtype=torch.float)
    
    # Find neighbors
    neighbors = structure.get_all_neighbors(6.0, include_index=True)
    
    # Create edge indices and features
    edge_indices = []
    edge_features = []
    
    for i, site_neighbors in enumerate(neighbors):
        for neighbor, distance, _, _ in site_neighbors:
            # Find neighbor index
            for j, site in enumerate(structure):
                if site == neighbor:
                    edge_indices.append([i, j])
                    edge_features.append([distance])
                    break
    
    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float)
    
    return atomic_numbers, edge_index, edge_attr
