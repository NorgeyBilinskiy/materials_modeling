"""
Prediction module for MPNN model.
"""

import os
import logging

import torch
from pymatgen.core import Structure

from src import create_graph_features

logger = logging.getLogger(__name__)


def predict_mpnn(
    model_path: str = "models/mpnn/best_model.pth",
    data_path: str = "data/",
    device: str = None,
) -> float:
    """
    Make prediction for NaCl formation energy using trained MPNN model.
    """
    logger.info("Starting MPNN prediction...")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Training a new MPNN model...")
        from .train import train_mpnn

        train_mpnn(epochs=50, data_path=data_path)
        model_path = "models/mpnn/best_model.pth"

    # Create model and load weights
    from .model import create_mpnn_model

    checkpoint = torch.load(model_path, map_location=device)
    hparams = checkpoint.get("hparams", {})
    model = create_mpnn_model(
        hidden_channels=hparams.get("hidden_channels", 64),
        num_layers=hparams.get("num_layers", 3),
        dropout=hparams.get("dropout", 0.2),
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create NaCl structure for prediction
    from pymatgen.core import Lattice

    lattice = Lattice.cubic(5.64)
    nacl_structure = Structure(
        lattice=lattice, species=["Na", "Cl"], coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )

    # Convert structure to graph
    x, edge_index, edge_attr, _ = create_graph_features(nacl_structure)

    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Make prediction
    with torch.no_grad():
        graph_data = graph_data.to(device)
        prediction = model(graph_data)
        predicted_energy = prediction.item()

    logger.info(f"MPNN predicted formation energy: {predicted_energy:.4f} eV/atom")

    # Compare with reference value
    reference_energy = -3.6
    error = abs(predicted_energy - reference_energy)
    logger.info(f"Reference value: {reference_energy} eV/atom")
    logger.info(f"Absolute error: {error:.4f} eV/atom")

    return predicted_energy


def predict_multiple_structures(
    model_path: str, structures: list[Structure], device: str = None
) -> list[float]:
    """Predict for multiple structures using a trained MPNN model."""
    logger.info(f"Making MPNN predictions for {len(structures)} structures...")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    from .model import create_mpnn_model

    model = create_mpnn_model()
    model.to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions: list[float] = []
    from torch_geometric.data import Data

    with torch.no_grad():
        for structure in structures:
            x, edge_index, edge_attr, _ = create_graph_features(structure)
            graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graph_data = graph_data.to(device)
            pred = model(graph_data)
            predictions.append(float(pred.item()))

    return predictions
