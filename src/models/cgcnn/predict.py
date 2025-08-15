"""
Prediction module for CGCNN model.
"""

import os
import logging
from typing import List, Dict


import numpy as np
import torch
from pymatgen.core import Structure

from .train import load_trained_model
from src import create_graph_features

logger = logging.getLogger(__name__)


def predict_cgcnn(
    model_path: str = "models/cgcnn/best_model.pth",
    data_path: str = "data/",
    device: str = None,
) -> float:
    """
    Make prediction for NaCl formation energy using trained CGCNN model.

    Args:
        model_path: Path to trained model
        data_path: Path to data directory
        device: Device to use for prediction

    Returns:
        Predicted formation energy in eV/atom
    """
    logger.info("Starting CGCNN prediction...")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Training a new CGCNN model...")
        from .train import train_cgcnn

        train_cgcnn(epochs=50, data_path=data_path)
        model_path = "models/cgcnn/best_model.pth"

    model = load_trained_model(model_path, device)
    model.eval()

    # Create NaCl structure for prediction
    from pymatgen.core import Lattice

    # Standard NaCl structure
    lattice = Lattice.cubic(5.64)  # Lattice parameter in Angstroms
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

    logger.info(f"CGCNN predicted formation energy: {predicted_energy:.4f} eV/atom")

    # Compare with reference value
    reference_energy = -3.6  # eV/atom
    error = abs(predicted_energy - reference_energy)
    logger.info(f"Reference value: {reference_energy} eV/atom")
    logger.info(f"Absolute error: {error:.4f} eV/atom")

    return predicted_energy


def predict_multiple_structures(
    model_path: str, structures: List[Structure], device: str = None
) -> List[float]:
    """
    Make predictions for multiple crystal structures.

    Args:
        model_path: Path to trained model
        structures: List of pymatgen Structure objects
        device: Device to use for prediction

    Returns:
        List of predicted formation energies
    """
    logger.info(f"Making predictions for {len(structures)} structures...")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_trained_model(model_path, device)
    model.eval()

    predictions = []

    for i, structure in enumerate(structures):
        # Convert structure to graph
        x, edge_index, edge_attr, _ = create_graph_features(structure)

        # Create PyTorch Geometric Data object
        from torch_geometric.data import Data

        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # Make prediction
        with torch.no_grad():
            graph_data = graph_data.to(device)
            prediction = model(graph_data)
            predicted_energy = prediction.item()
            predictions.append(predicted_energy)

        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1}/{len(structures)} structures")

    logger.info(f"Completed predictions for {len(structures)} structures")
    return predictions


def predict_with_uncertainty(
    model_path: str, structure: Structure, num_samples: int = 100, device: str = None
) -> Dict[str, float]:
    """
    Make prediction with uncertainty estimation using Monte Carlo dropout.

    Args:
        model_path: Path to trained model
        structure: pymatgen Structure object
        num_samples: Number of Monte Carlo samples
        device: Device to use for prediction

    Returns:
        Dictionary with mean prediction and uncertainty
    """
    logger.info(
        f"Making prediction with uncertainty estimation ({num_samples} samples)..."
    )

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_trained_model(model_path, device)
    model.train()  # Enable dropout for uncertainty estimation

    # Convert structure to graph
    x, edge_index, edge_attr, _ = create_graph_features(structure)

    # Create PyTorch Geometric Data object
    from torch_geometric.data import Data

    graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Make multiple predictions
    predictions = []
    with torch.no_grad():
        graph_data = graph_data.to(device)
        for _ in range(num_samples):
            prediction = model(graph_data)
            predictions.append(prediction.item())

    # Calculate statistics
    predictions = np.array(predictions)
    mean_prediction = np.mean(predictions)
    std_prediction = np.std(predictions)

    result = {
        "mean": mean_prediction,
        "std": std_prediction,
        "min": np.min(predictions),
        "max": np.max(predictions),
    }

    logger.info(f"Prediction: {mean_prediction:.4f} ± {std_prediction:.4f} eV/atom")
    return result


def evaluate_model_performance(
    model_path: str, data_path: str = "data/", device: str = None
) -> Dict[str, float]:
    """
    Evaluate model performance on test set.

    Args:
        model_path: Path to trained model
        data_path: Path to data directory
        device: Device to use for evaluation

    Returns:
        Dictionary with performance metrics
    """
    logger.info("Evaluating CGCNN model performance...")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_trained_model(model_path, device)
    model.eval()

    # Load test dataset
    from .train import load_datasets

    _, _, test_dataset = load_datasets(data_path)

    from torch_geometric.loader import DataLoader as GeometricDataLoader

    test_loader = GeometricDataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate
    predictions = []
    targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())

    predictions = np.array(predictions)
    targets = np.array(targets)

    # Calculate metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)

    # Calculate R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    metrics = {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}

    logger.info(f"Test MAE: {mae:.4f} eV/atom")
    logger.info(f"Test RMSE: {rmse:.4f} eV/atom")
    logger.info(f"Test R²: {r2:.4f}")

    return metrics
