"""
Training module for SchNet model.
"""

import os
import logging
import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from typing import Dict, Any

from .model import create_schnet_model

logger = logging.getLogger(__name__)


def train_schnet(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    data_path: str = "data/",
    model_save_path: str = "models/schnet/",
    device: str = None,
) -> Dict[str, Any]:
    """
    Train SchNet model for NaCl formation energy prediction.
    """
    logger.info("Starting SchNet training...")

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model save directory
    os.makedirs(model_save_path, exist_ok=True)

    # Load datasets (reuse from CGCNN)
    from src.models.cgcnn.train import load_datasets

    train_dataset, val_dataset, test_dataset = load_datasets(data_path)

    # Create data loaders
    train_loader = GeometricDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = create_schnet_model().to(device)

    # Create loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Training history
    history = {"train_loss": [], "val_loss": [], "best_val_loss": float("inf")}

    logger.info(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss /= num_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                outputs = model(batch)
                loss = criterion(outputs, batch.y)
                val_loss += loss.item()
                num_val_batches += 1

        val_loss /= num_val_batches

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                os.path.join(model_save_path, "best_model.pth"),
            )

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    logger.info("SchNet training completed!")
    return history
