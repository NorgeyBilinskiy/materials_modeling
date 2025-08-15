"""
Training module for SchNet model.
"""

import os
import logging
from typing import Dict, Any

import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader

from .model import create_schnet_model
from ...utils import set_random_seeds
from ...utils.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


def train_schnet(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    data_path: str = "data/",
    model_save_path: str = "models/schnet/",
    device: str = None,
    hidden_channels: int = 64,
    num_interactions: int = 3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    random_seed: int = 42,
    early_stopping_patience: int = 20,
    lr_scheduler_patience: int = 10,
    lr_scheduler_factor: float = 0.5,
) -> Dict[str, Any]:
    """
    Train SchNet model for NaCl formation energy prediction.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        data_path: Path to data directory
        model_save_path: Path to save trained model
        device: Device to use for training
        hidden_channels: Number of hidden channels
        num_interactions: Number of interaction blocks
        dropout: Dropout rate
        weight_decay: Weight decay for optimizer
        random_seed: Random seed for reproducibility
        early_stopping_patience: Number of epochs without improvement before early stopping
        lr_scheduler_patience: Patience for ReduceLROnPlateau
        lr_scheduler_factor: LR decay factor for ReduceLROnPlateau
    """
    logger.info("Starting SchNet training...")

    # Set random seeds for reproducibility
    set_random_seeds(random_seed)

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
    model = create_schnet_model(
        hidden_channels=hidden_channels,
        num_interactions=num_interactions,
        dropout=dropout,
    ).to(device)

    # Create loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    logger.info(f"Training for {epochs} epochs...")

    epochs_without_improve = 0

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

        # Step LR scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            epochs_without_improve = 0
            hparams = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_channels": hidden_channels,
                "num_interactions": num_interactions,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "early_stopping_patience": early_stopping_patience,
                "lr_scheduler_patience": lr_scheduler_patience,
                "lr_scheduler_factor": lr_scheduler_factor,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "history": history,
                    "hparams": hparams,
                },
                os.path.join(model_save_path, "best_model.pth"),
            )
        else:
            epochs_without_improve += 1

        # Early stopping
        if epochs_without_improve >= early_stopping_patience:
            logger.info(
                f"Early stopping at epoch {epoch + 1} (no improvement for {early_stopping_patience} epochs)"
            )
            break

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

    logger.info("SchNet training completed!")
    # Evaluate on test set and compute detailed metrics
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for batch in GeometricDataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        ):
            batch = batch.to(device)
            outputs = model(batch)
            y_pred_all.append(outputs.detach().cpu().view(-1))
            y_true_all.append(batch.y.detach().cpu().view(-1))

    if y_true_all and y_pred_all:
        import torch as _torch

        y_true_cat = _torch.cat(y_true_all).numpy()
        y_pred_cat = _torch.cat(y_pred_all).numpy()
        test_metrics = compute_regression_metrics(y_true_cat, y_pred_cat)
        history["test_metrics"] = test_metrics
        logger.info(
            "Test metrics - RMSE: %.6f, MAE: %.6f, R2: %.6f, MSE: %.6f"
            % (
                test_metrics.get("rmse", float("nan")),
                test_metrics.get("mae", float("nan")),
                test_metrics.get("r2", float("nan")),
                test_metrics.get("mse", float("nan")),
            )
        )
    else:
        logger.warning("Could not compute test metrics: empty predictions or targets")
    # Save training history and hyperparameters
    try:
        hparams = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_channels": hidden_channels,
            "num_interactions": num_interactions,
            "dropout": dropout,
            "weight_decay": weight_decay,
        }
        import json

        with open(os.path.join(model_save_path, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(model_save_path, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)
    except Exception:
        pass

    return history
