"""
Training module for CGCNN model.
"""

import os
import json
import logging

import torch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from tqdm import tqdm
from typing import Dict, Any

from .model import CGCNN, CGCNNLoss, create_cgcnn_model
from ...utils import set_random_seeds
from ...utils.metrics import compute_regression_metrics

logger = logging.getLogger(__name__)


def train_cgcnn(
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    data_path: str = "data/",
    model_save_path: str = "models/cgcnn/",
    device: str = None,
    # tunable model/opt params
    hidden_channels: int = 64,
    num_layers: int = 3,
    dropout: float = 0.2,
    weight_decay: float = 1e-5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Train CGCNN model for NaCl formation energy prediction.

    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        data_path: Path to data directory
        model_save_path: Path to save trained model
        device: Device to use for training
        hidden_channels: Number of hidden channels
        num_layers: Number of layers
        dropout: Dropout rate
        weight_decay: Weight decay for optimizer
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary containing training history
    """
    logger.info("Starting CGCNN training...")

    # Set random seeds for reproducibility
    set_random_seeds(random_seed)

    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model save directory
    os.makedirs(model_save_path, exist_ok=True)

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(data_path)

    # Create data loaders
    train_loader = GeometricDataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = GeometricDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = GeometricDataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Create model
    model = create_cgcnn_model(
        num_node_features=1,
        num_edge_features=1,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    # Create loss function and optimizer
    criterion = CGCNNLoss(loss_type="mse")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_mae": [],
        "val_mae": [],
        "best_val_loss": float("inf"),
        "best_epoch": 0,
    }

    logger.info(f"Training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in train_pbar:
            batch = batch.to(device)

            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch.y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_mae += torch.mean(torch.abs(outputs - batch.y)).item()
            num_batches += 1

            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "mae": f"{torch.mean(torch.abs(outputs - batch.y)).item():.4f}",
                }
            )

        train_loss /= num_batches
        train_mae /= num_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        num_val_batches = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for batch in val_pbar:
                batch = batch.to(device)

                outputs = model(batch)
                loss = criterion(outputs, batch.y)

                val_loss += loss.item()
                val_mae += torch.mean(torch.abs(outputs - batch.y)).item()
                num_val_batches += 1

                val_pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "mae": f"{torch.mean(torch.abs(outputs - batch.y)).item():.4f}",
                    }
                )

        val_loss /= num_val_batches
        val_mae /= num_val_batches

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)

        # Save best model
        if val_loss < history["best_val_loss"]:
            history["best_val_loss"] = val_loss
            history["best_epoch"] = epoch
            hparams = {
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "hidden_channels": hidden_channels,
                "num_layers": num_layers,
                "dropout": dropout,
                "weight_decay": weight_decay,
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

            logger.info(
                f"New best model saved at epoch {epoch + 1} with val_loss: {val_loss:.4f}"
            )

        # Log progress
        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}"
            )

    # Save final model
    hparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "dropout": dropout,
        "weight_decay": weight_decay,
    }
    torch.save(
        {
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "history": history,
            "hparams": hparams,
        },
        os.path.join(model_save_path, "final_model.pth"),
    )

    # Save training history
    with open(os.path.join(model_save_path, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    # Save hyperparameters
    try:
        with open(os.path.join(model_save_path, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=2)
    except Exception:
        pass

    # Evaluate on test set and compute detailed metrics
    y_true_all = []
    y_pred_all = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            y_pred_all.append(outputs.detach().cpu().view(-1))
            y_true_all.append(batch.y.detach().cpu().view(-1))

    if y_true_all and y_pred_all:
        y_true_cat = torch.cat(y_true_all).numpy()
        y_pred_cat = torch.cat(y_pred_all).numpy()
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

    logger.info("CGCNN training completed!")
    return history


def load_datasets(data_path: str):
    """
    Load train, validation, and test datasets.

    Args:
        data_path: Path to data directory

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    processed_path = os.path.join(data_path, "processed")

    # Load datasets
    train_dataset = torch.load(
        os.path.join(processed_path, "train.pt"), weights_only=False
    )
    val_dataset = torch.load(os.path.join(processed_path, "val.pt"), weights_only=False)
    test_dataset = torch.load(
        os.path.join(processed_path, "test.pt"), weights_only=False
    )

    logger.info(
        f"Loaded datasets - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_dataset, val_dataset, test_dataset


def evaluate_model(model: CGCNN, data_loader, criterion, device: str) -> tuple:
    """
    Evaluate model on given data loader.

    Args:
        model: Trained CGCNN model
        data_loader: Data loader for evaluation
        criterion: Loss function
        device: Device to use

    Returns:
        Tuple of (loss, mae)
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)

            outputs = model(batch)
            loss = criterion(outputs, batch.y)

            total_loss += loss.item()
            total_mae += torch.mean(torch.abs(outputs - batch.y)).item()
            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def load_trained_model(model_path: str, device: str = None) -> CGCNN:
    """
    Load a trained CGCNN model.

    Args:
        model_path: Path to saved model
        device: Device to load model on

    Returns:
        Loaded CGCNN model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint to infer hyperparameters
    checkpoint = torch.load(model_path, map_location=device)
    hparams = checkpoint.get("hparams", {})

    # Create model using saved hyperparameters if available
    model = create_cgcnn_model(
        num_node_features=1,
        num_edge_features=1,
        hidden_channels=hparams.get("hidden_channels", 64),
        num_layers=hparams.get("num_layers", 3),
        dropout=hparams.get("dropout", 0.2),
    )
    model.to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    logger.info(f"Loaded CGCNN model from {model_path}")
    return model
