import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from loguru import logger


def save_results_to_json(
    pred_results: dict,
    histories: dict,
    log_file: Path,
    final_metrics: Dict[str, Any] | None = None,
) -> None:
    """Save prediction results, training histories, and final metrics to JSON file with metadata.

    final_metrics: optional mapping model_name -> metrics dict (rmse, mae, r2, mse)
    """
    # Prepare results data
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "log_file": str(log_file),
            "description": "Materials modeling results - predictions, training histories, and final metrics",
        },
        "predictions": {},
        "training_histories": {},
        "final_metrics": {},
    }

    # Convert predictions to structured format
    if pred_results:
        for model_name, by_material in pred_results.items():
            results_data["predictions"][model_name] = {}
            for material, value in by_material.items():
                results_data["predictions"][model_name][material] = {
                    "value": float(value),
                    "unit": "eV/atom",
                    "formatted": f"{value:.4f} eV/atom",
                }
    else:
        logger.warning("No prediction results to save")

    # Convert training histories to structured format
    if histories:
        for model_name, history in histories.items():
            if isinstance(history, dict) and "error" not in history:
                # Fallbacks for models that don't store best_epoch/best_val_loss explicitly
                val_loss_list = history.get("val_loss") or []
                best_epoch = history.get("best_epoch")
                best_val_loss = history.get("best_val_loss")

                if (best_epoch is None or best_epoch == "N/A") and val_loss_list:
                    try:
                        best_epoch = min(
                            range(len(val_loss_list)), key=lambda i: val_loss_list[i]
                        )
                    except Exception:
                        best_epoch = "N/A"

                if (best_val_loss is None or best_val_loss == "N/A") and val_loss_list:
                    try:
                        best_val_loss = float(min(val_loss_list))
                    except Exception:
                        best_val_loss = "N/A"

                # Extract key metrics from training history
                results_data["training_histories"][model_name] = {
                    "status": "success",
                    "best_val_loss": best_val_loss
                    if best_val_loss is not None
                    else "N/A",
                    "best_epoch": best_epoch if best_epoch is not None else "N/A",
                    "final_train_loss": history.get("train_loss", [])[-1]
                    if history.get("train_loss")
                    else "N/A",
                    "final_val_loss": history.get("val_loss", [])[-1]
                    if history.get("val_loss")
                    else "N/A",
                    "total_epochs": len(history.get("train_loss", [])),
                }
            else:
                results_data["training_histories"][model_name] = {
                    "status": "error",
                    "error_message": str(history.get("error", "Unknown error"))
                    if isinstance(history, dict)
                    else str(history),
                }
    else:
        logger.warning("No training histories to save")

    # Add final metrics if provided
    if final_metrics:
        for model_name, metrics in final_metrics.items():
            try:
                results_data["final_metrics"][model_name] = {
                    "rmse": float(metrics.get("rmse"))
                    if metrics.get("rmse") is not None
                    else None,
                    "mae": float(metrics.get("mae"))
                    if metrics.get("mae") is not None
                    else None,
                    "r2": float(metrics.get("r2"))
                    if metrics.get("r2") is not None
                    else None,
                    "mse": float(metrics.get("mse"))
                    if metrics.get("mse") is not None
                    else None,
                }
            except Exception:
                results_data["final_metrics"][model_name] = metrics
    else:
        logger.warning("No final metrics provided")

    # Save to JSON file
    try:
        with open("result_models.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info("Results, histories and final metrics saved to result_models.json")
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}")
