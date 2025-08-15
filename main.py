import sys
import os
import json
from pathlib import Path
from datetime import datetime

from loguru import logger

from src import MaterialsDataManager
from src import prepare_datasets_from_cif
from src import train_and_validate_all_models
from src import run_predictions_for_selected_materials
from src import Config
from src import set_random_seeds


def setup_logging():
    """Setup logging to both console and file."""
    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate timestamp for log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"materials_modeling_{timestamp}.log"

    # Remove default handler and add custom ones
    logger.remove()

    # Console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # File handler with detailed format
    logger.add(
        str(log_file),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
    )

    logger.info(f"Logging setup complete. Log file: {log_file}")
    logger.info(f"Logs directory: {logs_dir.absolute()}")
    return log_file


def save_results_to_json(pred_results: dict, histories: dict, log_file: Path) -> None:
    """Save prediction results and training histories to JSON file with metadata."""
    # Prepare results data
    results_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "log_file": str(log_file),
            "description": "Materials modeling results - predictions and training histories",
        },
        "predictions": {},
        "training_histories": {},
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
                # Extract key metrics from training history
                results_data["training_histories"][model_name] = {
                    "status": "success",
                    "best_val_loss": history.get("best_val_loss", "N/A"),
                    "best_epoch": history.get("best_epoch", "N/A"),
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

    # Save to JSON file
    try:
        with open("result_models.json", "w", encoding="utf-8") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results and training histories saved to result_models.json")
    except Exception as e:
        logger.error(f"Failed to save results to JSON: {e}")


def check_existing_data(cif_structures_dir: str) -> bool:
    """
    Check if there are existing CIF structures in the directory.

    Args:
        cif_structures_dir: Path to cif_structures directory

    Returns:
        bool: True if data exists and is sufficient, False otherwise
    """
    cif_dir = Path(cif_structures_dir)

    if not cif_dir.exists():
        logger.info("CIF structures directory does not exist")
        return False

    # Check if there are any material subdirectories
    material_dirs = [d for d in cif_dir.iterdir() if d.is_dir()]

    if not material_dirs:
        logger.info("No material subdirectories found in CIF structures")
        return False

    # Check if each material directory has CIF files
    total_cif_files = 0
    for material_dir in material_dirs:
        cif_files = list(material_dir.glob("*.cif"))
        if cif_files:
            logger.info(f"Found {len(cif_files)} CIF files for {material_dir.name}")
            total_cif_files += len(cif_files)
        else:
            logger.warning(f"No CIF files found in {material_dir.name}")

    if total_cif_files == 0:
        logger.info("No CIF files found in any material directory")
        return False

    logger.info(
        f"Found existing data: {total_cif_files} CIF files across {len(material_dirs)} materials"
    )
    return True


def main():
    """Main function to execute the materials data pipeline."""
    # Setup logging first
    log_file = setup_logging()

    try:
        logger.info("Starting Materials Data Processing Pipeline")

        # Set random seeds for reproducibility
        cfg = Config()
        reproducibility_config = cfg.get_reproducibility_config()
        if reproducibility_config.get("apply_seed", True):
            seed = reproducibility_config.get("random_seed", 42)
            set_random_seeds(seed)
        else:
            logger.info("Random seeds not applied (disabled in config)")

        # Check if data already exists
        cif_structures_dir = os.path.join("temporary_data", "cif_structures")
        data_exists = check_existing_data(cif_structures_dir)

        if data_exists:
            logger.info("Existing CIF structures found. Skipping data download.")
            materials_data = {
                "existing_data": True
            }  # Mark that we're using existing data
        else:
            logger.info("No existing CIF structures found. Downloading data...")

            logger.info("Initializing MaterialsDataManager...")
            manager = MaterialsDataManager()
            logger.info("MaterialsDataManager initialized successfully")

            logger.info(f"Materials to process: {manager.materials_list}")
            logger.info(f"Data will be saved to: {manager.config.temporary_data_dir}")

            logger.info("Executing data pipeline...")
            materials_data = manager.get_materials_data_with_save()

            if not materials_data:
                logger.error("Failed to download materials data")
                return None

            logger.info("Pipeline completed successfully!")

            materials_with_data = [k for k, v in materials_data.items() if not v.empty]
            materials_without_data = [k for k, v in materials_data.items() if v.empty]

            logger.info(f"Total materials processed: {len(materials_data)}")
            logger.info(f"Materials with data: {len(materials_with_data)}")
            logger.info(f"Materials without data: {len(materials_without_data)}")

            if materials_with_data:
                logger.info(f"Materials with data: {', '.join(materials_with_data)}")

                for material, data in materials_data.items():
                    if not data.empty:
                        logger.info(
                            f"{material}: {len(data)} records, {len(data.columns)} columns"
                        )

                        cif_structures = (
                            data[data["structure_cif"].str.len() > 0]
                            if "structure_cif" in data.columns
                            else data
                        )
                        logger.info(
                            f"   CIF structures available: {len(cif_structures)}"
                        )

            if materials_without_data:
                logger.warning(
                    f"Materials without data: {', '.join(materials_without_data)}"
                )

            logger.info("Crystal structures saved as .cif files for CGCNN models")
            logger.info(
                f"Check directory: {manager.config.temporary_data_dir}/cif_structures/"
            )

        # Prepare datasets from CIFs (this will work with existing or newly downloaded data)
        logger.info("Preparing datasets from CIF files...")
        prepare_datasets_from_cif()

        # Train and validate all models
        logger.info("Training and validating models...")
        histories = train_and_validate_all_models(config=cfg)
        trained_ok = [
            m
            for m, h in histories.items()
            if not isinstance(h, dict) or "error" not in h
        ]
        logger.info(
            f"Models trained: {', '.join(trained_ok) if trained_ok else 'none'}"
        )

        # Run predictions for selected materials
        logger.info("Running predictions for selected materials...")
        pred_results = run_predictions_for_selected_materials()
        if pred_results:
            for model_name, by_material in pred_results.items():
                for material, value in by_material.items():
                    logger.info(
                        f"Prediction [{model_name}] {material}: {value:.4f} eV/atom"
                    )

            # Save results to JSON file
            save_results_to_json(pred_results, histories, log_file)

        return materials_data

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your API token and configuration files")
        return None

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return None


if __name__ == "__main__":
    result = main()

    if result is None:
        logger.error("Script execution failed")
        sys.exit(1)
    elif isinstance(result, dict) and "existing_data" in result:
        logger.info("Script completed successfully using existing data")
        sys.exit(0)
    elif isinstance(result, dict) and not result:
        logger.warning("Script completed but no data was retrieved")
        sys.exit(0)
    else:
        logger.info("Script completed successfully")
        sys.exit(0)
