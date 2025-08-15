import os
from loguru import logger

from . import (
    MaterialsDataManager,
    prepare_datasets_from_cif,
    train_and_validate_all_models,
    run_predictions_for_selected_materials,
    Config,
    set_random_seeds,
)
from .logging_setup import setup_logging
from .results_manager import save_results_to_json
from .data_checker import check_existing_data


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

        # Collect and log final metrics from histories
        final_metrics = {}
        for model_name, hist in histories.items():
            if isinstance(hist, dict) and "test_metrics" in hist:
                metrics = hist["test_metrics"]
                final_metrics[model_name] = metrics
                logger.info(
                    f"Final [{model_name}] metrics - RMSE: {metrics.get('rmse'):.6f}, MAE: {metrics.get('mae'):.6f}, R2: {metrics.get('r2'):.6f}, MSE: {metrics.get('mse'):.6f}"
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

            # Save results to JSON file (with final metrics)
            save_results_to_json(
                pred_results, histories, log_file, final_metrics=final_metrics
            )

        return materials_data

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your API token and configuration files")
        return None

    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return None
