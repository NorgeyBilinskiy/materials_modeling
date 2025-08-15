import sys

from loguru import logger

from src import MaterialsDataManager
from src import prepare_datasets_from_cif
from src.train_validate_models import train_and_validate_all_models
from src.predict_models import run_predictions_for_selected_materials


def main():
    """Main function to execute the materials data pipeline."""
    try:
        logger.info("Starting Materials Data Processing Pipeline")

        logger.info("Initializing MaterialsDataManager...")
        manager = MaterialsDataManager()
        logger.info("MaterialsDataManager initialized successfully")

        logger.info(f"Materials to process: {manager.materials_list}")
        logger.info(f"Data will be saved to: {manager.config.temporary_data_dir}")

        logger.info("Executing data pipeline...")
        materials_data = manager.get_materials_data_with_save()

        if materials_data:
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

            # Prepare datasets from CIFs
            logger.info("Preparing datasets from CIF files...")
            prepare_datasets_from_cif()

            # Train and validate all models
            logger.info("Training and validating models...")
            histories = train_and_validate_all_models()
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

            return materials_data

        else:
            logger.warning("No materials data was retrieved")
            return {}

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
    elif isinstance(result, dict) and not result:
        logger.warning("Script completed but no data was retrieved")
        sys.exit(0)
    else:
        logger.info("Script completed successfully")
        sys.exit(0)
