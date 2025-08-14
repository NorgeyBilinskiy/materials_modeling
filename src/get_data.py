from pathlib import Path
from typing import Dict

from loguru import logger
import pandas as pd

from .config import Config
from .loading_data import MaterialsProjectClient


class MaterialsDataManager:
    """
    Class for managing materials data from Materials Project API.
    """

    def __init__(self):
        """Initialize materials data manager."""
        self.config = Config()
        self.api_token = self.config.get_material_project_info()

        if not self.api_token:
            raise ValueError(
                "Materials Project API token not configured. Check configuration."
            )

        self.client = MaterialsProjectClient(self.api_token)
        self.materials_list = self.config.get_materials_list()

        if not self.materials_list:
            logger.warning("Materials list is empty in configuration")

    def get_materials_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get data for all materials from configuration from Materials Project.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with material data, where key is substance formula

        Raises:
            Exception: For API request errors
        """
        if not self.materials_list:
            logger.warning("Materials list is empty in configuration")
            return {}

        logger.info(
            f"Retrieving data for {len(self.materials_list)} materials from Materials Project..."
        )
        logger.info(f"Materials: {self.materials_list}")

        materials_data = {}

        try:
            for material in self.materials_list:
                logger.info(f"Getting data for {material}...")

                try:
                    material_data = self.client.get_compound_data(
                        material, return_format="dataframe"
                    )

                    if not material_data.empty:
                        materials_data[material] = material_data
                        logger.info(
                            f"Successfully retrieved {len(material_data)} records for {material}"
                        )

                        logger.info(
                            f"Data columns for {material}: {list(material_data.columns)}"
                        )
                        logger.info(f"Data size for {material}: {material_data.shape}")
                    else:
                        logger.warning(
                            f"No data found for {material} in Materials Project"
                        )
                        materials_data[material] = pd.DataFrame()

                except Exception as e:
                    logger.error(f"Error getting data for {material}: {e}")
                    materials_data[material] = pd.DataFrame()
                    continue

            logger.info(
                f"Data retrieval completed. Successfully retrieved data for {len([k for k, v in materials_data.items() if not v.empty])} materials"
            )
            return materials_data

        except Exception as e:
            logger.error(f"Critical error during data retrieval: {e}")
            raise

    def get_materials_data_with_save(
        self, save_dir: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for all materials and save crystal structures as .cif files for CGCNN models.

        Args:
            save_dir: Directory for saving data (default uses config directory)

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with material data
        """
        materials_data = self.get_materials_data()

        if not materials_data:
            logger.warning("No data to save")
            return {}

        if save_dir is None:
            save_dir = str(self.config.temporary_data_dir)

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save crystal structures as .cif files for CGCNN
        cif_save_dir = save_path / "cif_structures"
        self.client.save_structures_as_cif(materials_data, str(cif_save_dir))

        return materials_data
