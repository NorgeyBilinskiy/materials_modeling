import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from dotenv import load_dotenv
from loguru import logger

from .utils import FileValidator, FileHandler, DirectoryValidator


class Config:
    """Configuration class for managing application settings and paths."""
    
    def __init__(self):
        # === Path Configurations ===
        self.BASE_DIR = Path.cwd()
        self.PATH_TO_VALIDATE: Dict[str, Path] = {
            "materialsproject_token": self.BASE_DIR / "settings" / "materialsproject_api" / "token.env",
            "formulas_substances": self.BASE_DIR / "settings" / "materials" / "selection_materials.yaml",
            "predict_substances": self.BASE_DIR / "settings" / "materials" / "predict_material.yaml",
        }

        # === Path Validation ===
        logger.info("Starting validation of configuration file paths...")
        self._validate_config_paths()
        logger.info("All configuration file paths validated successfully.")

        # === Directory Creation ===
        self.temporary_data_dir = self.BASE_DIR / "temporary_data"
        self._ensure_directories_exist()

        # === Environment Configuration ===
        self._load_environment_variables()
        self._validate_configuration()

    def _validate_config_paths(self) -> None:
        """Validate that all required configuration files exist."""
        for name, path in self.PATH_TO_VALIDATE.items():
            try:
                FileValidator.validate_file_path(str(path))
                logger.debug(f"Configuration file '{name}' validated: {path}")
            except Exception as e:
                logger.error(f"Failed to validate configuration file '{name}': {e}")
                raise

    def _ensure_directories_exist(self) -> None:
        """Create necessary directories if they don't exist."""
        try:
            DirectoryValidator.create_directory_if_not_exists(str(self.temporary_data_dir))
            logger.debug(f"Temporary data directory ensured: {self.temporary_data_dir}")
        except Exception as e:
            logger.error(f"Failed to create temporary data directory: {e}")
            raise

    def _load_environment_variables(self) -> None:
        """Load environment variables from configuration files."""
        try:
            load_dotenv(self.PATH_TO_VALIDATE["materialsproject_token"])
            logger.debug("Environment variables loaded from token file")
        except Exception as e:
            logger.warning(f"Failed to load environment variables: {e}")

    def _validate_configuration(self) -> None:
        """Validate that all required configuration values are present."""
        self.API_TOKEN_MATERIALS_PROJECT = os.getenv("MATERIALS_PROJECT_API_KEY")
        
        if not self.API_TOKEN_MATERIALS_PROJECT:
            logger.warning("MATERIALS_PROJECT_API_KEY not found in environment variables")
            self.API_TOKEN_MATERIALS_PROJECT = None
        else:
            logger.debug("Materials Project API key loaded successfully")

    def get_material_project_info(self) -> Optional[str]:
        """Get the Materials Project API token.
        
        Returns:
            Optional[str]: The API token if available, None otherwise
        """
        return self.API_TOKEN_MATERIALS_PROJECT

    def get_temporary_data_path(self) -> Path:
        """Get the path to the temporary data directory.
        
        Returns:
            Path: Path to the temporary data directory
        """
        return self.temporary_data_dir

    def get_materials_list(self) -> List[str]:
        """Get the list of materials from the YAML configuration file.
        
        Returns:
            List[str]: List of material formulas from the configuration file
            
        Raises:
            Exception: If there's an error reading or parsing the YAML file
        """
        try:
            yaml_data = FileHandler.load_yaml(str(self.PATH_TO_VALIDATE["formulas_substances"]))
            materials = yaml_data.get("formulas_substances", [])
            
            if not materials:
                logger.warning("No materials found in the YAML configuration file")
                return []
                
            logger.debug(f"Loaded {len(materials)} materials from configuration: {materials}")
            return materials
            
        except Exception as e:
            logger.error(f"Failed to load materials list from YAML file: {e}")
            raise

    def get_predict_materials_list(self) -> List[str]:
        """Get the list of materials for prediction from the YAML configuration file.
        
        Returns:
            List[str]: List of material formulas for prediction
            
        Raises:
            Exception: If there's an error reading or parsing the YAML file
        """
        try:
            yaml_data = FileHandler.load_yaml(str(self.PATH_TO_VALIDATE["predict_substances"]))
            materials = yaml_data.get("predict_substances", [])
            
            if not materials:
                logger.warning("No materials for prediction found in the YAML configuration file")
                return []
                
            logger.debug(f"Loaded {len(materials)} materials for prediction: {materials}")
            return materials
            
        except Exception as e:
            logger.error(f"Failed to load prediction materials list from YAML file: {e}")
            raise
