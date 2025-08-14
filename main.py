import sys

from loguru import logger

from src import MaterialsDataManager


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
                        logger.info(f"{material}: {len(data)} records, {len(data.columns)} columns")
                        
                        cif_structures = data[data['structure_cif'].str.len() > 0] if 'structure_cif' in data.columns else data
                        logger.info(f"   CIF structures available: {len(cif_structures)}")
            
            if materials_without_data:
                logger.warning(f"Materials without data: {', '.join(materials_without_data)}")
            
            logger.info("Crystal structures saved as .cif files for CGCNN models")
            logger.info(f"Check directory: {manager.config.temporary_data_dir}/cif_structures/")
            
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