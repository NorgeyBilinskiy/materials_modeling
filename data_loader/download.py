"""
Data download module for NaCl crystal structures.
Downloads data from Materials Project and other sources.
"""

import json
import os
from typing import Dict, Any

from dotenv import load_dotenv
from loguru import logger
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure

# Load environment variables
load_dotenv("settings/.env")

def download_nacl_data(data_path: str = "data/") -> None:
    """
    Download NaCl crystal structure data from Materials Project.
    
    Args:
        data_path: Path to save the downloaded data
    """
    logger.info("Starting NaCl data download...")
    
    # Create data directory
    os.makedirs(data_path, exist_ok=True)
    
    # Download from Materials Project
    download_from_materials_project(data_path)
    
    logger.info("Data download completed!")

def download_from_materials_project(data_path: str) -> None:
    """
    Download NaCl data from Materials Project API.
    
    Args:
        data_path: Path to save the data
    """
    # Get API key from environment variables
    api_key = get_api_key()
    
    logger.info("Downloading data from Materials Project...")
    
    try:
        # Initialize MPRester with API key
        with MPRester(api_key) as mpr:
            # Search for NaCl materials
            nacl_entries = mpr.query(criteria={"formula_pretty": "NaCl"}, 
                                   properties=["material_id", "formula_pretty", 
                                             "formation_energy_per_atom", "structure"])
            
            if not nacl_entries:
                logger.warning("No NaCl materials found in Materials Project")
                return
            
            # Process and save each NaCl entry
            for entry in nacl_entries:
                material_id = entry["material_id"]
                filename = f"nacl_{material_id}.json"
                
                # Save structure data
                with open(os.path.join(data_path, filename), "w") as f:
                    json.dump(entry, f, indent=2)
                
                # Save structure as CIF file
                structure = entry["structure"]
                cif_filename = f"nacl_{material_id}.cif"
                structure.to(filename=os.path.join(data_path, cif_filename))
                
                logger.info(f"Downloaded and saved {material_id}: {filename}, {cif_filename}")
        
        logger.info(f"Successfully downloaded {len(nacl_entries)} NaCl materials from Materials Project")
        
    except Exception as e:
        logger.error(f"Failed to download from Materials Project: {e}")
        raise





def get_api_key() -> str:
    """
    Get Materials Project API key from environment variables.
    
    Returns:
        API key string
        
    Raises:
        ValueError: If API key is not set or invalid
    """
    api_key = os.getenv("MATERIALS_PROJECT_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "Please set MATERIALS_PROJECT_API_KEY in settings/.env file. "
            "Get your API key from https://materialsproject.org/api"
        )
    return api_key


def get_nacl_reference_data() -> Dict[str, Any]:
    """
    Get reference data for NaCl crystal structure.
    
    Returns:
        Dictionary with reference properties
    """
    return {
        "formula": "NaCl",
        "formation_energy_per_atom": -3.6,  # eV/atom
        "lattice_parameter": 5.64,  # Angstroms
        "space_group": "Fm-3m",
        "crystal_system": "cubic",
        "density": 2.17,  # g/cm³
        "melting_point": 801,  # Celsius
        "boiling_point": 1413,  # Celsius
        "band_gap": 8.9,  # eV
        "bulk_modulus": 23.9,  # GPa
        "shear_modulus": 12.6,  # GPa
        "poisson_ratio": 0.25
    }
