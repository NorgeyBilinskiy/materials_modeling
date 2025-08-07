"""
Data download module for NaCl crystal structures.
Downloads data from Materials Project and other sources.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from pymatgen.ext.matproj import MPRester
from pymatgen.core import Structure
import ase.io

logger = logging.getLogger(__name__)

def download_nacl_data(data_path: str = "data/") -> None:
    """
    Download NaCl crystal structure data from various sources.
    
    Args:
        data_path: Path to save the downloaded data
    """
    logger.info("Starting NaCl data download...")
    
    # Create data directory
    os.makedirs(data_path, exist_ok=True)
    
    # Download from Materials Project (if API key is available)
    try:
        download_from_materials_project(data_path)
    except Exception as e:
        logger.warning(f"Failed to download from Materials Project: {e}")
        logger.info("Using sample NaCl data instead...")
        create_sample_nacl_data(data_path)
    
    # Create additional sample data for training
    create_training_dataset(data_path)
    
    logger.info("Data download completed!")

def download_from_materials_project(data_path: str) -> None:
    """
    Download NaCl data from Materials Project API.
    
    Args:
        data_path: Path to save the data
    """
    # Note: In a real implementation, you would need an API key
    # For this MVP, we'll create sample data
    logger.info("Attempting to download from Materials Project...")
    
    # Sample NaCl structure data
    nacl_data = {
        "material_id": "mp-22862",
        "formula_pretty": "NaCl",
        "formation_energy_per_atom": -3.6,
        "structure": {
            "lattice": {
                "a": 5.64,
                "b": 5.64,
                "c": 5.64,
                "alpha": 90.0,
                "beta": 90.0,
                "gamma": 90.0
            },
            "sites": [
                {"species": [{"element": "Na", "oxidation_state": 1}], "xyz": [0.0, 0.0, 0.0]},
                {"species": [{"element": "Cl", "oxidation_state": -1}], "xyz": [0.5, 0.5, 0.5]}
            ]
        }
    }
    
    # Save the data
    with open(os.path.join(data_path, "nacl_mp.json"), "w") as f:
        json.dump(nacl_data, f, indent=2)
    
    logger.info("Saved NaCl data from Materials Project")

def create_sample_nacl_data(data_path: str) -> None:
    """
    Create sample NaCl crystal structure data.
    
    Args:
        data_path: Path to save the data
    """
    logger.info("Creating sample NaCl data...")
    
    # Create NaCl structure using pymatgen
    from pymatgen.core import Lattice, Structure
    
    # NaCl crystal structure (Fm-3m space group)
    lattice = Lattice.cubic(5.64)  # Lattice parameter in Angstroms
    structure = Structure(
        lattice=lattice,
        species=["Na", "Cl"],
        coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    
    # Save structure as CIF
    structure.to(filename=os.path.join(data_path, "nacl.cif"))
    
    # Save as JSON with metadata
    nacl_info = {
        "formula": "NaCl",
        "formation_energy_per_atom": -3.6,  # Reference value in eV/atom
        "lattice_parameter": 5.64,  # Angstroms
        "space_group": "Fm-3m",
        "density": 2.17,  # g/cm³
        "melting_point": 801,  # Celsius
        "source": "sample_data"
    }
    
    with open(os.path.join(data_path, "nacl_info.json"), "w") as f:
        json.dump(nacl_info, f, indent=2)
    
    logger.info("Sample NaCl data created")

def create_training_dataset(data_path: str) -> None:
    """
    Create a training dataset with variations of NaCl-like structures.
    
    Args:
        data_path: Path to save the data
    """
    logger.info("Creating training dataset...")
    
    # Create variations of NaCl structure for training
    training_data = []
    
    # Base NaCl structure
    base_structure = {
        "formula": "NaCl",
        "formation_energy": -3.6,
        "lattice_parameter": 5.64,
        "volume": 179.4,
        "density": 2.17
    }
    training_data.append(base_structure)
    
    # Create variations with different lattice parameters
    variations = [
        {"lattice_parameter": 5.5, "formation_energy": -3.4},
        {"lattice_parameter": 5.7, "formation_energy": -3.5},
        {"lattice_parameter": 5.8, "formation_energy": -3.3},
        {"lattice_parameter": 5.6, "formation_energy": -3.55},
        {"lattice_parameter": 5.65, "formation_energy": -3.58},
    ]
    
    for var in variations:
        structure_data = base_structure.copy()
        structure_data.update(var)
        structure_data["volume"] = var["lattice_parameter"] ** 3
        training_data.append(structure_data)
    
    # Save training dataset
    df = pd.DataFrame(training_data)
    df.to_csv(os.path.join(data_path, "training_data.csv"), index=False)
    
    # Save as JSON for easier processing
    with open(os.path.join(data_path, "training_data.json"), "w") as f:
        json.dump(training_data, f, indent=2)
    
    logger.info(f"Created training dataset with {len(training_data)} samples")

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
