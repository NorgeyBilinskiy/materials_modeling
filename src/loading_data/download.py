"""
Module for downloading data from Materials Project.

Contains a client class for accessing Materials Project API and helper
functions. The class provides a method that takes a chemical compound
formula (e.g., "NaCl") and returns structured information about found materials.
"""

from typing import Dict, Any, List, Optional, Union

import pandas as pd
from loguru import logger
from pymatgen.ext.matproj import MPRester
from pymatgen.io.cif import CifWriter


class MaterialsProjectClient:
    """
    Client for accessing Materials Project API.

    Provides method `get_compound_data` that returns data
    for materials matching a given formula in convenient format.
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize client.

        Args:
            api_key: API key for Materials Project access.
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("API key cannot be empty")

        self.api_key: str = api_key
        logger.debug("MaterialsProjectClient initialized successfully")

    def get_compound_data(
        self, formula: str, return_format: str = "dataframe"
    ) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Get crystal structures of materials in CIF format for CGCNN models.

        Args:
            formula: Chemical formula, e.g., "NaCl", "SiO2", etc.
            return_format: Return data format:
                          - "dataframe": pandas DataFrame (default)
                          - "dict": list of dictionaries

        Returns:
            Material data in selected format. Each record contains:
            - material_id: material identifier in Materials Project
            - formula: human-readable formula
            - structure_cif: crystal structure in CIF format
        """
        logger.info(f"Requesting crystal structures for formula: {formula}")

        try:
            with MPRester(self.api_key) as mpr:
                entries = mpr.summary.search(formula=formula)

            if not entries:
                logger.warning(f"No materials found for formula '{formula}'")
                if return_format == "dataframe":
                    return pd.DataFrame()
                return []

            results: List[Dict[str, Any]] = []
            for entry in entries:
                if isinstance(entry, dict):
                    material_id = entry.get("material_id")
                    formula_pretty = entry.get("formula_pretty", formula)
                else:
                    material_id = getattr(entry, "material_id", None)
                    formula_pretty = getattr(entry, "formula_pretty", formula)

                if not material_id:
                    logger.warning(f"Could not get material_id for entry: {entry}")
                    continue

                try:
                    with MPRester(self.api_key) as mpr_detail:
                        structure = mpr_detail.get_structure_by_material_id(material_id)

                        cif_text: str = ""
                        if structure is not None:
                            try:
                                cif_text = str(CifWriter(structure))
                            except Exception as err:
                                logger.error(
                                    f"Failed to generate CIF for {material_id}: {err}"
                                )
                                cif_text = ""
                        else:
                            logger.warning(f"Structure not found for {material_id}")

                        results.append(
                            {
                                "material_id": material_id,
                                "formula": formula_pretty,
                                "structure_cif": cif_text,
                            }
                        )

                except Exception as detail_err:
                    logger.warning(
                        f"Failed to get structure for {material_id}: {detail_err}"
                    )
                    results.append(
                        {
                            "material_id": material_id,
                            "formula": formula_pretty,
                            "structure_cif": "",
                        }
                    )

            logger.info(f"Retrieved {len(results)} records for formula '{formula}'")

            if return_format == "dataframe":
                df = pd.DataFrame(results)
                return df
            else:
                return results

        except Exception as e:
            logger.error(f"Error requesting Materials Project: {e}")
            raise

    def get_material_by_id(self, material_id: str) -> Optional[Dict[str, Any]]:
        """
        Get crystal structure of material by its ID for CGCNN models.

        Args:
            material_id: Material ID in Materials Project

        Returns:
            Dictionary with material information or None if not found
        """
        logger.info(f"Requesting crystal structure by ID: {material_id}")

        try:
            with MPRester(self.api_key) as mpr:
                structure = mpr.get_structure_by_material_id(material_id)

                if not structure:
                    logger.warning(f"Structure with ID '{material_id}' not found")
                    return None

                cif_text: str = ""
                try:
                    cif_text = str(CifWriter(structure))
                except Exception as err:
                    logger.error(f"Failed to generate CIF: {err}")
                    cif_text = ""

                return {
                    "material_id": material_id,
                    "structure_cif": cif_text,
                }

        except Exception as e:
            logger.error(f"Error requesting structure by ID: {e}")
            raise

    def search_materials(
        self, criteria: Dict[str, Any], properties: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Search crystal structures of materials by criteria for CGCNN models.

        Args:
            criteria: Dictionary with search criteria
            properties: List of properties to return (if None, basic CGCNN properties used)

        Returns:
            DataFrame with search results
        """
        if properties is None:
            properties = [
                "material_id",
                "formula_pretty",
                "structure_cif",
            ]

        logger.info(f"Searching crystal structures by criteria: {criteria}")

        try:
            with MPRester(self.api_key) as mpr:
                if "formula" in criteria:
                    entries = mpr.summary.search(formula=criteria["formula"])
                elif "material_id" in criteria:
                    entries = mpr.summary.search(material_ids=[criteria["material_id"]])
                else:
                    entries = mpr.summary.search()

            if not entries:
                logger.warning("No crystal structures found for given criteria")
                return pd.DataFrame()

            results = []
            for entry in entries:
                material_id = None
                formula_pretty = None

                if isinstance(entry, dict):
                    material_id = entry.get("material_id")
                    formula_pretty = entry.get("formula_pretty", "")
                else:
                    material_id = getattr(entry, "material_id", None)
                    formula_pretty = getattr(entry, "formula_pretty", "")

                if material_id:
                    try:
                        with MPRester(self.api_key) as mpr_detail:
                            structure = mpr_detail.get_structure_by_material_id(
                                material_id
                            )
                            cif_text = str(CifWriter(structure)) if structure else ""
                    except Exception as e:
                        logger.warning(f"Failed to get CIF for {material_id}: {e}")
                        cif_text = ""

                    result = {
                        "material_id": material_id,
                        "formula_pretty": formula_pretty,
                        "structure_cif": cif_text,
                    }
                    results.append(result)

            df = pd.DataFrame(results)
            logger.info(f"Found {len(df)} crystal structures")
            return df

        except Exception as e:
            logger.error(f"Error searching crystal structures: {e}")
            raise

    def save_structures_as_cif(
        self, materials_data: Dict[str, pd.DataFrame], save_dir: str = "cif_structures"
    ) -> None:
        """
        Save crystal structures as separate .cif files for CGCNN models.

        Args:
            materials_data: Dictionary with material data
            save_dir: Directory for saving .cif files
        """
        from pathlib import Path

        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        total_saved = 0

        for material, data in materials_data.items():
            if data.empty:
                continue

            material_dir = save_path / material.lower()
            material_dir.mkdir(exist_ok=True)

            for _, row in data.iterrows():
                material_id = row.get("material_id", "unknown")
                cif_content = row.get("structure_cif", "")

                if cif_content and cif_content.strip():
                    cif_filename = f"{material_id}.cif"
                    cif_filepath = material_dir / cif_filename

                    try:
                        with open(cif_filepath, "w", encoding="utf-8") as f:
                            f.write(cif_content)
                        total_saved += 1
                    except Exception as e:
                        logger.error(f"Error saving {cif_filepath}: {e}")
                else:
                    logger.warning(f"Empty CIF content for {material_id}")

        logger.info(f"Saved {total_saved} .cif files to {save_path}")
