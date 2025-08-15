from pathlib import Path

from loguru import logger


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
