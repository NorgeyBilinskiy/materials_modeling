from typing import Dict, Any

from loguru import logger

from .data_preprocessing import prepare_datasets_from_cif
from .models import (
    train_cgcnn,
    train_megnet,
    train_schnet,
    train_mpnn,
)


def train_and_validate_all_models() -> Dict[str, Any]:
    """Prepare datasets, then train and validate all models.

    Returns mapping model_name -> training history dict.
    """
    logger.info("Preparing datasets from CIF files...")
    paths = prepare_datasets_from_cif()
    data_path = str(paths.get('data_path'))

    logger.info("Starting training for all models...")
    histories: Dict[str, Any] = {}

    try:
        histories['cgcnn'] = train_cgcnn(data_path=data_path)
    except Exception as e:
        logger.error(f"CGCNN training failed: {e}")
        histories['cgcnn'] = {'error': str(e)}

    try:
        histories['megnet'] = train_megnet(data_path=data_path)
    except Exception as e:
        logger.error(f"MEGNet training failed: {e}")
        histories['megnet'] = {'error': str(e)}

    try:
        histories['schnet'] = train_schnet(data_path=data_path)
    except Exception as e:
        logger.error(f"SchNet training failed: {e}")
        histories['schnet'] = {'error': str(e)}

    try:
        histories['mpnn'] = train_mpnn(data_path=data_path)
    except Exception as e:
        logger.error(f"MPNN training failed: {e}")
        histories['mpnn'] = {'error': str(e)}

    logger.info("Training complete for all models")
    return histories


__all__ = ['train_and_validate_all_models']


