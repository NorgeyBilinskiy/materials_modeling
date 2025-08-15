"""
Module for managing random seeds to ensure reproducibility.
"""

import random

import numpy as np
import torch
from loguru import logger


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value to use for all libraries
    """
    logger.info(f"Setting random seed to {seed} for reproducibility")

    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Random seeds set successfully for all libraries")


def set_torch_seeds(seed: int = 42) -> None:
    """Set only PyTorch random seeds.

    Args:
        seed: Random seed value to use for PyTorch
    """
    logger.info(f"Setting PyTorch random seed to {seed}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("PyTorch random seeds set successfully")


def set_numpy_seeds(seed: int = 42) -> None:
    """Set only NumPy random seeds.

    Args:
        seed: Random seed value to use for NumPy
    """
    logger.info(f"Setting NumPy random seed to {seed}")

    np.random.seed(seed)

    logger.info("NumPy random seeds set successfully")


def set_python_seeds(seed: int = 42) -> None:
    """Set only Python random seeds.

    Args:
        seed: Random seed value to use for Python random
    """
    logger.info(f"Setting Python random seed to {seed}")

    random.seed(seed)

    logger.info("Python random seeds set successfully")
