"""
Data loader module for NaCl formation energy prediction.
Handles downloading and preprocessing of crystal structure data.
"""

from .download import download_nacl_data, get_api_key, get_nacl_reference_data
from .preprocess import preprocess_data

__all__ = ['download_nacl_data', 'get_api_key', 'get_nacl_reference_data', 'preprocess_data']
