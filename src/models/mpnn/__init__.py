"""
MPNN (Message Passing Neural Network) module.
Implementation of the MPNN model for graph property prediction.
"""

from .model import MPNN
from .train import train_mpnn
from .predict import predict_mpnn

__all__ = ['MPNN', 'train_mpnn', 'predict_mpnn']
