"""
MEGNet (MatErials Graph Network) module.
Implementation of the MEGNet model for materials property prediction.
"""

from .model import MEGNet
from .train import train_megnet
from .predict import predict_megnet

__all__ = ['MEGNet', 'train_megnet', 'predict_megnet']
