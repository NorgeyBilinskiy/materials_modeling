#!/usr/bin/env python3
"""
Main CLI script for NaCl formation energy prediction using graph neural networks.
Supports training and prediction with different models: CGCNN, MEGNet, SchNet, MPNN.
"""

import click
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_loader.download import download_nacl_data
from data_loader.preprocess import preprocess_data
from models.cgcnn.train import train_cgcnn
from models.cgcnn.predict import predict_cgcnn
from models.megnet.train import train_megnet
from models.megnet.predict import predict_megnet
from models.schnet.train import train_schnet
from models.schnet.predict import predict_schnet
from models.mpnn.train import train_mpnn
from models.mpnn.predict import predict_mpnn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

@click.group()
def cli():
    """NaCl Formation Energy Prediction with Graph Neural Networks"""
    pass

@cli.command()
@click.option('--method', type=click.Choice(['cgcnn', 'megnet', 'schnet', 'mpnn']), 
              required=True, help='Model to train')
@click.option('--epochs', default=100, help='Number of training epochs')
@click.option('--batch-size', default=32, help='Batch size for training')
@click.option('--lr', default=0.001, help='Learning rate')
@click.option('--data-path', default='data/', help='Path to data directory')
def train(method, epochs, batch_size, lr, data_path):
    """Train a specific model"""
    logger.info(f"Starting training for {method.upper()} model")
    
    # Download and preprocess data if needed
    if not os.path.exists(os.path.join(data_path, 'processed')):
        logger.info("Downloading and preprocessing data...")
        download_nacl_data(data_path)
        preprocess_data(data_path)
    
    # Train the selected model
    if method == 'cgcnn':
        train_cgcnn(epochs, batch_size, lr, data_path)
    elif method == 'megnet':
        train_megnet(epochs, batch_size, lr, data_path)
    elif method == 'schnet':
        train_schnet(epochs, batch_size, lr, data_path)
    elif method == 'mpnn':
        train_mpnn(epochs, batch_size, lr, data_path)
    
    logger.info(f"Training completed for {method.upper()}")

@cli.command()
@click.option('--method', type=click.Choice(['cgcnn', 'megnet', 'schnet', 'mpnn']), 
              required=True, help='Model to use for prediction')
@click.option('--model-path', help='Path to trained model')
@click.option('--data-path', default='data/', help='Path to data directory')
def predict(method, model_path, data_path):
    """Make predictions using a trained model"""
    logger.info(f"Starting prediction with {method.upper()} model")
    
    # Use default model path if not specified
    if not model_path:
        model_path = f'models/{method}/best_model.pth'
    
    # Make predictions
    if method == 'cgcnn':
        result = predict_cgcnn(model_path, data_path)
    elif method == 'megnet':
        result = predict_megnet(model_path, data_path)
    elif method == 'schnet':
        result = predict_schnet(model_path, data_path)
    elif method == 'mpnn':
        result = predict_mpnn(model_path, data_path)
    
    logger.info(f"Prediction result: {result:.4f} eV/atom")
    return result

@cli.command()
@click.option('--data-path', default='data/', help='Path to data directory')
def compare(data_path):
    """Compare all models and their predictions"""
    logger.info("Comparing all models...")
    
    results = {}
    methods = ['cgcnn', 'megnet', 'schnet', 'mpnn']
    
    for method in methods:
        model_path = f'models/{method}/best_model.pth'
        if os.path.exists(model_path):
            try:
                if method == 'cgcnn':
                    results[method] = predict_cgcnn(model_path, data_path)
                elif method == 'megnet':
                    results[method] = predict_megnet(model_path, data_path)
                elif method == 'schnet':
                    results[method] = predict_schnet(model_path, data_path)
                elif method == 'mpnn':
                    results[method] = predict_mpnn(model_path, data_path)
            except Exception as e:
                logger.warning(f"Failed to predict with {method}: {e}")
                results[method] = None
        else:
            logger.warning(f"Model not found for {method}")
            results[method] = None
    
    # Print comparison
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    print(f"{'Model':<10} {'Prediction (eV/atom)':<20} {'Error vs Reference':<20}")
    print("-"*50)
    
    reference = -3.6  # Reference value for NaCl
    for method, result in results.items():
        if result is not None:
            error = abs(result - reference)
            print(f"{method.upper():<10} {result:<20.4f} {error:<20.4f}")
        else:
            print(f"{method.upper():<10} {'N/A':<20} {'N/A':<20}")
    
    print("-"*50)
    print(f"Reference value: {reference} eV/atom")
    print("="*50)

@cli.command()
@click.option('--data-path', default='data/', help='Path to data directory')
def setup(data_path):
    """Setup the project by downloading and preprocessing data"""
    logger.info("Setting up the project...")
    
    # Download data
    download_nacl_data(data_path)
    
    # Preprocess data
    preprocess_data(data_path)
    
    logger.info("Project setup completed!")

@cli.command()
def info():
    """Display project information"""
    print("NaCl Formation Energy Prediction with Graph Neural Networks")
    print("="*60)
    print("Available models:")
    print("- CGCNN: Crystal Graph Convolutional Neural Network")
    print("- MEGNet: MatErials Graph Network")
    print("- SchNet: Schrödinger Network")
    print("- MPNN: Message Passing Neural Network")
    print("\nReference value for NaCl formation energy: -3.6 eV/atom")
    print("\nUsage examples:")
    print("  python run.py train --method cgcnn")
    print("  python run.py predict --method cgcnn")
    print("  python run.py compare")
    print("  python run.py setup")

if __name__ == '__main__':
    cli()
