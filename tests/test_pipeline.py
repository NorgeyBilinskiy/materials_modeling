"""
Tests for the NaCl formation energy prediction pipeline.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
from pymatgen.core import Structure, Lattice

from src.data_loader import create_sample_nacl_data
from src.data_loader import preprocess_data, create_graph_features
from src.models.cgcnn import create_cgcnn_model
from src.models import create_megnet_model
from src.models.schnet.model import create_schnet_model
from src.models.mpnn.model import create_mpnn_model


class TestDataPipeline(unittest.TestCase):
    """Test data loading and preprocessing pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_data_dir)
    
    def test_data_download(self):
        """Test data download functionality."""
        # Test sample data creation
        create_sample_nacl_data(self.test_data_dir)
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "nacl.cif")))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "nacl_info.json")))
        
        # Test training dataset creation
        from src.data_loader import create_training_dataset
        create_training_dataset(self.test_data_dir)
        
        # Check if training data was created
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "training_data.json")))
        self.assertTrue(os.path.exists(os.path.join(self.test_data_dir, "training_data.csv")))
    
    def test_data_preprocessing(self):
        """Test data preprocessing functionality."""
        # Create sample data first
        create_sample_nacl_data(self.test_data_dir)
        from src.data_loader import create_training_dataset
        create_training_dataset(self.test_data_dir)
        
        # Test preprocessing
        preprocess_data(self.test_data_dir)
        
        # Check if processed data was created
        processed_dir = os.path.join(self.test_data_dir, "processed")
        self.assertTrue(os.path.exists(processed_dir))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, "train.pt")))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, "val.pt")))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, "test.pt")))
        self.assertTrue(os.path.exists(os.path.join(processed_dir, "metadata.json")))
    
    def test_graph_features(self):
        """Test graph feature creation."""
        # Create NaCl structure
        lattice = Lattice.cubic(5.64)
        structure = Structure(
            lattice=lattice,
            species=["Na", "Cl"],
            coords=[[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        
        # Test graph feature creation
        node_features, edge_index, edge_attr = create_graph_features(structure)
        
        # Check output types and shapes
        self.assertIsInstance(node_features, torch.Tensor)
        self.assertIsInstance(edge_index, torch.Tensor)
        self.assertIsInstance(edge_attr, torch.Tensor)
        
        # Check shapes
        self.assertEqual(node_features.shape[0], 2)  # 2 atoms
        self.assertEqual(edge_index.shape[0], 2)  # 2D edge indices
        self.assertEqual(edge_attr.shape[1], 1)  # 1 edge feature (distance)


class TestModels(unittest.TestCase):
    """Test model creation and basic functionality."""
    
    def test_cgcnn_model(self):
        """Test CGCNN model creation."""
        model = create_cgcnn_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))
        
        # Test forward pass with dummy data
        from torch_geometric.data import Data
        dummy_data = Data(
            x=torch.tensor([[11], [17]], dtype=torch.long),  # Na, Cl
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[2.8], [2.8]], dtype=torch.float),
            batch=torch.tensor([0, 0], dtype=torch.long)
        )
        
        with torch.no_grad():
            output = model(dummy_data)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, (1,))  # Single prediction
    
    def test_megnet_model(self):
        """Test MEGNet model creation."""
        model = create_megnet_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))
        
        # Test forward pass with dummy data
        from torch_geometric.data import Data
        dummy_data = Data(
            x=torch.tensor([[11], [17]], dtype=torch.long),  # Na, Cl
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[2.8], [2.8]], dtype=torch.float),
            batch=torch.tensor([0, 0], dtype=torch.long)
        )
        
        with torch.no_grad():
            output = model(dummy_data)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, (1,))  # Single prediction
    
    def test_schnet_model(self):
        """Test SchNet model creation."""
        model = create_schnet_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))
        
        # Test forward pass with dummy data
        from torch_geometric.data import Data
        dummy_data = Data(
            x=torch.tensor([[11], [17]], dtype=torch.long),  # Na, Cl
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[2.8], [2.8]], dtype=torch.float),
            pos=torch.tensor([[0, 0, 0], [2.82, 2.82, 2.82]], dtype=torch.float),
            batch=torch.tensor([0, 0], dtype=torch.long)
        )
        
        with torch.no_grad():
            output = model(dummy_data)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, (1,))  # Single prediction
    
    def test_mpnn_model(self):
        """Test MPNN model creation."""
        model = create_mpnn_model()
        
        # Check model structure
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'forward'))
        
        # Test forward pass with dummy data
        from torch_geometric.data import Data
        dummy_data = Data(
            x=torch.tensor([[11], [17]], dtype=torch.long),  # Na, Cl
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            edge_attr=torch.tensor([[2.8], [2.8]], dtype=torch.float),
            batch=torch.tensor([0, 0], dtype=torch.long)
        )
        
        with torch.no_grad():
            output = model(dummy_data)
            self.assertIsInstance(output, torch.Tensor)
            self.assertEqual(output.shape, (1,))  # Single prediction


class TestEndToEnd(unittest.TestCase):
    """Test end-to-end pipeline functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_data_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_data_dir)
    
    def test_full_pipeline(self):
        """Test complete pipeline from data to prediction."""
        # 1. Create data
        create_sample_nacl_data(self.test_data_dir)
        from src.data_loader import create_training_dataset
        create_training_dataset(self.test_data_dir)
        
        # 2. Preprocess data
        preprocess_data(self.test_data_dir)
        
        # 3. Load processed data
        processed_dir = os.path.join(self.test_data_dir, "processed")
        train_dataset = torch.load(os.path.join(processed_dir, "train.pt"))
        
        # 4. Create model
        model = create_cgcnn_model()
        
        # 5. Test prediction
        if len(train_dataset) > 0:
            sample_data = train_dataset[0]
            with torch.no_grad():
                prediction = model(sample_data)
                self.assertIsInstance(prediction, torch.Tensor)
                self.assertEqual(prediction.shape, (1,))
        
        # 6. Check that prediction is reasonable
        # (should be negative for formation energy)
        if len(train_dataset) > 0:
            sample_data = train_dataset[0]
            with torch.no_grad():
                prediction = model(sample_data)
                # Formation energy should typically be negative
                self.assertLess(prediction.item(), 10.0)  # Reasonable upper bound


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
