# Materials Modeling with Graph Neural Networks

## Project Overview

This project demonstrates the application of various Graph Neural Network (GNN) architectures for predicting material properties, specifically formation energy of crystalline structures. The project compares the effectiveness of four state-of-the-art approaches: **CGCNN**, **MEGNet**, **SchNet**, and **MPNN**.

## Project Goals and Objectives

### Primary Goals
- **Predict formation energy** of crystalline materials using deep learning
- **Compare different GNN architectures** for materials science applications
- **Implement hyperparameter optimization** using Bayesian optimization
- **Provide reproducible results** with configurable random seeds
- **Create a containerized solution** for easy deployment and sharing

### Scientific Objectives
- Achieve accurate predictions of formation energy (target: NaCl ~-3.6 eV/atom)
- Demonstrate the effectiveness of graph-based approaches for crystal structures
- Provide insights into which GNN architecture works best for specific material types

## Theoretical Background

### Formation Energy
Formation energy is a key thermodynamic property that determines the stability of crystalline structures. For NaCl (table salt), the reference value is approximately **-2.1 eV/atom**.

### Why Graph Neural Networks?
Crystal structures can be naturally represented as graphs where:
- **Nodes** represent atoms
- **Edges** represent chemical bonds and interactions
- **Graph topology** captures the 3D spatial relationships

## Models in the Project

### 1. **CGCNN (Crystal Graph Convolutional Neural Network)**
- **Purpose**: Specifically designed for crystalline structures
- **Architecture**: Uses graph convolutions to process atomic connections
- **Features**: Handles periodic boundary conditions, crystal symmetry
- **Best for**: Inorganic crystals, materials with well-defined unit cells

### 2. **MEGNet (MatErials Graph Network)**
- **Purpose**: Universal architecture for materials science
- **Architecture**: Combines graph neural networks with global states
- **Features**: Global feature aggregation, multi-task learning capability
- **Best for**: Diverse material properties, transfer learning

### 3. **SchNet (Schrödinger Network)**
- **Purpose**: Physics-informed neural network based on quantum mechanics
- **Architecture**: Uses continuous filters for modeling interactions
- **Features**: Continuous convolution, physical interpretability
- **Best for**: Molecular systems, quantum chemistry applications

### 4. **MPNN (Message Passing Neural Network)**
- **Purpose**: General framework for graph neural networks
- **Architecture**: Message passing between graph nodes
- **Features**: Flexible architecture, adaptable to various graph types
- **Best for**: General graph learning, baseline comparisons

## Project Structure

```
materials_modeling/
├── Dockerfile                          # Container definition
├── docker-compose.yml                  # Service orchestration
├── main.py                            # Main application entry point
├── pyproject.toml                     # Project dependencies (uv)
├── requirements.txt                    # Python package requirements
├── settings/                          # Configuration files
│   ├── materials/                     # Material selection and prediction
│   │   ├── selection_materials.yaml   # Materials for training
│   │   └── predict_material.yaml      # Materials for prediction
│   ├── models/                        # Model configuration
│   │   ├── hyperparameter_limits.yaml # HPO bounds and settings
│   │   └── models.yaml                # Training epochs and model settings
│   ├── preprocessing_data_training/    # Data preprocessing settings
│   │   └── data_preprocessing.yaml    # Data splitting, scaling, graph features
│   └── materialsproject_api/          # API configuration
├── src/                               # Source code
│   ├── config.py                      # Configuration management
│   ├── data_preprocessing.py          # Data preprocessing and graph creation (multi-dimensional node/edge features, RBF)
│   ├── get_data.py                    # Data download from Materials Project
│   ├── loading_data/                 # Data downloading utilities (Materials Project)
│   │   └── download.py               # Download & caching logic
│   ├── models/                        # Model implementations
│   │   ├── cgcnn/                    # CGCNN model
│   │   ├── megnet/                    # MEGNet model
│   │   ├── schnet/                    # SchNet model
│   │   ├── mpnn/                      # MPNN model
│   │   └── hpo.py                     # Hyperparameter optimization
│   ├── train_validate_models.py       # Training orchestration
│   ├── predict_models.py              # Prediction pipeline
│   ├── app.py                         # Main application logic (called from main.py)
│   └── utils/                         # Utility functions (seed, file utils, metrics)
├── models/                            # Trained model checkpoints
├── temporary_data/                    # Downloaded and processed data
├── logs/                              # Application logs
└── result_models.json                 # Prediction results and training histories
```

## Configuration Management

The project uses a comprehensive YAML-based configuration system located in the `settings/` directory:

### 📁 **`settings/materials/`**
- **`selection_materials.yaml`**: Defines materials for training and validation
- **`predict_material.yaml`**: Specifies materials and structures for prediction

### 📁 **`settings/models/`**
- **`hyperparameter_limits.yaml`**: Defines hyperparameter search bounds for Bayesian optimization
- **`models.yaml`**: Configures training epochs and model-specific settings

### 📁 **`settings/preprocessing_data_training/`**
- **`data_preprocessing.yaml`**: Controls data splitting ratios, target scaling, graph features, and reproducibility

### Key Configuration Features
- **Data splitting ratios** (train/test splits)
- **Hyperparameter optimization** settings (initial points, iterations)
- **Training epochs** (HPO trials vs. final training)
- **Random seeds** for reproducibility
- **Graph creation parameters** (neighbor cutoff, minimum edges, RBF expansion)
- **Target scaling** options (standard, minmax, robust)

#### Graph Features (example)
```yaml
graph_features:
  neighbor_cutoff: 5.0      # neighbor search radius, Å
  min_edges: 1              # minimal edges per graph (fallback FC if sparse)
  rbf:
    num_gaussians: 32       # number of Gaussian bases for distance expansion
    gamma: null             # if null, chosen automatically from spacing
```

## Quick Start

### 🐳 **Docker Compose (Recommended)**

```bash
# Clone the repository
git clone https://github.com/NorgeyBilinskiy/materials_modeling.git
cd materials_modeling

# Build and run with Docker Compose
docker-compose up --build

# Run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### 🐳 **Docker Commands**

```bash
# Build image
docker build -t materials-modeling .

# Run container
docker run -it --rm \
  -v $(pwd)/temporary_data:/app/temporary_data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/settings:/app/settings \
  materials-modeling

# Run with specific environment variables
docker run -it --rm \
  -e MATERIALS_PROJECT_TOKEN=your_token_here \
  materials-modeling
```

### 💻 **Local Installation**

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies with uv
uv sync

# Or with pip
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

### Basic Execution

```bash
# Run complete pipeline (download, train, predict)
python main.py

# Check existing data before downloading
python main.py  # Automatically detects existing CIF files
```

### Configuration Customization

1. **Modify material lists** in `settings/materials/`
2. **Adjust training parameters** in `settings/models/`
3. **Change data preprocessing** in `settings/preprocessing_data_training/`
4. **Set random seeds** for reproducibility

### Expected Results

For NaCl prediction:
- **Reference value**: ~-2.1 eV/atom
- **CGCNN**: Expected close to reference
- **MEGNet**: Stable predictions with good accuracy
- **SchNet**: Physically justified results
- **MPNN**: Baseline predictions for comparison

## Features

### 🔬 **Advanced Data Processing**
- Automatic CIF file download from Materials Project
- Crystal structure to graph conversion
- Multi-dimensional node features: Z, group, period, electronegativity, covalent radius, atomic mass, Mendeleev number
- Edge features: bond distance + Gaussian RBF expansion (configurable)
- Configurable data splitting strategies
- Target variable scaling options

### 🚀 **Hyperparameter Optimization**
- Bayesian optimization with configurable bounds
- Separate training regimes for HPO trials and final training
- Automatic hyperparameter saving and loading
- Learning rate scheduling (ReduceLROnPlateau) and early stopping for stability

### 📊 **Comprehensive Logging**
- Console and file logging with rotation
- Detailed training progress tracking
- Error handling and debugging information

### 🔄 **Reproducibility**
- Configurable random seeds for all components
- Deterministic operations in PyTorch
- Saved model checkpoints with hyperparameters

## Requirements

### System Requirements
- **Python**: 3.11+
- **Memory**: 8GB+ RAM recommended
- **Storage**: 2GB+ for models and data

### Python Dependencies
- **PyTorch**: 2.0+
- **PyTorch Geometric**: Latest version
- **Materials Project API**: MPRester
- **Scikit-learn**: For data preprocessing
- **Bayesian Optimization**: For hyperparameter tuning

### Docker Requirements
- **Docker**: 20.10+
- **Docker Compose**: 2.0+

## Data Sources

- **Materials Project** (https://materialsproject.org/): Primary source for crystal structures and properties
- **CIF Files**: Standard format for crystallographic information
- **Formation Energy**: Calculated using density functional theory (DFT)

## Output Files

### 📁 **`logs/`**
- Timestamped log files with rotation
- Debug and info level logging
- Training progress and error tracking

### 📁 **`models/`**
- Trained model checkpoints
- Best and final model states
- Hyperparameter configurations

### 📄 **`result_models.json`**
- Structured prediction results
- Training history summaries (best_epoch, losses)
- Final metrics per model: RMSE, MAE, R², MSE
- Metadata and timestamps

#### Example final metrics section
```json
{
  "final_metrics": {
    "cgcnn": { "rmse": 0.18, "mae": 0.10, "r2": 0.91, "mse": 0.03 },
    "megnet": { "rmse": 0.43, "mae": 0.33, "r2": 0.52, "mse": 0.19 }
  }
}
```

## Metrics
During training, test metrics are computed and logged for each model:
- RMSE, MAE, R², MSE
These are also saved to `result_models.json` under `final_metrics`.

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce batch size in hyperparameter limits
- **Download failures**: Check Materials Project API token
- **Model loading errors / size mismatches**: If you recently changed graph features (e.g., enabled RBF or multi-dimensional node features), old checkpoints may be incompatible. Delete outdated files in `models/**/best_model.pth` and retrain.
- **Embedding expects Long indices**: This occurs if a checkpoint was trained with integer atom indices, but the current model receives continuous features. Retrain with the new feature pipeline, or ensure the model is constructed with continuous input projection (already handled in code).
- **Empty validation sets**: Check material structure assignments

### Debug Mode
Enable detailed logging by modifying `data_preprocessing.yaml`:
```yaml
logging:
  level: "DEBUG"
  show_progress: true
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Author

**This project was created by a Data Analyst.**

- **GitHub**: [NorgeyBilinskiy](https://github.com/NorgeyBilinskiy)
- **Telegram**: [@Norgey](https://t.me/Norgey)

## Acknowledgments

- Materials Project team for providing the API and database
- PyTorch Geometric community for graph neural network tools
- Open source materials science community

---

*For questions and support, please reach out via GitHub or Telegram.*
