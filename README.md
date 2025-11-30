# Amazon ML Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-89%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A portfolio-quality multimodal machine learning system for the **Amazon ML Challenge 2025**. This system predicts product prices using text (product descriptions), images, and tabular features with a target SMAPE of **< 9%**.

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **Multimodal Architecture** | DeBERTa-v3-small (text) + EfficientNet-B2 (images) + tabular features with cross-modal attention |
| **Memory Efficient** | Optimized for 6GB VRAM using LoRA (1.6% trainable params), gradient checkpointing, FP16 mixed precision |
| **Resumable Training** | Comprehensive checkpoint system - stop/resume at any point with automatic 30-minute saves |
| **2-Level Ensemble** | Stacking with LightGBM, XGBoost, CatBoost + Ridge/ElasticNet meta-learners + isotonic calibration |
| **Property-Based Testing** | 89 tests covering all 43 properties from requirements specification |

## 📁 Project Structure

```
amazon-ml-challenge-25/
├── config.py                    # Centralized configuration (paths, hyperparameters)
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation
│
├── src/                         # Source code
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset with multimodal loading
│   │   ├── downloader.py       # Resumable image downloader with retry logic
│   │   └── feature_engineering.py  # IPQ, TF-IDF, text statistics extraction
│   ├── models/
│   │   ├── multimodal.py       # OptimizedMultimodalModel with cross-modal attention
│   │   ├── losses.py           # HuberSMAPE, FocalSMAPE, GBDT objectives
│   │   └── utils.py            # EMA, model save/load utilities
│   ├── training/
│   │   ├── train_neural_net.py # Neural network training with LoRA
│   │   ├── train_gbdt.py       # LightGBM, XGBoost, CatBoost training
│   │   └── train_ensemble.py   # 2-level stacking ensemble
│   └── utils/
│       ├── checkpoint.py       # CheckpointManager with auto-cleanup
│       ├── metrics.py          # SMAPE, MAE, RMSE, quantile analysis
│       └── visualization.py    # Training curves, prediction plots
│
├── scripts/                     # Pipeline execution scripts
│   ├── run_stage1_setup.py     # Data download and verification
│   ├── run_stage2_features.py  # Feature engineering
│   ├── run_stage3_neural_net.py # Neural network training
│   ├── run_stage4_gbdt.py      # GBDT model training
│   ├── run_stage5_ensemble.py  # Ensemble training
│   └── create_submission.py    # Generate competition submission
│
├── tests/                       # Comprehensive test suite (89 tests)
│   ├── test_*_properties.py    # Property-based tests for each module
│   ├── test_integration.py     # End-to-end pipeline tests
│   └── test_performance.py     # Memory and speed constraint tests
│
├── data/                        # Data files (CSV, images, features)
├── models/                      # Trained model artifacts
├── checkpoints/                 # Training checkpoints
├── logs/                        # Training logs
└── predictions/                 # Model predictions and submissions
```

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10 or higher
- **CUDA**: 11.8+ (for GPU training)
- **Git**: For cloning the repository

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/coder-royswarnajit/amazon-ml-challenge-25.git
cd amazon-ml-challenge-25

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install package in development mode
pip install -e .

# 5. Verify installation
python -c "from src.models.multimodal import OptimizedMultimodalModel; print('✓ Installation successful')"
```

### Running the Complete Pipeline

```bash
# Stage 1: Download and prepare data
python scripts/run_stage1_setup.py

# Stage 2: Extract features (IPQ, TF-IDF, text statistics)
python scripts/run_stage2_features.py

# Stage 3: Train neural network (with LoRA fine-tuning)
python scripts/run_stage3_neural_net.py

# Stage 4: Train GBDT models (LightGBM, XGBoost, CatBoost)
python scripts/run_stage4_gbdt.py

# Stage 5: Train 2-level ensemble
python scripts/run_stage5_ensemble.py

# Generate final submission
python scripts/create_submission.py
```

Each stage creates checkpoints, allowing you to resume from any point if interrupted.

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA RTX 3050 6GB | RTX 3060 8GB+ |
| **CPU** | 4 cores | 8 cores (Ryzen 7 5800H) |
| **RAM** | 16GB | 32GB |
| **Storage** | 50GB | 100GB SSD |
| **CUDA** | 11.8 | 12.0+ |

The system is specifically optimized for **6GB VRAM** using:
- LoRA fine-tuning (only 1.6% of parameters trainable)
- Gradient checkpointing
- FP16 mixed precision training
- Gradient accumulation (effective batch size = 32)

## ⚙️ Configuration

All settings are centralized in `config.py`:

```python
# Key configuration options
class Config:
    # Paths
    DATA_DIR = "data/"
    MODELS_DIR = "models/"
    CHECKPOINTS_DIR = "checkpoints/"
    
    # Model Architecture
    TEXT_MODEL = "microsoft/deberta-v3-small"
    IMAGE_MODEL = "efficientnet_b2"
    HIDDEN_DIM = 256
    
    # Training
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION = 4  # Effective batch = 32
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    
    # LoRA
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    
    # Checkpointing
    CHECKPOINT_INTERVAL = 1800  # 30 minutes
    MAX_CHECKPOINTS = 3
```

## 🧪 Testing

The project includes **89 comprehensive tests** covering all requirements:

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest tests/test_config_properties.py -v          # Configuration tests
pytest tests/test_dataset_properties.py -v         # Dataset tests
pytest tests/test_feature_engineering_properties.py -v  # Feature engineering
pytest tests/test_model_properties.py -v           # Model architecture
pytest tests/test_losses_properties.py -v          # Loss functions
pytest tests/test_training_properties.py -v        # Training utilities
pytest tests/test_gbdt_ensemble_properties.py -v   # GBDT and ensemble
pytest tests/test_submission_properties.py -v      # Submission format
pytest tests/test_integration.py -v                # End-to-end tests
pytest tests/test_performance.py -v                # Memory/speed tests

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Coverage by Property

| Category | Properties | Tests |
|----------|------------|-------|
| Data Download | 1-4 | 4 |
| Feature Engineering | 5-10 | 12 |
| Checkpoint System | 11-13 | 6 |
| Model Architecture | 14-16 | 5 |
| Training | 17-21 | 6 |
| GBDT Models | 22-23 | 2 |
| Ensemble | 24-27 | 4 |
| Dataset | 28-33 | 6 |
| Metrics | 34-37 | 8 |
| Configuration | 38-40 | 3 |
| Submission | 41-43 | 5 |
| Integration | - | 7 |
| Performance | - | 6 |

## 📊 Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    OptimizedMultimodalModel                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ Text Encoder │  │Image Encoder │  │  Tabular Projection  │  │
│  │ DeBERTa-v3   │  │EfficientNet  │  │   Linear(N → 256)    │  │
│  │  + LoRA      │  │   -B2        │  │                      │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
│         ▼                 ▼                      │              │
│  ┌────────────────────────────────┐              │              │
│  │    Cross-Modal Attention       │              │              │
│  │  (Bidirectional Text ↔ Image)  │              │              │
│  └────────────────┬───────────────┘              │              │
│                   │                              │              │
│                   ▼                              ▼              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Gated Fusion                           │  │
│  │           (Concatenate + Learned Gate)                    │  │
│  └────────────────────────┬─────────────────────────────────┘  │
│                           │                                    │
│                           ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Regression Head (3-layer MLP)                │  │
│  │              768 → 256 → 64 → 1                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config.py
BATCH_SIZE = 4  # Default is 8

# Or increase gradient accumulation
GRADIENT_ACCUMULATION = 8  # Default is 4
```

#### 2. Slow Image Downloads
```bash
# Increase parallel workers in config.py
DOWNLOAD_WORKERS = 8  # Default is 4

# Downloads automatically resume if interrupted
```

#### 3. Missing Dependencies
```bash
# Install additional tokenizer dependencies
pip install sentencepiece tiktoken protobuf

# Reinstall transformers with all extras
pip install transformers[torch] --upgrade
```

#### 4. Checkpoint Not Found
```bash
# List available checkpoints
python -c "from src.utils.checkpoint import CheckpointManager; \
           cm = CheckpointManager('checkpoints'); \
           print(cm.get_latest_checkpoint())"
```

#### 5. Feature Engineering Errors
```bash
# Verify data files exist
ls -la data/train.csv data/test.csv

# Re-run feature extraction
python scripts/run_stage2_features.py --force
```

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Enable pin_memory** in DataLoader (enabled by default)
3. **Pre-download images** before training session
4. **Monitor GPU memory** with `nvidia-smi -l 1`

## 📈 Expected Results

| Metric | Target | Typical Range |
|--------|--------|---------------|
| **Validation SMAPE** | < 9.0% | 7.5% - 8.5% |
| **Training Time** | ~8 hours | 6-10 hours |
| **GPU Memory** | < 6GB | 4.5-5.5 GB |
| **Checkpoint Size** | < 500MB | 300-400 MB |

## 🏗️ Pipeline Stages

### Stage 1: Data Setup
- Downloads training/test CSV files
- Downloads ~150K product images with resume capability
- Verifies data integrity

### Stage 2: Feature Engineering
- Extracts IPQ (Item Pack Quantity) features
- Computes TF-IDF vectors (max 5000 features)
- Extracts text statistics (length, word count, etc.)
- Identifies quality/discount keywords
- Saves features to pickle files

### Stage 3: Neural Network Training
- Loads pre-trained DeBERTa-v3-small and EfficientNet-B2
- Applies LoRA fine-tuning to text encoder
- Trains with HuberSMAPE loss
- Uses EMA for model averaging
- Saves best model based on validation SMAPE

### Stage 4: GBDT Training
- Trains LightGBM with custom SMAPE objective
- Trains XGBoost with custom SMAPE objective
- Trains CatBoost with early stopping
- Optionally optimizes hyperparameters with Optuna

### Stage 5: Ensemble Training
- Creates level-1 meta-features from base model predictions
- Trains Ridge, ElasticNet, and shallow LightGBM meta-learners
- Optimizes level-2 weights to minimize SMAPE
- Applies isotonic calibration

### Stage 6: Submission Generation
- Loads best ensemble predictions
- Converts from log space to original price space
- Validates all test samples have predictions
- Saves submission CSV

## 📝 API Documentation

### Core Classes

```python
# Feature Engineering
from src.data.feature_engineering import FeatureEngineer
fe = FeatureEngineer()
features = fe.engineer_features(df)
fe.save_features(features, "features.pkl")

# Dataset
from src.data.dataset import AmazonMLDataset, get_dataloader
dataset = AmazonMLDataset(df, features, is_training=True)
loader = get_dataloader(dataset, batch_size=8)

# Model
from src.models.multimodal import OptimizedMultimodalModel
model = OptimizedMultimodalModel(num_tabular_features=50)

# Training
from src.training.train_neural_net import train_neural_network
train_neural_network(model, train_loader, val_loader, epochs=10)

# Checkpointing
from src.utils.checkpoint import CheckpointManager
cm = CheckpointManager("checkpoints/")
cm.save_checkpoint(model, optimizer, epoch, step, metrics)
state = cm.load_checkpoint("checkpoint.pt")

# Metrics
from src.utils.metrics import calculate_smape, evaluate_predictions
smape = calculate_smape(predictions, targets)
metrics = evaluate_predictions(predictions, targets)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest tests/ -v`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**M S Rishav Subhin**
- GitHub: [msrishav-28](https://github.com/msrishav-28)

## 🙏 Acknowledgments

- Amazon ML Challenge 2025 organizers
- Hugging Face for Transformers library
- PyTorch team for the deep learning framework
- Microsoft for DeBERTa model
