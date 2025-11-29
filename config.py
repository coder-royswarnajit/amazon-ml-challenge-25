"""
Centralized configuration for Amazon ML Price Prediction system.

This module provides all configuration parameters including:
- File paths for data, models, checkpoints, and outputs
- Model hyperparameters for neural networks and GBDT models
- Training settings (batch size, learning rate, epochs)
- Feature engineering parameters (IPQ patterns, unit conversions)
"""

from pathlib import Path
from typing import Dict, List
import re


class Config:
    """Configuration class for the Amazon ML Price Prediction system."""
    
    # ==================== PATHS ====================
    # Base directories
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    IMAGE_DIR = DATA_DIR / "images"
    MODEL_DIR = BASE_DIR / "models"
    CHECKPOINT_DIR = BASE_DIR / "checkpoints"
    PRED_DIR = BASE_DIR / "predictions"
    LOG_DIR = BASE_DIR / "logs"
    
    # Data files
    TRAIN_CSV = DATA_DIR / "train.csv"
    TEST_CSV = DATA_DIR / "test.csv"
    SUBMISSION_CSV = PRED_DIR / "submission.csv"
    
    # Feature files
    TRAIN_FEATURES_PKL = DATA_DIR / "train_features.pkl"
    VAL_FEATURES_PKL = DATA_DIR / "val_features.pkl"
    TEST_FEATURES_PKL = DATA_DIR / "test_features.pkl"
    
    # Model files
    NEURAL_NET_MODEL = MODEL_DIR / "neural_net_best.pt"
    LIGHTGBM_MODEL = MODEL_DIR / "lightgbm.txt"
    XGBOOST_MODEL = MODEL_DIR / "xgboost.json"
    CATBOOST_MODEL = MODEL_DIR / "catboost.cbm"
    ENSEMBLE_ARTIFACTS = MODEL_DIR / "ensemble_artifacts.pkl"
    
    # ==================== RANDOM SEED ====================
    RANDOM_SEED = 42
    
    # ==================== DATA PROCESSING ====================
    # Train/validation split
    VAL_SIZE = 0.15
    
    # Image settings
    IMAGE_SIZE = 224
    IMAGE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    IMAGE_STD = [0.229, 0.224, 0.225]
    
    # Text settings
    MAX_TEXT_LENGTH = 128
    TEXT_MODEL_NAME = "microsoft/deberta-v3-small"
    
    # Download settings
    DOWNLOAD_BATCH_SIZE = 5000
    DOWNLOAD_MAX_WORKERS = 40
    DOWNLOAD_TIMEOUT = 10
    DOWNLOAD_MAX_RETRIES = 3
    
    # ==================== FEATURE ENGINEERING ====================
    # TF-IDF settings
    TFIDF_MAX_FEATURES = 100
    TFIDF_MIN_DF = 5
    TFIDF_MAX_DF = 0.95
    
    # IPQ (Item Pack Quantity) extraction patterns
    IPQ_PATTERNS = [
        r'(\d+(?:\.\d+)?)\s*(kg|kilogram|kilograms)',
        r'(\d+(?:\.\d+)?)\s*(g|gram|grams)',
        r'(\d+(?:\.\d+)?)\s*(mg|milligram|milligrams)',
        r'(\d+(?:\.\d+)?)\s*(l|liter|liters|litre|litres)',
        r'(\d+(?:\.\d+)?)\s*(ml|milliliter|milliliters|millilitre|millilitres)',
        r'(\d+(?:\.\d+)?)\s*(m|meter|meters|metre|metres)',
        r'(\d+(?:\.\d+)?)\s*(cm|centimeter|centimeters|centimetre|centimetres)',
        r'(\d+(?:\.\d+)?)\s*(mm|millimeter|millimeters|millimetre|millimetres)',
        r'pack\s+of\s+(\d+)',
        r'(\d+)\s*pack',
        r'(\d+)\s*count',
        r'(\d+)\s*pieces?',
        r'(\d+)\s*pcs',
    ]
    
    # Unit conversions to standard units
    UNIT_CONVERSIONS: Dict[str, tuple] = {
        # Weight conversions to grams
        'kg': ('weight', 1000.0),
        'kilogram': ('weight', 1000.0),
        'kilograms': ('weight', 1000.0),
        'g': ('weight', 1.0),
        'gram': ('weight', 1.0),
        'grams': ('weight', 1.0),
        'mg': ('weight', 0.001),
        'milligram': ('weight', 0.001),
        'milligrams': ('weight', 0.001),
        
        # Volume conversions to milliliters
        'l': ('volume', 1000.0),
        'liter': ('volume', 1000.0),
        'liters': ('volume', 1000.0),
        'litre': ('volume', 1000.0),
        'litres': ('volume', 1000.0),
        'ml': ('volume', 1.0),
        'milliliter': ('volume', 1.0),
        'milliliters': ('volume', 1.0),
        'millilitre': ('volume', 1.0),
        'millilitres': ('volume', 1.0),
        
        # Length conversions to millimeters
        'm': ('length', 1000.0),
        'meter': ('length', 1000.0),
        'meters': ('length', 1000.0),
        'metre': ('length', 1000.0),
        'metres': ('length', 1000.0),
        'cm': ('length', 10.0),
        'centimeter': ('length', 10.0),
        'centimeters': ('length', 10.0),
        'centimetre': ('length', 10.0),
        'centimetres': ('length', 10.0),
        'mm': ('length', 1.0),
        'millimeter': ('length', 1.0),
        'millimeters': ('length', 1.0),
        'millimetre': ('length', 1.0),
        'millimetres': ('length', 1.0),
        
        # Count (no conversion needed)
        'pack': ('count', 1.0),
        'count': ('count', 1.0),
        'pieces': ('count', 1.0),
        'piece': ('count', 1.0),
        'pcs': ('count', 1.0),
    }
    
    # Quality keywords
    QUALITY_KEYWORDS = [
        'premium', 'quality', 'best', 'top', 'excellent', 'superior',
        'professional', 'luxury', 'deluxe', 'authentic', 'genuine',
        'certified', 'original', 'branded', 'high-quality'
    ]
    
    # Discount keywords
    DISCOUNT_KEYWORDS = [
        'discount', 'sale', 'offer', 'deal', 'save', 'off',
        'clearance', 'bargain', 'cheap', 'affordable', 'budget'
    ]
    
    # ==================== NEURAL NETWORK HYPERPARAMETERS ====================
    # Model architecture
    IMAGE_MODEL_NAME = "efficientnet_b2"
    HIDDEN_DIM = 512
    TABULAR_HIDDEN_DIM = 128
    REGRESSOR_HIDDEN_DIMS = [256, 128]
    DROPOUT_RATE = 0.15
    ATTENTION_HEADS = 8
    
    # LoRA settings
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.1
    LORA_TARGET_MODULES = ["query_proj", "key_proj", "value_proj"]
    
    # Training settings
    BATCH_SIZE = 12
    NUM_EPOCHS = 15
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    GRADIENT_ACCUMULATION_STEPS = 4
    MAX_GRAD_NORM = 1.0
    
    # EMA settings
    EMA_DECAY = 0.9999
    
    # Mixed precision
    USE_FP16 = True
    
    # Loss function
    LOSS_TYPE = "huber_smape"  # Options: "huber_smape", "focal_smape"
    HUBER_DELTA = 1.0
    
    # Checkpoint settings
    CHECKPOINT_INTERVAL_MINUTES = 30
    MAX_CHECKPOINTS_TO_KEEP = 3
    
    # ==================== GBDT HYPERPARAMETERS ====================
    # LightGBM
    LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 1000,
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }
    
    # XGBoost
    XGBOOST_PARAMS = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 1000,
        'random_state': RANDOM_SEED,
        'verbosity': 0,
    }
    
    # CatBoost
    CATBOOST_PARAMS = {
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'task_type': 'GPU',
        'devices': '0',
    }
    
    # Optuna settings
    OPTUNA_N_TRIALS = 50
    OPTUNA_TIMEOUT = 3600  # 1 hour
    
    # ==================== ENSEMBLE SETTINGS ====================
    # Level-1 meta-learners
    RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0]
    ELASTICNET_ALPHAS = [0.1, 1.0, 10.0]
    ELASTICNET_L1_RATIOS = [0.1, 0.5, 0.9]
    
    # Shallow LightGBM for meta-learning
    META_LIGHTGBM_PARAMS = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'n_estimators': 100,
        'random_state': RANDOM_SEED,
        'verbose': -1,
    }
    
    # Level-2 optimization
    LEVEL2_OPTIMIZATION_METHOD = 'SLSQP'
    LEVEL2_BOUNDS = (0.0, 1.0)  # Weights must be non-negative
    
    # ==================== EVALUATION SETTINGS ====================
    # Metrics
    SMAPE_EPSILON = 1e-10
    N_QUANTILES = 5
    
    # Test-time augmentation
    USE_TTA = True
    N_TTA = 3
    
    # ==================== VISUALIZATION SETTINGS ====================
    FIGURE_DPI = 300
    FIGURE_SIZE = (12, 8)
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.DATA_DIR / 'processed',  # Processed features directory
            cls.IMAGE_DIR,
            cls.MODEL_DIR,
            cls.CHECKPOINT_DIR,
            cls.PRED_DIR,
            cls.LOG_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device (CUDA if available, else CPU)."""
        import torch
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Create singleton instance
config = Config()

# Create directories on import
config.create_directories()


# ==================== COMPATIBILITY DICTIONARIES ====================
# These provide dictionary-style access for backward compatibility with scripts

PATHS = {
    'base_dir': Config.BASE_DIR,
    'data_dir': Config.DATA_DIR,
    'raw_dir': Config.DATA_DIR,  # Raw data directory (same as data_dir)
    'processed_dir': Config.DATA_DIR / 'processed',
    'images_dir': Config.IMAGE_DIR,
    'models_dir': Config.MODEL_DIR,
    'checkpoints_dir': Config.CHECKPOINT_DIR,
    'predictions_dir': Config.PRED_DIR,
    'logs_dir': Config.LOG_DIR,
    'train_csv': Config.TRAIN_CSV,
    'test_csv': Config.TEST_CSV,
    'submission_csv': Config.SUBMISSION_CSV,
}

DATA_CONFIG = {
    'kaggle_competition': 'amazon-ml-challenge-2025',
    'val_size': Config.VAL_SIZE,
    'random_seed': Config.RANDOM_SEED,
    'image_size': Config.IMAGE_SIZE,
    'image_mean': Config.IMAGE_MEAN,
    'image_std': Config.IMAGE_STD,
    'max_text_length': Config.MAX_TEXT_LENGTH,
    'download_batch_size': Config.DOWNLOAD_BATCH_SIZE,
    'download_max_workers': Config.DOWNLOAD_MAX_WORKERS,
    'download_timeout': Config.DOWNLOAD_TIMEOUT,
}

FEATURE_CONFIG = {
    'tfidf_features': Config.TFIDF_MAX_FEATURES,
    'tfidf_min_df': Config.TFIDF_MIN_DF,
    'tfidf_max_df': Config.TFIDF_MAX_DF,
    'ipq_patterns': Config.IPQ_PATTERNS,
    'unit_conversions': Config.UNIT_CONVERSIONS,
    'quality_keywords': Config.QUALITY_KEYWORDS,
    'discount_keywords': Config.DISCOUNT_KEYWORDS,
}

MODEL_CONFIG = {
    'text_config': {
        'model_name': Config.TEXT_MODEL_NAME,
        'max_length': Config.MAX_TEXT_LENGTH,
    },
    'image_config': {
        'model_name': Config.IMAGE_MODEL_NAME,
        'image_size': Config.IMAGE_SIZE,
    },
    'hidden_dim': Config.HIDDEN_DIM,
    'tabular_hidden_dim': Config.TABULAR_HIDDEN_DIM,
    'regressor_hidden_dims': Config.REGRESSOR_HIDDEN_DIMS,
    'dropout_rate': Config.DROPOUT_RATE,
    'attention_heads': Config.ATTENTION_HEADS,
    'lora_r': Config.LORA_R,
    'lora_alpha': Config.LORA_ALPHA,
    'lora_dropout': Config.LORA_DROPOUT,
    'lora_target_modules': Config.LORA_TARGET_MODULES,
}

TRAIN_CONFIG = {
    'batch_size': Config.BATCH_SIZE,
    'nn_epochs': Config.NUM_EPOCHS,
    'learning_rate': Config.LEARNING_RATE,
    'weight_decay': Config.WEIGHT_DECAY,
    'warmup_ratio': Config.WARMUP_RATIO,
    'gradient_accumulation_steps': Config.GRADIENT_ACCUMULATION_STEPS,
    'max_grad_norm': Config.MAX_GRAD_NORM,
    'ema_decay': Config.EMA_DECAY,
    'use_fp16': Config.USE_FP16,
    'loss_type': Config.LOSS_TYPE,
    'huber_delta': Config.HUBER_DELTA,
    'checkpoint_interval_minutes': Config.CHECKPOINT_INTERVAL_MINUTES,
    'max_checkpoints_to_keep': Config.MAX_CHECKPOINTS_TO_KEEP,
    'use_tta': Config.USE_TTA,
    'n_tta': Config.N_TTA,
    'random_seed': Config.RANDOM_SEED,
}

GBDT_CONFIG = {
    'lightgbm_params': Config.LIGHTGBM_PARAMS,
    'xgboost_params': Config.XGBOOST_PARAMS,
    'catboost_params': Config.CATBOOST_PARAMS,
    'optuna_n_trials': Config.OPTUNA_N_TRIALS,
    'optuna_timeout': Config.OPTUNA_TIMEOUT,
    'random_seed': Config.RANDOM_SEED,
}

ENSEMBLE_CONFIG = {
    'ridge_alphas': Config.RIDGE_ALPHAS,
    'elasticnet_alphas': Config.ELASTICNET_ALPHAS,
    'elasticnet_l1_ratios': Config.ELASTICNET_L1_RATIOS,
    'meta_lightgbm_params': Config.META_LIGHTGBM_PARAMS,
    'level2_optimization_method': Config.LEVEL2_OPTIMIZATION_METHOD,
    'level2_bounds': Config.LEVEL2_BOUNDS,
}

EVAL_CONFIG = {
    'smape_epsilon': Config.SMAPE_EPSILON,
    'n_quantiles': Config.N_QUANTILES,
}
