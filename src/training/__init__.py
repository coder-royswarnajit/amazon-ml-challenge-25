"""
Training Module for Amazon ML Challenge

This module provides training functionality for all model types:
- Neural network training with LoRA fine-tuning
- GBDT model training (LightGBM, XGBoost, CatBoost)
- Ensemble training with 2-level stacking
"""

from .train_neural_net import (
    setup_lora_model,
    train_neural_network,
    predict_with_tta
)

from .train_gbdt import (
    optimize_lightgbm,
    train_lightgbm,
    optimize_xgboost,
    train_xgboost,
    train_catboost,
    train_all_gbdt_models
)

from .train_ensemble import (
    StackingEnsemble,
    optimize_base_model_weights,
    train_ensemble,
    create_submission
)

__all__ = [
    # Neural network training
    'setup_lora_model',
    'train_neural_network',
    'predict_with_tta',
    
    # GBDT training
    'optimize_lightgbm',
    'train_lightgbm',
    'optimize_xgboost',
    'train_xgboost',
    'train_catboost',
    'train_all_gbdt_models',
    
    # Ensemble training
    'StackingEnsemble',
    'optimize_base_model_weights',
    'train_ensemble',
    'create_submission'
]
