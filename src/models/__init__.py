"""
Models package for Amazon ML Price Prediction.

This module exports:
- Multimodal model architecture (OptimizedMultimodalModel, CrossModalAttention, GatedFusion)
- Custom loss functions (HuberSMAPELoss, FocalSMAPELoss, SMAPELoss)
- GBDT custom objectives (lgb_smape_objective, xgb_smape_objective)
- Model utilities (ModelEMA, save_model, load_model, apply_lora)
"""

from .multimodal import (
    OptimizedMultimodalModel,
    CrossModalAttention,
    GatedFusion,
    TabularProjection,
    RegressionHead,
    create_model
)

from .losses import (
    HuberSMAPELoss,
    FocalSMAPELoss,
    SMAPELoss,
    lgb_smape_objective,
    lgb_smape_eval,
    xgb_smape_objective,
    xgb_smape_eval,
    get_loss_function
)

from .utils import (
    ModelEMA,
    save_model,
    load_model,
    apply_lora,
    freeze_model_layers,
    count_parameters,
    get_parameter_groups,
    GradientClipper
)

__all__ = [
    # Multimodal model
    'OptimizedMultimodalModel',
    'CrossModalAttention',
    'GatedFusion',
    'TabularProjection',
    'RegressionHead',
    'create_model',
    
    # Loss functions
    'HuberSMAPELoss',
    'FocalSMAPELoss',
    'SMAPELoss',
    'lgb_smape_objective',
    'lgb_smape_eval',
    'xgb_smape_objective',
    'xgb_smape_eval',
    'get_loss_function',
    
    # Utilities
    'ModelEMA',
    'save_model',
    'load_model',
    'apply_lora',
    'freeze_model_layers',
    'count_parameters',
    'get_parameter_groups',
    'GradientClipper',
]
