"""
Custom loss functions for SMAPE optimization.

This module provides custom loss functions optimized for the SMAPE metric,
including Huber-smoothed SMAPE for neural networks and custom objectives
for LightGBM and XGBoost.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from config import config


class HuberSMAPELoss(nn.Module):
    """
    Huber-smoothed SMAPE loss for robust neural network training.
    
    Combines the SMAPE metric with Huber smoothing to reduce sensitivity
    to outliers while still optimizing the target metric directly.
    
    Args:
        epsilon: Small constant to avoid division by zero
        delta: Huber delta parameter - errors larger than delta are treated linearly
    
    Validates: Requirements 5.1
    """
    
    def __init__(self, epsilon: float = 1e-10, delta: float = 1.0):
        """
        Initialize HuberSMAPELoss.
        
        Args:
            epsilon: Small constant for numerical stability
            delta: Huber transition point (default: 1.0)
        """
        super().__init__()
        self.epsilon = epsilon
        self.delta = delta
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Huber-smoothed SMAPE loss.
        
        Args:
            y_pred: Predicted values in log space (batch_size,)
            y_true: Target values in log space (batch_size,)
            
        Returns:
            Scalar loss value
        """
        # Convert from log space to original space
        pred_exp = torch.expm1(y_pred)
        true_exp = torch.expm1(y_true)
        
        # Ensure positive predictions
        pred_exp = torch.clamp(pred_exp, min=self.epsilon)
        true_exp = torch.clamp(true_exp, min=self.epsilon)
        
        # SMAPE components
        numerator = torch.abs(pred_exp - true_exp)
        denominator = (torch.abs(true_exp) + torch.abs(pred_exp)) / 2.0 + self.epsilon
        
        # Raw SMAPE per sample
        smape = numerator / denominator
        
        # Apply Huber smoothing
        # For small errors: quadratic loss (0.5 * smape^2 / delta)
        # For large errors: linear loss (smape - 0.5 * delta)
        mask = smape <= self.delta
        huber_smape = torch.where(
            mask,
            0.5 * smape ** 2 / self.delta,
            smape - 0.5 * self.delta
        )
        
        # Return mean loss scaled to percentage
        return 100.0 * huber_smape.mean()


class FocalSMAPELoss(nn.Module):
    """
    Focal-weighted SMAPE loss for handling hard examples.
    
    Applies focal weighting to down-weight easy examples and focus training
    on hard-to-predict samples, similar to Focal Loss for classification.
    
    Args:
        epsilon: Small constant to avoid division by zero
        gamma: Focusing parameter (higher = more focus on hard examples)
        alpha: Balance parameter (default: 1.0)
    
    Validates: Requirements 5.1
    """
    
    def __init__(self, epsilon: float = 1e-10, gamma: float = 2.0, alpha: float = 1.0):
        """
        Initialize FocalSMAPELoss.
        
        Args:
            epsilon: Small constant for numerical stability
            gamma: Focusing parameter (default: 2.0)
            alpha: Balancing parameter (default: 1.0)
        """
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute focal-weighted SMAPE loss.
        
        Args:
            y_pred: Predicted values in log space (batch_size,)
            y_true: Target values in log space (batch_size,)
            
        Returns:
            Scalar loss value
        """
        # Convert from log space to original space
        pred_exp = torch.expm1(y_pred)
        true_exp = torch.expm1(y_true)
        
        # Ensure positive predictions
        pred_exp = torch.clamp(pred_exp, min=self.epsilon)
        true_exp = torch.clamp(true_exp, min=self.epsilon)
        
        # SMAPE components
        numerator = torch.abs(pred_exp - true_exp)
        denominator = (torch.abs(true_exp) + torch.abs(pred_exp)) / 2.0 + self.epsilon
        
        # Raw SMAPE per sample (scaled to [0, 2] range)
        smape = numerator / denominator
        
        # Normalize SMAPE to [0, 1] range for focal weighting
        normalized_smape = smape / 2.0
        
        # Focal weight: (smape)^gamma - harder examples get higher weight
        focal_weight = torch.pow(normalized_smape + self.epsilon, self.gamma)
        
        # Apply focal weighting
        focal_smape = self.alpha * focal_weight * smape
        
        # Return mean loss scaled to percentage
        return 100.0 * focal_smape.mean()


class SMAPELoss(nn.Module):
    """
    Standard SMAPE loss without smoothing.
    
    Directly optimizes SMAPE metric for neural network training.
    May be less stable than Huber-smoothed version.
    
    Args:
        epsilon: Small constant to avoid division by zero
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize SMAPELoss.
        
        Args:
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute standard SMAPE loss.
        
        Args:
            y_pred: Predicted values in log space (batch_size,)
            y_true: Target values in log space (batch_size,)
            
        Returns:
            Scalar loss value
        """
        # Convert from log space to original space
        pred_exp = torch.expm1(y_pred)
        true_exp = torch.expm1(y_true)
        
        # Ensure positive predictions
        pred_exp = torch.clamp(pred_exp, min=self.epsilon)
        true_exp = torch.clamp(true_exp, min=self.epsilon)
        
        # SMAPE formula
        numerator = torch.abs(pred_exp - true_exp)
        denominator = (torch.abs(true_exp) + torch.abs(pred_exp)) / 2.0 + self.epsilon
        
        smape = numerator / denominator
        
        return 100.0 * smape.mean()


# =============================================================================
# GBDT Custom Objectives and Evaluation Functions
# =============================================================================

def lgb_smape_objective(y_pred: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom SMAPE objective function for LightGBM.
    
    Returns gradient and hessian of the SMAPE loss with respect to predictions.
    
    Args:
        y_pred: Predicted values (in log space)
        dataset: LightGBM Dataset object containing labels
        
    Returns:
        Tuple of (gradient, hessian) arrays
        
    Validates: Requirements 6.1
    """
    y_true = dataset.get_label()
    
    # Convert from log space
    pred_exp = np.expm1(y_pred)
    true_exp = np.expm1(y_true)
    
    # Clip to ensure positive values
    epsilon = 1e-10
    pred_exp = np.clip(pred_exp, epsilon, None)
    true_exp = np.clip(true_exp, epsilon, None)
    
    # SMAPE components
    diff = pred_exp - true_exp
    sum_abs = np.abs(true_exp) + np.abs(pred_exp) + epsilon
    
    # Gradient of SMAPE w.r.t. pred_exp
    # d(SMAPE)/d(pred) = sign(diff) / (sum/2) - |diff| * sign(pred) / (sum/2)^2
    sign_diff = np.sign(diff)
    sign_pred = np.sign(pred_exp)
    
    grad_raw = (2.0 * sign_diff / sum_abs) - (2.0 * np.abs(diff) * sign_pred / (sum_abs ** 2))
    
    # Chain rule for log transform: d(pred_exp)/d(y_pred) = pred_exp
    grad = grad_raw * pred_exp
    
    # Simplified hessian (diagonal approximation for stability)
    hess = 2.0 * pred_exp * pred_exp / (sum_abs ** 2)
    hess = np.clip(hess, 0.001, 10.0)  # Numerical stability
    
    return grad, hess


def lgb_smape_eval(y_pred: np.ndarray, dataset) -> Tuple[str, float, bool]:
    """
    SMAPE evaluation function for LightGBM.
    
    Args:
        y_pred: Predicted values (in log space)
        dataset: LightGBM Dataset object containing labels
        
    Returns:
        Tuple of (metric_name, metric_value, is_higher_better)
        
    Validates: Requirements 6.1
    """
    y_true = dataset.get_label()
    
    # Convert from log space
    pred_exp = np.expm1(y_pred)
    true_exp = np.expm1(y_true)
    
    # Clip to ensure positive values
    epsilon = 1e-10
    pred_exp = np.clip(pred_exp, epsilon, None)
    true_exp = np.clip(true_exp, epsilon, None)
    
    # Calculate SMAPE
    numerator = np.abs(pred_exp - true_exp)
    denominator = (np.abs(true_exp) + np.abs(pred_exp)) / 2.0 + epsilon
    smape = 100.0 * np.mean(numerator / denominator)
    
    return ('smape', smape, False)  # Lower is better


def xgb_smape_objective(y_pred: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom SMAPE objective function for XGBoost.
    
    Returns gradient and hessian of the SMAPE loss with respect to predictions.
    
    Args:
        y_pred: Predicted values (in log space)
        dtrain: XGBoost DMatrix object containing labels
        
    Returns:
        Tuple of (gradient, hessian) arrays
        
    Validates: Requirements 6.2
    """
    y_true = dtrain.get_label()
    
    # Convert from log space
    pred_exp = np.expm1(y_pred)
    true_exp = np.expm1(y_true)
    
    # Clip to ensure positive values
    epsilon = 1e-10
    pred_exp = np.clip(pred_exp, epsilon, None)
    true_exp = np.clip(true_exp, epsilon, None)
    
    # SMAPE components
    diff = pred_exp - true_exp
    sum_abs = np.abs(true_exp) + np.abs(pred_exp) + epsilon
    
    # Gradient of SMAPE w.r.t. pred_exp
    sign_diff = np.sign(diff)
    sign_pred = np.sign(pred_exp)
    
    grad_raw = (2.0 * sign_diff / sum_abs) - (2.0 * np.abs(diff) * sign_pred / (sum_abs ** 2))
    
    # Chain rule for log transform
    grad = grad_raw * pred_exp
    
    # Simplified hessian (diagonal approximation for stability)
    hess = 2.0 * pred_exp * pred_exp / (sum_abs ** 2)
    hess = np.clip(hess, 0.001, 10.0)  # Numerical stability
    
    return grad, hess


def xgb_smape_eval(y_pred: np.ndarray, dtrain) -> Tuple[str, float]:
    """
    SMAPE evaluation function for XGBoost.
    
    Args:
        y_pred: Predicted values (in log space)
        dtrain: XGBoost DMatrix object containing labels
        
    Returns:
        Tuple of (metric_name, metric_value)
        
    Validates: Requirements 6.2
    """
    y_true = dtrain.get_label()
    
    # Convert from log space
    pred_exp = np.expm1(y_pred)
    true_exp = np.expm1(y_true)
    
    # Clip to ensure positive values
    epsilon = 1e-10
    pred_exp = np.clip(pred_exp, epsilon, None)
    true_exp = np.clip(true_exp, epsilon, None)
    
    # Calculate SMAPE
    numerator = np.abs(pred_exp - true_exp)
    denominator = (np.abs(true_exp) + np.abs(pred_exp)) / 2.0 + epsilon
    smape = 100.0 * np.mean(numerator / denominator)
    
    return ('smape', smape)


def get_loss_function(loss_type: str = None, **kwargs) -> nn.Module:
    """
    Factory function to create loss function based on config or specified type.
    
    Args:
        loss_type: Type of loss ('huber_smape', 'focal_smape', 'smape')
                   If None, uses config.LOSS_TYPE
        **kwargs: Additional keyword arguments for the loss function
        
    Returns:
        Loss function module
    """
    if loss_type is None:
        loss_type = config.LOSS_TYPE
    
    if loss_type == 'huber_smape':
        delta = kwargs.get('delta', config.HUBER_DELTA)
        return HuberSMAPELoss(delta=delta)
    elif loss_type == 'focal_smape':
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 1.0)
        return FocalSMAPELoss(gamma=gamma, alpha=alpha)
    elif loss_type == 'smape':
        return SMAPELoss()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Choose from: 'huber_smape', 'focal_smape', 'smape'")
