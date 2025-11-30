"""
Property-based tests for custom loss functions.

Tests verify correctness properties for:
- HuberSMAPE loss gradient correctness
- Focal SMAPE loss focusing behavior
- GBDT custom objective gradient correctness
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.losses import HuberSMAPELoss, FocalSMAPELoss, lgb_smape_objective, xgb_smape_objective
from config import config


# ==================== Test Strategies ====================

@st.composite
def prediction_target_strategy(draw, batch_size=None):
    """Generate valid predictions and targets tensors."""
    if batch_size is None:
        batch_size = draw(st.integers(min_value=2, max_value=64))
    
    # Generate positive values for predictions and targets (log space prices)
    preds = [draw(st.floats(min_value=0.1, max_value=15.0)) for _ in range(batch_size)]
    targets = [draw(st.floats(min_value=0.1, max_value=15.0)) for _ in range(batch_size)]
    
    return (
        torch.tensor(preds, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )


# ==================== Property Tests ====================

class TestHuberSMAPELossProperties:
    """Property-based tests for HuberSMAPE loss."""
    
    @given(prediction_target_strategy())
    @settings(max_examples=50, deadline=None)
    def test_huber_smape_loss_non_negative(self, pred_target):
        """
        **Property: Loss non-negativity**
        **Validates: Loss function always returns non-negative values**
        
        For any predictions and targets, the HuberSMAPE loss should be non-negative.
        """
        preds, targets = pred_target
        loss_fn = HuberSMAPELoss(delta=config.HUBER_DELTA)
        
        loss = loss_fn(preds, targets)
        
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    @given(prediction_target_strategy())
    @settings(max_examples=50, deadline=None)
    def test_huber_smape_loss_zero_for_identical(self, pred_target):
        """
        **Property: Zero loss for identical predictions**
        **Validates: Loss is zero when predictions equal targets**
        
        When predictions exactly match targets, the loss should be approximately zero.
        """
        _, targets = pred_target
        preds = targets.clone()  # Identical predictions
        loss_fn = HuberSMAPELoss(delta=config.HUBER_DELTA)
        
        loss = loss_fn(preds, targets)
        
        assert loss.item() < 1e-6, f"Loss should be ~0 for identical predictions, got {loss.item()}"
    
    @given(prediction_target_strategy())
    @settings(max_examples=30, deadline=None)
    def test_huber_smape_loss_gradient_exists(self, pred_target):
        """
        **Property: Gradient existence**
        **Validates: Requirements 5.1 - Loss function produces valid gradients**
        
        The loss function should produce valid gradients for backpropagation.
        """
        preds, targets = pred_target
        preds.requires_grad = True
        loss_fn = HuberSMAPELoss(delta=config.HUBER_DELTA)
        
        loss = loss_fn(preds, targets)
        loss.backward()
        
        assert preds.grad is not None, "Gradients should exist"
        assert not torch.isnan(preds.grad).any(), "Gradients should not contain NaN"
        assert not torch.isinf(preds.grad).any(), "Gradients should not contain Inf"


class TestFocalSMAPELossProperties:
    """Property-based tests for Focal SMAPE loss."""
    
    @given(prediction_target_strategy())
    @settings(max_examples=50, deadline=None)
    def test_focal_smape_loss_non_negative(self, pred_target):
        """
        **Property: Loss non-negativity**
        **Validates: Focal loss always returns non-negative values**
        """
        preds, targets = pred_target
        loss_fn = FocalSMAPELoss(gamma=2.0)
        
        loss = loss_fn(preds, targets)
        
        assert loss.item() >= 0, f"Loss should be non-negative, got {loss.item()}"
    
    @given(prediction_target_strategy())
    @settings(max_examples=30, deadline=None)
    def test_focal_smape_loss_gradient_exists(self, pred_target):
        """
        **Property: Gradient existence for focal loss**
        **Validates: Focal loss produces valid gradients**
        """
        preds, targets = pred_target
        preds.requires_grad = True
        loss_fn = FocalSMAPELoss(gamma=2.0)
        
        loss = loss_fn(preds, targets)
        loss.backward()
        
        assert preds.grad is not None, "Gradients should exist"
        assert not torch.isnan(preds.grad).any(), "Gradients should not contain NaN"


class TestGBDTObjectiveProperties:
    """Property-based tests for GBDT custom objectives."""
    
    @given(prediction_target_strategy())
    @settings(max_examples=50, deadline=None)
    def test_lgb_smape_objective_returns_gradient_hessian(self, pred_target):
        """
        **Property: LightGBM objective returns valid gradient and hessian**
        **Validates: Requirements 6.1 - Custom SMAPE objective for LightGBM**
        """
        import lightgbm as lgb
        
        preds, targets = pred_target
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        
        # Create a LightGBM Dataset object as the function expects
        lgb_dataset = lgb.Dataset(data=np.zeros((len(targets_np), 1)), label=targets_np)
        lgb_dataset.construct()
        
        grad, hess = lgb_smape_objective(preds_np, lgb_dataset)
        
        # Check shapes
        assert grad.shape == preds_np.shape, f"Gradient shape mismatch"
        assert hess.shape == preds_np.shape, f"Hessian shape mismatch"
        
        # Check for NaN/Inf
        assert not np.isnan(grad).any(), "Gradient should not contain NaN"
        assert not np.isnan(hess).any(), "Hessian should not contain NaN"
        assert not np.isinf(grad).any(), "Gradient should not contain Inf"
        assert not np.isinf(hess).any(), "Hessian should not contain Inf"
    
    @given(prediction_target_strategy())
    @settings(max_examples=50, deadline=None)
    def test_xgb_smape_objective_returns_gradient_hessian(self, pred_target):
        """
        **Property: XGBoost objective returns valid gradient and hessian**
        **Validates: Requirements 6.2 - Custom SMAPE objective for XGBoost**
        """
        import xgboost as xgb
        
        preds, targets = pred_target
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        
        # Create an XGBoost DMatrix as the function expects
        dmatrix = xgb.DMatrix(data=np.zeros((len(targets_np), 1)), label=targets_np)
        
        grad, hess = xgb_smape_objective(preds_np, dmatrix)
        
        # Check shapes
        assert grad.shape == preds_np.shape, f"Gradient shape mismatch"
        assert hess.shape == preds_np.shape, f"Hessian shape mismatch"
        
        # Check for NaN/Inf
        assert not np.isnan(grad).any(), "Gradient should not contain NaN"
        assert not np.isnan(hess).any(), "Hessian should not contain NaN"
    
    @given(prediction_target_strategy())
    @settings(max_examples=30, deadline=None)
    def test_lgb_objective_zero_gradient_for_identical(self, pred_target):
        """
        **Property: Zero gradient for identical predictions**
        **Validates: Gradient is approximately zero when predictions equal targets**
        """
        import lightgbm as lgb
        
        _, targets = pred_target
        preds = targets.clone()
        preds_np = preds.numpy()
        targets_np = targets.numpy()
        
        # Create a LightGBM Dataset object
        lgb_dataset = lgb.Dataset(data=np.zeros((len(targets_np), 1)), label=targets_np)
        lgb_dataset.construct()
        
        grad, _ = lgb_smape_objective(preds_np, lgb_dataset)
        
        # Gradient should be near zero for identical predictions
        assert np.abs(grad).mean() < 1e-5, f"Gradient should be ~0 for identical predictions, got mean {np.abs(grad).mean()}"
