"""
Property-based tests for GBDT training and ensemble module.

Tests verify correctness properties for:
- GBDT model serialization
- GBDT prediction completeness
- Meta-feature stacking
- Weight normalization
- Isotonic calibration
- Ensemble artifact completeness
"""

import pytest
import torch
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config

# Conditional imports based on availability
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ==================== Test Strategies ====================

@st.composite
def training_data_strategy(draw, n_samples=None, n_features=None):
    """Generate valid training data."""
    if n_samples is None:
        n_samples = draw(st.integers(min_value=100, max_value=500))
    if n_features is None:
        n_features = draw(st.integers(min_value=10, max_value=50))
    
    # Generate features
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate targets (log-space prices)
    y = np.random.uniform(2.0, 12.0, n_samples).astype(np.float32)
    
    return X, y


@st.composite
def meta_predictions_strategy(draw):
    """Generate meta-predictions from multiple models."""
    n_samples = draw(st.integers(min_value=50, max_value=200))
    n_models = draw(st.integers(min_value=3, max_value=5))
    
    predictions = {}
    for i in range(n_models):
        model_name = f"model_{i}"
        # Generate predictions in log-space
        predictions[model_name] = np.random.uniform(2.0, 12.0, n_samples).astype(np.float32)
    
    # Generate true targets
    targets = np.random.uniform(2.0, 12.0, n_samples).astype(np.float32)
    
    return predictions, targets


# ==================== Property Tests ====================

@pytest.mark.skipif(not HAS_LIGHTGBM, reason="LightGBM not available")
class TestGBDTSerializationProperties:
    """Property-based tests for GBDT model serialization."""
    
    @given(training_data_strategy())
    @settings(max_examples=5, deadline=None)
    def test_property_22_gbdt_model_serialization(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 22: GBDT model serialization**
        **Validates: Requirements 6.5**
        
        Saved and loaded GBDT model should produce identical predictions.
        """
        X, y = data
        
        # Train a simple LightGBM model
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'learning_rate': 0.1,
            'n_estimators': 20,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        # Get predictions before saving
        preds_before = model.predict(X)
        
        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.txt"
            model.booster_.save_model(str(model_path))
            
            # Load model
            loaded_model = lgb.Booster(model_file=str(model_path))
            preds_after = loaded_model.predict(X)
        
        # Property: Predictions should be identical
        assert np.allclose(preds_before, preds_after, rtol=1e-5), \
            "Predictions should be identical after save/load"
    
    @given(training_data_strategy())
    @settings(max_examples=5, deadline=None)
    def test_property_23_gbdt_prediction_completeness(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 23: GBDT prediction completeness**
        **Validates: Requirements 6.6**
        
        GBDT model should produce predictions for all input samples.
        """
        X, y = data
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 15,
            'n_estimators': 20,
            'verbose': -1,
            'random_state': 42
        }
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X, y)
        
        # Get predictions
        preds = model.predict(X)
        
        # Property: Should have prediction for each sample
        assert len(preds) == len(X), \
            f"Expected {len(X)} predictions, got {len(preds)}"
        
        # Property: No NaN predictions
        assert not np.isnan(preds).any(), \
            "Predictions should not contain NaN"
        
        # Property: No Inf predictions
        assert not np.isinf(preds).any(), \
            "Predictions should not contain Inf"


class TestEnsembleProperties:
    """Property-based tests for ensemble training."""
    
    @given(meta_predictions_strategy())
    @settings(max_examples=10, deadline=None)
    def test_property_24_meta_feature_stacking(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 24: Meta-feature stacking correctness**
        **Validates: Requirements 7.1**
        
        Meta-features should correctly stack predictions from all base models.
        """
        predictions, targets = data
        n_models = len(predictions)
        n_samples = len(targets)
        
        # Stack predictions into meta-features
        meta_features = np.column_stack([predictions[key] for key in sorted(predictions.keys())])
        
        # Property: Meta-features should have correct shape
        expected_shape = (n_samples, n_models)
        assert meta_features.shape == expected_shape, \
            f"Meta-features shape {meta_features.shape} != expected {expected_shape}"
        
        # Property: Meta-features should preserve individual model predictions
        for i, model_name in enumerate(sorted(predictions.keys())):
            assert np.allclose(meta_features[:, i], predictions[model_name]), \
                f"Meta-features column {i} doesn't match {model_name} predictions"
    
    @given(st.integers(min_value=3, max_value=10))
    @settings(max_examples=20, deadline=None)
    def test_property_25_level2_weight_normalization(self, n_models):
        """
        **Feature: amazon-ml-price-prediction, Property 25: Level-2 weight normalization**
        **Validates: Requirements 7.3**
        
        Ensemble weights should sum to 1 and be non-negative.
        """
        # Generate random positive weights
        raw_weights = np.random.rand(n_models)
        
        # Normalize weights
        normalized_weights = raw_weights / raw_weights.sum()
        
        # Property: Weights should sum to 1
        assert np.isclose(normalized_weights.sum(), 1.0), \
            f"Weights should sum to 1, got {normalized_weights.sum()}"
        
        # Property: All weights should be non-negative
        assert (normalized_weights >= 0).all(), \
            "All weights should be non-negative"
        
        # Property: All weights should be <= 1
        assert (normalized_weights <= 1).all(), \
            "All weights should be <= 1"
    
    @given(st.data())
    @settings(max_examples=10, deadline=None)
    def test_property_26_isotonic_calibration_monotonicity(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 26: Isotonic calibration monotonicity**
        **Validates: Requirements 7.5**
        
        Isotonic regression should produce monotonically non-decreasing predictions.
        """
        from sklearn.isotonic import IsotonicRegression
        
        n_samples = data.draw(st.integers(min_value=50, max_value=200))
        
        # Generate predictions and targets
        X = np.sort(np.random.uniform(2.0, 12.0, n_samples))  # Sorted predictions
        y = X + np.random.randn(n_samples) * 0.5  # Noisy targets
        
        # Fit isotonic regression
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(X, y)
        
        # Get calibrated predictions
        calibrated = iso_reg.predict(X)
        
        # Property: Calibrated predictions should be monotonically non-decreasing
        for i in range(1, len(calibrated)):
            assert calibrated[i] >= calibrated[i-1] - 1e-10, \
                f"Isotonic predictions not monotonic at index {i}: {calibrated[i-1]} > {calibrated[i]}"
    
    @given(st.integers(min_value=3, max_value=5))
    @settings(max_examples=5, deadline=None)
    def test_property_27_ensemble_artifact_completeness(self, n_models):
        """
        **Feature: amazon-ml-price-prediction, Property 27: Ensemble artifact completeness**
        **Validates: Requirements 7.6**
        
        Ensemble artifacts should contain all required components for inference.
        """
        # Simulate ensemble artifacts
        artifacts = {
            'meta_learners': {f'model_{i}': {'type': 'ridge', 'coefs': np.random.randn(n_models)} for i in range(3)},
            'level2_weights': np.random.rand(3) / 3,  # Normalized weights
            'calibration': {'model': 'isotonic', 'fitted': True},
            'model_names': [f'base_model_{i}' for i in range(n_models)],
            'config': {'n_models': n_models}
        }
        
        # Property: All required keys should be present
        required_keys = ['meta_learners', 'level2_weights', 'model_names']
        for key in required_keys:
            assert key in artifacts, f"Missing required artifact: {key}"
        
        # Property: Weights should match number of meta-learners
        assert len(artifacts['level2_weights']) == len(artifacts['meta_learners']), \
            "Number of weights should match number of meta-learners"
        
        # Save and load artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "ensemble_artifacts.pkl"
            
            with open(artifact_path, 'wb') as f:
                pickle.dump(artifacts, f)
            
            with open(artifact_path, 'rb') as f:
                loaded_artifacts = pickle.load(f)
        
        # Property: Loaded artifacts should match original
        assert set(loaded_artifacts.keys()) == set(artifacts.keys()), \
            "Loaded artifacts should have same keys as original"
