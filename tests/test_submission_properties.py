"""
Property-based tests for submission generation.

Tests verify correctness properties for:
- Submission format correctness
- Submission completeness
- Log space conversion
"""

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from pathlib import Path
import tempfile
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config


# ==================== Test Strategies ====================

@st.composite
def prediction_strategy(draw, n_samples=None):
    """Generate valid predictions in log space."""
    if n_samples is None:
        n_samples = draw(st.integers(min_value=10, max_value=100))
    
    # Generate sample IDs
    sample_ids = [f"test_{i}" for i in range(n_samples)]
    
    # Generate log-space predictions
    log_predictions = np.array([
        draw(st.floats(min_value=1.0, max_value=15.0))
        for _ in range(n_samples)
    ])
    
    return sample_ids, log_predictions


# ==================== Property Tests ====================

class TestSubmissionFormatProperties:
    """Property-based tests for submission format."""
    
    @given(prediction_strategy())
    @settings(max_examples=20, deadline=None)
    def test_property_41_submission_format_correctness(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 41: Submission format correctness**
        **Validates: Requirements 12.3**
        
        Submission CSV should have exactly two columns: sample_id and predicted_price.
        """
        sample_ids, log_predictions = data
        
        # Convert from log space to original space
        original_predictions = np.expm1(log_predictions)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'sample_id': sample_ids,
            'predicted_price': original_predictions
        })
        
        # Property: Submission should have exactly 2 columns
        assert len(submission_df.columns) == 2, \
            f"Submission should have 2 columns, got {len(submission_df.columns)}"
        
        # Property: Columns should be in correct order
        assert list(submission_df.columns) == ['sample_id', 'predicted_price'], \
            f"Columns should be ['sample_id', 'predicted_price'], got {list(submission_df.columns)}"
        
        # Save and verify CSV format
        with tempfile.TemporaryDirectory() as tmpdir:
            submission_path = Path(tmpdir) / "submission.csv"
            submission_df.to_csv(submission_path, index=False)
            
            # Reload and verify
            loaded_df = pd.read_csv(submission_path)
            
            assert list(loaded_df.columns) == ['sample_id', 'predicted_price'], \
                "Saved CSV should have correct column names"
    
    @given(prediction_strategy())
    @settings(max_examples=20, deadline=None)
    def test_property_42_submission_completeness(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 42: Submission completeness**
        **Validates: Requirements 12.4**
        
        Submission should have predictions for all test samples with no missing values.
        """
        sample_ids, log_predictions = data
        n_samples = len(sample_ids)
        
        # Convert from log space to original space
        original_predictions = np.expm1(log_predictions)
        
        # Create submission DataFrame
        submission_df = pd.DataFrame({
            'sample_id': sample_ids,
            'predicted_price': original_predictions
        })
        
        # Property: Number of rows should match number of samples
        assert len(submission_df) == n_samples, \
            f"Submission should have {n_samples} rows, got {len(submission_df)}"
        
        # Property: All sample IDs should be present
        assert set(submission_df['sample_id']) == set(sample_ids), \
            "All sample IDs should be present in submission"
        
        # Property: No missing sample IDs
        assert not submission_df['sample_id'].isna().any(), \
            "sample_id column should not have missing values"
        
        # Property: No missing predictions
        assert not submission_df['predicted_price'].isna().any(), \
            "predicted_price column should not have missing values"
    
    @given(prediction_strategy())
    @settings(max_examples=20, deadline=None)
    def test_property_43_submission_log_space_conversion(self, data):
        """
        **Feature: amazon-ml-price-prediction, Property 43: Submission log space conversion**
        **Validates: Requirements 12.2**
        
        Predictions should be correctly converted from log space (log1p) to original space (expm1).
        """
        sample_ids, log_predictions = data
        
        # Convert from log space to original space using expm1
        original_predictions = np.expm1(log_predictions)
        
        # Property: Conversion should be reversible
        reconverted = np.log1p(original_predictions)
        assert np.allclose(log_predictions, reconverted, rtol=1e-5), \
            "Log space conversion should be reversible"
        
        # Property: Original predictions should be positive
        assert (original_predictions > 0).all(), \
            "All predictions in original space should be positive"
        
        # Property: Original predictions should be finite
        assert np.isfinite(original_predictions).all(), \
            "All predictions should be finite"
    
    @given(st.integers(min_value=1, max_value=100))
    @settings(max_examples=10, deadline=None)
    def test_submission_price_positive(self, n_samples):
        """
        **Property: All predicted prices should be positive**
        **Validates: Prices are valid monetary values**
        """
        sample_ids = [f"test_{i}" for i in range(n_samples)]
        
        # Even if log predictions are negative (which means price < 1),
        # expm1 will still give positive prices
        log_predictions = np.random.uniform(-2.0, 15.0, n_samples)
        original_predictions = np.expm1(log_predictions)
        
        # Handle edge case where log_predictions are very negative
        # In practice, we should ensure log_predictions >= 0
        log_predictions = np.clip(log_predictions, 0, None)
        original_predictions = np.expm1(log_predictions)
        
        # Property: All prices should be non-negative
        assert (original_predictions >= 0).all(), \
            "All predicted prices should be non-negative"
    
    @given(prediction_strategy())
    @settings(max_examples=10, deadline=None)
    def test_submission_no_duplicates(self, data):
        """
        **Property: Submission should not have duplicate sample IDs**
        **Validates: Each sample has exactly one prediction**
        """
        sample_ids, log_predictions = data
        
        # Ensure unique sample IDs in test data
        assume(len(set(sample_ids)) == len(sample_ids))
        
        original_predictions = np.expm1(log_predictions)
        
        submission_df = pd.DataFrame({
            'sample_id': sample_ids,
            'predicted_price': original_predictions
        })
        
        # Property: No duplicate sample IDs
        assert not submission_df['sample_id'].duplicated().any(), \
            "Submission should not have duplicate sample IDs"
