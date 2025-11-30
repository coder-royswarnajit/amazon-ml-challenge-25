"""
Property-based tests for multimodal model architecture.

Tests verify correctness properties for:
- Forward pass shape consistency
- Tabular feature projection
- Cross-modal attention shapes
"""

import pytest
import torch
import numpy as np
from hypothesis import given, strategies as st, settings, assume
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.multimodal import OptimizedMultimodalModel, CrossModalAttention, GatedFusion, TabularProjection
from config import config


# ==================== Test Strategies ====================

@st.composite
def batch_strategy(draw):
    """Generate a valid batch size."""
    return draw(st.integers(min_value=1, max_value=8))


# ==================== Property Tests ====================

class TestMultimodalModelProperties:
    """Property-based tests for multimodal model architecture."""
    
    @given(batch_strategy())
    @settings(max_examples=5, deadline=None)
    def test_property_15_forward_pass_shape_consistency(self, batch_size):
        """
        **Feature: amazon-ml-price-prediction, Property 15: Multimodal forward pass shape consistency**
        **Validates: Requirements 4.5**
        
        For any valid input batch, the model should produce output with shape (batch_size,).
        """
        # Create model with correct parameter name
        n_tabular_features = 50
        model = OptimizedMultimodalModel(
            num_tabular_features=n_tabular_features,
            hidden_dim=config.HIDDEN_DIM,
            use_cross_attention=True
        )
        model.eval()
        
        # Create dummy inputs
        input_ids = torch.randint(0, 1000, (batch_size, config.MAX_TEXT_LENGTH))
        attention_mask = torch.ones(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long)
        image = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        tabular = torch.randn(batch_size, n_tabular_features)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_ids, attention_mask, image, tabular)
        
        # Property: Output should have shape (batch_size,)
        expected_shape = (batch_size,)
        assert output.shape == expected_shape, \
            f"Output shape {output.shape} != expected {expected_shape}"
        
        # Property: Output should not contain NaN
        assert not torch.isnan(output).any(), \
            "Output should not contain NaN values"
    
    @given(batch_strategy())
    @settings(max_examples=5, deadline=None)
    def test_property_16_tabular_feature_projection(self, batch_size):
        """
        **Feature: amazon-ml-price-prediction, Property 16: Tabular feature projection**
        **Validates: Requirements 4.4**
        
        The tabular projection layer should correctly transform features to hidden dimension.
        """
        n_tabular_features = 50
        hidden_dim = config.HIDDEN_DIM
        
        # Create tabular projection
        projection = TabularProjection(
            input_dim=n_tabular_features,
            hidden_dim=config.TABULAR_HIDDEN_DIM,
            output_dim=hidden_dim,
            dropout=config.DROPOUT_RATE
        )
        projection.eval()
        
        # Create dummy tabular input
        tabular = torch.randn(batch_size, n_tabular_features)
        
        # Forward pass
        with torch.no_grad():
            output = projection(tabular)
        
        # Property: Output should have shape (batch_size, hidden_dim)
        expected_shape = (batch_size, hidden_dim)
        assert output.shape == expected_shape, \
            f"Tabular projection output shape {output.shape} != expected {expected_shape}"
        
        # Property: Output should not contain NaN
        assert not torch.isnan(output).any(), \
            "Tabular projection output should not contain NaN"


class TestCrossModalAttentionProperties:
    """Property-based tests for cross-modal attention."""
    
    @given(batch_strategy())
    @settings(max_examples=5, deadline=None)
    def test_cross_modal_attention_shape_preservation(self, batch_size):
        """
        **Property: Cross-modal attention preserves sequence length**
        **Validates: Cross-attention produces correct output shapes**
        """
        hidden_dim = config.HIDDEN_DIM
        num_heads = config.ATTENTION_HEADS
        seq_len = 16  # Reduced for testing
        
        # Create cross-modal attention with correct parameters
        attention = CrossModalAttention(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=config.DROPOUT_RATE
        )
        attention.eval()
        
        # Create dummy inputs
        query = torch.randn(batch_size, seq_len, hidden_dim)
        key_value = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Forward pass
        with torch.no_grad():
            output = attention(query, key_value)
        
        # Property: Output should have same shape as query
        assert output.shape == query.shape, \
            f"Cross-attention output shape {output.shape} != query shape {query.shape}"


class TestGatedFusionProperties:
    """Property-based tests for gated fusion mechanism."""
    
    @given(batch_strategy())
    @settings(max_examples=5, deadline=None)
    def test_gated_fusion_output_shape(self, batch_size):
        """
        **Property: Gated fusion produces correct output dimension**
        **Validates: Fusion mechanism correctly combines modalities**
        """
        hidden_dim = config.HIDDEN_DIM
        text_dim = 768  # DeBERTa hidden size
        image_dim = 1408  # EfficientNet-B2 feature size
        
        # Create gated fusion with proper dimensions
        fusion = GatedFusion(
            text_dim=text_dim,
            image_dim=image_dim,
            hidden_dim=hidden_dim
        )
        fusion.eval()
        
        # Create dummy inputs (text and image features)
        text_feat = torch.randn(batch_size, text_dim)
        image_feat = torch.randn(batch_size, image_dim)
        
        # Forward pass
        with torch.no_grad():
            output = fusion(text_feat, image_feat)
        
        # Property: Output should have shape (batch_size, hidden_dim)
        expected_shape = (batch_size, hidden_dim)
        assert output.shape == expected_shape, \
            f"Gated fusion output shape {output.shape} != expected {expected_shape}"
        
        # Property: Output should not contain NaN
        assert not torch.isnan(output).any(), \
            "Gated fusion output should not contain NaN"


class TestModelUtilityProperties:
    """Property tests for model utility functions."""
    
    def test_model_parameter_count_positive(self):
        """
        **Property: Model has positive parameter count**
        **Validates: Model is properly initialized with learnable parameters**
        """
        n_tabular_features = 50
        model = OptimizedMultimodalModel(
            num_tabular_features=n_tabular_features,
            hidden_dim=config.HIDDEN_DIM
        )
        
        total_params = model.count_parameters()
        trainable_params = model.count_trainable_parameters()
        
        assert total_params > 0, "Model should have positive total parameters"
        assert trainable_params > 0, "Model should have positive trainable parameters"
        assert trainable_params <= total_params, "Trainable params should be <= total params"
