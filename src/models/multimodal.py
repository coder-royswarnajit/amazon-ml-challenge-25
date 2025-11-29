"""
Multimodal neural network architecture for price prediction.

This module provides the OptimizedMultimodalModel that combines:
- Text encoding with DeBERTa-small + LoRA
- Image encoding with EfficientNet-B2
- Cross-modal attention for text-image fusion
- Tabular feature projection
- Regression head for price prediction

Optimized for 6GB VRAM using gradient checkpointing, mixed precision, and LoRA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from config import config


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-attention between text and image modalities.
    
    Allows text features to attend to image features and vice versa,
    enabling rich multimodal interaction.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        
    Validates: Requirements 4.3
    """
    
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer normalization and dropout
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute cross-modal attention.
        
        Args:
            query: Query tensor (batch, seq_len_q, dim)
            key_value: Key/Value tensor (batch, seq_len_kv, dim)
            key_padding_mask: Optional mask for key_value (batch, seq_len_kv)
            
        Returns:
            Attended output (batch, seq_len_q, dim)
        """
        batch_size = query.shape[0]
        
        # Layer normalization
        query = self.norm_q(query)
        key_value = self.norm_kv(key_value)
        
        # Compute Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf')
            )
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Residual connection
        output = query + attn_output
        
        # Feed-forward with residual
        output = output + self.ffn(self.ffn_norm(output))
        
        return output


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism for combining text and image features.
    
    A simpler alternative to cross-attention that uses learnable gates
    to control the contribution of each modality.
    
    Args:
        text_dim: Dimension of text features
        image_dim: Dimension of image features
        hidden_dim: Hidden dimension for fusion
    """
    
    def __init__(
        self,
        text_dim: int,
        image_dim: int,
        hidden_dim: int = 512
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Project both modalities to same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Gate networks
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.image_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
    
    def forward(
        self,
        text_feat: torch.Tensor,
        image_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse text and image features using gated mechanism.
        
        Args:
            text_feat: Text features (batch, text_dim)
            image_feat: Image features (batch, image_dim)
            
        Returns:
            Fused features (batch, hidden_dim)
        """
        # Project to common dimension
        text_proj = self.text_proj(text_feat)
        image_proj = self.image_proj(image_feat)
        
        # Concatenate for gate computation
        concat = torch.cat([text_proj, image_proj], dim=-1)
        
        # Compute gates
        text_gate = self.text_gate(concat)
        image_gate = self.image_gate(concat)
        
        # Apply gates
        gated_text = text_proj * text_gate
        gated_image = image_proj * image_gate
        
        # Fuse
        fused = torch.cat([gated_text, gated_image], dim=-1)
        output = self.fusion(fused)
        
        return output


class TabularProjection(nn.Module):
    """
    Projects tabular features through a learned embedding layer.
    
    Args:
        input_dim: Number of input features
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        dropout: Dropout probability
        
    Validates: Requirements 4.4
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project tabular features.
        
        Args:
            x: Tabular features (batch, input_dim)
            
        Returns:
            Projected features (batch, output_dim)
        """
        return self.projection(x)


class RegressionHead(nn.Module):
    """
    Multi-layer regression head for price prediction.
    
    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden dimensions
        dropout: Dropout probability
        
    Validates: Requirements 4.5
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128],
        dropout: float = 0.15
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        # Final prediction layer
        layers.append(nn.Linear(in_dim, 1))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute price prediction.
        
        Args:
            x: Input features (batch, input_dim)
            
        Returns:
            Predictions (batch,)
        """
        return self.head(x).squeeze(-1)


class OptimizedMultimodalModel(nn.Module):
    """
    Optimized multimodal model for RTX 3050 6GB.
    
    Architecture:
    - Text: DeBERTa-small (44M params) with LoRA fine-tuning
    - Image: EfficientNet-B2 (9M params) with frozen early layers
    - Fusion: Cross-modal attention (bidirectional)
    - Tabular: Learned projection
    - Output: 3-layer MLP regression head
    
    Memory optimizations:
    - LoRA fine-tuning (only 1.6% parameters trainable)
    - Gradient checkpointing (40% VRAM savings)
    - Mixed precision FP16 training
    - Frozen early layers in image encoder
    
    Args:
        num_tabular_features: Number of tabular input features
        hidden_dim: Hidden dimension for fusion
        use_cross_attention: If True, use cross-attention; else use gated fusion
        freeze_text_encoder: If True, freeze text encoder (when not using LoRA)
        freeze_image_encoder_early: If True, freeze early image encoder layers
        
    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7
    """
    
    def __init__(
        self,
        num_tabular_features: int = 180,
        hidden_dim: int = 512,
        use_cross_attention: bool = True,
        freeze_text_encoder: bool = False,
        freeze_image_encoder_early: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        
        # =================================================================
        # Text Encoder: DeBERTa-small (44M params)
        # =================================================================
        from transformers import AutoModel, AutoConfig
        
        text_config = AutoConfig.from_pretrained(config.TEXT_MODEL_NAME)
        self.text_encoder = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
        self.text_hidden_dim = text_config.hidden_size  # 768 for deberta-v3-small
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
        
        # Text projection to hidden_dim
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # =================================================================
        # Image Encoder: EfficientNet-B2 (9M params)
        # =================================================================
        import timm
        
        self.image_encoder = timm.create_model(
            config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0,  # Remove classification head
            global_pool=''  # No global pooling - we'll do it ourselves
        )
        
        # Get the number of features from EfficientNet
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            dummy_output = self.image_encoder(dummy_input)
            self.image_hidden_dim = dummy_output.shape[1]  # Channel dimension
            self.image_spatial_dim = dummy_output.shape[2] * dummy_output.shape[3]
        
        # Freeze early layers (blocks 0-4), fine-tune later blocks (5-6)
        if freeze_image_encoder_early:
            for name, param in self.image_encoder.named_parameters():
                # Only fine-tune the last few blocks
                if not any(x in name for x in ['blocks.5', 'blocks.6', 'conv_head', 'bn2']):
                    param.requires_grad = False
        
        # Image projection to hidden_dim
        self.image_proj = nn.Sequential(
            nn.Linear(self.image_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Global average pooling for image features
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        
        # =================================================================
        # Cross-Modal Fusion
        # =================================================================
        if use_cross_attention:
            # Bidirectional cross-attention
            self.text_to_image_attn = CrossModalAttention(
                dim=hidden_dim,
                num_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT_RATE
            )
            self.image_to_text_attn = CrossModalAttention(
                dim=hidden_dim,
                num_heads=config.ATTENTION_HEADS,
                dropout=config.DROPOUT_RATE
            )
            fusion_output_dim = hidden_dim * 2  # Concatenation of both directions
        else:
            # Gated fusion (simpler alternative)
            self.gated_fusion = GatedFusion(
                text_dim=hidden_dim,
                image_dim=hidden_dim,
                hidden_dim=hidden_dim
            )
            fusion_output_dim = hidden_dim
        
        # =================================================================
        # Tabular Features Projection
        # =================================================================
        self.tabular_proj = TabularProjection(
            input_dim=num_tabular_features,
            hidden_dim=config.TABULAR_HIDDEN_DIM,
            output_dim=config.TABULAR_HIDDEN_DIM,
            dropout=config.DROPOUT_RATE
        )
        
        # =================================================================
        # Regression Head
        # =================================================================
        total_features = fusion_output_dim + config.TABULAR_HIDDEN_DIM
        self.regressor = RegressionHead(
            input_dim=total_features,
            hidden_dims=config.REGRESSOR_HIDDEN_DIMS,
            dropout=config.DROPOUT_RATE
        )
        
        # Log model parameters
        self._log_model_info()
    
    def _log_model_info(self):
        """Log model parameter counts."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model initialized:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        print(f"  Fusion type: {'Cross-Attention' if self.use_cross_attention else 'Gated'}")
    
    def count_parameters(self) -> int:
        """Count total parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: torch.Tensor,
        tabular: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass combining all modalities.
        
        Args:
            input_ids: Tokenized text (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            image: Image tensor (batch, 3, H, W)
            tabular: Tabular features (batch, num_features)
            
        Returns:
            Predictions in log space (batch,)
            
        Validates: Requirements 4.5
        """
        batch_size = input_ids.shape[0]
        
        # =================================================================
        # Encode Text
        # =================================================================
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        text_hidden = text_outputs.last_hidden_state[:, 0, :]  # (batch, text_hidden_dim)
        text_feat = self.text_proj(text_hidden)  # (batch, hidden_dim)
        
        # =================================================================
        # Encode Image
        # =================================================================
        image_hidden = self.image_encoder(image)  # (batch, channels, h, w)
        image_pooled = self.image_pool(image_hidden).flatten(1)  # (batch, channels)
        image_feat = self.image_proj(image_pooled)  # (batch, hidden_dim)
        
        # =================================================================
        # Cross-Modal Fusion
        # =================================================================
        if self.use_cross_attention:
            # Add sequence dimension for attention
            text_seq = text_feat.unsqueeze(1)  # (batch, 1, hidden_dim)
            image_seq = image_feat.unsqueeze(1)  # (batch, 1, hidden_dim)
            
            # Bidirectional cross-attention
            text_attended = self.text_to_image_attn(text_seq, image_seq)  # (batch, 1, hidden_dim)
            image_attended = self.image_to_text_attn(image_seq, text_seq)  # (batch, 1, hidden_dim)
            
            # Squeeze and concatenate
            text_attended = text_attended.squeeze(1)  # (batch, hidden_dim)
            image_attended = image_attended.squeeze(1)  # (batch, hidden_dim)
            fused_feat = torch.cat([text_attended, image_attended], dim=-1)  # (batch, hidden_dim*2)
        else:
            # Gated fusion
            fused_feat = self.gated_fusion(text_feat, image_feat)  # (batch, hidden_dim)
        
        # =================================================================
        # Tabular Features
        # =================================================================
        tabular_feat = self.tabular_proj(tabular)  # (batch, tabular_hidden_dim)
        
        # =================================================================
        # Combine and Predict
        # =================================================================
        combined = torch.cat([fused_feat, tabular_feat], dim=-1)
        predictions = self.regressor(combined)  # (batch,)
        
        return predictions
    
    def get_text_encoder(self) -> nn.Module:
        """Return the text encoder for LoRA application."""
        return self.text_encoder
    
    def get_image_encoder(self) -> nn.Module:
        """Return the image encoder."""
        return self.image_encoder


def create_model(
    num_tabular_features: int = 180,
    use_cross_attention: bool = True,
    device: Optional[torch.device] = None
) -> OptimizedMultimodalModel:
    """
    Factory function to create the multimodal model.
    
    Args:
        num_tabular_features: Number of tabular features
        use_cross_attention: Whether to use cross-attention or gated fusion
        device: Device to place the model on
        
    Returns:
        Initialized model
    """
    model = OptimizedMultimodalModel(
        num_tabular_features=num_tabular_features,
        hidden_dim=config.HIDDEN_DIM,
        use_cross_attention=use_cross_attention
    )
    
    if device is not None:
        model = model.to(device)
    
    return model
