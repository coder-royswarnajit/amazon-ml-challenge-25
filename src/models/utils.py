"""
Model utilities for training and inference.

This module provides:
- ModelEMA: Exponential Moving Average for model parameters
- save_model / load_model: Checkpoint utilities
- LoRA application helper
"""

import copy
import logging
from pathlib import Path
from typing import Dict, Any, Optional, OrderedDict, Union

import torch
import torch.nn as nn

from config import config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model parameters that is updated with
    exponential moving average. This typically improves generalization.
    
    Usage:
        ema = ModelEMA(model, decay=0.9999)
        for batch in dataloader:
            loss = train_step(model, batch)
            ema.update()  # Update shadow parameters
        
        # For evaluation:
        ema.apply_shadow()  # Copy shadow params to model
        evaluate(model)
        ema.restore()  # Restore original params
    
    Args:
        model: PyTorch model to track
        decay: EMA decay rate (higher = slower update)
        
    Validates: Requirements 5.6
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize ModelEMA.
        
        Args:
            model: Model whose parameters to track
            decay: EMA decay rate (default: 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        
        # Initialize shadow parameters
        self._initialize_shadow()
    
    def _initialize_shadow(self) -> None:
        """Initialize shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self) -> None:
        """
        Update shadow parameters with EMA of current model parameters.
        
        Should be called after each optimization step.
        
        Formula: shadow = decay * shadow + (1 - decay) * param
        
        Validates: Property 20 (EMA update consistency)
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name in self.shadow:
                        # EMA update: new_avg = decay * old_avg + (1 - decay) * current
                        self.shadow[name] = (
                            self.decay * self.shadow[name] +
                            (1.0 - self.decay) * param.data
                        )
                    else:
                        # New parameter, just copy
                        self.shadow[name] = param.data.clone()
    
    def apply_shadow(self) -> None:
        """
        Apply shadow parameters to model for evaluation.
        
        Backs up current parameters so they can be restored later.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Backup current parameters
                self.backup[name] = param.data.clone()
                # Apply shadow parameters
                if name in self.shadow:
                    param.data = self.shadow[name].clone()
    
    def restore(self) -> None:
        """
        Restore original parameters after evaluation.
        
        Should be called after apply_shadow() and evaluation.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.backup:
                    param.data = self.backup[name]
        
        # Clear backup
        self.backup = {}
    
    def get_shadow_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get the shadow parameters as a state dict.
        
        Returns:
            Dictionary of shadow parameter tensors
        """
        return {name: tensor.clone() for name, tensor in self.shadow.items()}
    
    def load_shadow_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        Load shadow parameters from a state dict.
        
        Args:
            state_dict: Dictionary of shadow parameter tensors
        """
        for name, tensor in state_dict.items():
            if name in self.shadow:
                self.shadow[name] = tensor.clone()


def save_model(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    step: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    ema: Optional[ModelEMA] = None,
    additional_state: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save model checkpoint with optional training state.
    
    Args:
        model: Model to save
        filepath: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Optional current epoch
        step: Optional current step
        metrics: Optional metrics dictionary
        ema: Optional ModelEMA instance
        additional_state: Optional additional state to save
        
    Returns:
        Path to saved checkpoint
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if step is not None:
        checkpoint['step'] = step
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.get_shadow_state_dict()
    
    if additional_state is not None:
        checkpoint.update(additional_state)
    
    torch.save(checkpoint, filepath)
    
    size_mb = filepath.stat().st_size / (1024 * 1024)
    logger.info(f"Model saved: {filepath} ({size_mb:.1f} MB)")
    
    return filepath


def load_model(
    model: nn.Module,
    filepath: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema: Optional[ModelEMA] = None,
    device: Optional[torch.device] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint with optional training state.
    
    Args:
        model: Model to load weights into
        filepath: Path to checkpoint
        optimizer: Optional optimizer to load state into
        ema: Optional ModelEMA to load state into
        device: Device to map tensors to
        strict: Whether to strictly enforce state dict keys match
        
    Returns:
        Dictionary with loaded checkpoint data (epoch, step, metrics, etc.)
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    if device is None:
        device = config.get_device()
    
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
    
    # Load optimizer state if available
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load EMA state if available
    if ema is not None and 'ema_state_dict' in checkpoint:
        ema.load_shadow_state_dict(checkpoint['ema_state_dict'])
    
    logger.info(f"Model loaded: {filepath}")
    if 'epoch' in checkpoint:
        logger.info(f"  Epoch: {checkpoint['epoch']}")
    if 'metrics' in checkpoint:
        logger.info(f"  Metrics: {checkpoint['metrics']}")
    
    return checkpoint


def apply_lora(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to a transformer model.
    
    This function wraps the PEFT library's LoRA implementation
    to apply low-rank adapters to specified modules.
    
    Args:
        model: Model to apply LoRA to
        r: LoRA rank
        alpha: LoRA alpha parameter (scaling)
        dropout: LoRA dropout
        target_modules: List of module names to apply LoRA to
        
    Returns:
        Model with LoRA applied
        
    Validates: Requirements 4.8, Property 14 (LoRA parameter efficiency)
    """
    try:
        from peft import get_peft_model, LoraConfig, TaskType
    except ImportError:
        logger.warning("PEFT not installed. Skipping LoRA application.")
        return model
    
    if target_modules is None:
        target_modules = config.LORA_TARGET_MODULES
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    
    model = get_peft_model(model, lora_config)
    
    # Log parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = 100 * trainable_params / total_params
    
    logger.info(f"LoRA applied:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_ratio:.2f}%")
    
    return model


def freeze_model_layers(
    model: nn.Module,
    freeze_patterns: Optional[list] = None,
    unfreeze_patterns: Optional[list] = None
) -> None:
    """
    Freeze or unfreeze model layers based on name patterns.
    
    Args:
        model: Model to modify
        freeze_patterns: List of patterns to freeze (if name contains pattern)
        unfreeze_patterns: List of patterns to unfreeze
    """
    for name, param in model.named_parameters():
        should_freeze = False
        should_unfreeze = False
        
        if freeze_patterns:
            should_freeze = any(p in name for p in freeze_patterns)
        
        if unfreeze_patterns:
            should_unfreeze = any(p in name for p in unfreeze_patterns)
        
        # Unfreeze takes precedence
        if should_unfreeze:
            param.requires_grad = True
        elif should_freeze:
            param.requires_grad = False


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    Count model parameters.
    
    Args:
        model: Model to count parameters for
        only_trainable: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_parameter_groups(
    model: nn.Module,
    base_lr: float,
    weight_decay: float = 0.01,
    layer_decay: float = 0.95,
    no_decay_patterns: Optional[list] = None
) -> list:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Args:
        model: Model to create parameter groups for
        base_lr: Base learning rate
        weight_decay: Weight decay value
        layer_decay: Learning rate decay per layer (e.g., 0.95)
        no_decay_patterns: Patterns for parameters without weight decay
        
    Returns:
        List of parameter group dictionaries for optimizer
    """
    if no_decay_patterns is None:
        no_decay_patterns = ['bias', 'LayerNorm', 'layer_norm', 'bn']
    
    # Collect all parameters with their layer depths
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine if weight decay should be applied
        apply_decay = not any(p in name for p in no_decay_patterns)
        wd = weight_decay if apply_decay else 0.0
        
        # Determine layer depth for learning rate decay
        # This is a simplified version - for transformer models,
        # you might want to extract layer indices
        depth = name.count('.')
        lr_scale = layer_decay ** depth
        
        # Create group key
        group_key = (wd, lr_scale)
        
        if group_key not in param_groups:
            param_groups[group_key] = {
                'params': [],
                'weight_decay': wd,
                'lr': base_lr * lr_scale
            }
        
        param_groups[group_key]['params'].append(param)
    
    return list(param_groups.values())


class GradientClipper:
    """
    Utility class for gradient clipping with logging.
    
    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm (default: 2.0 for L2)
        
    Validates: Requirements 5.5, Property 19 (Gradient clipping enforcement)
    """
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.grad_norms = []
    
    def clip(self, model: nn.Module) -> float:
        """
        Clip gradients and return the gradient norm before clipping.
        
        Args:
            model: Model whose gradients to clip
            
        Returns:
            Gradient norm before clipping
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        
        self.grad_norms.append(float(grad_norm))
        
        return float(grad_norm)
    
    def get_average_norm(self) -> float:
        """Get average gradient norm."""
        if not self.grad_norms:
            return 0.0
        return sum(self.grad_norms) / len(self.grad_norms)
    
    def reset(self) -> None:
        """Reset gradient norm history."""
        self.grad_norms = []
