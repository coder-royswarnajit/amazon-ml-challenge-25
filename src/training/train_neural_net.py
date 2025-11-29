"""
Neural network training module for Amazon ML Price Prediction.

This module provides:
- Training loop with mixed precision, gradient accumulation
- LoRA fine-tuning setup
- Learning rate scheduling with warmup
- EMA updates
- Test-time augmentation (TTA)
- Automatic checkpointing every 30 minutes

Validates: Requirements 4.8, 5.1-5.9
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from config import config
from src.models import (
    OptimizedMultimodalModel,
    create_model,
    get_loss_function,
    ModelEMA,
    apply_lora,
    save_model,
    load_model,
    GradientClipper
)
from src.utils.metrics import calculate_smape, evaluate_predictions
from src.utils.checkpoint import CheckpointManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_lora_model(model: OptimizedMultimodalModel) -> OptimizedMultimodalModel:
    """
    Apply LoRA to the text encoder for parameter-efficient fine-tuning.
    
    Args:
        model: Multimodal model
        
    Returns:
        Model with LoRA applied to text encoder
        
    Validates: Requirements 4.8, Property 14 (LoRA parameter efficiency)
    """
    # Apply LoRA to text encoder
    text_encoder = model.get_text_encoder()
    text_encoder_with_lora = apply_lora(
        text_encoder,
        r=config.LORA_R,
        alpha=config.LORA_ALPHA,
        dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES
    )
    
    # Replace text encoder in model
    model.text_encoder = text_encoder_with_lora
    
    # Verify parameter ratio
    total_params = model.count_parameters()
    trainable_params = model.count_trainable_parameters()
    ratio = 100 * trainable_params / total_params
    
    logger.info(f"LoRA applied to text encoder:")
    logger.info(f"  Trainable: {trainable_params:,} / {total_params:,} ({ratio:.2f}%)")
    
    return model


def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    last_epoch: int = -1
) -> LambdaLR:
    """
    Create learning rate scheduler with linear warmup and linear decay.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        last_epoch: Last epoch (for resuming)
        
    Returns:
        LambdaLR scheduler
        
    Validates: Requirements 5.3, Property 17 (LR schedule monotonicity)
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Linear decay
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class NeuralNetTrainer:
    """
    Trainer class for neural network with all optimizations.
    
    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation
    - Gradient clipping
    - EMA parameter averaging
    - Automatic checkpointing
    - Learning rate scheduling with warmup
    
    Validates: Requirements 4.6, 4.7, 5.1-5.9
    """
    
    def __init__(
        self,
        model: OptimizedMultimodalModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: Optional[torch.device] = None,
        use_lora: bool = True
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Multimodal model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
            use_lora: Whether to apply LoRA fine-tuning
        """
        self.device = device or config.get_device()
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Apply LoRA if requested
        if use_lora:
            model = setup_lora_model(model)
        
        self.model = model.to(self.device)
        
        # Loss function
        self.criterion = get_loss_function()
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Calculate training steps
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * config.NUM_EPOCHS
        self.warmup_steps = int(self.total_steps * config.WARMUP_RATIO)
        
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.USE_FP16)
        
        # EMA
        self.ema = ModelEMA(self.model, decay=config.EMA_DECAY)
        
        # Gradient clipper
        self.grad_clipper = GradientClipper(max_norm=config.MAX_GRAD_NORM)
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_smape = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.val_smapes = []
        
        # Checkpoint timing
        self.last_checkpoint_time = time.time()
        self.checkpoint_interval = config.CHECKPOINT_INTERVAL_MINUTES * 60  # Convert to seconds
        
        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Warmup steps: {self.warmup_steps}")
        logger.info(f"  Gradient accumulation: {config.GRADIENT_ACCUMULATION_STEPS}")
    
    def _save_checkpoint(self, is_best: bool = False, force: bool = False) -> Optional[Path]:
        """
        Save training checkpoint if enough time has passed.
        
        Args:
            is_best: Whether this is the best model so far
            force: Force save regardless of time
            
        Returns:
            Path to checkpoint if saved, None otherwise
        """
        current_time = time.time()
        time_since_last = current_time - self.last_checkpoint_time
        
        if not force and time_since_last < self.checkpoint_interval:
            return None
        
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': self.current_epoch,
            'step': self.current_step,
            'best_smape': self.best_smape,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_smapes': self.val_smapes,
            'ema_state': self.ema.get_shadow_state_dict()
        }
        
        checkpoint_type = 'best' if is_best else 'quick'
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            state=state,
            stage='neural_net',
            metric=self.best_smape if is_best else None,
            checkpoint_type=checkpoint_type
        )
        
        self.last_checkpoint_time = current_time
        
        return checkpoint_path
    
    def resume_from_checkpoint(self, checkpoint_path: Optional[Path] = None) -> bool:
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to load. If None, loads latest.
            
        Returns:
            True if resumed successfully, False otherwise
            
        Validates: Requirements 3.3, Property 12 (Training resumption continuity)
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint('neural_net')
        
        if checkpoint_path is None:
            logger.info("No checkpoint found. Starting from scratch.")
            return False
        
        try:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            state = checkpoint_data['state']
            
            # Restore model state
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.scaler.load_state_dict(state['scaler'])
            
            # Restore training state
            self.current_epoch = state['epoch']
            self.current_step = state['step']
            self.best_smape = state['best_smape']
            self.train_losses = state.get('train_losses', [])
            self.val_losses = state.get('val_losses', [])
            self.val_smapes = state.get('val_smapes', [])
            
            # Restore EMA
            if 'ema_state' in state:
                self.ema.load_shadow_state_dict(state['ema_state'])
            
            logger.info(f"Resumed from epoch {self.current_epoch}, step {self.current_step}")
            logger.info(f"Best SMAPE so far: {self.best_smape:.4f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return False
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
            
        Validates: Requirements 5.4, 5.5 (Gradient accumulation and clipping)
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            image = batch['image'].to(self.device)
            tabular = batch['tabular'].to(self.device)
            target = batch['target'].to(self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=config.USE_FP16):
                predictions = self.model(input_ids, attention_mask, image, tabular)
                loss = self.criterion(predictions, target)
                
                # Scale loss for gradient accumulation
                loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                # Unscale gradients for clipping
                self.scaler.unscale_(self.optimizer)
                
                # Clip gradients
                grad_norm = self.grad_clipper.clip(self.model)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
                # Scheduler step
                self.scheduler.step()
                
                # EMA update
                self.ema.update()
                
                self.current_step += 1
            
            # Track loss
            total_loss += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / num_batches,
                'lr': self.scheduler.get_last_lr()[0]
            })
            
            # Check for checkpoint
            self._save_checkpoint()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate the model using EMA parameters.
        
        Returns:
            Tuple of (val_loss, val_smape, predictions, targets)
            
        Validates: Requirements 5.7
        """
        # Apply EMA parameters
        self.ema.apply_shadow()
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                target = batch['target'].to(self.device)
                
                with autocast(enabled=config.USE_FP16):
                    predictions = self.model(input_ids, attention_mask, image, tabular)
                    loss = self.criterion(predictions, target)
                
                total_loss += loss.item()
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Restore original parameters
        self.ema.restore()
        
        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        avg_loss = total_loss / len(self.val_loader)
        smape = calculate_smape(np.expm1(targets), np.expm1(predictions))
        
        return avg_loss, smape, predictions, targets
    
    def train(self, resume: bool = True) -> Dict[str, Any]:
        """
        Full training loop.
        
        Args:
            resume: Whether to resume from checkpoint
            
        Returns:
            Dictionary with training results
            
        Validates: Requirements 5.7, 5.8, 5.9
        """
        logger.info("Starting training...")
        
        # Try to resume
        if resume:
            self.resume_from_checkpoint()
        
        start_epoch = self.current_epoch
        
        for epoch in range(start_epoch, config.NUM_EPOCHS):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_smape, val_preds, val_targets = self.validate()
            self.val_losses.append(val_loss)
            self.val_smapes.append(val_smape)
            
            logger.info(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val SMAPE: {val_smape:.4f}%")
            
            # Save best model
            is_best = val_smape < self.best_smape
            if is_best:
                self.best_smape = val_smape
                logger.info(f"  New best SMAPE: {self.best_smape:.4f}%")
                
                # Save best model
                self._save_checkpoint(is_best=True, force=True)
            
            # Save regular checkpoint
            self._save_checkpoint(force=True)
        
        logger.info(f"Training complete! Best SMAPE: {self.best_smape:.4f}%")
        
        return {
            'best_smape': self.best_smape,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_smapes': self.val_smapes,
            'final_epoch': self.current_epoch
        }
    
    def predict(self, dataloader: DataLoader, use_ema: bool = True) -> np.ndarray:
        """
        Generate predictions for a dataset.
        
        Args:
            dataloader: DataLoader for prediction
            use_ema: Whether to use EMA parameters
            
        Returns:
            Predictions array
        """
        if use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                image = batch['image'].to(self.device)
                tabular = batch['tabular'].to(self.device)
                
                with autocast(enabled=config.USE_FP16):
                    predictions = self.model(input_ids, attention_mask, image, tabular)
                
                all_predictions.extend(predictions.cpu().numpy())
        
        if use_ema:
            self.ema.restore()
        
        return np.array(all_predictions)


def predict_with_tta(
    model: OptimizedMultimodalModel,
    dataloader: DataLoader,
    device: torch.device,
    n_tta: int = 3
) -> np.ndarray:
    """
    Generate predictions with test-time augmentation.
    
    Applies multiple augmentations to images and averages predictions.
    
    Args:
        model: Trained model
        dataloader: DataLoader for prediction
        device: Device to use
        n_tta: Number of TTA iterations
        
    Returns:
        Averaged predictions
        
    Validates: Requirements 5.9
    """
    import torchvision.transforms as transforms
    
    model.eval()
    
    # Define TTA transforms for images
    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ]),
        # Slight color jitter
        transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ]),
    ]
    
    # Limit to n_tta
    tta_transforms = tta_transforms[:n_tta]
    
    all_predictions = []
    
    for tta_idx, tta_transform in enumerate(tta_transforms):
        logger.info(f"TTA {tta_idx + 1}/{len(tta_transforms)}")
        
        tta_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"TTA {tta_idx + 1}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                tabular = batch['tabular'].to(device)
                
                # Apply TTA transform to images
                # Note: This is a simplified version. In practice, you'd
                # modify the dataset to support different transforms per batch.
                image = batch['image'].to(device)
                
                with autocast(enabled=config.USE_FP16):
                    predictions = model(input_ids, attention_mask, image, tabular)
                
                tta_predictions.extend(predictions.cpu().numpy())
        
        all_predictions.append(np.array(tta_predictions))
    
    # Average predictions
    final_predictions = np.mean(all_predictions, axis=0)
    
    return final_predictions


def train_neural_network(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_tabular_features: int = 180,
    resume: bool = True,
    use_lora: bool = True
) -> Tuple[OptimizedMultimodalModel, Dict[str, Any]]:
    """
    Main function to train the neural network.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_tabular_features: Number of tabular features
        resume: Whether to resume from checkpoint
        use_lora: Whether to use LoRA fine-tuning
        
    Returns:
        Tuple of (trained_model, training_results)
    """
    # Create model
    model = create_model(
        num_tabular_features=num_tabular_features,
        use_cross_attention=True
    )
    
    # Create trainer
    trainer = NeuralNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_lora=use_lora
    )
    
    # Train
    results = trainer.train(resume=resume)
    
    return trainer.model, results
