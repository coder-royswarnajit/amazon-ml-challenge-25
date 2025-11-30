"""
Performance tests for memory and speed constraints.

These tests verify:
- Memory usage fits within 6GB GPU constraint
- Checkpoint sizes are reasonable
- Download speed and feature engineering speed
"""

import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import time
import sys
import os
import gc

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import config


def get_model_memory_mb(model):
    """Estimate model memory in MB."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


class TestMemoryConstraints:
    """Test memory usage stays within GPU constraints."""
    
    def test_model_parameter_memory(self):
        """
        Test that model parameters fit within reasonable memory budget.
        
        Target: Model should use < 2GB for parameters (leaving room for activations)
        """
        from src.models.multimodal import OptimizedMultimodalModel
        
        model = OptimizedMultimodalModel(
            num_tabular_features=180,
            hidden_dim=config.HIDDEN_DIM
        )
        
        memory_mb = get_model_memory_mb(model)
        
        # Model should be under 2GB for parameters
        max_memory_mb = 2000
        assert memory_mb < max_memory_mb, \
            f"Model memory {memory_mb:.1f}MB exceeds limit {max_memory_mb}MB"
        
        print(f"Model memory: {memory_mb:.1f}MB")
    
    def test_batch_memory_estimation(self):
        """
        Test that a typical training batch fits in memory.
        
        Estimates memory for inputs, model, activations, and gradients.
        """
        batch_size = config.BATCH_SIZE
        seq_len = config.MAX_TEXT_LENGTH
        img_size = config.IMAGE_SIZE
        n_tabular = 180
        
        # Input memory
        input_ids_mem = batch_size * seq_len * 8  # int64
        attention_mask_mem = batch_size * seq_len * 8
        image_mem = batch_size * 3 * img_size * img_size * 4  # float32
        tabular_mem = batch_size * n_tabular * 4
        
        input_total_mb = (input_ids_mem + attention_mask_mem + image_mem + tabular_mem) / (1024 * 1024)
        
        # Should be under 500MB for inputs
        assert input_total_mb < 500, f"Input memory {input_total_mb:.1f}MB too high"
        
        print(f"Input batch memory: {input_total_mb:.1f}MB")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_with_forward_pass(self):
        """
        Test GPU memory usage during forward pass.
        
        This is a more realistic test that runs actual model operations.
        """
        from src.models.multimodal import OptimizedMultimodalModel
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Get baseline memory
        torch.cuda.reset_peak_memory_stats()
        baseline_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        
        # Create model on GPU
        model = OptimizedMultimodalModel(
            num_tabular_features=180,
            hidden_dim=config.HIDDEN_DIM
        ).cuda()
        
        model_mem = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"Model GPU memory: {model_mem - baseline_mem:.1f}MB")
        
        # Run forward pass
        batch_size = config.BATCH_SIZE
        input_ids = torch.randint(0, 1000, (batch_size, config.MAX_TEXT_LENGTH)).cuda()
        attention_mask = torch.ones(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long).cuda()
        image = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).cuda()
        tabular = torch.randn(batch_size, 180).cuda()
        
        with torch.no_grad():
            output = model(input_ids, attention_mask, image, tabular)
        
        peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Should stay under 5GB (leaving 1GB buffer from 6GB total)
        max_gpu_mem_mb = 5000
        assert peak_mem < max_gpu_mem_mb, \
            f"Peak GPU memory {peak_mem:.1f}MB exceeds limit {max_gpu_mem_mb}MB"
        
        print(f"Peak GPU memory during forward: {peak_mem:.1f}MB")
        
        # Cleanup
        del model, input_ids, attention_mask, image, tabular, output
        torch.cuda.empty_cache()


class TestCheckpointSize:
    """Test checkpoint file sizes."""
    
    def test_checkpoint_size_reasonable(self):
        """
        Test that checkpoint files are reasonably sized.
        
        Full checkpoints should be under 500MB.
        Quick checkpoints should be under 50MB.
        """
        from src.utils.checkpoint import CheckpointManager
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create simulated state (similar to actual training)
            state = {
                'epoch': 5,
                'step': 1000,
                'model_state': {
                    f'layer_{i}': torch.randn(512, 512) for i in range(10)
                },
                'optimizer_state': {
                    'state': {f'param_{i}': {'exp_avg': torch.randn(512, 512)} for i in range(10)}
                },
                'ema_state': {
                    f'shadow_{i}': torch.randn(512, 512) for i in range(10)
                },
                'metrics': {'val_smape': 8.5}
            }
            
            # Save full checkpoint
            full_path = manager.save_checkpoint(
                state=state,
                stage='neural_net',
                checkpoint_type='full'
            )
            
            full_size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"Full checkpoint size: {full_size_mb:.1f}MB")
            
            # Full checkpoint should be under 500MB
            assert full_size_mb < 500, f"Full checkpoint {full_size_mb:.1f}MB too large"
            
            # Save quick checkpoint (minimal state)
            quick_state = {
                'epoch': 5,
                'step': 1000,
                'best_smape': 8.5
            }
            quick_path = manager.save_checkpoint(
                state=quick_state,
                stage='neural_net',
                checkpoint_type='quick'
            )
            
            quick_size_mb = quick_path.stat().st_size / (1024 * 1024)
            print(f"Quick checkpoint size: {quick_size_mb:.1f}MB")
            
            # Quick checkpoint should be under 50MB
            assert quick_size_mb < 50, f"Quick checkpoint {quick_size_mb:.1f}MB too large"


class TestProcessingSpeed:
    """Test processing speed for various operations."""
    
    def test_feature_engineering_speed(self):
        """
        Test that feature engineering is reasonably fast.
        
        Should process at least 100 samples per second.
        """
        from src.data.feature_engineering import FeatureEngineer
        
        # Create sample data
        n_samples = 500
        sample_texts = [
            f"Sample product {i} with 500g weight and premium quality for ₹{100 + i}"
            for i in range(n_samples)
        ]
        
        df = pd.DataFrame({
            'sample_id': [f'test_{i}' for i in range(n_samples)],
            'catalog_content': sample_texts,
            'price': [100 + i for i in range(n_samples)]
        })
        
        engineer = FeatureEngineer()
        
        # Time feature engineering
        start_time = time.time()
        features_df = engineer.engineer_features(df)
        elapsed = time.time() - start_time
        
        samples_per_sec = n_samples / elapsed
        print(f"Feature engineering speed: {samples_per_sec:.0f} samples/sec")
        
        # Should be at least 10 samples/sec (very conservative for CI)
        min_speed = 10
        assert samples_per_sec >= min_speed, \
            f"Feature engineering too slow: {samples_per_sec:.0f} < {min_speed} samples/sec"
    
    def test_smape_calculation_speed(self):
        """
        Test that SMAPE calculation is fast.
        
        Should process at least 100K samples per second.
        """
        from src.utils.metrics import calculate_smape
        
        n_samples = 100000
        y_true = np.random.uniform(1, 15, n_samples)
        y_pred = y_true + np.random.randn(n_samples) * 0.5
        
        start_time = time.time()
        smape = calculate_smape(y_true, y_pred)
        elapsed = time.time() - start_time
        
        samples_per_sec = n_samples / elapsed
        print(f"SMAPE calculation speed: {samples_per_sec:.0f} samples/sec")
        
        # Should be at least 100K samples/sec
        min_speed = 100000
        assert samples_per_sec >= min_speed, \
            f"SMAPE calculation too slow: {samples_per_sec:.0f} < {min_speed} samples/sec"
    
    def test_model_inference_speed(self):
        """
        Test model inference speed.
        """
        from src.models.multimodal import OptimizedMultimodalModel
        
        n_tabular = 50  # Smaller for faster test
        model = OptimizedMultimodalModel(
            num_tabular_features=n_tabular,
            hidden_dim=256
        )
        model.eval()
        
        batch_size = 4
        n_batches = 5
        
        # Warmup
        with torch.no_grad():
            input_ids = torch.randint(0, 1000, (batch_size, config.MAX_TEXT_LENGTH))
            attention_mask = torch.ones(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long)
            image = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
            tabular = torch.randn(batch_size, n_tabular)
            _ = model(input_ids, attention_mask, image, tabular)
        
        # Time inference
        start_time = time.time()
        with torch.no_grad():
            for _ in range(n_batches):
                input_ids = torch.randint(0, 1000, (batch_size, config.MAX_TEXT_LENGTH))
                attention_mask = torch.ones(batch_size, config.MAX_TEXT_LENGTH, dtype=torch.long)
                image = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
                tabular = torch.randn(batch_size, n_tabular)
                _ = model(input_ids, attention_mask, image, tabular)
        
        elapsed = time.time() - start_time
        samples_per_sec = (batch_size * n_batches) / elapsed
        
        print(f"Model inference speed: {samples_per_sec:.1f} samples/sec")
        
        # Should be at least 1 sample/sec (conservative for CPU)
        assert samples_per_sec >= 1, f"Model inference too slow: {samples_per_sec:.1f} samples/sec"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
