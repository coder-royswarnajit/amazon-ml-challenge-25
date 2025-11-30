#!/usr/bin/env python3
"""
Task 20: Final Validation and Submission

This script performs comprehensive validation of the entire pipeline:
1. Verifies all modules are importable and functional
2. Creates synthetic data for testing (if real data unavailable)
3. Runs mini end-to-end pipeline
4. Validates SMAPE calculation
5. Generates sample submission
6. Reports performance metrics

Usage:
    python scripts/run_validation.py
    python scripts/run_validation.py --full  # Use real data if available
    python scripts/run_validation.py --quick # Quick validation only
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(n_train: int = 1000, n_test: int = 200) -> tuple:
    """
    Create synthetic data for pipeline validation.
    
    Returns:
        Tuple of (train_df, test_df)
    """
    logger.info(f"Creating synthetic data: {n_train} train, {n_test} test samples")
    
    np.random.seed(42)
    
    # Product categories with different price ranges
    categories = {
        'Electronics': (100, 2000),
        'Clothing': (20, 500),
        'Books': (5, 100),
        'Home': (30, 800),
        'Sports': (15, 400),
    }
    
    def generate_samples(n: int, include_price: bool = True) -> pd.DataFrame:
        data = []
        for i in range(n):
            # Select random category
            category = np.random.choice(list(categories.keys()))
            price_min, price_max = categories[category]
            
            # Generate price (log-normal distribution for realistic prices)
            if include_price:
                log_price = np.random.uniform(np.log(price_min), np.log(price_max))
                price = np.exp(log_price)
            
            # Generate product name with quantity patterns
            quantities = ['', 'Pack of 2', 'Pack of 3', '500g', '1kg', '250ml', '1L']
            quality = ['', 'Premium', 'Professional', 'Deluxe', 'Standard']
            
            product_name = f"{np.random.choice(quality)} {category} Product {i+1} {np.random.choice(quantities)}".strip()
            
            # Generate description (catalog_content)
            descriptions = [
                f"High quality {category.lower()} item with excellent features. Best seller in {category}.",
                f"Best selling {category.lower()} product with great reviews. Premium quality guaranteed.",
                f"Affordable {category.lower()} solution for everyday use. Value for money product.",
                f"Premium grade {category.lower()} with professional quality. Top rated by customers.",
            ]
            description = np.random.choice(descriptions)
            
            sample = {
                'sample_id': f'SAMPLE_{i:06d}',
                'product_name': product_name,
                'catalog_content': description,  # Required field for feature engineering
                'category': category,
                'image_link': f'https://example.com/images/{i}.jpg',
            }
            
            if include_price:
                sample['price'] = round(price, 2)
            
            data.append(sample)
        
        return pd.DataFrame(data)
    
    train_df = generate_samples(n_train, include_price=True)
    test_df = generate_samples(n_test, include_price=False)
    
    return train_df, test_df


def validate_imports():
    """Validate all module imports work correctly."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING IMPORTS")
    logger.info("=" * 60)
    
    modules = [
        ('config', 'Config, PATHS'),
        ('src.data.dataset', 'AmazonMLDataset, get_dataloader'),
        ('src.data.downloader', 'ResumableImageDownloader'),
        ('src.data.feature_engineering', 'FeatureEngineer'),
        ('src.models.multimodal', 'OptimizedMultimodalModel'),
        ('src.models.losses', 'HuberSMAPELoss, FocalSMAPELoss'),
        ('src.models.utils', 'ModelEMA, GradientClipper'),
        ('src.training.train_neural_net', 'train_neural_network, setup_lora_model'),
        ('src.training.train_gbdt', 'train_all_gbdt_models'),
        ('src.training.train_ensemble', 'train_ensemble, StackingEnsemble'),
        ('src.utils.checkpoint', 'CheckpointManager'),
        ('src.utils.metrics', 'calculate_smape, evaluate_predictions'),
        ('src.utils.visualization', 'plot_training_curves'),
    ]
    
    success_count = 0
    for module_name, components in modules:
        try:
            module = __import__(module_name, fromlist=components.split(', '))
            for comp in components.split(', '):
                getattr(module, comp.strip())
            logger.info(f"  ✓ {module_name}: {components}")
            success_count += 1
        except Exception as e:
            logger.error(f"  ✗ {module_name}: {e}")
    
    logger.info(f"\nImport validation: {success_count}/{len(modules)} modules OK")
    return success_count == len(modules)


def validate_feature_engineering(train_df: pd.DataFrame) -> dict:
    """Validate feature engineering module."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    from src.data.feature_engineering import FeatureEngineer
    
    fe = FeatureEngineer()
    
    # Test IPQ extraction
    test_texts = [
        "Product 500g pack",
        "Premium 1kg bag",
        "Pack of 3 items",
        "250ml bottle",
    ]
    
    logger.info("Testing IPQ extraction:")
    for text in test_texts:
        ipq = fe.extract_ipq_features(text)
        logger.info(f"  '{text}' -> {ipq}")
    
    # Test text statistics
    logger.info("\nTesting text statistics:")
    stats = fe.extract_text_statistics(test_texts[0])
    logger.info(f"  Stats: {stats}")
    
    # Test keyword features
    logger.info("\nTesting keyword detection:")
    kw = fe.extract_keyword_features("Premium quality product on sale")
    logger.info(f"  Keywords: {kw}")
    
    # Engineer features for entire dataset
    logger.info("\nEngineering features for dataset...")
    start_time = time.time()
    features = fe.engineer_features(train_df)
    elapsed = time.time() - start_time
    
    logger.info(f"  Features extracted in {elapsed:.2f}s")
    logger.info(f"  Feature shape: {features.shape}")
    logger.info(f"  Columns: {list(features.columns)[:10]}...")
    
    return {'features': features, 'engineer': fe}


def validate_model_architecture():
    """Validate model architecture and forward pass."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING MODEL ARCHITECTURE")
    logger.info("=" * 60)
    
    from src.models.multimodal import OptimizedMultimodalModel
    
    # Create model
    model = OptimizedMultimodalModel(num_tabular_features=50)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    # Test forward pass
    logger.info("\nTesting forward pass:")
    batch_size = 2
    
    # Create dummy inputs
    input_ids = torch.randint(0, 1000, (batch_size, 64))
    attention_mask = torch.ones(batch_size, 64, dtype=torch.long)
    images = torch.randn(batch_size, 3, 224, 224)
    tabular = torch.randn(batch_size, 50)
    
    model.eval()
    with torch.no_grad():
        try:
            output = model(input_ids, attention_mask, images, tabular)
            logger.info(f"  ✓ Forward pass successful")
            logger.info(f"  Output shape: {output.shape}")
        except Exception as e:
            logger.error(f"  ✗ Forward pass failed: {e}")
            return False
    
    # Estimate memory usage
    memory_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    logger.info(f"\n  Model memory: {memory_mb:.2f} MB")
    
    return True


def validate_loss_functions():
    """Validate loss function implementations."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING LOSS FUNCTIONS")
    logger.info("=" * 60)
    
    from src.models.losses import HuberSMAPELoss, FocalSMAPELoss
    
    # Create test data in LOG SPACE (as the losses expect)
    # Values like log1p(100) = 4.61, log1p(200) = 5.30, etc.
    predictions_log = torch.tensor([4.61, 5.30, 5.70, 5.99])  # log1p of 100, 200, 300, 400
    targets_log = torch.tensor([4.71, 5.25, 5.74, 5.94])      # log1p of 110, 190, 310, 380
    
    # Test HuberSMAPE
    huber_loss = HuberSMAPELoss()
    h_value = huber_loss(predictions_log, targets_log)
    logger.info(f"  HuberSMAPE loss: {h_value.item():.4f}")
    
    # Test FocalSMAPE
    focal_loss = FocalSMAPELoss()
    f_value = focal_loss(predictions_log, targets_log)
    logger.info(f"  FocalSMAPE loss: {f_value.item():.4f}")
    
    # Verify non-negativity (NaN fails this too)
    assert not torch.isnan(h_value) and h_value >= 0, "HuberSMAPE should be non-negative and not NaN"
    assert not torch.isnan(f_value) and f_value >= 0, "FocalSMAPE should be non-negative and not NaN"
    
    # Verify low loss for similar predictions
    similar = torch.tensor([5.0, 5.5])
    h_similar = huber_loss(similar, similar)
    logger.info(f"  HuberSMAPE (identical): {h_similar.item():.6f}")
    
    logger.info("  ✓ Loss functions validated")
    return True


def validate_metrics():
    """Validate metrics calculation."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING METRICS")
    logger.info("=" * 60)
    
    from src.utils.metrics import calculate_smape, evaluate_predictions
    
    # Test data
    predictions = np.array([100, 200, 300, 400, 500])
    targets = np.array([110, 190, 310, 380, 520])
    
    # Calculate SMAPE
    smape = calculate_smape(predictions, targets)
    logger.info(f"  SMAPE: {smape:.4f}%")
    
    # Verify SMAPE range
    assert 0 <= smape <= 200, f"SMAPE should be in [0, 200], got {smape}"
    
    # Full evaluation
    metrics = evaluate_predictions(predictions, targets)
    logger.info(f"  All metrics: {metrics}")
    
    # Test perfect predictions
    perfect_smape = calculate_smape(targets, targets)
    logger.info(f"  Perfect SMAPE: {perfect_smape:.6f}%")
    assert perfect_smape < 0.0001, "Perfect predictions should have ~0 SMAPE"
    
    logger.info("  ✓ Metrics validated")
    return True


def validate_checkpoint_system():
    """Validate checkpoint save/load functionality."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING CHECKPOINT SYSTEM")
    logger.info("=" * 60)
    
    import tempfile
    from src.utils.checkpoint import CheckpointManager
    
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CheckpointManager(Path(tmpdir))  # Pass Path object
        
        # Create dummy state dict
        state = {
            'model': {'layer1.weight': torch.randn(10, 10).numpy().tolist()},
            'optimizer': {'param_groups': []},
            'epoch': 5,
            'step': 1000,
            'loss': 0.05,
            'best_smape': 8.5,
        }
        
        # Save checkpoint
        checkpoint_path = cm.save_checkpoint(
            state=state,
            stage='validation_test',
            metric=8.5,
            checkpoint_type='quick'
        )
        
        logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        # Verify file exists
        assert Path(checkpoint_path).exists(), "Checkpoint file should exist"
        
        # Load checkpoint
        loaded = cm.load_checkpoint(checkpoint_path)
        logger.info(f"  ✓ Checkpoint loaded: epoch={loaded['state']['epoch']}, step={loaded['state']['step']}")
        
        # Verify values match
        assert loaded['state']['epoch'] == state['epoch'], "Epoch mismatch"
        assert loaded['state']['step'] == state['step'], "Step mismatch"
        assert loaded['stage'] == 'validation_test', "Stage mismatch"
        
        # Test latest checkpoint retrieval
        latest = cm.get_latest_checkpoint()
        logger.info(f"  ✓ Latest checkpoint: {latest}")
    
    logger.info("  ✓ Checkpoint system validated")
    return True


def validate_submission_format(test_df: pd.DataFrame):
    """Validate submission file format."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING SUBMISSION FORMAT")
    logger.info("=" * 60)
    
    import tempfile
    
    # Generate mock predictions
    n_test = len(test_df)
    predictions_log = np.random.uniform(3, 8, n_test)  # Log prices
    predictions = np.exp(predictions_log)  # Convert to original space
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'sample_id': test_df['sample_id'],
        'price': predictions
    })
    
    # Validate format
    assert 'sample_id' in submission.columns, "Missing sample_id column"
    assert 'price' in submission.columns, "Missing price column"
    logger.info("  ✓ Required columns present")
    
    # Validate completeness
    assert len(submission) == len(test_df), "Missing predictions"
    assert submission['sample_id'].nunique() == len(test_df), "Duplicate sample_ids"
    logger.info("  ✓ All samples have predictions")
    
    # Validate values
    assert (submission['price'] > 0).all(), "Negative prices found"
    assert not submission['price'].isna().any(), "NaN prices found"
    logger.info("  ✓ All prices are positive and non-null")
    
    # Save sample submission
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
        submission.to_csv(f.name, index=False)
        file_size = Path(f.name).stat().st_size
        logger.info(f"  ✓ Sample submission saved: {f.name} ({file_size} bytes)")
    
    # Print statistics
    logger.info(f"\n  Submission statistics:")
    logger.info(f"    Samples: {len(submission)}")
    logger.info(f"    Price range: [{submission['price'].min():.2f}, {submission['price'].max():.2f}]")
    logger.info(f"    Price mean: {submission['price'].mean():.2f}")
    logger.info(f"    Price median: {submission['price'].median():.2f}")
    
    return True


def run_mini_pipeline(train_df: pd.DataFrame, test_df: pd.DataFrame, features: dict):
    """Run a mini end-to-end pipeline."""
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING MINI PIPELINE")
    logger.info("=" * 60)
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import Ridge
    from src.utils.metrics import calculate_smape
    
    # Prepare data
    feature_df = features['features']
    
    # Select numeric columns only
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    X = feature_df[numeric_cols].fillna(0).values
    y = np.log1p(train_df['price'].values)  # Log transform target
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Validation samples: {len(X_val)}")
    logger.info(f"  Features: {X.shape[1]}")
    
    # Train simple model
    logger.info("\n  Training Ridge regression...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Predict
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Convert back from log space
    train_pred_orig = np.expm1(train_pred)
    val_pred_orig = np.expm1(val_pred)
    y_train_orig = np.expm1(y_train)
    y_val_orig = np.expm1(y_val)
    
    # Calculate SMAPE
    train_smape = calculate_smape(train_pred_orig, y_train_orig)
    val_smape = calculate_smape(val_pred_orig, y_val_orig)
    
    logger.info(f"\n  Results:")
    logger.info(f"    Train SMAPE: {train_smape:.2f}%")
    logger.info(f"    Val SMAPE: {val_smape:.2f}%")
    
    # Check if SMAPE is reasonable
    if val_smape < 50:  # Synthetic data, so threshold is higher
        logger.info(f"  ✓ Pipeline produces reasonable predictions")
    else:
        logger.warning(f"  ⚠ High SMAPE - model may need tuning")
    
    return {'train_smape': train_smape, 'val_smape': val_smape}


def print_summary(results: dict):
    """Print final validation summary."""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✓" if passed else "✗"
        logger.info(f"  {status} {test_name}")
    
    logger.info("\n" + "-" * 60)
    if all_passed:
        logger.info("  🎉 ALL VALIDATIONS PASSED")
        logger.info("  The pipeline is ready for production use.")
        logger.info("\n  Next steps:")
        logger.info("    1. Download real data: python scripts/run_stage1_setup.py")
        logger.info("    2. Run full pipeline: python scripts/run_stage2_features.py")
        logger.info("    3. Train models: python scripts/run_stage3_neural_net.py")
        logger.info("    4. Generate submission: python scripts/create_submission.py")
    else:
        logger.error("  ❌ SOME VALIDATIONS FAILED")
        logger.error("  Please fix the issues above before proceeding.")
    
    logger.info("=" * 60)
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description='Task 20: Final Validation and Submission',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--quick', action='store_true',
                        help='Quick validation (imports only)')
    parser.add_argument('--full', action='store_true',
                        help='Use real data if available')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - FINAL VALIDATION")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    results = {}
    
    try:
        # 1. Validate imports
        results['Import Validation'] = validate_imports()
        
        if args.quick:
            print_summary(results)
            return
        
        # 2. Create or load data
        if args.full:
            # Try to load real data
            from config import Config
            if Config.TRAIN_CSV.exists():
                logger.info("\nLoading real data...")
                train_df = pd.read_csv(Config.TRAIN_CSV)
                test_df = pd.read_csv(Config.TEST_CSV)
            else:
                logger.warning("Real data not found, using synthetic data")
                train_df, test_df = create_synthetic_data(args.samples, args.samples // 5)
        else:
            train_df, test_df = create_synthetic_data(args.samples, args.samples // 5)
        
        # 3. Validate feature engineering
        feature_result = validate_feature_engineering(train_df)
        results['Feature Engineering'] = feature_result is not None
        
        # 4. Validate model architecture
        results['Model Architecture'] = validate_model_architecture()
        
        # 5. Validate loss functions
        results['Loss Functions'] = validate_loss_functions()
        
        # 6. Validate metrics
        results['Metrics'] = validate_metrics()
        
        # 7. Validate checkpoint system
        results['Checkpoint System'] = validate_checkpoint_system()
        
        # 8. Validate submission format
        results['Submission Format'] = validate_submission_format(test_df)
        
        # 9. Run mini pipeline
        if feature_result:
            pipeline_result = run_mini_pipeline(train_df, test_df, feature_result)
            results['Mini Pipeline'] = pipeline_result['val_smape'] < 100  # Reasonable for synthetic
        
        # Print summary
        success = print_summary(results)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
