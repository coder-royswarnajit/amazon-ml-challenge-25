#!/usr/bin/env python3
"""
Stage 2: Feature Engineering

This script handles:
1. Text feature extraction (IPQ, text stats, keywords)
2. TF-IDF vectorization
3. Category encoding
4. Feature saving
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, DATA_CONFIG, FEATURE_CONFIG
from src.data.feature_engineering import (
    ProductFeatureEngineer,
    extract_ipq_features,
    extract_text_statistics,
    extract_keyword_features
)
from src.utils.checkpoint import CheckpointManager

# Configure logging
PATHS['logs_dir'].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATHS['logs_dir'] / 'stage2_features.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load raw training and test data"""
    logger.info("Loading raw data...")
    
    train_path = PATHS['raw_dir'] / 'train.csv'
    test_path = PATHS['raw_dir'] / 'test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"  Training samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    
    return train_df, test_df


def engineer_features(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                      resume_from: str = None):
    """Run feature engineering pipeline"""
    
    checkpoint_manager = CheckpointManager(PATHS['checkpoints_dir'])
    
    # Check for resume
    if resume_from:
        checkpoint = checkpoint_manager.load_checkpoint(Path(resume_from))
        if checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            return checkpoint.get('train_features'), checkpoint.get('test_features')
    
    # Initialize feature engineer
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    engineer = ProductFeatureEngineer(
        max_tfidf_features=FEATURE_CONFIG.get('tfidf_features', 100)
    )
    
    # Fit on training data
    logger.info("\nFitting feature engineering pipeline...")
    train_features = engineer.fit_transform(train_df, 'catalog_content')
    
    # Transform test data
    logger.info("\nTransforming test data...")
    test_features = engineer.transform(test_df, 'catalog_content')
    
    # Log feature information
    logger.info(f"\nFeature engineering complete:")
    logger.info(f"  Training features shape: {train_features.shape}")
    logger.info(f"  Test features shape: {test_features.shape}")
    
    # Feature summary
    feature_cols = train_features.columns.tolist()
    logger.info(f"\nFeature categories:")
    
    ipq_cols = [c for c in feature_cols if c.startswith('ipq_')]
    text_cols = [c for c in feature_cols if c.startswith('text_')]
    keyword_cols = [c for c in feature_cols if c.startswith('has_')]
    tfidf_cols = [c for c in feature_cols if c.startswith('tfidf_')]
    other_cols = [c for c in feature_cols if c not in ipq_cols + text_cols + keyword_cols + tfidf_cols]
    
    logger.info(f"  IPQ features: {len(ipq_cols)}")
    logger.info(f"  Text statistics: {len(text_cols)}")
    logger.info(f"  Keyword features: {len(keyword_cols)}")
    logger.info(f"  TF-IDF features: {len(tfidf_cols)}")
    logger.info(f"  Other features: {len(other_cols)}")
    
    # Check for NaN values
    train_nan = train_features.isna().sum().sum()
    test_nan = test_features.isna().sum().sum()
    logger.info(f"\nNaN values: train={train_nan}, test={test_nan}")
    
    # Save checkpoint
    checkpoint_manager.save_checkpoint(
        state={
            'train_features': train_features,
            'test_features': test_features,
            'feature_columns': feature_cols,
            'engineer_state': engineer.__dict__ if hasattr(engineer, '__dict__') else None
        },
        stage='feature_engineering',
        checkpoint_type='full'
    )
    
    return train_features, test_features


def save_features(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """Save engineered features to disk"""
    logger.info("\nSaving features...")
    
    processed_dir = PATHS['processed_dir']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = processed_dir / 'train_features.parquet'
    test_path = processed_dir / 'test_features.parquet'
    
    # Save as parquet for efficiency
    train_features.to_parquet(train_path, index=False)
    test_features.to_parquet(test_path, index=False)
    
    logger.info(f"  Saved: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    logger.info(f"  Saved: {test_path} ({test_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Also save as CSV for inspection
    train_features.head(100).to_csv(processed_dir / 'train_features_sample.csv', index=False)
    logger.info(f"  Saved sample: {processed_dir / 'train_features_sample.csv'}")


def validate_features(train_features: pd.DataFrame, test_features: pd.DataFrame):
    """Validate feature quality"""
    logger.info("\n" + "=" * 60)
    logger.info("FEATURE VALIDATION")
    logger.info("=" * 60)
    
    # Check column alignment
    train_cols = set(train_features.columns)
    test_cols = set(test_features.columns)
    
    if train_cols != test_cols:
        missing_in_test = train_cols - test_cols
        missing_in_train = test_cols - train_cols
        logger.warning(f"Column mismatch!")
        if missing_in_test:
            logger.warning(f"  Missing in test: {missing_in_test}")
        if missing_in_train:
            logger.warning(f"  Missing in train: {missing_in_train}")
    else:
        logger.info("✓ Column alignment: OK")
    
    # Check for constant features
    constant_cols = []
    for col in train_features.columns:
        if train_features[col].nunique() <= 1:
            constant_cols.append(col)
    
    if constant_cols:
        logger.warning(f"Constant features (consider removing): {constant_cols[:5]}...")
    else:
        logger.info("✓ No constant features")
    
    # Check for highly correlated features
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        corr_matrix = train_features[numeric_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [(col, row, corr_matrix.loc[row, col]) 
                     for col in upper.columns for row in upper.index 
                     if upper.loc[row, col] > 0.95]
        
        if high_corr:
            logger.warning(f"Highly correlated features (>0.95): {len(high_corr)} pairs")
        else:
            logger.info("✓ No highly correlated feature pairs (>0.95)")
    
    # Feature statistics
    logger.info("\nFeature statistics:")
    stats = train_features.describe().T
    logger.info(f"  Mean range: [{stats['mean'].min():.2f}, {stats['mean'].max():.2f}]")
    logger.info(f"  Std range: [{stats['std'].min():.2f}, {stats['std'].max():.2f}]")
    
    logger.info("\n✓ Feature validation complete")


def main():
    parser = argparse.ArgumentParser(
        description='Stage 2: Feature Engineering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run feature engineering
  python run_stage2_features.py

  # Resume from checkpoint
  python run_stage2_features.py --resume checkpoints/feature_engineering_xxx.pt

  # Validate only (using existing features)
  python run_stage2_features.py --validate-only
        """
    )
    
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing features')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - STAGE 2: FEATURE ENGINEERING")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    try:
        if args.validate_only:
            # Load existing features
            processed_dir = PATHS['processed_dir']
            train_features = pd.read_parquet(processed_dir / 'train_features.parquet')
            test_features = pd.read_parquet(processed_dir / 'test_features.parquet')
        else:
            # Load raw data
            train_df, test_df = load_data()
            
            # Engineer features
            train_features, test_features = engineer_features(
                train_df, test_df, resume_from=args.resume
            )
            
            # Save features
            save_features(train_features, test_features)
        
        # Validate features
        validate_features(train_features, test_features)
        
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2 COMPLETE")
        logger.info("=" * 60)
        logger.info("Next step: Run stage 3 for neural network training")
        logger.info("  python scripts/run_stage3_neural_net.py")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Stage 2 failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
