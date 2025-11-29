#!/usr/bin/env python3
"""
Stage 4: GBDT Model Training

This script handles:
1. LightGBM training with Optuna optimization
2. XGBoost training with Optuna optimization
3. CatBoost training
4. Prediction generation for ensemble
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, TRAIN_CONFIG, GBDT_CONFIG
from src.training.train_gbdt import (
    train_lightgbm, train_xgboost, train_catboost, train_gbdt_models
)
from src.utils.metrics import calculate_smape, evaluate_predictions
from src.utils.checkpoint import CheckpointManager
from src.utils.visualization import plot_predictions, plot_feature_importance

# Configure logging
PATHS['logs_dir'].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATHS['logs_dir'] / 'stage4_gbdt.log')
    ]
)
logger = logging.getLogger(__name__)


def load_data():
    """Load raw data and features"""
    logger.info("Loading data...")
    
    # Load raw data
    train_df = pd.read_csv(PATHS['raw_dir'] / 'train.csv')
    test_df = pd.read_csv(PATHS['raw_dir'] / 'test.csv')
    
    # Load features
    processed_dir = PATHS['processed_dir']
    train_features = pd.read_parquet(processed_dir / 'train_features.parquet')
    test_features = pd.read_parquet(processed_dir / 'test_features.parquet')
    
    logger.info(f"  Training samples: {len(train_df)}")
    logger.info(f"  Test samples: {len(test_df)}")
    logger.info(f"  Feature columns: {train_features.shape[1]}")
    
    return train_df, test_df, train_features, test_features


def prepare_data(train_df, test_df, train_features, test_features, val_size=0.1):
    """Prepare data for GBDT training"""
    logger.info("\nPreparing data...")
    
    # Split training data (use same split as neural network)
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=val_size,
        random_state=42
    )
    
    # Extract features and targets
    X_train = train_features.iloc[train_idx].values
    X_val = train_features.iloc[val_idx].values
    X_test = test_features.values
    
    y_train = train_df.iloc[train_idx]['price'].values
    y_val = train_df.iloc[val_idx]['price'].values
    
    # Log-transform targets for better GBDT performance
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    logger.info(f"  X_train: {X_train.shape}")
    logger.info(f"  X_val: {X_val.shape}")
    logger.info(f"  X_test: {X_test.shape}")
    
    # Handle NaN values
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    
    data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_train_log': y_train_log,
        'y_val_log': y_val_log,
        'feature_names': train_features.columns.tolist(),
        'train_df': train_df.iloc[train_idx].reset_index(drop=True),
        'val_df': train_df.iloc[val_idx].reset_index(drop=True),
        'test_df': test_df
    }
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='Stage 4: GBDT Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training with optimization
  python run_stage4_gbdt.py

  # Skip optimization (use default params)
  python run_stage4_gbdt.py --no-optimize

  # Train only specific model
  python run_stage4_gbdt.py --models lightgbm

  # Quick test
  python run_stage4_gbdt.py --n-trials 5 --debug
        """
    )
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['lightgbm', 'xgboost', 'catboost'],
                        choices=['lightgbm', 'xgboost', 'catboost'],
                        help='Models to train')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode with fewer samples')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - STAGE 4: GBDT TRAINING")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Models to train: {args.models}")
    
    try:
        # Load data
        train_df, test_df, train_features, test_features = load_data()
        
        # Debug mode
        if args.debug:
            logger.info("\n*** DEBUG MODE - Using reduced data ***")
            train_df = train_df.head(1000)
            test_df = test_df.head(200)
            train_features = train_features.head(1000)
            test_features = test_features.head(200)
        
        # Prepare data
        data = prepare_data(train_df, test_df, train_features, test_features)
        
        # Config
        config = {
            **GBDT_CONFIG,
            'n_trials': args.n_trials,
            'optimize': not args.no_optimize
        }
        
        # Train all GBDT models
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING GBDT MODELS")
        logger.info("=" * 60)
        
        results = train_gbdt_models(
            X_train=data['X_train'],
            y_train=data['y_train'],
            X_val=data['X_val'],
            y_val=data['y_val'],
            X_test=data['X_test'],
            config=config,
            optimize=not args.no_optimize,
            models_to_train=args.models
        )
        
        # Evaluate and save predictions
        predictions_dir = PATHS['predictions_dir']
        predictions_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        
        all_val_preds = {}
        all_test_preds = {}
        
        for model_name, model_results in results.items():
            val_preds = model_results['val_preds']
            test_preds = model_results['test_preds']
            
            # Calculate metrics
            val_smape = calculate_smape(data['y_val'], val_preds)
            logger.info(f"\n{model_name}:")
            logger.info(f"  Validation SMAPE: {val_smape:.4f}")
            
            # Save predictions
            all_val_preds[model_name] = val_preds
            all_test_preds[model_name] = test_preds
            
            np.save(predictions_dir / f'{model_name}_val_preds.npy', val_preds)
            np.save(predictions_dir / f'{model_name}_test_preds.npy', test_preds)
            
            # Plot predictions
            plot_predictions(
                data['y_val'], val_preds,
                split_name=f'{model_name} Validation',
                save_path=PATHS['logs_dir'] / f'{model_name}_predictions.png',
                smape=val_smape
            )
            
            # Save feature importance (if available)
            if 'feature_importance' in model_results:
                importance = model_results['feature_importance']
                importance_dict = dict(zip(data['feature_names'], importance))
                plot_feature_importance(
                    importance_dict,
                    top_k=30,
                    save_path=PATHS['logs_dir'] / f'{model_name}_feature_importance.png',
                    title=f'{model_name} Feature Importance'
                )
        
        # Save combined predictions
        val_pred_df = pd.DataFrame({
            'sample_id': data['val_df']['sample_id'],
            'true_price': data['y_val'],
            **{f'{k}_pred': v for k, v in all_val_preds.items()}
        })
        val_pred_df.to_csv(predictions_dir / 'gbdt_val_predictions.csv', index=False)
        
        test_pred_df = pd.DataFrame({
            'sample_id': data['test_df']['sample_id'],
            **{f'{k}_pred': v for k, v in all_test_preds.items()}
        })
        test_pred_df.to_csv(predictions_dir / 'gbdt_test_predictions.csv', index=False)
        
        # Save best params
        best_params = {name: res.get('best_params', {}) for name, res in results.items()}
        with open(PATHS['models_dir'] / 'gbdt_best_params.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        
        logger.info(f"\nPredictions saved to {predictions_dir}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4 SUMMARY")
        logger.info("=" * 60)
        
        for model_name, model_results in results.items():
            val_smape = calculate_smape(data['y_val'], model_results['val_preds'])
            logger.info(f"  {model_name}: SMAPE = {val_smape:.4f}")
        
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4 COMPLETE")
        logger.info("=" * 60)
        logger.info("Next step: Run stage 5 for ensemble training")
        logger.info("  python scripts/run_stage5_ensemble.py")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Stage 4 failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
