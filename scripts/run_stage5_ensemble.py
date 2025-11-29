#!/usr/bin/env python3
"""
Stage 5: Ensemble Training

This script handles:
1. Loading base model predictions
2. Training 2-level stacking ensemble
3. Isotonic calibration
4. Final prediction generation
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

from config import PATHS, TRAIN_CONFIG
from src.training.train_ensemble import (
    train_ensemble, StackingEnsemble, optimize_base_model_weights
)
from src.utils.metrics import calculate_smape, evaluate_predictions, calculate_metrics_by_quantile
from src.utils.visualization import (
    plot_predictions, plot_error_distribution, plot_ensemble_weights, plot_model_comparison
)

# Configure logging
PATHS['logs_dir'].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATHS['logs_dir'] / 'stage5_ensemble.log')
    ]
)
logger = logging.getLogger(__name__)


def load_predictions():
    """Load base model predictions"""
    logger.info("Loading base model predictions...")
    
    predictions_dir = PATHS['predictions_dir']
    
    # Load neural network predictions
    nn_val_preds = np.load(predictions_dir / 'nn_val_preds.npy')
    nn_test_preds = np.load(predictions_dir / 'nn_test_preds.npy')
    
    # Load GBDT predictions
    lgb_val_preds = np.load(predictions_dir / 'lightgbm_val_preds.npy')
    lgb_test_preds = np.load(predictions_dir / 'lightgbm_test_preds.npy')
    
    xgb_val_preds = np.load(predictions_dir / 'xgboost_val_preds.npy')
    xgb_test_preds = np.load(predictions_dir / 'xgboost_test_preds.npy')
    
    cat_val_preds = np.load(predictions_dir / 'catboost_val_preds.npy')
    cat_test_preds = np.load(predictions_dir / 'catboost_test_preds.npy')
    
    val_preds_dict = {
        'neural_net': nn_val_preds,
        'lightgbm': lgb_val_preds,
        'xgboost': xgb_val_preds,
        'catboost': cat_val_preds
    }
    
    test_preds_dict = {
        'neural_net': nn_test_preds,
        'lightgbm': lgb_test_preds,
        'xgboost': xgb_test_preds,
        'catboost': cat_test_preds
    }
    
    for name, preds in val_preds_dict.items():
        logger.info(f"  {name}: val={len(preds)}, test={len(test_preds_dict[name])}")
    
    return val_preds_dict, test_preds_dict


def load_train_predictions():
    """Load training set predictions for stacking"""
    logger.info("Loading training set predictions...")
    
    predictions_dir = PATHS['predictions_dir']
    
    # Try to load training predictions (generated during out-of-fold validation)
    train_preds_dict = {}
    
    models = ['neural_net', 'lightgbm', 'xgboost', 'catboost']
    
    for model in models:
        train_path = predictions_dir / f'{model}_train_preds.npy'
        if train_path.exists():
            train_preds_dict[model] = np.load(train_path)
        else:
            logger.warning(f"Training predictions not found for {model}")
    
    return train_preds_dict


def load_targets():
    """Load validation targets"""
    logger.info("Loading targets...")
    
    # Load raw data
    train_df = pd.read_csv(PATHS['raw_dir'] / 'train.csv')
    
    # Use same split as other stages
    train_idx, val_idx = train_test_split(
        range(len(train_df)),
        test_size=0.1,
        random_state=42
    )
    
    y_train = train_df.iloc[train_idx]['price'].values
    y_val = train_df.iloc[val_idx]['price'].values
    
    val_df = train_df.iloc[val_idx].reset_index(drop=True)
    
    logger.info(f"  Training samples: {len(y_train)}")
    logger.info(f"  Validation samples: {len(y_val)}")
    
    return y_train, y_val, val_df


def main():
    parser = argparse.ArgumentParser(
        description='Stage 5: Ensemble Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full ensemble training
  python run_stage5_ensemble.py

  # Load existing ensemble
  python run_stage5_ensemble.py --load-ensemble models/ensemble

  # Simple weighted average only
  python run_stage5_ensemble.py --simple-only
        """
    )
    
    parser.add_argument('--load-ensemble', type=str, default=None,
                        help='Load existing ensemble from path')
    parser.add_argument('--simple-only', action='store_true',
                        help='Use simple weighted average only (no stacking)')
    parser.add_argument('--no-isotonic', action='store_true',
                        help='Disable isotonic calibration')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - STAGE 5: ENSEMBLE TRAINING")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    try:
        # Load predictions
        val_preds_dict, test_preds_dict = load_predictions()
        
        # Load targets
        y_train, y_val, val_df = load_targets()
        
        # Try to load training predictions
        train_preds_dict = load_train_predictions()
        
        # If no training predictions, use validation predictions (for demo)
        if not train_preds_dict:
            logger.warning("No training predictions found, using validation predictions")
            train_preds_dict = val_preds_dict
            y_train = y_val
        
        # Base model performance
        logger.info("\n" + "=" * 60)
        logger.info("BASE MODEL PERFORMANCE")
        logger.info("=" * 60)
        
        model_metrics = {}
        for name, preds in val_preds_dict.items():
            smape = calculate_smape(y_val, preds)
            model_metrics[name] = {'smape': smape}
            logger.info(f"  {name}: SMAPE = {smape:.4f}")
        
        if args.simple_only:
            # Simple weighted average
            logger.info("\n" + "=" * 60)
            logger.info("SIMPLE WEIGHTED AVERAGE")
            logger.info("=" * 60)
            
            weights, val_preds = optimize_base_model_weights(val_preds_dict, y_val)
            
            # Apply weights to test
            test_preds = np.zeros(len(next(iter(test_preds_dict.values()))))
            for name, weight in weights.items():
                test_preds += weight * test_preds_dict[name].ravel()
            
            ensemble_smape = calculate_smape(y_val, val_preds)
            
            logger.info(f"\nOptimized weights: {weights}")
            logger.info(f"Ensemble SMAPE: {ensemble_smape:.4f}")
            
            results = {
                'val_predictions': val_preds,
                'test_predictions': test_preds,
                'val_smape': ensemble_smape,
                'weights': weights
            }
            
        elif args.load_ensemble:
            # Load existing ensemble
            logger.info("\n" + "=" * 60)
            logger.info(f"LOADING ENSEMBLE FROM {args.load_ensemble}")
            logger.info("=" * 60)
            
            ensemble = StackingEnsemble.load(Path(args.load_ensemble))
            val_preds = ensemble.predict(val_preds_dict)
            test_preds = ensemble.predict(test_preds_dict)
            
            ensemble_smape = calculate_smape(y_val, val_preds)
            
            results = {
                'ensemble': ensemble,
                'val_predictions': val_preds,
                'test_predictions': test_preds,
                'val_smape': ensemble_smape
            }
            
        else:
            # Full stacking ensemble
            logger.info("\n" + "=" * 60)
            logger.info("TRAINING STACKING ENSEMBLE")
            logger.info("=" * 60)
            
            results = train_ensemble(
                train_preds_dict=train_preds_dict,
                val_preds_dict=val_preds_dict,
                test_preds_dict=test_preds_dict,
                y_train=y_train,
                y_val=y_val,
                config=TRAIN_CONFIG,
                save_dir=PATHS['models_dir'] / 'ensemble'
            )
            
            val_preds = results['val_predictions']
            test_preds = results['test_predictions']
            ensemble_smape = results['val_smape']
        
        # Final evaluation
        logger.info("\n" + "=" * 60)
        logger.info("FINAL ENSEMBLE EVALUATION")
        logger.info("=" * 60)
        
        metrics = evaluate_predictions(y_val, val_preds, 'validation')
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        # Quantile analysis
        logger.info("\nSMAPE by price quantile:")
        quantile_df = calculate_metrics_by_quantile(y_val, val_preds)
        print(quantile_df.to_string())
        
        # Save predictions
        predictions_dir = PATHS['predictions_dir']
        
        np.save(predictions_dir / 'ensemble_val_preds.npy', val_preds)
        np.save(predictions_dir / 'ensemble_test_preds.npy', test_preds)
        
        # Save with sample IDs
        test_df = pd.read_csv(PATHS['raw_dir'] / 'test.csv')
        
        val_pred_df = pd.DataFrame({
            'sample_id': val_df['sample_id'],
            'true_price': y_val,
            'predicted_price': val_preds
        })
        val_pred_df.to_csv(predictions_dir / 'ensemble_val_predictions.csv', index=False)
        
        test_pred_df = pd.DataFrame({
            'sample_id': test_df['sample_id'],
            'predicted_price': test_preds
        })
        test_pred_df.to_csv(predictions_dir / 'ensemble_test_predictions.csv', index=False)
        
        logger.info(f"\nPredictions saved to {predictions_dir}")
        
        # Create visualizations
        logger.info("\nCreating visualizations...")
        
        plot_predictions(
            y_val, val_preds,
            split_name='Ensemble Validation',
            save_path=PATHS['logs_dir'] / 'ensemble_predictions.png',
            smape=ensemble_smape
        )
        
        plot_error_distribution(
            y_val, val_preds,
            save_path=PATHS['logs_dir'] / 'ensemble_error_distribution.png',
            title='Ensemble Error Distribution'
        )
        
        # Model comparison
        model_metrics['ensemble'] = {'smape': ensemble_smape}
        plot_model_comparison(
            model_metrics,
            metric_name='smape',
            save_path=PATHS['logs_dir'] / 'model_comparison.png',
            title='Model SMAPE Comparison'
        )
        
        # Ensemble weights
        if 'simple_weights' in results:
            plot_ensemble_weights(
                results['simple_weights'],
                save_path=PATHS['logs_dir'] / 'ensemble_base_weights.png',
                title='Base Model Weights'
            )
        
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5 COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Final Validation SMAPE: {ensemble_smape:.4f}")
        
        if ensemble_smape < 9.0:
            logger.info("✓ Target SMAPE < 9% achieved!")
        else:
            logger.warning(f"Target SMAPE < 9% not achieved (current: {ensemble_smape:.4f})")
        
        logger.info("\nNext step: Create final submission")
        logger.info("  python scripts/create_submission.py")
        
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}")
        logger.error("Make sure you have run stages 3 and 4 first")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Stage 5 failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
