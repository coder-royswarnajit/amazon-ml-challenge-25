#!/usr/bin/env python3
"""
Create Submission

This script handles:
1. Loading best ensemble predictions
2. Converting from log space if needed
3. Creating submission DataFrame
4. Validating submission format
5. Saving final submission CSV
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

from config import PATHS

# Configure logging
PATHS['logs_dir'].mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATHS['logs_dir'] / 'create_submission.log')
    ]
)
logger = logging.getLogger(__name__)


def load_predictions(source: str = 'ensemble'):
    """Load predictions from specified source"""
    predictions_dir = PATHS['predictions_dir']
    
    if source == 'ensemble':
        preds_path = predictions_dir / 'ensemble_test_preds.npy'
    elif source == 'neural_net':
        preds_path = predictions_dir / 'nn_test_preds.npy'
    elif source in ['lightgbm', 'xgboost', 'catboost']:
        preds_path = predictions_dir / f'{source}_test_preds.npy'
    else:
        preds_path = Path(source)
    
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    
    predictions = np.load(preds_path)
    logger.info(f"Loaded predictions from {preds_path}: {len(predictions)} samples")
    
    return predictions


def load_sample_ids():
    """Load test sample IDs"""
    test_path = PATHS['raw_dir'] / 'test.csv'
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    test_df = pd.read_csv(test_path)
    sample_ids = test_df['sample_id'].values
    
    logger.info(f"Loaded {len(sample_ids)} sample IDs from test.csv")
    
    return sample_ids


def validate_submission(submission: pd.DataFrame, sample_ids: np.ndarray):
    """Validate submission format and completeness"""
    logger.info("\nValidating submission...")
    
    errors = []
    warnings = []
    
    # Check columns
    required_cols = ['sample_id', 'price']
    for col in required_cols:
        if col not in submission.columns:
            errors.append(f"Missing required column: {col}")
    
    if 'sample_id' in submission.columns:
        # Check for all sample IDs
        submission_ids = set(submission['sample_id'].values)
        expected_ids = set(sample_ids)
        
        missing = expected_ids - submission_ids
        extra = submission_ids - expected_ids
        
        if missing:
            errors.append(f"Missing {len(missing)} sample IDs")
        if extra:
            warnings.append(f"Extra {len(extra)} sample IDs (will be ignored)")
    
    if 'price' in submission.columns:
        # Check for valid prices
        prices = submission['price'].values
        
        if np.any(np.isnan(prices)):
            errors.append(f"Found {np.sum(np.isnan(prices))} NaN prices")
        
        if np.any(prices < 0):
            errors.append(f"Found {np.sum(prices < 0)} negative prices")
        
        if np.any(np.isinf(prices)):
            errors.append(f"Found {np.sum(np.isinf(prices))} infinite prices")
        
        # Price statistics
        logger.info(f"Price statistics:")
        logger.info(f"  Min: ₹{np.nanmin(prices):.2f}")
        logger.info(f"  Max: ₹{np.nanmax(prices):.2f}")
        logger.info(f"  Mean: ₹{np.nanmean(prices):.2f}")
        logger.info(f"  Median: ₹{np.nanmedian(prices):.2f}")
        logger.info(f"  Std: ₹{np.nanstd(prices):.2f}")
    
    # Report
    if errors:
        for error in errors:
            logger.error(f"✗ {error}")
        return False
    
    for warning in warnings:
        logger.warning(f"⚠ {warning}")
    
    logger.info("✓ Submission validation passed")
    return True


def create_submission(
    predictions: np.ndarray,
    sample_ids: np.ndarray,
    from_log_space: bool = False,
    clip_min: float = 0.0,
    clip_max: float = None
) -> pd.DataFrame:
    """Create submission DataFrame"""
    
    # Convert from log space if needed
    if from_log_space:
        predictions = np.expm1(predictions)
        logger.info("Converted predictions from log space")
    
    # Clip to valid range
    predictions = np.clip(predictions, clip_min, clip_max)
    
    # Create DataFrame
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Ensure correct types
    submission['sample_id'] = submission['sample_id'].astype(str)
    submission['price'] = submission['price'].astype(float)
    
    return submission


def main():
    parser = argparse.ArgumentParser(
        description='Create Submission File',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create submission from ensemble predictions
  python create_submission.py

  # Use specific model predictions
  python create_submission.py --source neural_net

  # Custom predictions file
  python create_submission.py --source /path/to/predictions.npy

  # With log space conversion
  python create_submission.py --from-log-space
        """
    )
    
    parser.add_argument('--source', type=str, default='ensemble',
                        help='Prediction source: ensemble, neural_net, lightgbm, xgboost, catboost, or path')
    parser.add_argument('--from-log-space', action='store_true',
                        help='Convert predictions from log space')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file path')
    parser.add_argument('--clip-min', type=float, default=0.0,
                        help='Minimum price clip value')
    parser.add_argument('--clip-max', type=float, default=None,
                        help='Maximum price clip value')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - CREATE SUBMISSION")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Source: {args.source}")
    
    try:
        # Load predictions
        predictions = load_predictions(args.source)
        
        # Load sample IDs
        sample_ids = load_sample_ids()
        
        # Verify length match
        if len(predictions) != len(sample_ids):
            raise ValueError(f"Length mismatch: predictions={len(predictions)}, sample_ids={len(sample_ids)}")
        
        # Create submission
        submission = create_submission(
            predictions=predictions,
            sample_ids=sample_ids,
            from_log_space=args.from_log_space,
            clip_min=args.clip_min,
            clip_max=args.clip_max
        )
        
        # Validate
        if not validate_submission(submission, sample_ids):
            logger.error("Submission validation failed!")
            sys.exit(1)
        
        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = PATHS['predictions_dir'] / f'submission_{args.source}_{timestamp}.csv'
        
        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save submission
        submission.to_csv(output_path, index=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("SUBMISSION CREATED")
        logger.info("=" * 60)
        logger.info(f"Output: {output_path}")
        logger.info(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
        logger.info(f"Samples: {len(submission)}")
        
        # Also create a copy as 'submission.csv' for easy upload
        latest_path = PATHS['predictions_dir'] / 'submission.csv'
        submission.to_csv(latest_path, index=False)
        logger.info(f"Also saved as: {latest_path}")
        
        # Preview
        logger.info("\nSubmission preview:")
        print(submission.head(10).to_string(index=False))
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Failed to create submission: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
