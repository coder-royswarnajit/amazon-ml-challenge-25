#!/usr/bin/env python3
"""
Stage 1: Setup and Data Download

This script handles:
1. Environment setup verification
2. Data download from Kaggle
3. Image download from URLs
4. Data validation
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import PATHS, DATA_CONFIG
from src.data.downloader import DataDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PATHS['logs_dir'] / 'stage1_setup.log')
    ]
)
logger = logging.getLogger(__name__)


def verify_environment():
    """Verify all required directories and dependencies exist"""
    logger.info("Verifying environment...")
    
    # Create required directories
    for name, path in PATHS.items():
        if isinstance(path, Path) and 'dir' in name:
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"  ✓ {name}: {path}")
    
    # Check for required packages
    required_packages = [
        'torch', 'transformers', 'timm', 'peft',
        'lightgbm', 'xgboost', 'catboost',
        'pandas', 'numpy', 'sklearn', 'optuna'
    ]
    
    missing = []
    for pkg in required_packages:
        try:
            __import__(pkg)
            logger.info(f"  ✓ Package: {pkg}")
        except ImportError:
            missing.append(pkg)
            logger.warning(f"  ✗ Missing package: {pkg}")
    
    if missing:
        logger.error(f"Missing packages: {missing}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def download_data(skip_images: bool = False, max_images: int = None):
    """Download data from Kaggle and images from URLs"""
    
    downloader = DataDownloader(
        data_dir=PATHS['data_dir'],
        raw_dir=PATHS['raw_dir'],
        images_dir=PATHS['images_dir']
    )
    
    # Download Kaggle data
    logger.info("\n" + "=" * 60)
    logger.info("DOWNLOADING KAGGLE DATA")
    logger.info("=" * 60)
    
    kaggle_success = downloader.download_kaggle_data()
    
    if not kaggle_success:
        logger.warning("Kaggle download failed. Please download manually:")
        logger.warning(f"  kaggle competitions download -c {DATA_CONFIG['kaggle_competition']}")
        logger.warning(f"  Extract to: {PATHS['raw_dir']}")
    
    # Download images
    if not skip_images:
        logger.info("\n" + "=" * 60)
        logger.info("DOWNLOADING IMAGES")
        logger.info("=" * 60)
        
        train_path = PATHS['raw_dir'] / 'train.csv'
        test_path = PATHS['raw_dir'] / 'test.csv'
        
        if train_path.exists():
            logger.info("Downloading training images...")
            downloader.download_images_from_csv(
                train_path,
                image_column='image_link',
                id_column='sample_id',
                max_images=max_images
            )
        else:
            logger.warning(f"Training file not found: {train_path}")
        
        if test_path.exists():
            logger.info("Downloading test images...")
            downloader.download_images_from_csv(
                test_path,
                image_column='image_link',
                id_column='sample_id',
                max_images=max_images
            )
        else:
            logger.warning(f"Test file not found: {test_path}")
    else:
        logger.info("Skipping image download (--skip-images flag)")


def validate_data():
    """Validate downloaded data"""
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATING DATA")
    logger.info("=" * 60)
    
    import pandas as pd
    
    # Check CSV files
    train_path = PATHS['raw_dir'] / 'train.csv'
    test_path = PATHS['raw_dir'] / 'test.csv'
    
    if train_path.exists():
        train_df = pd.read_csv(train_path)
        logger.info(f"Training data: {len(train_df)} samples")
        logger.info(f"  Columns: {list(train_df.columns)}")
        logger.info(f"  Price range: [{train_df['price'].min():.2f}, {train_df['price'].max():.2f}]")
        logger.info(f"  Price mean: {train_df['price'].mean():.2f}")
    else:
        logger.error(f"Training file not found: {train_path}")
        return False
    
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        logger.info(f"Test data: {len(test_df)} samples")
        logger.info(f"  Columns: {list(test_df.columns)}")
    else:
        logger.error(f"Test file not found: {test_path}")
        return False
    
    # Check images
    images_dir = PATHS['images_dir']
    if images_dir.exists():
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        logger.info(f"Downloaded images: {len(image_files)}")
        
        # Check image coverage
        train_ids = set(train_df['sample_id'].astype(str))
        test_ids = set(test_df['sample_id'].astype(str))
        
        image_ids = {f.stem for f in image_files}
        train_coverage = len(train_ids & image_ids) / len(train_ids) * 100
        test_coverage = len(test_ids & image_ids) / len(test_ids) * 100
        
        logger.info(f"  Training image coverage: {train_coverage:.1f}%")
        logger.info(f"  Test image coverage: {test_coverage:.1f}%")
    else:
        logger.warning(f"Images directory not found: {images_dir}")
    
    logger.info("\n✓ Data validation complete")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Stage 1: Setup and Data Download',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full setup (download everything)
  python run_stage1_setup.py

  # Skip image download
  python run_stage1_setup.py --skip-images

  # Download only first 1000 images
  python run_stage1_setup.py --max-images 1000

  # Validate only (no download)
  python run_stage1_setup.py --validate-only
        """
    )
    
    parser.add_argument('--skip-images', action='store_true',
                        help='Skip downloading images')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to download per file')
    parser.add_argument('--validate-only', action='store_true',
                        help='Only validate existing data, no download')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AMAZON ML CHALLENGE - STAGE 1: SETUP")
    logger.info("=" * 60)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    try:
        # Verify environment
        if not verify_environment():
            logger.error("Environment verification failed")
            sys.exit(1)
        
        # Download data (unless validate-only)
        if not args.validate_only:
            download_data(
                skip_images=args.skip_images,
                max_images=args.max_images
            )
        
        # Validate data
        if not validate_data():
            logger.error("Data validation failed")
            sys.exit(1)
        
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1 COMPLETE")
        logger.info("=" * 60)
        logger.info("Next step: Run stage 2 for feature engineering")
        logger.info("  python scripts/run_stage2_features.py")
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Stage 1 failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
