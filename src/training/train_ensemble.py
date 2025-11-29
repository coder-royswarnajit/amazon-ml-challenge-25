"""
Ensemble Training Module for Amazon ML Challenge

Implements 2-level stacking ensemble with:
- Level 0: Base model predictions (Neural Network, LightGBM, XGBoost, CatBoost)
- Level 1: Meta-learners (Ridge, ElasticNet, Shallow LightGBM)
- Level 2: Final weighted combination with isotonic calibration
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import json
from datetime import datetime

from sklearn.linear_model import Ridge, ElasticNet, RidgeCV, ElasticNetCV
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from scipy.optimize import minimize
import lightgbm as lgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.metrics import calculate_smape, smape_scorer
from src.utils.checkpoint import CheckpointManager
from config import TRAIN_CONFIG, PATHS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StackingEnsemble:
    """
    2-Level Stacking Ensemble with Isotonic Calibration
    
    Level 0: Base model predictions
    Level 1: Meta-learners (Ridge, ElasticNet, LightGBM)
    Level 2: Weighted combination + Isotonic calibration
    """
    
    def __init__(
        self,
        n_folds: int = 5,
        use_isotonic: bool = True,
        optimize_weights: bool = True,
        random_state: int = 42
    ):
        self.n_folds = n_folds
        self.use_isotonic = use_isotonic
        self.optimize_weights = optimize_weights
        self.random_state = random_state
        
        # Meta-learners
        self.meta_ridge = None
        self.meta_elasticnet = None
        self.meta_lgbm = None
        
        # Ensemble components
        self.meta_weights = None
        self.isotonic_model = None
        
        # Fitted flag
        self.is_fitted = False
        
    def fit(
        self,
        train_preds: Dict[str, np.ndarray],
        val_preds: Dict[str, np.ndarray],
        y_train: np.ndarray,
        y_val: np.ndarray
    ) -> 'StackingEnsemble':
        """
        Fit the stacking ensemble
        
        Args:
            train_preds: Dict of base model predictions on training set
            val_preds: Dict of base model predictions on validation set
            y_train: Training targets
            y_val: Validation targets
            
        Returns:
            self
        """
        logger.info("Fitting 2-level stacking ensemble...")
        
        # Prepare stacked features
        X_train_stack = self._prepare_stack_features(train_preds)
        X_val_stack = self._prepare_stack_features(val_preds)
        
        logger.info(f"Stack features shape: {X_train_stack.shape}")
        
        # Level 1: Train meta-learners with cross-validation
        logger.info("Training Level 1 meta-learners...")
        meta_preds_train, meta_preds_val = self._train_meta_learners(
            X_train_stack, y_train, X_val_stack
        )
        
        # Level 2: Optimize meta-learner weights
        logger.info("Optimizing Level 2 weights...")
        self._optimize_meta_weights(meta_preds_val, y_val)
        
        # Level 2: Train isotonic calibration
        if self.use_isotonic:
            logger.info("Training isotonic calibration...")
            self._train_isotonic(meta_preds_val, y_val)
        
        self.is_fitted = True
        
        # Evaluate on validation
        final_val_preds = self.predict(val_preds)
        val_smape = calculate_smape(y_val, final_val_preds)
        logger.info(f"Ensemble validation SMAPE: {val_smape:.4f}")
        
        return self
    
    def _prepare_stack_features(self, preds_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Stack base model predictions as features"""
        # Ensure consistent ordering
        model_names = sorted(preds_dict.keys())
        return np.column_stack([preds_dict[name].ravel() for name in model_names])
    
    def _train_meta_learners(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train meta-learners and get predictions"""
        
        meta_preds_train = []
        meta_preds_val = []
        
        # 1. Ridge Regression with CV
        logger.info("  Training Ridge CV...")
        alphas = np.logspace(-4, 4, 20)
        self.meta_ridge = RidgeCV(
            alphas=alphas,
            cv=self.n_folds,
            scoring='neg_mean_absolute_error'
        )
        self.meta_ridge.fit(X_train, y_train)
        
        # Cross-val predictions on training
        ridge_train_preds = cross_val_predict(
            Ridge(alpha=self.meta_ridge.alpha_),
            X_train, y_train,
            cv=self.n_folds
        )
        meta_preds_train.append(ridge_train_preds)
        meta_preds_val.append(self.meta_ridge.predict(X_val))
        
        logger.info(f"    Ridge best alpha: {self.meta_ridge.alpha_:.6f}")
        
        # 2. ElasticNet with CV
        logger.info("  Training ElasticNet CV...")
        self.meta_elasticnet = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            alphas=np.logspace(-4, 2, 20),
            cv=self.n_folds,
            max_iter=5000,
            random_state=self.random_state
        )
        self.meta_elasticnet.fit(X_train, y_train)
        
        # Cross-val predictions on training
        elasticnet_train_preds = cross_val_predict(
            ElasticNet(
                alpha=self.meta_elasticnet.alpha_,
                l1_ratio=self.meta_elasticnet.l1_ratio_
            ),
            X_train, y_train,
            cv=self.n_folds
        )
        meta_preds_train.append(elasticnet_train_preds)
        meta_preds_val.append(self.meta_elasticnet.predict(X_val))
        
        logger.info(f"    ElasticNet alpha: {self.meta_elasticnet.alpha_:.6f}, "
                   f"l1_ratio: {self.meta_elasticnet.l1_ratio_:.2f}")
        
        # 3. Shallow LightGBM
        logger.info("  Training Shallow LightGBM...")
        lgbm_train_preds = self._train_shallow_lgbm(X_train, y_train, X_val)
        meta_preds_train.append(lgbm_train_preds[0])
        meta_preds_val.append(lgbm_train_preds[1])
        
        return np.column_stack(meta_preds_train), np.column_stack(meta_preds_val)
    
    def _train_shallow_lgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train shallow LightGBM for meta-learning"""
        
        # Shallow LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 15,  # Very shallow
            'max_depth': 4,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': self.random_state,
            'verbose': -1
        }
        
        # Cross-validation predictions
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        train_preds = np.zeros(len(y_train))
        val_preds = np.zeros(len(X_val))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_vl = X_train[train_idx], X_train[val_idx]
            y_tr, y_vl = y_train[train_idx], y_train[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_vl, y_vl)],
                callbacks=[lgb.early_stopping(20, verbose=False)]
            )
            
            train_preds[val_idx] = model.predict(X_vl)
            val_preds += model.predict(X_val) / self.n_folds
        
        # Train final model on all training data
        self.meta_lgbm = lgb.LGBMRegressor(**params)
        self.meta_lgbm.fit(X_train, y_train)
        
        return train_preds, val_preds
    
    def _optimize_meta_weights(
        self,
        meta_preds: np.ndarray,
        y_true: np.ndarray
    ) -> None:
        """Optimize weights for meta-learner predictions"""
        
        if not self.optimize_weights:
            # Equal weights
            n_meta = meta_preds.shape[1]
            self.meta_weights = np.ones(n_meta) / n_meta
            return
        
        def objective(weights):
            """SMAPE objective for weight optimization"""
            weights = np.abs(weights)
            weights = weights / np.sum(weights)  # Normalize
            weighted_pred = meta_preds @ weights
            return calculate_smape(y_true, weighted_pred)
        
        # Initial weights (equal)
        n_meta = meta_preds.shape[1]
        x0 = np.ones(n_meta) / n_meta
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        # Normalize final weights
        weights = np.abs(result.x)
        self.meta_weights = weights / np.sum(weights)
        
        logger.info(f"    Optimized meta weights: Ridge={self.meta_weights[0]:.4f}, "
                   f"ElasticNet={self.meta_weights[1]:.4f}, LightGBM={self.meta_weights[2]:.4f}")
    
    def _train_isotonic(
        self,
        meta_preds: np.ndarray,
        y_true: np.ndarray
    ) -> None:
        """Train isotonic regression for calibration"""
        
        # Get weighted predictions
        weighted_pred = meta_preds @ self.meta_weights
        
        # Fit isotonic regression
        self.isotonic_model = IsotonicRegression(
            y_min=0,  # Prices should be positive
            out_of_bounds='clip'
        )
        self.isotonic_model.fit(weighted_pred, y_true)
    
    def predict(
        self,
        base_preds: Dict[str, np.ndarray],
        return_intermediate: bool = False
    ) -> np.ndarray:
        """
        Generate ensemble predictions
        
        Args:
            base_preds: Dict of base model predictions
            return_intermediate: If True, return intermediate predictions too
            
        Returns:
            Final predictions (or tuple with intermediate if requested)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Prepare stack features
        X_stack = self._prepare_stack_features(base_preds)
        
        # Level 1: Meta-learner predictions
        meta_preds = np.column_stack([
            self.meta_ridge.predict(X_stack),
            self.meta_elasticnet.predict(X_stack),
            self.meta_lgbm.predict(X_stack)
        ])
        
        # Level 2: Weighted combination
        weighted_pred = meta_preds @ self.meta_weights
        
        # Level 2: Isotonic calibration
        if self.use_isotonic and self.isotonic_model is not None:
            final_pred = self.isotonic_model.predict(weighted_pred)
        else:
            final_pred = weighted_pred
        
        # Ensure non-negative predictions
        final_pred = np.maximum(final_pred, 0)
        
        if return_intermediate:
            return final_pred, meta_preds, weighted_pred
        return final_pred
    
    def save(self, path: Path) -> None:
        """Save ensemble to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save state
        state = {
            'n_folds': self.n_folds,
            'use_isotonic': self.use_isotonic,
            'optimize_weights': self.optimize_weights,
            'random_state': self.random_state,
            'meta_weights': self.meta_weights,
            'is_fitted': self.is_fitted
        }
        
        with open(path / 'ensemble_state.json', 'w') as f:
            json.dump({k: v if not isinstance(v, np.ndarray) else v.tolist() 
                      for k, v in state.items()}, f, indent=2)
        
        # Save models
        with open(path / 'meta_ridge.pkl', 'wb') as f:
            pickle.dump(self.meta_ridge, f)
        
        with open(path / 'meta_elasticnet.pkl', 'wb') as f:
            pickle.dump(self.meta_elasticnet, f)
        
        with open(path / 'meta_lgbm.pkl', 'wb') as f:
            pickle.dump(self.meta_lgbm, f)
        
        if self.isotonic_model is not None:
            with open(path / 'isotonic_model.pkl', 'wb') as f:
                pickle.dump(self.isotonic_model, f)
        
        logger.info(f"Ensemble saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'StackingEnsemble':
        """Load ensemble from disk"""
        path = Path(path)
        
        # Load state
        with open(path / 'ensemble_state.json', 'r') as f:
            state = json.load(f)
        
        ensemble = cls(
            n_folds=state['n_folds'],
            use_isotonic=state['use_isotonic'],
            optimize_weights=state['optimize_weights'],
            random_state=state['random_state']
        )
        
        ensemble.meta_weights = np.array(state['meta_weights'])
        ensemble.is_fitted = state['is_fitted']
        
        # Load models
        with open(path / 'meta_ridge.pkl', 'rb') as f:
            ensemble.meta_ridge = pickle.load(f)
        
        with open(path / 'meta_elasticnet.pkl', 'rb') as f:
            ensemble.meta_elasticnet = pickle.load(f)
        
        with open(path / 'meta_lgbm.pkl', 'rb') as f:
            ensemble.meta_lgbm = pickle.load(f)
        
        isotonic_path = path / 'isotonic_model.pkl'
        if isotonic_path.exists():
            with open(isotonic_path, 'rb') as f:
                ensemble.isotonic_model = pickle.load(f)
        
        logger.info(f"Ensemble loaded from {path}")
        return ensemble


def optimize_base_model_weights(
    preds_dict: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Optimize weights for base model predictions (simple weighted average)
    
    Args:
        preds_dict: Dict of model name -> predictions
        y_true: True target values
        
    Returns:
        Tuple of (weights dict, weighted predictions)
    """
    model_names = sorted(preds_dict.keys())
    preds_matrix = np.column_stack([preds_dict[name].ravel() for name in model_names])
    
    def objective(weights):
        weights = np.abs(weights)
        weights = weights / np.sum(weights)
        weighted_pred = preds_matrix @ weights
        return calculate_smape(y_true, weighted_pred)
    
    n_models = len(model_names)
    x0 = np.ones(n_models) / n_models
    
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 1000}
    )
    
    weights = np.abs(result.x)
    weights = weights / np.sum(weights)
    
    weights_dict = {name: w for name, w in zip(model_names, weights)}
    weighted_preds = preds_matrix @ weights
    
    return weights_dict, weighted_preds


def train_ensemble(
    train_preds_dict: Dict[str, np.ndarray],
    val_preds_dict: Dict[str, np.ndarray],
    test_preds_dict: Dict[str, np.ndarray],
    y_train: np.ndarray,
    y_val: np.ndarray,
    config: Optional[Dict] = None,
    save_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Train 2-level stacking ensemble with isotonic calibration
    
    Args:
        train_preds_dict: Dict of base model predictions on training set
        val_preds_dict: Dict of base model predictions on validation set
        test_preds_dict: Dict of base model predictions on test set
        y_train: Training targets
        y_val: Validation targets
        config: Optional configuration dict
        save_dir: Directory to save ensemble
        
    Returns:
        Dict with ensemble model and predictions
    """
    logger.info("=" * 60)
    logger.info("ENSEMBLE TRAINING")
    logger.info("=" * 60)
    
    config = config or TRAIN_CONFIG
    
    # Log base model performance
    logger.info("\nBase model validation performance:")
    for name, preds in val_preds_dict.items():
        smape = calculate_smape(y_val, preds)
        logger.info(f"  {name}: SMAPE = {smape:.4f}")
    
    # Simple weighted average baseline
    logger.info("\nOptimizing simple weighted average...")
    simple_weights, simple_val_preds = optimize_base_model_weights(val_preds_dict, y_val)
    simple_smape = calculate_smape(y_val, simple_val_preds)
    
    logger.info(f"Simple weighted average SMAPE: {simple_smape:.4f}")
    logger.info(f"Weights: {simple_weights}")
    
    # Train stacking ensemble
    ensemble = StackingEnsemble(
        n_folds=5,
        use_isotonic=True,
        optimize_weights=True,
        random_state=42
    )
    
    ensemble.fit(train_preds_dict, val_preds_dict, y_train, y_val)
    
    # Generate predictions
    val_predictions = ensemble.predict(val_preds_dict)
    test_predictions = ensemble.predict(test_preds_dict)
    
    # Evaluate
    ensemble_val_smape = calculate_smape(y_val, val_predictions)
    
    logger.info("\n" + "=" * 60)
    logger.info("ENSEMBLE RESULTS")
    logger.info("=" * 60)
    logger.info(f"Simple weighted average SMAPE: {simple_smape:.4f}")
    logger.info(f"Stacking ensemble SMAPE: {ensemble_val_smape:.4f}")
    logger.info(f"Improvement: {simple_smape - ensemble_val_smape:.4f}")
    
    # Save ensemble
    if save_dir is not None:
        ensemble.save(save_dir)
    
    # Generate simple weighted average predictions for comparison
    simple_test_preds = np.zeros(len(next(iter(test_preds_dict.values()))))
    for name, weight in simple_weights.items():
        simple_test_preds += weight * test_preds_dict[name].ravel()
    
    results = {
        'ensemble': ensemble,
        'val_predictions': val_predictions,
        'test_predictions': test_predictions,
        'val_smape': ensemble_val_smape,
        'simple_weights': simple_weights,
        'simple_val_smape': simple_smape,
        'simple_test_predictions': simple_test_preds,
        'meta_weights': {
            'ridge': ensemble.meta_weights[0],
            'elasticnet': ensemble.meta_weights[1],
            'lightgbm': ensemble.meta_weights[2]
        }
    }
    
    return results


def create_submission(
    sample_ids: np.ndarray,
    predictions: np.ndarray,
    output_path: Path,
    description: str = ''
) -> pd.DataFrame:
    """
    Create submission DataFrame and save to CSV
    
    Args:
        sample_ids: Array of sample IDs
        predictions: Array of price predictions
        output_path: Path to save submission
        description: Optional description for logging
        
    Returns:
        Submission DataFrame
    """
    # Ensure non-negative predictions
    predictions = np.maximum(predictions, 0)
    
    submission = pd.DataFrame({
        'sample_id': sample_ids,
        'price': predictions
    })
    
    # Save submission
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    # Log statistics
    logger.info(f"\nSubmission {description}:")
    logger.info(f"  Samples: {len(submission)}")
    logger.info(f"  Price range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    logger.info(f"  Mean price: {predictions.mean():.2f}")
    logger.info(f"  Median price: {np.median(predictions):.2f}")
    logger.info(f"  Saved to: {output_path}")
    
    return submission


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Ensemble Training Module - Example Usage")
    print("=" * 60)
    
    np.random.seed(42)
    n_train = 1000
    n_val = 200
    n_test = 500
    
    # Synthetic targets
    y_train = np.abs(np.random.randn(n_train) * 100 + 500)
    y_val = np.abs(np.random.randn(n_val) * 100 + 500)
    
    # Synthetic base model predictions (with some correlation to targets)
    train_preds = {
        'neural_net': y_train + np.random.randn(n_train) * 50,
        'lightgbm': y_train + np.random.randn(n_train) * 45,
        'xgboost': y_train + np.random.randn(n_train) * 48,
        'catboost': y_train + np.random.randn(n_train) * 47
    }
    
    val_preds = {
        'neural_net': y_val + np.random.randn(n_val) * 50,
        'lightgbm': y_val + np.random.randn(n_val) * 45,
        'xgboost': y_val + np.random.randn(n_val) * 48,
        'catboost': y_val + np.random.randn(n_val) * 47
    }
    
    test_preds = {
        'neural_net': np.abs(np.random.randn(n_test) * 100 + 500),
        'lightgbm': np.abs(np.random.randn(n_test) * 100 + 500),
        'xgboost': np.abs(np.random.randn(n_test) * 100 + 500),
        'catboost': np.abs(np.random.randn(n_test) * 100 + 500)
    }
    
    # Train ensemble
    results = train_ensemble(
        train_preds, val_preds, test_preds,
        y_train, y_val,
        save_dir=Path('models/ensemble')
    )
    
    print(f"\nFinal validation SMAPE: {results['val_smape']:.4f}")
    print(f"Test predictions shape: {results['test_predictions'].shape}")
