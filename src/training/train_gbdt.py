"""
GBDT training module for Amazon ML Price Prediction.

This module provides training for:
- LightGBM with custom SMAPE objective
- XGBoost with custom SMAPE objective
- CatBoost with GPU support
- Optuna hyperparameter optimization

Validates: Requirements 6.1-6.6
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Callable
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, Pool

from config import config
from src.models.losses import (
    lgb_smape_objective,
    lgb_smape_eval,
    xgb_smape_objective,
    xgb_smape_eval
)
from src.utils.metrics import calculate_smape, evaluate_predictions

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def optimize_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimize LightGBM hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets (log space)
        X_val: Validation features
        y_val: Validation targets (log space)
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        
    Returns:
        Best hyperparameters
        
    Validates: Requirements 6.4
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.warning("Optuna not installed. Using default parameters.")
        return config.LIGHTGBM_PARAMS.copy()
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'random_state': config.RANDOM_SEED,
            
            # Hyperparameters to optimize
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)  # Suppress output
            ]
        )
        
        # Evaluate
        val_pred = model.predict(X_val)
        smape = calculate_smape(np.expm1(y_val), np.expm1(val_pred))
        
        return smape
    
    # Create study
    sampler = TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler
    )
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    logger.info(f"Starting LightGBM hyperparameter optimization ({n_trials} trials)...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    logger.info(f"Best SMAPE: {study.best_value:.4f}%")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Merge with base params
    best_params = config.LIGHTGBM_PARAMS.copy()
    best_params.update(study.best_params)
    
    return best_params


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    use_custom_objective: bool = True,
    optimize: bool = False
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """
    Train LightGBM model with optional custom SMAPE objective.
    
    Args:
        X_train: Training features
        y_train: Training targets (log space)
        X_val: Validation features
        y_val: Validation targets (log space)
        params: Model parameters (if None, uses config defaults)
        use_custom_objective: Whether to use custom SMAPE objective
        optimize: Whether to run hyperparameter optimization
        
    Returns:
        Tuple of (trained_model, metrics)
        
    Validates: Requirements 6.1, 6.5, 6.6
    """
    logger.info("Training LightGBM...")
    
    # Get parameters
    if optimize:
        params = optimize_lightgbm(X_train, y_train, X_val, y_val)
    elif params is None:
        params = config.LIGHTGBM_PARAMS.copy()
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Prepare callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=100)
    ]
    
    # Train model
    if use_custom_objective:
        # Remove objective from params - we'll use custom
        train_params = {k: v for k, v in params.items() if k != 'objective'}
        train_params['metric'] = 'None'  # We use custom eval
        
        model = lgb.train(
            train_params,
            train_data,
            num_boost_round=params.get('n_estimators', 1000),
            valid_sets=[val_data],
            fobj=lgb_smape_objective,
            feval=lgb_smape_eval,
            callbacks=callbacks
        )
    else:
        model = lgb.train(
            params,
            train_data,
            num_boost_round=params.get('n_estimators', 1000),
            valid_sets=[val_data],
            callbacks=callbacks
        )
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_smape = calculate_smape(np.expm1(y_train), np.expm1(train_pred))
    val_smape = calculate_smape(np.expm1(y_val), np.expm1(val_pred))
    
    logger.info(f"LightGBM - Train SMAPE: {train_smape:.4f}%, Val SMAPE: {val_smape:.4f}%")
    
    # Save model
    model_path = config.LIGHTGBM_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"LightGBM model saved: {model_path}")
    
    metrics = {
        'train_smape': train_smape,
        'val_smape': val_smape,
        'best_iteration': model.best_iteration
    }
    
    return model, metrics


def optimize_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training targets (log space)
        X_val: Validation features
        y_val: Validation targets (log space)
        n_trials: Number of Optuna trials
        timeout: Timeout in seconds
        
    Returns:
        Best hyperparameters
        
    Validates: Requirements 6.4
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        logger.warning("Optuna not installed. Using default parameters.")
        return config.XGBOOST_PARAMS.copy()
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'booster': 'gbtree',
            'verbosity': 0,
            'random_state': config.RANDOM_SEED,
            
            # Hyperparameters to optimize
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train model
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            evals=[(dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        # Evaluate
        val_pred = model.predict(dval)
        smape = calculate_smape(np.expm1(y_val), np.expm1(val_pred))
        
        return smape
    
    # Create study
    sampler = TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(
        direction='minimize',
        sampler=sampler
    )
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    logger.info(f"Starting XGBoost hyperparameter optimization ({n_trials} trials)...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )
    
    logger.info(f"Best SMAPE: {study.best_value:.4f}%")
    logger.info(f"Best parameters: {study.best_params}")
    
    # Merge with base params
    best_params = config.XGBOOST_PARAMS.copy()
    best_params.update(study.best_params)
    
    return best_params


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    use_custom_objective: bool = True,
    optimize: bool = False
) -> Tuple[xgb.Booster, Dict[str, float]]:
    """
    Train XGBoost model with optional custom SMAPE objective.
    
    Args:
        X_train: Training features
        y_train: Training targets (log space)
        X_val: Validation features
        y_val: Validation targets (log space)
        params: Model parameters (if None, uses config defaults)
        use_custom_objective: Whether to use custom SMAPE objective
        optimize: Whether to run hyperparameter optimization
        
    Returns:
        Tuple of (trained_model, metrics)
        
    Validates: Requirements 6.2, 6.5, 6.6
    """
    logger.info("Training XGBoost...")
    
    # Get parameters
    if optimize:
        params = optimize_xgboost(X_train, y_train, X_val, y_val)
    elif params is None:
        params = config.XGBOOST_PARAMS.copy()
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    if use_custom_objective:
        # Remove objective from params
        train_params = {k: v for k, v in params.items() 
                       if k not in ['objective', 'eval_metric', 'n_estimators']}
        train_params['disable_default_eval_metric'] = 1
        
        model = xgb.train(
            train_params,
            dtrain,
            num_boost_round=params.get('n_estimators', 1000),
            evals=[(dtrain, 'train'), (dval, 'val')],
            obj=xgb_smape_objective,
            custom_metric=xgb_smape_eval,
            early_stopping_rounds=100,
            verbose_eval=100
        )
    else:
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 1000),
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=100,
            verbose_eval=100
        )
    
    # Evaluate
    train_pred = model.predict(xgb.DMatrix(X_train))
    val_pred = model.predict(dval)
    
    train_smape = calculate_smape(np.expm1(y_train), np.expm1(train_pred))
    val_smape = calculate_smape(np.expm1(y_val), np.expm1(val_pred))
    
    logger.info(f"XGBoost - Train SMAPE: {train_smape:.4f}%, Val SMAPE: {val_smape:.4f}%")
    
    # Save model
    model_path = config.XGBOOST_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"XGBoost model saved: {model_path}")
    
    metrics = {
        'train_smape': train_smape,
        'val_smape': val_smape,
        'best_iteration': model.best_iteration
    }
    
    return model, metrics


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
    use_gpu: bool = True
) -> Tuple[CatBoostRegressor, Dict[str, float]]:
    """
    Train CatBoost model with GPU support.
    
    Args:
        X_train: Training features
        y_train: Training targets (log space)
        X_val: Validation features
        y_val: Validation targets (log space)
        params: Model parameters (if None, uses config defaults)
        use_gpu: Whether to use GPU
        
    Returns:
        Tuple of (trained_model, metrics)
        
    Validates: Requirements 6.3, 6.5, 6.6
    """
    logger.info("Training CatBoost...")
    
    # Get parameters
    if params is None:
        params = config.CATBOOST_PARAMS.copy()
    
    # Adjust for GPU availability
    if not use_gpu:
        params['task_type'] = 'CPU'
        if 'devices' in params:
            del params['devices']
    
    # Create model
    model = CatBoostRegressor(**params)
    
    # Create pools
    train_pool = Pool(X_train, label=y_train)
    val_pool = Pool(X_val, label=y_val)
    
    # Train
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100,
        verbose=100
    )
    
    # Evaluate
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    train_smape = calculate_smape(np.expm1(y_train), np.expm1(train_pred))
    val_smape = calculate_smape(np.expm1(y_val), np.expm1(val_pred))
    
    logger.info(f"CatBoost - Train SMAPE: {train_smape:.4f}%, Val SMAPE: {val_smape:.4f}%")
    
    # Save model
    model_path = config.CATBOOST_MODEL
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    logger.info(f"CatBoost model saved: {model_path}")
    
    metrics = {
        'train_smape': train_smape,
        'val_smape': val_smape,
        'best_iteration': model.best_iteration_
    }
    
    return model, metrics


def train_all_gbdt_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    optimize: bool = False,
    use_custom_objectives: bool = True
) -> Dict[str, Tuple[Any, Dict[str, float]]]:
    """
    Train all GBDT models (LightGBM, XGBoost, CatBoost).
    
    Args:
        X_train: Training features
        y_train: Training targets (log space)
        X_val: Validation features
        y_val: Validation targets (log space)
        optimize: Whether to run hyperparameter optimization
        use_custom_objectives: Whether to use custom SMAPE objectives
        
    Returns:
        Dictionary mapping model name to (model, metrics) tuple
    """
    results = {}
    
    # Train LightGBM
    lgb_model, lgb_metrics = train_lightgbm(
        X_train, y_train, X_val, y_val,
        use_custom_objective=use_custom_objectives,
        optimize=optimize
    )
    results['lightgbm'] = (lgb_model, lgb_metrics)
    
    # Train XGBoost
    xgb_model, xgb_metrics = train_xgboost(
        X_train, y_train, X_val, y_val,
        use_custom_objective=use_custom_objectives,
        optimize=optimize
    )
    results['xgboost'] = (xgb_model, xgb_metrics)
    
    # Train CatBoost
    try:
        cb_model, cb_metrics = train_catboost(
            X_train, y_train, X_val, y_val,
            use_gpu=True
        )
        results['catboost'] = (cb_model, cb_metrics)
    except Exception as e:
        logger.warning(f"CatBoost GPU training failed: {e}")
        logger.info("Retrying with CPU...")
        cb_model, cb_metrics = train_catboost(
            X_train, y_train, X_val, y_val,
            use_gpu=False
        )
        results['catboost'] = (cb_model, cb_metrics)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("GBDT TRAINING SUMMARY")
    logger.info("=" * 60)
    for name, (model, metrics) in results.items():
        logger.info(f"{name.upper():12s} - Val SMAPE: {metrics['val_smape']:.4f}%")
    logger.info("=" * 60)
    
    return results


def generate_gbdt_predictions(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Generate predictions from all GBDT models.
    
    Args:
        models: Dictionary of trained models
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Dictionary with predictions for each model
        
    Validates: Requirements 6.6
    """
    predictions = {}
    
    for name, model in models.items():
        logger.info(f"Generating predictions for {name}...")
        
        if name == 'lightgbm':
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
        elif name == 'xgboost':
            train_pred = model.predict(xgb.DMatrix(X_train))
            val_pred = model.predict(xgb.DMatrix(X_val))
            test_pred = model.predict(xgb.DMatrix(X_test))
        elif name == 'catboost':
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
        else:
            logger.warning(f"Unknown model type: {name}")
            continue
        
        predictions[name] = {
            'train': train_pred,
            'val': val_pred,
            'test': test_pred
        }
    
    return predictions


def save_gbdt_predictions(
    predictions: Dict[str, Dict[str, np.ndarray]],
    output_dir: Optional[Path] = None
) -> None:
    """
    Save GBDT predictions to disk.
    
    Args:
        predictions: Dictionary of predictions
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = config.PRED_DIR / 'gbdt'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, preds in predictions.items():
        model_dir = output_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        np.save(model_dir / 'train_pred.npy', preds['train'])
        np.save(model_dir / 'val_pred.npy', preds['val'])
        np.save(model_dir / 'test_pred.npy', preds['test'])
    
    logger.info(f"GBDT predictions saved to: {output_dir}")


def load_gbdt_predictions(
    input_dir: Optional[Path] = None
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load GBDT predictions from disk.
    
    Args:
        input_dir: Input directory
        
    Returns:
        Dictionary of predictions
    """
    if input_dir is None:
        input_dir = config.PRED_DIR / 'gbdt'
    
    predictions = {}
    
    for model_dir in input_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            predictions[model_name] = {
                'train': np.load(model_dir / 'train_pred.npy'),
                'val': np.load(model_dir / 'val_pred.npy'),
                'test': np.load(model_dir / 'test_pred.npy')
            }
    
    logger.info(f"GBDT predictions loaded from: {input_dir}")
    
    return predictions
