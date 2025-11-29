"""
Visualization Module for Amazon ML Challenge

Provides plotting functions for:
- Training curves (loss, metrics)
- Prediction analysis (actual vs predicted)
- Error distribution analysis
- Feature importance visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import seaborn as sns
import logging

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


def setup_figure(figsize: Tuple[int, int] = (12, 8)) -> Tuple[plt.Figure, Any]:
    """Create figure with consistent styling"""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    val_smapes: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
    title: str = "Training Curves"
) -> plt.Figure:
    """
    Plot training and validation curves
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: Optional list of validation losses per epoch
        val_smapes: Optional list of validation SMAPE values per epoch
        save_path: Optional path to save the figure
        title: Title for the plot
        
    Returns:
        Matplotlib figure
    """
    n_plots = 1 + (val_losses is not None) + (val_smapes is not None)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_losses) + 1)
    plot_idx = 0
    
    # Training loss
    axes[plot_idx].plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss')
    if val_losses is not None:
        axes[plot_idx].plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss')
    axes[plot_idx].set_xlabel('Epoch', fontsize=12)
    axes[plot_idx].set_ylabel('Loss', fontsize=12)
    axes[plot_idx].set_title('Training & Validation Loss', fontsize=14)
    axes[plot_idx].legend(fontsize=10)
    axes[plot_idx].grid(True, alpha=0.3)
    plot_idx += 1
    
    # Validation SMAPE
    if val_smapes is not None and plot_idx < n_plots:
        axes[plot_idx].plot(epochs, val_smapes, 'g-', linewidth=2, marker='o', markersize=4)
        axes[plot_idx].set_xlabel('Epoch', fontsize=12)
        axes[plot_idx].set_ylabel('SMAPE (%)', fontsize=12)
        axes[plot_idx].set_title('Validation SMAPE', fontsize=14)
        axes[plot_idx].grid(True, alpha=0.3)
        
        # Add best epoch marker
        best_idx = np.argmin(val_smapes)
        axes[plot_idx].axvline(x=best_idx + 1, color='red', linestyle='--', alpha=0.7, 
                               label=f'Best: {val_smapes[best_idx]:.4f} @ epoch {best_idx + 1}')
        axes[plot_idx].legend(fontsize=10)
        plot_idx += 1
    
    # Learning rate (if we have separate losses)
    if val_losses is not None and val_smapes is not None:
        # Plot loss vs SMAPE correlation
        axes[plot_idx].scatter(val_losses, val_smapes, c=epochs, cmap='viridis', s=50)
        axes[plot_idx].set_xlabel('Validation Loss', fontsize=12)
        axes[plot_idx].set_ylabel('Validation SMAPE (%)', fontsize=12)
        axes[plot_idx].set_title('Loss vs SMAPE Correlation', fontsize=14)
        cbar = plt.colorbar(axes[plot_idx].collections[0], ax=axes[plot_idx])
        cbar.set_label('Epoch')
        axes[plot_idx].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training curves saved to {save_path}")
    
    return fig


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = 'Validation',
    save_path: Optional[Path] = None,
    smape: Optional[float] = None
) -> plt.Figure:
    """
    Plot predicted vs actual values with residuals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        split_name: Name of the data split
        save_path: Optional path to save the figure
        smape: Optional SMAPE value to display
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.5, 1, 1])
    
    # Scatter plot: Actual vs Predicted
    ax1 = fig.add_subplot(gs[0])
    
    # Use log scale for better visualization
    y_true_pos = np.maximum(y_true, 1)
    y_pred_pos = np.maximum(y_pred, 1)
    
    scatter = ax1.scatter(y_true_pos, y_pred_pos, alpha=0.5, s=10, c='steelblue')
    
    # Perfect prediction line
    lims = [min(y_true_pos.min(), y_pred_pos.min()), 
            max(y_true_pos.max(), y_pred_pos.max())]
    ax1.plot(lims, lims, 'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Actual Price (₹)', fontsize=12)
    ax1.set_ylabel('Predicted Price (₹)', fontsize=12)
    
    title = f'{split_name}: Predicted vs Actual'
    if smape is not None:
        title += f'\nSMAPE: {smape:.4f}'
    ax1.set_title(title, fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = fig.add_subplot(gs[1])
    
    residuals = y_pred - y_true
    percent_error = 100 * (y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    
    ax2.scatter(y_true_pos, percent_error, alpha=0.5, s=10, c='steelblue')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax2.set_xscale('log')
    ax2.set_xlabel('Actual Price (₹)', fontsize=12)
    ax2.set_ylabel('Percent Error (%)', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=14)
    ax2.set_ylim(-100, 100)
    ax2.grid(True, alpha=0.3)
    
    # Error histogram
    ax3 = fig.add_subplot(gs[2])
    
    ax3.hist(percent_error, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
    ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax3.axvline(x=np.median(percent_error), color='green', linestyle='--', 
                linewidth=2, label=f'Median: {np.median(percent_error):.2f}%')
    ax3.set_xlabel('Percent Error (%)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=14)
    ax3.set_xlim(-100, 100)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Prediction plot saved to {save_path}")
    
    return fig


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[Path] = None,
    title: str = 'Error Distribution Analysis'
) -> plt.Figure:
    """
    Plot comprehensive error distribution analysis
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Calculate various error metrics
    abs_error = np.abs(y_pred - y_true)
    percent_error = 100 * (y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    smape_error = 100 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    
    # Absolute error distribution
    ax1 = axes[0, 0]
    ax1.hist(abs_error, bins=50, edgecolor='white', alpha=0.7, color='steelblue')
    ax1.axvline(x=np.mean(abs_error), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: ₹{np.mean(abs_error):.2f}')
    ax1.axvline(x=np.median(abs_error), color='green', linestyle='--',
                linewidth=2, label=f'Median: ₹{np.median(abs_error):.2f}')
    ax1.set_xlabel('Absolute Error (₹)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Absolute Error Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # SMAPE distribution
    ax2 = axes[0, 1]
    ax2.hist(smape_error, bins=50, edgecolor='white', alpha=0.7, color='coral')
    ax2.axvline(x=np.mean(smape_error), color='red', linestyle='--',
                linewidth=2, label=f'Mean SMAPE: {np.mean(smape_error):.2f}%')
    ax2.set_xlabel('SMAPE per Sample (%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('SMAPE Distribution', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Error by price range
    ax3 = axes[1, 0]
    
    # Create price bins
    price_bins = [0, 100, 500, 1000, 5000, 10000, 50000, np.inf]
    price_labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', '10K-50K', '50K+']
    bin_indices = np.digitize(y_true, price_bins) - 1
    
    bin_smapes = []
    for i in range(len(price_labels)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_smape = np.mean(smape_error[mask])
            bin_smapes.append(bin_smape)
        else:
            bin_smapes.append(0)
    
    bars = ax3.bar(price_labels, bin_smapes, color='teal', edgecolor='white', alpha=0.7)
    ax3.axhline(y=np.mean(smape_error), color='red', linestyle='--', 
                linewidth=2, label=f'Overall SMAPE: {np.mean(smape_error):.2f}%')
    ax3.set_xlabel('Price Range (₹)', fontsize=12)
    ax3.set_ylabel('Mean SMAPE (%)', fontsize=12)
    ax3.set_title('SMAPE by Price Range', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Add counts on bars
    for bar, label in zip(bars, price_labels):
        idx = price_labels.index(label)
        count = np.sum(bin_indices == idx)
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # QQ plot
    ax4 = axes[1, 1]
    
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(percent_error, dist='norm')
    ax4.scatter(osm, osr, s=10, alpha=0.5, color='steelblue')
    ax4.plot(osm, intercept + slope * osm, 'r-', linewidth=2, label=f'R² = {r**2:.4f}')
    ax4.set_xlabel('Theoretical Quantiles', fontsize=12)
    ax4.set_ylabel('Sample Quantiles', fontsize=12)
    ax4.set_title('Q-Q Plot (Percent Error)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Error distribution plot saved to {save_path}")
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_k: int = 20,
    save_path: Optional[Path] = None,
    title: str = 'Feature Importance'
) -> plt.Figure:
    """
    Plot feature importance bar chart
    
    Args:
        importance_dict: Dictionary of feature name -> importance value
        top_k: Number of top features to display
        save_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
    features = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]
    
    fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
    
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color='steelblue', edgecolor='white', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, importance in zip(bars, importances):
        ax.text(bar.get_width() + 0.01 * max(importances), bar.get_y() + bar.get_height() / 2,
                f'{importance:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    return fig


def plot_ensemble_weights(
    weights_dict: Dict[str, float],
    save_path: Optional[Path] = None,
    title: str = 'Ensemble Model Weights'
) -> plt.Figure:
    """
    Plot ensemble model weights as a pie/bar chart
    
    Args:
        weights_dict: Dictionary of model name -> weight
        save_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    models = list(weights_dict.keys())
    weights = list(weights_dict.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Pie chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(weights, labels=models, autopct='%1.1f%%',
                                        colors=colors, explode=[0.02] * len(models))
    ax1.set_title('Weight Distribution', fontsize=14)
    
    # Bar chart
    ax2 = axes[1]
    bars = ax2.bar(models, weights, color=colors, edgecolor='white', alpha=0.8)
    ax2.set_ylabel('Weight', fontsize=12)
    ax2.set_title('Model Weights', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, weight in zip(bars, weights):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{weight:.4f}', ha='center', va='bottom', fontsize=10)
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Ensemble weights plot saved to {save_path}")
    
    return fig


def plot_model_comparison(
    model_metrics: Dict[str, Dict[str, float]],
    metric_name: str = 'smape',
    save_path: Optional[Path] = None,
    title: str = 'Model Comparison'
) -> plt.Figure:
    """
    Plot comparison of multiple models
    
    Args:
        model_metrics: Dict of model_name -> {metric_name: value}
        metric_name: Name of the metric to compare
        save_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(model_metrics.keys())
    values = [model_metrics[m].get(metric_name, 0) for m in models]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(models)))
    
    # Sort by value
    sorted_indices = np.argsort(values)
    models = [models[i] for i in sorted_indices]
    values = [values[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    bars = ax.barh(models, values, color=colors, edgecolor='white', alpha=0.8)
    
    ax.set_xlabel(metric_name.upper(), fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, value in zip(bars, values):
        ax.text(bar.get_width() + 0.01 * max(values), bar.get_y() + bar.get_height() / 2,
                f'{value:.4f}', va='center', fontsize=10)
    
    # Highlight best model
    best_idx = 0  # Already sorted
    bars[best_idx].set_edgecolor('gold')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    return fig


def plot_price_distribution(
    y_train: np.ndarray,
    y_val: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
    title: str = 'Price Distribution'
) -> plt.Figure:
    """
    Plot price distribution comparison
    
    Args:
        y_train: Training prices
        y_val: Optional validation prices
        y_pred: Optional predicted prices
        save_path: Optional path to save the figure
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log-scale histogram
    ax1 = axes[0]
    
    # Use log-transformed values for binning
    y_train_log = np.log1p(y_train)
    bins = np.linspace(y_train_log.min(), y_train_log.max(), 50)
    
    ax1.hist(y_train_log, bins=bins, alpha=0.7, label='Train', color='steelblue', edgecolor='white')
    
    if y_val is not None:
        y_val_log = np.log1p(y_val)
        ax1.hist(y_val_log, bins=bins, alpha=0.7, label='Validation', color='coral', edgecolor='white')
    
    if y_pred is not None:
        y_pred_log = np.log1p(y_pred)
        ax1.hist(y_pred_log, bins=bins, alpha=0.5, label='Predicted', color='green', edgecolor='white')
    
    ax1.set_xlabel('log(Price + 1)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Log-Scale Price Distribution', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Box plot comparison
    ax2 = axes[1]
    
    data_to_plot = [y_train]
    labels = ['Train']
    
    if y_val is not None:
        data_to_plot.append(y_val)
        labels.append('Validation')
    
    if y_pred is not None:
        data_to_plot.append(y_pred)
        labels.append('Predicted')
    
    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['steelblue', 'coral', 'green'][:len(data_to_plot)]
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.set_yscale('log')
    ax2.set_ylabel('Price (₹, log scale)', fontsize=12)
    ax2.set_title('Price Distribution Box Plot', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics
    for i, data in enumerate(data_to_plot):
        median = np.median(data)
        mean = np.mean(data)
        ax2.annotate(f'μ={mean:.0f}\nM={median:.0f}', 
                     xy=(i + 1, median), 
                     xytext=(i + 1.2, median * 1.5),
                     fontsize=8, ha='left')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Price distribution plot saved to {save_path}")
    
    return fig


def create_training_report(
    train_history: Dict[str, List[float]],
    val_history: Dict[str, List[float]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = 'Model',
    save_dir: Optional[Path] = None
) -> None:
    """
    Create a comprehensive training report with multiple plots
    
    Args:
        train_history: Dictionary of training metrics over epochs
        val_history: Dictionary of validation metrics over epochs
        y_true: True validation values
        y_pred: Predicted validation values
        model_name: Name of the model for the report
        save_dir: Directory to save all plots
    """
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating training report for {model_name}")
    
    # Training curves
    train_losses = train_history.get('loss', [])
    val_losses = val_history.get('loss', [])
    val_smapes = val_history.get('smape', [])
    
    if train_losses:
        plot_training_curves(
            train_losses, val_losses, val_smapes,
            save_path=save_dir / f'{model_name.lower()}_training_curves.png' if save_dir else None,
            title=f'{model_name} Training Curves'
        )
    
    # Prediction analysis
    from src.utils.metrics import calculate_smape
    smape = calculate_smape(y_true, y_pred)
    
    plot_predictions(
        y_true, y_pred,
        split_name='Validation',
        save_path=save_dir / f'{model_name.lower()}_predictions.png' if save_dir else None,
        smape=smape
    )
    
    # Error distribution
    plot_error_distribution(
        y_true, y_pred,
        save_path=save_dir / f'{model_name.lower()}_error_dist.png' if save_dir else None,
        title=f'{model_name} Error Distribution'
    )
    
    logger.info(f"Training report saved to {save_dir}")


if __name__ == "__main__":
    # Example usage
    print("Visualization Module - Example Usage")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 500
    y_true = np.abs(np.random.randn(n_samples) * 100 + 500)
    y_pred = y_true + np.random.randn(n_samples) * 50
    
    # Training history
    epochs = 20
    train_losses = [1.0 - 0.03 * i + np.random.randn() * 0.05 for i in range(epochs)]
    val_losses = [1.1 - 0.025 * i + np.random.randn() * 0.08 for i in range(epochs)]
    val_smapes = [15.0 - 0.3 * i + np.random.randn() * 0.5 for i in range(epochs)]
    
    # Create plots
    fig1 = plot_training_curves(train_losses, val_losses, val_smapes)
    fig2 = plot_predictions(y_true, y_pred, smape=8.5)
    fig3 = plot_error_distribution(y_true, y_pred)
    
    # Feature importance
    importance = {f'feature_{i}': np.random.rand() for i in range(30)}
    fig4 = plot_feature_importance(importance)
    
    # Ensemble weights
    weights = {'Neural Net': 0.35, 'LightGBM': 0.30, 'XGBoost': 0.20, 'CatBoost': 0.15}
    fig5 = plot_ensemble_weights(weights)
    
    plt.show()
    print("\nVisualization examples complete!")
