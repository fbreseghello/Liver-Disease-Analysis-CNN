"""
Visualization utilities for liver disease analysis.
Provides functions for creating informative plots and charts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def plot_feature_distributions(
    df: pd.DataFrame,
    features: List[str],
    target: str = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: str = None
):
    """
    Plot distribution of features.
    
    Args:
        df: Input dataframe
        features: List of features to plot
        target: Target column for color coding
        figsize: Figure size
        save_path: Path to save figure
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        if df[feature].dtype in ['int64', 'float64']:
            if target:
                for label in df[target].unique():
                    subset = df[df[target] == label]
                    ax.hist(subset[feature], alpha=0.5, label=str(label), bins=30)
                ax.legend()
            else:
                ax.hist(df[feature], bins=30, edgecolor='black', alpha=0.7)
        else:
            if target:
                pd.crosstab(df[feature], df[target]).plot(kind='bar', ax=ax, stacked=False)
            else:
                df[feature].value_counts().plot(kind='bar', ax=ax, edgecolor='black')
        
        ax.set_title(f'Distribution of {feature}', fontsize=12, fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(
    df: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 10),
    save_path: str = None
):
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: Input dataframe
        figsize: Figure size
        save_path: Path to save figure
    """
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr = df.select_dtypes(include=[np.number]).corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Create heatmap
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8}
    )
    
    plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    disp.plot(cmap='Blues', ax=ax, values_format='d')
    ax.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def plot_class_distribution(
    y: pd.Series,
    title: str = 'Class Distribution',
    figsize: Tuple[int, int] = (10, 6),
    save_path: str = None
):
    """
    Plot distribution of target classes.
    
    Args:
        y: Target variable
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    counts = y.value_counts()
    colors = sns.color_palette('husl', len(counts))
    ax1.bar(counts.index, counts.values, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title(f'{title} - Counts', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (idx, val) in enumerate(counts.items()):
        ax1.text(i, val, str(val), ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.05] * len(counts)
    )
    ax2.set_title(f'{title} - Proportions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def plot_pca_variance(
    pca: PCA,
    figsize: Tuple[int, int] = (12, 5),
    save_path: str = None
):
    """
    Plot PCA explained variance.
    
    Args:
        pca: Fitted PCA object
        figsize: Figure size
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Individual variance
    ax1.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        alpha=0.7,
        edgecolor='black'
    )
    ax1.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Variance Ratio', fontsize=12, fontweight='bold')
    ax1.set_title('Explained Variance by Component', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3, axis='y')
    
    # Cumulative variance
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(range(1, len(cumsum) + 1), cumsum, marker='o', linewidth=2)
    ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    ax2.set_xlabel('Number of Components', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative Variance Ratio', fontsize=12, fontweight='bold')
    ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    top_n: int = 15,
    figsize: Tuple[int, int] = (10, 8),
    save_path: str = None
):
    """
    Plot feature importance.
    
    Args:
        feature_names: List of feature names
        importances: Feature importance values
        top_n: Number of top features to show
        figsize: Figure size
        save_path: Path to save figure
    """
    # Create dataframe and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=figsize)
    
    colors = sns.color_palette('viridis', len(importance_df))
    plt.barh(range(len(importance_df)), importance_df['importance'], color=colors, edgecolor='black')
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Feature', fontsize=12, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to {save_path}")
    
    plt.show()
