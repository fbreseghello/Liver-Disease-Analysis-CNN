"""
General utilities for liver disease analysis project.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import json
import os


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    Load data from CSV file.
    
    Args:
        filepath: Path to CSV file
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath, **kwargs)
        print(f"✓ Data loaded successfully from {filepath}")
        print(f"  Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def split_features_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into features and target.
    
    Args:
        df: Input dataframe
        target_column: Name of target column
        
    Returns:
        Tuple of (features, target)
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"✓ Data split into features and target")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        stratify: Whether to stratify by target
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if stratify else None
    )
    
    print(f"✓ Data split into train and test sets")
    print(f"  Train size: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"  Test size: {X_test.shape[0]} ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance training data.
    
    Args:
        X_train: Training features
        y_train: Training target
        random_state: Random seed
        k_neighbors: Number of neighbors for SMOTE
        
    Returns:
        Tuple of (X_balanced, y_balanced)
    """
    print("Applying SMOTE to balance classes...")
    print(f"  Original distribution: {np.bincount(y_train)}")
    
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"✓ SMOTE applied successfully")
    print(f"  Balanced distribution: {np.bincount(y_balanced)}")
    print(f"  New shape: {X_balanced.shape}")
    
    return X_balanced, y_balanced


def save_results(
    results: Dict[str, Any],
    filepath: str
):
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Path to save JSON
    """
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    print(f"✓ Results saved to {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Dictionary of results
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    print(f"✓ Results loaded from {filepath}")
    return results


def print_dataset_info(df: pd.DataFrame):
    """
    Print comprehensive dataset information.
    
    Args:
        df: Input dataframe
    """
    print("=" * 70)
    print("DATASET INFORMATION")
    print("=" * 70)
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\n" + "-" * 70)
    print("Column Information:")
    print("-" * 70)
    
    info_df = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Null %': (df.isnull().sum() / len(df) * 100).values.round(2),
        'Unique': df.nunique().values
    })
    
    print(info_df.to_string(index=False))
    
    print("\n" + "-" * 70)
    print("Numeric Features Summary:")
    print("-" * 70)
    print(df.describe().round(2))
    
    print("\n" + "=" * 70)


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    try:
        import random
        random.seed(seed)
    except:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except:
        pass
    
    print(f"✓ Random seeds set to {seed}")
