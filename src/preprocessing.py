"""
Preprocessing utilities for liver disease analysis.
Contains functions for data cleaning, imputation, and transformation.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from typing import Tuple, List


class DataPreprocessor:
    """Handle data preprocessing for liver disease dataset."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        
    def create_preprocessing_pipeline(
        self, 
        numeric_features: List[str], 
        categorical_features: List[str]
    ) -> ColumnTransformer:
        """
        Create preprocessing pipeline for numeric and categorical features.
        
        Args:
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            
        Returns:
            Fitted ColumnTransformer object
        """
        # Numeric transformer: KNN Imputation + Standard Scaling
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)),
            ('scaler', StandardScaler())
        ])
        
        # Categorical transformer: One-Hot Encoding
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(
        self, 
        X: pd.DataFrame, 
        numeric_features: List[str], 
        categorical_features: List[str]
    ) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Input features
            numeric_features: List of numeric column names
            categorical_features: List of categorical column names
            
        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            self.create_preprocessing_pipeline(numeric_features, categorical_features)
        
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Store feature names for later use
        try:
            # Get feature names from transformers
            num_features = numeric_features
            cat_features = self.preprocessor.named_transformers_['cat']\
                .named_steps['onehot'].get_feature_names_out(categorical_features)
            self.feature_names = list(num_features) + list(cat_features)
        except:
            self.feature_names = None
            
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature array
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(X)


def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for missing values in dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with missing value statistics
    """
    missing_data = pd.DataFrame({
        'Column': df.columns,
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values
    })
    
    missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values(
        'Missing_Percentage', ascending=False
    ).reset_index(drop=True)
    
    return missing_data


def get_feature_types(df: pd.DataFrame, exclude_columns: List[str] = None) -> Tuple[List[str], List[str]]:
    """
    Automatically detect numeric and categorical features.
    
    Args:
        df: Input dataframe
        exclude_columns: Columns to exclude (e.g., target variable)
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    if exclude_columns is None:
        exclude_columns = []
    
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove excluded columns
    numeric_features = [f for f in numeric_features if f not in exclude_columns]
    categorical_features = [f for f in categorical_features if f not in exclude_columns]
    
    return numeric_features, categorical_features
