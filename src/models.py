"""
Model building and evaluation utilities for liver disease analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
import optuna
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import joblib


class ModelBuilder:
    """Build and optimize machine learning models."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model builder.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        
    def create_model_from_params(self, params: Dict[str, Any]) -> VotingClassifier:
        """
        Create ensemble model from hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            VotingClassifier with configured estimators
        """
        # Logistic Regression
        lr_l1_ratio = params.get('lr_l1_ratio', None)
        lr_solver = params.get('lr_solver1', params.get('lr_solver2', 'saga'))
        
        lr = LogisticRegression(
            penalty=params['lr_penalty'],
            tol=params['lr_tol'],
            C=params['lr_C'],
            solver=lr_solver,
            l1_ratio=lr_l1_ratio,
            random_state=self.random_state,
            max_iter=1000
        )
        
        # K-Nearest Neighbors
        knn = KNeighborsClassifier(
            n_neighbors=params['knn_neighbors'],
            weights=params['knn_weights'],
            p=params['knn_p']
        )
        
        # Support Vector Machine
        svm = SVC(
            C=params['svm_C'],
            kernel=params['svm_kernel'],
            degree=params.get('svm_degree', 3),
            tol=params['svm_tol'],
            gamma=params.get('svm_gamma', 'scale'),
            random_state=self.random_state,
            probability=True
        )
        
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=params['rf_estimators'],
            criterion=params['rf_criterion'],
            max_depth=params['rf_max_depth'],
            min_samples_split=params['rf_min_samples_split'],
            min_samples_leaf=params['rf_min_samples_leaf'],
            max_features=params.get('rf_max_features', 'sqrt'),
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Naive Bayes
        nb = GaussianNB(var_smoothing=params['nb_smoothing'])
        
        # Ensemble Model
        vc = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('knn', knn),
                ('svm', svm),
                ('rf', rf),
                ('nb', nb)
            ],
            weights=[
                params['lr_w'],
                params['knn_w'],
                params['svm_w'],
                params['rf_w'],
                params['nb_w']
            ],
            voting='soft',
            n_jobs=-1
        )
        
        return vc
    
    def evaluate_model(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        target_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            target_names: Names of target classes
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = metrics.recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = metrics.f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = metrics.confusion_matrix(y_test, y_pred)
        
        # Classification report
        report = metrics.classification_report(
            y_test, 
            y_pred, 
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred
        }
        
        return results
    
    def save_model(self, model, filepath: str):
        """
        Save model to disk.
        
        Args:
            model: Model to save
            filepath: Path to save model
        """
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from disk.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


def print_evaluation_results(results: Dict[str, Any]):
    """
    Print formatted evaluation results.
    
    Args:
        results: Dictionary from evaluate_model
    """
    print("=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    print("=" * 60)
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    print("\nClassification Report:")
    report_df = pd.DataFrame(results['classification_report']).transpose()
    print(report_df.round(3))
