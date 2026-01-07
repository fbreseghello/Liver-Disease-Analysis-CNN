"""
Train Liver Disease Prediction Model
=====================================

This script trains a machine learning model for liver disease prediction
using the Hepatitis C dataset.

Usage:
    python train_model.py
    python train_model.py --trials 200 --test-size 0.3
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import preprocessing, models, utils, visualization
from src.config import *
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train liver disease prediction model')
    parser.add_argument('--data', type=str, default=HEPATITIS_DATA, 
                        help='Path to dataset CSV file')
    parser.add_argument('--trials', type=int, default=N_TRIALS,
                        help='Number of Optuna trials')
    parser.add_argument('--test-size', type=float, default=TEST_SIZE,
                        help='Test set size (0.0 to 1.0)')
    parser.add_argument('--output', type=str, default=MODEL_DIR,
                        help='Output directory for models')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()


def objective(trial, X_train, y_train, X_val, y_val, random_state):
    """Optuna objective function."""
    
    # Logistic Regression
    lr_penalty = trial.suggest_categorical('lr_penalty', ['l1', 'l2', 'elasticnet'])
    lr_l1_ratio = None
    
    if lr_penalty == 'l1':
        lr_solver = trial.suggest_categorical('lr_solver1', ['liblinear', 'saga'])
    elif lr_penalty == 'l2':
        lr_solver = trial.suggest_categorical('lr_solver2', ['newton-cg', 'lbfgs', 'sag'])
    else:
        lr_solver = 'saga'
        lr_l1_ratio = trial.suggest_float('lr_l1_ratio', 0.0, 1.0)
    
    params = {
        'lr_penalty': lr_penalty,
        'lr_tol': trial.suggest_float('lr_tol', 1e-5, 1e-2, log=True),
        'lr_C': trial.suggest_float('lr_C', 0.01, 10.0, log=True),
        'lr_w': trial.suggest_float('lr_w', 0.1, 2.0),
        
        'knn_neighbors': trial.suggest_int('knn_neighbors', 2, 50),
        'knn_weights': trial.suggest_categorical('knn_weights', ['uniform', 'distance']),
        'knn_p': trial.suggest_categorical('knn_p', [1, 2]),
        'knn_w': trial.suggest_float('knn_w', 0.1, 2.0),
        
        'svm_C': trial.suggest_float('svm_C', 0.01, 10.0, log=True),
        'svm_kernel': trial.suggest_categorical('svm_kernel', ['rbf', 'poly']),
        'svm_tol': trial.suggest_float('svm_tol', 1e-5, 1e-2, log=True),
        'svm_gamma': trial.suggest_categorical('svm_gamma', ['scale', 'auto']),
        'svm_w': trial.suggest_float('svm_w', 0.1, 2.0),
        
        'rf_estimators': trial.suggest_int('rf_estimators', 50, 500, step=50),
        'rf_criterion': trial.suggest_categorical('rf_criterion', ['gini', 'entropy']),
        'rf_max_depth': trial.suggest_int('rf_max_depth', 3, 50),
        'rf_min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
        'rf_min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
        'rf_max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
        'rf_w': trial.suggest_float('rf_w', 0.1, 2.0),
        
        'nb_smoothing': trial.suggest_float('nb_smoothing', 1e-10, 1e-5, log=True),
        'nb_w': trial.suggest_float('nb_w', 0.1, 2.0)
    }
    
    # Add solver params
    if lr_penalty == 'l1':
        params['lr_solver1'] = lr_solver
    elif lr_penalty == 'l2':
        params['lr_solver2'] = lr_solver
    
    if lr_l1_ratio is not None:
        params['lr_l1_ratio'] = lr_l1_ratio
    
    if params['svm_kernel'] == 'poly':
        params['svm_degree'] = trial.suggest_int('svm_degree', 2, 5)
    
    # Build and train model
    builder = models.ModelBuilder(random_state=random_state)
    model = builder.create_model_from_params(params)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_val, y_pred)
    
    return accuracy


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 70)
    print("LIVER DISEASE PREDICTION MODEL TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.data}")
    print(f"  Trials: {args.trials}")
    print(f"  Test size: {args.test_size}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {args.output}")
    print()
    
    # Set random seeds
    utils.set_random_seeds(args.seed)
    
    # Load data
    print("\n1. Loading data...")
    df = utils.load_data(args.data, index_col=0)
    utils.print_dataset_info(df)
    
    # Split features and target
    print("\n2. Splitting features and target...")
    X, y = utils.split_features_target(df, TARGET_COLUMN)
    
    # Identify feature types
    numeric_features, categorical_features = preprocessing.get_feature_types(
        X, exclude_columns=[]
    )
    print(f"  Numeric features: {numeric_features}")
    print(f"  Categorical features: {categorical_features}")
    
    # Preprocess data
    print("\n3. Preprocessing data...")
    preprocessor = preprocessing.DataPreprocessor(random_state=args.seed)
    X_transformed = preprocessor.fit_transform(X, numeric_features, categorical_features)
    
    # Encode target
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split into train, validation, and test
    print("\n4. Splitting into train/validation/test sets...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_transformed, y_encoded,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_encoded
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VALIDATION_SIZE,
        random_state=args.seed,
        stratify=y_temp
    )
    
    print(f"  Train: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    
    # Apply SMOTE
    print("\n5. Applying SMOTE...")
    X_train_bal, y_train_bal = utils.apply_smote(
        X_train, y_train,
        random_state=args.seed,
        k_neighbors=SMOTE_K_NEIGHBORS
    )
    
    # Optimize hyperparameters
    print("\n6. Optimizing hyperparameters with Optuna...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='liver_disease_classification',
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train_bal, y_train_bal, X_val, y_val, args.seed),
        n_trials=args.trials,
        show_progress_bar=True
    )
    
    print(f"\n✓ Optimization completed!")
    print(f"  Best validation accuracy: {study.best_value:.4f}")
    
    # Train final model
    print("\n7. Training final model with best parameters...")
    builder = models.ModelBuilder(random_state=args.seed)
    best_model = builder.create_model_from_params(study.best_params)
    best_model.fit(X_train_bal, y_train_bal)
    
    # Evaluate on test set
    print("\n8. Evaluating on test set...")
    results = builder.evaluate_model(
        best_model, X_test, y_test,
        target_names=le.classes_
    )
    models.print_evaluation_results(results)
    
    # Save model and preprocessor
    print(f"\n9. Saving model and preprocessor...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    model_path = Path(args.output) / "best_model.pkl"
    preprocessor_path = Path(args.output) / "preprocessor.pkl"
    label_encoder_path = Path(args.output) / "label_encoder.pkl"
    
    builder.save_model(best_model, str(model_path))
    builder.save_model(preprocessor, str(preprocessor_path))
    builder.save_model(le, str(label_encoder_path))
    
    # Save results
    results_path = Path(OUTPUT_DIR) / "training_results.json"
    utils.save_results({
        'best_accuracy': study.best_value,
        'best_params': study.best_params,
        'test_accuracy': results['accuracy'],
        'test_f1_score': results['f1_score']
    }, str(results_path))
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nModel saved to: {model_path}")
    print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Results saved to: {results_path}")
    

if __name__ == "__main__":
    main()
