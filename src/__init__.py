"""
Liver Disease Analysis - Utility Modules
=========================================

This package contains utility modules for the liver disease analysis project.

Modules:
--------
- preprocessing: Data preprocessing and transformation utilities
- models: Model building and evaluation utilities  
- visualization: Plotting and visualization functions
- utils: General utility functions

Usage:
------
    from src import preprocessing, models, visualization, utils
    
    # Load and preprocess data
    df = utils.load_data('data/HepatitisCdata.csv')
    preprocessor = preprocessing.DataPreprocessor()
    
    # Build and train model
    builder = models.ModelBuilder()
    
    # Visualize results
    visualization.plot_confusion_matrix(y_true, y_pred)
"""

__version__ = '2.0.0'
__author__ = 'Felipe Breseghello'

from . import preprocessing
from . import models
from . import visualization
from . import utils

__all__ = [
    'preprocessing',
    'models',
    'visualization',
    'utils'
]
