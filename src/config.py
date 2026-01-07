# Configuration file for Liver Disease Analysis

# Random seed for reproducibility
RANDOM_STATE = 42

# Data paths
DATA_DIR = "data"
HEPATITIS_DATA = "data/HepatitisCdata.csv"
HEART_DATA = "data/heart.csv"
STROKE_DATA = "data/healthcare-dataset-stroke-data.csv"

# Model paths
MODEL_DIR = "models"
BEST_MODEL_PATH = "models/best_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

# Output paths
OUTPUT_DIR = "outputs"
RESULTS_PATH = "outputs/results.json"
PLOTS_DIR = "outputs/plots"

# Model training parameters
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
SMOTE_K_NEIGHBORS = 5

# Optuna parameters
N_TRIALS = 100
OPTUNA_TIMEOUT = 3600  # 1 hour in seconds
OPTUNA_N_JOBS = -1  # Use all CPU cores

# Feature names (Hepatitis C dataset)
NUMERIC_FEATURES = [
    'Age', 'ALB', 'ALP', 'ALT', 'AST', 
    'BIL', 'CHE', 'CHOL', 'CREA', 'GGT', 'PROT'
]

CATEGORICAL_FEATURES = ['Sex']

TARGET_COLUMN = 'Category'

# Target labels
TARGET_LABELS = {
    '0=Blood Donor': 'Blood Donor',
    '0s=suspect Blood Donor': 'Suspect Donor',
    '1=Hepatitis': 'Hepatitis',
    '2=Fibrosis': 'Fibrosis',
    '3=Cirrhosis': 'Cirrhosis'
}

# Visualization settings
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
FIGURE_DPI = 300
COLOR_PALETTE = 'husl'
