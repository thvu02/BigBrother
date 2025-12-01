RANDOM_SEED = 42
ENHANCED_CENSUS_PATH = 'data/enhanced_census_data.csv'
ENHANCED_AD_PATH = 'data/enhanced_ad_data.csv'

RESULTS_OUTPUT_PATH = 'results/'
SAVE_INTERMEDIATE_RESULTS = True
VERBOSE = True

### Differential privacy variables

EPSILON = 0.5  # Privacy budget
DELTA = 1e-5   # Failure probability for (ε,δ)-DP

# Income sensitivity for Laplace mechanism
INCOME_SENSITIVITY = 50000
INCOME_SENSITIVITY_GAUSSIAN = 100000

# k-anonymity threshold for Multi-Layer DP
K_ANONYMITY_THRESHOLD = 40

ADAPTIVE_BUDGET = {
    'income': 0.424,      # 42.4% - Most vulnerable
    'education': 0.328,   # 32.8% - Medium vulnerable
    'occupation': 0.248   # 24.8% - Least vulnerable
}

DPSGD_CLIP_NORM = 1.0
DPSGD_N_EPOCHS = 100
DPSGD_BATCH_SIZE = 32

### ML model variables

ML_TEST_SIZE = 0.3
ML_RANDOM_STATE = 42

RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10

GB_N_ESTIMATORS = 100
GB_MAX_DEPTH = 5

NN_HIDDEN_LAYERS = (100, 50)
NN_MAX_ITER = 500

### income classification variables

INCOME_THRESHOLDS = {
    'low': 50000,
    'medium': 100000,
    'high': 150000
}

INCOME_LABELS = {
    'low': '<50k',
    'medium': '50-100k',
    'high': '100-150k',
    'very_high': '150k+'
}

### quasi-identifier variables

ZIP_GENERALIZATION = {
    'original': 3,    # 3-digit ZIP (e.g., "900XX")
    'multilayer': 2   # 2-digit ZIP (e.g., "90XXX")
}

AGE_GROUPS_STANDARD = {
    '18-24': (18, 24),
    '25-34': (25, 34),
    '35-44': (35, 44),
    '45-54': (45, 54),
    '55-64': (55, 64),
    '65+': (65, 100)
}

AGE_GROUPS_COARSE = {
    '18-34': (18, 34),
    '35-54': (35, 54),
    '55+': (55, 100)
}

### utility assessment variables

UTILITY_THRESHOLDS = {
    'excellent': 0.80,
    'good': 0.65,
    'moderate': 0.50
}
