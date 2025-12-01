import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
import pickle

import config
from utils import (
    classify_income,
    reidentification_attack,
    test_all_models,
    test_all_models_with_dp,
    apply_original_dp,
    apply_adaptive_dp,
    apply_multilayer_dp,
    DPNeuralNetwork,
    assess_utility
)

warnings.filterwarnings('ignore')
np.random.seed(config.RANDOM_SEED)
sns.set_style('whitegrid')

census_enhanced = pd.read_csv(config.ENHANCED_CENSUS_PATH)
ad_enhanced = pd.read_csv(config.ENHANCED_AD_PATH)

print(f'Census records: {len(census_enhanced):,}')
print(f'Ad records: {len(ad_enhanced):,}')
print(f'Unique users: {ad_enhanced["user_id"].nunique():,}')

# prepare feature matrix for ML
all_interests = set()
for interests_str in ad_enhanced['ad_interests']:
    all_interests.update(interests_str.split(','))
all_interests = sorted(list(all_interests))

user_interests = ad_enhanced.groupby('user_id')['ad_interests'].apply(lambda x: ','.join(x)).reset_index()

feature_matrix = []
for _, row in user_interests.iterrows():
    user_interest_set = set(row['ad_interests'].split(','))
    features = [1 if interest in user_interest_set else 0 for interest in all_interests]
    feature_matrix.append(features)

X = pd.DataFrame(feature_matrix, columns=all_interests, index=user_interests['user_id'])
print(f'Feature matrix: {X.shape} ({len(all_interests)} unique interests)')

### BASELINE ATTACKS (NO DEFENSES)

print('\n3.1 Reidentification Attack')
print('-'*80)

reident_baseline = reidentification_attack(ad_enhanced, census_enhanced)
reident_acc_baseline = reident_baseline['correct'].mean()
reident_k_baseline = reident_baseline['k_anonymity'].mean()

print(f'Accuracy: {reident_acc_baseline:.2%}')
print(f'Average k-anonymity: {reident_k_baseline:.2f}')
print(f'Interpretation: {reident_acc_baseline*100:.1f}% of users successfully reidentified')

print('\n3.2 Reconstruction Attacks (Multiple ML Models)')
print('-'*80)

baseline_reconstruction = {}

income_classes = census_enhanced['income'].apply(classify_income).nunique()
income_random_baseline = 1.0 / income_classes
print(f'\nINCOME ({income_classes} classes, random baseline = {income_random_baseline:.2%}):')
income_res = test_all_models(X, census_enhanced, 'income', classify_income)
baseline_reconstruction['income'] = income_res
for model, acc in income_res.items():
    print(f'  {model:20s}: {acc:.2%} (+{(acc-income_random_baseline)*100:.1f}% above random)')

edu_classes = census_enhanced['education'].nunique()
edu_random_baseline = 1.0 / edu_classes
print(f'\nEDUCATION ({edu_classes} classes, random baseline = {edu_random_baseline:.2%}):')
edu_res = test_all_models(X, census_enhanced, 'education')
baseline_reconstruction['education'] = edu_res
for model, acc in edu_res.items():
    print(f'  {model:20s}: {acc:.2%} (+{(acc-edu_random_baseline)*100:.1f}% above random)')

occ_classes = census_enhanced['occupation'].nunique()
occ_random_baseline = 1.0 / occ_classes
print(f'\nOCCUPATION ({occ_classes} classes, random baseline = {occ_random_baseline:.2%}):')
occ_res = test_all_models(X, census_enhanced, 'occupation')
baseline_reconstruction['occupation'] = occ_res
for model, acc in occ_res.items():
    print(f'  {model:20s}: {acc:.2%} (+{(acc-occ_random_baseline)*100:.1f}% above random)')

baseline_avg = np.mean([
    np.mean(list(baseline_reconstruction['income'].values())),
    np.mean(list(baseline_reconstruction['education'].values())),
    np.mean(list(baseline_reconstruction['occupation'].values()))
])
print(f'\nAverage baseline accuracy: {baseline_avg:.2%}')

### DIFFERENTIAL PRIVACY METHODS

all_dp_methods = {}

print(f'\n4.1 Method 1: Original Laplace + Randomized Response (epsilon={config.EPSILON})')
print('-'*80)

census_original_dp = apply_original_dp(census_enhanced)
print(f'p_true (randomized response): {np.exp(config.EPSILON)/(np.exp(config.EPSILON)+1):.2%}')
print('Applied: Laplace noise to income, Randomized Response to categorical')

print('\nReidentification attack on Original Laplace DP:')
reident_original_dp = reidentification_attack(ad_enhanced, census_original_dp)
reident_acc_original_dp = reident_original_dp['correct'].mean()
reident_k_original_dp = reident_original_dp['k_anonymity'].mean()
print(f'  Accuracy: {reident_acc_original_dp:.2%} (baseline: {reident_acc_baseline:.2%}, reduction: {reident_acc_baseline-reident_acc_original_dp:.2%})')
print(f'  k-anonymity: {reident_k_original_dp:.2f} (baseline: {reident_k_baseline:.2f}, improvement: {reident_k_original_dp-reident_k_baseline:.2f})')

print(f'\n4.2 Method 2: Adaptive Budget Allocation (epsilon={config.EPSILON})')
print('-'*80)

census_adaptive_dp = apply_adaptive_dp(census_enhanced)
print('Budget allocation:')
print(f'  Income: {config.ADAPTIVE_BUDGET["income"]*100:.1f}% (eps={config.EPSILON*config.ADAPTIVE_BUDGET["income"]:.3f}) - Most vulnerable')
print(f'  Education: {config.ADAPTIVE_BUDGET["education"]*100:.1f}% (eps={config.EPSILON*config.ADAPTIVE_BUDGET["education"]:.3f}) - Medium vulnerable')
print(f'  Occupation: {config.ADAPTIVE_BUDGET["occupation"]*100:.1f}% (eps={config.EPSILON*config.ADAPTIVE_BUDGET["occupation"]:.3f}) - Least vulnerable')

print('\nReidentification attack on Adaptive Budget DP:')
reident_adaptive_dp = reidentification_attack(ad_enhanced, census_adaptive_dp)
reident_acc_adaptive_dp = reident_adaptive_dp['correct'].mean()
reident_k_adaptive_dp = reident_adaptive_dp['k_anonymity'].mean()
print(f'  Accuracy: {reident_acc_adaptive_dp:.2%} (baseline: {reident_acc_baseline:.2%}, reduction: {reident_acc_baseline-reident_acc_adaptive_dp:.2%})')
print(f'  k-anonymity: {reident_k_adaptive_dp:.2f} (baseline: {reident_k_baseline:.2f}, improvement: {reident_k_adaptive_dp-reident_k_baseline:.2f})')

print(f'\n4.3 Method 3: Multi-Layer DP (epsilon={config.EPSILON}, delta={config.DELTA})')
print('-'*80)

census_multilayer_dp = apply_multilayer_dp(census_enhanced)
print(f'Records retained: {len(census_multilayer_dp)}/{len(census_enhanced)} ({len(census_multilayer_dp)/len(census_enhanced)*100:.1f}%)')
print('Applied: Gaussian noise + Enhanced generalization + k-suppression')

print('\nReidentification attack on Multi-Layer DP:')
reident_multilayer_dp = reidentification_attack(ad_enhanced, census_multilayer_dp)
reident_acc_multilayer_dp = reident_multilayer_dp['correct'].mean()
reident_k_multilayer_dp = reident_multilayer_dp['k_anonymity'].mean()
print(f'  Accuracy: {reident_acc_multilayer_dp:.2%} (baseline: {reident_acc_baseline:.2%}, reduction: {reident_acc_baseline-reident_acc_multilayer_dp:.2%})')
print(f'  k-anonymity: {reident_k_multilayer_dp:.2f} (baseline: {reident_k_baseline:.2f}, improvement: {reident_k_multilayer_dp-reident_k_baseline:.2f})')

print(f'\n4.4 Method 4: DP-SGD for Neural Networks (epsilon={config.EPSILON}, delta={config.DELTA})')
print('-'*80)

print('Training DP-SGD models for each attribute...')
dp_sgd_results = {}

for attr in ['income', 'education', 'occupation']:
    census_indexed = census_enhanced.set_index('person_id')
    if attr == 'income':
        y = census_indexed.loc[X.index]['income'].apply(classify_income)
    else:
        y = census_indexed.loc[X.index][attr]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.ML_TEST_SIZE, random_state=config.ML_RANDOM_STATE, stratify=y
    )

    dp_sgd = DPNeuralNetwork()
    dp_sgd.fit(X_train, y_train)
    y_pred = dp_sgd.predict(X_test)

    dp_sgd_results[attr] = accuracy_score(y_test, y_pred)

print('DP-SGD Results:')
for attr, acc in dp_sgd_results.items():
    baseline_nn = baseline_reconstruction[attr]['Neural Network']
    print(f'  {attr.capitalize():12s}: {acc:.2%} (baseline: {baseline_nn:.2%}, reduction: {baseline_nn-acc:.2%})')

### TESTING ALL DP METHODS WITH ALL ML MODELS

print('\n5.1 Original Laplace DP - All Models:')
print('-'*80)
original_dp_results = test_all_models_with_dp(X, census_original_dp, baseline_reconstruction)
all_dp_methods['Original Laplace'] = original_dp_results

for attr in ['income', 'education', 'occupation']:
    if original_dp_results[attr]:
        print(f'\n{attr.upper()}:')
        for model, acc in original_dp_results[attr].items():
            baseline = baseline_reconstruction[attr][model]
            print(f'  {model:20s}: {acc:.2%} (baseline: {baseline:.2%}, reduction: {baseline-acc:.2%})')

print('\n5.2 Adaptive Budget DP - All Models:')
print('-'*80)
adaptive_dp_results = test_all_models_with_dp(X, census_adaptive_dp, baseline_reconstruction)
all_dp_methods['Adaptive Budget'] = adaptive_dp_results

for attr in ['income', 'education', 'occupation']:
    if adaptive_dp_results[attr]:
        print(f'\n{attr.upper()}:')
        for model, acc in adaptive_dp_results[attr].items():
            baseline = baseline_reconstruction[attr][model]
            print(f'  {model:20s}: {acc:.2%} (baseline: {baseline:.2%}, reduction: {baseline-acc:.2%})')

print('\n5.3 Multi-Layer DP - All Models:')
print('-'*80)
multilayer_dp_results = test_all_models_with_dp(X, census_multilayer_dp, baseline_reconstruction)
all_dp_methods['Multi-Layer'] = multilayer_dp_results

for attr in ['income', 'education', 'occupation']:
    if multilayer_dp_results[attr]:
        print(f'\n{attr.upper()}:')
        for model, acc in multilayer_dp_results[attr].items():
            baseline = baseline_reconstruction[attr][model]
            print(f'  {model:20s}: {acc:.2%} (baseline: {baseline:.2%}, reduction: {baseline-acc:.2%})')

### UTILITY ASSESSMENT

print('\n6.1 Utility Assessment for Adaptive Budget DP:')
print('-'*80)
utility_metrics = assess_utility(census_enhanced, census_adaptive_dp)

print(f'Income MAE: ${utility_metrics["income_mae"]:,.2f}')
print(f'Education TVD: {utility_metrics["education_tvd"]:.4f} (similarity: {(1-utility_metrics["education_tvd"])*100:.2f}%)')
print(f'Occupation TVD: {utility_metrics["occupation_tvd"]:.4f} (similarity: {(1-utility_metrics["occupation_tvd"])*100:.2f}%)')
print(f'Education JSD: {utility_metrics["education_jsd"]:.4f} (quality: {(1-utility_metrics["education_jsd"])*100:.2f}%)')
print(f'Occupation JSD: {utility_metrics["occupation_jsd"]:.4f} (quality: {(1-utility_metrics["occupation_jsd"])*100:.2f}%)')
print(f'Education Distribution: {utility_metrics["education_dist_acc"]*100:.2f}% preserved')
print(f'Occupation Distribution: {utility_metrics["occupation_dist_acc"]*100:.2f}% preserved')
print(f'\nOVERALL UTILITY SCORE: {utility_metrics["overall_utility"]*100:.2f}%')

if utility_metrics['overall_utility'] >= config.UTILITY_THRESHOLDS['excellent']:
    assessment = 'EXCELLENT - Dataset is highly usable'
elif utility_metrics['overall_utility'] >= config.UTILITY_THRESHOLDS['good']:
    assessment = 'GOOD - Dataset is usable for many tasks'
elif utility_metrics['overall_utility'] >= config.UTILITY_THRESHOLDS['moderate']:
    assessment = 'MODERATE - Dataset has limited utility'
else:
    assessment = 'POOR - Dataset utility is significantly degraded'

print(f'Assessment: {assessment}')

### save results to .pkl file

results = {
    'baseline_reconstruction': baseline_reconstruction,
    'baseline_reidentification': {
        'acc': reident_acc_baseline,
        'k': reident_k_baseline
    },
    'all_dp_methods': all_dp_methods,
    'dp_reidentification': {
        'Original Laplace': {
            'acc': reident_acc_original_dp,
            'k': reident_k_original_dp
        },
        'Adaptive Budget': {
            'acc': reident_acc_adaptive_dp,
            'k': reident_k_adaptive_dp
        },
        'Multi-Layer': {
            'acc': reident_acc_multilayer_dp,
            'k': reident_k_multilayer_dp
        }
    },
    'dp_sgd_results': dp_sgd_results,
    'utility_metrics': utility_metrics
}

with open('results/results.pkl', 'wb') as f:
    pickle.dump(results, f)
