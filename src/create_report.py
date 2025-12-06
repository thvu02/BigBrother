import numpy as np
import pickle
import sys

try:
    with open('results/results.pkl', 'rb') as f:
        results = pickle.load(f)
except FileNotFoundError:
    sys.exit(1)

baseline_reconstruction = results['baseline_reconstruction']
baseline_reident = results['baseline_reidentification']
all_dp_methods = results['all_dp_methods']
dp_reident = results['dp_reidentification']
utility_metrics = results['utility_metrics']

baseline_avg = np.mean([
    np.mean(list(baseline_reconstruction['income'].values())),
    np.mean(list(baseline_reconstruction['education'].values())),
    np.mean(list(baseline_reconstruction['occupation'].values()))
])

### Reconstruction Attack 

reident_acc_baseline = baseline_reident['acc']
reident_k_baseline = baseline_reident['k']

print('\n1.1 Average Reconstruction Attack Accuracy by Method:')
print('-'*80)
print(f'{"Method":<25} {"Income":<12} {"Education":<12} {"Occupation":<12} {"Average"}')
print('-'*80)

baseline_income_avg = np.mean(list(baseline_reconstruction['income'].values()))
baseline_edu_avg = np.mean(list(baseline_reconstruction['education'].values()))
baseline_occ_avg = np.mean(list(baseline_reconstruction['occupation'].values()))
print(f'{"Baseline (No Defense)":<25} {baseline_income_avg:>10.2%} {baseline_edu_avg:>10.2%} {baseline_occ_avg:>10.2%} {baseline_avg:>10.2%}')

for method_name, method_results in all_dp_methods.items():
    if method_results['income'] and method_results['education'] and method_results['occupation']:
        income_avg = np.mean(list(method_results['income'].values()))
        edu_avg = np.mean(list(method_results['education'].values()))
        occ_avg = np.mean(list(method_results['occupation'].values()))
        overall_avg = np.mean([income_avg, edu_avg, occ_avg])
        print(f'{method_name:<25} {income_avg:>10.2%} {edu_avg:>10.2%} {occ_avg:>10.2%} {overall_avg:>10.2%}')

### Reidentification Attack

print('\n2.1 Reidentification Results:')
print('-'*80)
print(f'{"Method":<25} {"Accuracy":<12} {"k-anonymity":<12} {"Acc Reduction":<15} {"k Improvement"}')
print('-'*80)
print(f'{"Baseline (No Defense)":<25} {reident_acc_baseline:>10.2%} {reident_k_baseline:>10.2f} {"-":>13s} {"-":>13s}')

for method_name in ['Laplace', 'Adaptive Budget', 'Multi-Layer']:
    if method_name in dp_reident:
        acc = dp_reident[method_name]['acc']
        k = dp_reident[method_name]['k']
        acc_reduction = reident_acc_baseline - acc
        k_improvement = k - reident_k_baseline
        print(f'{method_name + " DP":<25} {acc:>10.2%} {k:>10.2f} {acc_reduction:>13.2%} {k_improvement:>13.2f}')

### Utility assessment

print('\n3.1 Utility Comparison Summary:')
print('-'*80)
print(f'{"Method":<25} {"Overall Utility":<20} {"Assessment"}')
print('-'*80)
for method_name in ['Laplace', 'Adaptive Budget', 'Multi-Layer']:
    if method_name in utility_metrics:
        util = utility_metrics[method_name]['overall_utility']
        if util >= 0.80:
            assessment = 'EXCELLENT'
        elif util >= 0.65:
            assessment = 'GOOD'
        elif util >= 0.50:
            assessment = 'MODERATE'
        else:
            assessment = 'POOR'
        print(f'{method_name:<25} {util*100:>18.2f}% {assessment:>12s}')