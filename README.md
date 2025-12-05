# BigBrother: Privacy Attack & Defense Analysis

## Overview

BigBrother is a comprehensive privacy research project that demonstrates real-world privacy vulnerabilities in data publishing and evaluates differential privacy (DP) defense mechanisms. The project simulates realistic privacy attacks on census and advertising data, then implements and compares four different differential privacy techniques to protect against these attacks.

## Project Purpose

This project addresses the critical question: **Can publicly available data be used to infer private information about individuals?**

The answer is demonstrated through:
1. **Reidentification Attacks**: Linking anonymous ad profiles to census records using quasi-identifiers
2. **Reconstruction Attacks**: Inferring sensitive attributes (income, education, occupation) from behavioral data using machine learning

The project then evaluates how well differential privacy techniques can defend against these attacks while maintaining data utility.

## Key Features

- **Realistic Data Simulation**: Generates correlated advertising and census data with 80% demographic-to-interest correlation
- **Multiple Attack Vectors**: Implements both reidentification and ML-based reconstruction attacks
- **Four DP Defense Methods**:
  1. Original Laplace DP (baseline differential privacy)
  2. Adaptive Budget DP (sensitivity-based epsilon allocation)
  3. Multi-Layer DP (Gaussian noise + generalization + k-suppression)
  4. DP-SGD (differential privacy in neural network training)
- **Comprehensive Evaluation**: Measures privacy protection, attack success reduction, and data utility preservation
- **Rich Visualizations**: Generates detailed charts and heatmaps for analysis

## Project Structure

```
BigBrother/
  data/
    synthetic_census_data.csv         # Original census data
    enhanced_census_data.csv          # Census with 7-digit ZIP codes
    enhanced_ad_data.csv              # Correlated advertising data
  src/
    config.py                         # Configuration parameters
    utils.py                          # Core DP and attack implementations
    create_enhanced_dataset.py        # Dataset generation
    privacy_analysis.py               # Main analysis script
    create_report.py                  # Generate text reports
    create_graphics.py                # Generate visualizations
  results/
    results.pkl                       # Serialized results
    privacy_analysis_visualization.png
    privacy_analysis_key_findings.png
  README.md
```

## Methodology

### Phase 1: Data Generation
- **Census Data**: Enhanced with 7-digit ZIP codes and age groups
- **Ad Data**: Generated with 80% correlation to demographics
  - Income-based interests (e.g., luxury_travel for high income)
  - Education-based interests (e.g., graduate_schools for advanced degrees)
  - Occupation-based interests (e.g., programming for tech workers)
- **Quasi-Identifiers**: Age group, gender, ZIP code prefix

### Phase 2: Baseline Attacks (No Defense)
**Reidentification Attack**:
- Links ad user IDs to census person IDs using quasi-identifiers
- Measures k-anonymity (how many people share the same quasi-identifiers)
- Calculates reidentification accuracy

**Reconstruction Attacks**:
- Uses 4 ML models: Random Forest, Gradient Boosting, SVM, Neural Network
- Predicts sensitive attributes from ad interests:
  - **Income**: Classified into 4 categories (<50k, 50-100k, 100-150k, 150k+)
  - **Education**: 6 levels (from "Less than High School" to "Doctorate")
  - **Occupation**: 11 categories (e.g., Computer and Mathematical, Healthcare)

### Phase 3: Differential Privacy Defenses

#### Method 1: Original Laplace DP (epsilon=0.5)
- **Income**: Laplace noise (sensitivity = $50,000)
- **Categorical Attributes**: Randomized response
- **Quasi-Identifiers**: 3-digit ZIP generalization, age groups

#### Method 2: Adaptive Budget DP (epsilon=0.5)
- **Smart Budget Allocation**:
  - Income: 42.4% (most vulnerable)
  - Education: 32.8% (medium vulnerable)
  - Occupation: 24.8% (least vulnerable)
- Allocates more privacy budget to more sensitive attributes

#### Method 3: Multi-Layer DP (epsilon=0.5, delta=1e-5)
- **Gaussian Noise**: For (epsilon, delta)-differential privacy
- **Enhanced Generalization**: 2-digit ZIP codes, coarser age groups
- **k-Suppression**: Removes records with k-anonymity < 40
- **Trade-off**: Record suppression (~80-90% retention)

#### Method 4: DP-SGD (epsilon=0.5, delta=1e-5)
- **Gradient Clipping**: Norm clipping = 1.0
- **Noise Injection**: Adds calibrated noise to gradients
- **Neural Network**: MLPClassifier with (100, 50) hidden layers
- **Focus**: Protects reconstruction attacks only

### Phase 4: Utility Assessment
Measures data quality preservation using:
- **Mean Absolute Error (MAE)**: For numerical attributes (income)
- **Total Variation Distance (TVD)**: Distribution similarity
- **Jensen-Shannon Divergence (JSD)**: Information-theoretic distance
- **Distribution Accuracy**: Category proportion preservation

## Configuration

Key parameters in `config.py`:

```python
EPSILON = 0.5                 # Privacy budget
DELTA = 1e-5                  # Failure probability for (epsilon, delta)-DP
K_ANONYMITY_THRESHOLD = 40    # For Multi-Layer DP

ADAPTIVE_BUDGET = {
    'income': 0.424,          # 42.4% of total budget
    'education': 0.328,       # 32.8%
    'occupation': 0.248       # 24.8%
}

INCOME_THRESHOLDS = {
    'low': 50000,
    'medium': 100000,
    'high': 150000
}
```

## How to Run

### 1. Generate Enhanced Datasets
```bash
python src/create_enhanced_dataset.py
```
Creates:
- `data/enhanced_census_data.csv`
- `data/enhanced_ad_data.csv`

### 2. Run Privacy Analysis
```bash
python src/privacy_analysis.py
```
Performs all attacks and DP defenses, saves results to `results/results.pkl`

### 3. Generate Report
```bash
python src/create_report.py
```
Displays comprehensive text-based analysis

### 4. Create Visualizations
```bash
python src/create_graphics.py
```
Generates:
- `results/privacy_analysis_visualization.png` (comprehensive dashboard)
- `results/privacy_analysis_key_findings.png` (key findings)

## Key Findings

### Finding 1: Privacy Vulnerability Without Protection
- **Reidentification**: High success rate with quasi-identifiers
- **Reconstruction**: ML models significantly above random baseline
- **Implication**: Public data can reveal sensitive private information

### Finding 2: Adaptive Budget DP Provides Optimal Balance
- **Privacy Protection**: Significant reduction in attack accuracy
- **Data Utility**: Maintains 90%+ utility score
- **Model-Agnostic**: Effective across all 4 ML models (<1% variance)
- **Recommendation**: Best choice for production deployments

### Finding 3: DP-SGD Offers Strongest Reconstruction Protection
- **Attack Accuracy**: Reduces reconstruction below random baseline
- **Trade-off**: Very low data utility (~40%)
- **Use Case**: When maximum privacy is required regardless of utility

### Finding 4: Multi-Layer DP Maximizes Reidentification Protection
- **k-Anonymity**: Highest k-anonymity improvement
- **Quasi-Identifier Protection**: Enhanced generalization + suppression
- **Trade-off**: Record suppression (some data loss)

### Finding 5: Utility Preservation is Achievable
- **Adaptive Budget DP**: 90%+ utility score (EXCELLENT)
- **Distribution Preservation**: 89-95% for categorical attributes
- **Suitable For**: Aggregate statistics, distribution analysis
- **Not Suitable For**: Individual-level analysis

## Attack Success Metrics

### Baseline (No Defense)
- **Reidentification Accuracy**: Varies based on quasi-identifier uniqueness
- **Reconstruction Accuracy**: 40-70% depending on attribute and model
- **k-Anonymity**: Low (vulnerable to linkage attacks)

### With Differential Privacy
- **Reconstruction Reduction**: 15-60% depending on method
- **Reidentification Reduction**: 80-100% with proper generalization
- **k-Anonymity Improvement**: 5-50x increase with Multi-Layer DP

## Technologies Used

- **Python 3.x**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn (RandomForest, GradientBoosting, SVM, MLP)
- **Visualization**: matplotlib, seaborn
- **Privacy**: Custom DP implementations (Laplace, Gaussian, Randomized Response)

## Privacy Budget Allocation Rationale

The Adaptive Budget method allocates epsilon based on:
1. **Sensitivity Analysis**: How much information each attribute reveals
2. **Correlation with Other Attributes**: Income strongly correlates with occupation/education
3. **Reconstruction Vulnerability**: Empirical attack success rates

## Utility Thresholds

| Score | Assessment | Suitable For |
|-------|-----------|--------------|
| >=80% | EXCELLENT | Most analytical tasks |
| >=65% | GOOD | Distribution analysis, aggregations |
| >=50% | MODERATE | Limited utility, rough statistics only |
| <50% | POOR | Significantly degraded utility |

## Recommendations

### For Production Data Publishing
1. **Use Adaptive Budget DP** with epsilon=0.5
2. Apply quasi-identifier generalization (3-digit ZIP, age groups)
3. Monitor k-anonymity metrics
4. Validate utility meets requirements (aim for >=80%)

### For Maximum Privacy (High-Risk Data)
1. **Combine Multiple Techniques**: Multi-Layer DP + Adaptive Budget
2. Use smaller epsilon (epsilon=0.1-0.3)
3. Accept utility trade-offs
4. Consider k-suppression for rare combinations

### For Research/ML Applications
1. **Use DP-SGD** for training ML models on sensitive data
2. Consider federated learning for additional protection
3. Validate model performance degradation is acceptable

## References

This project demonstrates techniques from:
- Dwork, C., & Roth, A. (2014). The Algorithmic Foundations of Differential Privacy
- Abadi, M., et al. (2016). Deep Learning with Differential Privacy (DP-SGD)
- Sweeney, L. (2002). k-anonymity: A model for protecting privacy