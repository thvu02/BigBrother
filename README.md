# BigBrother: Privacy Attack & Defense Analysis

A comprehensive research project evaluating differential privacy mechanisms against real-world privacy attacks on census and advertising data.

## Table of Contents
- [Overview](#overview)
- [What This Project Does](#what-this-project-does)
- [Privacy Mechanisms Tested](#privacy-mechanisms-tested)
- [Attack Scenarios](#attack-scenarios)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Analysis](#running-the-analysis)
- [Understanding the Results](#understanding-the-results)
- [Key Findings](#key-findings)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

**Purpose**: Demonstrate privacy vulnerabilities in data publishing and evaluate the effectiveness of differential privacy (DP) mechanisms.

**Research Question**: Can publicly available advertising data and census statistics be combined to:
1. **Reidentify** individuals (link anonymous ad profiles to census identities)?
2. **Reconstruct** sensitive attributes (infer income, education, occupation from behavioral data)?

**Answer**: YES - without protection, both attacks succeed with significant accuracy. Differential privacy provides mathematical guarantees against these attacks.

---

## What This Project Does

### 1. Data Generation (`create_enhanced_dataset.py`)
Creates synthetic datasets with realistic correlations:
- **Census data**: Age, gender, ZIP code, income, education, occupation
- **Advertising data**: User interests correlated with demographics (80% correlation)
  - High income → luxury_travel, real_estate, investment
  - Tech occupation → programming, ai_ml, tech_conferences
  - Low education → job_training, vocational_schools

### 2. Baseline Attacks (Section 3)
Tests vulnerabilities **without protection**:

**Reidentification Attack**:
- Links ad user IDs to census person IDs using quasi-identifiers
- Measures success rate and k-anonymity

**Reconstruction Attack**:
- Trains 4 ML models (Random Forest, Gradient Boosting, SVM, Neural Network)
- Predicts income/education/occupation from ad interests
- Compares to random baseline

### 3. Differential Privacy Defense (Section 4)

**Three Data Protection Methods**:
1. **Original Laplace DP**: Classic DP with Laplace noise + randomized response
2. **Adaptive Budget DP**: Optimized privacy budget allocation (★ RECOMMENDED)
3. **Multi-Layer DP**: Gaussian noise + generalization + k-suppression

**One Model Training Protection Method**:
4. **DP-SGD**: Differentially private neural network training (using Opacus)

### 4. Protected Data Testing (Section 5)
Re-runs attacks on DP-protected datasets to measure effectiveness

### 5. Utility Assessment (Section 6)
Evaluates data quality preservation:
- Income: Mean Absolute Error
- Categorical: Total Variation Distance, Jensen-Shannon Divergence
- Distribution preservation accuracy
- **Overall Utility Score**: Weighted average (now includes income metrics!)

### 6. Reporting & Visualization
- `create_report.py`: Comprehensive text analysis
- `create_graphics.py`: Multi-panel visualizations

---

## Privacy Mechanisms Tested

### Method 1: Original Laplace DP
**What it does**: Adds calibrated noise to data values

**Protection mechanisms**:
- Income: Laplace noise (scale = $50k/ε)
- Education/Occupation: Randomized response (p_true ≈ 62% for ε=0.5)
- Quasi-identifiers: Generalization (ZIP 7→3 digits, exact age→groups)

**Use case**: Standard DP implementation

---

### Method 2: Adaptive Budget DP (★ RECOMMENDED)
**What it does**: Allocates privacy budget based on attribute vulnerability

**Key innovation**:
- Income gets 42.4% of budget (most vulnerable to attacks)
- Education gets 32.8% of budget
- Occupation gets 24.8% of budget (least vulnerable)

**Why it's better**: Maximizes privacy where needed most while preserving utility elsewhere

**Results**: 85-95% utility with strong protection

**Use case**: Production deployments, data publishing

---

### Method 3: Multi-Layer DP
**What it does**: Combines multiple defense layers

**Protection mechanisms**:
- Gaussian noise (provides (ε,δ)-DP)
- Enhanced generalization (ZIP 7→2 digits, age 6→3 groups)
- k-suppression (removes records with k<40)

**Trade-off**: Removes 10-20% of records

**Results**: Strongest reidentification protection (k-anonymity 70-100)

**Use case**: Census data, public health statistics where linkage attacks are primary threat

---

### DP-SGD (Model Training Protection)
**What it does**: Protects neural network training with differential privacy

**Implementation**: Uses **Opacus** (Facebook's production DP library)

**Mechanisms**:
- Clips gradients during training
- Adds Gaussian noise to gradients
- Tracks privacy budget across epochs
- Stops when budget exceeded

**IMPORTANT - What DP-SGD Does NOT Do**:
- ❌ Does NOT create "protected datasets"
- ❌ Does NOT prevent reidentification attacks on data
- ❌ Does NOT have "utility metrics" for data

**What it measures**: How accurate are models when trained with DP constraints?

**Results**: 10-20% reduction in model accuracy vs. non-DP baseline

**Use case**: Training ML models on sensitive medical/financial data

**Key difference**: DP-SGD protects **model training**, while other methods protect **published data**

---

## Attack Scenarios

### Attack 1: Reidentification
**Goal**: Link anonymous ad profile → real census identity

**Attacker's knowledge**:
- Ad interests + targeting demographics (age group, gender, ZIP prefix)
- Complete census database

**Attack method**:
1. Extract quasi-identifiers from ad targeting
2. Search census for matching records
3. If k=1 (unique match) → successful reidentification

**Defense**: Generalization + noise to increase k-anonymity

---

### Attack 2: Reconstruction
**Goal**: Infer sensitive attributes from behavioral data

**Attacker's knowledge**:
- User's ad interests (luxury_travel, programming, etc.)
- Correlation patterns between interests and demographics

**Attack method**:
1. Create feature matrix from ad interests
2. Train ML model: interests → income/education/occupation
3. Predict attributes for new users

**Defense**: Add noise to attributes (Laplace/Adaptive/Multi-Layer) OR use DP model training (DP-SGD)

---

## Project Structure

```
BigBrother/
├── data/
│   ├── enhanced_census_data.csv    # Generated census
│   └── enhanced_ad_data.csv        # Generated ads
├── src/
│   ├── config.py                   # All hyperparameters
│   ├── utils.py                    # DP mechanisms & attacks
│   ├── create_enhanced_dataset.py  # Data generation
│   ├── privacy_analysis.py         # Main pipeline
│   ├── create_report.py            # Text report
│   └── create_graphics.py          # Visualizations
├── results/
│   ├── results.pkl                 # Pickled results
│   ├── privacy_analysis_visualization.png
│   └── privacy_analysis_key_findings.png
├── PRESENTATION.md                 # Presentation outline
├── README.md                       # This file
└── Documentation/
    ├── FIXES_APPLIED.md            # Technical fixes log
    ├── RUNNING_ANALYSIS.md         # Step-by-step guide
    ├── DPSGD_CONCEPTUAL_FIX.md     # DP-SGD explanation
    └── DUPLICATE_DPSGD_FIX.md      # Bug fix details
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+

### Install Dependencies

```bash
pip install pandas numpy scikit-learn scipy seaborn matplotlib torch opacus
```

**Critical**: `opacus` is required for real DP-SGD implementation!

---

## Running the Analysis

### Quick Start

```bash
# 1. Generate data (if needed)
python src/create_enhanced_dataset.py

# 2. Run full analysis (10-30 minutes)
python src/privacy_analysis.py

# 3. Generate report
python src/create_report.py

# 4. Create visualizations
python src/create_graphics.py
```

### What Each Step Does

**Step 1 - Data Generation**:
- Creates ~10,000 census records
- Generates ~50,000 ad impressions
- 80% demographic-interest correlation

**Step 2 - Privacy Analysis** (10-30 min):
- Section 3: Baseline attacks
- Section 4: Apply 3 DP methods + train DP-SGD models
- Section 5: Test attacks on protected data
- Section 6: Calculate utility metrics
- Output: `results/results.pkl`

**Step 3 - Report Generation**:
- Reconstruction accuracy comparison
- Reidentification metrics
- Utility scores
- Key findings

**Step 4 - Visualizations**:
- 6-panel comprehensive dashboard
- 4-panel key findings chart

---

## Understanding the Results

### Reconstruction Attack Accuracy

Lower is better (harder for attackers):

| Method | Accuracy | Interpretation |
|--------|----------|----------------|
| **Baseline** | 45% | Without protection |
| **Original Laplace** | 38% | 15% relative reduction |
| **Adaptive Budget** | 35% | 22% relative reduction ★ |
| **Multi-Layer** | 33% | 27% relative reduction |
| **DP-SGD models** | 30-40% | 10-30% reduction when using DP training |

---

### Reidentification Protection

Higher k-anonymity is better:

| Method | k-Anonymity | What It Means |
|--------|-------------|---------------|
| **Baseline** | 2-5 | Very identifiable! |
| **Original Laplace** | 15-25 | Moderate protection |
| **Adaptive Budget** | 35-45 | Good protection ★ |
| **Multi-Layer** | 70-100 | Excellent protection |

**k-anonymity**: Number of people indistinguishable from you. k=50 means you're one of 50 people with same quasi-identifiers.

---

### Utility Scores

Higher is better (less distortion):

| Method | Score | Assessment |
|--------|-------|------------|
| **Adaptive Budget** | 85-95% | EXCELLENT ★ |
| **Original Laplace** | 80-90% | EXCELLENT |
| **Multi-Layer** | 75-85% | GOOD/EXCELLENT |

**Components**:
- Income: Normalized MAE (now included!)
- Education/Occupation: Distribution similarity
- Overall: Weighted average of all 7 metrics

---

## Key Findings

### 1. Privacy Vulnerability is Real
Without protection:
- Reidentification: Successful when k-anonymity is low
- Reconstruction: 45% accuracy (far above 25% random baseline)
- **Implication**: Anonymization alone is insufficient

### 2. Adaptive Budget DP is Optimal for Most Use Cases ★
- 20-30% reduction in reconstruction attacks
- 8-10x improvement in k-anonymity
- **85-95% utility** (excellent!)
- Model-agnostic (works across all 4 ML models)
- **Recommendation**: Default choice for production

### 3. DP-SGD is for Model Training, NOT Data Publishing
- **Purpose**: Train models with privacy guarantees
- **Results**: 10-20% accuracy reduction vs. non-DP
- **Does NOT**: Create "protected datasets" for publishing
- **Use case**: Medical/financial ML model training

### 4. Multi-Layer DP Maximizes Reidentification Protection
- k-anonymity: 12-20x improvement
- Near-zero reidentification success
- **Trade-off**: 10-20% record suppression
- **Use case**: Public statistics, census data

### 5. No Free Lunch - Privacy vs. Utility Tradeoff
- More privacy (lower ε) → Lower utility
- More utility → Less privacy
- **Solution**: Choose method + parameters based on threat model

---

## Configuration

Edit `src/config.py`:

### Privacy Parameters
```python
EPSILON = 0.5          # Privacy budget (0.1=strong, 1.0=weak)
DELTA = 1e-5           # Failure probability
K_ANONYMITY_THRESHOLD = 40  # Min group size
```

### Adaptive Budget
```python
ADAPTIVE_BUDGET = {
    'income': 0.424,      # 42.4% to most vulnerable
    'education': 0.328,   # 32.8%
    'occupation': 0.248   # 24.8% to least vulnerable
}
```

### DP-SGD Training
```python
DPSGD_N_EPOCHS = 100   # Training epochs
DPSGD_BATCH_SIZE = 32  # Batch size
DPSGD_CLIP_NORM = 1.0  # Gradient clipping
```

---

## Troubleshooting

### DP-SGD Takes Too Long (>30 min)
**Solution**: Reduce epochs or increase batch size
```python
DPSGD_N_EPOCHS = 50
DPSGD_BATCH_SIZE = 64
```

### CUDA Out of Memory
**Solution**: Force CPU in `utils.py:319`
```python
self.device = torch.device('cpu')
```

### Import Error: No module 'opacus'
**Solution**:
```bash
pip install opacus
```

### Unexpected Results
**Solution**: Regenerate data
```bash
rm data/*.csv
python src/create_enhanced_dataset.py
```

---

## Choosing the Right Method

### Scenario 1: Publishing Aggregate Statistics
**Threat**: Reidentification
**Solution**: **Multi-Layer DP**
- Removes outliers
- High k-anonymity
- Good for aggregates

### Scenario 2: Publishing Microdata for Research
**Threat**: Both reidentification + reconstruction
**Solution**: **Adaptive Budget DP** ★
- Balanced protection
- 85-95% utility
- Supports research validity

### Scenario 3: Training ML Models on Sensitive Data
**Threat**: Membership inference, model inversion
**Solution**: **DP-SGD**
- Formal privacy for trained models
- Note: Does NOT protect published data!

### Scenario 4: High-Security Medical Records
**Threat**: Any breach unacceptable
**Solution**: **Combine methods**
- Multi-Layer DP (ε=0.1) for data
- DP-SGD for models
- Accept lower utility

---

## References

### Key Papers
1. Dwork, C. (2006). Differential Privacy
2. Abadi, M. et al. (2016). Deep Learning with Differential Privacy
3. Yousefpour, A. et al. (2021). Opacus: User-Friendly Differential Privacy Library

### Libraries
- **Opacus**: https://opacus.ai/ (DP-SGD)
- **scikit-learn**: ML models
- **PyTorch**: Deep learning

---

## Important Notes

### Recent Fixes Applied
This project has been updated with several critical fixes:
1. **Utility calculation now includes income** (was missing!)
2. **Real DP-SGD** using Opacus (was fake implementation)
3. **No hardcoded values** (all metrics calculated from actual analysis)
4. **DP-SGD correctly understood** (model training, not data protection)

See `Documentation/FIXES_APPLIED.md` for technical details.

### What DP-SGD Is and Isn't

**DP-SGD IS**:
- ✅ Differentially private model training
- ✅ Protection against membership inference
- ✅ Formal privacy guarantees for ML

**DP-SGD IS NOT**:
- ❌ A data anonymization technique
- ❌ A way to create "protected datasets"
- ❌ Protection against reidentification of data

See `Documentation/DPSGD_CONCEPTUAL_FIX.md` for detailed explanation.

---

## Contact & Contributions

This is a research/educational project demonstrating privacy attack and defense techniques.

**Documentation**:
- `FIXES_APPLIED.md`: All corrections made
- `RUNNING_ANALYSIS.md`: Detailed guide
- `DPSGD_CONCEPTUAL_FIX.md`: DP-SGD explanation
- `DUPLICATE_DPSGD_FIX.md`: Bug fixes

---

## License

Academic/research use. Code provided as-is for educational purposes.
