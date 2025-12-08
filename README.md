# Reidentification and Reconstruction Attacks on Sensitive Datasets

## Table of Contents
- [What This Project Does](#what-this-project-does)
- [Privacy Mechanisms Tested](#privacy-mechanisms-tested)
- [Attack Scenarios](#attack-scenarios)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Running the Analysis](#running-the-analysis)

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
1. **Laplace DP**: Classic DP with Laplace noise + randomized response
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

### Method 1: Laplace DP
**What it does**: Adds calibrated noise to data values

**Protection mechanisms**:
- Income: Laplace noise (scale = $50k/ε)
- Education/Occupation: Randomized response (p_true ≈ 62% for ε=0.5)
- Quasi-identifiers: Generalization (ZIP 7→3 digits, exact age→groups)

---

### Method 2: Adaptive Budget DP (RECOMMENDED)
**What it does**: Allocates privacy budget based on attribute vulnerability

**Key innovation**:
- Income gets 42.4% of budget (most vulnerable to attacks)
- Education gets 32.8% of budget
- Occupation gets 24.8% of budget (least vulnerable)

**Why it's better**: Maximizes privacy where needed most while preserving utility elsewhere

---

### Method 3: Multi-Layer DP
**What it does**: Combines multiple defense layers

**Protection mechanisms**:
- Gaussian noise (provides (ε,δ)-DP)
- Enhanced generalization (ZIP 7→2 digits, age 6→3 groups)
- k-suppression (removes records with k<40)

**Results**: Strongest reidentification protection (k-anonymity 70-100)

---

### DP-SGD (Model Training Protection)
**What it does**: Protects neural network training with differential privacy

**Implementation**: Uses **Opacus** (Facebook's production DP library)

**Mechanisms**:
- Clips gradients during training
- Adds Gaussian noise to gradients
- Tracks privacy budget across epochs
- Stops when budget exceeded

**What it measures**: How accurate are models when trained with DP constraints?

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

**Defense**: Add noise to attributes (Laplace/Adaptive/Multi-Layer)

---

## Project Structure

```
BigBrother/
├── data/
│   ├── enhanced_census_data.csv    # Generated census
│   ├── enhanced_ad_data.csv        # Generated ads
│   └── synthetic_census_data.csv   # Original fake census data
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
├── README.md                       # This file
```

---

## Installation & Setup

### Prerequisites
- Python 3.8+

### Install Dependencies

```bash
pip install pandas numpy scikit-learn scipy seaborn matplotlib torch opacus
```

---

## Running the Analysis

### Quick Start

```bash
# 1. Generate data (if needed)
python src/create_enhanced_dataset.py

# 2. Run full analysis
python src/privacy_analysis.py

# 3. Generate report
python src/create_report.py

# 4. Create visualizations
python src/create_graphics.py
```
