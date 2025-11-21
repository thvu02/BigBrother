# Privacy Attack Project

This project demonstrates privacy attacks on synthetic datasets, including reidentification and reconstruction attacks.

## Project Structure

```
Project/
├── src/                          # Python source scripts
│   ├── reidentification_attack.py    # Reidentification attack implementation
├── notebooks/                    # Jupyter notebooks
│   └── reconstruction_attack.ipynb   # Interactive notebook with visualizations
├── data/                         # Input data files
│   ├── synthetic_census_data.csv
│   └── synthetic_facebook_ad_data.csv
├── output/                       # Generated output files
│   ├── reid_matches.csv         # Reidentification matches (from reidentification attack)
│   └── reconstruction_predictions.csv  # Predicted attributes (from reconstruction attack)
└── README.md                     # This file
```

## Usage

### 1. Reidentification Attack

Run the reidentification attack script to link ad records to census records:

```bash
cd src
python reidentification_attack.py
```

This will:
- Load census and ad datasets from `../data/`
- Perform record linkage using quasi-identifiers (age group, gender, zip prefix)
- Generate `reid_matches.csv` in `../output/`



### 2. Reconstruction Attack

```bash
cd notebooks
jupyter notebook reconstruction_attack.ipynb
```

The notebook provides:
- Interactive exploration of the data
- Visualizations of model performance
- Confusion matrices
- Feature importance plots
- Comprehensive metrics displays

## Requirements

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

## Files Description

### Source Files (`src/`)

- **`reidentification_attack.py`**: Performs record linkage between ad records and census records using common quasi-identifiers.

### Notebooks (`notebooks/`)

- **`reconstruction_attack.ipynb`**: Interactive Jupyter notebook version of the reconstruction attack with enhanced visualizations and metrics display.

### Data Files (`data/`)

- **`synthetic_census_data.csv`**: Synthetic census records with personal attributes including income, education, occupation, etc.

- **`synthetic_facebook_ad_data.csv`**: Synthetic Facebook ad records with targeting information including ad interests.

### Output Files (`output/`)

- **`reid_matches.csv`**: Results from the reidentification attack, linking ad records to census records with match types (unique, ambiguous, none).

- **`reconstruction_predictions.csv`**: Predictions of Census attributes (income, education, occupation) for all ad records.

## Notes

- The reidentification attack must be run before the reconstruction attack, as the reconstruction attack uses the linked records from `reid_matches.csv`.

- All file paths are relative to the project root directory, so scripts should be run from their respective directories (`src/` for Python scripts, `notebooks/` for notebooks).

