# Electricity Theft and Abnormal Consumption Detection

Machine learning system that flags electricity consumers whose smart-meter readings suggest meter bypass, tampering, or other abnormal consumption.

- **Full report:** [`Project_Report.md`](Project_Report.md)
- **Notebook:** [`notebooks/electricity_theft_detection.ipynb`](notebooks/electricity_theft_detection.ipynb)
- **Submission texts (LinkedIn post, email):** [`Submission_Drafts.md`](Submission_Drafts.md)

## Problem

Electricity theft causes large non-technical losses for utilities. Smart meters record daily kWh, which can reveal theft through level drops, zero-reading gaps, and abnormal spikes. This project turns that history into a per-consumer risk score.

## Dataset

The data is **synthetic**, because real utility meter data is confidential.

| File | Contents |
|---|---|
| `data/raw/daily_consumption.csv` | 365,000 daily readings: 1,000 consumers × 365 days (`consumer_id, date, kwh`) |
| `data/raw/consumer_labels.csv` | Ground truth per consumer: `theft_type` (none / bypass / intermittent / spike_drop) and `is_theft` |
| `data/processed/consumer_features.csv` | 15 engineered features per consumer |

Theft prevalence is 15%. Normal consumers include benign anomalies (vacant homes, genuine growth, legitimate spikes) so that the task is not trivially separable.

## Repository structure

```
├── README.md
├── Project_Report.md              # full methodology, results, and discussion
├── Submission_Drafts.md           # LinkedIn post and email drafts
├── requirements.txt
├── data/
│   ├── raw/                       # generated readings and labels
│   └── processed/                 # engineered features
├── notebooks/
│   └── electricity_theft_detection.ipynb   # full pipeline (executed, with outputs)
├── src/
│   ├── generate_dataset.py        # synthetic data generator
│   ├── features.py                # feature engineering (shared by notebook and prototype)
│   └── prototype.py               # command-line scorer for new readings
├── models/
│   └── best_model.joblib          # trained production model + feature list
└── outputs/
    ├── cv_results.csv             # 5-fold cross-validation results
    ├── test_results.csv           # held-out test results
    ├── recall_by_theft_type.csv   # error analysis
    ├── feature_importance_rf.csv
    ├── predictions_all_consumers.csv
    └── figures/                   # all plots used in the report
```

## Setup

```bash
pip install -r requirements.txt
```

Python 3.10 or newer is recommended.

## How to run

1. **Regenerate the dataset (optional; a fixed seed reproduces the same data):**
   ```bash
   python src/generate_dataset.py --consumers 1000 --days 365 --seed 42
   ```
2. **Recompute features (optional):**
   ```bash
   cd src && python -c "import pandas as pd; from features import build_feature_table; \
   r=pd.read_csv('../data/raw/daily_consumption.csv', parse_dates=['date']); \
   build_feature_table(r).to_csv('../data/processed/consumer_features.csv', index=False)"
   ```
3. **Run the full analysis:** open `notebooks/electricity_theft_detection.ipynb` in Jupyter and run all cells. It trains and compares five supervised models plus an unsupervised baseline, tunes a Random Forest, and saves the production model to `models/`.
4. **Score new consumers with the prototype:**
   ```bash
   python src/prototype.py path/to/readings.csv --out outputs/predictions.csv
   ```
   The input CSV needs the columns `consumer_id, date, kwh`. Consumers with fewer than 30 days of data are skipped.

## Methods in brief

- **Features (15):** level and seasonal change, coefficient of variation, zero-reading ratio and longest zero run, spike and drop ratios, spike magnitude, skew, kurtosis, lag-1 autocorrelation, and weekend uplift.
- **Models:** Logistic Regression, Random Forest (also tuned by grid search), Gradient Boosting, SVM (RBF), KNN, and Isolation Forest (unsupervised baseline).
- **Evaluation:** stratified 5-fold cross-validation on 80% of the data, then one run on the untouched 20% test set. Metrics: precision, recall, F1, ROC-AUC, PR-AUC.

## Headline results (held-out test set, 200 consumers, 30 thefts)

| Model | F1 | ROC-AUC | Recall |
|---|---|---|---|
| **Gradient Boosting** (selected) | **0.871** | **0.987** | 0.90 |
| Random Forest | 0.867 | 0.980 | 0.87 |
| SVM (RBF) | 0.831 | 0.982 | 0.90 |
| Logistic Regression | 0.784 | 0.977 | 0.97 |
| Isolation Forest (no labels) | 0.644 | 0.892 | 0.63 |

Bypass theft is the hardest type to catch (73% recall). Intermittent and spike-and-drop theft were caught 100% of the time in the test set. See the report for limitations.

## Limitations

- Synthetic data: performance on real utility data will differ.
- Only 30 theft cases in the test set, so the metrics carry wide uncertainty.
- Daily readings only. Hourly data would likely improve bypass detection.
