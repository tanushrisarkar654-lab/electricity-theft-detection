# Electricity Theft and Abnormal Consumption Detection

**Project type:** Machine learning, supervised and unsupervised classification
**Team:** [Your name(s) and roll number(s)]
**Course:** [Course name]

---

## 1. Problem Statement and Objectives

Electricity theft, through meter bypass, tampering, or interference, causes large non-technical losses for utilities and distorts consumption records. Smart meters record daily consumption, so theft often shows up as a level drop, runs of zero readings, or abnormal spikes.

**Problem:** Given a consumer's daily kWh history, decide whether the consumer is normal or likely stealing electricity or behaving abnormally.

**Objectives**
1. Build a labelled consumer dataset with realistic theft signatures.
2. Engineer features that capture those signatures.
3. Implement and compare at least three ML algorithms.
4. Evaluate with stratified cross-validation and an untouched test set, using metrics suited to class imbalance.
5. Analyse errors by theft type and interpret the model.
6. Deliver a working prototype that scores new consumers.

## 2. Dataset

**Source:** Synthetic. Real utility meter data is confidential, so `src/generate_dataset.py` simulates it. The seed is fixed (42), so the data is reproducible.

| Property | Value |
|---|---|
| Consumers | 1,000 |
| Days per consumer | 365 (2025) |
| Total readings | 365,000 daily kWh values |
| Theft consumers | 150 (15%) |
| Theft types | Bypass (48), intermittent (58), spike-and-drop (44) |
| Normal consumers | 850, including benign anomalies |

**Simulation design**

- *Normal consumption:* per-consumer base level (lognormal), annual seasonality, a weekend uplift, slow trend, and multiplicative daily noise (σ = 0.28).
- *Benign anomalies (in normal consumers, about 22%):* vacant home or holiday shutdown (8%, drop to 20–40% for 30–60 days), genuine growth such as a new appliance or larger household (8%, level rises 30–60%), and legitimate high-usage spikes (6%).
- *Bypass theft:* from a random start date, recorded consumption is cut to 55–85% of true usage.
- *Intermittent theft:* 5–20% of days are recorded at 5–40% of normal.
- *Spike-and-drop theft:* short 1.6–2.6× spikes, followed by a sustained 10–30% under-report from a later date.

The benign anomalies and moderate theft strengths were added deliberately. An earlier version with strong theft signals gave every model a perfect score, which showed the task was too easy to test anything. The current version produces the differences in performance reported here.

## 3. Feature Engineering

Each consumer's 365-day series is reduced to 15 features (`src/features.py`):

| Group | Features | Targets |
|---|---|---|
| Level | `mean_kwh`, `median_kwh`, `std_kwh` | Scale (not used alone, since it varies by household) |
| Variability | `cv`, `skew`, `kurtosis` | Irregular or shape-changing usage |
| Zero readings | `zero_ratio`, `longest_zero_run` | Intermittent meter interference |
| Level change | `level_change` (second half vs first half), `quarter_change` (last vs first quarter) | Bypass and under-reporting |
| Spikes and drops | `spike_ratio`, `drop_ratio`, `spike_magnitude` | Abrupt changes |
| Temporal | `autocorr_lag1`, `weekend_ratio` | Loss of normal patterns |

The features are computed by the same function in the notebook and the prototype, and the notebook checks that the precomputed table matches a fresh computation.

## 4. Preprocessing and Experimental Design

- **Split:** stratified 80/20 train/test split (800 training, 200 test consumers, 30 thefts in the test set). Stratification keeps the 15% theft rate in both sets.
- **Scaling:** standardisation inside each pipeline, fitted on training folds only, to prevent leakage.
- **Imbalance:** `class_weight="balanced"` where supported.
- **Validation:** 5-fold stratified cross-validation on the training set for model comparison. The test set was used once for the final comparison and for nothing else.

## 5. Models

| # | Algorithm | Type | Notes |
|---|---|---|---|
| 1 | Logistic Regression | Supervised, linear | Interpretable baseline |
| 2 | Random Forest | Supervised, bagged trees | 300 trees; also grid-tuned |
| 3 | Gradient Boosting | Supervised, boosted trees | 200 estimators, depth 3 |
| 4 | SVM (RBF kernel) | Supervised, kernel margin | C = 2, probability outputs |
| 5 | K-Nearest Neighbours | Supervised, distance-based | k = 7, distance-weighted |
| 6 | Isolation Forest | Unsupervised | Trained without labels; a baseline for "no labels available" |

Random Forest hyperparameters were tuned with grid search (5-fold CV, F1 scoring) over the number of trees (200, 400), maximum depth (none, 6, 10), and minimum leaf size (1, 3). The best settings were 400 trees, unlimited depth, and minimum leaf 3, with CV F1 0.882.

## 6. Evaluation Metrics

Accuracy alone is misleading at 15% prevalence (predicting "normal" for everyone scores 85%). The main metrics are therefore:

- **Precision:** of flagged consumers, the share that are thieves. This sets the cost of wasted inspections.
- **Recall:** of thieves, the share caught. This sets the cost of missed losses.
- **F1:** the harmonic mean of precision and recall.
- **ROC-AUC and PR-AUC:** threshold-independent ranking quality. PR-AUC is the more informative measure under imbalance.

## 7. Results

### 7.1 Cross-validation (training set, 5-fold, mean ± std)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.953 ± 0.006 | 0.797 ± 0.044 | 0.925 ± 0.049 | 0.854 ± 0.017 | **0.985 ± 0.009** | 0.940 ± 0.035 |
| SVM (RBF) | 0.954 ± 0.005 | 0.807 ± 0.012 | **0.908 ± 0.031** | **0.855 ± 0.017** | 0.981 ± 0.009 | 0.892 ± 0.056 |
| Random Forest | **0.959 ± 0.015** | **0.931 ± 0.019** | 0.783 ± 0.107 | 0.847 ± 0.065 | 0.982 ± 0.016 | **0.944 ± 0.032** |
| Gradient Boosting | 0.954 ± 0.018 | 0.869 ± 0.051 | 0.817 ± 0.114 | 0.838 ± 0.071 | 0.977 ± 0.021 | 0.928 ± 0.044 |
| KNN | 0.953 ± 0.013 | 0.886 ± 0.039 | 0.783 ± 0.055 | 0.831 ± 0.048 | 0.955 ± 0.020 | 0.907 ± 0.035 |

The CV results are close, and the standard deviations overlap for the top models. Logistic Regression had the best CV ROC-AUC. Random Forest had the highest precision and PR-AUC.

### 7.2 Held-out test set (200 consumers, 30 thefts)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **Gradient Boosting** | **0.960** | 0.844 | 0.900 | **0.871** | **0.987** | **0.949** |
| Random Forest | 0.960 | **0.867** | 0.867 | 0.867 | 0.980 | 0.921 |
| KNN | 0.955 | 0.862 | 0.833 | 0.847 | 0.967 | 0.910 |
| SVM (RBF) | 0.945 | 0.771 | 0.900 | 0.831 | 0.982 | 0.922 |
| Logistic Regression | 0.920 | 0.659 | **0.967** | 0.784 | 0.977 | 0.865 |
| Isolation Forest (no labels) | 0.895 | 0.655 | 0.633 | 0.644 | 0.892 | 0.688 |

**Selected model:** Gradient Boosting, with the best test F1 and ROC-AUC. The tuned Random Forest (test F1 0.857, ROC-AUC 0.980, PR-AUC 0.929) did not beat it, so the default Gradient Boosting was kept. The model was refit on all 1,000 consumers before saving to `models/best_model.joblib`.

### 7.3 Error analysis by theft type (test set, Gradient Boosting baseline in the notebook's best-model cell)

| Theft type | Thefts in test | Caught | Recall |
|---|---|---|---|
| Bypass | 11 | 8 | 73% |
| Intermittent | 12 | 12 | 100% |
| Spike-and-drop | 7 | 7 | 100% |

Bypass is hardest because a 55–85% cut in consumption overlaps with a vacant home or a seasonal dip, which the benign anomalies in normal consumers imitate. Intermittent zeros and spikes are more distinctive, which matches the feature importances.

### 7.4 Feature importance (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | `level_change` | 0.189 |
| 2 | `cv` | 0.176 |
| 3 | `quarter_change` | 0.147 |
| 4 | `spike_magnitude` | 0.080 |
| 5 | `drop_ratio` | 0.070 |

Level change and variability dominate, which fits the bypass and spike-and-drop signatures. Zero-reading features matter less here because intermittent theft leaves a clear signal in the variability features as well.

### 7.5 Isolation Forest (unsupervised)

Isolation Forest, trained without labels and with a 15% contamination setting, reached F1 0.644 and ROC-AUC 0.892. It finds many anomalies but cannot tell theft from vacancy or growth. It is useful as a triage step when no labelled data exists, not as a replacement for supervised models.

## 8. Prototype

`src/prototype.py` loads the saved model and scores any CSV of daily readings:

```bash
python src/prototype.py data/raw/daily_consumption.csv --out outputs/predictions_all_consumers.csv
```

On the 1,000-consumer dataset it flagged 148 consumers. The output is sorted by risk, so inspectors can work from the top. The notebook also has an in-line `detect_theft()` function for single consumers.

*Note:* the 99.8% agreement against labels in `outputs/predictions_all_consumers.csv` is in-sample. The model was refit on these same consumers, so it is optimistic. The test-set results in Section 7.2 are the honest estimate.

## 9. Discussion

- **Ranking is strong, thresholds need tuning.** ROC-AUC is about 0.98 for most supervised models, but at the default 0.5 threshold precision and recall trade off. A utility should set the threshold from the cost of an inspection against the revenue lost to theft.
- **Simple models are competitive.** Logistic Regression ranks consumers almost as well as the ensemble methods. For a utility that must explain flags to customers and regulators, that is a useful result.
- **Benign anomalies are the main risk.** Vacant homes and genuine growth look like theft to a model. Any real deployment needs context such as occupancy records, tariff changes, or customer history.

## 10. Limitations

1. **Synthetic data.** The theft and benign patterns were designed by us. Real results will differ, and the notebook should be rerun on utility data.
2. **Small test theft count.** Thirty thefts means one or two errors move recall by several points. The differences between the top four models are within noise.
3. **Daily granularity.** Hourly or half-hourly data would reveal more about bypass and load-shape anomalies.
4. **No external context.** Weather, occupancy, and tariff data are not used.
5. **Perfect labels assumed.** Real labels come from inspections and are noisy.

## 11. Future Work

- Validate on real utility data with inspection outcomes as labels.
- Use hourly data and compare each consumer with similar consumers (peer-group normalisation).
- Add temporal deep-learning models (LSTM or residuals from a seasonal forecast).
- Choose the decision threshold from inspection costs and expected recovery.
- Add explainability (SHAP values) per flagged consumer to support field inspection.

## 12. Repository and Reproduction

See `README.md` for the file layout and run instructions. All results in this report come from executing `notebooks/electricity_theft_detection.ipynb` end to end on the dataset generated with seed 42.

## 13. Conclusion

Supervised ensemble and linear models detect electricity theft and abnormal consumption well on this synthetic benchmark, with ROC-AUC near 0.98 and F1 around 0.87 on held-out consumers. Level-change and variability features carry most of the signal, and bypass theft is the hardest case to catch. The results depend on synthetic data, so the main next step is validation on real smart-meter data.
