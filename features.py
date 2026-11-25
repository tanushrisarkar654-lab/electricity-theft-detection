"""
Feature engineering for consumer-level electricity theft detection.

Each consumer's daily kWh series is summarised into a fixed-length feature
vector. The features are chosen to capture the three theft signatures:
  - level drops (bypass)          -> ratio of late vs early consumption
  - zero / near-zero readings     -> zero_ratio, longest zero run
  - abrupt anomalies              -> spike ratio, big day-to-day drops
  - irregularity / shape changes  -> cv, skew, kurtosis, autocorrelation
"""
import numpy as np
import pandas as pd
from scipy import stats


def series_features(x: np.ndarray, base_threshold: float = None) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    std = x.std(ddof=1)
    median = np.median(x)
    eps = 1e-9

    # Zero-based signatures
    near_zero = x < 0.1 * max(median, eps)
    zero_ratio = near_zero.mean()
    longest_zero_run = _longest_run(near_zero)

    # Level change: first vs last part of the year
    half = n // 2
    first_half, second_half = x[:half].mean(), x[half:].mean()
    level_change = second_half / (first_half + eps)

    last_quarter = x[-n // 4:].mean()
    first_quarter = x[: n // 4].mean()
    quarter_change = last_quarter / (first_quarter + eps)

    # Spikes and abrupt drops (day-to-day relative change)
    rel = x[1:] / (x[:-1] + eps)
    spike_ratio = (rel > 2.0).mean()
    drop_ratio = (rel < 0.5).mean()

    # Spike magnitude vs typical usage
    spike_magnitude = np.percentile(x, 99) / (median + eps)

    # Irregularity
    cv = std / (mean + eps)
    skew = stats.skew(x)
    kurt = stats.kurtosis(x)
    ac1 = pd.Series(x).autocorr(lag=1)
    ac1 = 0.0 if np.isnan(ac1) else ac1

    # Weekly pattern (weekend vs weekday uplift); assumes series starts Monday-aligned
    idx = np.arange(n)
    weekend = (idx % 7) >= 5
    weekend_ratio = x[weekend].mean() / (x[~weekend].mean() + eps)

    return {
        "mean_kwh": mean,
        "median_kwh": median,
        "std_kwh": std,
        "cv": cv,
        "zero_ratio": zero_ratio,
        "longest_zero_run": longest_zero_run,
        "level_change": level_change,
        "quarter_change": quarter_change,
        "spike_ratio": spike_ratio,
        "drop_ratio": drop_ratio,
        "spike_magnitude": spike_magnitude,
        "skew": skew,
        "kurtosis": kurt,
        "autocorr_lag1": ac1,
        "weekend_ratio": weekend_ratio,
    }


def _longest_run(bool_arr: np.ndarray) -> int:
    best = cur = 0
    for v in bool_arr:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def build_feature_table(readings: pd.DataFrame) -> pd.DataFrame:
    """readings: consumer_id, date, kwh  ->  one row per consumer."""
    rows = []
    for cid, grp in readings.sort_values(["consumer_id", "date"]).groupby("consumer_id"):
        feats = series_features(grp["kwh"].to_numpy())
        feats["consumer_id"] = cid
        rows.append(feats)
    return pd.DataFrame(rows)[["consumer_id"] + [c for c in pd.DataFrame(rows).columns if c != "consumer_id"]]


FEATURE_COLUMNS = [
    "mean_kwh", "median_kwh", "std_kwh", "cv", "zero_ratio", "longest_zero_run",
    "level_change", "quarter_change", "spike_ratio", "drop_ratio",
    "spike_magnitude", "skew", "kurtosis", "autocorr_lag1", "weekend_ratio",
]
