"""
Synthetic smart-meter dataset generator for Electricity Theft and Abnormal
Consumption Detection.

Simulates daily kWh readings for N consumers over D days. A fraction of
consumers are "thieves" whose readings are manipulated in one of three ways:

  1. bypass      - a meter bypass / tampering that reduces recorded consumption
                   from some day onward (recorded = true * factor, factor 0.3-0.6)
  2. intermittent- random days are recorded as zero or near zero (meter
                   interference, magnet attacks, intermittent bypass)
  3. spike_drop  - abnormal high-usage spikes (e.g. illegal heating loads or
                   hooked-in appliances) followed by under-reporting

Outputs (relative to project root):
  data/raw/daily_consumption.csv   consumer_id, date, kwh
  data/raw/consumer_labels.csv     consumer_id, theft_type, is_theft

Usage:
  python src/generate_dataset.py --consumers 1000 --days 365 --seed 42
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"


def simulate_consumer(rng, n_days, base_kwh, is_theft, theft_type):
    days = np.arange(n_days)
    # Seasonality: annual cycle and a weekend uplift
    annual = 1 + 0.25 * np.cos(2 * np.pi * days / 365.0)
    weekly = np.where((days % 7) >= 5, 1.12, 1.0)
    trend = 1 + rng.normal(0, 0.10) * days / n_days
    # Day-to-day noise is deliberately large so theft signatures overlap with normal variation
    noise = rng.lognormal(mean=0.0, sigma=0.28, size=n_days)
    true_kwh = base_kwh * annual * weekly * trend * noise

    recorded = true_kwh.copy()

    # Benign anomalies in NORMAL consumers (realistic false-positive sources)
    if not is_theft:
        r = rng.random()
        if r < 0.08:  # vacant home / holiday shutdown for 30-60 days
            s = rng.integers(0, n_days - 60)
            recorded[s:s + rng.integers(30, 61)] *= rng.uniform(0.2, 0.4)
        elif r < 0.16:  # genuine growth: new appliance, larger household
            start = rng.integers(int(0.3 * n_days), int(0.8 * n_days))
            recorded[start:] *= rng.uniform(1.3, 1.6)
        elif r < 0.22:  # occasional legitimate high-usage spikes
            for s in rng.integers(0, n_days - 5, size=rng.integers(2, 5)):
                recorded[s:s + rng.integers(2, 5)] *= rng.uniform(1.8, 2.5)

    # Theft signatures (moderate strength, overlapping with benign behaviour)
    if is_theft:
        if theft_type == "bypass":
            start = rng.integers(int(0.1 * n_days), int(0.7 * n_days))
            recorded[start:] *= rng.uniform(0.55, 0.85)
        elif theft_type == "intermittent":
            frac = rng.uniform(0.05, 0.20)
            idx = rng.choice(n_days, size=int(frac * n_days), replace=False)
            recorded[idx] *= rng.uniform(0.05, 0.4, size=idx.size)
        elif theft_type == "spike_drop":
            for s in rng.integers(0, n_days - 10, size=rng.integers(2, 6)):
                recorded[s:s + rng.integers(2, 6)] *= rng.uniform(1.6, 2.6)
            start = rng.integers(int(0.3 * n_days), int(0.8 * n_days))
            recorded[start:] *= rng.uniform(0.7, 0.9)
    return np.round(np.clip(recorded, 0, None), 2)


def generate(n_consumers=500, n_days=365, theft_rate=0.15, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    types = ["bypass", "intermittent", "spike_drop"]

    n_theft = int(round(theft_rate * n_consumers))
    theft_flags = np.zeros(n_consumers, dtype=bool)
    theft_flags[:n_theft] = True
    rng.shuffle(theft_flags)

    records, labels = [], []
    for i in range(n_consumers):
        cid = f"C{i + 1:04d}"
        base = rng.lognormal(mean=np.log(10.0), sigma=0.45)  # ~10 kWh/day typical
        t_type = rng.choice(types) if theft_flags[i] else "none"
        series = simulate_consumer(rng, n_days, base, theft_flags[i], t_type)
        records.append(pd.DataFrame({"consumer_id": cid, "date": dates, "kwh": series}))
        labels.append({"consumer_id": cid, "theft_type": t_type, "is_theft": int(theft_flags[i])})

    return pd.concat(records, ignore_index=True), pd.DataFrame(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumers", type=int, default=1000)
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--theft_rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    readings, labels = generate(args.consumers, args.days, args.theft_rate, args.seed)
    readings.to_csv(RAW_DIR / "daily_consumption.csv", index=False)
    labels.to_csv(RAW_DIR / "consumer_labels.csv", index=False)
    print(f"Wrote {len(readings):,} readings for {args.consumers} consumers")
    print(labels["theft_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
