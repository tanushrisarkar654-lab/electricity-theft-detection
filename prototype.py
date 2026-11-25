"""
Command-line prototype: score consumers from a CSV of daily readings.

Input CSV columns: consumer_id, date, kwh   (same format as data/raw/daily_consumption.csv)
Output: one row per consumer with risk_score and verdict.

Usage:
  python src/prototype.py data/raw/daily_consumption.csv --out outputs/predictions.csv
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_feature_table  # noqa: E402

MIN_DAYS = 30


def score(readings: pd.DataFrame, model_path: Path, threshold: float = 0.5) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    counts = readings.groupby("consumer_id").size()
    short = counts[counts < MIN_DAYS].index.tolist()
    if short:
        print(f"Skipping {len(short)} consumers with fewer than {MIN_DAYS} days", file=sys.stderr)
    readings = readings[~readings.consumer_id.isin(short)]

    feats = build_feature_table(readings)
    prob = bundle["model"].predict_proba(feats[bundle["features"]].values)[:, 1]
    out = pd.DataFrame({
        "consumer_id": feats["consumer_id"],
        "risk_score": prob.round(3),
        "verdict": ["SUSPECTED THEFT / ABNORMAL" if p >= threshold else "normal" for p in prob],
    })
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description="Score consumers for electricity theft risk")
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best_model.joblib")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    readings = pd.read_csv(args.input_csv, parse_dates=["date"])
    result = score(readings, args.model, args.threshold)
    print(result.head(10).to_string(index=False))
    print(f"\nFlagged {int((result.verdict != 'normal').sum())} of {len(result)} consumers")
    if args.out:
        result.to_csv(args.out, index=False)
        print(f"Saved predictions to {args.out}")


if __name__ == "__main__":
    main()
