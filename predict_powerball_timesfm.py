#!/usr/bin/env python3
"""
Next-draw Powerball prediction using Google's TimesFM 2.5 (200M) foundation
time-series model (google/timesfm-2.5-200m-pytorch).

Each white-ball position is treated as a univariate series over the draw history
and forecast one step ahead, zero-shot, on the GPU. As with every other model in
this sandbox, TimesFM has no real predictive edge on an i.i.d. lottery -- it
forecasts close to each position's marginal with wide uncertainty. Run
``evaluation_harness.py`` (which includes the ``timesfm`` family) for the honest,
baseline-relative verdict.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from powerball_extra_models import timesfm_forecast
from autoresearch_powerball import WHITE_BALL_COLUMNS, load_dataset, next_drawing_date

DATA = "powerball_games_only.csv"
WHITE_MIN, WHITE_MAX = 1, 69


def dedupe_within_range(values, lo=WHITE_MIN, hi=WHITE_MAX):
    """Make the five forecast numbers distinct, nudging collisions outward."""
    out = []
    used = set()
    for v in values:
        v = int(np.clip(round(v), lo, hi))
        step = 0
        while v in used:
            step += 1
            cand = v + step if (v + step) <= hi else v - step
            if cand in used:
                cand = v - step if (v - step) >= lo else v + step
            v = int(np.clip(cand, lo, hi))
            if step > hi:
                break
        used.add(v)
        out.append(v)
    return out


def main() -> None:
    print("=" * 70)
    print("POWERBALL PREDICTION USING TIMESFM 2.5 (200M) FOUNDATION MODEL")
    print("=" * 70)

    df = load_dataset(Path(DATA))
    print(f"Loaded {len(df)} drawings ({df['date'].min().date()} .. {df['date'].max().date()})")

    series_list = [df[col].to_numpy(dtype=float) for col in WHITE_BALL_COLUMNS]
    print("Forecasting next draw for all 5 positions (zero-shot, GPU)...")
    point, quantiles = timesfm_forecast(series_list, horizon=1)

    raw = [float(point[i, 0]) for i in range(5)]
    predicted = dedupe_within_range(raw)

    next_date = next_drawing_date(df["date"].max())
    print(f"\nTarget draw date: {next_date.strftime('%Y-%m-%d')}")
    print("\nPer-position forecast (point + 0.1/0.9 deciles):")
    for i, col in enumerate(WHITE_BALL_COLUMNS):
        q = quantiles[i, 0]
        print(f"  {col}: {raw[i]:5.1f} -> {predicted[i]:2d}   "
              f"[p10={q[1]:.1f}, p50={q[5]:.1f}, p90={q[9]:.1f}]")

    print(f"\nPredicted Numbers: {predicted}")
    print(f"Sorted: {sorted(predicted)}")

    # Save prediction CSV
    out = {
        "prediction_date": next_date.strftime("%Y-%m-%d"),
        "model": "TimesFM-2.5-200M",
        **{f"ball_{i}": predicted[i - 1] for i in range(1, 6)},
        "sorted": str(sorted(predicted)),
    }
    pd.DataFrame([out]).to_csv("timesfm_predictions.csv", index=False)
    print("\n✓ Predictions saved to timesfm_predictions.csv")

    # Log to the shared historical tracker (best-effort)
    try:
        from prediction_tracker import PredictionTracker

        tracker = PredictionTracker()
        tracker.add_prediction(
            target_date=next_date.strftime("%Y-%m-%d"),
            model_type="timesfm",
            numbers=predicted,
            model_details={
                "model": "google/timesfm-2.5-200m-pytorch",
                "device": "cuda",
                "horizon": 1,
                "data_points": int(len(df)),
            },
        )
        tracker.save_history()
        print("✓ Stored prediction in historical tracker")
    except Exception as exc:
        print(f"⚠ Could not store in historical tracker: {exc}")

    print("\n" + "=" * 70)
    print("DISCLAIMER: Lottery draws are random. A foundation forecaster has no")
    print("edge on winning numbers; it reverts to each position's marginal.")
    print("=" * 70)


if __name__ == "__main__":
    main()
