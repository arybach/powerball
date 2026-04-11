#!/usr/bin/env python3
"""
Autonomous experiment loop for Powerball prediction models.

Implements an autoresearch-style workflow for two model families:
1) Fourier-on-bins predictor
2) RandomForest-on-bins predictor

For each model:
- run a baseline configuration first
- mutate hyperparameters in a loop
- evaluate with walk-forward backtesting
- keep or discard configs based on objective score
- log every experiment to TSV
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


WHITE_BALL_COLUMNS = [f"ball_{i}" for i in range(1, 6)]


@dataclass
class EvalResult:
    score: float
    mae: float
    bin_accuracy: float
    hits: float
    objective: str
    status: str
    description: str


def next_drawing_date(last_date: pd.Timestamp) -> pd.Timestamp:
    # Powerball draws Monday/Wednesday/Saturday.
    weekdays = {0, 2, 5}
    candidate = last_date + timedelta(days=1)
    while candidate.weekday() not in weekdays:
        candidate += timedelta(days=1)
    return candidate


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def compute_bin_edges(train_df: pd.DataFrame, n_bins: int) -> Dict[str, np.ndarray]:
    edges: Dict[str, np.ndarray] = {}
    for col in WHITE_BALL_COLUMNS:
        values = train_df[col].astype(float).to_numpy()
        q = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        q = np.unique(q)
        if len(q) < n_bins + 1:
            q = np.linspace(values.min(), values.max() + 0.1, n_bins + 1)
        edges[col] = q
    return edges


def to_bins(values: np.ndarray, edges: np.ndarray, n_bins: int) -> np.ndarray:
    b = np.digitize(values, edges) - 1
    b = np.clip(b, 0, n_bins - 1) + 1
    return b


def bin_to_number(bin_value: float, edges: np.ndarray, n_bins: int) -> int:
    idx = int(np.clip(np.round(bin_value) - 1, 0, n_bins - 1))
    return int((edges[idx] + edges[idx + 1]) / 2)


def fourier_predict_next_bin(
    series_bins: np.ndarray,
    next_index: int,
    harmonics: int,
    ridge_alpha: float,
) -> float:
    x = np.arange(len(series_bins), dtype=float)
    y = series_bins.astype(float)
    if len(x) < 8:
        return float(np.median(y))

    x_scale = np.pi * (x / max(1.0, x[-1]))
    x_next = np.pi * (next_index / max(1.0, x[-1]))

    cols = [np.ones_like(x_scale)]
    next_cols = [np.array([1.0])]
    for k in range(1, harmonics + 1):
        cols.append(np.sin(k * x_scale))
        cols.append(np.cos(k * x_scale))
        next_cols.append(np.sin(np.array([k * x_next])))
        next_cols.append(np.cos(np.array([k * x_next])))

    X = np.column_stack(cols)
    X_next = np.column_stack(next_cols)

    xtx = X.T @ X + ridge_alpha * np.eye(X.shape[1])
    xty = X.T @ y
    coeff = np.linalg.solve(xtx, xty)
    pred = float((X_next @ coeff)[0])
    return pred


def random_forest_predict_next_bin(
    series_bins: np.ndarray,
    seq_len: int,
    n_estimators: int,
    max_depth: int,
    min_samples_split: int,
    random_state: int,
) -> float:
    if len(series_bins) <= seq_len:
        return float(np.median(series_bins))

    X: List[np.ndarray] = []
    y: List[int] = []
    for i in range(seq_len, len(series_bins)):
        X.append(series_bins[i - seq_len : i])
        y.append(int(series_bins[i]))

    if len(X) < 8:
        return float(np.median(series_bins))

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1,
    )
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    model.fit(X_arr, y_arr)
    pred = float(model.predict(series_bins[-seq_len:].reshape(1, -1))[0])
    return pred


def score_metrics(mae: float, bin_accuracy: float, hits: float, objective: str) -> float:
    # Lower is better for all objectives.
    if objective == "balanced":
        return mae - (4.0 * bin_accuracy) - (1.5 * hits)
    if objective == "bin_focus":
        return mae - (7.0 * bin_accuracy) - (1.0 * hits)
    raise ValueError(f"Unknown objective: {objective}")


def evaluate_config(
    df: pd.DataFrame,
    model_type: str,
    config: Dict,
    eval_draws: int,
    random_state: int,
    objective: str,
) -> EvalResult:
    n = len(df)
    start = max(120, n - eval_draws)
    if start >= n:
        raise ValueError("Not enough rows to evaluate")

    abs_errors: List[float] = []
    exact_bin_matches = 0
    total_bin_preds = 0
    hit_counts: List[int] = []

    for target_idx in range(start, n):
        train_df = df.iloc[:target_idx]
        target_row = df.iloc[target_idx]

        n_bins = int(config["n_bins"])
        edges = compute_bin_edges(train_df, n_bins)

        predicted_nums: List[int] = []
        actual_nums = [int(target_row[c]) for c in WHITE_BALL_COLUMNS]

        for col in WHITE_BALL_COLUMNS:
            train_series = train_df[col].to_numpy()
            train_bins = to_bins(train_series, edges[col], n_bins)
            actual_bin = int(to_bins(np.array([target_row[col]]), edges[col], n_bins)[0])

            if model_type == "fourier":
                pred_bin = fourier_predict_next_bin(
                    train_bins,
                    next_index=len(train_bins),
                    harmonics=int(config["harmonics"]),
                    ridge_alpha=float(config["ridge_alpha"]),
                )
            elif model_type == "random_forest":
                pred_bin = random_forest_predict_next_bin(
                    train_bins,
                    seq_len=int(config["sequence_length"]),
                    n_estimators=int(config["n_estimators"]),
                    max_depth=int(config["max_depth"]),
                    min_samples_split=int(config["min_samples_split"]),
                    random_state=random_state,
                )
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            pred_bin_rounded = int(np.clip(np.round(pred_bin), 1, n_bins))
            predicted_number = bin_to_number(pred_bin_rounded, edges[col], n_bins)
            predicted_nums.append(predicted_number)

            abs_errors.append(abs(int(target_row[col]) - predicted_number))
            exact_bin_matches += int(pred_bin_rounded == actual_bin)
            total_bin_preds += 1

        hit_counts.append(len(set(predicted_nums).intersection(set(actual_nums))))

    mae = float(np.mean(abs_errors))
    bin_accuracy = float(exact_bin_matches / max(1, total_bin_preds))
    hits = float(np.mean(hit_counts))

    score = score_metrics(mae, bin_accuracy, hits, objective)

    return EvalResult(
        score=score,
        mae=mae,
        bin_accuracy=bin_accuracy,
        hits=hits,
        objective=objective,
        status="ok",
        description="evaluated",
    )


def mutate_fourier_config(best: Dict, rng: np.random.Generator) -> Dict:
    c = dict(best)
    c["harmonics"] = int(np.clip(c["harmonics"] + rng.integers(-2, 3), 2, 24))
    c["n_bins"] = int(np.clip(c["n_bins"] + rng.integers(-1, 2), 4, 10))
    c["ridge_alpha"] = float(np.clip(c["ridge_alpha"] * rng.choice([0.5, 0.8, 1.25, 1.6]), 1e-6, 5.0))
    return c


def mutate_rf_config(best: Dict, rng: np.random.Generator) -> Dict:
    c = dict(best)
    c["sequence_length"] = int(np.clip(c["sequence_length"] + rng.integers(-3, 4), 6, 40))
    c["n_bins"] = int(np.clip(c["n_bins"] + rng.integers(-1, 2), 4, 10))
    c["n_estimators"] = int(np.clip(c["n_estimators"] + rng.integers(-30, 31), 80, 240))
    c["max_depth"] = int(np.clip(c["max_depth"] + rng.integers(-4, 5), 4, 40))
    c["min_samples_split"] = int(np.clip(c["min_samples_split"] + rng.integers(-1, 2), 2, 15))
    return c


def predict_next_numbers(df: pd.DataFrame, model_type: str, config: Dict, random_state: int) -> List[int]:
    train_df = df.copy()
    n_bins = int(config["n_bins"])
    edges = compute_bin_edges(train_df, n_bins)
    predicted: List[int] = []

    for col in WHITE_BALL_COLUMNS:
        train_bins = to_bins(train_df[col].to_numpy(), edges[col], n_bins)

        if model_type == "fourier":
            pred_bin = fourier_predict_next_bin(
                train_bins,
                next_index=len(train_bins),
                harmonics=int(config["harmonics"]),
                ridge_alpha=float(config["ridge_alpha"]),
            )
        else:
            pred_bin = random_forest_predict_next_bin(
                train_bins,
                seq_len=int(config["sequence_length"]),
                n_estimators=int(config["n_estimators"]),
                max_depth=int(config["max_depth"]),
                min_samples_split=int(config["min_samples_split"]),
                random_state=random_state,
            )

        pred_bin_rounded = int(np.clip(np.round(pred_bin), 1, n_bins))
        predicted.append(bin_to_number(pred_bin_rounded, edges[col], n_bins))

    return predicted


def append_result_tsv(tsv_path: Path, row: Dict) -> None:
    header = [
        "timestamp",
        "objective",
        "model",
        "experiment",
        "score",
        "mae",
        "bin_accuracy",
        "avg_hits",
        "status",
        "description",
        "params_json",
    ]
    if not tsv_path.exists():
        tsv_path.write_text("\t".join(header) + "\n", encoding="utf-8")

    values = [
        row["timestamp"],
        row["objective"],
        row["model"],
        row["experiment"],
        f"{row['score']:.6f}",
        f"{row['mae']:.6f}",
        f"{row['bin_accuracy']:.6f}",
        f"{row['avg_hits']:.6f}",
        row["status"],
        row["description"],
        json.dumps(row["params"], sort_keys=True),
    ]
    with tsv_path.open("a", encoding="utf-8") as f:
        f.write("\t".join(values) + "\n")


def run_model_loop(
    df: pd.DataFrame,
    model_type: str,
    baseline: Dict,
    iterations: int,
    eval_draws: int,
    tsv_path: Path,
    random_state: int,
    objective: str,
) -> Tuple[Dict, EvalResult]:
    rng = np.random.default_rng(random_state + (11 if model_type == "fourier" else 29))

    print(f"\n=== {model_type.upper()} LOOP ({objective}) ===")

    best_config = dict(baseline)
    best_eval = evaluate_config(df, model_type, best_config, eval_draws, random_state, objective)
    append_result_tsv(
        tsv_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "objective": objective,
            "model": model_type,
            "experiment": "baseline",
            "score": best_eval.score,
            "mae": best_eval.mae,
            "bin_accuracy": best_eval.bin_accuracy,
            "avg_hits": best_eval.hits,
            "status": "keep",
            "description": "baseline",
            "params": best_config,
        },
    )
    print(
        f"baseline score={best_eval.score:.4f} mae={best_eval.mae:.4f} "
        f"bin_acc={best_eval.bin_accuracy:.4f} hits={best_eval.hits:.4f}"
    )

    for i in range(1, iterations + 1):
        if model_type == "fourier":
            candidate = mutate_fourier_config(best_config, rng)
        else:
            candidate = mutate_rf_config(best_config, rng)

        ev = evaluate_config(df, model_type, candidate, eval_draws, random_state, objective)
        keep = ev.score < best_eval.score
        status = "keep" if keep else "discard"

        append_result_tsv(
            tsv_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "objective": objective,
                "model": model_type,
                "experiment": f"iter_{i}",
                "score": ev.score,
                "mae": ev.mae,
                "bin_accuracy": ev.bin_accuracy,
                "avg_hits": ev.hits,
                "status": status,
                "description": f"mutated from current best, iter {i}",
                "params": candidate,
            },
        )

        print(
            f"iter {i:02d}: score={ev.score:.4f} mae={ev.mae:.4f} "
            f"bin_acc={ev.bin_accuracy:.4f} hits={ev.hits:.4f} -> {status}"
        )

        if keep:
            best_config = candidate
            best_eval = ev

    return best_config, best_eval


def write_summary(
    output_dir: Path,
    objective: str,
    fourier_best: Dict,
    rf_best: Dict,
    fourier_eval: EvalResult,
    rf_eval: EvalResult,
    next_date: pd.Timestamp,
    fourier_pred: List[int],
    rf_pred: List[int],
) -> None:
    summary_path = output_dir / "AUTORESEARCH_SUMMARY.md"
    text = f"""# Powerball Autoresearch Summary

Generated: {datetime.now().isoformat(timespec='seconds')}

Objective: {objective}

## Best Fourier Configuration

- Score: {fourier_eval.score:.6f}
- MAE: {fourier_eval.mae:.6f}
- Bin accuracy: {fourier_eval.bin_accuracy:.6f}
- Avg hits: {fourier_eval.hits:.6f}
- Params: `{json.dumps(fourier_best, sort_keys=True)}`

## Best Random Forest Configuration

- Score: {rf_eval.score:.6f}
- MAE: {rf_eval.mae:.6f}
- Bin accuracy: {rf_eval.bin_accuracy:.6f}
- Avg hits: {rf_eval.hits:.6f}
- Params: `{json.dumps(rf_best, sort_keys=True)}`

## Next Drawing Predictions

- Target date: {next_date.strftime('%Y-%m-%d')}
- Fourier numbers: {fourier_pred}
- Fourier sorted: {sorted(fourier_pred)}
- Random Forest numbers: {rf_pred}
- Random Forest sorted: {sorted(rf_pred)}
"""
    summary_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch loop for Powerball models")
    parser.add_argument("--data", default="powerball_games_only.csv", help="Input CSV dataset")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per model")
    parser.add_argument("--eval-draws", type=int, default=120, help="Walk-forward evaluation horizon")
    parser.add_argument("--output-dir", default=".", help="Directory for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--objective",
        default="balanced",
        choices=["balanced", "bin_focus"],
        help="Scoring objective used for keep/discard decisions",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(Path(args.data))
    print(f"Loaded {len(df)} rows from {args.data}")

    fourier_baseline = {
        "harmonics": 15,
        "n_bins": 6,
        "ridge_alpha": 0.01,
    }
    rf_baseline = {
        "sequence_length": 15,
        "n_bins": 6,
        "n_estimators": 120,
        "max_depth": 20,
        "min_samples_split": 5,
    }

    tsv_path = output_dir / "autoresearch_results.tsv"

    fourier_best, fourier_eval = run_model_loop(
        df,
        "fourier",
        fourier_baseline,
        iterations=args.iterations,
        eval_draws=args.eval_draws,
        tsv_path=tsv_path,
        random_state=args.seed,
        objective=args.objective,
    )
    rf_best, rf_eval = run_model_loop(
        df,
        "random_forest",
        rf_baseline,
        iterations=args.iterations,
        eval_draws=args.eval_draws,
        tsv_path=tsv_path,
        random_state=args.seed,
        objective=args.objective,
    )

    next_date = next_drawing_date(df["date"].max())
    fourier_pred = predict_next_numbers(df, "fourier", fourier_best, args.seed)
    rf_pred = predict_next_numbers(df, "random_forest", rf_best, args.seed)

    best_configs = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": args.objective,
        "fourier": {"params": fourier_best, "metrics": fourier_eval.__dict__},
        "random_forest": {"params": rf_best, "metrics": rf_eval.__dict__},
    }
    (output_dir / "autoresearch_best_configs.json").write_text(
        json.dumps(best_configs, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    pred_df = pd.DataFrame(
        [
            {
                "target_date": next_date.strftime("%Y-%m-%d"),
                "model": "fourier_autoresearch",
                **{f"ball_{i}": fourier_pred[i - 1] for i in range(1, 6)},
                "sorted": str(sorted(fourier_pred)),
            },
            {
                "target_date": next_date.strftime("%Y-%m-%d"),
                "model": "random_forest_autoresearch",
                **{f"ball_{i}": rf_pred[i - 1] for i in range(1, 6)},
                "sorted": str(sorted(rf_pred)),
            },
        ]
    )
    pred_df.to_csv(output_dir / "autoresearch_predictions.csv", index=False)

    write_summary(
        output_dir,
        args.objective,
        fourier_best,
        rf_best,
        fourier_eval,
        rf_eval,
        next_date,
        fourier_pred,
        rf_pred,
    )

    print("\nAutoresearch run complete.")
    print(f"Results TSV: {tsv_path}")
    print(f"Best configs: {output_dir / 'autoresearch_best_configs.json'}")
    print(f"Predictions: {output_dir / 'autoresearch_predictions.csv'}")


if __name__ == "__main__":
    main()
