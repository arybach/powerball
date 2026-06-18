#!/usr/bin/env python3
"""
Autonomous experiment loop for Powerball prediction models.

Implements an autoresearch-style workflow over a configurable set of model
families, all expressed in a common per-position *bin-prediction* framework:

  fourier            -- ridge-fit Fourier series on each position's bin sequence
  random_forest      -- RandomForest on lagged bins
  dirichlet          -- Bayesian Dirichlet-multinomial with optional recency
  gradient_boosting  -- XGBoost on lagged bins (GPU)
  neural             -- LSTM / GRU / Transformer over the bin sequence (GPU)
  ensemble           -- number-level average of the per-family best models

For each family:
- run a baseline configuration first
- mutate hyperparameters in a loop
- evaluate with walk-forward backtesting
- keep or discard configs based on objective score
- log every experiment to TSV

Evaluation protocol notes
-------------------------
Cheap families (fourier / random_forest / dirichlet) are refit at every
walk-forward step on an expanding window (bin edges recomputed each step).
Expensive GPU families (gradient_boosting / neural) use a *train-once* protocol:
bin edges and model weights are fixed from the pre-evaluation window and the
model is rolled forward over the held-out tail. Both are valid out-of-sample
backtests; the unified, fully apples-to-apples comparison lives in
``evaluation_harness.py``.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from powerball_extra_models import DirichletModel, make_model


WHITE_BALL_COLUMNS = [f"ball_{i}" for i in range(1, 6)]

# Families refit each step (expanding window) vs. trained once before the tail.
# (TimesFM is zero-shot, so "train once" just stores context.)
STEPWISE_MODELS = {"fourier", "random_forest", "dirichlet"}
TRAINONCE_MODELS = {"gradient_boosting", "neural", "timesfm"}
BASE_FAMILIES = ["fourier", "random_forest", "dirichlet", "gradient_boosting", "neural"]
# timesfm is selectable via --models but kept out of the lightweight default.
SELECTABLE_FAMILIES = BASE_FAMILIES + ["timesfm"]


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


# --------------------------------------------------------------------------- #
# Low-level per-position bin predictors (stepwise families)
# --------------------------------------------------------------------------- #
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


def stepwise_predict_bin(model_type: str, train_bins: np.ndarray, config: Dict, random_state: int) -> float:
    """Point bin prediction for the next draw from an expanding-window history."""
    if model_type == "fourier":
        return fourier_predict_next_bin(
            train_bins,
            next_index=len(train_bins),
            harmonics=int(config["harmonics"]),
            ridge_alpha=float(config["ridge_alpha"]),
        )
    if model_type == "random_forest":
        return random_forest_predict_next_bin(
            train_bins,
            seq_len=int(config["sequence_length"]),
            n_estimators=int(config["n_estimators"]),
            max_depth=int(config["max_depth"]),
            min_samples_split=int(config["min_samples_split"]),
            random_state=random_state,
        )
    if model_type == "dirichlet":
        m = DirichletModel(
            n_bins=int(config["n_bins"]),
            alpha_prior=float(config["alpha_prior"]),
            recency_halflife=float(config["recency_halflife"]),
        )
        m.fit(train_bins)
        return m.predict_point()
    raise ValueError(f"Not a stepwise model: {model_type}")


# --------------------------------------------------------------------------- #
# Walk-forward prediction (returns per-step numbers/bins for any family)
# --------------------------------------------------------------------------- #
def walk_forward(
    df: pd.DataFrame,
    model_type: str,
    config: Dict,
    eval_draws: int,
    random_state: int,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    n = len(df)
    start = max(120, n - eval_draws)
    if start >= n:
        raise ValueError("Not enough rows to evaluate")
    n_bins = int(config["n_bins"])
    steps = n - start

    pred_nums = np.zeros((steps, 5), dtype=int)
    actual_nums = np.zeros((steps, 5), dtype=int)
    pred_bins = np.zeros((steps, 5), dtype=int)
    actual_bins = np.zeros((steps, 5), dtype=int)

    if model_type in TRAINONCE_MODELS:
        train0 = df.iloc[:start]
        edges = compute_bin_edges(train0, n_bins)
        models = {}
        full_bins = {}
        for col in WHITE_BALL_COLUMNS:
            fb = to_bins(df[col].to_numpy(), edges[col], n_bins)
            full_bins[col] = fb
            m = make_model(model_type, random_state=random_state, device=device, **config)
            m.fit(fb[:start])
            models[col] = m
        for s, ti in enumerate(range(start, n)):
            for j, col in enumerate(WHITE_BALL_COLUMNS):
                context = full_bins[col][:ti]
                pb = models[col].predict_point(context)
                pbr = int(np.clip(np.round(pb), 1, n_bins))
                pred_bins[s, j] = pbr
                pred_nums[s, j] = bin_to_number(pbr, edges[col], n_bins)
                actual_bins[s, j] = int(full_bins[col][ti])
                actual_nums[s, j] = int(df[col].iloc[ti])
    else:
        for s, ti in enumerate(range(start, n)):
            train_df = df.iloc[:ti]
            target_row = df.iloc[ti]
            edges = compute_bin_edges(train_df, n_bins)
            for j, col in enumerate(WHITE_BALL_COLUMNS):
                train_bins = to_bins(train_df[col].to_numpy(), edges[col], n_bins)
                pb = stepwise_predict_bin(model_type, train_bins, config, random_state)
                pbr = int(np.clip(np.round(pb), 1, n_bins))
                pred_bins[s, j] = pbr
                pred_nums[s, j] = bin_to_number(pbr, edges[col], n_bins)
                actual_bins[s, j] = int(to_bins(np.array([target_row[col]]), edges[col], n_bins)[0])
                actual_nums[s, j] = int(target_row[col])

    return {
        "pred_nums": pred_nums,
        "actual_nums": actual_nums,
        "pred_bins": pred_bins,
        "actual_bins": actual_bins,
    }


def metrics_from_arrays(arrays: Dict[str, np.ndarray], objective: str) -> EvalResult:
    pred_nums = arrays["pred_nums"]
    actual_nums = arrays["actual_nums"]
    mae = float(np.mean(np.abs(actual_nums - pred_nums)))
    bin_accuracy = float(np.mean(arrays["pred_bins"] == arrays["actual_bins"]))
    hits = float(np.mean([len(set(p) & set(a)) for p, a in zip(pred_nums, actual_nums)]))
    score = score_metrics(mae, bin_accuracy, hits, objective)
    return EvalResult(score, mae, bin_accuracy, hits, objective, "ok", "evaluated")


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
    device: str = "cuda",
) -> EvalResult:
    arrays = walk_forward(df, model_type, config, eval_draws, random_state, device)
    return metrics_from_arrays(arrays, objective)


def evaluate_ensemble(
    df: pd.DataFrame,
    members: List[Tuple[str, Dict]],
    eval_draws: int,
    random_state: int,
    objective: str,
    device: str = "cuda",
) -> EvalResult:
    """Number-level average of each member's per-step prediction."""
    member_arrays = [walk_forward(df, fam, cfg, eval_draws, random_state, device) for fam, cfg in members]
    actual_nums = member_arrays[0]["actual_nums"]
    stacked = np.stack([a["pred_nums"] for a in member_arrays], axis=0)  # (M, steps, 5)
    ens_nums = np.clip(np.round(stacked.mean(axis=0)), 1, 69).astype(int)

    # Bin accuracy under a fixed 6-bin scheme from the pre-eval window.
    n = len(df)
    start = max(120, n - eval_draws)
    edges = compute_bin_edges(df.iloc[:start], 6)
    ens_bins = np.zeros_like(ens_nums)
    act_bins = np.zeros_like(ens_nums)
    for j, col in enumerate(WHITE_BALL_COLUMNS):
        ens_bins[:, j] = to_bins(ens_nums[:, j], edges[col], 6)
        act_bins[:, j] = to_bins(actual_nums[:, j], edges[col], 6)

    return metrics_from_arrays(
        {"pred_nums": ens_nums, "actual_nums": actual_nums, "pred_bins": ens_bins, "actual_bins": act_bins},
        objective,
    )


# --------------------------------------------------------------------------- #
# Mutation operators
# --------------------------------------------------------------------------- #
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


def mutate_dirichlet_config(best: Dict, rng: np.random.Generator) -> Dict:
    c = dict(best)
    c["n_bins"] = int(np.clip(c["n_bins"] + rng.integers(-1, 2), 4, 10))
    c["alpha_prior"] = float(np.clip(c["alpha_prior"] * rng.choice([0.5, 0.8, 1.25, 2.0]), 1e-3, 50.0))
    step = rng.choice([0.0, 50.0, 100.0, 200.0, -50.0, -100.0])
    c["recency_halflife"] = float(np.clip(c["recency_halflife"] + step, 0.0, 1000.0))
    return c


def mutate_gbm_config(best: Dict, rng: np.random.Generator) -> Dict:
    c = dict(best)
    c["seq_len"] = int(np.clip(c["seq_len"] + rng.integers(-4, 5), 5, 40))
    c["n_bins"] = int(np.clip(c["n_bins"] + rng.integers(-1, 2), 4, 10))
    c["n_estimators"] = int(np.clip(c["n_estimators"] + rng.integers(-60, 61), 50, 500))
    c["max_depth"] = int(np.clip(c["max_depth"] + rng.integers(-2, 3), 2, 12))
    c["learning_rate"] = float(np.clip(c["learning_rate"] * rng.choice([0.5, 0.7, 1.3, 1.8]), 0.01, 0.5))
    c["subsample"] = float(np.clip(c["subsample"] + rng.choice([-0.1, 0.0, 0.1]), 0.5, 1.0))
    return c


def mutate_neural_config(best: Dict, rng: np.random.Generator) -> Dict:
    c = dict(best)
    if rng.random() < 0.3:
        c["arch"] = str(rng.choice(["lstm", "gru", "transformer"]))
    c["seq_len"] = int(np.clip(c["seq_len"] + rng.integers(-5, 6), 6, 40))
    c["n_bins"] = int(np.clip(c["n_bins"] + rng.integers(-1, 2), 4, 10))
    c["hidden"] = int(rng.choice([32, 64, 96, 128]))
    c["layers"] = int(rng.choice([1, 2]))
    c["epochs"] = int(np.clip(c["epochs"] + rng.integers(-20, 21), 20, 150))
    c["lr"] = float(np.clip(c["lr"] * rng.choice([0.5, 0.7, 1.3, 2.0]), 1e-3, 5e-2))
    return c


def mutate_timesfm_config(best: Dict, rng: np.random.Generator) -> Dict:
    # TimesFM is zero-shot; the only knobs are the bin scheme and context length.
    c = dict(best)
    c["context_len"] = int(rng.choice([256, 512, 1024, 2048]))
    c["n_bins"] = int(np.clip(c["n_bins"] + rng.integers(-1, 2), 4, 10))
    return c


MUTATORS: Dict[str, Callable] = {
    "fourier": mutate_fourier_config,
    "random_forest": mutate_rf_config,
    "dirichlet": mutate_dirichlet_config,
    "gradient_boosting": mutate_gbm_config,
    "neural": mutate_neural_config,
    "timesfm": mutate_timesfm_config,
}

BASELINES: Dict[str, Dict] = {
    "fourier": {"harmonics": 15, "n_bins": 6, "ridge_alpha": 0.01},
    "random_forest": {"sequence_length": 15, "n_bins": 6, "n_estimators": 120, "max_depth": 20, "min_samples_split": 5},
    "dirichlet": {"n_bins": 6, "alpha_prior": 1.0, "recency_halflife": 0.0},
    "gradient_boosting": {"n_bins": 6, "seq_len": 15, "n_estimators": 200, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.9},
    "neural": {"n_bins": 6, "seq_len": 20, "arch": "lstm", "hidden": 64, "layers": 1, "dropout": 0.1, "epochs": 60, "lr": 0.01},
    "timesfm": {"n_bins": 6, "context_len": 512},
}

# Per-family RNG offsets so mutation streams differ but stay reproducible.
_RNG_OFFSET = {"fourier": 11, "random_forest": 29, "dirichlet": 43, "gradient_boosting": 67, "neural": 89, "timesfm": 101}


# --------------------------------------------------------------------------- #
# Next-draw prediction for the chosen best config
# --------------------------------------------------------------------------- #
def predict_next_numbers(df: pd.DataFrame, model_type: str, config: Dict, random_state: int, device: str = "cuda") -> List[int]:
    n_bins = int(config["n_bins"])
    edges = compute_bin_edges(df, n_bins)
    predicted: List[int] = []

    if model_type in TRAINONCE_MODELS:
        for col in WHITE_BALL_COLUMNS:
            full_bins = to_bins(df[col].to_numpy(), edges[col], n_bins)
            m = make_model(model_type, random_state=random_state, device=device, **config)
            m.fit(full_bins)
            pbr = int(np.clip(np.round(m.predict_point()), 1, n_bins))
            predicted.append(bin_to_number(pbr, edges[col], n_bins))
    else:
        for col in WHITE_BALL_COLUMNS:
            train_bins = to_bins(df[col].to_numpy(), edges[col], n_bins)
            pb = stepwise_predict_bin(model_type, train_bins, config, random_state)
            pbr = int(np.clip(np.round(pb), 1, n_bins))
            predicted.append(bin_to_number(pbr, edges[col], n_bins))

    return predicted


def predict_next_ensemble(df: pd.DataFrame, members: List[Tuple[str, Dict]], random_state: int, device: str = "cuda") -> List[int]:
    preds = np.array([predict_next_numbers(df, fam, cfg, random_state, device) for fam, cfg in members], dtype=float)
    return [int(x) for x in np.clip(np.round(preds.mean(axis=0)), 1, 69).astype(int)]


# --------------------------------------------------------------------------- #
# TSV logging + experiment loop
# --------------------------------------------------------------------------- #
def append_result_tsv(tsv_path: Path, row: Dict) -> None:
    header = [
        "timestamp", "objective", "model", "experiment", "score", "mae",
        "bin_accuracy", "avg_hits", "status", "description", "params_json",
    ]
    if not tsv_path.exists():
        tsv_path.write_text("\t".join(header) + "\n", encoding="utf-8")

    values = [
        row["timestamp"], row["objective"], row["model"], row["experiment"],
        f"{row['score']:.6f}", f"{row['mae']:.6f}", f"{row['bin_accuracy']:.6f}",
        f"{row['avg_hits']:.6f}", row["status"], row["description"],
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
    device: str = "cuda",
) -> Tuple[Dict, EvalResult]:
    rng = np.random.default_rng(random_state + _RNG_OFFSET.get(model_type, 7))
    mutate_fn = MUTATORS[model_type]

    print(f"\n=== {model_type.upper()} LOOP ({objective}, {iterations} iters) ===")

    best_config = dict(baseline)
    best_eval = evaluate_config(df, model_type, best_config, eval_draws, random_state, objective, device)
    append_result_tsv(
        tsv_path,
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "objective": objective, "model": model_type, "experiment": "baseline",
            "score": best_eval.score, "mae": best_eval.mae,
            "bin_accuracy": best_eval.bin_accuracy, "avg_hits": best_eval.hits,
            "status": "keep", "description": "baseline", "params": best_config,
        },
    )
    print(
        f"baseline score={best_eval.score:.4f} mae={best_eval.mae:.4f} "
        f"bin_acc={best_eval.bin_accuracy:.4f} hits={best_eval.hits:.4f}"
    )

    for i in range(1, iterations + 1):
        candidate = mutate_fn(best_config, rng)
        try:
            ev = evaluate_config(df, model_type, candidate, eval_draws, random_state, objective, device)
        except Exception as exc:  # a bad mutation shouldn't kill the loop
            print(f"iter {i:02d}: error ({exc}); skipping")
            continue
        keep = ev.score < best_eval.score
        status = "keep" if keep else "discard"

        append_result_tsv(
            tsv_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "objective": objective, "model": model_type, "experiment": f"iter_{i}",
                "score": ev.score, "mae": ev.mae, "bin_accuracy": ev.bin_accuracy,
                "avg_hits": ev.hits, "status": status,
                "description": f"mutated from current best, iter {i}", "params": candidate,
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


def write_summary(output_dir: Path, objective: str, results: Dict[str, Tuple[Dict, EvalResult]],
                  next_date: pd.Timestamp, predictions: Dict[str, List[int]]) -> None:
    lines = [
        "# Powerball Autoresearch Summary", "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}", "",
        f"Objective: {objective}", "",
        f"Target draw date: {next_date.strftime('%Y-%m-%d')}", "",
        "## Best configuration per family", "",
        "| Family | Score | MAE | Bin acc | Avg hits | Prediction |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for fam, (cfg, ev) in results.items():
        pred = sorted(predictions.get(fam, []))
        lines.append(
            f"| {fam} | {ev.score:.4f} | {ev.mae:.4f} | {ev.bin_accuracy:.4f} | {ev.hits:.4f} | {pred} |"
        )
    lines += ["", "## Best params", ""]
    for fam, (cfg, ev) in results.items():
        lines.append(f"- **{fam}**: `{json.dumps(cfg, sort_keys=True)}`")
    (output_dir / "AUTORESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoresearch loop for Powerball models")
    parser.add_argument("--data", default="powerball_games_only.csv", help="Input CSV dataset")
    parser.add_argument("--iterations", type=int, default=10, help="Iterations per (cheap) model")
    parser.add_argument("--heavy-iterations", type=int, default=None,
                        help="Iterations for GPU models (gbm/neural); defaults to max(5, iterations//2)")
    parser.add_argument("--eval-draws", type=int, default=120, help="Walk-forward evaluation horizon")
    parser.add_argument("--output-dir", default=".", help="Directory for artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cuda", help="Torch/XGBoost device (cuda or cpu)")
    parser.add_argument("--models", default=",".join(BASE_FAMILIES),
                        help="Comma-separated subset of: " + ", ".join(SELECTABLE_FAMILIES))
    parser.add_argument("--no-ensemble", action="store_true", help="Skip the stacking ensemble")
    parser.add_argument(
        "--objective", default="balanced", choices=["balanced", "bin_focus"],
        help="Scoring objective used for keep/discard decisions",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    families = [f.strip() for f in args.models.split(",") if f.strip()]
    unknown = [f for f in families if f not in SELECTABLE_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown model families: {unknown}")
    heavy_iters = args.heavy_iterations if args.heavy_iterations is not None else max(5, args.iterations // 2)

    df = load_dataset(Path(args.data))
    print(f"Loaded {len(df)} rows from {args.data}  | families={families} device={args.device}")

    tsv_path = output_dir / "autoresearch_results.tsv"
    results: Dict[str, Tuple[Dict, EvalResult]] = {}
    for fam in families:
        iters = heavy_iters if fam in TRAINONCE_MODELS else args.iterations
        best_config, best_eval = run_model_loop(
            df, fam, BASELINES[fam], iterations=iters, eval_draws=args.eval_draws,
            tsv_path=tsv_path, random_state=args.seed, objective=args.objective, device=args.device,
        )
        results[fam] = (best_config, best_eval)

    next_date = next_drawing_date(df["date"].max())
    predictions: Dict[str, List[int]] = {
        fam: predict_next_numbers(df, fam, cfg, args.seed, args.device) for fam, (cfg, _) in results.items()
    }

    # Stacking ensemble over the per-family best configs.
    if not args.no_ensemble and len(families) >= 2:
        members = [(fam, results[fam][0]) for fam in families]
        print("\n=== ENSEMBLE (stack of per-family bests) ===")
        ens_eval = evaluate_ensemble(df, members, args.eval_draws, args.seed, args.objective, args.device)
        results["ensemble"] = ({"members": families}, ens_eval)
        predictions["ensemble"] = predict_next_ensemble(df, members, args.seed, args.device)
        append_result_tsv(
            tsv_path,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "objective": args.objective, "model": "ensemble", "experiment": "stack",
                "score": ens_eval.score, "mae": ens_eval.mae, "bin_accuracy": ens_eval.bin_accuracy,
                "avg_hits": ens_eval.hits, "status": "keep",
                "description": "number-level average of per-family bests", "params": {"members": families},
            },
        )
        print(
            f"ensemble score={ens_eval.score:.4f} mae={ens_eval.mae:.4f} "
            f"bin_acc={ens_eval.bin_accuracy:.4f} hits={ens_eval.hits:.4f}"
        )

    best_configs = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": args.objective,
        "families": {fam: {"params": cfg, "metrics": ev.__dict__} for fam, (cfg, ev) in results.items()},
    }
    (output_dir / "autoresearch_best_configs.json").write_text(
        json.dumps(best_configs, indent=2, sort_keys=True), encoding="utf-8"
    )

    pred_rows = [
        {
            "target_date": next_date.strftime("%Y-%m-%d"),
            "model": f"{fam}_autoresearch",
            **{f"ball_{i}": predictions[fam][i - 1] for i in range(1, 6)},
            "sorted": str(sorted(predictions[fam])),
        }
        for fam in results
    ]
    pd.DataFrame(pred_rows).to_csv(output_dir / "autoresearch_predictions.csv", index=False)

    write_summary(output_dir, args.objective, results, next_date, predictions)

    print("\nAutoresearch run complete.")
    print(f"Results TSV: {tsv_path}")
    print(f"Best configs: {output_dir / 'autoresearch_best_configs.json'}")
    print(f"Predictions: {output_dir / 'autoresearch_predictions.csv'}")


if __name__ == "__main__":
    main()
