#!/usr/bin/env python3
"""
Honest evaluation harness for the Powerball models.

The point of this module is rigor, not predictions. Powerball draws are i.i.d.
uniform, so the null hypothesis is that *no* model beats chance. This harness is
built to make it hard to fool ourselves:

1. Honest baselines. Every model is compared not just to uniform-random number
   picking, but to the per-position empirical-frequency baseline that captures
   the order-statistics / bin-imbalance structure of the sorted draw. Beating
   uniform-random is trivial and meaningless; beating the marginal baseline is
   the only thing that would matter.

2. One unified protocol for all families. A single walk-forward with a fixed
   refit cadence and fixed bin edges (from the pre-eval window) is applied
   identically to every model, so the comparison is apples-to-apples.

3. Calibration metrics. For models that emit a bin distribution we report
   multiclass Brier score and log-loss against the marginal baseline -- a model
   that is merely re-deriving the marginal will tie it, not beat it.

4. Negative control. We shuffle each position's bin series (destroying any
   temporal structure) and re-evaluate. Any "edge" that survives shuffling was
   never temporal -- it was structure the marginal baseline already has.

5. Bootstrap confidence bands. Avg-hits and MAE differences vs. the marginal
   baseline get a bootstrap 95% CI, and we only call a model a "winner" if the
   interval excludes zero.

Output: EVALUATION_REPORT.md plus evaluation_metrics.csv.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import autoresearch_powerball as ar
from powerball_extra_models import (
    DirichletModel,
    GradientBoostingModel,
    MarginalFrequencyModel,
    NeuralSequenceModel,
)

WHITE = ar.WHITE_BALL_COLUMNS
N_BINS = 6
WHITE_MAX = 69


# --------------------------------------------------------------------------- #
# Model adapters: every evaluated model exposes fit(series)/predict_proba()/
# predict_point() on a single position's bin series. Fourier/RF are wrapped
# around the autoresearch functions so we evaluate the *same* code that the
# tuning loop uses.
# --------------------------------------------------------------------------- #
class _UniformBaseline:
    def __init__(self, n_bins=N_BINS, **_):
        self.n_bins = n_bins

    def fit(self, series):
        return self

    def predict_proba(self, context=None):
        return np.ones(self.n_bins) / self.n_bins

    def predict_point(self, context=None):
        return (self.n_bins + 1) / 2.0


class _FourierAdapter:
    def __init__(self, n_bins=N_BINS, harmonics=15, ridge_alpha=0.01, **_):
        self.n_bins, self.harmonics, self.ridge_alpha = n_bins, harmonics, ridge_alpha
        self._series = np.array([3.0])

    def fit(self, series):
        self._series = np.asarray(series, dtype=float)
        return self

    def predict_point(self, context=None):
        s = self._series if context is None else np.asarray(context, dtype=float)
        return ar.fourier_predict_next_bin(s, len(s), self.harmonics, self.ridge_alpha)

    def predict_proba(self, context=None):  # point model -> soft one-hot around the bin
        pt = np.clip(self.predict_point(context), 1, self.n_bins)
        bins = np.arange(1, self.n_bins + 1)
        p = np.exp(-0.5 * ((bins - pt) / 0.75) ** 2)
        return p / p.sum()


class _RFAdapter:
    def __init__(self, n_bins=N_BINS, sequence_length=15, n_estimators=120, max_depth=20,
                 min_samples_split=5, random_state=42, **_):
        self.n_bins = n_bins
        self.kw = dict(seq_len=sequence_length, n_estimators=n_estimators, max_depth=max_depth,
                       min_samples_split=min_samples_split, random_state=random_state)
        self._series = np.array([3])

    def fit(self, series):
        self._series = np.asarray(series, dtype=int)
        return self

    def predict_point(self, context=None):
        s = self._series if context is None else np.asarray(context, dtype=int)
        return ar.random_forest_predict_next_bin(s, **self.kw)

    def predict_proba(self, context=None):
        pt = np.clip(self.predict_point(context), 1, self.n_bins)
        bins = np.arange(1, self.n_bins + 1)
        p = np.exp(-0.5 * ((bins - pt) / 0.75) ** 2)
        return p / p.sum()


# Builders keyed by family name. Each is constructed fresh per refit.
def _builders(device: str, best_configs: Optional[Dict] = None) -> Dict[str, callable]:
    cfg = best_configs or {}

    def merged(family, defaults):
        d = dict(defaults)
        d.update(cfg.get(family, {}).get("params", {}) if family in cfg else {})
        # The harness evaluates every model on one common bin grid, so the bin
        # count is fixed here regardless of what the autoresearch loop tuned.
        d["n_bins"] = N_BINS
        return d

    return {
        "uniform_baseline": lambda: _UniformBaseline(n_bins=N_BINS),
        "marginal_baseline": lambda: MarginalFrequencyModel(n_bins=N_BINS),
        "fourier": lambda: _FourierAdapter(**merged("fourier", ar.BASELINES["fourier"])),
        "random_forest": lambda: _RFAdapter(**merged("random_forest", ar.BASELINES["random_forest"])),
        "dirichlet": lambda: DirichletModel(**merged("dirichlet", ar.BASELINES["dirichlet"])),
        "gradient_boosting": lambda: GradientBoostingModel(device=device, **merged("gradient_boosting", ar.BASELINES["gradient_boosting"])),
        "neural": lambda: NeuralSequenceModel(device=device, **merged("neural", ar.BASELINES["neural"])),
    }


# --------------------------------------------------------------------------- #
# Unified walk-forward (same protocol for every family)
# --------------------------------------------------------------------------- #
def _bin_series(df: pd.DataFrame, edges: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {col: ar.to_bins(df[col].to_numpy(), edges[col], N_BINS) for col in WHITE}


def walk_forward_eval(
    df: pd.DataFrame,
    builder,
    eval_draws: int,
    refit_every: int,
    shuffle: bool = False,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Roll a model forward over the last ``eval_draws`` draws with a fixed
    bin scheme and a refit cadence. Returns per-step predictions + the bin
    probability assigned to the realised bin (for calibration)."""
    n = len(df)
    start = max(120, n - eval_draws)
    edges = ar.compute_bin_edges(df.iloc[:start], N_BINS)
    full_bins = _bin_series(df, edges)

    if shuffle:  # negative control: destroy temporal order within the train region
        rng = np.random.default_rng(seed)
        full_bins = {c: v.copy() for c, v in full_bins.items()}
        for c in WHITE:
            perm = rng.permutation(start)
            full_bins[c][:start] = full_bins[c][:start][perm]

    steps = n - start
    pred_nums = np.zeros((steps, 5), dtype=int)
    actual_nums = np.zeros((steps, 5), dtype=int)
    pred_bins = np.zeros((steps, 5), dtype=int)
    actual_bins = np.zeros((steps, 5), dtype=int)
    true_bin_proba = np.zeros((steps, 5), dtype=float)
    proba_full = np.zeros((steps, 5, N_BINS), dtype=float)

    models = {col: None for col in WHITE}
    for s, ti in enumerate(range(start, n)):
        if s % refit_every == 0:  # refit on everything strictly before ti
            for col in WHITE:
                models[col] = builder().fit(full_bins[col][:ti])
        for j, col in enumerate(WHITE):
            context = full_bins[col][:ti]
            proba = np.asarray(models[col].predict_proba(context), dtype=float)
            proba = proba / proba.sum()
            pbr = int(np.clip(np.round(models[col].predict_point(context)), 1, N_BINS))
            ab = int(full_bins[col][ti])
            pred_bins[s, j] = pbr
            pred_nums[s, j] = ar.bin_to_number(pbr, edges[col], N_BINS)
            actual_bins[s, j] = ab
            actual_nums[s, j] = int(df[col].iloc[ti])
            true_bin_proba[s, j] = proba[ab - 1]
            proba_full[s, j] = proba

    return {
        "pred_nums": pred_nums, "actual_nums": actual_nums,
        "pred_bins": pred_bins, "actual_bins": actual_bins,
        "true_bin_proba": true_bin_proba, "proba_full": proba_full,
    }


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def point_metrics(a: Dict[str, np.ndarray]) -> Dict[str, float]:
    pred, act = a["pred_nums"], a["actual_nums"]
    per_step_hits = np.array([len(set(p) & set(t)) for p, t in zip(pred, act)], dtype=float)
    return {
        "mae": float(np.mean(np.abs(act - pred))),
        "bin_accuracy": float(np.mean(a["pred_bins"] == a["actual_bins"])),
        "avg_hits": float(per_step_hits.mean()),
        "_per_step_abs": np.abs(act - pred).mean(axis=1),  # per-step MAE for bootstrap
        "_per_step_hits": per_step_hits,
    }


def calibration_metrics(a: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Multiclass Brier and log-loss over bins, averaged across positions/steps."""
    proba = a["proba_full"]  # (steps, 5, n_bins)
    actual_bins = a["actual_bins"]  # 1-based
    onehot = np.zeros_like(proba)
    for s in range(proba.shape[0]):
        for j in range(5):
            onehot[s, j, actual_bins[s, j] - 1] = 1.0
    brier = float(np.mean(np.sum((proba - onehot) ** 2, axis=-1)))
    p_true = np.clip(np.take_along_axis(proba, (actual_bins[..., None] - 1), axis=-1)[..., 0], 1e-12, 1.0)
    logloss = float(-np.mean(np.log(p_true)))
    return {"brier": brier, "log_loss": logloss}


def bootstrap_ci_diff(metric_model: np.ndarray, metric_ref: np.ndarray, n_boot=2000, seed=0,
                      lower_is_better=True) -> Tuple[float, float, float, bool]:
    """Bootstrap 95% CI of mean(model - ref). Returns (diff, lo, hi, beats_ref)."""
    rng = np.random.default_rng(seed)
    diff = metric_model - metric_ref
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mean_diff = float(diff.mean())
    # "beats ref" means improvement direction with CI excluding 0
    beats = (hi < 0) if lower_is_better else (lo > 0)
    return mean_diff, float(lo), float(hi), bool(beats)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def run(df: pd.DataFrame, families: List[str], eval_draws: int, refit_every: int,
        device: str, seed: int, best_configs: Optional[Dict], output_dir: Path) -> None:
    builders = _builders(device, best_configs)
    families = [f for f in families if f in builders]

    print(f"Evaluating {families} over last {eval_draws} draws (refit every {refit_every}).")

    real: Dict[str, Dict] = {}
    shuf: Dict[str, Dict] = {}
    for fam in families:
        print(f"  - {fam} ...", flush=True)
        real[fam] = walk_forward_eval(df, builders[fam], eval_draws, refit_every, shuffle=False, seed=seed)
        shuf[fam] = walk_forward_eval(df, builders[fam], eval_draws, refit_every, shuffle=True, seed=seed)

    ref = point_metrics(real["marginal_baseline"])
    rows = []
    for fam in families:
        pm = point_metrics(real[fam])
        cm = calibration_metrics(real[fam])
        sm = point_metrics(shuf[fam])
        d_hits, lo_h, hi_h, beats_h = bootstrap_ci_diff(
            pm["_per_step_hits"], ref["_per_step_hits"], seed=seed, lower_is_better=False)
        d_mae, lo_m, hi_m, beats_m = bootstrap_ci_diff(
            pm["_per_step_abs"], ref["_per_step_abs"], seed=seed, lower_is_better=True)
        rows.append({
            "family": fam,
            "mae": pm["mae"], "bin_accuracy": pm["bin_accuracy"], "avg_hits": pm["avg_hits"],
            "brier": cm["brier"], "log_loss": cm["log_loss"],
            "shuffled_avg_hits": sm["avg_hits"], "shuffled_mae": sm["mae"],
            "d_hits_vs_marginal": d_hits, "hits_ci_lo": lo_h, "hits_ci_hi": hi_h,
            "d_mae_vs_marginal": d_mae, "mae_ci_lo": lo_m, "mae_ci_hi": hi_m,
            "beats_marginal_hits": beats_h, "beats_marginal_mae": beats_m,
        })

    metrics_df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_dir / "evaluation_metrics.csv", index=False)

    _write_report(metrics_df, df, eval_draws, refit_every, output_dir)
    print(f"\nWrote {output_dir / 'EVALUATION_REPORT.md'}")
    print(f"Wrote {output_dir / 'evaluation_metrics.csv'}")


def _write_report(m: pd.DataFrame, df: pd.DataFrame, eval_draws: int, refit_every: int, output_dir: Path) -> None:
    def fmt(x, n=4):
        return f"{x:.{n}f}"

    lines = [
        "# Powerball Model Evaluation Report", "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Data: {len(df)} draws, {df['date'].min().date()} .. {df['date'].max().date()}",
        f"Protocol: unified walk-forward over the last {eval_draws} draws, "
        f"refit every {refit_every} step(s), fixed {N_BINS}-bin scheme from the pre-eval window.",
        "",
        "## TL;DR",
        "",
        "Powerball is i.i.d. uniform; the honest null is *no model beats chance*. "
        "The bar that matters is the **marginal-frequency baseline** (per-position empirical "
        "bin frequencies), which already captures the sorted-draw order-statistics structure. "
        "A model only demonstrates real signal if it beats the marginal baseline with a "
        "bootstrap 95% CI that excludes zero **and** that edge does not survive the shuffled "
        "negative control.",
        "",
    ]

    winners_hits = m[(m.family.isin(["fourier", "random_forest", "dirichlet", "gradient_boosting", "neural"])) & (m.beats_marginal_hits)]
    winners_mae = m[(m.family.isin(["fourier", "random_forest", "dirichlet", "gradient_boosting", "neural"])) & (m.beats_marginal_mae)]
    if len(winners_hits) == 0 and len(winners_mae) == 0:
        lines += ["**Verdict: no model beats the marginal baseline at the 95% level on either "
                  "avg-hits or MAE.** This is the expected result for a fair lottery — the models "
                  "are re-deriving the draw's structural marginals, not finding temporal signal.", ""]
    else:
        names = sorted(set(winners_hits.family) | set(winners_mae.family))
        lines += [f"**Apparent winners vs. marginal (95% CI excludes 0): {names}.** "
                  "Check the negative-control columns below — if the shuffled edge is similar, "
                  "the 'win' is structural, not temporal.", ""]

    lines += [
        "## Point metrics (lower MAE better; higher bin-acc / avg-hits better)", "",
        "| Family | MAE | Bin acc | Avg hits | Brier | Log-loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in m.iterrows():
        lines.append(f"| {r.family} | {fmt(r.mae)} | {fmt(r.bin_accuracy)} | {fmt(r.avg_hits)} "
                     f"| {fmt(r.brier)} | {fmt(r.log_loss)} |")

    lines += [
        "", "## Negative control (shuffled bin order)", "",
        "If a model has real temporal signal, shuffling the history should degrade it. "
        "Similar real vs. shuffled numbers ⇒ no temporal signal.", "",
        "| Family | Avg hits (real) | Avg hits (shuffled) | MAE (real) | MAE (shuffled) |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, r in m.iterrows():
        lines.append(f"| {r.family} | {fmt(r.avg_hits)} | {fmt(r.shuffled_avg_hits)} "
                     f"| {fmt(r.mae)} | {fmt(r.shuffled_mae)} |")

    lines += [
        "", "## Bootstrap 95% CI vs. marginal baseline", "",
        "`d_hits` = mean(model − marginal) avg-hits per draw (positive favours the model); "
        "`d_mae` = mean(model − marginal) MAE (negative favours the model). "
        "A model 'beats' the baseline only if the whole CI is on the favourable side of 0.", "",
        "| Family | d_hits | hits 95% CI | beats? | d_mae | mae 95% CI | beats? |",
        "|---|---:|---|:--:|---:|---|:--:|",
    ]
    for _, r in m.iterrows():
        lines.append(
            f"| {r.family} | {fmt(r.d_hits_vs_marginal,3)} | [{fmt(r.hits_ci_lo,3)}, {fmt(r.hits_ci_hi,3)}] "
            f"| {'✅' if r.beats_marginal_hits else '—'} "
            f"| {fmt(r.d_mae_vs_marginal,3)} | [{fmt(r.mae_ci_lo,3)}, {fmt(r.mae_ci_hi,3)}] "
            f"| {'✅' if r.beats_marginal_mae else '—'} |")

    lines += [
        "", "## How to read this", "",
        "- **Brier / log-loss**: probabilistic accuracy over bins. The marginal baseline is hard "
        "to beat here by construction; a model that ties it is just re-learning the marginal.",
        "- **uniform_baseline** exists only to show how easy it is to look good: beating uniform "
        "random is trivial and proves nothing.",
        "- **Bin accuracy > 1/n_bins** is expected from bin imbalance alone (sorted balls), not signal.",
        "",
        "_Lottery draws are random. None of this provides a real edge on winning numbers; "
        "the harness exists to demonstrate exactly that, rigorously._",
    ]
    (output_dir / "EVALUATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Honest evaluation harness for Powerball models")
    p.add_argument("--data", default="powerball_games_only.csv")
    p.add_argument("--eval-draws", type=int, default=200)
    p.add_argument("--refit-every", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--best-configs", default=None,
                   help="Optional autoresearch_best_configs.json to evaluate tuned configs")
    p.add_argument("--families", default="uniform_baseline,marginal_baseline,fourier,random_forest,dirichlet,gradient_boosting,neural")
    p.add_argument("--output-dir", default=".")
    args = p.parse_args()

    df = ar.load_dataset(Path(args.data))
    best_configs = None
    if args.best_configs and Path(args.best_configs).exists():
        data = json.loads(Path(args.best_configs).read_text())
        best_configs = data.get("families", data)

    families = [f.strip() for f in args.families.split(",") if f.strip()]
    run(df, families, args.eval_draws, args.refit_every, args.device, args.seed, best_configs, Path(args.output_dir))


if __name__ == "__main__":
    main()
