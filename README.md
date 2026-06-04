# Powerball Research Sandbox

This repository is a Powerball data and modeling sandbox focused on:

- Parsing official Florida Lottery Powerball PDFs into tabular data.
- Running a range of prediction experiments: Fourier series, Random Forest,
  gradient boosting (XGBoost), Bayesian Dirichlet-multinomial, neural sequence
  models (LSTM/GRU/Transformer), and a stacking ensemble.
- Tracking and comparing model outputs over time.
- Running autonomous experiment loops inspired by Karpathy's autoresearch pattern.
- Evaluating every model **honestly** against baselines, a negative control,
  calibration metrics, and bootstrap confidence bands — because for a fair
  lottery the correct result is that nothing beats chance.

## Important Disclaimer

Lottery drawings are random events. These models do not provide a proven statistical edge for gambling. This project is for experimentation, data engineering, and model workflow development.

## Repository Components

- Data extraction:
  - parse_powerball_pdf.py
  - run-powerball-parser.sh
- Fourier modeling:
  - predict_powerball_fourier.py
  - predict_powerball_fourier_bins.py
  - run-fourier-prediction.sh
  - run-fourier-bin-prediction.sh
- Tree modeling (Random Forest / Gradient Boosting):
  - predict_powerball_bins_gpu_enhanced.py
    (set `RF_BACKEND=xgboost` to use GPU XGBoost as a stronger drop-in for the RF)
- Additional model families (shared library):
  - powerball_extra_models.py
    - Bayesian Dirichlet-multinomial (`dirichlet`)
    - Gradient boosting on lagged bins (`gradient_boosting`, GPU XGBoost)
    - Neural sequence models LSTM / GRU / Transformer (`neural`, GPU torch)
    - Marginal-frequency baseline and a probability-averaging stacking ensemble
- Honest evaluation harness:
  - evaluation_harness.py
- Comparison and tracking:
  - prediction_tracker.py
  - run-prediction-comparison.sh
  - analyze-prediction-history.sh
  - plot_time_series.py
- Autoresearch workflow (spans all model families + ensemble):
  - autoresearch_powerball.py
  - run-autoresearch-powerball.sh
- One-shot full pipeline:
  - run-full-pipeline.sh

## Setup

Use the provided virtual environment bootstrap:

```bash
./setup-powerball-env.sh
```

Then activate if needed:

```bash
source venv-powerball/bin/activate
```

## Typical Workflows

### 1) Parse and refresh data

```bash
./run-powerball-parser.sh
```

### 2) Run baseline model scripts

```bash
./run-fourier-prediction.sh
./run-fourier-bin-prediction.sh
./run-prediction-comparison.sh
```

### 3) Run autoresearch experiments

```bash
./run-autoresearch-powerball.sh
```

The loop now tunes all model families — `fourier`, `random_forest`, `dirichlet`,
`gradient_boosting`, `neural` — and adds a stacking `ensemble` of the per-family
bests. Optional runtime controls:

```bash
# Subset of families, fewer GPU-model iterations, both objectives
ITERATIONS=30 HEAVY_ITERATIONS=12 EVAL_DRAWS=120 \
MODELS=fourier,dirichlet,gradient_boosting,neural \
OBJECTIVES=balanced,bin_focus DEVICE=cuda ./run-autoresearch-powerball.sh
```

### 4) Honest evaluation (does anything beat chance?)

```bash
python3 evaluation_harness.py --data powerball_games_only.csv \
    --eval-draws 200 --refit-every 10 --device cuda \
    --best-configs results_*/balanced/autoresearch_best_configs.json \
    --output-dir results_eval
```

This runs one unified walk-forward for every family **and** honest baselines
(uniform-random and per-position marginal frequency), a shuffled-history
**negative control**, calibration (Brier / log-loss), and bootstrap 95% CIs vs.
the marginal baseline. It writes `EVALUATION_REPORT.md` + `evaluation_metrics.csv`.
The expected (and observed) result for a fair lottery: **no model beats the
marginal baseline** — the models re-derive the draw's structural marginals, not
temporal signal.

### 5) One-shot full pipeline

Refresh data → all single-model predictions → autoresearch (all families, both
objectives) → evaluation report, GPU-accelerated:

```bash
RF_BACKEND=xgboost DEVICE=cuda ./run-full-pipeline.sh
```

## Explicit Integration with Karpathy autoresearch program.md

This repository includes a direct adaptation of the workflow defined in:

https://github.com/karpathy/autoresearch/blob/master/program.md

The integration is implemented in autoresearch_powerball.py and orchestrated by run-autoresearch-powerball.sh.

The adapted loop follows the same core pattern:

1. Establish baseline run per model family.
2. Mutate configuration for each experiment iteration.
3. Evaluate on a fixed backtest window.
4. Log each run with metrics and parameters to a TSV artifact.
5. Keep or discard changes by objective score.
6. Persist best configurations and next-draw predictions.

### Objective Variants

Two objective modes are supported and can run side-by-side:

- balanced
- bin_focus

Each objective writes separate artifacts under a timestamped results directory and produces a merged OBJECTIVE_COMPARISON report.

## Artifacts

Common generated artifacts include:

- powerball_games_only.csv
- historical_predictions.csv
- results_YYYYMMDD_HHMMSS/<objective>/autoresearch_results.tsv
- results_YYYYMMDD_HHMMSS/<objective>/autoresearch_best_configs.json
  (now keyed under `families.<family>` covering all model families + ensemble)
- results_YYYYMMDD_HHMMSS/<objective>/autoresearch_predictions.csv
- results_YYYYMMDD_HHMMSS/OBJECTIVE_COMPARISON.md
- results_YYYYMMDD_HHMMSS/EVALUATION_REPORT.md (honest baseline comparison)
- results_YYYYMMDD_HHMMSS/evaluation_metrics.csv
- enhanced_random_forest_predictions.csv (tree model; XGBoost when RF_BACKEND=xgboost)

## Notes

- The repository currently keeps a local virtual environment folder (venv-powerball). Third-party package documentation files inside that environment are not project documentation.
- Project documentation has been intentionally consolidated into this single README.md.
