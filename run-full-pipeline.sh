#!/bin/bash
# Master pipeline: refresh data, regenerate every model's predictions, run the
# autoresearch loop over all model families, and produce the honest evaluation
# report. GPU-accelerated (RTX) where the model supports it.
#
# Env overrides:
#   ITERATIONS, HEAVY_ITERATIONS, EVAL_DRAWS, OBJECTIVES, SEED, DEVICE, MODELS
#   EVAL_HARNESS_DRAWS, REFIT_EVERY, RF_BACKEND (random_forest|xgboost)
#   SKIP_REFRESH=1 to reuse existing data
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PY=venv-powerball/bin/python
[ -x "$PY" ] || PY=python3

SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EVAL_HARNESS_DRAWS="${EVAL_HARNESS_DRAWS:-200}"
REFIT_EVERY="${REFIT_EVERY:-10}"

echo "======================================================================="
echo "POWERBALL FULL PIPELINE  ($(date))"
echo "Device: $DEVICE | autoresearch creates a timestamped results_* dir"
echo "======================================================================="

# 1) Refresh data ------------------------------------------------------------
if [ "${SKIP_REFRESH:-0}" != "1" ]; then
    echo; echo "[1/5] Refreshing data from Florida Lottery PDF..."
    "$PY" parse_powerball_pdf.py
else
    echo; echo "[1/5] SKIP_REFRESH=1 -> reusing existing data"
fi

# 2) Standalone single-model predictions ------------------------------------
echo; echo "[2/5] Fourier (numbers) prediction..."
"$PY" predict_powerball_fourier.py

echo; echo "[2/5] Fourier (bins) prediction..."
"$PY" predict_powerball_fourier_bins.py

echo; echo "[2/5] Tree-bins prediction (backend=${RF_BACKEND:-random_forest})..."
RF_BACKEND="${RF_BACKEND:-random_forest}" "$PY" predict_powerball_bins_gpu_enhanced.py

echo; echo "[2/5] TimesFM 2.5 foundation-model prediction..."
"$PY" predict_powerball_timesfm.py

# 3) Autoresearch over all model families -----------------------------------
echo; echo "[3/5] Autoresearch loop (all families, both objectives)..."
ITERATIONS="${ITERATIONS:-20}" HEAVY_ITERATIONS="${HEAVY_ITERATIONS:-12}" \
EVAL_DRAWS="${EVAL_DRAWS:-120}" OBJECTIVES="${OBJECTIVES:-balanced,bin_focus}" \
SEED="$SEED" DEVICE="$DEVICE" MODELS="${MODELS:-fourier,random_forest,dirichlet,gradient_boosting,neural,timesfm}" \
    bash run-autoresearch-powerball.sh

# run-autoresearch-powerball.sh creates its own results_* dir; capture the latest.
AR_DIR="$(ls -dt results_* | head -1)"

# 4) Honest evaluation harness ----------------------------------------------
echo; echo "[4/5] Evaluation harness (baselines, negative control, calibration)..."
BEST_CONFIGS=""
if [ -f "$AR_DIR/balanced/autoresearch_best_configs.json" ]; then
    BEST_CONFIGS="--best-configs $AR_DIR/balanced/autoresearch_best_configs.json"
fi
"$PY" evaluation_harness.py \
    --data powerball_games_only.csv \
    --eval-draws "$EVAL_HARNESS_DRAWS" \
    --refit-every "$REFIT_EVERY" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $BEST_CONFIGS \
    --output-dir "$AR_DIR"

# 5) Done --------------------------------------------------------------------
echo; echo "[5/5] Pipeline complete."
echo "Artifacts in: $AR_DIR"
echo "  - OBJECTIVE_COMPARISON.md      (autoresearch best per family/objective)"
echo "  - <objective>/AUTORESEARCH_SUMMARY.md"
echo "  - <objective>/autoresearch_predictions.csv"
echo "  - EVALUATION_REPORT.md         (honest baseline comparison)"
echo "  - evaluation_metrics.csv"
echo "Standalone predictions:"
echo "  - powerball_predictions.csv, fourier_bin_predictions.csv"
echo "  - enhanced_random_forest_predictions.csv"
echo "  - historical_predictions.csv (tracker)"
