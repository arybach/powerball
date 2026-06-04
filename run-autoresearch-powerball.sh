#!/bin/bash
set -euo pipefail

echo "======================================================================="
echo "POWERBALL AUTORESEARCH LOOP (FOURIER + RANDOM FOREST)"
echo "======================================================================="
echo "Date: $(date)"
echo ""

if [ -f "venv-powerball/bin/activate" ]; then
    echo "Activating virtual environment..."
    source venv-powerball/bin/activate
fi

RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

ITERATIONS="${ITERATIONS:-30}"
EVAL_DRAWS="${EVAL_DRAWS:-120}"
SEED="${SEED:-42}"
OBJECTIVES="${OBJECTIVES:-balanced,bin_focus}"
HEAVY_ITERATIONS="${HEAVY_ITERATIONS:-12}"
DEVICE="${DEVICE:-cuda}"
MODELS="${MODELS:-fourier,random_forest,dirichlet,gradient_boosting,neural}"

echo "Results directory: $RESULTS_DIR"
echo "Iterations/model: $ITERATIONS (heavy/GPU models: $HEAVY_ITERATIONS)"
echo "Evaluation horizon: $EVAL_DRAWS"
echo "Seed: $SEED"
echo "Device: $DEVICE"
echo "Models: $MODELS"
echo "Objectives: $OBJECTIVES"
echo ""

IFS=',' read -r -a OBJECTIVE_ARRAY <<< "$OBJECTIVES"

for objective in "${OBJECTIVE_ARRAY[@]}"; do
    objective_dir="$RESULTS_DIR/$objective"
    mkdir -p "$objective_dir"

    echo "======================================================================="
    echo "Running objective: $objective"
    echo "Output: $objective_dir"
    echo "======================================================================="

    python3 autoresearch_powerball.py \
        --data powerball_games_only.csv \
        --iterations "$ITERATIONS" \
        --heavy-iterations "$HEAVY_ITERATIONS" \
        --eval-draws "$EVAL_DRAWS" \
        --seed "$SEED" \
        --device "$DEVICE" \
        --models "$MODELS" \
        --objective "$objective" \
        --output-dir "$objective_dir"
done

RESULTS_DIR_ENV="$RESULTS_DIR" python3 - <<'PY'
import json
import os
from pathlib import Path
from datetime import datetime

results_dir = Path(os.environ["RESULTS_DIR_ENV"])
rows = []

for obj_dir in sorted([d for d in results_dir.iterdir() if d.is_dir()]):
    cfg_file = obj_dir / "autoresearch_best_configs.json"
    if not cfg_file.exists():
        continue
    data = json.loads(cfg_file.read_text(encoding="utf-8"))
    objective = data.get("objective", obj_dir.name)
    families = data.get("families", {})
    for fam, info in families.items():
        mtr = info.get("metrics", {})
        rows.append((objective, fam, mtr.get("score", 0.0), mtr.get("mae", 0.0),
                     mtr.get("bin_accuracy", 0.0), mtr.get("hits", 0.0)))

rows.sort(key=lambda x: (x[1], x[2]))

lines = []
lines.append("# Objective Comparison")
lines.append("")
lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
lines.append("")
lines.append("| Objective | Model | Score (lower better) | MAE | Bin Accuracy | Avg Hits |")
lines.append("|---|---|---:|---:|---:|---:|")
for objective, model, score, mae, bin_acc, hits in rows:
    lines.append(f"| {objective} | {model} | {score:.6f} | {mae:.6f} | {bin_acc:.6f} | {hits:.6f} |")

(results_dir / "OBJECTIVE_COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"Wrote {results_dir / 'OBJECTIVE_COMPARISON.md'}")
PY

echo ""
echo "Done. Artifacts:"
echo "  - $RESULTS_DIR/<objective>/autoresearch_results.tsv"
echo "  - $RESULTS_DIR/<objective>/autoresearch_best_configs.json"
echo "  - $RESULTS_DIR/<objective>/autoresearch_predictions.csv"
echo "  - $RESULTS_DIR/<objective>/AUTORESEARCH_SUMMARY.md"
echo "  - $RESULTS_DIR/OBJECTIVE_COMPARISON.md"
