# Powerball Research Sandbox

This repository is a Powerball data and modeling sandbox focused on:

- Parsing official Florida Lottery Powerball PDFs into tabular data.
- Running Fourier-based and Random-Forest-based prediction experiments.
- Tracking and comparing model outputs over time.
- Running autonomous experiment loops inspired by Karpathy's autoresearch pattern.

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
- Random Forest modeling:
  - predict_powerball_bins_gpu_enhanced.py
- Comparison and tracking:
  - prediction_tracker.py
  - run-prediction-comparison.sh
  - analyze-prediction-history.sh
  - plot_time_series.py
- Autoresearch workflow:
  - autoresearch_powerball.py
  - run-autoresearch-powerball.sh

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

Optional runtime controls:

```bash
ITERATIONS=30 EVAL_DRAWS=10 OBJECTIVES=balanced,bin_focus ./run-autoresearch-powerball.sh
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
- results_YYYYMMDD_HHMMSS/<objective>/autoresearch_predictions.csv
- results_YYYYMMDD_HHMMSS/OBJECTIVE_COMPARISON.md

## Notes

- The repository currently keeps a local virtual environment folder (venv-powerball). Third-party package documentation files inside that environment are not project documentation.
- Project documentation has been intentionally consolidated into this single README.md.
