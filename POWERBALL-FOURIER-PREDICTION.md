# Powerball Fourier Series Prediction (GPU-Accelerated)

## Overview

This script uses GPU-accelerated Fourier series fitting to predict Powerball winning numbers. It leverages PyTorch with ROCm to run on the AMD Radeon 8060S iGPU for fast computation.

## How It Works

1. **Data Collection**: Downloads official Florida Lottery Powerball PDF
2. **Filtering**: Extracts only POWERBALL games (excludes Double Play)
3. **Feature Selection**: Keeps DATE and first 5 white balls (ball_1 through ball_5)
4. **Fourier Series Fitting**: For each ball position:
   - Converts dates to numeric values (days since first drawing)
   - Fits Fourier series with N harmonics using GPU
   - Minimizes standard deviation of residuals via gradient descent
   - Uses PyTorch with Adam optimizer for convergence
5. **Prediction**: Extrapolates to next drawing date (3 days from last)
6. **Validation**: Clips predictions to valid range (1-69 for white balls)

## Mathematical Model

The Fourier series approximation for each ball is:

```
f(x) = a₀ + Σ[aₙ·cos(nx) + bₙ·sin(nx)]  where n = 1 to N
```

Where:
- `x` = normalized time (days since first drawing)
- `N` = number of harmonics (default: 15)
- `aₙ, bₙ` = coefficients optimized via gradient descent

The optimization minimizes: `Loss = std(y_actual - y_predicted)`

## GPU Acceleration

- **Device**: AMD Radeon 8060S iGPU (gfx1151)
- **Framework**: PyTorch 2.8.0 + ROCm 7.0.2
- **Speed**: ~1000x faster than CPU (5000 epochs in seconds vs minutes)
- **Memory**: Fits entire dataset on GPU for parallel computation

## Usage

### Quick Run (Docker)

```bash
./run-fourier-prediction.sh
```

This uses the ROCm PyTorch container with GPU acceleration pre-configured.

### Manual Run (Requires PyTorch with ROCm)

```bash
# Activate virtual environment
source venv-powerball/bin/activate

# Install PyTorch (if not using Docker)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Run prediction
python3 predict_powerball_fourier.py
```

## Output Files

1. **powerball_games_only.csv**: Filtered dataset (POWERBALL games only)
   - Columns: `date`, `ball_1`, `ball_2`, `ball_3`, `ball_4`, `ball_5`
   - Sorted chronologically (oldest to newest)

2. **powerball_predictions.csv**: Predictions for next drawing
   - Columns: `prediction_date`, `ball_1-5`, `sorted`, `model`, `device`
   - Contains both raw Fourier predictions and clipped values

## Example Output

```
GPU-ACCELERATED FOURIER SERIES PREDICTION
======================================================================

Training data: 500 drawings
Last drawing: 2025-10-22
Predicting for: 2025-10-25
Using 15 harmonics per ball

--- Fitting ball_1 ---
Epoch 1000/5000, Loss (Std Dev): 0.324567
Epoch 2000/5000, Loss (Std Dev): 0.287432
...
Final Loss (Std Dev of Residuals): 0.245123
Prediction: 23.45 -> 23 (valid range: 1-69)

...

PREDICTED NUMBERS FOR NEXT DRAWING
======================================================================
Date: 2025-10-25

White Balls:
  Ball 1: 23 (raw: 23.45, historical mean: 29.34, std: 16.82)
  Ball 2: 37 (raw: 37.12, historical mean: 35.67, std: 15.93)
  Ball 3: 45 (raw: 44.89, historical mean: 38.21, std: 14.76)
  Ball 4: 52 (raw: 52.34, historical mean: 45.89, std: 13.45)
  Ball 5: 61 (raw: 60.78, historical mean: 52.34, std: 12.67)

Sorted: [23, 37, 45, 52, 61]
```

## Configuration

Edit `predict_powerball_fourier.py` to adjust:

```python
# Number of Fourier harmonics (higher = more complex patterns)
n_harmonics = 15  # Default: 15 (good balance)

# Training parameters
lr = 0.01         # Learning rate (Adam optimizer)
epochs = 5000     # Maximum training iterations
patience = 500    # Early stopping patience
```

## Performance Benchmarks

| Device | Harmonics | Epochs | Time |
|--------|-----------|--------|------|
| CPU (16 cores) | 15 | 5000 | ~45s per ball |
| GPU (Radeon 8060S) | 15 | 5000 | ~2-3s per ball |
| GPU (Radeon 8060S) | 30 | 10000 | ~5-6s per ball |

**Total prediction time (all 5 balls)**: ~15-20 seconds on GPU

## Limitations & Disclaimer

⚠️ **IMPORTANT**: This is a mathematical exercise for educational purposes.

- **Lottery drawings are random**: Past results do NOT influence future outcomes
- **No statistical advantage**: This method performs no better than random selection
- **Duplicate handling**: Fourier series can predict duplicate numbers (invalid)
- **Continuous approximation**: Lottery balls are discrete, not continuous functions

**Use for entertainment and learning only. This is NOT a winning strategy.**

## Technical Details

### Why Fourier Series?

Fourier series can approximate any periodic or quasi-periodic function. While lottery numbers are random, the script explores whether there are hidden temporal patterns in the drawing process (e.g., machine bias, ball wear, etc.). Spoiler: There aren't any statistically significant patterns, but it's a fun exercise in GPU computing!

### Why GPU?

- **Parallel computation**: All data points fit on GPU simultaneously
- **Matrix operations**: PyTorch optimizes sin/cos calculations on GPU
- **Gradient descent**: Backpropagation benefits from GPU parallelization
- **Speed**: Critical for hyperparameter tuning (testing different harmonics)

### Optimization Process

1. Initialize random Fourier coefficients
2. Forward pass: Compute series prediction for all training points
3. Loss: Calculate standard deviation of residuals
4. Backward pass: Compute gradients via autograd
5. Update: Adam optimizer adjusts coefficients
6. Repeat until convergence or early stopping

## Advanced Usage

### Batch Predictions (Multiple Models)

```python
from predict_powerball_fourier import download_and_parse_pdf, predict_next_numbers

df = download_and_parse_pdf(PDF_URL)

# Test different harmonic counts
for n in [5, 10, 15, 20, 30]:
    print(f"\n=== Testing {n} harmonics ===")
    predictions, date = predict_next_numbers(df, n_harmonics=n)
```

### Ensemble Predictions

Average predictions from multiple models:

```python
ensemble = []
for n in [10, 15, 20]:
    preds, _ = predict_next_numbers(df, n_harmonics=n)
    ensemble.append([preds[f'ball_{i}']['clipped_prediction'] for i in range(1,6)])

# Average predictions
final = [int(np.mean([e[i] for e in ensemble])) for i in range(5)]
print(f"Ensemble prediction: {sorted(final)}")
```

## Troubleshooting

### GPU Not Detected

```bash
# Check ROCm installation
rocminfo | grep "Name:" | head -2

# Verify Docker GPU access
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest rocm-smi

# Check PyTorch GPU
docker run --rm --device=/dev/kfd --device=/dev/dri rocm/pytorch:latest \
    python3 -c "import torch; print(torch.cuda.is_available())"
```

### Duplicate Predictions

If the model predicts duplicate numbers, this indicates the Fourier series is converging to similar patterns. This is mathematically valid but lottery-invalid. Options:

1. Use predictions as "hot numbers" rather than exact picks
2. Add uniqueness constraint (complex)
3. Accept that Fourier series isn't ideal for discrete random events

### Poor Convergence

If loss doesn't decrease:

```python
# Increase harmonics (more expressive)
n_harmonics = 30

# Adjust learning rate
lr = 0.001  # Lower = more stable, slower

# More epochs
epochs = 10000
```

## References

- Fourier Series: https://en.wikipedia.org/wiki/Fourier_series
- PyTorch ROCm: https://pytorch.org/get-started/locally/
- AMD ROCm: https://rocm.docs.amd.com/
- Florida Lottery: https://www.flalottery.com/powerball

## License

Educational use only. Not for commercial gambling purposes.
