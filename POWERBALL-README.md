# Florida Lottery Powerball PDF Parser

Python script to extract and analyze Florida Lottery Powerball drawing data from the official PDF.

## Installation

Due to Ubuntu's externally-managed Python environment, we use a virtual environment:

```bash
# One-time setup (creates venv-powerball/ directory)
./setup-powerball-env.sh
```

This will create a virtual environment and install all dependencies.

## Usage

### Quick Run

```bash
# Run parser in virtual environment (easiest method)
./run-powerball-parser.sh

# Run Fourier series prediction
./run-fourier-prediction.sh

# Run Random Forest bin classification prediction
./run-bin-prediction.sh

# Run BOTH models and compare predictions (RECOMMENDED!)
./run-prediction-comparison.sh
```

### Manual Run

```bash
# Activate virtual environment
source venv-powerball/bin/activate

# Run parser
python parse_powerball_pdf.py

# Deactivate when done
deactivate
```

This will:
1. Download the latest Powerball PDF from Florida Lottery
2. Extract all drawing data
3. Create a pandas DataFrame
4. Save to CSV and Excel files
5. Display analysis and statistics

### Output Files

#### Data Files
- `powerball_data.csv` - All drawing data in CSV format
- `powerball_data.xlsx` - Excel format (requires openpyxl)
- `powerball_games_only.csv` - Filtered Powerball games (excludes Double Play)
- `/tmp/powerball_raw_text.txt` - Raw extracted text for debugging

#### Prediction Files
- `powerball_predictions.csv` - Fourier series predictions
- `powerball_bin_predictions.csv` - Random Forest bin predictions
- `random_forest_bin_model.joblib` - Trained Random Forest model
- `bin_predictions_visualization.png` - Analysis charts and visualizations
- `results_YYYYMMDD_HHMMSS/` - Comprehensive prediction comparison results

### DataFrame Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Drawing date |
| `ball_1` to `ball_5` | int | Five white ball numbers (1-69) |
| `powerball` | int | Red Powerball number (1-26) |
| `power_play` | str | Power Play multiplier (optional) |
| `jackpot` | str | Jackpot amount (optional) |
| `numbers_sorted` | list | White balls in sorted order |
| `sum_white_balls` | int | Sum of five white balls |
| `total_sum` | int | Sum of all six numbers |
| `has_even_count` | int | Count of even numbers |
| `has_odd_count` | int | Count of odd numbers |
| `raw_line` | str | Original text line for debugging |

## Python API Usage

```python
from parse_powerball_pdf import download_pdf, extract_text_from_pdf, parse_powerball_data, create_dataframe

# Download and parse
pdf_content = download_pdf("https://files.floridalottery.com/exptkt/pb.pdf")
text = extract_text_from_pdf(pdf_content)
data = parse_powerball_data(text)
df = create_dataframe(data)

# Analysis
print(df.head())
print(df.describe())

# Find most common numbers
all_balls = pd.concat([df['ball_1'], df['ball_2'], df['ball_3'], 
                       df['ball_4'], df['ball_5']])
print(all_balls.value_counts().head(10))

# Filter by date
recent = df[df['date'] >= '2024-01-01']

# Find patterns
print(df['has_even_count'].value_counts())
print(df['sum_white_balls'].describe())
```

## Analysis Features

The script automatically calculates:

- **Most common white ball numbers**
- **Most common Powerball numbers**
- **Sum statistics** (min, max, average)
- **Even/Odd distribution**
- **Date range coverage**

## Customization

If the PDF format changes, update the regex patterns in `parse_powerball_data()`:

```python
# Adjust this pattern to match the actual PDF format
pattern1 = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{4})\s+'  # Date
    r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # 5 balls
    r'(\d{1,2})'  # Powerball
)
```

Check `/tmp/powerball_raw_text.txt` to see the actual format and adjust accordingly.

## Troubleshooting

### No data parsed

If you see "No data was parsed!", the PDF format may have changed:

1. Check `/tmp/powerball_raw_text.txt` to see the raw extracted text
2. Update the regex patterns in `parse_powerball_data()`
3. The script includes two patterns for flexibility

### PDF download fails

```python
# Use alternative URL or local file
pdf_content = open('pb.pdf', 'rb').read()
text = extract_text_from_pdf(pdf_content)
```

### Date parsing errors

The script tries multiple date formats. If dates aren't parsing:

```python
# Manually specify format
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
```

## Example Analysis Queries

```python
# Load saved data
df = pd.read_csv('powerball_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Most recent drawing
print(df.iloc[0])

# Drawings in 2024
df_2024 = df[df['date'].dt.year == 2024]

# High jackpot drawings
high_jackpot = df[df['jackpot'].notna()]

# Numbers that appear together frequently
from itertools import combinations
pairs = []
for _, row in df.iterrows():
    nums = [row['ball_1'], row['ball_2'], row['ball_3'], row['ball_4'], row['ball_5']]
    pairs.extend(combinations(sorted(nums), 2))

pair_counts = pd.Series(pairs).value_counts()
print("Most common number pairs:")
print(pair_counts.head(20))

# Average days between same powerball number
for num in range(1, 27):
    subset = df[df['powerball'] == num].copy()
    if len(subset) > 1:
        subset['days_between'] = subset['date'].diff().abs().dt.days
        print(f"Powerball {num}: avg {subset['days_between'].mean():.1f} days")
```

## Notes

- The PDF contains historical data from Florida Lottery
- Drawing dates are typically Wednesday and Saturday
- White balls range from 1-69
- Powerball ranges from 1-26
- Power Play multiplier (2X, 3X, 4X, 5X, or 10X) is optional
- Data is sorted newest to oldest by default

## Random Forest Bin Classification Model

### Overview

The Random Forest bin classification model (`predict_powerball_bins_gpu.py`) takes a different approach to prediction by:

1. **Binning Historical Numbers**: Converts lottery numbers to standardized bins based on statistical deviations
2. **Sequence Analysis**: Uses sequences of previous bin patterns as features
3. **GPU Acceleration**: Leverages RAPIDS cuML or PyTorch for faster training
4. **Pattern Recognition**: Assumes Linear Congruential Generator (LCG) patterns create repeatable bin sequences

### Bin Classification System

Numbers are classified into 6 bins representing standard deviation bands:

- **Bin 0**: [-3σ, -2σ) - Very low numbers
- **Bin 1**: [-2σ, -1σ) - Low numbers  
- **Bin 2**: [-1σ, 0) - Below average
- **Bin 3**: [0, +1σ) - Above average
- **Bin 4**: [+1σ, +2σ) - High numbers
- **Bin 5**: [+2σ, +3σ] - Very high numbers

### Model Features

- **Automatic Optimization**: Finds optimal sequence length (5-30 drawings)
- **Multi-Ball Training**: Separate Random Forest for each ball position
- **GPU Acceleration**: Uses cuML if available, falls back to sklearn
- **Feature Engineering**: Flattened sequences of historical bin patterns
- **Statistical Foundation**: Based on mean/std calculations from historical data

### Usage Example

```bash
# Run with GPU acceleration (Docker)
./run-bin-prediction.sh

# Or run locally
source venv-powerball/bin/activate
python predict_powerball_bins_gpu.py
```

### Model Output

```
Ball 1: Predicted Bin: 2 → Number: 15
Ball 2: Predicted Bin: 4 → Number: 45  
Ball 3: Predicted Bin: 1 → Number: 8
Ball 4: Predicted Bin: 3 → Number: 35
Ball 5: Predicted Bin: 5 → Number: 62

Predicted Numbers (sorted): [8, 15, 35, 45, 62]
```

### Theory Behind LCG Pattern Recognition

The model assumes lottery systems use Linear Congruential Generators with the formula:
```
X(n+1) = (a × X(n) + c) mod m
```

Where the final lottery numbers are: `lottery_number = floor(X(n) / large_divisor) + 1`

Since LCG sequences eventually repeat, and the modulo/division operations preserve some statistical patterns, sequences of numbers falling into specific statistical bins may exhibit predictable patterns over time.

### Model Parameters

- **Sequence Length**: Auto-optimized (typically 10-20 drawings)
- **Random Forest**: 200 trees, max depth 15, sqrt features
- **Bins**: 6 bins spanning -3σ to +3σ
- **GPU Support**: RAPIDS cuML or CPU sklearn

## Comprehensive Prediction Comparison

### Overview

The master prediction script (`run-prediction-comparison.sh`) runs both models and provides detailed comparison:

1. **Downloads Fresh Data**: Gets latest PDF from Florida Lottery
2. **Runs Fourier Model**: Time-series harmonic analysis
3. **Runs Random Forest Model**: Bin classification with sequences  
4. **Compares Results**: Statistical analysis and consensus detection
5. **Generates Reports**: Comprehensive analysis and visualizations

### Usage

```bash
# Run comprehensive prediction comparison
./run-prediction-comparison.sh
```

### Output Structure

```
results_YYYYMMDD_HHMMSS/
├── fourier_predictions.csv           # Fourier model predictions
├── random_forest_predictions.csv     # Random Forest predictions  
├── powerball_games_only.csv          # Fresh lottery data
├── DETAILED_COMPARISON.md             # Comprehensive analysis
├── bin_predictions_visualization.png # Charts and graphs
├── fourier_output.log                # Execution logs
├── random_forest_output.log          # Execution logs
└── random_forest_bin_model.joblib    # Trained model
```

### Interpretation

- **Consensus**: When models agree on 2+ numbers, higher confidence
- **No Consensus**: Different mathematical approaches, consider both
- **Statistical Analysis**: Range, sums, distribution patterns
- **Model Strengths**: Fourier (time-series), Random Forest (patterns)

## License

This script is for educational and analytical purposes only. Lottery data is provided by Florida Lottery.
