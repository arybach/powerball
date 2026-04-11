#!/bin/bash
# Master script to run both Fourier and Random Forest predictions
# Fetches fresh PDF data and compares both model predictions

echo "======================================================================="
echo "POWERBALL PREDICTION COMPARISON - FOURIER vs RANDOM FOREST"
echo "======================================================================="
echo "Date: $(date)"
echo ""

# Set up environment
export HSA_OVERRIDE_GFX_VERSION=11.5.1
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx1151

# Create results directory
RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to check if virtual environment exists and activate it
activate_venv() {
    if [ -f "venv-powerball/bin/activate" ]; then
        echo "✓ Activating virtual environment..."
        source venv-powerball/bin/activate
        echo "✓ Virtual environment activated"
    else
        echo "⚠ Virtual environment not found, using system Python"
    fi
}

# Function to run Fourier prediction
run_fourier() {
    echo ""
    echo "======================================================================="
    echo "1. RUNNING FOURIER SERIES PREDICTION"
    echo "======================================================================="
    
    # Check for bin-based Fourier option
    FOURIER_MODE=${FOURIER_MODE:-"numbers"}  # Default to numbers, can be set to "bins"
    
    if [ "$FOURIER_MODE" = "bins" ]; then
        echo "Using bin-based Fourier analysis with ROCm PyTorch..."
        ./run-fourier-bin-prediction.sh "$RESULTS_DIR"
        return
    fi
    
    echo "Using number-based Fourier analysis..."
    if command -v docker &> /dev/null && [ -f "Dockerfile" ]; then
        echo "Using Docker with ROCm GPU acceleration..."
        
        docker run --rm \
            --device=/dev/kfd \
            --device=/dev/dri \
            --group-add video \
            --security-opt seccomp=unconfined \
            -v /projects/powerball:/workspace \
            -w /workspace \
            -e HSA_OVERRIDE_GFX_VERSION=11.5.1 \
            -e HIP_VISIBLE_DEVICES=0 \
            -e PYTORCH_ROCM_ARCH=gfx1151 \
            rocm/pytorch:latest \
            bash -c "
                echo 'Installing dependencies...'
                pip install -q requests PyPDF2
                
                echo 'Running Fourier Series Prediction...'
                python3 predict_powerball_fourier.py
            " 2>&1 | tee "$RESULTS_DIR/fourier_output.log"
    else
        echo "Docker not available, running locally..."
        activate_venv
        
        echo "Running Fourier Series Prediction..."
        python3 predict_powerball_fourier.py 2>&1 | tee "$RESULTS_DIR/fourier_output.log"
    fi
    
    # Copy Fourier results
    if [ -f "powerball_predictions.csv" ]; then
        cp powerball_predictions.csv "$RESULTS_DIR/fourier_predictions.csv"
        echo "✓ Fourier predictions saved to $RESULTS_DIR/fourier_predictions.csv"
    elif [ -f "powerball_predictions_latest.csv" ]; then
        cp powerball_predictions_latest.csv "$RESULTS_DIR/fourier_predictions.csv"
        echo "✓ Fourier predictions saved to $RESULTS_DIR/fourier_predictions.csv (from fallback output)"
    fi
    
    if [ -f "powerball_games_only.csv" ]; then
        cp powerball_games_only.csv "$RESULTS_DIR/powerball_games_only.csv"
        echo "✓ Game data saved to $RESULTS_DIR/powerball_games_only.csv"
    fi
}

# Function to run Random Forest prediction
run_random_forest() {
    echo ""
    echo "======================================================================="
    echo "2. RUNNING ENHANCED RANDOM FOREST BIN PREDICTION"
    echo "======================================================================="
    
    # Try Docker first, fall back to local execution
    if command -v docker &> /dev/null && [ -f "Dockerfile.ml" ]; then
        echo "Using Docker with ROCm GPU acceleration..."
        
        # Build image if needed
        if ! docker images | grep -q powerball-ml; then
            echo "Building ML Docker image..."
            docker build -f Dockerfile.ml -t powerball-ml .
        fi
        
        echo "Installing ML dependencies..."
        docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace powerball-ml \
            bash -c "pip install cuml-cu12 scipy || echo 'cuML installation failed, will use CPU'"
        
        echo "Checking for RAPIDS cuML (GPU acceleration)..."
        docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace powerball-ml \
            python3 -c "
try:
    import cuml
    print('✓ cuML GPU acceleration available')
except ImportError:
    print('⚠ cuML not available, using CPU sklearn')
"
        
        echo "Running Enhanced Random Forest Bin Prediction..."
        docker run --rm --gpus all -v "$(pwd)":/workspace -w /workspace powerball-ml \
            python3 predict_powerball_bins_gpu_enhanced.py > "$RESULTS_DIR/random_forest_output.log" 2>&1
    else
        echo "Docker not available, using local enhanced model..."
        activate_venv
        
        echo "Running Enhanced Random Forest Bin Prediction locally..."
        python3 predict_powerball_bins_gpu_enhanced.py > "$RESULTS_DIR/random_forest_output.log" 2>&1
    fi
    
    # Copy enhanced results (try both filenames for compatibility)
    if [ -f "enhanced_random_forest_predictions.csv" ]; then
        cp enhanced_random_forest_predictions.csv "$RESULTS_DIR/random_forest_predictions.csv"
        echo "✓ Enhanced Random Forest predictions saved to $RESULTS_DIR/random_forest_predictions.csv"
    elif [ -f "random_forest_predictions.csv" ]; then
        cp random_forest_predictions.csv "$RESULTS_DIR/"
        echo "✓ Random Forest predictions saved to $RESULTS_DIR/random_forest_predictions.csv"
    fi
    
    if [ -f "enhanced_random_forest_bin_model.joblib" ]; then
        cp enhanced_random_forest_bin_model.joblib "$RESULTS_DIR/random_forest_bin_model.joblib"
        echo "✓ Enhanced trained model saved to $RESULTS_DIR/random_forest_bin_model.joblib"
    elif [ -f "random_forest_bin_model.joblib" ]; then
        cp random_forest_bin_model.joblib "$RESULTS_DIR/"
        echo "✓ Trained model saved to $RESULTS_DIR/random_forest_bin_model.joblib"
    fi
    
    if [ -f "bin_predictions_visualization.png" ]; then
        cp bin_predictions_visualization.png "$RESULTS_DIR/"
        echo "✓ Visualization saved to $RESULTS_DIR/bin_predictions_visualization.png"
    fi
}

# Function to store predictions in historical tracker
store_predictions() {
    echo ""
    echo "======================================================================="
    echo "3. STORING PREDICTIONS IN HISTORY"
    echo "======================================================================="
    
    activate_venv
    
    python3 -c "
from prediction_tracker import PredictionTracker
import pandas as pd
import hashlib
from datetime import datetime, timedelta

# Initialize tracker
tracker = PredictionTracker('$RESULTS_DIR/historical_predictions.csv')

# Calculate data hash for version tracking
try:
    with open('$RESULTS_DIR/powerball_games_only.csv', 'r') as f:
        data_content = f.read()
    data_hash = hashlib.md5(data_content.encode()).hexdigest()[:8]
except:
    data_hash = 'unknown'

print(f'Data hash: {data_hash}')

# Load and store Fourier predictions
try:
    fourier_df = pd.read_csv('$RESULTS_DIR/fourier_predictions.csv')
    if len(fourier_df) > 0:
        latest_fourier = fourier_df.iloc[-1]
        fourier_numbers = [int(latest_fourier[f'ball_{i}']) for i in range(1, 6)]
        
        fourier_details = {
            'harmonics': 15,
            'device': str(latest_fourier.get('device', 'unknown')),
            'model_type': 'fourier_series'
        }
        
        tracker.add_prediction(
            target_date=str(latest_fourier['prediction_date']),
            model_type='fourier',
            numbers=fourier_numbers,
            model_details=fourier_details,
            data_hash=data_hash
        )
        print(f'✓ Stored Fourier prediction: {fourier_numbers}')
except Exception as e:
    print(f'⚠ Could not store Fourier prediction: {e}')

# Load and store Random Forest predictions
try:
    rf_df = pd.read_csv('$RESULTS_DIR/random_forest_predictions.csv')
    if len(rf_df) > 0:
        latest_rf = rf_df.iloc[-1]
        rf_numbers = [int(latest_rf[f'ball_{i}']) for i in range(1, 6)]
        
        rf_details = {
            'sequence_length': int(latest_rf.get('sequence_length', 0)),
            'model_info': str(latest_rf.get('model', '')),
            'bins': [int(latest_rf[f'bin_{i}']) for i in range(1, 6)]
        }
        
        tracker.add_prediction(
            target_date=str(latest_rf['prediction_date']),
            model_type='random_forest',
            numbers=rf_numbers,
            model_details=rf_details,
            data_hash=data_hash
        )
        print(f'✓ Stored Random Forest prediction: {rf_numbers}')
except Exception as e:
    print(f'⚠ Could not store Random Forest prediction: {e}')

# Save the updated history
tracker.save_history()

# Quick consistency check if we have multiple predictions
if len(tracker.history_df) > 1:
    print('\\n--- Quick Consistency Check ---')
    tracker.analyze_consistency(days_back=7)
"
}

# Function to compare predictions
compare_predictions() {
    echo ""
    echo "======================================================================="
    echo "4. COMPARING PREDICTIONS"
    echo "======================================================================="
    
    activate_venv
    
    python3 -c "
import pandas as pd
from datetime import datetime

print('PREDICTION COMPARISON SUMMARY')
print('=' * 50)

# Load Fourier predictions
try:
    fourier_df = pd.read_csv('$RESULTS_DIR/fourier_predictions.csv')
    print('\n📊 FOURIER SERIES PREDICTION:')
    if len(fourier_df) > 0:
        latest_fourier = fourier_df.iloc[-1]
        fourier_numbers = [latest_fourier[f'ball_{i}'] for i in range(1, 6)]
        fourier_sorted = sorted(fourier_numbers)
        print(f'  Date: {latest_fourier[\"prediction_date\"]}')
        print(f'  Numbers: {fourier_numbers}')
        print(f'  Sorted: {fourier_sorted}')
        print(f'  Model: {latest_fourier.get(\"model\", \"Fourier Series\")}')
    else:
        print('  ⚠ No Fourier predictions found')
        fourier_numbers = []
except FileNotFoundError:
    print('  ⚠ Fourier predictions file not found')
    fourier_numbers = []

# Load Random Forest predictions  
try:
    rf_df = pd.read_csv('$RESULTS_DIR/random_forest_predictions.csv')
    print('\n🌲 RANDOM FOREST BIN PREDICTION:')
    if len(rf_df) > 0:
        latest_rf = rf_df.iloc[-1]
        rf_numbers = [latest_rf[f'ball_{i}'] for i in range(1, 6)]
        rf_sorted = sorted(rf_numbers)
        rf_bins = [latest_rf[f'bin_{i}'] for i in range(1, 6)]
        print(f'  Date: {latest_rf[\"prediction_date\"]}')
        print(f'  Numbers: {rf_numbers}')
        print(f'  Sorted: {rf_sorted}')
        print(f'  Bins: {rf_bins}')
        print(f'  Model: {latest_rf.get(\"model\", \"Random Forest\")}')
        print(f'  Sequence Length: {latest_rf.get(\"sequence_length\", \"N/A\")}')
    else:
        print('  ⚠ No Random Forest predictions found')
        rf_numbers = []
except FileNotFoundError:
    print('  ⚠ Random Forest predictions file not found')
    rf_numbers = []

# Compare predictions
if fourier_numbers and rf_numbers:
    print('\n🔍 COMPARISON ANALYSIS:')
    
    # Number overlap
    fourier_set = set(fourier_numbers)
    rf_set = set(rf_numbers)
    overlap = fourier_set.intersection(rf_set)
    
    print(f'  Common numbers: {sorted(list(overlap)) if overlap else \"None\"}')
    print(f'  Overlap count: {len(overlap)}/5 numbers')
    
    # Range analysis
    fourier_range = max(fourier_numbers) - min(fourier_numbers)
    rf_range = max(rf_numbers) - min(rf_numbers)
    
    print(f'  Fourier range: {fourier_range} (min: {min(fourier_numbers)}, max: {max(fourier_numbers)})')
    print(f'  Random Forest range: {rf_range} (min: {min(rf_numbers)}, max: {max(rf_numbers)})')
    
    # Sum comparison
    fourier_sum = sum(fourier_numbers)
    rf_sum = sum(rf_numbers)
    
    print(f'  Fourier sum: {fourier_sum}')
    print(f'  Random Forest sum: {rf_sum}')
    print(f'  Difference: {abs(fourier_sum - rf_sum)}')
    
    # Consensus recommendation
    if len(overlap) >= 2:
        print(f'\n✅ CONSENSUS: Both models agree on {len(overlap)} numbers: {sorted(list(overlap))}')
    elif len(overlap) == 1:
        print(f'\n⚖️ PARTIAL CONSENSUS: Both models agree on 1 number: {list(overlap)[0]}')
    else:
        print('\n❌ NO CONSENSUS: Models predict completely different numbers')
        
    print(f'\n📋 FINAL RECOMMENDATIONS:')
    print(f'  Fourier Choice: {fourier_sorted}')
    print(f'  Random Forest Choice: {rf_sorted}')
    if overlap:
        consensus_nums = sorted(list(overlap))
        remaining_fourier = [n for n in fourier_sorted if n not in overlap]
        remaining_rf = [n for n in rf_sorted if n not in overlap]
        print(f'  Hybrid (consensus + Fourier): {consensus_nums + remaining_fourier[:5-len(consensus_nums)]}')
        print(f'  Hybrid (consensus + RF): {consensus_nums + remaining_rf[:5-len(consensus_nums)]}')

print('\n🎯 Next Drawing Expected: Saturday or Wednesday')
print('📅 Prediction Date: ' + str(datetime.now().strftime('%Y-%m-%d')))

print('\n' + '=' * 70)
print('DISCLAIMER: These are mathematical models for educational purposes.')
print('Lottery drawings are designed to be random. Use predictions responsibly.')
print('=' * 70)
" 2>&1 | tee "$RESULTS_DIR/comparison_summary.txt"
}

# Function to generate time series analysis
generate_time_series_analysis() {
    echo ""
    echo "======================================================================="
    echo "6. GENERATING TIME SERIES ANALYSIS"
    echo "======================================================================="
    
    activate_venv
    
    echo "Creating actual vs predicted time series plots..."
    
    python3 -c "
from plot_time_series import PowerballTimeSeriesPlotter
import os
import shutil

try:
    # Create plotter instance
    plotter = PowerballTimeSeriesPlotter('$RESULTS_DIR/powerball_games_only.csv')
    
    # Generate time series analysis (smaller scope for speed)
    plotter.create_comprehensive_analysis(
        prediction_days=45,  # Last 45 days for reasonable speed
        save_dir='$RESULTS_DIR/time_series_analysis'
    )
    
    print('✓ Time series analysis generated successfully')
    
except Exception as e:
    print(f'⚠ Time series analysis failed: {e}')
    # Create basic info file anyway
    os.makedirs('$RESULTS_DIR/time_series_analysis', exist_ok=True)
    with open('$RESULTS_DIR/time_series_analysis/analysis_failed.txt', 'w') as f:
        f.write(f'Time series analysis failed: {e}\\n')
        f.write('Run ./run-time-series-analysis.sh separately for detailed analysis\\n')
"
    
    if [ -d "$RESULTS_DIR/time_series_analysis" ]; then
        echo "✓ Time series analysis saved to $RESULTS_DIR/time_series_analysis/"
    else
        echo "⚠ Time series analysis was not generated"
    fi
}

# Function to create summary report
create_summary() {
    echo ""
    echo "======================================================================="
    echo "5. CREATING SUMMARY REPORT"
    echo "======================================================================="
    
    # Create comprehensive summary
    cat > "$RESULTS_DIR/PREDICTION_SUMMARY.md" << EOF
# Powerball Prediction Summary - $(date)

## Run Information
- **Date**: $(date)
- **Directory**: $RESULTS_DIR
- **Models**: Fourier Series + Random Forest Bin Classification

## Files Generated
- \`fourier_predictions.csv\` - Fourier series predictions
- \`random_forest_predictions.csv\` - Random Forest bin predictions
- \`fourier_output.log\` - Fourier execution log
- \`random_forest_output.log\` - Random Forest execution log
- \`comparison_summary.txt\` - Model comparison analysis
- \`powerball_games_only.csv\` - Fresh game data
- \`bin_predictions_visualization.png\` - Analysis charts
- \`random_forest_bin_model.joblib\` - Trained model

## Model Descriptions

### Fourier Series Model
- **Method**: GPU-accelerated Fourier series fitting
- **Approach**: Time series prediction using harmonic analysis
- **Features**: Minimizes standard deviation of residuals
- **Output**: Direct number predictions for each ball position

### Random Forest Bin Classification
- **Method**: Sequence-based bin classification with Random Forest
- **Approach**: Predicts statistical bins, converts to numbers
- **Features**: Historical bin sequence patterns (optimized length)
- **Output**: Bin classifications + approximate number predictions
- **Training Split**: 67% training / 33% testing chronologically

## Usage
This summary provides predictions from both mathematical approaches for comparison and analysis.

EOF

    echo "✓ Summary report created: $RESULTS_DIR/PREDICTION_SUMMARY.md"
    
    # List all generated files
    echo ""
    echo "📁 Generated Files:"
    ls -la "$RESULTS_DIR/" | grep -v "^total" | awk '{print "  " $9 " (" $5 " bytes)"}'
}

# Main execution
main() {
    echo "Starting comprehensive prediction run..."
    echo ""
    
    # Ensure we're in the right directory
    if [ ! -f "predict_powerball_fourier.py" ] || [ ! -f "predict_powerball_bins_gpu_enhanced.py" ]; then
        echo "❌ Error: Prediction scripts not found in current directory"
        echo "Please run this script from the powerball project directory"
        exit 1
    fi
    
    # Run both models
    run_fourier
    run_random_forest
    
    # Store predictions in historical tracker
    store_predictions
    
    # Compare and analyze
    compare_predictions
    
    # Create summary
    create_summary
    
    # Generate time series analysis
    generate_time_series_analysis
    
    echo ""
    echo "======================================================================="
    echo "✅ PREDICTION RUN COMPLETED SUCCESSFULLY"
    echo "======================================================================="
    echo "📊 Results saved to: $RESULTS_DIR/"
    echo "📈 View comparison: cat $RESULTS_DIR/comparison_summary.txt"
    echo "📋 Full summary: cat $RESULTS_DIR/PREDICTION_SUMMARY.md"
    echo ""
    echo "🎲 Ready for next Powerball drawing!"
    echo "======================================================================="
}

# Execute main function
main "$@"