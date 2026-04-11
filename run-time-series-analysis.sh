#!/bin/bash
# Generate comprehensive time series plots for actual vs predicted numbers

echo "======================================================================="
echo "POWERBALL TIME SERIES ANALYSIS - ACTUAL vs PREDICTED"
echo "======================================================================="
echo "Date: $(date)"
echo ""

# Activate virtual environment if available
if [ -f "venv-powerball/bin/activate" ]; then
    echo "✓ Activating virtual environment..."
    source venv-powerball/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found, using system Python"
fi

# Ensure required dependencies are installed (matplotlib and seaborn already installed in venv)
echo "Checking plotting dependencies..."
python3 -c "import matplotlib, seaborn; print('✓ Plotting libraries available')" || {
    echo "Installing plotting dependencies in virtual environment..."
    pip install matplotlib seaborn
}

# Check if data file exists
if [ ! -f "powerball_games_only.csv" ]; then
    echo "❌ powerball_games_only.csv not found!"
    echo "Run ./run-powerball-parser.sh or ./run-prediction-comparison.sh first"
    exit 1
fi

echo "✓ Found lottery data file"
echo ""

# Run the time series analysis
echo "🔍 Generating time series analysis..."
echo "This will create:"
echo "  • Actual vs Predicted time series plots for each ball"
echo "  • Prediction accuracy heatmaps"
echo "  • Statistical summary of model performance"
echo ""

# Use the virtual environment python explicitly
/projects/powerball/venv-powerball/bin/python plot_time_series.py

# Check results
if [ -d "time_series_analysis" ]; then
    echo ""
    echo "📊 ANALYSIS RESULTS:"
    echo "✓ Time series plots: time_series_analysis/"
    
    # List generated files
    echo ""
    echo "📁 Generated Files:"
    ls -la time_series_analysis/ | grep -v "^total" | awk '{print "  " $9 " (" $5 " bytes)"}'
    
    echo ""
    echo "🎯 KEY PLOTS:"
    echo "  📈 Main Analysis: time_series_analysis/actual_vs_predicted_timeseries.png"
    echo "  🔥 Accuracy Map: time_series_analysis/prediction_accuracy_heatmap.png"
    echo "  📊 Statistics: time_series_analysis/summary_statistics.txt"
    
    # Show quick stats if available
    if [ -f "time_series_analysis/summary_statistics.txt" ]; then
        echo ""
        echo "📋 QUICK SUMMARY:"
        head -15 time_series_analysis/summary_statistics.txt | sed 's/^/  /'
    fi
    
else
    echo "❌ Analysis failed - no output directory created"
    exit 1
fi

echo ""
echo "======================================================================="
echo "✅ TIME SERIES ANALYSIS COMPLETE"
echo "======================================================================="
echo "📊 View plots in: time_series_analysis/"
echo "🔍 This analysis shows how well our models predict actual lottery numbers"
echo "📈 Lower MAE values indicate better prediction accuracy"
echo "🎲 Use these insights to evaluate model performance over time"
echo "======================================================================="