#!/bin/bash
# Analyze historical predictions for consistency and changes

echo "======================================================================="
echo "POWERBALL HISTORICAL PREDICTION ANALYSIS"
echo "======================================================================="
echo "Date: $(date)"
echo ""

# Activate virtual environment if available
if [ -f "venv-powerball/bin/activate" ]; then
    echo "✓ Activating virtual environment..."
    source venv-powerball/bin/activate
else
    echo "⚠ Virtual environment not found"
fi

# Check for historical predictions file
HISTORY_FILES=(
    "historical_predictions.csv"
    "results_*/historical_predictions.csv"
)

FOUND_FILE=""
for pattern in "${HISTORY_FILES[@]}"; do
    for file in $pattern; do
        if [ -f "$file" ]; then
            FOUND_FILE="$file"
            break 2
        fi
    done
done

if [ -z "$FOUND_FILE" ]; then
    echo "❌ No historical predictions found!"
    echo "Run ./run-prediction-comparison.sh first to generate predictions."
    exit 1
fi

echo "📊 Using historical data: $FOUND_FILE"
echo ""

# Run comprehensive analysis
python3 -c "
from prediction_tracker import PredictionTracker
import pandas as pd
import os

# Initialize tracker with found file
tracker = PredictionTracker('$FOUND_FILE')

print('Running comprehensive historical analysis...')
print()

# Summary statistics
tracker.get_summary_stats()

# Consistency analysis for both models
print()
print('='*70)
print('DETAILED CONSISTENCY ANALYSIS')
print('='*70)

# Analyze last 30 days
tracker.analyze_consistency(model_type='fourier', days_back=30)
tracker.analyze_consistency(model_type='random_forest', days_back=30)

# Model agreement analysis
tracker.analyze_model_agreement()

# Create visualization
try:
    viz_path = 'historical_prediction_analysis.png'
    tracker.create_consistency_visualization(viz_path)
    print(f'\\n✓ Analysis visualization saved to {viz_path}')
except Exception as e:
    print(f'\\n⚠ Could not create visualization: {e}')

# Detailed change analysis
print('\\n' + '='*70)
print('PREDICTION CHANGE DETECTION')  
print('='*70)

# Check for any prediction changes for same target dates
history_df = tracker.history_df

if len(history_df) > 0:
    # Group by target date and model
    changes_found = False
    
    for target_date in history_df['target_drawing_date'].unique():
        for model in ['fourier', 'random_forest']:
            target_model_preds = history_df[
                (history_df['target_drawing_date'] == target_date) &
                (history_df['model_type'] == model)
            ].sort_values('prediction_made_date')
            
            if len(target_model_preds) > 1:
                changes_found = True
                print(f'\\nTarget: {target_date.strftime(\"%Y-%m-%d\")} | Model: {model.upper()}')
                
                prev_numbers = None
                for _, pred in target_model_preds.iterrows():
                    current_numbers = [pred[f'ball_{i}'] for i in range(1, 6)]
                    pred_time = pred['prediction_made_date'].strftime('%m/%d %H:%M')
                    
                    if prev_numbers is None:
                        print(f'  {pred_time}: {current_numbers} (initial)')
                    else:
                        if current_numbers != prev_numbers:
                            changed_positions = []
                            for i in range(5):
                                if current_numbers[i] != prev_numbers[i]:
                                    changed_positions.append(f'ball_{i+1}: {prev_numbers[i]}→{current_numbers[i]}')
                            
                            print('  {}: {} (CHANGED: {})'.format(pred_time, current_numbers, ', '.join(changed_positions)))
                        else:
                            print(f'  {pred_time}: {current_numbers} (same)')
                    
                    prev_numbers = current_numbers
    
    if not changes_found:
        print('✅ No prediction changes detected - models are consistent!')
        print('(Each target date has only one prediction per model)')
else:
    print('No historical data to analyze.')

print('\\n' + '='*70)
print('ANALYSIS COMPLETE')
print('='*70)
print('• Check historical_prediction_analysis.png for visualizations')
print('• Run prediction comparison again to add more data points')
print('• Consistent predictions indicate stable model behavior')
print('• Changes may indicate new data or model improvements')
print('='*70)
"

echo ""
echo "✅ Historical prediction analysis complete!"
echo "📈 Check historical_prediction_analysis.png for visual analysis"