#!/usr/bin/env python3
"""
Historical Prediction Tracker for Powerball Models
Stores and analyzes prediction consistency over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class PredictionTracker:
    """
    Tracks historical predictions from both Fourier and Random Forest models
    to analyze consistency and changes over time.
    """
    
    def __init__(self, storage_file: str = "historical_predictions.csv"):
        self.storage_file = storage_file
        self.load_historical_data()
    
    def load_historical_data(self):
        """Load existing historical predictions."""
        if os.path.exists(self.storage_file):
            self.history_df = pd.read_csv(self.storage_file)
            self.history_df['prediction_made_date'] = pd.to_datetime(self.history_df['prediction_made_date'])
            self.history_df['target_drawing_date'] = pd.to_datetime(self.history_df['target_drawing_date'])
            print(f"✓ Loaded {len(self.history_df)} historical predictions from {self.storage_file}")
        else:
            self.history_df = pd.DataFrame(columns=[
                'prediction_made_date', 'target_drawing_date', 'model_type',
                'ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'powerball',
                'sorted_numbers', 'model_details', 'data_hash'
            ])
            print(f"✓ Created new prediction history tracker: {self.storage_file}")
    
    def add_prediction(self,
                      target_date: str,
                      model_type: str,
                      numbers: List[int],
                      model_details: Dict,
                      data_hash: str = None,
                      powerball: Optional[int] = None):
        """
        Add a new prediction to the historical record.

        Args:
            target_date: Date the prediction is for (YYYY-MM-DD)
            model_type: 'fourier' or 'random_forest'
            numbers: List of 5 predicted white-ball numbers
            model_details: Dict with model-specific details
            data_hash: Hash of training data to detect data changes
            powerball: Optional predicted red Powerball (1-26)
        """
        prediction_made = datetime.now()

        new_prediction = {
            'prediction_made_date': prediction_made,
            'target_drawing_date': pd.to_datetime(target_date),
            'model_type': model_type,
            'ball_1': numbers[0],
            'ball_2': numbers[1],
            'ball_3': numbers[2],
            'ball_4': numbers[3],
            'ball_5': numbers[4],
            'powerball': powerball,
            'sorted_numbers': str(sorted(numbers)),
            'model_details': json.dumps(model_details),
            'data_hash': data_hash or "unknown"
        }

        self.history_df = pd.concat([
            self.history_df,
            pd.DataFrame([new_prediction])
        ], ignore_index=True)

        pb_str = f" + PB {powerball}" if powerball is not None else ""
        print(f"✓ Added {model_type} prediction for {target_date}: {numbers}{pb_str}")
    
    def save_history(self):
        """Save historical predictions to CSV."""
        self.history_df.to_csv(self.storage_file, index=False)
        print(f"✓ Saved {len(self.history_df)} predictions to {self.storage_file}")
    
    def analyze_consistency(self, model_type: str = None, days_back: int = 30):
        """
        Analyze prediction consistency over time.
        
        Args:
            model_type: 'fourier', 'random_forest', or None for both
            days_back: Number of days to analyze
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        if model_type:
            df_filtered = self.history_df[
                (self.history_df['model_type'] == model_type) &
                (self.history_df['prediction_made_date'] >= cutoff_date)
            ].copy()
        else:
            df_filtered = self.history_df[
                self.history_df['prediction_made_date'] >= cutoff_date
            ].copy()
        
        if len(df_filtered) == 0:
            print(f"No predictions found for {model_type or 'any model'} in last {days_back} days")
            return None
        
        print(f"\n{'='*60}")
        print(f"PREDICTION CONSISTENCY ANALYSIS - LAST {days_back} DAYS")
        if model_type:
            print(f"Model: {model_type.upper()}")
        print(f"{'='*60}")
        
        # Group by target drawing date
        for target_date in df_filtered['target_drawing_date'].unique():
            target_predictions = df_filtered[
                df_filtered['target_drawing_date'] == target_date
            ].sort_values('prediction_made_date')
            
            if len(target_predictions) <= 1:
                continue
                
            print(f"\nTarget Drawing: {target_date.strftime('%Y-%m-%d')}")
            print(f"Predictions made: {len(target_predictions)}")
            
            # Check for changes in predictions
            previous_numbers = None
            changes_detected = []
            
            for _, pred in target_predictions.iterrows():
                current_numbers = [pred[f'ball_{i}'] for i in range(1, 6)]
                pred_time = pred['prediction_made_date'].strftime('%Y-%m-%d %H:%M')
                
                print(f"  {pred_time}: {current_numbers} ({pred['model_type']})")
                
                if previous_numbers is not None:
                    if current_numbers != previous_numbers:
                        changes = self._compare_predictions(previous_numbers, current_numbers)
                        changes_detected.append({
                            'time': pred_time,
                            'changes': changes
                        })
                        print(f"    ⚠ Changed from previous: {changes}")
                
                previous_numbers = current_numbers
            
            if changes_detected:
                print(f"  📊 Total changes detected: {len(changes_detected)}")
            else:
                print(f"  ✅ Consistent predictions (no changes)")
        
        return df_filtered
    
    def _compare_predictions(self, old_nums: List[int], new_nums: List[int]) -> Dict:
        """Compare two predictions and return change details."""
        old_set = set(old_nums)
        new_set = set(new_nums)
        
        return {
            'numbers_changed': len(old_set.symmetric_difference(new_set)),
            'numbers_added': sorted(list(new_set - old_set)),
            'numbers_removed': sorted(list(old_set - new_set)),
            'old_sum': sum(old_nums),
            'new_sum': sum(new_nums),
            'sum_change': sum(new_nums) - sum(old_nums)
        }
    
    def analyze_model_agreement(self, target_date: str = None):
        """
        Analyze agreement between Fourier and Random Forest models.
        """
        if target_date:
            df_filtered = self.history_df[
                self.history_df['target_drawing_date'] == pd.to_datetime(target_date)
            ]
        else:
            # Use most recent predictions for each model
            df_filtered = self.history_df.groupby(['target_drawing_date', 'model_type']).tail(1)
        
        print(f"\n{'='*60}")
        print(f"MODEL AGREEMENT ANALYSIS")
        if target_date:
            print(f"Target Date: {target_date}")
        else:
            print("All Recent Predictions")
        print(f"{'='*60}")
        
        # Group by target date
        for target_date_val in df_filtered['target_drawing_date'].unique():
            target_preds = df_filtered[
                df_filtered['target_drawing_date'] == target_date_val
            ]
            
            fourier_preds = target_preds[target_preds['model_type'] == 'fourier']
            rf_preds = target_preds[target_preds['model_type'] == 'random_forest']
            
            if len(fourier_preds) == 0 or len(rf_preds) == 0:
                continue
                
            print(f"\nTarget Date: {target_date_val.strftime('%Y-%m-%d')}")
            
            # Get latest prediction for each model
            fourier_latest = fourier_preds.iloc[-1]
            rf_latest = rf_preds.iloc[-1]
            
            fourier_nums = [fourier_latest[f'ball_{i}'] for i in range(1, 6)]
            rf_nums = [rf_latest[f'ball_{i}'] for i in range(1, 6)]
            
            print(f"  Fourier: {fourier_nums}")
            print(f"  Random Forest: {rf_nums}")
            
            # Calculate overlap
            overlap = set(fourier_nums).intersection(set(rf_nums))
            print(f"  Agreement: {len(overlap)}/5 numbers ({sorted(list(overlap)) if overlap else 'None'})")
            
            # Calculate similarity metrics
            fourier_sum = sum(fourier_nums)
            rf_sum = sum(rf_nums)
            sum_diff = abs(fourier_sum - rf_sum)
            
            print(f"  Sum difference: {sum_diff} (Fourier: {fourier_sum}, RF: {rf_sum})")
    
    def create_consistency_visualization(self, save_path: str = "prediction_consistency.png"):
        """Create visualization of prediction consistency over time."""
        if len(self.history_df) == 0:
            print("No historical data to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Powerball Prediction Consistency Analysis', fontsize=16)
        
        # 1. Prediction timeline
        ax1 = axes[0, 0]
        
        for model in ['fourier', 'random_forest']:
            model_data = self.history_df[self.history_df['model_type'] == model]
            if len(model_data) > 0:
                # Plot sum of predictions over time
                model_data = model_data.copy()
                model_data['prediction_sum'] = (
                    model_data['ball_1'] + model_data['ball_2'] + 
                    model_data['ball_3'] + model_data['ball_4'] + model_data['ball_5']
                )
                
                ax1.plot(model_data['prediction_made_date'], 
                        model_data['prediction_sum'], 
                        'o-', label=model.replace('_', ' ').title(), alpha=0.7)
        
        ax1.set_title('Prediction Sum Timeline')
        ax1.set_xlabel('Prediction Date')
        ax1.set_ylabel('Sum of Predicted Numbers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Model agreement over time
        ax2 = axes[0, 1]
        
        # Calculate agreement for each target date
        agreement_data = []
        for target_date in self.history_df['target_drawing_date'].unique():
            target_preds = self.history_df[
                self.history_df['target_drawing_date'] == target_date
            ]
            
            fourier_preds = target_preds[target_preds['model_type'] == 'fourier']
            rf_preds = target_preds[target_preds['model_type'] == 'random_forest']
            
            if len(fourier_preds) > 0 and len(rf_preds) > 0:
                fourier_latest = fourier_preds.iloc[-1]
                rf_latest = rf_preds.iloc[-1]
                
                fourier_nums = set([fourier_latest[f'ball_{i}'] for i in range(1, 6)])
                rf_nums = set([rf_latest[f'ball_{i}'] for i in range(1, 6)])
                
                overlap = len(fourier_nums.intersection(rf_nums))
                agreement_data.append({
                    'target_date': target_date,
                    'agreement': overlap
                })
        
        if agreement_data:
            agreement_df = pd.DataFrame(agreement_data)
            ax2.plot(agreement_df['target_date'], agreement_df['agreement'], 'ro-', alpha=0.7)
            ax2.set_title('Model Agreement Over Time')
            ax2.set_xlabel('Target Drawing Date')
            ax2.set_ylabel('Number of Agreeing Numbers (0-5)')
            ax2.set_ylim(-0.5, 5.5)
            ax2.grid(True, alpha=0.3)
        
        # 3. Prediction distribution
        ax3 = axes[1, 0]
        
        all_predictions = []
        for i in range(1, 6):
            all_predictions.extend(self.history_df[f'ball_{i}'].values)
        
        ax3.hist(all_predictions, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of All Predicted Numbers')
        ax3.set_xlabel('Predicted Number')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Model comparison statistics
        ax4 = axes[1, 1]
        
        model_stats = []
        for model in ['fourier', 'random_forest']:
            model_data = self.history_df[self.history_df['model_type'] == model]
            if len(model_data) > 0:
                all_nums = []
                for i in range(1, 6):
                    all_nums.extend(model_data[f'ball_{i}'].values)
                
                model_stats.append({
                    'model': model.replace('_', ' ').title(),
                    'mean': np.mean(all_nums),
                    'std': np.std(all_nums),
                    'min': np.min(all_nums),
                    'max': np.max(all_nums)
                })
        
        if model_stats:
            stats_df = pd.DataFrame(model_stats)
            x = range(len(stats_df))
            
            ax4.bar([i - 0.2 for i in x], stats_df['mean'], 0.4, 
                   label='Mean', alpha=0.7)
            ax4.bar([i + 0.2 for i in x], stats_df['std'], 0.4, 
                   label='Std Dev', alpha=0.7)
            
            ax4.set_title('Model Statistics Comparison')
            ax4.set_xlabel('Model')
            ax4.set_ylabel('Value')
            ax4.set_xticks(x)
            ax4.set_xticklabels(stats_df['model'])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Consistency visualization saved to {save_path}")
        
        return fig
    
    def get_summary_stats(self):
        """Get summary statistics for historical predictions."""
        if len(self.history_df) == 0:
            print("No historical predictions to analyze")
            return None
        
        print(f"\n{'='*50}")
        print("HISTORICAL PREDICTION SUMMARY")
        print(f"{'='*50}")
        
        print(f"Total predictions: {len(self.history_df)}")
        print(f"Date range: {self.history_df['prediction_made_date'].min()} to {self.history_df['prediction_made_date'].max()}")
        
        model_counts = self.history_df['model_type'].value_counts()
        print(f"\nPredictions by model:")
        for model, count in model_counts.items():
            print(f"  {model}: {count}")
        
        target_dates = self.history_df['target_drawing_date'].unique()
        print(f"\nUnique target dates: {len(target_dates)}")
        
        # Check for data hash changes (indicating new training data)
        hash_changes = self.history_df['data_hash'].nunique()
        print(f"Data version changes: {hash_changes}")
        
        return {
            'total_predictions': len(self.history_df),
            'model_counts': model_counts.to_dict(),
            'target_dates': len(target_dates),
            'data_versions': hash_changes
        }


def main():
    """Example usage of PredictionTracker."""
    tracker = PredictionTracker()
    
    # Show current summary
    tracker.get_summary_stats()
    
    # Analyze consistency if we have data
    if len(tracker.history_df) > 0:
        print("\nAnalyzing Fourier model consistency...")
        tracker.analyze_consistency('fourier', days_back=7)
        
        print("\nAnalyzing Random Forest model consistency...")
        tracker.analyze_consistency('random_forest', days_back=7)
        
        print("\nAnalyzing model agreement...")
        tracker.analyze_model_agreement()
        
        # Create visualization
        tracker.create_consistency_visualization()
    else:
        print("\nNo historical predictions found. Use add_prediction() to start tracking.")


if __name__ == "__main__":
    main()