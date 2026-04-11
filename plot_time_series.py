#!/usr/bin/env python3
"""
Time Series Plotting for Powerball Predictions
Creates actual vs predicted visualizations for both Fourier and Random Forest models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import os

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class PowerballTimeSeriesPlotter:
    """
    Creates comprehensive time series plots comparing actual lottery numbers
    with Fourier and Random Forest predictions.
    """
    
    def __init__(self, data_file: str = "powerball_games_only.csv"):
        self.data_file = data_file
        self.load_data()
        
    def load_data(self):
        """Load the historical lottery data."""
        try:
            self.df = pd.read_csv(self.data_file)
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df = self.df.sort_values('date').reset_index(drop=True)
            print(f"✓ Loaded {len(self.df)} historical drawings")
        except FileNotFoundError:
            print(f"❌ Could not find {self.data_file}")
            raise
    
    def create_fourier_predictions_series(self, n_harmonics: int = 15, prediction_days: int = 30):
        """
        Generate Fourier predictions for the last N days to create a prediction time series.
        """
        print(f"Generating Fourier prediction series for last {prediction_days} days...")
        
        # Import the Fourier model
        try:
            from predict_powerball_fourier_bins import FourierSeriesBinGPU, BinClassifier
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except ImportError:
            print("❌ Could not import Fourier model")
            return None
        
        # Use data up to N days ago for training, predict the recent period
        cutoff_date = self.df['date'].max() - timedelta(days=prediction_days)
        train_df = self.df[self.df['date'] <= cutoff_date].copy()
        actual_df = self.df[self.df['date'] > cutoff_date].copy()
        
        if len(train_df) < 100 or len(actual_df) == 0:
            print("❌ Insufficient data for Fourier prediction series")
            return None
        
        print(f"Training on {len(train_df)} drawings, predicting {len(actual_df)} recent drawings")
        
        # Convert dates to numeric for training
        first_date = train_df['date'].min()
        train_df['days_since_start'] = (train_df['date'] - first_date).dt.days.values

        # Fit one classifier over the training set and reuse per-ball bin edges.
        classifier = BinClassifier()
        classifier.fit(train_df)
        train_bins_df = classifier.transform_to_bins(train_df)
        
        predictions = {}
        
        for ball_num in range(1, 6):
            col_name = f'ball_{ball_num}'
            
            # Fit Fourier model on training data
            x_train = train_df['days_since_start'].values.astype(float)
            y_train = train_df[col_name].values.astype(float)
            
            # Use bin-based approach
            train_bins = train_bins_df[f'bin_{ball_num}'].values.astype(float)
            
            model = FourierSeriesBinGPU(n_harmonics=n_harmonics, device=device)
            model.fit(x_train, train_bins, lr=0.01, epochs=2000)
            
            # Predict for recent dates
            recent_predictions = []
            for _, row in actual_df.iterrows():
                days_since = (row['date'] - first_date).days
                x_pred = np.array([days_since], dtype=float)
                bin_pred = model.predict(x_pred)[0]
                # Convert bin back to number (use middle of bin range)
                bin_edges = classifier.bin_edges[col_name]
                bin_idx = int(np.clip(np.round(bin_pred) - 1, 0, len(bin_edges) - 2))
                number_pred = int((bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2)
                number_pred = max(1, min(69, number_pred))
                recent_predictions.append(number_pred)
            
            predictions[col_name] = recent_predictions
        
        # Create prediction DataFrame
        pred_df = actual_df[['date']].copy()
        for ball_num in range(1, 6):
            col_name = f'ball_{ball_num}'
            pred_df[f'{col_name}_fourier_pred'] = predictions[col_name]
        
        return pred_df
    
    def create_bin_predictions_series(self, prediction_days: int = 30):
        """
        Generate Random Forest bin predictions for the last N days.
        """
        print(f"Generating Random Forest prediction series for last {prediction_days} days...")
        
        try:
            from predict_powerball_bins_gpu_enhanced import EnhancedGPURandomForestBinPredictor
        except ImportError:
            print("❌ Could not import Random Forest model")
            return None
        
        # Use data up to N days ago for training
        cutoff_date = self.df['date'].max() - timedelta(days=prediction_days)
        train_df = self.df[self.df['date'] <= cutoff_date].copy()
        actual_df = self.df[self.df['date'] > cutoff_date].copy()
        
        if len(train_df) < 200 or len(actual_df) == 0:
            print("❌ Insufficient data for Random Forest prediction series")
            return None
        
        print(f"Training on {len(train_df)} drawings, predicting {len(actual_df)} recent drawings")
        
        # Train Random Forest model
        predictor = EnhancedGPURandomForestBinPredictor(sequence_length=21, use_gpu=False)
        predictor.fit(train_df, optimize_length=False, train_test_split_ratio=0.8)
        
        # Generate rolling predictions
        predictions = {}
        for ball_num in range(1, 6):
            predictions[f'ball_{ball_num}'] = []
            predictions[f'bin_{ball_num}'] = []
        
        # Use expanding window approach
        for i in range(len(actual_df)):
            # Use training data + actual data up to current point
            current_train_df = pd.concat([
                train_df,
                actual_df.iloc[:i]
            ]).reset_index(drop=True)
            
            if len(current_train_df) >= predictor.sequence_length:
                try:
                    pred_result = predictor.predict_next_bins(current_train_df)
                    
                    for ball_num in range(1, 6):
                        col_name = f'ball_{ball_num}'
                        predictions[col_name].append(pred_result[col_name]['number_prediction'])
                        predictions[f'bin_{ball_num}'].append(pred_result[col_name]['bin_prediction'])
                except:
                    # Use previous prediction or default
                    for ball_num in range(1, 6):
                        col_name = f'ball_{ball_num}'
                        last_val = predictions[col_name][-1] if predictions[col_name] else 35
                        predictions[col_name].append(last_val)
                        predictions[f'bin_{ball_num}'].append(2)
            else:
                # Not enough data, use historical mean
                for ball_num in range(1, 6):
                    col_name = f'ball_{ball_num}'
                    mean_val = int(current_train_df[col_name].mean())
                    predictions[col_name].append(mean_val)
                    predictions[f'bin_{ball_num}'].append(2)
        
        # Create prediction DataFrame
        pred_df = actual_df[['date']].copy()
        for ball_num in range(1, 6):
            col_name = f'ball_{ball_num}'
            pred_df[f'{col_name}_rf_pred'] = predictions[col_name]
            pred_df[f'{col_name}_rf_bin'] = predictions[f'bin_{ball_num}']
        
        return pred_df
    
    def plot_time_series_comparison(self, prediction_days: int = 60, save_path: str = None):
        """
        Create comprehensive time series plots comparing actual vs predicted values.
        """
        print(f"\nCreating time series comparison plots for last {prediction_days} days...")
        
        # Generate prediction series
        fourier_preds = self.create_fourier_predictions_series(prediction_days=prediction_days)
        rf_preds = self.create_bin_predictions_series(prediction_days=prediction_days)
        
        if fourier_preds is None and rf_preds is None:
            print("❌ Could not generate any predictions")
            return None
        
        # Get recent actual data
        recent_df = self.df.tail(prediction_days).copy()
        
        # Create the plot
        fig, axes = plt.subplots(5, 1, figsize=(16, 20))
        fig.suptitle(f'Powerball Time Series: Actual vs Predicted (Last {prediction_days} Days)', fontsize=16)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for ball_num in range(1, 6):
            ax = axes[ball_num - 1]
            col_name = f'ball_{ball_num}'
            
            # Plot actual values
            ax.plot(recent_df['date'], recent_df[col_name], 
                   'o-', color=colors[ball_num-1], linewidth=2, markersize=4,
                   label=f'Actual Ball {ball_num}', alpha=0.8)
            
            # Plot Fourier predictions if available
            if fourier_preds is not None:
                fourier_col = f'{col_name}_fourier_pred'
                if fourier_col in fourier_preds.columns:
                    ax.plot(fourier_preds['date'], fourier_preds[fourier_col],
                           's--', color='red', linewidth=1.5, markersize=3,
                           label='Fourier Prediction', alpha=0.7)
            
            # Plot Random Forest predictions if available
            if rf_preds is not None:
                rf_col = f'{col_name}_rf_pred'
                if rf_col in rf_preds.columns:
                    ax.plot(rf_preds['date'], rf_preds[rf_col],
                           '^:', color='green', linewidth=1.5, markersize=3,
                           label='Random Forest Prediction', alpha=0.7)
            
            # Formatting
            ax.set_title(f'Ball {ball_num} - Time Series Comparison', fontsize=12)
            ax.set_ylabel('Number Value', fontsize=10)
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(1, 69)
            
            # Format x-axis dates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, prediction_days//10)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # Add overall statistics
        self._add_prediction_statistics(axes, recent_df, fourier_preds, rf_preds)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Time series plot saved to {save_path}")
        
        return fig
    
    def _add_prediction_statistics(self, axes, actual_df, fourier_preds, rf_preds):
        """Add prediction accuracy statistics to the plots."""
        if fourier_preds is not None or rf_preds is not None:
            # Merge dataframes for comparison
            comparison_df = actual_df[['date'] + [f'ball_{i}' for i in range(1, 6)]].copy()
            
            if fourier_preds is not None:
                comparison_df = pd.merge(comparison_df, fourier_preds, on='date', how='inner')
            
            if rf_preds is not None:
                comparison_df = pd.merge(comparison_df, rf_preds, on='date', how='inner')
            
            # Calculate statistics for each ball
            stats_text = "Prediction Accuracy (MAE):\n"
            
            for ball_num in range(1, 6):
                col_name = f'ball_{ball_num}'
                actual_vals = comparison_df[col_name]
                
                ball_stats = f"Ball {ball_num}: "
                
                if fourier_preds is not None and f'{col_name}_fourier_pred' in comparison_df.columns:
                    fourier_mae = np.mean(np.abs(actual_vals - comparison_df[f'{col_name}_fourier_pred']))
                    ball_stats += f"Fourier MAE={fourier_mae:.1f} "
                
                if rf_preds is not None and f'{col_name}_rf_pred' in comparison_df.columns:
                    rf_mae = np.mean(np.abs(actual_vals - comparison_df[f'{col_name}_rf_pred']))
                    ball_stats += f"RF MAE={rf_mae:.1f}"
                
                # Add text to corresponding subplot
                axes[ball_num - 1].text(0.02, 0.98, ball_stats, 
                                       transform=axes[ball_num - 1].transAxes,
                                       verticalalignment='top', fontsize=8,
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def create_prediction_accuracy_heatmap(self, prediction_days: int = 30, save_path: str = None):
        """
        Create a heatmap showing prediction accuracy for each ball over time.
        """
        print(f"Creating prediction accuracy heatmap for last {prediction_days} days...")
        
        fourier_preds = self.create_fourier_predictions_series(prediction_days=prediction_days)
        rf_preds = self.create_bin_predictions_series(prediction_days=prediction_days)
        
        if fourier_preds is None and rf_preds is None:
            print("❌ Could not generate predictions for heatmap")
            return None
        
        recent_df = self.df.tail(prediction_days).copy()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Prediction Accuracy Heatmap (Last {prediction_days} Days)', fontsize=16)
        
        # Fourier accuracy heatmap
        if fourier_preds is not None:
            fourier_accuracy = np.zeros((5, len(recent_df)))
            
            comparison_df = pd.merge(recent_df, fourier_preds, on='date', how='inner')
            
            for i, ball_num in enumerate(range(1, 6)):
                col_name = f'ball_{ball_num}'
                pred_col = f'{col_name}_fourier_pred'
                
                if pred_col in comparison_df.columns:
                    actual = comparison_df[col_name].values
                    predicted = comparison_df[pred_col].values
                    errors = np.abs(actual - predicted)
                    fourier_accuracy[i, :len(errors)] = errors
            
            im1 = axes[0].imshow(fourier_accuracy, aspect='auto', cmap='RdYlBu_r', 
                               vmin=0, vmax=20)
            axes[0].set_title('Fourier Series - Absolute Prediction Errors')
            axes[0].set_ylabel('Ball Number')
            axes[0].set_yticks(range(5))
            axes[0].set_yticklabels([f'Ball {i+1}' for i in range(5)])
            
            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=axes[0])
            cbar1.set_label('Absolute Error')
        
        # Random Forest accuracy heatmap
        if rf_preds is not None:
            rf_accuracy = np.zeros((5, len(recent_df)))
            
            comparison_df = pd.merge(recent_df, rf_preds, on='date', how='inner')
            
            for i, ball_num in enumerate(range(1, 6)):
                col_name = f'ball_{ball_num}'
                pred_col = f'{col_name}_rf_pred'
                
                if pred_col in comparison_df.columns:
                    actual = comparison_df[col_name].values
                    predicted = comparison_df[pred_col].values
                    errors = np.abs(actual - predicted)
                    rf_accuracy[i, :len(errors)] = errors
            
            im2 = axes[1].imshow(rf_accuracy, aspect='auto', cmap='RdYlBu_r',
                               vmin=0, vmax=20)
            axes[1].set_title('Random Forest - Absolute Prediction Errors')
            axes[1].set_ylabel('Ball Number')
            axes[1].set_xlabel('Drawing Index (Recent → Older)')
            axes[1].set_yticks(range(5))
            axes[1].set_yticklabels([f'Ball {i+1}' for i in range(5)])
            
            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=axes[1])
            cbar2.set_label('Absolute Error')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Accuracy heatmap saved to {save_path}")
        
        return fig
    
    def create_comprehensive_analysis(self, prediction_days: int = 60, save_dir: str = "time_series_analysis"):
        """
        Create comprehensive time series analysis with multiple plots.
        """
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE TIME SERIES ANALYSIS")
        print(f"{'='*70}")
        
        # Create output directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate main time series plot
        ts_plot = self.plot_time_series_comparison(
            prediction_days=prediction_days,
            save_path=f"{save_dir}/actual_vs_predicted_timeseries.png"
        )
        
        # Generate accuracy heatmap
        heatmap_plot = self.create_prediction_accuracy_heatmap(
            prediction_days=min(prediction_days, 30),
            save_path=f"{save_dir}/prediction_accuracy_heatmap.png"
        )
        
        # Create summary statistics
        self._create_summary_statistics(prediction_days, save_dir)
        
        print(f"\n✅ Comprehensive analysis complete!")
        print(f"📁 Results saved to: {save_dir}/")
        print(f"📊 Main plot: actual_vs_predicted_timeseries.png")
        print(f"🔥 Heatmap: prediction_accuracy_heatmap.png")
        print(f"📈 Statistics: summary_statistics.txt")
    
    def _create_summary_statistics(self, prediction_days: int, save_dir: str):
        """Create detailed summary statistics file."""
        print("Generating summary statistics...")
        
        fourier_preds = self.create_fourier_predictions_series(prediction_days=prediction_days)
        rf_preds = self.create_bin_predictions_series(prediction_days=prediction_days)
        recent_df = self.df.tail(prediction_days).copy()
        
        with open(f"{save_dir}/summary_statistics.txt", "w") as f:
            f.write(f"POWERBALL PREDICTION ANALYSIS SUMMARY\n")
            f.write(f"{'='*50}\n\n")
            f.write(f"Analysis Period: Last {prediction_days} days\n")
            f.write(f"Total Drawings Analyzed: {len(recent_df)}\n")
            f.write(f"Date Range: {recent_df['date'].min()} to {recent_df['date'].max()}\n\n")
            
            # Fourier statistics
            if fourier_preds is not None:
                f.write(f"FOURIER SERIES MODEL PERFORMANCE\n")
                f.write(f"{'-'*40}\n")
                
                comparison_df = pd.merge(recent_df, fourier_preds, on='date', how='inner')
                
                for ball_num in range(1, 6):
                    col_name = f'ball_{ball_num}'
                    pred_col = f'{col_name}_fourier_pred'
                    
                    if pred_col in comparison_df.columns:
                        actual = comparison_df[col_name]
                        predicted = comparison_df[pred_col]
                        
                        mae = np.mean(np.abs(actual - predicted))
                        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                        exact_matches = np.sum(actual == predicted)
                        
                        f.write(f"Ball {ball_num}:\n")
                        f.write(f"  Mean Absolute Error: {mae:.2f}\n")
                        f.write(f"  Root Mean Square Error: {rmse:.2f}\n")
                        f.write(f"  Exact Matches: {exact_matches}/{len(actual)} ({exact_matches/len(actual)*100:.1f}%)\n\n")
            
            # Random Forest statistics
            if rf_preds is not None:
                f.write(f"RANDOM FOREST MODEL PERFORMANCE\n")
                f.write(f"{'-'*40}\n")
                
                comparison_df = pd.merge(recent_df, rf_preds, on='date', how='inner')
                
                for ball_num in range(1, 6):
                    col_name = f'ball_{ball_num}'
                    pred_col = f'{col_name}_rf_pred'
                    
                    if pred_col in comparison_df.columns:
                        actual = comparison_df[col_name]
                        predicted = comparison_df[pred_col]
                        
                        mae = np.mean(np.abs(actual - predicted))
                        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
                        exact_matches = np.sum(actual == predicted)
                        
                        f.write(f"Ball {ball_num}:\n")
                        f.write(f"  Mean Absolute Error: {mae:.2f}\n")
                        f.write(f"  Root Mean Square Error: {rmse:.2f}\n")
                        f.write(f"  Exact Matches: {exact_matches}/{len(actual)} ({exact_matches/len(actual)*100:.1f}%)\n\n")
            
            f.write(f"NOTES:\n")
            f.write(f"- MAE: Average absolute difference between predicted and actual\n")
            f.write(f"- RMSE: Root mean square error (penalizes larger errors more)\n")
            f.write(f"- Exact Matches: Number of perfect predictions\n")
            f.write(f"- Random baseline MAE ≈ 17 (for uniform distribution 1-69)\n")


def main():
    """Main execution function."""
    plotter = PowerballTimeSeriesPlotter()
    
    # Create comprehensive analysis
    plotter.create_comprehensive_analysis(
        prediction_days=90,  # Analyze last 90 days
        save_dir="time_series_analysis"
    )


if __name__ == "__main__":
    main()