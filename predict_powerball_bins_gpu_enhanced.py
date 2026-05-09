#!/usr/bin/env python3
"""
Enhanced GPU-Accelerated Random Forest Bin Prediction for Powerball

This version focuses specifically on using sequential bin hit patterns as predictors.
Uses sequence of historical bin hits to predict which bins the next 5 numbers will fall into.

The key improvement: Creates ball-specific sequential features that capture temporal patterns
in bin transitions for each ball position independently.

Author: Assistant
Date: October 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import joblib

# GPU acceleration
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    GPU_AVAILABLE = True
    print("✓ cuML GPU acceleration available")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ cuML not available, using CPU-only sklearn")

# PyTorch for GPU info
try:
    import torch
    if torch.cuda.is_available():
        print(f"PyTorch device: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("PyTorch: No CUDA GPU detected")
except ImportError:
    print("PyTorch not available for GPU detection")


# Target columns the predictor models. White balls 1..5 are drawn from 1-69
# without replacement; the red Powerball is drawn from a separate 1-26 pool.
TARGET_COLUMNS = ['ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5', 'powerball']

VALID_RANGES = {
    'ball_1': 69, 'ball_2': 69, 'ball_3': 69, 'ball_4': 69, 'ball_5': 69,
    'powerball': 26,
}


def bin_col_for(col_name: str) -> str:
    """Map a source column (e.g. 'ball_3', 'powerball') to its bin column."""
    if col_name == 'powerball':
        return 'bin_pb'
    return col_name.replace('ball_', 'bin_')


class BinClassifier:
    """
    Classify numbers into statistical bins based on historical distribution.
    """
    
    def __init__(self, n_bins: int = 6, target_columns: Optional[List[str]] = None):
        self.n_bins = n_bins
        self.bin_edges = {}
        self.fitted = False
        self.target_columns = list(target_columns) if target_columns is not None else list(TARGET_COLUMNS)

    def fit(self, df: pd.DataFrame):
        """Fit bin edges based on historical data."""
        for col_name in self.target_columns:
            if col_name not in df.columns:
                print(f"⚠ Skipping {col_name} (not in dataframe)")
                continue
            values = df[col_name].values

            # Quantile-based edges for an even sample distribution.
            bin_edges = np.quantile(values, np.linspace(0, 1, self.n_bins + 1))
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < self.n_bins + 1:
                # Fallback to equal-width bins (e.g. powerball with only 26 values).
                bin_edges = np.linspace(values.min(), values.max() + 0.1, self.n_bins + 1)

            self.bin_edges[col_name] = bin_edges

            print(f"{col_name} bins: {[f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]}")

        self.fitted = True

    def transform_to_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numbers to bin classifications."""
        if not self.fitted:
            raise ValueError("BinClassifier not fitted yet!")

        df_bins = df.copy()

        for col_name in self.target_columns:
            if col_name not in df.columns or col_name not in self.bin_edges:
                continue
            bin_col = bin_col_for(col_name)

            bins = np.digitize(df[col_name].values, self.bin_edges[col_name]) - 1
            bins = np.clip(bins, 0, self.n_bins - 1) + 1  # 1..n_bins

            df_bins[bin_col] = bins

        return df_bins


class SequentialBinFeatureExtractor:
    """
    Enhanced feature extractor that creates ball-specific sequential features.
    
    Key improvement: For each ball position, creates features from the sequence
    of bin hits for that specific ball, rather than flattening all bins together.
    """
    
    def __init__(self, sequence_length: int = 15):
        self.sequence_length = sequence_length
        print(f"Sequential feature extractor: {sequence_length} time steps per ball")
    
    def create_ball_specific_sequences(self, df_bins: pd.DataFrame, col_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences specifically for one target column ('ball_1'..'ball_5'
        or 'powerball').

        Returns:
            X: Features (previous sequence_length bin hits for this column)
            y: Targets (next bin hit for this column)
        """
        bin_col = bin_col_for(col_name)

        bin_sequence = df_bins[bin_col].values
        n_samples = len(bin_sequence)

        if n_samples <= self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length + 1} samples for {col_name}")
        
        X, y = [], []
        
        for i in range(self.sequence_length, n_samples):
            # Previous sequence_length bin hits for this ball as features
            sequence_features = bin_sequence[i - self.sequence_length:i]
            
            # Add additional temporal features
            enhanced_features = self._create_enhanced_features(sequence_features)
            
            X.append(enhanced_features)
            
            # Next bin hit for this ball as target
            y.append(bin_sequence[i])
        
        return np.array(X), np.array(y)
    
    def _create_enhanced_features(self, sequence: np.ndarray) -> np.ndarray:
        """
        Create enhanced features from a bin sequence.
        
        Features include:
        - Raw sequence values
        - Recent pattern indicators (last 3, last 5)
        - Trend features (is increasing, is decreasing)
        - Frequency features (mode, unique count)
        - Transition features (number of changes)
        """
        features = []
        
        # 1. Raw sequence (most important)
        features.extend(sequence)
        
        # 2. Recent pattern indicators
        if len(sequence) >= 3:
            features.extend(sequence[-3:])  # Last 3 values
        else:
            features.extend([0] * 3)
            
        if len(sequence) >= 5:
            features.extend(sequence[-5:])  # Last 5 values
        else:
            features.extend([0] * 5)
        
        # 3. Statistical features
        features.append(np.mean(sequence))  # Average bin
        features.append(np.std(sequence))   # Volatility
        features.append(np.median(sequence))  # Median bin
        
        # 4. Trend features
        if len(sequence) >= 2:
            recent_trend = np.mean(np.diff(sequence[-5:]) if len(sequence) >= 5 else np.diff(sequence))
            features.append(recent_trend)  # Recent trend direction
            features.append(1 if recent_trend > 0 else 0)  # Is trending up
            features.append(1 if recent_trend < 0 else 0)  # Is trending down
        else:
            features.extend([0, 0, 0])
        
        # 5. Pattern features
        unique_count = len(np.unique(sequence))
        features.append(unique_count)  # Diversity of bins
        
        # Most frequent bin in sequence
        from scipy.stats import mode
        try:
            mode_result = mode(sequence, keepdims=True)
            features.append(mode_result.mode[0])
            features.append(mode_result.count[0])  # Frequency of mode
        except:
            features.extend([np.median(sequence), 1])
        
        # 6. Transition features
        if len(sequence) >= 2:
            transitions = np.sum(np.diff(sequence) != 0)  # Number of bin changes
            features.append(transitions)
            
            # Recent stability (no changes in last 3)
            recent_changes = np.sum(np.diff(sequence[-3:]) != 0) if len(sequence) >= 3 else 0
            features.append(1 if recent_changes == 0 else 0)
        else:
            features.extend([0, 0])
        
        return np.array(features)
    
    def get_latest_sequence_for_ball(self, df_bins: pd.DataFrame, col_name: str) -> np.ndarray:
        """Get the most recent sequence for a target column for prediction."""
        bin_col = bin_col_for(col_name)
        bin_sequence = df_bins[bin_col].values

        if len(bin_sequence) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} samples for {col_name}")

        latest_sequence = bin_sequence[-self.sequence_length:]
        enhanced_features = self._create_enhanced_features(latest_sequence)

        return enhanced_features.reshape(1, -1)


class EnhancedGPURandomForestBinPredictor:
    """
    Enhanced GPU-Accelerated Random Forest for predicting bin classifications.
    
    Key improvements:
    1. Ball-specific sequential modeling
    2. Enhanced feature engineering with temporal patterns
    3. Better sequence length optimization
    4. Improved validation methodology
    """
    
    def __init__(self, sequence_length: int = 15, use_gpu: bool = None,
                 target_columns: Optional[List[str]] = None):
        self.sequence_length = sequence_length
        self.use_gpu = GPU_AVAILABLE if use_gpu is None else (use_gpu and GPU_AVAILABLE)
        self.target_columns = list(target_columns) if target_columns is not None else list(TARGET_COLUMNS)
        self.bin_classifier = BinClassifier(n_bins=6, target_columns=self.target_columns)
        self.feature_extractor = SequentialBinFeatureExtractor(sequence_length)
        self.models = {}  # One model per target column
        self.fitted = False
        self.df_test = None
        
        print(f"\n🚀 Enhanced GPU Random Forest Bin Predictor")
        print(f"Using GPU acceleration: {self.use_gpu}")
        print(f"Sequence length: {sequence_length}")
        print(f"Enhanced features per ball: ~{3 * sequence_length + 15} features")
    
    def optimize_sequence_length(self, df_train: pd.DataFrame, min_length: int = 8, max_length: int = 25) -> int:
        """
        Find optimal sequence length using enhanced ball-specific validation.
        """
        print("\n--- Enhanced Sequence Length Optimization ---")
        
        # Create temporary bin classifier
        temp_bin_classifier = BinClassifier(target_columns=self.target_columns)
        temp_bin_classifier.fit(df_train)
        df_bins = temp_bin_classifier.transform_to_bins(df_train)

        best_length = min_length
        best_score = 0
        scores_by_length = {}

        for seq_len in range(min_length, min(max_length + 1, len(df_train) // 3)):
            try:
                extractor = SequentialBinFeatureExtractor(seq_len)

                # Test on all target columns and average the performance
                ball_scores = []

                for col_name in self.target_columns:
                    try:
                        X, y = extractor.create_ball_specific_sequences(df_bins, col_name)
                        
                        if len(X) < 30:  # Need minimum samples
                            continue
                        
                        # Temporal split (use last 20% as validation)
                        split_idx = int(len(X) * 0.8)
                        X_train, X_val = X[:split_idx], X[split_idx:]
                        y_train, y_val = y[:split_idx], y[split_idx:]
                        
                        # Train quick model
                        if self.use_gpu and len(X_train) > 100:
                            model = CuMLRandomForestClassifier(
                                n_estimators=50,
                                max_depth=10,
                                random_state=42
                            )
                        else:
                            model = RandomForestClassifier(
                                n_estimators=50,
                                max_depth=10,
                                random_state=42,
                                n_jobs=-1
                            )
                        
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_val)
                        score = accuracy_score(y_val, predictions)
                        ball_scores.append(score)
                        
                    except Exception as e:
                        continue
                
                if ball_scores:
                    avg_score = np.mean(ball_scores)
                    scores_by_length[seq_len] = avg_score
                    print(f"Sequence length {seq_len:2d}: Avg accuracy = {avg_score:.4f} (balls: {len(ball_scores)})")
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_length = seq_len
                        
            except Exception as e:
                print(f"Error with sequence length {seq_len}: {e}")
                continue
        
        print(f"\n✓ Optimal sequence length: {best_length} (avg accuracy: {best_score:.4f})")
        
        # Show progression
        if len(scores_by_length) > 1:
            print("\nSequence Length Performance:")
            for length in sorted(scores_by_length.keys()):
                marker = " ← BEST" if length == best_length else ""
                print(f"  {length:2d}: {scores_by_length[length]:.4f}{marker}")
        
        return best_length
    
    def fit(self, df: pd.DataFrame, optimize_length: bool = True, train_test_split_ratio: float = 0.67):
        """
        Fit enhanced Random Forest models with ball-specific sequences.
        """
        print("\n" + "="*70)
        print("TRAINING ENHANCED GPU RANDOM FOREST BIN PREDICTOR")
        print("="*70)
        
        # Chronological split
        split_index = int(len(df) * train_test_split_ratio)
        df_train = df.iloc[:split_index].copy()
        df_test = df.iloc[split_index:].copy()
        
        print(f"\nDataset Split (Chronological):")
        print(f"Full dataset: {len(df)} drawings ({df['date'].min()} to {df['date'].max()})")
        print(f"Training: {len(df_train)} drawings ({df_train['date'].min()} to {df_train['date'].max()})")
        print(f"Testing: {len(df_test)} drawings ({df_test['date'].min()} to {df_test['date'].max()})")
        print(f"Split ratio: {train_test_split_ratio:.1%} train / {1-train_test_split_ratio:.1%} test")
        
        self.df_test = df_test
        
        # Optimize sequence length
        if optimize_length:
            optimal_length = self.optimize_sequence_length(df_train)
            self.sequence_length = optimal_length
            self.feature_extractor = SequentialBinFeatureExtractor(optimal_length)
        
        # Fit bin classifier
        self.bin_classifier.fit(df_train)
        df_train_bins = self.bin_classifier.transform_to_bins(df_train)
        
        # Show bin distribution
        print("\n--- Training Data Bin Distribution ---")
        for col_name in self.target_columns:
            bin_col = bin_col_for(col_name)
            if bin_col not in df_train_bins.columns:
                continue
            counts = df_train_bins[bin_col].value_counts().sort_index()
            print(f"{bin_col}: {dict(counts)}")

        # Train separate model for each target column
        for col_name in self.target_columns:
            if bin_col_for(col_name) not in df_train_bins.columns:
                continue

            print(f"\n--- Training Enhanced {col_name} Model ---")

            X_train, y_train = self.feature_extractor.create_ball_specific_sequences(df_train_bins, col_name)
            print(f"{col_name} training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            
            # Enhanced model configuration
            if self.use_gpu:
                model = CuMLRandomForestClassifier(
                    n_estimators=300,  # More trees for better pattern capture
                    max_depth=20,      # Deeper trees for complex patterns
                    max_features='sqrt',
                    min_samples_split=5,
                    random_state=42
                )
                print("  Using GPU-accelerated cuML Random Forest")
            else:
                model = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=20,
                    max_features='sqrt',
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
                print("  Using CPU sklearn Random Forest")
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Training performance
            train_pred = model.predict(X_train)
            train_acc = accuracy_score(y_train, train_pred)
            random_baseline = 1/6
            improvement = (train_acc - random_baseline) / random_baseline * 100
            
            print(f"  Training accuracy: {train_acc:.4f}")
            print(f"  Improvement over random: {improvement:+.1f}%")
            
            # Feature importance analysis
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                top_indices = np.argsort(importances)[-10:][::-1]
                print(f"  Top 10 feature importances:")
                for i, idx in enumerate(top_indices[:5]):
                    print(f"    {i+1}. Feature {idx}: {importances[idx]:.4f}")
                print(f"    ... (showing top 5 of {len(importances)} features)")
            
            self.models[col_name] = model
        
        self.fitted = True
        print(f"\n✅ All enhanced models trained successfully!")
        return self
    
    def evaluate_on_test_set(self):
        """Evaluate models on the held-out test set with enhanced metrics."""
        if not self.fitted or self.df_test is None:
            print("⚠ Model not fitted or no test data available")
            return
        
        print("\n" + "="*70)
        print("ENHANCED TEST SET EVALUATION")
        print("="*70)
        
        df_test_bins = self.bin_classifier.transform_to_bins(self.df_test)

        all_predictions = {c: [] for c in self.target_columns}
        all_actuals = {c: [] for c in self.target_columns}

        successful_predictions = 0

        for i in range(self.sequence_length, len(self.df_test)):
            try:
                hist_data = df_test_bins.iloc[:i]
                actual_row = df_test_bins.iloc[i]

                for col_name in self.target_columns:
                    if col_name not in self.models:
                        continue
                    bin_col = bin_col_for(col_name)

                    X_latest = self.feature_extractor.get_latest_sequence_for_ball(hist_data, col_name)
                    model = self.models[col_name]
                    bin_pred = model.predict(X_latest)[0]

                    all_predictions[col_name].append(bin_pred)
                    all_actuals[col_name].append(actual_row[bin_col])

                successful_predictions += 1

            except Exception as e:
                continue
        
        print(f"Test predictions made: {successful_predictions} out of {len(self.df_test) - self.sequence_length}")
        
        if successful_predictions > 0:
            # Detailed analysis per ball
            overall_predictions = []
            overall_actuals = []
            
            print(f"\n📊 Test Set Accuracy by Ball Position:")
            for col_name in self.target_columns:
                if col_name not in all_predictions:
                    continue
                if len(all_predictions[col_name]) > 0:
                    acc = accuracy_score(all_actuals[col_name], all_predictions[col_name])
                    random_baseline = 1/6
                    improvement = (acc - random_baseline) / random_baseline * 100
                    
                    print(f"  {col_name}: {acc:.4f} ({improvement:+.1f}% vs random)")
                    
                    overall_predictions.extend(all_predictions[col_name])
                    overall_actuals.extend(all_actuals[col_name])
            
            # Overall performance
            if len(overall_predictions) > 0:
                overall_acc = accuracy_score(overall_actuals, overall_predictions)
                random_baseline = 1/6
                improvement = (overall_acc - random_baseline) / random_baseline * 100
                
                print(f"\n🎯 Overall Test Performance:")
                print(f"   Accuracy: {overall_acc:.4f}")
                print(f"   Random Baseline: {random_baseline:.4f}")
                print(f"   Improvement: {improvement:+.1f}%")
                print(f"   Total Predictions: {len(overall_predictions)}")
                
                if overall_acc > random_baseline * 1.1:  # 10% improvement threshold
                    print("✅ Model significantly beats random baseline!")
                elif overall_acc > random_baseline:
                    print("✓ Model beats random baseline")
                else:
                    print("⚠ Model does not beat random baseline")
                
                # Bin prediction distribution
                print(f"\n📈 Prediction Distribution:")
                unique, counts = np.unique(overall_predictions, return_counts=True)
                for bin_num, count in zip(unique, counts):
                    pct = count / len(overall_predictions) * 100
                    print(f"   Bin {int(bin_num)}: {count:3d} predictions ({pct:5.1f}%)")
        else:
            print("❌ No successful test predictions made")
    
    def predict_next_bins(self, df: pd.DataFrame) -> Dict:
        """Predict bin classifications for the next drawing using enhanced features."""
        if not self.fitted:
            raise ValueError("Model not fitted yet!")

        df_bins = self.bin_classifier.transform_to_bins(df)

        predictions = {}

        for col_name in self.target_columns:
            if col_name not in self.models:
                continue

            X_latest = self.feature_extractor.get_latest_sequence_for_ball(df_bins, col_name)

            model = self.models[col_name]
            bin_pred = model.predict(X_latest)[0]
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    probas = model.predict_proba(X_latest)[0]
                    confidence = np.max(probas)
                except:
                    confidence = 0.5
            else:
                confidence = 0.5
            
            predictions[col_name] = {
                'bin_prediction': int(bin_pred),
                'confidence': float(confidence)
            }
        
        return predictions
    
    def save_model(self, filepath: str):
        """Save the trained models."""
        model_data = {
            'models': self.models,
            'bin_classifier': self.bin_classifier,
            'sequence_length': self.sequence_length,
            'use_gpu': self.use_gpu,
            'fitted': self.fitted
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Enhanced model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        """Load a trained model."""
        model_data = joblib.load(filepath)
        
        instance = cls(
            sequence_length=model_data['sequence_length'],
            use_gpu=model_data['use_gpu']
        )
        
        instance.models = model_data['models']
        instance.bin_classifier = model_data['bin_classifier']
        instance.fitted = model_data['fitted']
        
        print(f"✓ Enhanced model loaded from {filepath}")
        return instance


def convert_bin_predictions_to_numbers(bin_predictions: Dict, bin_classifier: BinClassifier) -> Dict:
    """
    Convert bin predictions back to actual number predictions.
    Uses the midpoint of each bin as the predicted number.
    """
    number_predictions = {}

    for col_name, pred in bin_predictions.items():
        bin_pred = pred['bin_prediction']
        confidence = pred['confidence']

        bin_edges = bin_classifier.bin_edges[col_name]
        bin_idx = bin_pred - 1

        if bin_idx < 0 or bin_idx >= len(bin_edges) - 1:
            predicted_number = int((bin_edges[0] + bin_edges[-1]) / 2)
        else:
            bin_start = bin_edges[bin_idx]
            bin_end = bin_edges[bin_idx + 1]
            predicted_number = int((bin_start + bin_end) / 2)

        valid_max = VALID_RANGES.get(col_name, 69)
        predicted_number = max(1, min(valid_max, predicted_number))

        number_predictions[col_name] = {
            'number_prediction': predicted_number,
            'bin_prediction': bin_pred,
            'confidence': confidence,
            'bin_range': f"{bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx+1]:.1f}" if 0 <= bin_idx < len(bin_edges)-1 else "unknown"
        }

    return number_predictions


def main():
    """Enhanced main function with comprehensive analysis."""
    print("="*70)
    print("ENHANCED POWERBALL BIN PREDICTION USING GPU RANDOM FOREST")
    print("="*70)
    
    from pathlib import Path
    project_dir = Path(__file__).resolve().parent
    data_path = project_dir / 'powerball_games_only.csv'
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Loaded {len(df)} drawings from {data_path.name}")
    except FileNotFoundError:
        print("❌ Data file not found. Please run the Fourier prediction script first.")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Create enhanced predictor
    predictor = EnhancedGPURandomForestBinPredictor(
        sequence_length=15,  # Will be optimized
        use_gpu=True
    )
    
    # Train with optimization
    predictor.fit(df, optimize_length=True, train_test_split_ratio=0.67)
    
    # Evaluate on test set
    predictor.evaluate_on_test_set()
    
    # Make prediction for next drawing
    print(f"\n" + "="*70)
    print("ENHANCED BIN PREDICTIONS FOR NEXT DRAWING")
    print("="*70)
    
    bin_predictions = predictor.predict_next_bins(df)
    number_predictions = convert_bin_predictions_to_numbers(bin_predictions, predictor.bin_classifier)
    
    # Display predictions
    prediction_date = (df['date'].max() + timedelta(days=3)).strftime('%Y-%m-%d')
    print(f"\nPrediction Date: {prediction_date}")
    print(f"Based on {predictor.sequence_length}-step enhanced sequences")
    
    predicted_numbers = []
    powerball_pred = None
    print(f"\nEnhanced Bin Predictions:")
    for i in range(1, 6):
        col_name = f'ball_{i}'
        if col_name not in number_predictions:
            continue
        pred = number_predictions[col_name]
        print(f"  {col_name}: Bin {pred['bin_prediction']} → Number {pred['number_prediction']} "
              f"(confidence: {pred['confidence']:.3f}, range: {pred['bin_range']})")
        predicted_numbers.append(pred['number_prediction'])

    if 'powerball' in number_predictions:
        powerball_pred = number_predictions['powerball']
        print(f"  powerball: Bin {powerball_pred['bin_prediction']} → Number {powerball_pred['number_prediction']} "
              f"(confidence: {powerball_pred['confidence']:.3f}, range: {powerball_pred['bin_range']})")

    print(f"\nPredicted White Balls: {predicted_numbers}")
    print(f"Sorted: {sorted(predicted_numbers)}")
    if powerball_pred is not None:
        print(f"Predicted Powerball: {powerball_pred['number_prediction']}")

    results = {
        'prediction_date': prediction_date,
        'model': f'Enhanced RandomForest-Bins (seq={predictor.sequence_length}, gpu={predictor.use_gpu})',
        'sequence_length': predictor.sequence_length,
        **{f'ball_{i}': predicted_numbers[i-1] for i in range(1, 6)},
        **{f'bin_{i}': bin_predictions[f'ball_{i}']['bin_prediction'] for i in range(1, 6)},
        'powerball': powerball_pred['number_prediction'] if powerball_pred is not None else None,
        'bin_pb': bin_predictions['powerball']['bin_prediction'] if 'powerball' in bin_predictions else None,
    }

    results_df = pd.DataFrame([results])
    results_df.to_csv(project_dir / 'enhanced_random_forest_predictions.csv', index=False)
    print(f"✓ Enhanced predictions saved to enhanced_random_forest_predictions.csv")

    predictor.save_model(str(project_dir / 'enhanced_random_forest_bin_model.joblib'))

    # Append to the rolling historical tracker so RF predictions accumulate
    # alongside Fourier ones for over-time comparison vs actual draws.
    try:
        import sys
        import hashlib
        sys.path.append(str(project_dir))
        from prediction_tracker import PredictionTracker

        tracker = PredictionTracker(str(project_dir / 'historical_predictions.csv'))
        data_hash = hashlib.md5(str(len(df)).encode()).hexdigest()[:8]

        rf_details = {
            'model_type': 'random_forest_bins',
            'sequence_length': predictor.sequence_length,
            'n_bins': predictor.bin_classifier.n_bins,
            'use_gpu': predictor.use_gpu,
            'data_points': len(df),
            'bins_white': [bin_predictions[f'ball_{i}']['bin_prediction'] for i in range(1, 6)],
            'bin_powerball': (bin_predictions['powerball']['bin_prediction']
                              if 'powerball' in bin_predictions else None),
            'confidence_white': [round(bin_predictions[f'ball_{i}']['confidence'], 4) for i in range(1, 6)],
            'confidence_powerball': (round(bin_predictions['powerball']['confidence'], 4)
                                     if 'powerball' in bin_predictions else None),
        }

        tracker.add_prediction(
            target_date=prediction_date,
            model_type='random_forest',
            numbers=predicted_numbers,
            model_details=rf_details,
            data_hash=data_hash,
            powerball=(powerball_pred['number_prediction'] if powerball_pred is not None else None),
        )
        tracker.save_history()
        print(f"✓ Stored RF prediction in historical tracker")
    except Exception as e:
        print(f"⚠ Could not store RF prediction in historical tracker: {e}")
    
    print(f"\n" + "="*70)
    print("ENHANCED PREDICTION COMPLETE")
    print("="*70)
    print("Key Improvements:")
    print("• Ball-specific sequential feature extraction")
    print("• Enhanced temporal pattern recognition")
    print("• Improved sequence length optimization")
    print("• Deeper Random Forest models (300 trees, depth 20)")
    print("• Advanced feature engineering (trend, stability, transitions)")
    print("="*70)


if __name__ == "__main__":
    main()