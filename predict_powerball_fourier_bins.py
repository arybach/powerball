#!/usr/bin/env python3
"""
GPU-Accelerated Fourier Series Powerball Bin Prediction
Modified to work with statistical bins instead of raw numbers for better pattern recognition.
Uses PyTorch with ROCm to fit Fourier series to historical bin sequences
and predict next bin classifications.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import PyPDF2
import io
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability (ROCm/CUDA)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    except:
        print("GPU detected but properties unavailable")

# PDF URL
PDF_URL = "https://files.floridalottery.com/exptkt/pb.pdf"


def download_and_parse_pdf(url: str) -> pd.DataFrame:
    """Download PDF and parse POWERBALL games only."""
    print(f"Downloading PDF from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        pdf_bytes = response.content
    except requests.exceptions.SSLError as exc:
        print(f"requests SSL handshake failed ({exc}); falling back to curl...")
        import subprocess
        result = subprocess.run(
            ["curl", "-sS", "--fail", "-L", url], capture_output=True, timeout=60
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"curl fallback failed: {result.stderr.decode(errors='replace')}"
            ) from exc
        pdf_bytes = result.stdout

    # Extract text
    pdf_file = io.BytesIO(pdf_bytes)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Parse lines - looking for format: DATE NUMBERS PB # [X#] POWERBALL [DP]
    # We only want lines ending with "POWERBALL" (not "POWERBALL DP")
    data = []
    pattern = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{2,4})\s+'  # Date
        r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # 5 white balls
        r'PB\s+(\d{1,2})'  # Powerball
        r'(?:\s+X\d+)?'  # Optional multiplier
        r'\s+POWERBALL\s*$'  # Must end with POWERBALL (not POWERBALL DP)
    )
    
    for line in text.split('\n'):
        line = line.strip()
        match = pattern.search(line)
        if match:
            try:
                date_str = match.group(1)
                date_obj = pd.to_datetime(date_str, format='%m/%d/%y', errors='coerce')
                
                data.append({
                    'date': date_obj,
                    'ball_1': int(match.group(2)),
                    'ball_2': int(match.group(3)),
                    'ball_3': int(match.group(4)),
                    'ball_4': int(match.group(5)),
                    'ball_5': int(match.group(6)),
                })
            except (ValueError, AttributeError) as e:
                continue
    
    df = pd.DataFrame(data)
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    
    print(f"Parsed {len(df)} POWERBALL drawings (excluding Double Play)")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


class BinClassifier:
    """
    Classify numbers into statistical bins based on historical distribution.
    """
    
    def __init__(self, n_bins: int = 6):
        self.n_bins = n_bins
        self.bin_edges = {}
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """Fit bin edges based on historical data."""
        print(f"\n--- Creating {self.n_bins} Bins for Fourier Analysis ---")
        for ball_num in range(1, 6):
            col_name = f'ball_{ball_num}'
            values = df[col_name].values
            
            # Create bins based on quantiles for even distribution
            bin_edges = np.quantile(values, np.linspace(0, 1, self.n_bins + 1))
            # Ensure unique edges
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < self.n_bins + 1:
                # Fallback to equal-width bins
                bin_edges = np.linspace(values.min(), values.max() + 0.1, self.n_bins + 1)
            
            self.bin_edges[col_name] = bin_edges
            
            print(f"{col_name} bins: {[f'{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}' for i in range(len(bin_edges)-1)]}")
        
        self.fitted = True
    
    def transform_to_bins(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numbers to bin classifications."""
        if not self.fitted:
            raise ValueError("BinClassifier not fitted yet!")
        
        df_bins = df.copy()
        
        for ball_num in range(1, 6):
            col_name = f'ball_{ball_num}'
            bin_col = f'bin_{ball_num}'
            
            # Use digitize to assign bins (1-indexed)
            bins = np.digitize(df[col_name].values, self.bin_edges[col_name]) - 1
            # Ensure bins are in valid range
            bins = np.clip(bins, 0, self.n_bins - 1) + 1  # Convert to 1-6 range
            
            df_bins[bin_col] = bins
        
        return df_bins
    
    def convert_bins_to_numbers(self, bin_predictions: Dict) -> Dict:
        """Convert bin predictions back to approximate numbers."""
        number_predictions = {}
        
        for ball_num in range(1, 6):
            col_name = f'ball_{ball_num}'
            bin_pred = bin_predictions[col_name]['bin_prediction']
            confidence = bin_predictions[col_name]['confidence']
            
            # Get bin edges for this ball
            bin_edges = self.bin_edges[col_name]
            
            # Convert bin number (1-6) to array index (0-5)
            bin_idx = bin_pred - 1
            
            if bin_idx < 0 or bin_idx >= len(bin_edges) - 1:
                # Fallback to middle range
                predicted_number = int((bin_edges[0] + bin_edges[-1]) / 2)
            else:
                # Use bin midpoint
                bin_start = bin_edges[bin_idx]
                bin_end = bin_edges[bin_idx + 1]
                predicted_number = int((bin_start + bin_end) / 2)
            
            # Ensure valid Powerball range (1-69)
            predicted_number = max(1, min(69, predicted_number))
            
            number_predictions[col_name] = {
                'number_prediction': predicted_number,
                'bin_prediction': bin_pred,
                'confidence': confidence,
                'bin_range': f"{bin_edges[bin_idx]:.1f}-{bin_edges[bin_idx+1]:.1f}" if 0 <= bin_idx < len(bin_edges)-1 else "unknown"
            }
        
        return number_predictions


class FourierSeriesBinGPU:
    """GPU-accelerated Fourier series fitting for bin sequences using PyTorch."""
    
    def __init__(self, n_harmonics: int = 10, device='cuda'):
        self.n_harmonics = n_harmonics
        self.device = device
        self.coeffs = None
        self.x_scale = None
        self.y_scale = None
        self.y_mean = None
        
    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 3000):
        """
        Fit Fourier series to bin sequence data using gradient descent.
        Bins (1-6) should have much clearer patterns than raw numbers (1-69).
        """
        # Normalize data for better convergence
        x_min, x_max = x.min(), x.max()
        self.x_scale = (x_min, x_max)
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1  # Normalize to [-1, 1]
        
        # For bins, we'll normalize differently - keep the discrete nature
        self.y_mean = y.mean()
        self.y_scale = y.std() if y.std() > 0 else 1.0
        y_norm = (y - self.y_mean) / self.y_scale
        
        # Convert to tensors
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32, device=self.device)
        
        # Initialize Fourier coefficients
        # [a0, a1, b1, a2, b2, ..., an, bn]
        self.coeffs = torch.randn(1 + 2 * self.n_harmonics, device=self.device, requires_grad=True)
        
        optimizer = torch.optim.AdamW([self.coeffs], lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=200, factor=0.8)
        
        best_loss = float('inf')
        patience_counter = 0
        patience_limit = 300
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute Fourier series
            y_pred = self._fourier_series(x_tensor)
            
            # Loss: standard deviation of residuals (like original, but for bins)
            residuals = y_tensor - y_pred
            loss = torch.std(residuals)
            
            # Add small regularization to prevent overfitting to discrete bin values
            reg_loss = 1e-4 * torch.norm(self.coeffs)
            total_loss = loss + reg_loss
            
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss)
            
            # Early stopping
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        final_loss = loss.item()
        print(f"Final Loss (Std Dev of Residuals): {final_loss:.6f}")
        return self
    
    def _fourier_series(self, x):
        """Compute Fourier series: a0/2 + Σ(an*cos(nx) + bn*sin(nx))"""
        # Initialize result with same shape as x
        result = torch.zeros_like(x) + (self.coeffs[0] / 2)  # a0/2 constant term
        
        for n in range(1, self.n_harmonics + 1):
            an = self.coeffs[2*n - 1]  # cosine coefficient
            bn = self.coeffs[2*n]      # sine coefficient
            result = result + an * torch.cos(n * torch.pi * x) + bn * torch.sin(n * torch.pi * x)
        
        return result
    
    def predict(self, x: np.ndarray):
        """Predict using fitted Fourier series."""
        if self.coeffs is None:
            raise ValueError("Model not fitted yet!")
        
        # Normalize x using same scaling
        x_min, x_max = self.x_scale
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        x_tensor = torch.tensor(x_norm, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            y_pred_norm = self._fourier_series(x_tensor)
            # Denormalize
            y_pred = y_pred_norm * self.y_scale + self.y_mean
        
        return y_pred.cpu().numpy()


def predict_next_drawing_bins(df: pd.DataFrame, n_harmonics: int = 15) -> Tuple[Dict, str]:
    """
    Predict the next drawing using Fourier series on bin sequences.
    
    Returns:
        predictions: Dict with bin predictions for each ball
        next_date: Next drawing date
    """
    print("\n" + "="*70)
    print("GPU-ACCELERATED FOURIER SERIES BIN PREDICTION")
    print("="*70)
    
    # Create bin classifier and transform data
    bin_classifier = BinClassifier(n_bins=6)
    bin_classifier.fit(df)
    df_bins = bin_classifier.transform_to_bins(df)
    
    # Show bin distribution
    print(f"\n--- Bin Distribution Analysis ---")
    for ball_num in range(1, 6):
        bin_col = f'bin_{ball_num}'
        counts = df_bins[bin_col].value_counts().sort_index()
        print(f"{bin_col}: {dict(counts)}")
    
    # Prepare training data
    x_train = np.arange(len(df_bins)).astype(float)
    next_date = (df['date'].max() + timedelta(days=3)).strftime('%Y-%m-%d')
    x_next = np.array([len(df_bins)], dtype=float)
    
    print(f"\nTraining data: {len(df_bins)} drawings")
    print(f"Last drawing: {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Predicting for: {next_date}")
    print(f"Using {n_harmonics} harmonics per ball")
    
    # Fit Fourier series for each ball's bin sequence
    predictions = {}
    
    for ball_num in range(1, 6):
        ball_col = f'ball_{ball_num}'
        bin_col = f'bin_{ball_num}'
        y_train = df_bins[bin_col].values.astype(float)
        
        print(f"\n--- Fitting {ball_col} bin sequence ---")
        model = FourierSeriesBinGPU(n_harmonics=n_harmonics, device=device)
        model.fit(x_train, y_train, lr=0.01, epochs=3000)
        
        # Predict next bin
        y_next = model.predict(x_next)[0]
        
        # Round to nearest bin and clip to valid range (1-6)
        y_next_clipped = np.clip(np.round(y_next), 1, 6)
        
        predictions[ball_col] = {
            'bin_prediction': int(y_next_clipped),
            'raw_prediction': y_next,
            'confidence': 1.0 / (1.0 + abs(y_next - y_next_clipped)),  # Confidence based on how close to integer
            'bin_mean': df_bins[bin_col].mean(),
            'bin_std': df_bins[bin_col].std(),
        }
        
        print(f"Bin prediction: {y_next:.2f} -> Bin {int(y_next_clipped)} (range: 1-6)")
        print(f"Historical bin stats - mean: {predictions[ball_col]['bin_mean']:.2f}, std: {predictions[ball_col]['bin_std']:.2f}")
    
    # Convert bins back to approximate numbers
    number_predictions = bin_classifier.convert_bins_to_numbers(predictions)
    
    return predictions, number_predictions, next_date, bin_classifier


def main():
    """Main execution with bin-based Fourier analysis."""
    print("="*70)
    print("POWERBALL BIN PREDICTION USING GPU-ACCELERATED FOURIER SERIES")
    print("="*70)
    
    # Download and parse data
    df = download_and_parse_pdf(PDF_URL)
    
    if df.empty:
        print("ERROR: No data parsed!")
        return
    
    # Save full data
    df.to_csv('powerball_games_only.csv', index=False)
    print(f"\n✓ Full data saved to powerball_games_only.csv")
    
    # Show sample
    print("\n--- Sample Data (last 5 drawings) ---")
    print(df[['date', 'ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']].tail())
    
    # Predict next drawing using bin-based Fourier series
    bin_predictions, number_predictions, next_date, bin_classifier = predict_next_drawing_bins(df, n_harmonics=15)
    
    print(f"\n" + "="*70)
    print("PREDICTED BINS AND NUMBERS FOR NEXT DRAWING")
    print("="*70)
    print(f"\nDate: {next_date}")
    print(f"Method: Fourier Series on Bin Sequences (n={15})")
    
    print("\nBin Predictions:")
    predicted_numbers = []
    for i in range(1, 6):
        ball_col = f'ball_{i}'
        bin_pred = bin_predictions[ball_col]
        num_pred = number_predictions[ball_col]
        
        print(f"  Ball {i}: Bin {bin_pred['bin_prediction']} → Number {num_pred['number_prediction']} "
              f"(confidence: {bin_pred['confidence']:.3f}, range: {num_pred['bin_range']})")
        
        predicted_numbers.append(num_pred['number_prediction'])
    
    # Sort and check for duplicates
    predicted_numbers_sorted = sorted(predicted_numbers)
    print(f"\nPredicted Numbers: {predicted_numbers}")
    print(f"Sorted: {predicted_numbers_sorted}")
    
    if len(set(predicted_numbers)) < 5:
        print("\n⚠ WARNING: Duplicate numbers detected!")
        print("Consider adjusting bin boundaries or using different post-processing.")
    
    # Save predictions in multiple formats
    # 1. Bin predictions
    bin_pred_df = pd.DataFrame([{
        'prediction_date': next_date,
        'model': f'Fourier-Bins (n={15})',
        **{f'bin_{i}': bin_predictions[f'ball_{i}']['bin_prediction'] for i in range(1, 6)},
        **{f'ball_{i}': predicted_numbers[i-1] for i in range(1, 6)},
        'sorted': str(predicted_numbers_sorted),
        'device': str(device)
    }])
    
    bin_pred_df.to_csv('fourier_bin_predictions.csv', index=False)
    print(f"\n✓ Bin predictions saved to fourier_bin_predictions.csv")
    
    # 2. Regular format (for compatibility)
    pred_df = pd.DataFrame([{
        'prediction_date': next_date,
        'ball_1': predicted_numbers[0],
        'ball_2': predicted_numbers[1],
        'ball_3': predicted_numbers[2],
        'ball_4': predicted_numbers[3],
        'ball_5': predicted_numbers[4],
        'sorted': str(predicted_numbers_sorted),
        'model': f'Fourier-Bins (n={15})',
        'device': str(device)
    }])
    
    pred_df.to_csv('powerball_predictions.csv', index=False)
    print(f"✓ Number predictions saved to powerball_predictions.csv")
    
    # Store in historical tracker
    try:
        import sys
        sys.path.append('.')
        from prediction_tracker import PredictionTracker
        import hashlib
        
        tracker = PredictionTracker('historical_predictions.csv')
        
        # Calculate data hash
        data_hash = hashlib.md5(str(len(df)).encode()).hexdigest()[:8]
        
        fourier_bin_details = {
            'harmonics': 15,
            'device': str(device),
            'model_type': 'fourier_bins',
            'data_hash': data_hash,
            'bins_used': 6
        }
        
        # Add prediction
        tracker.add_prediction(
            target_date=next_date,
            model_type='fourier_bins',
            numbers=predicted_numbers,
            model_details=fourier_bin_details,
            data_hash=data_hash
        )

        tracker.save_history()
        print("✓ Stored in historical tracker")
        
    except Exception as e:
        print(f"⚠ Could not store in historical tracker: {e}")
    
    print(f"\n" + "="*70)
    print("IMPORTANT DISCLAIMER")
    print("="*70)
    print("This is a mathematical exercise using Fourier series for bin pattern")
    print("prediction. Lottery numbers are drawn randomly, and past results do")
    print("not influence future outcomes. This bin-based approach may show")
    print("better pattern recognition than raw numbers, but still has no")
    print("statistical advantage over random number selection.")
    print("="*70)


if __name__ == "__main__":
    main()