#!/usr/bin/env python3
"""
GPU-Accelerated Fourier Series Powerball Prediction
Uses PyTorch with ROCm to fit Fourier series to historical data
and predict next winning numbers.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import PyPDF2
import io
import re
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# PDF URL
PDF_URL = "https://files.floridalottery.com/exptkt/pb.pdf"


def download_and_parse_pdf(url: str) -> pd.DataFrame:
    """Download PDF and parse POWERBALL games only."""
    print(f"Downloading PDF from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    
    # Extract text
    pdf_file = io.BytesIO(response.content)
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
                    'powerball': int(match.group(7)),
                })
            except (ValueError, AttributeError) as e:
                continue
    
    df = pd.DataFrame(data)
    df = df.sort_values('date', ascending=True).reset_index(drop=True)
    
    print(f"Parsed {len(df)} POWERBALL drawings (excluding Double Play)")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


class FourierSeriesGPU:
    """GPU-accelerated Fourier series fitting using PyTorch."""
    
    def __init__(self, n_harmonics: int = 10, device='cuda'):
        self.n_harmonics = n_harmonics
        self.device = device
        self.coeffs = None
        self.x_scale = None
        self.y_scale = None
        self.y_mean = None
        
    def fit(self, x: np.ndarray, y: np.ndarray, lr: float = 0.01, epochs: int = 5000):
        """
        Fit Fourier series to data using gradient descent.
        Minimizes standard deviation of residuals.
        """
        # Normalize data
        x_min, x_max = x.min(), x.max()
        self.x_scale = (x_min, x_max)
        x_norm = (x - x_min) / (x_max - x_min) * 2 * np.pi
        
        self.y_mean = y.mean()
        y_centered = y - self.y_mean
        y_std = y_centered.std()
        self.y_scale = y_std if y_std > 0 else 1.0
        y_norm = y_centered / self.y_scale
        
        # Convert to tensors
        x_t = torch.FloatTensor(x_norm).to(self.device)
        y_t = torch.FloatTensor(y_norm).to(self.device)
        
        # Initialize coefficients (a0, a1, b1, a2, b2, ..., an, bn)
        # Must be a parameter to be a leaf tensor
        coeffs = torch.nn.Parameter(torch.randn(2 * self.n_harmonics + 1, device=self.device) * 0.01)
        
        optimizer = torch.optim.Adam([coeffs], lr=lr)
        
        best_loss = float('inf')
        best_coeffs = coeffs.clone()
        patience = 500
        patience_counter = 0
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute Fourier series
            y_pred = coeffs[0]  # a0
            for n in range(1, self.n_harmonics + 1):
                a_n = coeffs[2 * n - 1]
                b_n = coeffs[2 * n]
                y_pred = y_pred + a_n * torch.cos(n * x_t) + b_n * torch.sin(n * x_t)
            
            # Loss: minimize standard deviation of residuals
            residuals = y_t - y_pred
            loss = torch.std(residuals)
            
            loss.backward()
            optimizer.step()
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_coeffs = coeffs.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss (Std Dev): {loss.item():.6f}")
        
        self.coeffs = best_coeffs.cpu().numpy()
        print(f"Final Loss (Std Dev of Residuals): {best_loss:.6f}")
        
        return self
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict values for new x."""
        if self.coeffs is None:
            raise ValueError("Model not fitted yet!")
        
        # Normalize x
        x_min, x_max = self.x_scale
        x_norm = (x - x_min) / (x_max - x_min) * 2 * np.pi
        
        # Convert to tensor
        x_t = torch.FloatTensor(x_norm).to(self.device)
        coeffs_t = torch.FloatTensor(self.coeffs).to(self.device)
        
        # Compute Fourier series
        y_pred = coeffs_t[0]
        for n in range(1, self.n_harmonics + 1):
            a_n = coeffs_t[2 * n - 1]
            b_n = coeffs_t[2 * n]
            y_pred = y_pred + a_n * torch.cos(n * x_t) + b_n * torch.sin(n * x_t)
        
        # Denormalize
        y_pred = y_pred.cpu().numpy() * self.y_scale + self.y_mean
        
        return y_pred


def predict_next_numbers(df: pd.DataFrame, n_harmonics: int = 15) -> Dict:
    """
    Fit Fourier series to each ball position and predict next drawing.
    """
    print("\n" + "="*70)
    print("GPU-ACCELERATED FOURIER SERIES PREDICTION")
    print("="*70)
    
    # Convert dates to numeric (days since first drawing)
    first_date = df['date'].min()
    df['days_since_start'] = (df['date'] - first_date).dt.days.values
    
    x_train = df['days_since_start'].values.astype(float)
    
    # Predict next drawing date (3-4 days from last)
    last_date = df['date'].max()
    next_date = last_date + timedelta(days=3)
    x_next = np.array([(next_date - first_date).days], dtype=float)
    
    print(f"\nTraining data: {len(df)} drawings")
    print(f"Last drawing: {last_date.strftime('%Y-%m-%d')}")
    print(f"Predicting for: {next_date.strftime('%Y-%m-%d')}")
    print(f"Using {n_harmonics} harmonics per ball")

    # Each entry: (column_name, valid_max). White balls 1-69, Powerball 1-26.
    targets = [(f'ball_{i}', 69) for i in range(1, 6)] + [('powerball', 26)]

    predictions = {}

    for col_name, valid_max in targets:
        if col_name not in df.columns:
            print(f"\n--- Skipping {col_name} (not present in data) ---")
            continue
        y_train = df[col_name].values.astype(float)

        print(f"\n--- Fitting {col_name} ---")
        model = FourierSeriesGPU(n_harmonics=n_harmonics, device=device)
        model.fit(x_train, y_train, lr=0.01, epochs=5000)

        y_next = model.predict(x_next)[0]
        y_next_clipped = np.clip(np.round(y_next), 1, valid_max)

        predictions[col_name] = {
            'raw_prediction': y_next,
            'clipped_prediction': int(y_next_clipped),
            'std_dev': df[col_name].std(),
            'mean': df[col_name].mean(),
            'valid_max': valid_max,
        }

        print(f"Prediction: {y_next:.2f} -> {int(y_next_clipped)} (valid range: 1-{valid_max})")

    return predictions, next_date


def main():
    """Main execution."""
    print("="*70)
    print("POWERBALL PREDICTION USING GPU-ACCELERATED FOURIER SERIES")
    print("="*70)
    
    project_dir = Path(__file__).resolve().parent

    # Download and parse data
    df = download_and_parse_pdf(PDF_URL)
    
    if df.empty:
        print("ERROR: No data parsed!")
        return
    
    # Save full data
    df.to_csv(project_dir / 'powerball_games_only.csv', index=False)
    print(f"\n✓ Full data saved to powerball_games_only.csv")
    
    # Show sample
    print("\n--- Sample Data (last 5 drawings) ---")
    sample_cols = ['date', 'ball_1', 'ball_2', 'ball_3', 'ball_4', 'ball_5']
    if 'powerball' in df.columns:
        sample_cols.append('powerball')
    print(df[sample_cols].tail())
    
    # Predict next numbers
    predictions, next_date = predict_next_numbers(df, n_harmonics=15)
    
    # Display predictions
    print("\n" + "="*70)
    print("PREDICTED NUMBERS FOR NEXT DRAWING")
    print("="*70)
    print(f"Date: {next_date.strftime('%Y-%m-%d')}")
    print("\nWhite Balls:")

    predicted_numbers = []
    for i in range(1, 6):
        pred = predictions[f'ball_{i}']
        predicted_numbers.append(pred['clipped_prediction'])
        print(f"  Ball {i}: {pred['clipped_prediction']:2d} "
              f"(raw: {pred['raw_prediction']:6.2f}, "
              f"historical mean: {pred['mean']:5.2f}, std: {pred['std_dev']:5.2f})")

    predicted_numbers_sorted = sorted(predicted_numbers)
    print(f"\nSorted: {predicted_numbers_sorted}")

    if len(set(predicted_numbers)) < 5:
        print("\n⚠ WARNING: Duplicate numbers detected!")
        print("Fourier series may not be ideal for this type of discrete prediction.")
        print("Consider using the predictions as guidance rather than exact values.")

    powerball_pred = predictions.get('powerball')
    if powerball_pred is not None:
        print(f"\nPowerball: {powerball_pred['clipped_prediction']:2d} "
              f"(raw: {powerball_pred['raw_prediction']:6.2f}, "
              f"historical mean: {powerball_pred['mean']:5.2f}, "
              f"std: {powerball_pred['std_dev']:5.2f})")

    pred_row = {
        'prediction_date': next_date,
        'ball_1': predicted_numbers[0],
        'ball_2': predicted_numbers[1],
        'ball_3': predicted_numbers[2],
        'ball_4': predicted_numbers[3],
        'ball_5': predicted_numbers[4],
        'powerball': powerball_pred['clipped_prediction'] if powerball_pred is not None else None,
        'sorted': str(predicted_numbers_sorted),
        'model': f'Fourier Series (n={15})',
        'device': str(device)
    }
    pred_df = pd.DataFrame([pred_row])
    
    prediction_output = project_dir / 'powerball_predictions.csv'
    try:
        pred_df.to_csv(prediction_output, index=False)
        print(f"\n✓ Predictions saved to {prediction_output.name}")
    except PermissionError:
        prediction_output = project_dir / 'powerball_predictions_latest.csv'
        pred_df.to_csv(prediction_output, index=False)
        print(f"\n⚠ powerball_predictions.csv not writable; saved to {prediction_output.name}")
    
    # Store in historical tracker
    try:
        import sys
        sys.path.append(str(project_dir))
        from prediction_tracker import PredictionTracker
        import hashlib
        
        tracker = PredictionTracker(str(project_dir / 'historical_predictions.csv'))
        
        # Calculate data hash
        data_hash = hashlib.md5(str(len(df)).encode()).hexdigest()[:8]
        
        fourier_details = {
            'harmonics': 15,
            'device': str(device),
            'model_type': 'fourier_series',
            'data_points': len(df)
        }
        
        tracker.add_prediction(
            target_date=next_date.strftime('%Y-%m-%d'),
            model_type='fourier',
            numbers=predicted_numbers,
            model_details=fourier_details,
            data_hash=data_hash,
            powerball=(powerball_pred['clipped_prediction'] if powerball_pred is not None else None)
        )
        
        tracker.save_history()
        print(f"✓ Stored prediction in historical tracker")
    except Exception as e:
        print(f"⚠ Could not store in historical tracker: {e}")
    
    print("\n" + "="*70)
    print("IMPORTANT DISCLAIMER")
    print("="*70)
    print("This is a mathematical exercise using Fourier series for time series")
    print("prediction. Lottery numbers are drawn randomly, and past results do")
    print("not influence future outcomes. This prediction has no statistical")
    print("advantage over random number selection.")
    print("="*70)


if __name__ == "__main__":
    main()
