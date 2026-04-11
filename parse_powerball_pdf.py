#!/usr/bin/env python3
"""
Florida Lottery Powerball PDF Parser
Extracts drawing data from the official Florida Lottery Powerball PDF
and converts it to a pandas DataFrame.
"""

import requests
import pandas as pd
import PyPDF2
import io
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple

# PDF URL
PDF_URL = "https://files.floridalottery.com/exptkt/pb.pdf"


def download_pdf(url: str) -> bytes:
    """Download PDF from URL."""
    print(f"Downloading PDF from {url}...")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    print(f"Downloaded {len(response.content)} bytes")
    return response.content


def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract all text from PDF."""
    print("Extracting text from PDF...")
    pdf_file = io.BytesIO(pdf_content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    text = ""
    for page_num, page in enumerate(pdf_reader.pages):
        print(f"Processing page {page_num + 1}/{len(pdf_reader.pages)}")
        text += page.extract_text() + "\n"
    
    return text


def parse_powerball_data(text: str) -> List[Dict]:
    """
    Parse Powerball drawing data from extracted text.
    
    Expected format (varies, adjust regex as needed):
    Date       Numbers                Powerball  Power Play  Jackpot
    01/01/2024 05 12 23 34 45          08         2X          $100M
    """
    data = []
    
    # Split text into lines
    lines = text.split('\n')
    
    # Pattern: DATE + 5 white balls + PB + powerball + optional multiplier + POWERBALL[/DP]
    # Example: 4/8/26 3 16 17 42 52 PB 3 X2 POWERBALL
    pattern1 = re.compile(
        r'(\d{1,2}/\d{1,2}/\d{2,4})\s+'  # Date
        r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})\s+'  # 5 white balls
        r'PB\s+(\d{1,2})'  # Powerball
        r'(?:\s+(X\d+))?'  # Power Play (optional)
        r'\s+POWERBALL(?:\s+DP)?\s*$'  # Powerball or Double Play line
    )
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Try pattern 1
        match = pattern1.search(line)
        if match:
            date_str = match.group(1)
            ball1 = int(match.group(2))
            ball2 = int(match.group(3))
            ball3 = int(match.group(4))
            ball4 = int(match.group(5))
            ball5 = int(match.group(6))
            powerball = int(match.group(7))
            power_play = match.group(8) if match.group(8) else None
            jackpot = None
            
            data.append({
                'date': date_str,
                'ball_1': ball1,
                'ball_2': ball2,
                'ball_3': ball3,
                'ball_4': ball4,
                'ball_5': ball5,
                'powerball': powerball,
                'power_play': power_play,
                'jackpot': jackpot,
                'raw_line': line
            })
    
    print(f"Parsed {len(data)} drawing records")
    return data


def create_dataframe(data: List[Dict]) -> pd.DataFrame:
    """Convert parsed data to pandas DataFrame with proper types."""
    if not data:
        print("No data parsed!")
        return pd.DataFrame()
    
    df = pd.DataFrame(data)
    
    # Convert date string to datetime
    try:
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
    except Exception:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Sort by date (newest first)
    df = df.sort_values('date', ascending=False).reset_index(drop=True)
    
    # Add computed columns
    df['numbers_sorted'] = df.apply(
        lambda row: sorted([row['ball_1'], row['ball_2'], row['ball_3'], 
                           row['ball_4'], row['ball_5']]), 
        axis=1
    )
    
    df['sum_white_balls'] = df['ball_1'] + df['ball_2'] + df['ball_3'] + df['ball_4'] + df['ball_5']
    df['total_sum'] = df['sum_white_balls'] + df['powerball']
    
    # Number frequency columns (for analysis)
    df['has_even_count'] = df.apply(
        lambda row: sum([1 for ball in [row['ball_1'], row['ball_2'], row['ball_3'], 
                                         row['ball_4'], row['ball_5']] if ball % 2 == 0]),
        axis=1
    )
    
    df['has_odd_count'] = 5 - df['has_even_count']
    
    return df


def analyze_data(df: pd.DataFrame):
    """Print basic statistics about the data."""
    if df.empty:
        print("No data to analyze")
        return
    
    print("\n" + "="*60)
    print("POWERBALL DATA ANALYSIS")
    print("="*60)
    print(f"\nTotal drawings: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Most common white balls
    print("\n--- Most Common White Balls ---")
    all_white_balls = pd.concat([df['ball_1'], df['ball_2'], df['ball_3'], 
                                  df['ball_4'], df['ball_5']])
    print(all_white_balls.value_counts().head(10))
    
    # Most common Powerball numbers
    print("\n--- Most Common Powerball Numbers ---")
    print(df['powerball'].value_counts().head(10))
    
    # Sum statistics
    print("\n--- Sum Statistics ---")
    print(f"Average sum of white balls: {df['sum_white_balls'].mean():.2f}")
    print(f"Min sum: {df['sum_white_balls'].min()}")
    print(f"Max sum: {df['sum_white_balls'].max()}")
    
    # Even/Odd distribution
    print("\n--- Even/Odd Distribution ---")
    print(df['has_even_count'].value_counts().sort_index())


def main():
    """Main execution function."""
    try:
        # Download PDF
        pdf_content = download_pdf(PDF_URL)
        
        # Extract text
        text = extract_text_from_pdf(pdf_content)
        
        # Save raw text for inspection (optional)
        with open('/tmp/powerball_raw_text.txt', 'w') as f:
            f.write(text)
        print("Raw text saved to /tmp/powerball_raw_text.txt for inspection")
        
        # Parse data
        data = parse_powerball_data(text)
        
        if not data:
            print("\n⚠ WARNING: No data was parsed!")
            print("The PDF format may have changed. Check /tmp/powerball_raw_text.txt")
            print("and adjust the regex patterns in parse_powerball_data().")
            return None
        
        # Create DataFrame
        df = create_dataframe(data)
        
        output_dir = Path(__file__).resolve().parent

        # Save to CSV
        output_csv = output_dir / 'powerball_data.csv'
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Data saved to {output_csv}")
        
        # Save to Excel (if openpyxl is installed)
        try:
            output_excel = output_dir / 'powerball_data.xlsx'
            df.to_excel(output_excel, index=False, sheet_name='Powerball')
            print(f"✓ Data saved to {output_excel}")
        except ImportError:
            print("(Install openpyxl for Excel export: pip install openpyxl)")
        
        # Display sample
        print("\n--- First 10 Records ---")
        print(df.head(10).to_string())
        
        # Analyze
        analyze_data(df)
        
        return df
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    df = main()
    
    if df is not None:
        print("\n" + "="*60)
        print("DataFrame is ready! Access it via the 'df' variable")
        print("="*60)
        print("\nExample usage:")
        print("  df.head()           # View first rows")
        print("  df.describe()       # Statistical summary")
        print("  df.info()           # Column info")
        print("  df['powerball'].value_counts()  # Frequency analysis")
