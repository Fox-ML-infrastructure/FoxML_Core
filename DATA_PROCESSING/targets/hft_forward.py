#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Add HFT forward return targets to MTF data.
Generates short-horizon targets for HFT training.
"""


import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_hft_targets(data_dir: str, output_dir: str):
    """Add HFT forward return targets to MTF data."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all parquet files
    parquet_files = glob.glob(f"{data_dir}/**/*.parquet", recursive=True)
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    for file_path in parquet_files:
        logger.info(f"Processing {file_path}")
        
        # Load data
        df = pd.read_parquet(file_path)
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['ts'], unit='ns')
        df = df.sort_values('datetime')
        
        # Calculate forward returns for HFT horizons
        # 5m data -> calculate returns for 15m, 30m, 60m, 120m ahead
        
        # 15m ahead (3 bars)
        df['fwd_ret_15m'] = df['close'].pct_change(3).shift(-3)
        
        # 30m ahead (6 bars) 
        df['fwd_ret_30m'] = df['close'].pct_change(6).shift(-6)
        
        # 60m ahead (12 bars)
        df['fwd_ret_60m'] = df['close'].pct_change(12).shift(-12)
        
        # 120m ahead (24 bars)
        df['fwd_ret_120m'] = df['close'].pct_change(24).shift(-24)
        
        # Same-day open to close (session-anchored)
        # Group by date and calculate open to close return
        df['date'] = df['datetime'].dt.date
        df['session_open'] = df.groupby('date')['open'].transform('first')
        df['session_close'] = df.groupby('date')['close'].transform('last')
        df['fwd_ret_oc_same_day'] = (df['session_close'] / df['session_open'] - 1)
        
        # Remove temporary columns
        df = df.drop(['datetime', 'date', 'session_open', 'session_close'], axis=1)
        
        # Create output path
        rel_path = os.path.relpath(file_path, data_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save updated data
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to {output_path}")
        
        # Log the new target columns
        new_targets = ['fwd_ret_15m', 'fwd_ret_30m', 'fwd_ret_60m', 'fwd_ret_120m', 'fwd_ret_oc_same_day']
        logger.info(f"Added targets: {new_targets}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add HFT forward return targets")
    parser.add_argument("--data-dir", default="5m_comprehensive_features_final", help="Input data directory")
    parser.add_argument("--output-dir", default="5m_comprehensive_features_hft", help="Output data directory")
    
    args = parser.parse_args()
    
    add_hft_targets(args.data_dir, args.output_dir)
