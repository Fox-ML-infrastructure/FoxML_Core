#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

def add_hft_targets(data_dir: str, output_dir: str, interval_minutes: float = 5.0):
    """
    Add HFT forward return targets to MTF data.
    
    Args:
        data_dir: Input data directory
        output_dir: Output data directory
        interval_minutes: Bar interval in minutes (default: 5.0 for 5m data)
    """
    
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
        
        # CRITICAL: Convert horizon_minutes to horizon_bars
        # Calculate forward returns for HFT horizons
        horizons_minutes = [15, 30, 60, 120]
        
        for horizon_minutes in horizons_minutes:
            horizon_bars = int(horizon_minutes / interval_minutes)
            if abs(horizon_bars * interval_minutes - horizon_minutes) > 0.01:
                logger.warning(
                    f"⚠️  Horizon {horizon_minutes}m is not a multiple of interval {interval_minutes}m. "
                    f"Using {horizon_bars} bars = {horizon_bars * interval_minutes:.1f}m"
                )
            
            # TIME CONTRACT: Label starts at t+1
            # Forward return: (close[t+horizon_bars] / close[t]) - 1
            # Using shift(-horizon_bars) to get future close, then compute return
            col_name = f'fwd_ret_{horizon_minutes}m'
            df[col_name] = (df['close'].shift(-horizon_bars) / df['close'] - 1)
        
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
    parser.add_argument("--interval-minutes", type=float, default=5.0, help="Bar interval in minutes (default: 5.0)")
    
    args = parser.parse_args()
    
    add_hft_targets(args.data_dir, args.output_dir, interval_minutes=args.interval_minutes)
