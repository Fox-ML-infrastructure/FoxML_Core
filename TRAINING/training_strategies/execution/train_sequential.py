#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
One-Command Sequential Model Training (2-Stage Approach)

Trains all 20 models (sequential + cross-sectional) using 2-stage approach:
Stage 1: CPU models (LightGBM, XGBoost, etc.)
Stage 2: GPU models (LSTM, Transformer, MLP, VAE, etc.)
with automatic training plan detection.

Usage:
    python -m TRAINING.training_strategies.train_sequential [data_dir] [symbols...]
    
    Or as a script:
    python TRAINING/training_strategies/train_sequential.py data AAPL MSFT GOOGL

Examples:
    python -m TRAINING.training_strategies.train_sequential data AAPL MSFT GOOGL
    python -m TRAINING.training_strategies.train_sequential data AAPL MSFT GOOGL --training-plan-dir results/METRICS/training_plan
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def main():
    parser = argparse.ArgumentParser(
        description="Train all models (sequential + cross-sectional) with 2-stage approach",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('data_dir', nargs='?', default='data', 
                       help='Data directory (default: data)')
    parser.add_argument('symbols', nargs='*', default=['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
                       help='Symbols to train on (default: AAPL MSFT GOOGL TSLA)')
    parser.add_argument('--output-dir', type=str, 
                       default=f"output/sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help='Output directory (default: output/sequential_YYYYMMDD_HHMMSS)')
    parser.add_argument('--training-plan-dir', type=str,
                       help='Path to training plan directory (auto-detected if not provided)')
    parser.add_argument('--no-training-plan', action='store_true',
                       help='Disable training plan auto-detection')
    
    # Pass through any additional arguments to main.py
    args, unknown_args = parser.parse_known_args()
    
    # Build command arguments for main.py
    cmd_args = [
        '--data-dir', args.data_dir,
        '--symbols'] + args.symbols + [
        '--model-types', 'sequential',
        '--output-dir', args.output_dir
    ]
    
    if args.training_plan_dir:
        cmd_args.extend(['--training-plan-dir', args.training_plan_dir])
    
    if args.no_training_plan:
        cmd_args.append('--no-training-plan')
    
    # Add any unknown args (for flexibility)
    cmd_args.extend(unknown_args)
    
    print("🚀 Training Sequential Models (2-Stage Approach)")
    print("=" * 50)
    print(f"Data dir: {args.data_dir}")
    print(f"Symbols: {' '.join(args.symbols)}")
    print(f"Output: {args.output_dir}")
    print()
    
    # Import and call main.py's main function
    # We'll call it directly with parsed arguments instead of manipulating sys.argv
    from .main import main as main_func
    
    # Create a mock argparse.Namespace for main.py
    # Since main.py uses argparse internally, we need to call it properly
    # The cleanest way is to temporarily modify sys.argv
    original_argv = sys.argv
    try:
        # main.py expects: python -m TRAINING.training_strategies.main [args...]
        sys.argv = ['train_sequential.py'] + cmd_args
        main_func()
    finally:
        sys.argv = original_argv
    
    print()
    print("✅ Sequential model training complete!")
    print(f"📁 Output: {args.output_dir}")

if __name__ == '__main__':
    main()
