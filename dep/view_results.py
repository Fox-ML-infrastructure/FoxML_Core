#!/usr/bin/env python3

# MIT License - see LICENSE file

"""
View Results
Generate summary report of training results.
"""


import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def collect_results():
    """Collect all training results"""
    models_dir = PROJECT_ROOT / "models"
    
    if not models_dir.exists():
        print("❌ No models directory found")
        return None
    
    # Find all metrics files
    metrics_files = list(models_dir.glob("*_metrics.json"))
    
    if not metrics_files:
        print("❌ No metrics files found")
        return None
    
    # Load all metrics
    results = []
    for metrics_file in metrics_files:
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
                results.append(metrics)
        except Exception as e:
            print(f"⚠️  Failed to load {metrics_file.name}: {e}")
    
    if not results:
        return None
    
    return pd.DataFrame(results)

def print_summary(df):
    """Print summary report"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                      MODEL TRAINING RESULTS SUMMARY                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
    
    print(f"Total Models Trained: {len(df)}")
    print(f"Symbols: {df['symbol'].nunique()}")
    print(f"Model Types: {', '.join(df['model'].unique())}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print()
    
    # Performance by model
    print("="*80)
    print("Performance by Model Type")
    print("="*80)
    
    model_stats = df.groupby('model').agg({
        'val_r2': ['mean', 'std', 'min', 'max'],
        'val_mse': ['mean', 'min'],
        'symbol': 'count'
    }).round(4)
    
    print(model_stats)
    print()
    
    # Top performers
    print("="*80)
    print("Top 10 Performers (by Validation R²)")
    print("="*80)
    
    top_10 = df.nlargest(10, 'val_r2')[['symbol', 'model', 'variant', 'val_r2', 'val_mse']]
    print(top_10.to_string(index=False))
    print()
    
    # Bottom performers (potential issues)
    if len(df) >= 10:
        print("="*80)
        print("Bottom 10 Performers (may need attention)")
        print("="*80)
        
        bottom_10 = df.nsmallest(10, 'val_r2')[['symbol', 'model', 'variant', 'val_r2', 'val_mse']]
        print(bottom_10.to_string(index=False))
        print()
    
    # Variant comparison (if multiple variants used)
    if df['variant'].nunique() > 1:
        print("="*80)
        print("Performance by Variant")
        print("="*80)
        
        variant_stats = df.groupby('variant').agg({
            'val_r2': ['mean', 'std'],
            'symbol': 'count'
        }).round(4)
        
        print(variant_stats)
        print()
    
    # Save detailed report
    report_file = PROJECT_ROOT / f"results/summary_{datetime.now().strftime('%Y-%m-%d')}.csv"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_file, index=False)
    print(f"📊 Detailed report saved to: {report_file}")
    print()

def main():
    print("Collecting results...")
    df = collect_results()
    
    if df is None:
        print("\n❌ No results found")
        print("\nMake sure you've run training:")
        print("  python SCRIPTS/train_single_model.py AAPL lightgbm")
        print("  OR")
        print("  bash SCRIPTS/run_full_pipeline.sh")
        return 1
    
    print(f"✅ Found {len(df)} trained models\n")
    print_summary(df)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

