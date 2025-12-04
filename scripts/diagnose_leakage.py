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
Diagnose why leakage is still appearing despite filtering.

This script checks:
1. Config cache status
2. Actual features being used
3. Perfect correlations
4. CV fold issues
5. Target degeneracy

Usage:
    python scripts/diagnose_leakage.py <target_column> [--symbol AAPL]
"""


import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit

# Add project root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils.leakage_filtering import filter_features_for_target, _load_leakage_config, _CONFIG_PATH


def diagnose_leakage(target_column: str, symbol: str = "AAPL", data_dir: Path = None):
    """Diagnose why leakage might still be appearing."""
    
    if data_dir is None:
        data_dir = _REPO_ROOT / "data" / "data_labeled" / "interval=5m"
    
    # Load data
    symbol_dir = data_dir / f"symbol={symbol}"
    parquet_file = symbol_dir / f"{symbol}.parquet"
    
    if not parquet_file.exists():
        print(f"ERROR: Data file not found: {parquet_file}")
        return
    
    df = pd.read_parquet(parquet_file)
    all_cols = df.columns.tolist()
    
    print("="*80)
    print("LEAKAGE DIAGNOSIS")
    print("="*80)
    print(f"Target: {target_column}")
    print(f"Symbol: {symbol}")
    print()
    
    # 1. Check config
    print("="*80)
    print("1. CONFIG STATUS")
    print("="*80)
    print(f"Config path: {_CONFIG_PATH}")
    print(f"Config exists: {_CONFIG_PATH.exists()}")
    
    if _CONFIG_PATH.exists():
        config = _load_leakage_config(force_reload=True)
        always_exclude = config.get('always_exclude', {})
        barrier_in_regex = any('barrier_' in p for p in always_exclude.get('regex_patterns', []))
        barrier_in_prefix = 'barrier_' in always_exclude.get('prefix_patterns', [])
        
        print(f"barrier_* in regex patterns: {barrier_in_regex}")
        print(f"barrier_* in prefix patterns: {barrier_in_prefix}")
        
        if not (barrier_in_regex or barrier_in_prefix):
            print("⚠️  PROBLEM: barrier_* patterns not found in config!")
        else:
            print("✓ barrier_* patterns found in config")
    print()
    
    # 2. Check filtering
    print("="*80)
    print("2. FILTERING CHECK")
    print("="*80)
    
    safe_columns = filter_features_for_target(all_cols, target_column, verbose=False)
    print(f"Safe features: {len(safe_columns)}")
    
    # Check for known leaks
    known_leaks = ['barrier_up_60m_0.8', 'barrier_down_60m_0.8', 'p_up_60m_0.8', 'p_down_60m_0.8']
    leaks_found = [f for f in safe_columns if f in known_leaks]
    
    # Check for horizon-based leaks
    from scripts.utils.leakage_filtering import _extract_horizon
    target_horizon = _extract_horizon(target_column, config)
    horizon_leaks = []
    if target_horizon:
        threshold = target_horizon / 4
        for col in safe_columns:
            col_horizon = _extract_horizon(col, config)
            if col_horizon and col_horizon >= threshold:
                horizon_leaks.append((col, col_horizon))
    
    # Check for HIGH/LOW-based leaks
    high_low_leaks = []
    target_lower = target_column.lower()
    if 'peak' in target_lower:
        for col in safe_columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['high', 'upper', 'max', 'top', 'ceiling']):
                if not any(allow in col_lower for allow in ['high_vol', 'high_freq', 'high_corr']):
                    high_low_leaks.append(col)
    elif 'valley' in target_lower:
        for col in safe_columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['low', 'lower', 'min', 'bottom', 'floor']):
                if not any(allow in col_lower for allow in ['low_vol', 'low_freq', 'low_corr']):
                    high_low_leaks.append(col)
    
    if leaks_found:
        print(f"⚠️  CRITICAL: {len(leaks_found)} known leaks in safe features:")
        for leak in leaks_found:
            print(f"  - {leak}")
        print()
        print("SOLUTION: Restart Python process to clear config cache!")
    
    if horizon_leaks:
        print(f"⚠️  CRITICAL: {len(horizon_leaks)} horizon-based leaks found:")
        for leak, horizon in sorted(horizon_leaks, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {leak} (horizon: {horizon}m, threshold: {threshold:.0f}m)")
        if len(horizon_leaks) > 10:
            print(f"  ... and {len(horizon_leaks) - 10} more")
        print()
        print("SOLUTION: These should be excluded by horizon overlap rule (target/4)")
        print("          Restart Python process to clear config cache!")
    
    if high_low_leaks:
        leak_type = "HIGH" if 'peak' in target_lower else "LOW"
        print(f"⚠️  CRITICAL: {len(high_low_leaks)} {leak_type}-based leaks found:")
        for leak in high_low_leaks[:10]:
            print(f"  - {leak}")
        if len(high_low_leaks) > 10:
            print(f"  ... and {len(high_low_leaks) - 10} more")
        print()
        print(f"SOLUTION: {leak_type}-based features should be excluded for {target_lower} targets")
        print("          Restart Python process to clear config cache!")
    
    if not leaks_found and not horizon_leaks and not high_low_leaks:
        print("✓ No known leaks in safe features")
    print()
    
    # 3. Check perfect correlations
    print("="*80)
    print("3. PERFECT CORRELATION CHECK")
    print("="*80)
    
    df_filtered = df[safe_columns + [target_column]].dropna(subset=[target_column])
    X = df_filtered[safe_columns]
    y = df_filtered[target_column]
    
    # Sample for speed
    sample_size = min(5000, len(X))
    X_sample = X.sample(n=sample_size, random_state=42)
    y_sample = y.loc[X_sample.index]
    
    perfect_corr = []
    for col in X_sample.columns:
        try:
            df_check = pd.DataFrame({'y': y_sample, 'x': X_sample[col]}).dropna()
            if len(df_check) > 100:
                corr = df_check['y'].corr(df_check['x'])
                if not pd.isna(corr) and abs(corr) >= 0.99:
                    perfect_corr.append((col, corr))
        except:
            pass
    
    if perfect_corr:
        print(f"⚠️  Found {len(perfect_corr)} features with |correlation| >= 0.99:")
        for col, corr in sorted(perfect_corr, key=lambda x: abs(x[1]), reverse=True)[:10]:
            print(f"  {col:50s} | {corr:.4f}")
        print()
        print("These are LEAKS - add to exclusion patterns!")
    else:
        print("✓ No perfect correlations found")
    print()
    
    # 4. Check target distribution
    print("="*80)
    print("4. TARGET DISTRIBUTION")
    print("="*80)
    print(f"Total samples: {len(y)}")
    print(f"Value counts:")
    print(y.value_counts().sort_index())
    print(f"Class balance: {y.mean():.1%} positive")
    
    if y.mean() > 0.99 or y.mean() < 0.01:
        print("⚠️  WARNING: Target is extremely imbalanced")
    print()
    
    # 5. Check CV folds
    print("="*80)
    print("5. CV FOLD CHECK")
    print("="*80)
    
    tscv = TimeSeriesSplit(n_splits=3)
    X_array = X.values
    y_array = y.values
    
    degenerate_folds = []
    for i, (train_idx, test_idx) in enumerate(tscv.split(X_array)):
        y_train = y_array[train_idx]
        y_test = y_array[test_idx]
        
        train_balance = y_train.mean()
        test_balance = y_test.mean()
        
        print(f"Fold {i+1}:")
        print(f"  Train: {len(train_idx)} samples, {train_balance:.1%} positive")
        print(f"  Test:  {len(test_idx)} samples, {test_balance:.1%} positive")
        
        if train_balance > 0.99 or train_balance < 0.01:
            print(f"  ⚠️  Train set is degenerate (all {int(train_balance > 0.5)})")
            degenerate_folds.append(f"Fold {i+1} train")
        if test_balance > 0.99 or test_balance < 0.01:
            print(f"  ⚠️  Test set is degenerate (all {int(test_balance > 0.5)})")
            degenerate_folds.append(f"Fold {i+1} test")
        print()
    
    if degenerate_folds:
        print(f"⚠️  WARNING: {len(degenerate_folds)} degenerate folds found")
        print("   This can cause perfect scores even without leaks!")
    else:
        print("✓ No degenerate folds found")
    print()
    
    # 6. Recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    
    if leaks_found:
        print("1. ⚠️  CRITICAL: Known leaks found in safe features")
        print("   → Restart Python process to clear config cache")
        print("   → The config was updated but the process is using cached version")
        print()
    
    if perfect_corr:
        print("2. ⚠️  Perfect correlations found")
        print("   → Add these features to CONFIG/excluded_features.yaml")
        print("   → Restart Python after updating config")
        print()
    
    if degenerate_folds:
        print("3. ⚠️  Degenerate folds found")
        print("   → This can cause perfect scores even without leaks")
        print("   → Consider using stratified CV or filtering degenerate targets")
        print()
    
    if not leaks_found and not perfect_corr and not degenerate_folds:
        print("✓ No obvious issues found")
        print()
        print("If models still show 100% accuracy:")
        print("  1. Check feature combinations (multiple features together)")
        print("  2. Verify CV is using TimeSeriesSplit correctly")
        print("  3. Check if models are overfitting (train vs CV scores)")
        print("  4. Look at feature importances for suspicious patterns")


def main():
    parser = argparse.ArgumentParser(description="Diagnose leakage issues")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--symbol", default="AAPL", help="Symbol to test")
    parser.add_argument("--data-dir", type=Path, help="Data directory")
    
    args = parser.parse_args()
    
    diagnose_leakage(args.target, args.symbol, args.data_dir)


if __name__ == "__main__":
    main()

