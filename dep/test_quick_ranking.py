#!/usr/bin/env python

# MIT License - see LICENSE file

"""
Quick test of target ranking - shows output immediately
Tests on 1 symbol, 1 target to verify everything works
"""


import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add project root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("="*80)
print("QUICK TEST: Target Ranking (1 symbol, 1 target)")
print("="*80)
print()

# Test 1: Check data availability
print("1️⃣  Checking data availability...")
data_dir = _REPO_ROOT / "data/data_labeled/interval=5m"
if not data_dir.exists():
    print(f"❌ Data directory not found: {data_dir}")
    sys.exit(1)

symbol_dirs = [d for d in data_dir.glob("symbol=*") if d.is_dir()]
if not symbol_dirs:
    print(f"❌ No symbol directories found in {data_dir}")
    sys.exit(1)

print(f"✅ Found {len(symbol_dirs)} symbols in dataset")

# Pick first available symbol
test_symbol = None
for symbol_dir in symbol_dirs:
    symbol_name = symbol_dir.name.replace("symbol=", "")
    parquet_file = symbol_dir / f"{symbol_name}.parquet"
    if parquet_file.exists():
        test_symbol = symbol_name
        break

if not test_symbol:
    print("❌ No valid symbol data found")
    sys.exit(1)

print(f"✅ Test symbol: {test_symbol}")
print()

# Test 2: Load data and check targets
print("2️⃣  Loading data and checking targets...")
df = pd.read_parquet(data_dir / f"symbol={test_symbol}" / f"{test_symbol}.parquet")
print(f"✅ Loaded {len(df)} rows")

# Find available targets
target_cols = [col for col in df.columns if col.startswith('y_')]
print(f"✅ Found {len(target_cols)} target columns")

if not target_cols:
    print("❌ No target columns found")
    sys.exit(1)

# Pick first binary target
test_target = None
for col in target_cols:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        test_target = col
        break

if not test_target:
    # Just use first target
    test_target = target_cols[0]

print(f"✅ Test target: {test_target}")
print()

# Test 3: Quick model training
print("3️⃣  Testing quick model training...")

from sklearn.model_selection import train_test_split
import lightgbm as lgb

# Prepare data
df_clean = df.dropna(subset=[test_target])
exclude_cols = [col for col in df_clean.columns 
               if col.startswith(('y_', 'fwd_ret_', 'barrier_', 'zigzag_', 'p_')) 
               or col in ['ts', 'datetime', 'symbol', 'interval', 'source']]
X = df_clean.drop(columns=exclude_cols, errors='ignore')
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    X = X.drop(columns=object_cols)
y = df_clean[test_target]

print(f"✅ Features: {X.shape[1]} columns, {X.shape[0]} rows")

# Sample for speed
if len(X) > 5000:
    X, _, y, _ = train_test_split(X, y, train_size=5000, random_state=42, stratify=y)
    print(f"✅ Sampled to 5000 rows for speed")

# Train quick model
model = lgb.LGBMClassifier(n_estimators=50, verbose=-1, random_state=42)
model.fit(X, y)
score = model.score(X, y)

print(f"✅ Model trained! Score: {score:.3f}")

# Get top features
importances = model.feature_importances_
top_indices = np.argsort(importances)[-10:][::-1]
print()
print("📊 Top 10 features:")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. {X.columns[idx]:30s} {importances[idx]:.1f}")

print()
print("="*80)
print("✅ ALL TESTS PASSED - Scripts should work!")
print("="*80)
print()
print("Next steps:")
print("  1. Run target ranking:")
print(f"     python SCRIPTS/rank_target_predictability.py --symbols {test_symbol}")
print()
print("  2. Monitor output:")
print("     ls -lh results/target_rankings/")
print()
print("  3. Run multi-model feature selection:")
print(f"     python SCRIPTS/multi_model_feature_selection.py --symbols {test_symbol} --target-column {test_target}")

