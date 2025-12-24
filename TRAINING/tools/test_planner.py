#!/usr/bin/env python3

# MIT License - see LICENSE file

"""Quick test to verify plan_for_family returns correct thread counts."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TRAINING.common.threads import plan_for_family

families = [
    "LightGBM",
    "QuantileLightGBM", 
    "XGBoost",
    "RewardBased",
    "Ensemble",
    "NGBoost",
    "MLP",
    "CNN1D"
]

print("Testing plan_for_family with total_threads=14:")
print("=" * 60)
for f in families:
    plan = plan_for_family(f, 14)
    status = "✅" if plan["OMP"] == 14 or f in ["RewardBased", "MLP", "CNN1D"] else "❌"
    print(f"{status} {f:20s} → OMP={plan['OMP']:2d} MKL={plan['MKL']}")

print("\n" + "=" * 60)
print("Expected:")
print("  - OMP-heavy (LGBM/XGB/QuantileLGBM): OMP=14")
print("  - BLAS-only (RewardBased): OMP=1")
print("  - GPU (MLP/CNN1D): OMP=1")

