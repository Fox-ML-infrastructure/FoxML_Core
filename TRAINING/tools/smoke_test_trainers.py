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
Smoke Test for All 20 Trainers

Tests that all trainers can import, train, and predict without numerical issues.
"""


import importlib, numpy as np, traceback, sys, os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import the registry
from models.registry import TRAINER_REGISTRY
FAMILIES = TRAINER_REGISTRY

def tiny_dataset(n=4096, d=64, seed=0):
    """Create a tiny dataset with some NaNs/Infs to test guards"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = (X[:, 0] * 0.1 + rng.standard_normal(n) * 0.01).astype(np.float32)
    
    # Sprinkle NaNs/Infs to test guards
    X[::701, 3] = np.inf
    X[::541, 7] = -np.inf
    X[::379, 9] = np.nan
    y[::997] = np.nan
    
    return X, y

def test_trainer(name, mod_name, cls_name):
    """Test a single trainer"""
    try:
        # Import trainer
        Trainer = getattr(importlib.import_module(mod_name), cls_name)
        
        # Create trainer instance
        trainer = Trainer()
        
        # Create test data
        X, y = tiny_dataset()
        
        # Train
        model = trainer.train(X, y)
        
        # Predict
        preds = trainer.predict(X[:128])
        
        # Check predictions are finite
        assert np.isfinite(preds).all(), f"{name} produced non-finite predictions"
        
        print(f"[OK] {name}")
        return True
        
    except Exception as e:
        print(f"[FAIL] {name}")
        print(f"  Error: {e}")
        print(f"  Traceback: {traceback.format_exc()}")
        return False

def main():
    """Run smoke tests for all trainers"""
    print("🧪 Running smoke tests for all 20 trainers...")
    print("=" * 60)
    
    X, y = tiny_dataset()
    print(f"📊 Test dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"📊 Data contains: {np.isnan(X).sum()} NaN features, {np.isnan(y).sum()} NaN targets")
    print(f"📊 Data contains: {np.isinf(X).sum()} Inf features, {np.isinf(y).sum()} Inf targets")
    print()
    
    results = {}
    for name, (mod, cls) in FAMILIES.items():
        results[name] = test_trainer(name, mod, cls)
    
    print()
    print("=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("🎉 All trainers passed smoke tests!")
        return 0
    else:
        print("⚠️  Some trainers failed - check logs above")
        return 1

if __name__ == "__main__":
    exit(main())
