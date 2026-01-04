#!/usr/bin/env python3
"""
Two-Process Strict Determinism Test

This test verifies that strict mode produces bitwise identical results
across two separate Python processes.

Run with:
    pytest TRAINING/contract_tests/test_determinism_strict.py -v

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""

import subprocess
import json
import os
import tempfile
import sys
from pathlib import Path

import pytest


# Skip if not running in strict mode or if lightgbm not available
def _can_run_strict_test():
    """Check if strict test can run."""
    try:
        import lightgbm
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _can_run_strict_test(), reason="LightGBM not available")
def test_strict_determinism_two_process():
    """
    Run same job in TWO SEPARATE PROCESSES.
    Assert bitwise identical results.
    
    Captures and compares:
    - Prediction bytes hash (bitwise)
    - Resolved params JSON (exact match)
    - Thread env snapshot
    - Key library versions
    """
    # Get project root
    project_root = Path(__file__).resolve().parents[2]
    launcher = project_root / "bin" / "run_deterministic.sh"
    
    if not launcher.exists():
        pytest.skip(f"Launcher not found: {launcher}")
    
    # Test script that runs a minimal training job
    test_script = '''
# Bootstrap MUST be first
import TRAINING.common.repro_bootstrap

import json
import hashlib
import os
import numpy as np
from TRAINING.common.determinism import (
    create_estimator, resolve_seed, load_reproducibility_config, seed_all
)

seed_all()
config = load_reproducibility_config()

# Create a simple dataset
# NOTE: We use seed_all() which sets np.random.seed internally
X = np.random.randn(100, 10)
y = np.random.randn(100)

# Create and fit model
seed = resolve_seed(config["seed"], "test", target="test_target")
base_config = {"n_estimators": 10}
model = create_estimator("lightgbm", base_config, seed, "regression")

# Capture the CANONICAL merged config we passed
canonical_input = {
    "library": "lightgbm",
    "base_config": base_config,
    "seed": seed,
    "problem_kind": "regression",
    "mode": config["mode"],
}

model.fit(X, y)

# Get predictions
preds = model.predict(X)
pred_hash = hashlib.sha256(preds.tobytes()).hexdigest()

# Get resolved params (what the model actually got)
resolved_params = {k: v for k, v in model.get_params().items() if v is not None}
resolved_params_json = json.dumps(resolved_params, sort_keys=True, default=str)

# Get library versions
def get_version(lib):
    try:
        return __import__(lib).__version__
    except:
        return "N/A"

versions = {
    "lightgbm": get_version("lightgbm"),
    "numpy": get_version("numpy"),
}

# Output
result = {
    "seed": seed,
    "pred_hash": pred_hash,
    "canonical_input": canonical_input,
    "resolved_params_json": resolved_params_json,
    "env": {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
        "REPRO_MODE": os.environ.get("REPRO_MODE"),
    },
    "versions": versions,
}
print(json.dumps(result, default=str))
'''
    
    # Write test script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        # Prepare environment
        env = {
            **os.environ,
            "PYTHONHASHSEED": "42",
            "REPRO_MODE": "strict",
            "PYTHONPATH": str(project_root),
        }
        
        # Run 1
        result1 = subprocess.run(
            [str(launcher), script_path],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
        
        if result1.returncode != 0:
            print(f"Run 1 stderr:\n{result1.stderr}")
            pytest.fail(f"Run 1 failed with code {result1.returncode}")
        
        # Parse output (last line is JSON)
        output1 = json.loads(result1.stdout.strip().split('\n')[-1])
        
        # Run 2 (SEPARATE PROCESS, same command line, same working directory)
        result2 = subprocess.run(
            [str(launcher), script_path],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
        
        if result2.returncode != 0:
            print(f"Run 2 stderr:\n{result2.stderr}")
            pytest.fail(f"Run 2 failed with code {result2.returncode}")
        
        output2 = json.loads(result2.stdout.strip().split('\n')[-1])
        
        # STRICT ASSERTIONS
        assert output1["seed"] == output2["seed"], \
            f"Seeds differ: {output1['seed']} vs {output2['seed']}"
        
        # Bitwise prediction hash match
        assert output1["pred_hash"] == output2["pred_hash"], \
            f"Predictions differ (bitwise): {output1['pred_hash'][:16]}... vs {output2['pred_hash'][:16]}..."
        
        # Resolved params exact match
        assert output1["resolved_params_json"] == output2["resolved_params_json"], \
            f"Resolved params differ"
        
        # Env vars set correctly
        assert output1["env"]["PYTHONHASHSEED"] == "42", \
            f"PYTHONHASHSEED not set correctly: {output1['env']['PYTHONHASHSEED']}"
        assert output1["env"]["OMP_NUM_THREADS"] == "1", \
            f"OMP_NUM_THREADS not 1 in strict mode: {output1['env']['OMP_NUM_THREADS']}"
        assert output1["env"]["MKL_NUM_THREADS"] == "1", \
            f"MKL_NUM_THREADS not 1 in strict mode: {output1['env']['MKL_NUM_THREADS']}"
        assert output1["env"]["REPRO_MODE"] == "strict", \
            f"REPRO_MODE not strict: {output1['env']['REPRO_MODE']}"
        
        # Versions match
        assert output1["versions"] == output2["versions"], \
            f"Library versions differ: {output1['versions']} vs {output2['versions']}"
        
        print("✅ Two-process strict determinism test PASSED")
        print(f"   Prediction hash: {output1['pred_hash'][:16]}...")
        print(f"   Seed: {output1['seed']}")
        print(f"   Versions: {output1['versions']}")
        
    finally:
        os.unlink(script_path)


@pytest.mark.skipif(not _can_run_strict_test(), reason="LightGBM not available")
def test_bootstrap_required_in_strict():
    """Test that strict mode fails if bootstrap wasn't run."""
    project_root = Path(__file__).resolve().parents[2]
    
    # Script that imports determinism without bootstrap
    test_script = '''
import os
os.environ["REPRO_MODE"] = "strict"

try:
    from TRAINING.common.determinism import create_estimator
    print("FAIL: Should have raised RuntimeError")
    exit(1)
except RuntimeError as e:
    if "repro_bootstrap" in str(e):
        print("PASS: Got expected RuntimeError about bootstrap")
        exit(0)
    else:
        print(f"FAIL: Wrong error: {e}")
        exit(1)
'''
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(test_script)
        script_path = f.name
    
    try:
        env = {
            **os.environ,
            "PYTHONPATH": str(project_root),
            "REPRO_MODE": "strict",
        }
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env=env,
        )
        
        # Check output
        if "PASS" in result.stdout:
            print("✅ Bootstrap enforcement test PASSED")
        else:
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            pytest.fail("Bootstrap enforcement not working")
            
    finally:
        os.unlink(script_path)


if __name__ == "__main__":
    test_strict_determinism_two_process()
    test_bootstrap_required_in_strict()
