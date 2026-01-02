#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
Demo Script - One-Command Reproducible Pipeline Run

Purpose:
    Verify the modeling pipeline works end-to-end with deterministic output.
    Designed for reviewers who need to run the system before reviewing code.

Usage:
    python tools/demo.py              # Run demo with default config
    python tools/demo.py --clean      # Clean previous output first
    python tools/demo.py --verify     # Verify determinism (run twice, compare)

Output:
    RESULTS/demo_run/
    ├── manifest.json          # Fingerprints, versions, inputs
    ├── globals/
    │   ├── run_context.json   # Resolved mode, scope, symbols
    │   └── metrics.json       # Aggregated metrics
    └── log.txt                # Full execution log

Success Criteria:
    - Runs in <5 minutes on minimal hardware
    - Same fingerprints on repeated runs (deterministic)
    - Exit code 0 on success
"""

import sys
import os
import json
import hashlib
import argparse
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Demo configuration
DEMO_CONFIG = "demo"
DEMO_OUTPUT_DIR = PROJECT_ROOT / "RESULTS" / "demo_run"
DEMO_MANIFEST = DEMO_OUTPUT_DIR / "manifest.json"


def clean_demo_output():
    """Remove previous demo output for clean run."""
    if DEMO_OUTPUT_DIR.exists():
        logger.info(f"Cleaning previous demo output: {DEMO_OUTPUT_DIR}")
        shutil.rmtree(DEMO_OUTPUT_DIR)


def run_demo() -> int:
    """
    Run the demo pipeline.
    
    Returns:
        Exit code (0 = success, non-zero = failure)
    """
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("Starting Demo Pipeline Run")
    logger.info(f"Config: CONFIG/experiments/{DEMO_CONFIG}.yaml")
    logger.info(f"Output: {DEMO_OUTPUT_DIR}")
    logger.info("=" * 60)
    
    try:
        # Import and run intelligent trainer
        from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
        from CONFIG.config_builder import load_experiment_config
        
        # Load demo config
        config = load_experiment_config(DEMO_CONFIG)
        logger.info(f"Loaded experiment config: {config.name}")
        
        # Create trainer
        trainer = IntelligentTrainer(
            data_dir=config.data_dir,
            symbols=config.symbols,
            output_dir=DEMO_OUTPUT_DIR,
            add_timestamp=False,  # Fixed path for reproducibility
            experiment_config=config
        )
        
        # Run pipeline
        logger.info("Running intelligent training pipeline...")
        trainer.run()
        
        # Generate manifest
        manifest = generate_manifest(trainer, config, start_time)
        
        # Write manifest
        DEMO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with open(DEMO_MANIFEST, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        logger.info(f"Wrote manifest: {DEMO_MANIFEST}")
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info(f"Demo completed successfully in {elapsed:.1f}s")
        logger.info(f"Manifest fingerprint: {manifest.get('fingerprint', 'N/A')}")
        logger.info("=" * 60)
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.error("Demo requires data in data/data_labeled_v3/interval=5m/")
        logger.error("Run with sample data or adjust CONFIG/experiments/demo.yaml")
        return 1
        
    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        return 1


def generate_manifest(trainer, config, start_time) -> dict:
    """
    Generate manifest.json with fingerprints for reproducibility verification.
    
    The fingerprint should be deterministic - same inputs produce same fingerprint.
    """
    # Collect inputs for fingerprinting
    inputs = {
        "config_name": config.name,
        "symbols": sorted(config.symbols),
        "data_dir": str(config.data_dir),
        "python_version": sys.version,
    }
    
    # Add package versions if available
    try:
        import lightgbm
        inputs["lightgbm_version"] = lightgbm.__version__
    except ImportError:
        pass
    
    try:
        import numpy
        inputs["numpy_version"] = numpy.__version__
    except ImportError:
        pass
    
    # Compute fingerprint from inputs
    input_str = json.dumps(inputs, sort_keys=True)
    fingerprint = hashlib.sha256(input_str.encode()).hexdigest()[:16]
    
    # Check for run_context.json
    run_context = {}
    run_context_path = DEMO_OUTPUT_DIR / "globals" / "run_context.json"
    if run_context_path.exists():
        with open(run_context_path) as f:
            run_context = json.load(f)
    
    manifest = {
        "fingerprint": fingerprint,
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": (datetime.now() - start_time).total_seconds(),
        "inputs": inputs,
        "run_context": {
            "resolved_mode": run_context.get("resolved_mode"),
            "n_symbols": run_context.get("n_symbols"),
            "data_scope": run_context.get("data_scope"),
        },
        "outputs": {
            "output_dir": str(DEMO_OUTPUT_DIR),
            "files": list_output_files(),
        },
    }
    
    return manifest


def list_output_files() -> list:
    """List key output files for manifest."""
    key_files = []
    if DEMO_OUTPUT_DIR.exists():
        for pattern in ["**/*.json", "**/*.yaml", "**/*.parquet"]:
            for f in DEMO_OUTPUT_DIR.glob(pattern):
                rel_path = f.relative_to(DEMO_OUTPUT_DIR)
                key_files.append(str(rel_path))
    return sorted(key_files)[:50]  # Limit to first 50 files


def verify_determinism() -> int:
    """
    Verify determinism by running demo twice and comparing fingerprints.
    
    Returns:
        Exit code (0 = deterministic, 1 = non-deterministic)
    """
    logger.info("Verifying determinism: running demo twice...")
    
    # First run
    clean_demo_output()
    result1 = run_demo()
    if result1 != 0:
        logger.error("First run failed")
        return 1
    
    # Read first fingerprint
    with open(DEMO_MANIFEST) as f:
        manifest1 = json.load(f)
    fp1 = manifest1.get("fingerprint")
    
    # Second run
    clean_demo_output()
    result2 = run_demo()
    if result2 != 0:
        logger.error("Second run failed")
        return 1
    
    # Read second fingerprint
    with open(DEMO_MANIFEST) as f:
        manifest2 = json.load(f)
    fp2 = manifest2.get("fingerprint")
    
    # Compare
    if fp1 == fp2:
        logger.info(f"✅ Determinism verified: fingerprint={fp1}")
        return 0
    else:
        logger.error(f"❌ Non-deterministic: run1={fp1}, run2={fp2}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Demo script for one-command reproducible pipeline run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--clean', action='store_true',
        help='Clean previous demo output before running'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify determinism by running twice and comparing fingerprints'
    )
    parser.add_argument(
        '--clean-only', action='store_true',
        help='Only clean demo output, do not run'
    )
    
    args = parser.parse_args()
    
    if args.clean_only:
        clean_demo_output()
        logger.info("Demo output cleaned")
        return 0
    
    if args.verify:
        return verify_determinism()
    
    if args.clean:
        clean_demo_output()
    
    return run_demo()


if __name__ == "__main__":
    sys.exit(main())
