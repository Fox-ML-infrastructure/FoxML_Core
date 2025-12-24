# MIT License - see LICENSE file

"""
Run Feature Selection for Multiple Targets (GPU-Accelerated)

Runs select_features.py for each important target variable to create
specialized feature sets. This is far more powerful than using one
universal feature set, as different targets require different features.

Automatically detects and uses GPU acceleration (CUDA/OpenCL) if available.
Each target is processed sequentially when using GPU to avoid memory conflicts.

Usage:
    # All targets, all symbols (GPU accelerated if available)
    python SCRIPTS/run_multi_target_selection.py
    
    # Test with specific symbols
    python SCRIPTS/run_multi_target_selection.py --symbols AAPL,MSFT,GOOGL
    
    # Specific targets only
    python SCRIPTS/run_multi_target_selection.py --targets peak_60m,return_5m
    
GPU Note:
    - Config: Set device: "cuda" in CONFIG/feature_selection_config.yaml
    - Each target processes sequentially on GPU for memory efficiency
    - Much faster than CPU parallel processing for large datasets
"""


import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import logging
import yaml
import json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

def check_gpu_status() -> Dict[str, any]:
    """Check if GPU acceleration is available and configured."""
    gpu_info = {
        "gpu_available": False,
        "device": "cpu",
        "gpu_name": None
    }
    
    # Check config
    config_path = _REPO_ROOT / "CONFIG" / "feature_selection_config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        device = config.get('lightgbm', {}).get('device', 'cpu').lower()
        gpu_info["device"] = device
        
        if device in ['cuda', 'gpu']:
            # Try to detect GPU
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    gpu_info["gpu_available"] = True
                    gpu_info["gpu_name"] = result.stdout.strip().split('\n')[0]
            except:
                pass
    
    return gpu_info

def load_target_configs(config_path: Path = None) -> Dict[str, Dict]:
    """Load target configurations from YAML file."""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "target_configs.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ Target configs not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Extract and filter enabled targets
    targets = config.get('targets', {})
    enabled_targets = {
        name: cfg for name, cfg in targets.items()
        if cfg.get('enabled', True)
    }
    
    logger.info(f"Loaded {len(enabled_targets)} enabled targets from {config_path}")
    return enabled_targets

def run_feature_selection(
    target_name: str,
    config: Dict,
    symbols: str = None,
    num_workers: int = 12,
    data_dir: Path = None,
    base_output_dir: Path = None
) -> Dict:
    """
    Run feature selection for a single target.
    
    Returns:
        dict with results metadata
    """
    if data_dir is None:
        data_dir = _REPO_ROOT / "data" / "data_labeled" / "interval=5m"
    
    if base_output_dir is None:
        base_output_dir = _REPO_ROOT / "DATA_PROCESSING" / "data" / "features"
    
    output_dir = base_output_dir / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🎯 Running Feature Selection: {target_name}")
    logger.info(f"{'='*80}")
    logger.info(f"Target: {config['target_column']}")
    logger.info(f"Description: {config.get('description', 'N/A')}")
    logger.info(f"Use Case: {config.get('use_case', 'N/A')}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'-'*80}")
    
    # Build command
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "select_features.py"),
        "--target-column", config['target_column'],
        "--top-n", str(config['top_n']),
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--num-workers", str(num_workers),
        "--method", config.get('method', 'mean')
    ]
    
    if symbols:
        cmd.extend(["--symbols", symbols])
    
    # Run
    start_time = datetime.now()
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        elapsed = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✅ Completed {target_name} in {elapsed:.1f}s")
        
        return {
            "target_name": target_name,
            "target_column": config['target_column'],
            "status": "success",
            "elapsed_seconds": elapsed,
            "output_dir": str(output_dir),
            "selected_features_file": str(output_dir / "selected_features.txt")
        }
    
    except subprocess.CalledProcessError as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.error(f"❌ Failed {target_name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        
        return {
            "target_name": target_name,
            "target_column": config['target_column'],
            "status": "failed",
            "elapsed_seconds": elapsed,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Run feature selection for multiple targets with GPU support.")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (optional, for testing)")
    parser.add_argument("--targets", type=str, help="Comma-separated list of target names to run (default: all enabled)")
    parser.add_argument("--num-workers", type=int, default=12, help="Number of parallel workers per target (ignored if GPU mode)")
    parser.add_argument("--data-dir", type=Path, help="Data directory (default: from config)")
    parser.add_argument("--output-dir", type=Path, help="Base output directory (default: from config)")
    parser.add_argument("--config", type=Path, help="Path to custom target configs YAML (overrides default)")
    
    args = parser.parse_args()
    
    # Check GPU status
    gpu_info = check_gpu_status()
    use_gpu = gpu_info["device"] in ['cuda', 'gpu'] and gpu_info["gpu_available"]
    
    # Load target configurations
    target_configs = load_target_configs(args.config)
    if not target_configs:
        return 1
    
    # Determine which targets to run
    if args.targets:
        target_names = args.targets.split(',')
        targets_to_run = {name: target_configs[name] for name in target_names if name in target_configs}
        if not targets_to_run:
            logger.error(f"❌ No valid targets found. Available: {', '.join(target_configs.keys())}")
            return 1
    else:
        targets_to_run = target_configs
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Multi-Target Feature Selection")
    logger.info(f"{'='*80}")
    logger.info(f"Targets to run: {len(targets_to_run)}")
    for name in targets_to_run:
        logger.info(f"  • {name}: {targets_to_run[name]['target_column']}")
    
    # Log GPU/CPU mode
    if use_gpu:
        logger.info(f"🎮 Device: {gpu_info['device'].upper()} ({gpu_info['gpu_name']})")
        logger.info(f"Processing: Sequential per target (GPU processes symbols one-by-one)")
        logger.info(f"Note: Each target runs on GPU sequentially to avoid memory conflicts")
    else:
        logger.info(f"🖥️  Device: CPU")
        logger.info(f"Workers per target: {args.num_workers}")
    
    if args.symbols:
        logger.info(f"Symbols (test mode): {args.symbols}")
    else:
        logger.info(f"Symbols: ALL (full universe)")
    logger.info(f"{'='*80}\n")
    
    # Run feature selection for each target
    results = []
    for target_name, config in targets_to_run.items():
        result = run_feature_selection(
            target_name=target_name,
            config=config,
            symbols=args.symbols,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            base_output_dir=args.output_dir
        )
        results.append(result)
    
    # Save summary
    if args.output_dir:
        summary_file = args.output_dir / "multi_target_summary.json"
    else:
        summary_file = _REPO_ROOT / "DATA_PROCESSING" / "data" / "features" / "multi_target_summary.json"
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "targets_run": len(results),
        "symbols": args.symbols if args.symbols else "ALL",
        "device": gpu_info["device"],
        "gpu_name": gpu_info.get("gpu_name"),
        "gpu_accelerated": use_gpu,
        "results": results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 Summary")
    logger.info(f"{'='*80}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    logger.info(f"✅ Successful: {len(successful)}/{len(results)}")
    logger.info(f"❌ Failed: {len(failed)}/{len(results)}")
    logger.info(f"\nTotal time: {sum(r['elapsed_seconds'] for r in results):.1f}s")
    
    if use_gpu:
        logger.info(f"🎮 Accelerated by: {gpu_info['gpu_name']} ({gpu_info['device'].upper()} mode)")
    else:
        logger.info(f"🖥️  Processed on: CPU ({args.num_workers} workers per target)")
    
    logger.info(f"Summary saved: {summary_file}")
    
    if successful:
        logger.info(f"\n💡 Feature sets created:")
        for r in successful:
            logger.info(f"  • {r['target_name']}: {r['selected_features_file']}")
    
    if failed:
        logger.warning(f"\n⚠️  Failed targets:")
        for r in failed:
            logger.warning(f"  • {r['target_name']}: {r.get('error', 'Unknown error')}")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Multi-target feature selection complete!")
    logger.info(f"{'='*80}")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())

