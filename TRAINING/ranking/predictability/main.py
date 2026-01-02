# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python SCRIPTS/rank_target_predictability.py
  
  # Test on specific symbols first
  python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python SCRIPTS/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Add project root FIRST (before any scripts.* imports)
# TRAINING/ranking/rank_target_predictability.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# CRITICAL: Set up determinism BEFORE importing any ML libraries
# This ensures reproducible results across runs
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
    base_seed = get_cfg("pipeline.determinism.base_seed", default=42)
except ImportError:
    _CONFIG_AVAILABLE = False
    base_seed = 42  # FALLBACK_DEFAULT_OK

# Import determinism system FIRST (before any ML libraries)
from TRAINING.common.determinism import set_global_determinism, seed_for, stable_seed_from

# Set global determinism immediately
BASE_SEED = set_global_determinism(
    base_seed=base_seed,
    threads=None,  # Auto-detect optimal thread count
    deterministic_algorithms=False,  # Allow parallel algorithms for performance
    prefer_cpu_tree_train=False,  # Use GPU when available
    tf_on=False,  # TensorFlow not needed for ranking
    strict_mode=False  # Allow optimizations
)

# Try to import config loader (for other config access)
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Already handled above

# Import logging config utilities
try:
    from CONFIG.logging_config_utils import get_module_logging_config, get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False
    # Fallback: create a simple config-like object
    class _DummyLoggingConfig:
        def __init__(self):
            self.gpu_detail = False
            self.cv_detail = False
            self.edu_hints = False
            self.detail = False

# Import checkpoint utility (after path is set)
from TRAINING.orchestration.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.common.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.common.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.ranking.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.orchestration.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)

# Main entry point for target predictability ranking

# Import all dependencies
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.predictability.composite_score import calculate_composite_score
from TRAINING.ranking.predictability.data_loading import (
    load_target_configs, discover_all_targets, load_sample_data, 
    prepare_features_and_target, load_multi_model_config
)
from TRAINING.ranking.predictability.model_evaluation import (
    train_and_evaluate_models, evaluate_target_predictability
)
from TRAINING.ranking.predictability.reporting import (
    save_leak_report_summary, save_rankings, _get_recommendation
)

def main():
    parser = argparse.ArgumentParser(
        description="Rank target predictability across model families"
    )
    parser.add_argument("--symbols", type=str, default="AAPL,MSFT,GOOGL,TSLA,JPM",
                       help="Symbols to test on (default: 5 representative stocks)")
    parser.add_argument("--data-dir", type=Path,
                       default=_REPO_ROOT / "data/data_labeled/interval=5m")
    parser.add_argument("--output-dir", type=Path,
                       default=_REPO_ROOT / "results/target_rankings")
    parser.add_argument("--targets", type=str,
                       help="Specific targets to evaluate (comma-separated), default: all enabled")
    parser.add_argument("--discover-all", action="store_true",
                       help="Auto-discover and rank ALL targets from data (ignores config)")
    parser.add_argument("--model-families", type=str,
                       default=None,
                       help="Model families to use (default: use all enabled from multi_model_feature_selection.yaml)")
    parser.add_argument("--multi-model-config", type=Path,
                       default=None,
                       help="Path to multi-model config (default: CONFIG/multi_model_feature_selection.yaml)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint if available")
    parser.add_argument("--clear-checkpoint", action="store_true",
                       help="Clear existing checkpoint and start fresh")
    parser.add_argument("--min-cs", type=int, default=10,
                       help="Minimum cross-sectional size per timestamp (default: 10)")
    parser.add_argument("--max-cs-samples", type=int, default=None,
                       help="Maximum samples per timestamp for cross-sectional sampling (default: 1000)")
    parser.add_argument("--max-rows-per-symbol", type=int, default=50000,
                       help="Maximum rows to load per symbol (most recent rows, default: 50000)")
    
    args = parser.parse_args()
    
    # Parse inputs
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Load multi-model config
    multi_model_config = None
    if args.multi_model_config:
        multi_model_config = load_multi_model_config(args.multi_model_config)
    else:
        multi_model_config = load_multi_model_config()  # Try default path
    
    # Determine model families
    if args.model_families:
        model_families = [m.strip() for m in args.model_families.split(',')]
    elif multi_model_config:
        # Use enabled models from config
        model_families_dict = multi_model_config.get('model_families', {})
        if model_families_dict is None or not isinstance(model_families_dict, dict):
            logger.warning("model_families in config is None or not a dict. Using defaults.")
            model_families = ['lightgbm', 'random_forest', 'neural_network']
        else:
            model_families = [
                name for name, config in model_families_dict.items()
                if config is not None and isinstance(config, dict) and config.get('enabled', False)
            ]
        logger.info(f"Using {len(model_families)} model families from config: {', '.join(model_families)}")
    else:
        # Default fallback
        model_families = ['lightgbm', 'random_forest', 'neural_network']
        logger.info(f"Using default model families: {', '.join(model_families)}")
    
    logger.info("="*80)
    logger.info("Target Predictability Ranking")
    logger.info("="*80)
    logger.info(f"Test symbols: {', '.join(symbols)}")
    logger.info(f"Model families: {', '.join(model_families)}")
    
    # Discover or load targets
    if args.discover_all:
        logger.info("Auto-discovering ALL targets from data...")
        targets_to_eval = discover_all_targets(symbols[0], args.data_dir)
        logger.info(f"Found {len(targets_to_eval)} valid targets\n")
    else:
        # Load target configs
        target_configs = load_target_configs()
        
        # Filter targets
        if args.targets:
            requested = [t.strip() for t in args.targets.split(',')]
            targets_to_eval = {k: v for k, v in target_configs.items() if k in requested}
        else:
            # Only evaluate enabled targets
            targets_to_eval = {k: v for k, v in target_configs.items() if v.get('enabled', False)}
        
        logger.info(f"Evaluating {len(targets_to_eval)} targets\n")
    
    # Initialize checkpoint manager
    checkpoint_file = args.output_dir / "checkpoint.json"
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda item: item if isinstance(item, str) else item[0]  # target
    )
    
    # Clear checkpoint if requested
    if args.clear_checkpoint:
        checkpoint.clear()
        logger.info("Cleared checkpoint - starting fresh")
    
    # Load completed targets
    completed = checkpoint.load_completed()
    logger.info(f"Found {len(completed)} completed targets in checkpoint")
    
    # Evaluate each target
    results = []
    total_targets = len(targets_to_eval)
    completed_count = 0
    skipped_count = 0
    
    for idx, (target, target_config) in enumerate(targets_to_eval.items(), 1):
        # Check if already completed
        if target in completed:
            if args.resume:
                logger.info(f"[{idx}/{total_targets}] Skipping {target} (already completed)")
                result = TargetPredictabilityScore.from_dict(completed[target])
                if result.mean_r2 != -999.0:
                    results.append(result)
                skipped_count += 1
                continue
            elif not args.resume:
                # If not resuming, skip silently
                skipped_count += 1
                continue
        
        # Evaluate target
        logger.info(f"[{idx}/{total_targets}] Evaluating {target}...")
        try:
            result = evaluate_target_predictability(
                target, target_config, symbols, args.data_dir, model_families, multi_model_config,
                output_dir=args.output_dir, min_cs=args.min_cs, max_cs_samples=args.max_cs_samples,
                max_rows_per_symbol=args.max_rows_per_symbol
            )
            
            # Save checkpoint after each target
            checkpoint.save_item(target, result.to_dict())
            
            # Skip degenerate targets (marked with auc = -999)
            if result.auc != -999.0:
                results.append(result)
                completed_count += 1
            else:
                logger.info(f"  Skipped degenerate target: {target}")
        
        except Exception as e:
            logger.error(f"  Failed to evaluate {target}: {e}")
            checkpoint.mark_failed(target, str(e))
            # Continue with next target
    
    logger.info(f"\nCompleted: {completed_count}, Skipped: {skipped_count}, Total: {total_targets}")
    
    # Get all results (including from checkpoint)
    all_results = results
    if args.resume:
        # Merge with checkpoint results
        checkpoint_results = [
            TargetPredictabilityScore.from_dict(v)
            for k, v in completed.items()
            if k not in [r.target for r in results]  # Avoid duplicates
        ]
        all_results = results + checkpoint_results
    
    # Save rankings
    save_rankings(all_results, args.output_dir)
    
    # Analyze stability for all targets/methods (non-invasive hook)
    try:
        from TRAINING.stability.feature_importance import analyze_all_stability_hook
        logger.info("\n" + "="*60)
        logger.info("Feature Importance Stability Analysis")
        logger.info("="*60)
        analyze_all_stability_hook(output_dir=args.output_dir)
    except Exception as e:
        logger.debug(f"Stability analysis failed (non-critical): {e}")
    
    # Print summary
    logger.info("="*80)
    logger.info("TARGET PREDICTABILITY RANKINGS")
    logger.info("="*80)
    
    all_results = sorted(all_results, key=lambda x: x.composite_score, reverse=True)
    
    for i, result in enumerate(all_results, 1):
        leakage_indicator = f" [{result.leakage_flag}]" if result.leakage_flag != "OK" else ""
        logger.info(f"\n{i:2d}. {result.target:25s} | Score: {result.composite_score:.3f}{leakage_indicator}")
        # Use task-appropriate metric name
        if result.task_type == TaskType.REGRESSION:
            metric_name = "R²"
        elif result.task_type == TaskType.BINARY_CLASSIFICATION:
            metric_name = "ROC-AUC"
        else:
            metric_name = "Accuracy"
        logger.info(f"    {metric_name}: {result.auc:.3f} ± {result.std_score:.3f}")
        logger.info(f"    Importance: {result.mean_importance:.2f}")
        logger.info(f"    Recommendation: {_get_recommendation(result)}")
        if result.leakage_flag != "OK":
            logger.info(f"    LEAKAGE FLAG: {result.leakage_flag}")
    
    logger.info("\n" + "="*80)
    logger.info("Target ranking complete!")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"Checkpoint saved to: {checkpoint_file}")
    logger.info("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

