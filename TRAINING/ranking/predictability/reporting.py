"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Logger not yet initialized, will be set up below

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



# Import dependencies
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore

def save_leak_report_summary(
    output_dir: Path,
    all_leaks: Dict[str, Dict[str, List[Tuple[str, float]]]]
) -> None:
    """
    Save a summary of all detected leaks across all targets.
    
    Args:
        output_dir: Directory to save the report
        all_leaks: Dict of {target_name: {model_name: [(feature, importance), ...]}}
    """
    report_file = output_dir / "leak_detection_summary.txt"
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("LEAK DETECTION SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write("This report lists features with suspiciously high importance (>50%)\n")
        f.write("which may indicate data leakage (future information in features).\n\n")
        
        total_leaks = sum(len(leaks) for leaks in all_leaks.values())
        f.write(f"Total targets with suspicious features: {len(all_leaks)}\n")
        f.write(f"Total suspicious feature detections: {total_leaks}\n\n")
        
        for target_name, model_leaks in sorted(all_leaks.items()):
            f.write(f"\n{'='*80}\n")
            f.write(f"Target: {target_name}\n")
            f.write(f"{'='*80}\n")
            
            for model_name, features in model_leaks.items():
                if features:
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"{'-'*80}\n")
                    for feat, imp in sorted(features, key=lambda x: x[1], reverse=True):
                        f.write(f"  {feat:60s} | {imp:.1%}\n")
        
        f.write(f"\n\n{'='*80}\n")
        f.write("RECOMMENDATIONS:\n")
        f.write(f"{'='*80}\n")
        f.write("1. Review features with >50% importance - they likely contain future information\n")
        f.write("2. Check for:\n")
        f.write("   - Centered moving averages (center=True)\n")
        f.write("   - Backward shifts (.shift(-1) instead of .shift(1))\n")
        f.write("   - High/Low data that matches target definition\n")
        f.write("   - Features computed from the same barrier logic as the target\n")
        f.write("3. Add suspicious features to leakage_filtering.py exclusion list\n")
        f.write("4. Re-run ranking after fixing leaks\n")
    
    logger.info(f"Leak detection summary saved to: {report_file}")


def save_rankings(
    results: List[TargetPredictabilityScore],
    output_dir: Path
):
    """
    Save target predictability rankings.
    
    New structure:
    - CSV (reproducibility artifact) → REPRODUCIBILITY/TARGET_RANKING/target_predictability_rankings.csv
    - YAML (decision log) → DECISION/TARGET_RANKING/target_prioritization.yaml
    
    Args:
        results: List of TargetPredictabilityScore objects
        output_dir: Base output directory (RESULTS/{run}/), not target_rankings subdirectory
    """
    # Determine base output directory (handle both old and new call patterns)
    if output_dir.name == "target_rankings":
        base_output_dir = output_dir.parent
    else:
        base_output_dir = output_dir
    
    # Target-first structure: target prioritization is global (ranking of all targets)
    from TRAINING.orchestration.utils.target_first_paths import (
        get_globals_dir, get_target_decision_dir, ensure_target_structure
    )
    globals_dir = get_globals_dir(base_output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    
    # Target-first structure only - no legacy directories needed
    
    # Handle empty results
    if not results:
        logger.warning("No valid targets to rank - all targets were skipped (insufficient features, degenerate, or failed)")
        # Create empty CSV file with headers
        empty_df = pd.DataFrame(columns=[
            'rank', 'target_name', 'target_column', 'composite_score', 'task_type',
            'mean_score', 'std_score', 'mean_r2', 'std_r2', 'mean_importance',
            'consistency', 'n_models', 'leakage_flag', 'recommendation'
        ])
        empty_df.to_csv(repro_dir / "target_predictability_rankings.csv", index=False)
        logger.info(f"Saved empty rankings file to {repro_dir / 'target_predictability_rankings.csv'}")
        return
    
    # Sort by composite score
    results = sorted(results, key=lambda x: x.composite_score, reverse=True)
    
    # Create DataFrame
    df = pd.DataFrame([{
        'rank': i + 1,
        'target_name': r.target_name,
        'target_column': r.target_column,
        'composite_score': r.composite_score,
        'task_type': r.task_type.name,
        'mean_score': r.mean_score,
        'std_score': r.std_score,
        'mean_r2': r.mean_score,  # Backward compatibility
        'std_r2': r.std_score,  # Backward compatibility
        'mean_importance': r.mean_importance,
        'consistency': r.consistency,
        'n_models': r.n_models,
        'leakage_flag': r.leakage_flag,
        **{f'{model}_r2': score for model, score in r.model_scores.items()},
        'recommendation': _get_recommendation(r)
    } for i, r in enumerate(results)])
    
    # Log suspicious targets
    suspicious = df[df['leakage_flag'] != 'OK']
    if len(suspicious) > 0:
        logger.warning(f"\nFOUND {len(suspicious)} SUSPICIOUS TARGETS (possible leakage):")
        for _, row in suspicious.iterrows():
            logger.warning(
                f"  {row['target_name']:25s} | R²={row['mean_r2']:.3f} | "
                f"Composite={row['composite_score']:.3f} | Flag: {row['leakage_flag']}"
            )
        logger.warning("Review these targets - they may have leaked features or be degenerate!")
    
    # Save CSV to target-first structure (globals/ for global ranking) only
    try:
        target_csv_path = globals_dir / "target_predictability_rankings.csv"
        df.to_csv(target_csv_path, index=False)
        logger.info(f"✅ Saved rankings CSV to {target_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to save rankings CSV to target-first location: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Save YAML with recommendations to DECISION (decision log)
    yaml_data = {
        'target_rankings': [
            {
            'rank': i + 1,
            'target': r.target_name,
            'composite_score': float(r.composite_score),
            'task_type': r.task_type.name,
            'mean_score': float(r.mean_score),
            'mean_r2': float(r.mean_score),  # Backward compatibility
            'leakage_flag': r.leakage_flag,
            'recommendation': _get_recommendation(r)
            }
            for i, r in enumerate(results)
        ]
    }
    
    # Save to globals/ (target-first primary location - this is a global ranking)
    globals_yaml_path = globals_dir / "target_prioritization.yaml"
    with open(globals_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    logger.info(f"Saved target prioritization YAML to {globals_yaml_path}")
    
    # Also save per-target slices for fast local inspection
    for i, r in enumerate(results):
        target_name_clean = r.target_name.replace('/', '_').replace('\\', '_')
        try:
            ensure_target_structure(base_output_dir, target_name_clean)
            target_decision_dir = get_target_decision_dir(base_output_dir, target_name_clean)
            target_yaml_path = target_decision_dir / "target_prioritization.yaml"
            # Save per-target slice with rank and recommendation
            target_yaml_data = {
                'rank': i + 1,
                'target': r.target_name,
                'composite_score': float(r.composite_score),
                'task_type': r.task_type.name,
                'mean_score': float(r.mean_score),
                'leakage_flag': r.leakage_flag,
                'recommendation': _get_recommendation(r)
            }
            with open(target_yaml_path, 'w') as f:
                yaml.dump(target_yaml_data, f, default_flow_style=False)
            logger.debug(f"Saved per-target prioritization to {target_yaml_path}")
        except Exception as e:
            logger.debug(f"Failed to save per-target prioritization for {target_name_clean}: {e}")
    
    # Target-first structure only - no legacy writes


def _get_recommendation(score: TargetPredictabilityScore) -> str:
    """Get recommendation based on predictability score"""
    if score.composite_score >= 0.7:
        return "PRIORITIZE - Strong predictive signal"
    elif score.composite_score >= 0.5:
        return "ENABLE - Good predictive signal"
    elif score.composite_score >= 0.3:
        return "TEST - Moderate signal, worth exploring"
    else:
        return "DEPRIORITIZE - Weak signal, low ROI"


