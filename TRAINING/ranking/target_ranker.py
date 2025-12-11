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
Target Ranking Module

Extracted from SCRIPTS/rank_target_predictability.py to enable integration
into the training pipeline. All leakage-free behavior is preserved by
reusing the original functions.
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings

# Add project root to path for imports
# TRAINING/ranking/target_ranker.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Import original functions to preserve leakage-free behavior
from TRAINING.ranking.rank_target_predictability import (
    TargetPredictabilityScore,
    evaluate_target_predictability as _evaluate_target_predictability,
    discover_all_targets as _discover_all_targets,
    load_target_configs as _load_target_configs,
    save_rankings as _save_rankings
)

# Import auto-rerun wrapper if available
try:
    from TRAINING.ranking.rank_target_predictability import evaluate_target_with_autofix
except ImportError:
    # Fallback: use regular evaluation
    evaluate_target_with_autofix = _evaluate_target_predictability
from TRAINING.utils.task_types import TargetConfig, TaskType
from TRAINING.utils.leakage_filtering import reload_feature_configs

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass

# Import new config system (optional - for backward compatibility)
try:
    from CONFIG.config_builder import build_target_ranking_config
    from CONFIG.config_schemas import ExperimentConfig, TargetRankingConfig
    _NEW_CONFIG_AVAILABLE = True
except ImportError:
    _NEW_CONFIG_AVAILABLE = False
    # Logger not yet initialized, will be set up below
    pass

# Suppress expected warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

logger = logging.getLogger(__name__)


def evaluate_target_predictability(
    target_name: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: Optional[int] = None,  # Load from config if None
    max_cs_samples: Optional[int] = None,  # Load from config if None
    max_rows_per_symbol: Optional[int] = None,  # Load from config if None
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config
    experiment_config: Optional[Any] = None  # Optional ExperimentConfig (for data.bar_interval)
) -> TargetPredictabilityScore:
    """
    Evaluate predictability of a single target across symbols.
    
    This is a wrapper around the original function to preserve all
    leakage-free behavior (PurgedTimeSeriesSplit, leakage filtering, etc.).
    
    Args:
        target_name: Display name of target
        target_config: TargetConfig object or dict with target config
        symbols: List of symbols to evaluate on
        data_dir: Directory containing symbol data
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
        output_dir: Optional output directory for results
        min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
        max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
        max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)
    
    Returns:
        TargetPredictabilityScore object with predictability metrics
    """
    # Load from config if not provided
    if min_cs is None:
        try:
            from CONFIG.config_loader import get_cfg
            min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
        except Exception:
            min_cs = 10
    
    if max_cs_samples is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
        except Exception:
            max_cs_samples = 1000
    
    if max_rows_per_symbol is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
        except Exception:
            max_rows_per_symbol = 50000
    
    return _evaluate_target_predictability(
        target_name=target_name,
        target_config=target_config,
        symbols=symbols,
        data_dir=data_dir,
        model_families=model_families,
        multi_model_config=multi_model_config,
        output_dir=output_dir,
        min_cs=min_cs,
        max_cs_samples=max_cs_samples,
        max_rows_per_symbol=max_rows_per_symbol,
        explicit_interval=None,  # Will be passed from rank_targets
        experiment_config=None  # Will be passed from rank_targets
    )


def discover_targets(
    symbol: str,
    data_dir: Path
) -> Dict[str, TargetConfig]:
    """
    Auto-discover all valid targets from data (non-degenerate).
    
    Preserves all leakage-free filtering (excludes first_touch targets, etc.).
    
    Args:
        symbol: Symbol to use for discovery
        data_dir: Directory containing symbol data
    
    Returns:
        Dict mapping target_name -> TargetConfig
    """
    return _discover_all_targets(symbol, data_dir)


def load_target_configs() -> Dict[str, Dict]:
    """
    Load target configurations from CONFIG/target_configs.yaml.
    
    Returns:
        Dict mapping target_name -> target config dict
    """
    return _load_target_configs()


def rank_targets(
    targets: Dict[str, TargetConfig | Dict[str, Any]],
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: Optional[int] = None,  # Load from config if None
    max_cs_samples: Optional[int] = None,  # Load from config if None
    max_rows_per_symbol: Optional[int] = None,  # Load from config if None
    top_n: Optional[int] = None,
    max_targets_to_evaluate: Optional[int] = None,  # Limit number of targets to evaluate (for faster testing)
    target_ranking_config: Optional['TargetRankingConfig'] = None,  # New typed config (optional)
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (e.g., "5m")
    experiment_config: Optional[Any] = None  # Optional ExperimentConfig (for data.bar_interval)
) -> List[TargetPredictabilityScore]:
    """
    Rank multiple targets by predictability.
    
    This function evaluates all targets and returns them sorted by
    composite predictability score. All leakage-free behavior is preserved.
    
    Args:
        targets: Dict mapping target_name -> TargetConfig or config dict
        symbols: List of symbols to evaluate on
        data_dir: Directory containing symbol data
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict [LEGACY]
        output_dir: Optional output directory for results
        min_cs: Minimum cross-sectional size per timestamp (loads from config if None)
        max_cs_samples: Maximum samples per timestamp for cross-sectional sampling (loads from config if None)
        max_rows_per_symbol: Maximum rows to load per symbol (loads from config if None)
        top_n: Optional limit on number of top targets to return (after ranking)
        max_targets_to_evaluate: Optional limit on number of targets to evaluate (for faster testing)
        target_ranking_config: Optional TargetRankingConfig object [NEW - preferred]
    
    Returns:
        List of TargetPredictabilityScore objects, sorted by composite_score (descending)
    """
    # Load from config if not provided
    if min_cs is None:
        try:
            from CONFIG.config_loader import get_cfg
            min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
        except Exception:
            min_cs = 10
    
    if max_cs_samples is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
        except Exception:
            max_cs_samples = 1000
    
    if max_rows_per_symbol is None:
        try:
            from CONFIG.config_loader import get_cfg
            max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
        except Exception:
            max_rows_per_symbol = 50000
    
    results = []
    
    # NEW: Use typed config if provided
    if target_ranking_config is not None and _NEW_CONFIG_AVAILABLE:
        # Extract values from typed config
        if target_ranking_config.model_families:
            # Convert to list of enabled family names
            model_families = [
                name for name, cfg in target_ranking_config.model_families.items()
                if cfg.get('enabled', False)
            ]
        if target_ranking_config.data_dir:
            data_dir = target_ranking_config.data_dir
        if target_ranking_config.symbols:
            symbols = target_ranking_config.symbols
        if target_ranking_config.max_samples_per_symbol:
            max_rows_per_symbol = target_ranking_config.max_samples_per_symbol
        # Build multi_model_config dict from typed config for backward compat
        multi_model_config = {
            'model_families': target_ranking_config.model_families,
            'cross_validation': target_ranking_config.cross_validation,
            'sampling': target_ranking_config.sampling,
            'ranking': target_ranking_config.ranking
        }
    
    # Limit targets to evaluate if specified (for faster testing)
    all_targets_count = len(targets)
    targets_to_evaluate = targets
    if max_targets_to_evaluate is not None and max_targets_to_evaluate > 0:
        # Take first N targets (they're already in a reasonable order from discovery)
        target_items = list(targets.items())[:max_targets_to_evaluate]
        targets_to_evaluate = dict(target_items)
        logger.info(f"Limiting evaluation to {len(targets_to_evaluate)} targets (out of {all_targets_count} total) for faster testing")
    
    total_to_evaluate = len(targets_to_evaluate)
    logger.info(f"Ranking {total_to_evaluate} targets across {len(symbols)} symbols")
    logger.info(f"Model families: {', '.join(model_families)}")
    
    # Load auto-rerun config
    auto_rerun_enabled = False
    max_reruns = 3
    rerun_on_perfect_train_acc = True
    rerun_on_high_auc_only = False
    
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            auto_rerun_cfg = leakage_cfg.get('auto_rerun', {})
            auto_rerun_enabled = auto_rerun_cfg.get('enabled', False)
            max_reruns = int(auto_rerun_cfg.get('max_reruns', 3))
            rerun_on_perfect_train_acc = auto_rerun_cfg.get('rerun_on_perfect_train_acc', True)
            rerun_on_high_auc_only = auto_rerun_cfg.get('rerun_on_high_auc_only', False)
        except Exception:
            pass
    
    # Evaluate each target (use targets_to_evaluate, not original targets dict)
    for idx, (target_name, target_config) in enumerate(targets_to_evaluate.items(), 1):
        logger.info(f"[{idx}/{total_to_evaluate}] Evaluating {target_name}...")
        
        try:
            # Use auto-rerun wrapper if enabled, otherwise use regular evaluation
            if auto_rerun_enabled:
                result = evaluate_target_with_autofix(
                    target_name=target_name,
                    target_config=target_config,
                    symbols=symbols,
                    data_dir=data_dir,
                    model_families=model_families,
                    multi_model_config=multi_model_config,
                    output_dir=output_dir,
                    min_cs=min_cs,
                    max_cs_samples=max_cs_samples,
                    max_rows_per_symbol=max_rows_per_symbol,
                    max_reruns=max_reruns,
                    rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                    rerun_on_high_auc_only=rerun_on_high_auc_only,
                    explicit_interval=explicit_interval,
                    experiment_config=experiment_config
                )
            else:
                result = evaluate_target_predictability(
                    target_name=target_name,
                    target_config=target_config,
                    symbols=symbols,
                    data_dir=data_dir,
                    model_families=model_families,
                    multi_model_config=multi_model_config,
                    output_dir=output_dir,
                    min_cs=min_cs,
                    max_cs_samples=max_cs_samples,
                    max_rows_per_symbol=max_rows_per_symbol,
                    explicit_interval=explicit_interval,
                    experiment_config=experiment_config
                )
            
            # Skip degenerate/failed targets (marked with mean_score = -999)
            # Also skip targets with unresolved leakage or suspicious scores
            # SUSPICIOUS/SUSPICIOUS_STRONG targets are excluded - they likely have structural leakage
            skip_statuses = [
                "LEAKAGE_UNRESOLVED", 
                "LEAKAGE_UNRESOLVED_MAX_RETRIES",
                "SUSPICIOUS",
                "SUSPICIOUS_STRONG"
            ]
            
            if result.mean_score != -999.0 and result.status not in skip_statuses:
                results.append(result)
            else:
                # Provide more specific reason for skipping
                if result.status in skip_statuses:
                    reason = result.status
                    if result.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                        logger.warning(
                            f"  ⚠️  Excluded {target_name} ({reason}) - "
                            f"High score ({result.mean_score:.3f}) suggests structural leakage. "
                            f"Review target construction and label logic."
                        )
                    else:
                        logger.info(f"  Skipped {target_name} ({reason})")
                else:
                    reason = result.leakage_flag if result.leakage_flag != "OK" else "degenerate/failed"
                    logger.info(f"  Skipped {target_name} ({reason})")
        
        except Exception as e:
            logger.error(f"  Failed to evaluate {target_name}: {e}")
            # Continue with next target
    
    # Sort by composite score (descending)
    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    # Apply top_n limit if specified
    if top_n is not None and top_n > 0:
        results = results[:top_n]
        logger.info(f"Returning top {len(results)} targets")
    
    # Save rankings if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_rankings(results, output_dir)
    
    return results

