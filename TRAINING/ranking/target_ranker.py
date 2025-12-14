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
from TRAINING.ranking.target_routing import (
    _compute_target_routing_decisions,
    _save_dual_view_rankings
)

# Import auto-rerun wrapper if available
# Note: evaluate_target_with_autofix may not accept view/symbol, so we'll handle that
try:
    from TRAINING.ranking.rank_target_predictability import evaluate_target_with_autofix
    _AUTOFIX_AVAILABLE = True
except ImportError:
    # Fallback: use regular evaluation
    evaluate_target_with_autofix = None
    _AUTOFIX_AVAILABLE = False
from TRAINING.utils.task_types import TargetConfig, TaskType
from TRAINING.utils.leakage_filtering import reload_feature_configs

# Import parallel execution utilities
try:
    from TRAINING.common.parallel_exec import execute_parallel, get_max_workers
    _PARALLEL_AVAILABLE = True
except ImportError:
    _PARALLEL_AVAILABLE = False
    logger.warning("Parallel execution utilities not available; will run sequentially")

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
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    view: str = "CROSS_SECTIONAL",  # "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", or "LOSO"
    symbol: Optional[str] = None  # Required for SYMBOL_SPECIFIC and LOSO views
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
        # First check experiment config if available
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config")
        else:
            # Fallback to pipeline config
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
        explicit_interval=explicit_interval,
        experiment_config=experiment_config,
        view=view,
        symbol=symbol
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
    # Load from config if not provided - check experiment config first
    if min_cs is None:
        # First check experiment config if available (read from YAML data section)
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                    exp_data = exp_yaml.get('data', {})
                    if 'min_cs' in exp_data:
                        min_cs = exp_data['min_cs']
                        logger.debug(f"Using min_cs={min_cs} from experiment config")
            except Exception:
                pass
        
        # Fallback to pipeline config
        if min_cs is None:
            try:
                from CONFIG.config_loader import get_cfg
                min_cs = int(get_cfg("pipeline.data_limits.min_cross_sectional_samples", default=10, config_name="pipeline_config"))
            except Exception:
                min_cs = 10
    
    if max_cs_samples is None:
        # First check experiment config if available (read from YAML data section)
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                    exp_data = exp_yaml.get('data', {})
                    if 'max_cs_samples' in exp_data:
                        max_cs_samples = exp_data['max_cs_samples']
                        logger.debug(f"Using max_cs_samples={max_cs_samples} from experiment config")
            except Exception:
                pass
        
        # Fallback to pipeline config
        if max_cs_samples is None:
            try:
                from CONFIG.config_loader import get_cfg
                max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
            except Exception:
                max_cs_samples = 1000
    
    if max_rows_per_symbol is None:
        # First check experiment config if available
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config")
        else:
            # Fallback to pipeline config
            try:
                from CONFIG.config_loader import get_cfg
                max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
            except Exception:
                max_rows_per_symbol = 50000
    
    # Results storage: separate by view
    results_cs = []  # Cross-sectional results
    results_sym = {}  # Symbol-specific results: {symbol: [results]}
    results_loso = {}  # LOSO results: {symbol: [results]} (optional)
    
    # Load dual-view config (experiment config takes precedence over global config)
    enable_symbol_specific = True  # Default: enable symbol-specific evaluation
    enable_loso = False  # Default: disable LOSO (optional, high value)
    
    # First, try to load from experiment config (per-experiment control)
    exp_target_ranking = {}
    if experiment_config:
        try:
            import yaml
            exp_name = experiment_config.name
            exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
            if exp_file.exists():
                with open(exp_file, 'r') as f:
                    exp_yaml = yaml.safe_load(f) or {}
                exp_target_ranking = exp_yaml.get('target_ranking', {})
                if 'enable_symbol_specific' in exp_target_ranking:
                    enable_symbol_specific = bool(exp_target_ranking['enable_symbol_specific'])
                    logger.debug(f"Using enable_symbol_specific={enable_symbol_specific} from experiment config")
                if 'enable_loso' in exp_target_ranking:
                    enable_loso = bool(exp_target_ranking['enable_loso'])
                    logger.debug(f"Using enable_loso={enable_loso} from experiment config")
        except Exception as e:
            logger.debug(f"Failed to load target_ranking from experiment config: {e}")
    
    # Fallback to global config if not set in experiment config
    if _CONFIG_AVAILABLE:
        try:
            from CONFIG.config_loader import get_cfg
            ranking_cfg = get_cfg("target_ranking", default={}, config_name="target_ranking_config")
            # Only use global config if experiment config didn't set these
            if 'enable_symbol_specific' not in exp_target_ranking:
                enable_symbol_specific = ranking_cfg.get('enable_symbol_specific', enable_symbol_specific)
            if 'enable_loso' not in exp_target_ranking:
                enable_loso = ranking_cfg.get('enable_loso', enable_loso)
        except Exception:
            pass
    
    # Results list for backward compatibility (will contain cross-sectional + aggregated symbol results)
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
    
    # Load parallel execution config
    parallel_targets = False
    if _CONFIG_AVAILABLE:
        try:
            from CONFIG.config_loader import get_cfg
            multi_target_cfg = get_cfg("multi_target", default={}, config_name="target_configs")
            parallel_targets = multi_target_cfg.get('parallel_targets', False)
        except Exception:
            pass
    
    # Check if parallel execution is globally enabled
    parallel_enabled = _PARALLEL_AVAILABLE and parallel_targets
    if parallel_enabled:
        try:
            from CONFIG.config_loader import get_cfg
            parallel_global = get_cfg("threading.parallel.enabled", default=True, config_name="threading_config")
            parallel_enabled = parallel_enabled and parallel_global
        except Exception:
            pass
    
    # Helper function for parallel target evaluation (must be picklable)
    def _evaluate_single_target(item):
        """Evaluate a single target - wrapper for parallel execution"""
        target_name, target_config = item
        result_data = {
            'target_name': target_name,
            'result_cs': None,
            'result_sym_dict': {},
            'result_loso_dict': {},
            'error': None
        }
        
        try:
            # View A: Cross-sectional evaluation (always run)
            if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                try:
                    result_cs = evaluate_target_with_autofix(
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
                        experiment_config=experiment_config,
                        view="CROSS_SECTIONAL",
                        symbol=None
                    )
                except TypeError:
                    result_cs = evaluate_target_with_autofix(
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
                result_cs = evaluate_target_predictability(
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
                    experiment_config=experiment_config,
                    view="CROSS_SECTIONAL",
                    symbol=None
                )
            
            result_data['result_cs'] = result_cs
            
            # View B: Symbol-specific evaluation (if enabled and cross-sectional succeeded)
            skip_statuses = ["LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES", "SUSPICIOUS", "SUSPICIOUS_STRONG"]
            cs_succeeded = result_cs.mean_score != -999.0 and result_cs.status not in skip_statuses
            
            if enable_symbol_specific and cs_succeeded:
                result_sym_dict = {}
                for symbol in symbols:
                    try:
                        if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                            try:
                                result_sym = evaluate_target_with_autofix(
                                    target_name=target_name,
                                    target_config=target_config,
                                    symbols=[symbol],
                                    data_dir=data_dir,
                                    model_families=model_families,
                                    multi_model_config=multi_model_config,
                                    output_dir=output_dir,
                                    min_cs=1,
                                    max_cs_samples=max_cs_samples,
                                    max_rows_per_symbol=max_rows_per_symbol,
                                    max_reruns=max_reruns,
                                    rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                    rerun_on_high_auc_only=rerun_on_high_auc_only,
                                    explicit_interval=explicit_interval,
                                    experiment_config=experiment_config,
                                    view="SYMBOL_SPECIFIC",
                                    symbol=symbol
                                )
                            except TypeError:
                                result_sym = evaluate_target_with_autofix(
                                    target_name=target_name,
                                    target_config=target_config,
                                    symbols=[symbol],
                                    data_dir=data_dir,
                                    model_families=model_families,
                                    multi_model_config=multi_model_config,
                                    output_dir=output_dir,
                                    min_cs=1,
                                    max_cs_samples=max_cs_samples,
                                    max_rows_per_symbol=max_rows_per_symbol,
                                    max_reruns=max_reruns,
                                    rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                    rerun_on_high_auc_only=rerun_on_high_auc_only,
                                    explicit_interval=explicit_interval,
                                    experiment_config=experiment_config
                                )
                        else:
                            result_sym = evaluate_target_predictability(
                                target_name=target_name,
                                target_config=target_config,
                                symbols=[symbol],
                                data_dir=data_dir,
                                model_families=model_families,
                                multi_model_config=multi_model_config,
                                output_dir=output_dir,
                                min_cs=1,
                                max_cs_samples=max_cs_samples,
                                max_rows_per_symbol=max_rows_per_symbol,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                view="SYMBOL_SPECIFIC",
                                symbol=symbol
                            )
                        
                        if result_sym.mean_score != -999.0:
                            result_sym_dict[symbol] = result_sym
                    except Exception as e:
                        logger.warning(f"    Failed to evaluate {target_name} for symbol {symbol}: {e}")
                        continue
                
                result_data['result_sym_dict'] = result_sym_dict
            
            # View C: LOSO evaluation (if enabled)
            if enable_loso:
                result_loso_dict = {}
                for symbol in symbols:
                    try:
                        result_loso_sym = evaluate_target_predictability(
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
                            experiment_config=experiment_config,
                            view="LOSO",
                            symbol=symbol
                        )
                        result_loso_dict[symbol] = result_loso_sym
                    except Exception as e:
                        logger.warning(f"    Failed LOSO evaluation for {target_name} on symbol {symbol}: {e}")
                        continue
                
                result_data['result_loso_dict'] = result_loso_dict
                
        except Exception as e:
            result_data['error'] = str(e)
            logger.exception(f"  Failed to evaluate {target_name}: {e}")
        
        return result_data
    
    # Evaluate targets (parallel or sequential)
    if parallel_enabled and len(targets_to_evaluate) > 1:
        logger.info(f"🚀 Parallel target evaluation enabled ({len(targets_to_evaluate)} targets)")
        parallel_results = execute_parallel(
            _evaluate_single_target,
            targets_to_evaluate.items(),
            max_workers=None,  # Auto-detect from config
            task_type="process",  # CPU-bound
            desc="Evaluating targets",
            show_progress=True
        )
        
        # Process parallel results
        for item, result_data in parallel_results:
            target_name = result_data['target_name']
            if result_data['error']:
                logger.error(f"  ❌ {target_name}: {result_data['error']}")
                continue
            
            result_cs = result_data['result_cs']
            result_sym_dict = result_data.get('result_sym_dict', {})
            result_loso_dict = result_data.get('result_loso_dict', {})
            
            # Process cross-sectional result
            skip_statuses = ["LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES", "SUSPICIOUS", "SUSPICIOUS_STRONG"]
            cs_succeeded = result_cs.mean_score != -999.0 and result_cs.status not in skip_statuses
            if cs_succeeded:
                results_cs.append(result_cs)
                results.append(result_cs)
            else:
                reason = result_cs.status if result_cs.status in skip_statuses else (result_cs.leakage_flag if result_cs.leakage_flag != "OK" else "degenerate/failed")
                if result_cs.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                    logger.warning(f"  ⚠️  Excluded {target_name} CROSS_SECTIONAL ({reason}) - High score suggests structural leakage")
                else:
                    logger.info(f"  Skipped {target_name} CROSS_SECTIONAL ({reason})")
            
            # Store symbol-specific results
            if enable_symbol_specific:
                if target_name not in results_sym:
                    results_sym[target_name] = {}
                for symbol, result_sym in result_sym_dict.items():
                    if result_sym.mean_score != -999.0 and result_sym.status not in skip_statuses:
                        results_sym[target_name][symbol] = result_sym
                    else:
                        reason = result_sym.status if result_sym.status in skip_statuses else "degenerate/failed"
                        logger.debug(f"    Skipped {target_name} SYMBOL_SPECIFIC ({symbol}): {reason}")
            
            # Store LOSO results
            if enable_loso:
                if target_name not in results_loso:
                    results_loso[target_name] = {}
                for symbol, result_loso_sym in result_loso_dict.items():
                    if result_loso_sym.mean_score != -999.0 and result_loso_sym.status not in skip_statuses:
                        results_loso[target_name][symbol] = result_loso_sym
    else:
        # Sequential evaluation (original code path)
        if parallel_enabled and len(targets_to_evaluate) == 1:
            logger.info("Running sequentially (only 1 target)")
        elif not parallel_enabled:
            logger.info("Parallel execution disabled (parallel_targets=false or not available)")
        
        # Evaluate each target in dual views
        for idx, (target_name, target_config) in enumerate(targets_to_evaluate.items(), 1):
            logger.info(f"[{idx}/{total_to_evaluate}] Evaluating {target_name}...")
            
            try:
                # View A: Cross-sectional evaluation (always run)
                logger.info(f"  View A: CROSS_SECTIONAL")
                if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                    # Try with view/symbol, fallback to without if not supported
                    try:
                        result_cs = evaluate_target_with_autofix(
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
                            experiment_config=experiment_config,
                            view="CROSS_SECTIONAL",
                            symbol=None
                        )
                    except TypeError:
                        # Fallback: autofix doesn't support view/symbol yet
                        logger.debug("evaluate_target_with_autofix doesn't support view/symbol, using without")
                        result_cs = evaluate_target_with_autofix(
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
                    result_cs = evaluate_target_predictability(
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
                        experiment_config=experiment_config,
                        view="CROSS_SECTIONAL",
                        symbol=None
                    )
                
                # Store cross-sectional result FIRST (needed for gating symbol-specific)
                skip_statuses = [
                    "LEAKAGE_UNRESOLVED", 
                    "LEAKAGE_UNRESOLVED_MAX_RETRIES",
                    "SUSPICIOUS",
                    "SUSPICIOUS_STRONG"
                ]
                
                cs_succeeded = result_cs.mean_score != -999.0 and result_cs.status not in skip_statuses
                if cs_succeeded:
                    results_cs.append(result_cs)
                    results.append(result_cs)  # Backward compatibility
                else:
                    reason = result_cs.status if result_cs.status in skip_statuses else (result_cs.leakage_flag if result_cs.leakage_flag != "OK" else "degenerate/failed")
                    if result_cs.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                        logger.warning(f"  ⚠️  Excluded {target_name} CROSS_SECTIONAL ({reason}) - High score suggests structural leakage")
                    else:
                        logger.info(f"  Skipped {target_name} CROSS_SECTIONAL ({reason})")
                
                # View B: Symbol-specific evaluation (if enabled)
                result_sym_dict = {}
                if enable_symbol_specific:
                    logger.info(f"  View B: SYMBOL_SPECIFIC (evaluating {len(symbols)} symbols)")
                    
                    # Gate: Only evaluate if cross-sectional succeeded (no point without baseline)
                    if not cs_succeeded:
                        logger.warning(f"  ⚠️  Skipping symbol-specific evaluation for {target_name} (cross-sectional failed: mean_score={result_cs.mean_score}, status={result_cs.status})")
                    else:
                        logger.info(f"  ✅ Cross-sectional succeeded for {target_name}, proceeding with symbol-specific evaluation")
                        for symbol in symbols:
                            logger.info(f"    Evaluating {target_name} for symbol {symbol}...")
                            try:
                                if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                                    # Try with view/symbol, fallback to without if not supported
                                    try:
                                        result_sym = evaluate_target_with_autofix(
                                            target_name=target_name,
                                            target_config=target_config,
                                            symbols=[symbol],  # Single symbol
                                            data_dir=data_dir,
                                            model_families=model_families,
                                            multi_model_config=multi_model_config,
                                            output_dir=output_dir,
                                            min_cs=1,  # Single symbol, min_cs=1
                                            max_cs_samples=max_cs_samples,
                                            max_rows_per_symbol=max_rows_per_symbol,
                                            max_reruns=max_reruns,
                                            rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                            rerun_on_high_auc_only=rerun_on_high_auc_only,
                                            explicit_interval=explicit_interval,
                                            experiment_config=experiment_config,
                                            view="SYMBOL_SPECIFIC",
                                            symbol=symbol
                                        )
                                    except TypeError:
                                        # Fallback: autofix doesn't support view/symbol yet
                                        logger.debug(f"evaluate_target_with_autofix doesn't support view/symbol for {symbol}, using without")
                                        result_sym = evaluate_target_with_autofix(
                                        target_name=target_name,
                                        target_config=target_config,
                                        symbols=[symbol],
                                        data_dir=data_dir,
                                        model_families=model_families,
                                        multi_model_config=multi_model_config,
                                        output_dir=output_dir,
                                        min_cs=1,
                                        max_cs_samples=max_cs_samples,
                                        max_rows_per_symbol=max_rows_per_symbol,
                                        max_reruns=max_reruns,
                                        rerun_on_perfect_train_acc=rerun_on_perfect_train_acc,
                                        rerun_on_high_auc_only=rerun_on_high_auc_only,
                                        explicit_interval=explicit_interval,
                                        experiment_config=experiment_config
                                    )
                                else:
                                    result_sym = evaluate_target_predictability(
                                    target_name=target_name,
                                    target_config=target_config,
                                    symbols=[symbol],  # Single symbol
                                    data_dir=data_dir,
                                    model_families=model_families,
                                    multi_model_config=multi_model_config,
                                    output_dir=output_dir,
                                    min_cs=1,  # Single symbol, min_cs=1
                                    max_cs_samples=max_cs_samples,
                                    max_rows_per_symbol=max_rows_per_symbol,
                                    explicit_interval=explicit_interval,
                                    experiment_config=experiment_config,
                                    view="SYMBOL_SPECIFIC",
                                    symbol=symbol
                                    )
                                
                                # Gate: Skip if result is degenerate (mean_score = -999)
                                if result_sym.mean_score == -999.0:
                                    logger.warning(f"    ⚠️  Skipped {target_name} for symbol {symbol}: degenerate result (mean_score=-999.0, status={result_sym.status})")
                                    continue
                                
                                logger.info(f"    ✅ {target_name} for {symbol}: mean_score={result_sym.mean_score:.4f}, status={result_sym.status}")
                                result_sym_dict[symbol] = result_sym
                            except Exception as e:
                                logger.error(f"    ❌ Failed to evaluate {target_name} for symbol {symbol}: {e}", exc_info=True)
                                continue
                        
                        logger.info(f"  📊 Symbol-specific results for {target_name}: {len(result_sym_dict)}/{len(symbols)} symbols succeeded")
                
                # View C: LOSO evaluation (optional, if enabled)
                result_loso_dict = {}
                if enable_loso:
                    logger.info(f"  View C: LOSO (evaluating {len(symbols)} symbols)")
                    for symbol in symbols:
                        try:
                            result_loso_sym = evaluate_target_predictability(
                                target_name=target_name,
                                target_config=target_config,
                                symbols=symbols,  # All symbols for training
                                data_dir=data_dir,
                                model_families=model_families,
                                multi_model_config=multi_model_config,
                                output_dir=output_dir,
                                min_cs=min_cs,
                                max_cs_samples=max_cs_samples,
                                max_rows_per_symbol=max_rows_per_symbol,
                                explicit_interval=explicit_interval,
                                experiment_config=experiment_config,
                                view="LOSO",
                                symbol=symbol
                            )
                            result_loso_dict[symbol] = result_loso_sym
                        except Exception as e:
                            logger.warning(f"    Failed LOSO evaluation for {target_name} on symbol {symbol}: {e}")
                            continue
                
                # Store symbol-specific results
                if enable_symbol_specific:
                    if target_name not in results_sym:
                        results_sym[target_name] = {}
                    stored_count = 0
                    for symbol, result_sym in result_sym_dict.items():
                        if result_sym.mean_score != -999.0 and result_sym.status not in skip_statuses:
                            results_sym[target_name][symbol] = result_sym
                            stored_count += 1
                        else:
                            reason = result_sym.status if result_sym.status in skip_statuses else "degenerate/failed"
                            logger.warning(f"    ⚠️  Filtered out {target_name} SYMBOL_SPECIFIC ({symbol}): {reason} (mean_score={result_sym.mean_score})")
                    if stored_count > 0:
                        logger.info(f"  ✅ Stored {stored_count} symbol-specific results for {target_name}")
                    elif len(result_sym_dict) > 0:
                        logger.warning(f"  ⚠️  All {len(result_sym_dict)} symbol-specific results for {target_name} were filtered out")
                
                # Store LOSO results
                if enable_loso:
                    if target_name not in results_loso:
                        results_loso[target_name] = {}
                    for symbol, result_loso_sym in result_loso_dict.items():
                        if result_loso_sym.mean_score != -999.0 and result_loso_sym.status not in skip_statuses:
                            results_loso[target_name][symbol] = result_loso_sym
            
            except Exception as e:
                logger.exception(f"  Failed to evaluate {target_name}: {e}")  # Better error logging with traceback
                # Continue with next target
    
    # Compute routing decisions and aggregate symbol-specific results
    logger.info("=" * 60)
    logger.info("DUAL-VIEW TARGET RANKING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Cross-sectional targets evaluated: {len(results_cs)}")
    if enable_symbol_specific:
        total_sym_results = sum(len(sym_results) for sym_results in results_sym.values())
        logger.info(f"Symbol-specific evaluations: {total_sym_results} (across {len(results_sym)} targets)")
    
    # Compute routing decisions for each target
    routing_decisions = _compute_target_routing_decisions(
        results_cs=results_cs,
        results_sym=results_sym,
        results_loso=results_loso if enable_loso else {}
    )
    
    # Log routing summary
    cs_only = sum(1 for r in routing_decisions.values() if r.get('route') == 'CROSS_SECTIONAL')
    sym_only = sum(1 for r in routing_decisions.values() if r.get('route') == 'SYMBOL_SPECIFIC')
    both = sum(1 for r in routing_decisions.values() if r.get('route') == 'BOTH')
    blocked = sum(1 for r in routing_decisions.values() if r.get('route') == 'BLOCKED')
    logger.info(f"Routing decisions: {cs_only} CROSS_SECTIONAL, {sym_only} SYMBOL_SPECIFIC, {both} BOTH, {blocked} BLOCKED")
    
    # Sort cross-sectional results by composite score (descending) for backward compatibility
    results.sort(key=lambda r: r.composite_score, reverse=True)
    
    # Apply top_n limit if specified (to cross-sectional results)
    if top_n is not None and top_n > 0:
        results = results[:top_n]
        logger.info(f"Returning top {len(results)} targets (cross-sectional)")
    
    # Save rankings if output_dir provided
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_rankings(results, output_dir)
        
        # Save dual-view results and routing decisions
        _save_dual_view_rankings(
            results_cs=results_cs,
            results_sym=results_sym,
            results_loso=results_loso if enable_loso else {},
            routing_decisions=routing_decisions,
            output_dir=output_dir
        )
    
    return results

