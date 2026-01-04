# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
from TRAINING.common.utils.task_types import TargetConfig, TaskType
from TRAINING.ranking.utils.leakage_filtering import reload_feature_configs

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
    from config_loader import get_safety_config, get_experiment_config_path, load_experiment_config
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
    target: str,
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
    symbol: Optional[str] = None,  # Required for SYMBOL_SPECIFIC and LOSO views
    scope_purpose: str = "ROUTING_EVAL",  # Default to ROUTING_EVAL for target ranking
    run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object for authoritative signatures
) -> TargetPredictabilityScore:
    """
    Evaluate predictability of a single target across symbols.
    
    This is a wrapper around the original function to preserve all
    leakage-free behavior (PurgedTimeSeriesSplit, leakage filtering, etc.).
    
    Args:
        target: Display name of target
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
        # First check experiment config if available (same pattern as model_evaluation.py)
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config attribute")
        else:
            # Try reading from experiment config YAML directly (same as model_evaluation.py)
            if experiment_config:
                try:
                    import yaml
                    exp_name = experiment_config.name
                    if _CONFIG_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        import yaml
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            with open(exp_file, 'r') as f:
                                exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
                        exp_data = exp_yaml.get('data', {})
                        if 'max_samples_per_symbol' in exp_data:
                            max_rows_per_symbol = exp_data['max_samples_per_symbol']
                            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config YAML")
                except Exception:
                    pass
            
            # Fallback to pipeline config
            if max_rows_per_symbol is None:
                try:
                    from CONFIG.config_loader import get_cfg
                    max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
                except Exception:
                    max_rows_per_symbol = 50000
    
    return _evaluate_target_predictability(
        target=target,
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
        symbol=symbol,
        scope_purpose=scope_purpose,
        run_identity=run_identity,  # NEW: Pass RunIdentity SST object
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
        Dict mapping target -> TargetConfig
    """
    return _discover_all_targets(symbol, data_dir)


def load_target_configs() -> Dict[str, Dict]:
    """
    Load target configurations from CONFIG/target_configs.yaml.
    
    Returns:
        Dict mapping target -> target config dict
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
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object for authoritative signatures
) -> List[TargetPredictabilityScore]:
    """
    Rank multiple targets by predictability.
    
    This function evaluates all targets and returns them sorted by
    composite predictability score. All leakage-free behavior is preserved.
    
    Args:
        targets: Dict mapping target -> TargetConfig or config dict
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
        # First check experiment config if available (same pattern as model_evaluation.py)
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config attribute")
        else:
            # Try reading from experiment config YAML directly (same as model_evaluation.py)
            if experiment_config:
                try:
                    import yaml
                    exp_name = experiment_config.name
                    if _CONFIG_AVAILABLE:
                        exp_yaml = load_experiment_config(exp_name)
                    else:
                        import yaml
                        exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                        if exp_file.exists():
                            with open(exp_file, 'r') as f:
                                exp_yaml = yaml.safe_load(f) or {}
                        else:
                            exp_yaml = {}
                        exp_data = exp_yaml.get('data', {})
                        if 'max_samples_per_symbol' in exp_data:
                            max_rows_per_symbol = exp_data['max_samples_per_symbol']
                            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config YAML")
                except Exception:
                    pass
            
            # Fallback to pipeline config
            if max_rows_per_symbol is None:
                try:
                    from CONFIG.config_loader import get_cfg
                    max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
                except Exception:
                    max_rows_per_symbol = 50000
    
    # Results storage: separate by view
    results_cs = []  # Cross-sectional results
    results_sym = {}  # Symbol-specific results: {target: {symbol: result}}
    symbol_skip_reasons = {}  # Skip reasons: {target: {symbol: {reason, status, ...}}}
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
    
    # Track all evaluated targets (for ensuring decision files are created)
    all_evaluated_targets = set()
    # Track all CS results (including failed ones) for decision computation
    all_cs_results = {}  # {target: TargetPredictabilityScore}
    
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
        # Sort targets alphabetically for consistent ordering
        sorted_target_items = sorted(targets.items(), key=lambda x: x[0])
        target_items = sorted_target_items[:max_targets_to_evaluate]
        targets_to_evaluate = dict(target_items)
        logger.info(f"Limiting evaluation to {len(targets_to_evaluate)} targets (out of {all_targets_count} total) for faster testing")
        selected_targets = list(targets_to_evaluate.keys())
        if len(selected_targets) <= 10:
            logger.debug(f"Selected targets: {selected_targets}")
        else:
            logger.debug(f"Selected targets: {selected_targets[:10]}... (showing first 10 of {len(selected_targets)})")
    
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
        target, target_config = item
        result_data = {
            'target': target,
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
                        target=target,
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
                        target=target,
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
                    target=target,
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
                    symbol=None,
                    run_identity=run_identity,  # NEW: Pass RunIdentity
                )
            
            result_data['result_cs'] = result_cs
            
            # View B: Symbol-specific evaluation (if enabled and cross-sectional succeeded)
            skip_statuses = ["LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES", "SUSPICIOUS", "SUSPICIOUS_STRONG"]
            cs_succeeded = result_cs.auc != -999.0 and result_cs.status not in skip_statuses
            
            if enable_symbol_specific and cs_succeeded:
                result_sym_dict = {}
                for symbol in symbols:
                    try:
                        if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                            try:
                                result_sym = evaluate_target_with_autofix(
                                    target=target,
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
                                    target=target,
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
                                target=target,
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
                                symbol=symbol,
                                run_identity=run_identity,  # NEW: Pass RunIdentity
                            )
                        
                        if result_sym.auc != -999.0:
                            result_sym_dict[symbol] = result_sym
                    except Exception as e:
                        logger.warning(f"    Failed to evaluate {target} for symbol {symbol}: {e}")
                        continue
                
                result_data['result_sym_dict'] = result_sym_dict
            
            # View C: LOSO evaluation (if enabled)
            if enable_loso:
                result_loso_dict = {}
                for symbol in symbols:
                    try:
                        result_loso_sym = evaluate_target_predictability(
                            target=target,
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
                            symbol=symbol,
                            run_identity=run_identity,  # NEW: Pass RunIdentity
                        )
                        result_loso_dict[symbol] = result_loso_sym
                    except Exception as e:
                        logger.warning(f"    Failed LOSO evaluation for {target} on symbol {symbol}: {e}")
                        continue
                
                result_data['result_loso_dict'] = result_loso_dict
                
        except Exception as e:
            result_data['error'] = str(e)
            logger.exception(f"  Failed to evaluate {target}: {e}")
        
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
            target = result_data['target']
            all_evaluated_targets.add(target)  # Track that this target was evaluated
            if result_data['error']:
                logger.error(f"  ❌ {target}: {result_data['error']}")
                continue
            
            result_cs = result_data['result_cs']
            result_sym_dict = result_data.get('result_sym_dict', {})
            result_loso_dict = result_data.get('result_loso_dict', {})
            
            # Track CS result (even if failed) for decision computation
            if result_cs:
                all_cs_results[target] = result_cs
            
            # Process cross-sectional result
            skip_statuses = ["LEAKAGE_UNRESOLVED", "LEAKAGE_UNRESOLVED_MAX_RETRIES", "SUSPICIOUS", "SUSPICIOUS_STRONG"]
            cs_succeeded = result_cs.auc != -999.0 and result_cs.status not in skip_statuses
            if cs_succeeded:
                results_cs.append(result_cs)
                results.append(result_cs)
            else:
                reason = result_cs.status if result_cs.status in skip_statuses else (result_cs.leakage_flag if result_cs.leakage_flag != "OK" else "degenerate/failed")
                if result_cs.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                    logger.warning(f"  ⚠️  Excluded {target} CROSS_SECTIONAL ({reason}) - High score suggests structural leakage")
                else:
                    logger.info(f"  Skipped {target} CROSS_SECTIONAL ({reason})")
            
            # Store symbol-specific results
            if enable_symbol_specific:
                if target not in results_sym:
                    results_sym[target] = {}
                for symbol, result_sym in result_sym_dict.items():
                    if result_sym.auc != -999.0 and result_sym.status not in skip_statuses:
                        results_sym[target][symbol] = result_sym
                    else:
                        reason = result_sym.status if result_sym.status in skip_statuses else "degenerate/failed"
                        logger.debug(f"    Skipped {target} SYMBOL_SPECIFIC ({symbol}): {reason}")
            
            # Store LOSO results
            if enable_loso:
                if target not in results_loso:
                    results_loso[target] = {}
                for symbol, result_loso_sym in result_loso_dict.items():
                    if result_loso_sym.auc != -999.0 and result_loso_sym.status not in skip_statuses:
                        results_loso[target][symbol] = result_loso_sym
            
            # Save routing decision immediately after target evaluation (incremental)
            if output_dir:
                from TRAINING.ranking.target_routing import (
                    _compute_single_target_routing_decision, _save_single_target_decision
                )
                target_sym_results = results_sym.get(target, {})
                target_skip_reasons = symbol_skip_reasons.get(target, {}) if symbol_skip_reasons else {}
                decision = _compute_single_target_routing_decision(
                    target=target,
                    result_cs=result_cs if cs_succeeded else None,
                    sym_results=target_sym_results,
                    symbol_skip_reasons=target_skip_reasons
                )
                _save_single_target_decision(target, decision, output_dir)
    else:
        # Sequential evaluation (original code path)
        if parallel_enabled and len(targets_to_evaluate) == 1:
            logger.info("Running sequentially (only 1 target)")
        elif not parallel_enabled:
            logger.info("Parallel execution disabled (parallel_targets=false or not available)")
        
        # Evaluate each target in dual views
        for idx, (target, target_config) in enumerate(targets_to_evaluate.items(), 1):
            all_evaluated_targets.add(target)  # Track that this target was evaluated
            logger.info(f"[{idx}/{total_to_evaluate}] Evaluating {target}...")
            
            try:
                # View A: Cross-sectional evaluation (always run)
                logger.info(f"  View A: CROSS_SECTIONAL")
                if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                    # Try with view/symbol, fallback to without if not supported
                    try:
                        result_cs = evaluate_target_with_autofix(
                            target=target,
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
                            target=target,
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
                        target=target,
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
                        symbol=None,
                        run_identity=run_identity,  # NEW: Pass RunIdentity
                    )
                
                # Store cross-sectional result FIRST (needed for gating symbol-specific)
                skip_statuses = [
                    "LEAKAGE_UNRESOLVED", 
                    "LEAKAGE_UNRESOLVED_MAX_RETRIES",
                    "SUSPICIOUS",
                    "SUSPICIOUS_STRONG"
                ]
                
                # Track CS result (even if failed) for decision computation
                if result_cs:
                    all_cs_results[target] = result_cs
                
                cs_succeeded = result_cs.auc != -999.0 and result_cs.status not in skip_statuses
                if cs_succeeded:
                    results_cs.append(result_cs)
                    results.append(result_cs)  # Backward compatibility
                else:
                    reason = result_cs.status if result_cs.status in skip_statuses else (result_cs.leakage_flag if result_cs.leakage_flag != "OK" else "degenerate/failed")
                    if result_cs.status in ["SUSPICIOUS", "SUSPICIOUS_STRONG"]:
                        logger.warning(f"  ⚠️  Excluded {target} CROSS_SECTIONAL ({reason}) - High score suggests structural leakage")
                    else:
                        logger.info(f"  Skipped {target} CROSS_SECTIONAL ({reason})")
                
                # View B: Symbol-specific evaluation (if enabled)
                result_sym_dict = {}
                if enable_symbol_specific:
                    logger.info(f"  View B: SYMBOL_SPECIFIC (evaluating {len(symbols)} symbols)")
                    
                    # Always evaluate symbol-specific, even if cross-sectional failed
                    # Some targets may work symbol-specifically even if cross-sectional fails
                    # Routing logic needs symbol-specific results to make decisions (e.g., "SYMBOL_SPECIFIC only: weak CS but some symbols work")
                    if not cs_succeeded:
                        logger.info(f"  ℹ️  Cross-sectional failed for {target} (auc={result_cs.auc}, status={result_cs.status}), but evaluating symbol-specific anyway (target may work per-symbol)")
                    else:
                        logger.info(f"  ✅ Cross-sectional succeeded for {target}, proceeding with symbol-specific evaluation")
                    
                    # Track per-symbol skip reasons and diagnostics (local to this target evaluation)
                    local_symbol_skip_reasons = {}  # {symbol: {reason, n_rows, n_train, n_val, n_pos_train, n_neg_train, n_pos_val, n_neg_val}}
                    
                    for symbol in symbols:
                        logger.info(f"    Evaluating {target} for symbol {symbol}...")
                        try:
                            if auto_rerun_enabled and _AUTOFIX_AVAILABLE:
                                # Try with view/symbol, fallback to without if not supported
                                try:
                                    result_sym = evaluate_target_with_autofix(
                                        target=target,
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
                                        target=target,
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
                                    target=target,
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
                                    symbol=symbol,
                                    run_identity=run_identity,  # NEW: Pass RunIdentity
                                )
                            
                            # Gate: Skip if result is degenerate (auc = -999)
                            if result_sym.auc == -999.0:
                                skip_reason = result_sym.status if result_sym.status != "OK" else "degenerate"
                                logger.warning(f"    ⚠️  Skipped {target} for symbol {symbol}: {skip_reason} (auc=-999.0, status={result_sym.status})")
                                local_symbol_skip_reasons[symbol] = {
                                    'reason': skip_reason,
                                    'status': result_sym.status,
                                    'leakage_flag': result_sym.leakage_flag,
                                    'auc': result_sym.auc
                                }
                                continue
                            
                            logger.info(f"    ✅ {target} for {symbol}: auc={result_sym.auc:.4f}, status={result_sym.status}")
                            result_sym_dict[symbol] = result_sym
                        except Exception as e:
                            skip_reason = f"exception: {type(e).__name__}"
                            logger.error(f"    ❌ Failed to evaluate {target} for symbol {symbol}: {e}", exc_info=True)
                            local_symbol_skip_reasons[symbol] = {
                                'reason': skip_reason,
                                'error': str(e),
                                'error_type': type(e).__name__
                            }
                            continue
                    
                    # Log summary of skip reasons
                    if local_symbol_skip_reasons:
                        logger.warning(f"  📋 Symbol-specific skip reasons for {target}: {len(local_symbol_skip_reasons)}/{len(symbols)} symbols skipped")
                        for sym, skip_info in local_symbol_skip_reasons.items():
                            reason = skip_info.get('reason', 'unknown')
                            logger.debug(f"    {sym}: {reason}")
                    
                    # Store skip reasons for routing decisions (use global symbol_skip_reasons dict)
                    if local_symbol_skip_reasons:
                        symbol_skip_reasons[target] = local_symbol_skip_reasons
                    
                    logger.info(f"  📊 Symbol-specific results for {target}: {len(result_sym_dict)}/{len(symbols)} symbols succeeded")
                
                # View C: LOSO evaluation (optional, if enabled)
                result_loso_dict = {}
                if enable_loso:
                    logger.info(f"  View C: LOSO (evaluating {len(symbols)} symbols)")
                    for symbol in symbols:
                        try:
                            result_loso_sym = evaluate_target_predictability(
                                target=target,
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
                                symbol=symbol,
                                run_identity=run_identity,  # NEW: Pass RunIdentity
                            )
                            result_loso_dict[symbol] = result_loso_sym
                        except Exception as e:
                            logger.warning(f"    Failed LOSO evaluation for {target} on symbol {symbol}: {e}")
                            continue
                
                # Store symbol-specific results
                if enable_symbol_specific and result_sym_dict:
                    if target not in results_sym:
                        results_sym[target] = {}
                    
                    stored_count = 0
                    for symbol, result_sym in result_sym_dict.items():
                        if result_sym.auc != -999.0 and result_sym.status not in skip_statuses:
                            results_sym[target][symbol] = result_sym
                            stored_count += 1
                        else:
                            reason = result_sym.status if result_sym.status in skip_statuses else "degenerate/failed"
                            logger.warning(f"    ⚠️  Filtered out {target} SYMBOL_SPECIFIC ({symbol}): {reason} (auc={result_sym.auc})")
                            # Add to skip reasons if not already there (use global symbol_skip_reasons dict)
                            if target not in symbol_skip_reasons:
                                symbol_skip_reasons[target] = {}
                            if symbol not in symbol_skip_reasons[target]:
                                symbol_skip_reasons[target][symbol] = {
                                    'reason': reason,
                                    'status': result_sym.status,
                                    'auc': result_sym.auc
                                }
                    if stored_count > 0:
                        logger.info(f"  ✅ Stored {stored_count} symbol-specific results for {target}")
                    elif len(result_sym_dict) > 0:
                        logger.warning(f"  ⚠️  All {len(result_sym_dict)} symbol-specific results for {target} were filtered out")
                    
                    # Save routing decision immediately after target evaluation (incremental)
                    if output_dir:
                        from TRAINING.ranking.target_routing import (
                            _compute_single_target_routing_decision, _save_single_target_decision
                        )
                        target_sym_results = results_sym.get(target, {})
                        target_skip_reasons = symbol_skip_reasons.get(target, {}) if symbol_skip_reasons else {}
                        decision = _compute_single_target_routing_decision(
                            target=target,
                            result_cs=result_cs if cs_succeeded else None,
                            sym_results=target_sym_results,
                            symbol_skip_reasons=target_skip_reasons
                        )
                        _save_single_target_decision(target, decision, output_dir)
                
                # Store LOSO results
                if enable_loso:
                    if target not in results_loso:
                        results_loso[target] = {}
                    for symbol, result_loso_sym in result_loso_dict.items():
                        if result_loso_sym.auc != -999.0 and result_loso_sym.status not in skip_statuses:
                            results_loso[target][symbol] = result_loso_sym
            
            except Exception as e:
                logger.exception(f"  Failed to evaluate {target}: {e}")  # Better error logging with traceback
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
    # Use global symbol_skip_reasons dict (already initialized at top of function)
    
    routing_decisions = _compute_target_routing_decisions(
        results_cs=results_cs,
        results_sym=results_sym,
        results_loso=results_loso if enable_loso else {},
        symbol_skip_reasons=symbol_skip_reasons
    )
    
    # CRITICAL: Ensure ALL evaluated targets have routing decisions, even if they're not in routing_decisions
    # This handles cases where CS failed and all symbols failed (target won't be in routing_decisions)
    if output_dir:
        from TRAINING.ranking.target_routing import (
            _compute_single_target_routing_decision, _save_single_target_decision
        )
        for target in all_evaluated_targets:
            if target not in routing_decisions:
                # Target was evaluated but not in routing_decisions (CS failed + all symbols failed)
                # Compute and save decision anyway
                logger.debug(f"Computing routing decision for {target} (not in routing_decisions, likely all evaluations failed)")
                target_sym_results = results_sym.get(target, {})
                target_skip_reasons = symbol_skip_reasons.get(target, {}) if symbol_skip_reasons else {}
                # Find CS result even if it failed (from all_cs_results which includes failed ones)
                result_cs = all_cs_results.get(target)
                decision = _compute_single_target_routing_decision(
                    target=target,
                    result_cs=result_cs,  # Will be None if CS was never evaluated or failed
                    sym_results=target_sym_results,
                    symbol_skip_reasons=target_skip_reasons
                )
                _save_single_target_decision(target, decision, output_dir)
                # Also add to routing_decisions so it's in the global file
                routing_decisions[target] = decision
    
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
        
        # Generate metrics rollups after all targets are evaluated
        try:
            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
            from datetime import datetime
            
            # Find the REPRODUCIBILITY directory (could be in output_dir or parent)
            repro_dir = output_dir / "REPRODUCIBILITY"
            if not repro_dir.exists() and output_dir.parent.exists():
                repro_dir = output_dir.parent / "REPRODUCIBILITY"
            
            if repro_dir.exists():
                # Use output_dir parent as base (where RESULTS/runs/ typically is)
                base_dir = output_dir.parent if (output_dir / "REPRODUCIBILITY").exists() else output_dir
                tracker = ReproducibilityTracker(output_dir=base_dir)
                # Generate run_id from output_dir name or timestamp
                run_id = output_dir.name if output_dir.name else datetime.now().strftime("%Y%m%d_%H%M%S")
                tracker.generate_metrics_rollups(stage="TARGET_RANKING", run_id=run_id)
                logger.debug("✅ Generated metrics rollups for TARGET_RANKING")
        except Exception as e:
            logger.debug(f"Failed to generate metrics rollups: {e}")
    
    return results

