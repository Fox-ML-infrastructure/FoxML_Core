# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

# ---- PATH BOOTSTRAP: ensure project root on sys.path in parent AND children ----
import os, sys
from pathlib import Path

# CRITICAL: Set LD_LIBRARY_PATH for conda CUDA libraries BEFORE any imports
# This must happen before TensorFlow tries to load CUDA libraries
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib = os.path.join(conda_prefix, "lib")
    conda_targets_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = []
    if conda_lib not in current_ld_path:
        new_paths.append(conda_lib)
    if conda_targets_lib not in current_ld_path:
        new_paths.append(conda_targets_lib)
    if new_paths:
        updated_ld_path = ":".join(new_paths + [current_ld_path] if current_ld_path else new_paths)
        os.environ["LD_LIBRARY_PATH"] = updated_ld_path

# Show TensorFlow warnings so user knows if GPU isn't working
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Removed - show warnings
# os.environ.setdefault("TF_LOGGING_VERBOSITY", "ERROR")  # Removed - show warnings

# project root: TRAINING/training_strategies/*.py -> parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure Python can import `common`, `model_fun`, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Propagate to spawned processes (spawned interpreter reads PYTHONPATH at startup)
os.environ.setdefault("PYTHONPATH", str(_PROJECT_ROOT))

# Set up all paths using centralized utilities
# Note: setup_all_paths already adds CONFIG to sys.path
from TRAINING.common.utils.path_setup import setup_all_paths
_PROJECT_ROOT, _TRAINING_ROOT, _CONFIG_DIR = setup_all_paths(_PROJECT_ROOT)

# Import config loader (CONFIG is already in sys.path from setup_all_paths)
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    # Only log at debug level to avoid misleading warnings
    logging.getLogger(__name__).debug("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
from TRAINING.common.utils.process_cleanup import setup_loky_cleanup_from_config
setup_loky_cleanup_from_config()

"""
Enhanced Training Script with Multiple Strategies - Full Original Functionality

Replicates ALL functionality from train_mtf_cross_sectional_gpu.py but with:
- Modular architecture
- 3 training strategies (single-task, multi-task, cascade)
- All 20 model families from original script
- GPU acceleration
- Memory management
- Batch processing
- Cross-sectional training
- Target discovery
- Data validation
"""

# ANTI-DEADLOCK: Process-level safety (before importing TF/XGB/sklearn)
import time as _t
# Make thread pools predictable (also avoids weird deadlocks)


# Import the isolation runner (moved to TRAINING/common/isolation_runner.py)
# Add TRAINING to path for local imports
_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Also add current directory for relative imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications - import from centralized constants
from TRAINING.common.family_constants import TF_FAMS, TORCH_FAMS, CPU_FAMS, TORCH_SEQ_FAMILIES


"""Core training functions."""

# Import dependencies
from TRAINING.training_strategies.execution.family_runners import _run_family_inproc, _run_family_isolated
from TRAINING.training_strategies.execution.data_preparation import prepare_training_data_cross_sectional
from TRAINING.training_strategies.utils import (
    FAMILY_CAPS, ALL_FAMILIES, tf_available, ngboost_available,
    _now, THREADS, CPU_ONLY,
    TORCH_SEQ_FAMILIES, build_sequences_from_features, _env_guard, safe_duration
)
# train_model_comprehensive is defined in this file, not in utils
from TRAINING.orchestration.routing.target_router import TaskSpec
from TRAINING.training_strategies.strategies.single_task import SingleTaskStrategy
from TRAINING.training_strategies.strategies.multi_task import MultiTaskStrategy
from TRAINING.training_strategies.strategies.cascade import CascadeStrategy

# Standard library imports
from typing import Dict, List, Any, Optional, Tuple
import logging
import json
import joblib

# Third-party imports
import numpy as np
import pandas as pd

# Setup logger
logger = logging.getLogger(__name__)

def train_model_comprehensive(family: str, X: np.ndarray, y: np.ndarray, 
                            target: str, strategy: str, feature_names: List[str],
                            caps: Dict[str, Any], routing_meta: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train model using modular trainers directly - enforces runtime policy and routing."""
    
    # CRITICAL: Normalize family name before all registry lookups
    from TRAINING.training_strategies.utils import normalize_family_name
    family = normalize_family_name(family)
    
    logger.info(f"🎯 Training {family} model with {strategy} strategy")
    
    # Extract routing info
    if routing_meta is None:
        routing_meta = {
            'spec': TaskSpec('regression', 'regression', ['rmse', 'mae']),
            'sample_weights': None,
            'group_sizes': None
        }
    
    # Safety check: ensure spec exists (defensive programming)
    spec = routing_meta.get('spec')
    if spec is None:
        # Fallback to default regression spec if missing
        spec = TaskSpec('regression', 'regression', ['rmse', 'mae'])
        routing_meta['spec'] = spec
        logger.warning(f"[{family}] routing_meta missing 'spec', using default regression spec")
    
    sample_weights = routing_meta.get('sample_weights')
    group_sizes = routing_meta.get('group_sizes')
    
    logger.info(f"[{family}] Task={spec.task}, Objective={spec.objective}, Has weights={sample_weights is not None}, Has groups={group_sizes is not None}")
    
    # Get runtime policy for this family (single source of truth)
    from common.runtime_policy import get_policy
    policy = get_policy(family)
    
    # Log policy decision
    if policy.force_isolation_reason:
        logger.info(f"[{family}] Policy: {policy.run_mode} mode ({policy.force_isolation_reason})")
    else:
        logger.info(f"[{family}] Policy: {policy.run_mode} mode, GPU={policy.needs_gpu}, backends={list(policy.backends)}")
    
    # Determine backend for logging
    if "tf" in policy.backends:
        backend = "TF"
    elif "torch" in policy.backends:
        backend = "PyTorch"
    elif "xgb" in policy.backends:
        backend = "XGBoost"
    elif policy.omp_user_api == "blas":
        backend = "BLAS"
    else:
        backend = "OpenMP"
    
    # Honor user override for in-process training (but policy can force isolation)
    user_wants_inproc = os.getenv("TRAINER_NO_ISOLATION", "0") in ("1", "true", "True")
    user_force_iso = os.getenv("TRAINER_FORCE_ISOLATION_FOR", "")
    family_force_isolated = family in [f.strip() for f in user_force_iso.replace(",", " ").split() if f.strip()]
    
    # Final decision: policy OR user override
    if policy.run_mode == "process" or family_force_isolated:
        USE_INPROC = False
    elif policy.run_mode == "inproc" and user_wants_inproc:
        USE_INPROC = True
    else:
        # Default to policy
        USE_INPROC = (policy.run_mode == "inproc")
    
    # Build trainer config with routing info
    from TRAINING.orchestration.routing.target_router import get_objective_for_family
    
    trainer_config = {
        "num_threads": THREADS,
        "objective": get_objective_for_family(family, spec),
        "task_type": spec.task,
    }
    
    # Add routing-specific config for supported families
    if family in ['LightGBM', 'QuantileLightGBM']:
        if spec.task == 'multiclass' and routing_meta.get('label_map'):
            trainer_config["num_class"] = len(routing_meta['label_map'])
        if group_sizes is not None:
            try:
                gs = np.asarray(group_sizes).ravel().tolist()
            except Exception:
                gs = group_sizes
            trainer_config["groups"] = gs
        if sample_weights is not None:
            try:
                sw = np.asarray(sample_weights).ravel().tolist()
            except Exception:
                sw = sample_weights
            trainer_config["sample_weight"] = sw
    
    elif family == 'XGBoost':
        if spec.task == 'multiclass' and routing_meta.get('label_map'):
            trainer_config["num_class"] = len(routing_meta['label_map'])
        if sample_weights is not None:
            try:
                sw = np.asarray(sample_weights).ravel().tolist()
            except Exception:
                sw = sample_weights
            trainer_config["sample_weight"] = sw
    
    logger.info(f"[{family}] Trainer config: {trainer_config}")
    
    # Execute based on decision
    if USE_INPROC:
        logger.info("🔄 [%s] using in-process training (no isolation) with %s threads", family, THREADS)
        print(f"🔄 [{family}] using in-process training with {THREADS} threads...")
        model = _run_family_inproc(
            family, X, y,
            total_threads=THREADS,
            trainer_kwargs={"config": trainer_config}
        )
    else:
        logger.info("🔄 [%s] using isolation runner (%s backend)…", family, backend)
        print(f"🔄 [{family}] using isolation runner ({backend} backend)...")
        # Pass None to use optimal thread planning from plan_for_family()
        model = _run_family_isolated(
            family, X, y,
            omp_threads=None,  # Use optimal planning
            mkl_threads=None,  # Use optimal planning
            trainer_kwargs={"config": trainer_config}
        )
    
    # Wrap model in strategy manager
    manager = SingleTaskStrategy({'family': family})
    manager.models[family] = model
    return {
        'model': model,
        'trainer': None, 'test_predictions': None, 'success': True,
        'family': family, 'target': target, 'strategy': strategy,
        'strategy_manager': manager
    }


def normalize_selected_features(
    target: str,
    target_feat_data: Any,
    symbols: List[str],
    route: Optional[str] = None
) -> Dict[str, Any]:
    """
    Normalize selected_features to canonical shape.
    
    Input shapes accepted:
    - List[str]: CROSS_SECTIONAL features
    - Dict with 'cross_sectional' key: CROSS_SECTIONAL features
    - Dict with symbol keys (e.g., {'AAPL': [...], 'MSFT': [...]}): SYMBOL_SPECIFIC
    - Dict with 'symbol_specific' key: SYMBOL_SPECIFIC
    - Dict with 'BOTH' structure: {'cross_sectional': [...], 'symbol_specific': {...}}
    
    Returns canonical shape:
    - CROSS_SECTIONAL: {'cross_sectional': [features...]}
    - SYMBOL_SPECIFIC: {'symbol_specific': {symbol: [features...]}}
    - BOTH: {'cross_sectional': [...], 'symbol_specific': {...}}
    
    Raises:
        ValueError: If shape cannot be normalized
    """
    # Known meta keys that indicate structure type
    META_KEYS = {'cross_sectional', 'symbol_specific', 'route', 'CROSS_SECTIONAL', 'SYMBOL_SPECIFIC'}
    
    if target_feat_data is None:
        return {'cross_sectional': []}
    
    # Case 1: Simple list -> CROSS_SECTIONAL
    if isinstance(target_feat_data, (list, tuple)):
        return {'cross_sectional': list(target_feat_data)}
    
    # Case 2: Dict
    if not isinstance(target_feat_data, dict):
        raise ValueError(
            f"selected_features for {target} must be list or dict, got {type(target_feat_data)}. "
            f"Value: {target_feat_data}"
        )
    
    keys = set(target_feat_data.keys())
    
    # Case 2a: Already normalized (has 'cross_sectional' or 'symbol_specific')
    if 'cross_sectional' in keys or 'CROSS_SECTIONAL' in keys:
        cs_key = 'cross_sectional' if 'cross_sectional' in keys else 'CROSS_SECTIONAL'
        cs_features = target_feat_data[cs_key]
        if isinstance(cs_features, (list, tuple)):
            result = {'cross_sectional': list(cs_features)}
            if 'symbol_specific' in keys or 'SYMBOL_SPECIFIC' in keys:
                ss_key = 'symbol_specific' if 'symbol_specific' in keys else 'SYMBOL_SPECIFIC'
                result['symbol_specific'] = target_feat_data[ss_key]
            return result
        else:
            raise ValueError(f"cross_sectional value must be list, got {type(cs_features)}")
    
    if 'symbol_specific' in keys or 'SYMBOL_SPECIFIC' in keys:
        ss_key = 'symbol_specific' if 'symbol_specific' in keys else 'SYMBOL_SPECIFIC'
        return {'symbol_specific': target_feat_data[ss_key]}
    
    # Case 2b: Dict keys are symbol names (SYMBOL_SPECIFIC shape)
    symbol_keys = keys & set(symbols)  # Intersection with known symbols
    if symbol_keys and not (keys & META_KEYS):
        # All keys are symbols -> SYMBOL_SPECIFIC
        return {'symbol_specific': {sym: target_feat_data[sym] for sym in symbol_keys}}
    
    # Case 2c: Unknown structure
    raise ValueError(
        f"selected_features for {target} has unrecognized dict structure. "
        f"Keys: {list(keys)}. "
        f"Expected: list[str], dict with 'cross_sectional' key, dict with symbol keys, or BOTH structure. "
        f"Route: {route}"
    )


def train_models_for_interval_comprehensive(interval: str, targets: List[str], 
                                           mtf_data: Dict[str, pd.DataFrame],
                                           families: List[str],
                                           strategy: str = 'single_task',
                                           output_dir: str = 'output',
                                           min_cs: int = 10,
                                           max_cs_samples: int = None,
                                           max_rows_train: int = None,
                                           target_features: Dict[str, Any] = None,
                                           target_families: Optional[Dict[str, List[str]]] = None,
                                           routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None,
                                           run_identity: Optional[Any] = None,  # NEW: RunIdentity for reproducibility tracking
                                           experiment_config: Optional[Any] = None) -> Dict[str, Any]:  # NEW: For SST fallback
    """Train models for a specific interval using comprehensive approach (replicates original script)."""
    
    logger.info(f"🎯 Training models for interval: {interval}")
    
    results = {
        'interval': interval,
        'targets': targets,
        'families': families,
        'strategy': strategy,
        'models': {},
        'metrics': {},
        'failed_targets': [],  # Track targets that failed data preparation
        'failed_reasons': {}   # Track why each target failed
    }
    
    # SST: Create or use run_identity for reproducibility tracking
    # Mirrors the pattern from FEATURE_SELECTION and multi_model_feature_selection.py
    effective_run_identity = run_identity
    if effective_run_identity is None:
        try:
            from TRAINING.common.utils.fingerprinting import create_stage_identity
            symbols = list(mtf_data.keys()) if mtf_data else []
            effective_run_identity = create_stage_identity(
                stage="TRAINING",
                symbols=symbols,
                experiment_config=experiment_config,
            )
            logger.debug(f"Created fallback TRAINING identity with train_seed={effective_run_identity.train_seed}")
        except Exception as e:
            logger.debug(f"Failed to create fallback identity for TRAINING: {e}")
    
    for j, target in enumerate(targets, 1):
        logger.info(f"🎯 [{j}/{len(targets)}] Training models for target: {target}")
        
        # Get families for this target (per-target families override global) - with validation
        target_families = families
        if target_families is not None and isinstance(target_families, dict) and target in target_families:
            try:
                per_target_families = target_families[target]
                if isinstance(per_target_families, list) and per_target_families:
                    target_families = per_target_families
                    logger.info(f"📋 Using per-target families for {target}: {target_families}")
                else:
                    logger.debug(f"Per-target families for {target} is empty or invalid, using global")
            except (KeyError, TypeError) as e:
                logger.debug(f"Could not get per-target families for {target}: {e}, using global")
        
        # Validate target_families is a list
        if not isinstance(target_families, list):
            logger.warning(f"target_families is not a list for {target}, got {type(target_families)}, using global")
            target_families = families
        
        if not target_families:
            logger.warning(f"No families available for {target}, using global families")
            target_families = families
        
        # Validate configured families are being attempted (diagnostic)
        try:
            from CONFIG.config_loader import get_cfg
            configured_families = get_cfg("training.model_families", default=None, config_name="training_config")
            if isinstance(configured_families, list) and configured_families:
                logger.info(f"📋 Config specifies {len(configured_families)} families: {configured_families}")
                # Check if all configured families are in the training list
                missing = set(configured_families) - set(target_families)
                if missing:
                    logger.warning(f"⚠️ Config specifies {len(missing)} families not in training list for {target}: {missing}")
                else:
                    logger.debug(f"✅ All configured families are in training list for {target}")
        except Exception as e:
            logger.debug(f"Could not validate configured families: {e}")
        
        # Prepare training data with cross-sectional sampling
        print(f"🔄 Preparing training data for target: {target}")  # Debug print
        prep_start = _t.time()
        
        # Use selected features for this target if provided
        # Normalize to canonical shape using single normalization function
        selected_features = None
        route_info = routing_decisions.get(target, {}) if routing_decisions else {}
        route = route_info.get('route', 'CROSS_SECTIONAL')
        
        # CRITICAL FIX: Check if cross-sectional is explicitly DISABLED in routing plan
        cs_info = route_info.get('cross_sectional', {})
        cs_route_status = cs_info.get('route', 'ENABLED') if isinstance(cs_info, dict) else 'ENABLED'
        
        if target_features and target in target_features:
            target_feat_data = target_features[target]
            
            # Handle different structures based on route
            if route == 'BLOCKED':
                # BLOCKED targets should be skipped entirely
                logger.warning(f"Skipping {target} (BLOCKED: {route_info.get('reason', 'suspicious score')})")
                results['failed_targets'].append(target)
                results['failed_reasons'][target] = f"BLOCKED: {route_info.get('reason', 'suspicious score')}"
                continue
            
            # Normalize selected_features to canonical shape
            symbols_list = list(mtf_data.keys())
            try:
                normalized = normalize_selected_features(
                    target=target,
                    target_feat_data=target_feat_data,
                    symbols=symbols_list,
                    route=route
                )
            except ValueError as e:
                logger.error(f"Failed to normalize selected_features for {target}: {e}")
                results['failed_targets'].append(target)
                results['failed_reasons'][target] = f"Normalization error: {e}"
                continue
            
            # Extract features based on route from normalized structure
            if route == 'CROSS_SECTIONAL':
                # CRITICAL FIX: Respect routing plan - skip CS training if DISABLED
                if cs_route_status == 'DISABLED':
                    logger.warning(
                        f"Skipping cross-sectional training for {target}: "
                        f"CS route is DISABLED in routing plan (reason: {cs_info.get('reason', 'unknown')})"
                    )
                    results['failed_targets'].append(target)
                    results['failed_reasons'][target] = f"CS: DISABLED in routing plan ({cs_info.get('reason', 'unknown')})"
                    continue
                
                # Extract cross-sectional features from normalized structure
                if 'cross_sectional' in normalized:
                    selected_features = normalized['cross_sectional']
                else:
                    raise ValueError(
                        f"Normalized structure for {target} missing 'cross_sectional' key for CROSS_SECTIONAL route. "
                        f"Keys: {list(normalized.keys())}"
                    )
                
                # Validate it's a list
                if selected_features is not None:
                    if not isinstance(selected_features, (list, tuple)):
                        raise TypeError(
                            f"selected_features for {target} must be list[str], got {type(selected_features)}. "
                            f"Value: {selected_features}"
                        )
                    elif len(selected_features) == 0:
                        logger.warning(f"selected_features for {target} is empty, will auto-discover features")
                        selected_features = None
                    else:
                        logger.info(f"Using {len(selected_features)} cross-sectional features for {target}")
            elif route == 'SYMBOL_SPECIFIC':
                # Extract symbol-specific features from normalized structure
                if 'symbol_specific' in normalized:
                    symbol_specific_features = normalized['symbol_specific']
                    if not isinstance(symbol_specific_features, dict):
                        raise TypeError(
                            f"SYMBOL_SPECIFIC route: normalized['symbol_specific'] must be dict, got {type(symbol_specific_features)}"
                        )
                    
                    # Validate dict structure: all values should be lists
                    for symbol, features in symbol_specific_features.items():
                        if not isinstance(features, (list, tuple)):
                            raise TypeError(
                                f"SYMBOL_SPECIFIC route: features for symbol {symbol} must be list, got {type(features)}"
                            )
                    
                    # Will handle per-symbol training below
                    logger.info(f"SYMBOL_SPECIFIC route: will train {len(symbol_specific_features)} separate models (one per symbol)")
                    # Store in target_features for per-symbol training below
                    target_features[target] = symbol_specific_features
                else:
                    raise ValueError(
                        f"Normalized structure for {target} missing 'symbol_specific' key for SYMBOL_SPECIFIC route. "
                        f"Keys: {list(normalized.keys())}"
                    )
            elif route == 'BOTH':
                # Extract both cross-sectional and symbol-specific from normalized structure
                if 'cross_sectional' in normalized:
                    selected_features = normalized['cross_sectional']
                    if selected_features and isinstance(selected_features, (list, tuple)):
                        logger.info(f"Using {len(selected_features)} cross-sectional features for {target} (BOTH route - CS training)")
                    else:
                        logger.warning(f"Cross-sectional features not found in BOTH structure for {target}")
                        selected_features = None
                else:
                    raise ValueError(
                        f"BOTH route normalized structure for {target} missing 'cross_sectional' key. "
                        f"Keys: {list(normalized.keys())}"
                    )
                
                # For BOTH route, also check if we need symbol-specific training
                # Extract symbol-specific features from normalized structure
                symbol_specific_features = normalized.get('symbol_specific', {})
                if symbol_specific_features and isinstance(symbol_specific_features, dict) and len(symbol_specific_features) > 0:
                    # Check routing plan to see which symbols need symbol-specific training
                    winner_symbols = route_info.get('winner_symbols', [])
                    if not winner_symbols:
                        # If no winner_symbols specified, check symbol_specific section of routing plan
                        sym_info = route_info.get('symbol_specific', {})
                        if isinstance(sym_info, dict):
                            # Get all symbols that have symbol-specific routing enabled
                            winner_symbols = [sym for sym, sym_data in sym_info.items() 
                                             if isinstance(sym_data, dict) and sym_data.get('route') == 'ENABLED']
                    
                    if winner_symbols:
                        # Filter symbol-specific features to only those symbols that need training
                        filtered_symbol_features = {sym: symbol_specific_features[sym] 
                                                   for sym in winner_symbols 
                                                   if sym in symbol_specific_features}
                        if filtered_symbol_features:
                            logger.info(f"BOTH route: Will train symbol-specific models for {len(filtered_symbol_features)} symbols: {list(filtered_symbol_features.keys())}")
                            # Temporarily store symbol-specific features for symbol-specific training path
                            # We'll restore the full structure after symbol-specific training
                            target_features[target] = filtered_symbol_features
                        else:
                            logger.warning(f"BOTH route: No symbol-specific features found for winner symbols: {winner_symbols}")
                    else:
                        logger.info(f"BOTH route: No symbol-specific training needed (no winner_symbols or all disabled)")
                else:
                    logger.warning(f"BOTH route: No symbol-specific features found in structure")
            else:
                # Unknown route, try to extract as list
                selected_features = target_feat_data
                if selected_features and not isinstance(selected_features, (list, tuple)):
                    logger.warning(f"Unknown route {route}, attempting to convert features to list")
                    try:
                        selected_features = list(selected_features) if selected_features else None
                    except Exception as e:
                        logger.error(f"Failed to convert features to list: {e}")
                        selected_features = None
        
        # CRITICAL: Run preflight validation BEFORE both CROSS_SECTIONAL and SYMBOL_SPECIFIC paths
        # This ensures invalid families are filtered out for all routes
        from TRAINING.training_strategies.utils import normalize_family_name
        from common.isolation_runner import TRAINER_MODULE_MAP
        
        # Get canonical family set from registries (must match MODMAP in family_runners.py)
        MODMAP_KEYS = {
            "lightgbm", "quantile_lightgbm", "xgboost", "reward_based", "gmm_regime",
            "change_point", "ngboost", "ensemble", "ftrl_proximal", "mlp", "neural_network", "vae",
            "gan", "meta_learning", "multi_task"
        }
        REGISTRY_KEYS = MODMAP_KEYS | set(TRAINER_MODULE_MAP.keys())
        
        # Normalize and validate requested families (PREFLIGHT - runs for all routes)
        validated_families = []
        skipped_families = []
        invalid_families = []
        
        for family in target_families:
            normalized = normalize_family_name(family)
            if normalized in ["mutual_information", "univariate_selection"]:
                skipped_families.append((family, normalized, "feature_selector_not_trainer"))
            elif normalized not in REGISTRY_KEYS:
                invalid_families.append((family, normalized))
            else:
                validated_families.append(normalized)
        
        # Log preflight results
        if invalid_families:
            known_missing = {'random_forest', 'catboost', 'lasso', 'elastic_net'}
            missing_normalized = {norm for _, norm in invalid_families}
            
            if missing_normalized & known_missing:
                # These families are used in feature selection but not as trainers (expected behavior)
                # Log at INFO level instead of WARNING to reduce noise
                logger.info(
                    f"ℹ️ Preflight [{target}]: {len(invalid_families)} families not in trainer registry "
                    f"(known missing: {sorted(missing_normalized & known_missing)}). "
                    f"These families are used in feature selection but not registered as trainers (expected)."
                )
            else:
                logger.warning(f"⚠️ Preflight [{target}]: {len(invalid_families)} families not in trainer registry:")
            for raw, norm in invalid_families:
                # Only log individual families at WARNING if they're not known missing
                if norm not in known_missing:
                    logger.warning(f"  - {raw} (normalized: {norm}) → SKIP")
                else:
                    logger.debug(f"  - {raw} (normalized: {norm}) → SKIP (known missing, expected)")
        if skipped_families:
            logger.info(f"ℹ️ Preflight [{target}]: {len(skipped_families)} families are selectors (not trainers):")
            for raw, norm, reason in skipped_families:
                logger.info(f"  - {raw} (normalized: {norm}) → SKIP ({reason})")
        
        if not validated_families:
            logger.error(f"❌ No valid trainer families after preflight validation for {target}. Requested: {target_families}")
            results['failed_targets'].append(target)
            results['failed_reasons'][target] = f"No valid trainer families (invalid: {[f[0] for f in invalid_families]}, selectors: {[f[0] for f in skipped_families]})"
            continue
        
        logger.info(f"✅ Preflight [{target}]: {len(validated_families)} valid trainer families, {len(skipped_families)} selectors skipped, {len(invalid_families)} invalid")
        
        # Handle SYMBOL_SPECIFIC route separately (per-symbol training)
        if route == 'SYMBOL_SPECIFIC' and isinstance(target_feat_data, dict):
            # Train separate models for each symbol
            logger.info(f"🔄 Training per-symbol models for {target} ({len(target_feat_data)} symbols)")
            
            for symbol, symbol_features in target_feat_data.items():
                if symbol not in mtf_data:
                    logger.warning(f"Skipping {symbol} for {target}: symbol not in mtf_data")
                    continue
                
                if not isinstance(symbol_features, (list, tuple)) or len(symbol_features) == 0:
                    logger.warning(f"Skipping {symbol} for {target}: no features available")
                    continue
                
                logger.info(f"  📊 Training {symbol} with {len(symbol_features)} features")
                
                # Apply runtime quarantine (dominance quarantine confirmed features) for symbol-specific
                if symbol_features and output_dir:
                    try:
                        from TRAINING.ranking.utils.dominance_quarantine import load_confirmed_quarantine
                        from pathlib import Path
                        output_dir_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
                        runtime_quarantine = load_confirmed_quarantine(
                            output_dir=output_dir_path,
                            target=target,
                            view="SYMBOL_SPECIFIC",
                            symbol=symbol
                        )
                        if runtime_quarantine:
                            symbol_features = [f for f in symbol_features if f not in runtime_quarantine]
                            logger.info(f"  🔒 {symbol}: Applied runtime quarantine: Removed {len(runtime_quarantine)} confirmed leaky features ({len(symbol_features)} remaining)")
                    except Exception as e:
                        logger.debug(f"Could not load runtime quarantine for {target}:{symbol}: {e}")
                
                # Prepare data for this symbol only
                symbol_mtf_data = {symbol: mtf_data[symbol]}
                X, y, feature_names, symbols_arr, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
                    symbol_mtf_data, target, feature_names=symbol_features, min_cs=1, max_cs_samples=max_cs_samples, routing_decisions=routing_decisions
                )
                
                if X is None or len(X) == 0:
                    logger.warning(f"❌ Failed to prepare data for {target}:{symbol}")
                    continue
                
                # Apply row cap if needed
                if max_rows_train and len(X) > max_rows_train:
                    from TRAINING.common.determinism import BASE_SEED, stable_seed_from
                    downsample_seed = stable_seed_from([target, symbol, 'downsample'])
                    rng = np.random.RandomState(downsample_seed)
                    idx = rng.choice(len(X), max_rows_train, replace=False)
                    X, y = X[idx], y[idx]
                    if time_vals is not None: time_vals = time_vals[idx]
                    if symbols_arr is not None: symbols_arr = symbols_arr[idx]
                    logger.info(f"✂️ Downsampled {symbol} to max_rows_train={max_rows_train}")
                
                # Extract routing info
                if isinstance(routing_meta, dict) and 'spec' in routing_meta:
                    logger.info(f"[Routing] {symbol}: Using task spec: {routing_meta['spec']}")
                else:
                    from TRAINING.orchestration.routing.target_router import route_target
                    route_info = route_target(target)
                    routing_meta = {
                        'target': target,
                        'spec': route_info['spec'],
                        'sample_weights': None,
                        'group_sizes': None
                    }
                
                # CRITICAL FIX: Use validated_families from preflight (not original target_families)
                # This prevents attempting to train invalid families (random_forest, catboost, etc.)
                families_to_train = validated_families
                if not families_to_train:
                    logger.warning(f"  ⚠️ No valid families to train for {target}:{symbol} (all filtered by preflight)")
                    continue
                
                # Train models for this symbol
                symbol_results = {}
                # Track family status for symbol-specific training
                symbol_family_status = {family: {"attempted": False, "saved": False, "error": None} for family in families_to_train}
                logger.info(f"🎯 Attempting to train {len(families_to_train)} model families for {target}:{symbol}: {families_to_train}")
                
                for family in families_to_train:
                    try:
                        symbol_family_status[family]["attempted"] = True
                        logger.info(f"  🤖 Training {family} for {target}:{symbol}")
                        model_result = train_model_comprehensive(
                            family, X, y, target, strategy, feature_names,
                            caps={}, routing_meta=routing_meta
                        )
                        
                        if model_result is None or not model_result.get('success', False):
                            symbol_family_status[family]["error"] = "Training returned None or success=False"
                            logger.warning(f"  ⚠️ {family} training for {target}:{symbol} did not succeed")
                            continue
                        
                        symbol_results[family] = model_result
                        
                        # Save model to canonical target-first structure using ArtifactPaths
                        try:
                            # Ensure output_dir is a Path object
                            output_dir_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
                            
                            # Find base run directory (parent of training_results/ if output_dir is training_results/)
                            base_run_dir = output_dir_path
                            if output_dir_path.name == 'training_results':
                                base_run_dir = output_dir_path.parent
                            elif (output_dir_path.parent / 'training_results').exists():
                                base_run_dir = output_dir_path.parent
                            
                            # Use ArtifactPaths to get canonical model directory
                            from TRAINING.orchestration.utils.artifact_paths import ArtifactPaths
                            model_dir = ArtifactPaths.model_dir(
                                run_root=base_run_dir,
                                target=target,
                                view="SYMBOL_SPECIFIC",
                                family=family,
                                symbol=symbol
                            )
                            model_dir.mkdir(parents=True, exist_ok=True)
                            logger.info(f"💾 Saving SYMBOL_SPECIFIC model for {symbol} to: {model_dir}")
                            
                            # Get the trained model from strategy manager
                            strategy_manager = model_result.get('strategy_manager')
                            if strategy_manager and hasattr(strategy_manager, 'models'):
                                models = strategy_manager.models
                                
                                # Import model wrapper for saving compatibility
                                from TRAINING.common.model_wrapper import wrap_model_for_saving, get_model_saving_info
                                
                                # Save each model component
                                for model_name, model in models.items():
                                    # Wrap model for saving compatibility
                                    wrapped_model = wrap_model_for_saving(model, family)
                                    
                                    # Get saving info
                                    save_info = get_model_saving_info(wrapped_model)
                                    
                                    # Determine file extensions based on model type
                                    # Save to canonical location using ArtifactPaths
                                    if save_info['is_lightgbm']:  # LightGBM
                                        model_path = ArtifactPaths.model_file(model_dir, family, extension='txt')
                                        wrapped_model.save_model(str(model_path))
                                        logger.info(f"  💾 LightGBM model saved: {model_path}")
                                        symbol_family_status[family]["saved"] = True
                                        
                                    elif save_info['is_tensorflow']:  # TensorFlow/Keras
                                        model_path = ArtifactPaths.model_file(model_dir, family, extension='keras')
                                        wrapped_model.save(str(model_path))
                                        logger.info(f"  💾 Keras model saved: {model_path}")
                                        symbol_family_status[family]["saved"] = True
                                        
                                    elif save_info['is_pytorch']:  # PyTorch models
                                        model_path = ArtifactPaths.model_file(model_dir, family, extension='pt')
                                        import torch
                                        
                                        # Extract the actual PyTorch model
                                        if hasattr(wrapped_model, 'core') and hasattr(wrapped_model.core, 'model'):
                                            torch_model = wrapped_model.core.model
                                        elif hasattr(wrapped_model, 'model'):
                                            torch_model = wrapped_model.model
                                        else:
                                            torch_model = wrapped_model
                                        
                                        # Save state dict + metadata
                                        torch.save({
                                            "state_dict": torch_model.state_dict(),
                                            "config": getattr(wrapped_model, "config", {}),
                                            "arch": family,
                                            "input_shape": X.shape
                                        }, str(model_path))
                                        logger.info(f"  💾 PyTorch model saved: {model_path}")
                                        symbol_family_status[family]["saved"] = True
                                        
                                    else:  # Scikit-learn models
                                        model_path = ArtifactPaths.model_file(model_dir, family, extension='joblib')
                                        wrapped_model.save(str(model_path))
                                        logger.info(f"  💾 Scikit-learn model saved: {model_path}")
                                        symbol_family_status[family]["saved"] = True
                                
                                # Save preprocessors if available
                                if wrapped_model.scaler is not None:
                                    scaler_path = ArtifactPaths.scaler_file(model_dir, family)
                                    joblib.dump(wrapped_model.scaler, scaler_path)
                                    logger.info(f"  💾 Scaler saved: {scaler_path}")
                                
                                if wrapped_model.imputer is not None:
                                    imputer_path = ArtifactPaths.imputer_file(model_dir, family)
                                    joblib.dump(wrapped_model.imputer, imputer_path)
                                    logger.info(f"  💾 Imputer saved: {imputer_path}")
                                
                                # Save metadata (match cross-sectional format)
                                # Define _pkg_ver BEFORE conditional blocks to avoid "referenced before assignment"
                                def _pkg_ver(pkg_name):
                                    try:
                                        import importlib.metadata
                                        return importlib.metadata.version(pkg_name)
                                    except:
                                        try:
                                            return __import__(pkg_name).__version__
                                        except:
                                            return "unknown"
                                
                                if save_info['is_lightgbm']:  # LightGBM - JSON format
                                    # Save metadata to canonical location using ArtifactPaths
                                    meta_path = ArtifactPaths.metadata_file(model_dir)
                                    
                                    import json
                                    metadata = {
                                        "family": family,
                                        "target": target,
                                        "symbol": symbol,
                                        "route": "SYMBOL_SPECIFIC",  # Add route indicator
                                        "view": "SYMBOL_SPECIFIC",  # Add view indicator
                                        "min_cs": 1,  # Per-symbol training doesn't use min_cs
                                        "features": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                                        "feature_names": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                                        "n_features": len(feature_names),
                                        "package_versions": {
                                            "numpy": _pkg_ver("numpy"),
                                            "pandas": _pkg_ver("pandas"),
                                            "sklearn": _pkg_ver("sklearn"),
                                            "lightgbm": _pkg_ver("lightgbm"),
                                            "xgboost": _pkg_ver("xgboost"),
                                            "tensorflow": _pkg_ver("tensorflow"),
                                            "ngboost": _pkg_ver("ngboost"),
                                        },
                                        "cli_args": {
                                            "min_cs": 1,
                                            "max_cs_samples": max_cs_samples,
                                            "cs_normalize": "per_ts_split",
                                            "cs_block": 32,
                                            "cs_winsor_p": 0.01,
                                            "cs_ddof": 1,
                                            "batch_id": 0,
                                            "families": [family],
                                            "symbol": symbol
                                        },
                                        "n_rows_train": len(X),
                                        "n_rows_val": 0,
                                        "train_timestamps": int(np.unique(time_vals).size) if time_vals is not None else len(X),
                                        "val_timestamps": 0,
                                        "time_col": None,
                                        "val_start_ts": None,
                                        "metrics": {
                                            "mean_IC": 0.0,
                                            "mean_RankIC": 0.0,
                                            "IC_IR": 0.0,
                                            "n_times": 0,
                                            "hit_rate": 0.0,
                                            "skipped_timestamps": 0,
                                            "total_timestamps": 0
                                        },
                                        "routing": {
                                            "route": "SYMBOL_SPECIFIC",
                                            "symbol": symbol,
                                            "view": "SYMBOL_SPECIFIC"
                                        }
                                    }
                                    
                                    # Add CV scores if available
                                    if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
                                        cv_scores = strategy_manager.cv_scores
                                        if cv_scores and len(cv_scores) > 0:
                                            metadata["cv_scores"] = [float(s) for s in cv_scores]
                                            metadata["cv_mean"] = float(np.mean(cv_scores))
                                            metadata["cv_std"] = float(np.std(cv_scores))
                                    
                                    # Save metadata to canonical location
                                    with open(meta_path, 'w') as f:
                                        json.dump(metadata, f, indent=2)
                                    logger.info(f"  💾 Metadata saved: {meta_path}")
                                
                                else:  # Other model types - save as JSON too
                                    # Save metadata to canonical location using ArtifactPaths
                                    meta_path = ArtifactPaths.metadata_file(model_dir)
                                    import json
                                    metadata = {
                                        "family": family,
                                        "target": target,
                                        "symbol": symbol,
                                        "route": "SYMBOL_SPECIFIC",  # Add route indicator
                                        "view": "SYMBOL_SPECIFIC",  # Add view indicator
                                        "min_cs": 1,
                                        "features": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                                        "feature_names": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                                        "n_features": len(feature_names),
                                        "n_rows_train": len(X),
                                        "train_timestamps": int(np.unique(time_vals).size) if time_vals is not None else len(X),
                                        "routing": {
                                            "route": "SYMBOL_SPECIFIC",
                                            "symbol": symbol,
                                            "view": "SYMBOL_SPECIFIC"
                                        }
                                    }
                                    
                                    # Add CV scores if available
                                    if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
                                        cv_scores = strategy_manager.cv_scores
                                        if cv_scores and len(cv_scores) > 0:
                                            metadata["cv_scores"] = [float(s) for s in cv_scores]
                                            metadata["cv_mean"] = float(np.mean(cv_scores))
                                            metadata["cv_std"] = float(np.std(cv_scores))
                                    
                                    # Save metadata to canonical location
                                    with open(meta_path, 'w') as f:
                                        json.dump(metadata, f, indent=2)
                                    logger.info(f"  💾 Metadata saved: {meta_path}")
                                    
                                    # SST: Create TrainingSnapshot for SYMBOL_SPECIFIC model
                                    try:
                                        from TRAINING.training_strategies.reproducibility import create_and_save_training_snapshot
                                        saved_model_path = str(model_path) if 'model_path' in locals() else None
                                        
                                        # Extract cohort_id from cohort_metadata if available
                                        symbol_cohort_id = None
                                        symbol_cohort_metadata = None
                                        if 'cohort_metadata' in locals() and cohort_metadata:
                                            symbol_cohort_metadata = cohort_metadata
                                            try:
                                                from TRAINING.training_strategies.reproducibility.io import compute_cohort_id_from_metadata
                                                symbol_cohort_id = compute_cohort_id_from_metadata(cohort_metadata, view="SYMBOL_SPECIFIC")
                                            except Exception as e:
                                                logger.debug(f"Failed to compute cohort_id for {symbol}: {e}")
                                        
                                        create_and_save_training_snapshot(
                                            target=target,
                                            model_family=family,
                                            model_result=model_result,
                                            output_dir=base_run_dir,
                                            view="SYMBOL_SPECIFIC",
                                            symbol=symbol,
                                            run_identity=effective_run_identity,
                                            model_path=saved_model_path,
                                            features_used=list(feature_names) if feature_names is not None else None,
                                            n_samples=len(X) if 'X' in locals() else None,
                                            cohort_id=symbol_cohort_id,
                                            cohort_metadata=symbol_cohort_metadata,
                                        )
                                    except Exception as ts_err:
                                        logger.debug(f"Training snapshot failed for {target}:{symbol} (non-critical): {ts_err}")
                            else:
                                # Fallback: save model directly if no strategy_manager
                                model_path = ArtifactPaths.model_file(model_dir, family, extension='joblib')
                                # joblib already imported at top of file (line 176)
                                joblib.dump(model_result.get('model'), model_path)
                                logger.info(f"  ✅ Saved {family} model for {target}:{symbol} to {model_path}")
                                symbol_family_status[family]["saved"] = True
                        except Exception as e:
                            symbol_family_status[family]["error"] = f"Save failed: {str(e)}"
                            logger.error(f"  ❌ Failed to save {family} model for {target}:{symbol}: {e}")
                    except Exception as e:
                        symbol_family_status[family]["error"] = f"Training failed: {str(e)}"
                        logger.error(f"  ❌ Training failed for {family} on {target}:{symbol}: {e}")
                        continue
                
                # Summary for this symbol
                symbol_failed = [f for f, s in symbol_family_status.items() if not s["saved"] and s["attempted"]]
                if symbol_failed:
                    logger.warning(f"  ⚠️ {len(symbol_failed)} families failed for {target}:{symbol}: {symbol_failed}")
                symbol_successful = [f for f, s in symbol_family_status.items() if s["saved"]]
                if symbol_successful:
                    logger.info(f"  ✅ {len(symbol_successful)} families saved for {target}:{symbol}: {symbol_successful}")
                    
                    # Save basic metadata to target-first structure with stage scoping
                    from TRAINING.orchestration.utils.target_first_paths import (
                        get_target_reproducibility_dir
                    )
                    repro_dir = get_target_reproducibility_dir(Path(output_dir), target, stage="TRAINING")
                    repro_dir.mkdir(parents=True, exist_ok=True)
                    meta_path_repro = repro_dir / f"meta_{family}_{symbol}_b0.json"
                    meta_path_legacy = symbol_target_dir / "meta_b0.json"
                    
                    import json
                    metadata = {
                        "family": family,
                        "target": target,
                        "symbol": symbol,
                        "route": "SYMBOL_SPECIFIC",  # Add route indicator
                        "view": "SYMBOL_SPECIFIC",  # Add view indicator
                        "n_features": len(feature_names) if feature_names else 0,
                        "n_rows_train": len(X),
                        "routing": {
                            "route": "SYMBOL_SPECIFIC",
                            "symbol": symbol,
                            "view": "SYMBOL_SPECIFIC"
                        }
                    }
                    # Save to target-first structure
                    with open(meta_path_repro, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.info(f"  💾 Basic metadata saved: {meta_path_repro}")
                    
                    # Also save to legacy location
                    with open(meta_path_legacy, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    logger.debug(f"  💾 Basic metadata saved to legacy location: {meta_path_legacy}")
                    
                    # Track reproducibility for symbol-specific model
                    if output_dir:
                        try:
                            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
                            from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                            
                            module_output_dir = Path(output_dir)
                            if module_output_dir.name != 'training_results':
                                module_output_dir = module_output_dir.parent / 'training_results'
                            
                            tracker = ReproducibilityTracker(
                                output_dir=module_output_dir,
                                search_previous_runs=True
                            )
                            
                            # Extract metrics using clean structure
                            strategy_manager = model_result.get('strategy_manager')
                            metrics = {}
                            if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
                                cv_scores = strategy_manager.cv_scores
                                if cv_scores and len(cv_scores) > 0:
                                    # Build clean training metrics for CV scores
                                    try:
                                        from TRAINING.ranking.predictability.metrics_schema import build_clean_training_metrics
                                        from TRAINING.common.utils.task_types import TaskType
                                        
                                        # Determine task type from model_result or route
                                        task_type = TaskType.REGRESSION  # Default
                                        if model_result.get('task_type'):
                                            task_type_str = model_result.get('task_type')
                                            if isinstance(task_type_str, str):
                                                task_type = TaskType[task_type_str.upper()]
                                        elif 'route' in locals() and route:
                                            # Infer from route if available
                                            if 'classification' in str(route).lower():
                                                task_type = TaskType.BINARY_CLASSIFICATION
                                        
                                        # Build clean metrics dict
                                        cv_metrics_result = {
                                            "val_auc": float(np.mean(cv_scores)),
                                            "val_auc_std": float(np.std(cv_scores)),
                                        }
                                        
                                        metrics = build_clean_training_metrics(
                                            model_result=cv_metrics_result,
                                            task_type=task_type,
                                            view="SYMBOL_SPECIFIC",  # Per-symbol training
                                            n_features=len(feature_names) if feature_names else None,
                                            n_samples=len(X) if 'X' in locals() else None,
                                        )
                                        metrics["metric_name"] = "CV Score"  # Add metadata field
                                    except Exception as e:
                                        # Fallback to flat structure if clean builder fails
                                        logger.debug(f"Failed to build clean training metrics, using fallback: {e}")
                                        metrics = {
                                            "metric_name": "CV Score",
                                            "auc": float(np.mean(cv_scores)),
                                            "std_score": float(np.std(cv_scores)),
                                            "composite_score": float(np.mean(cv_scores))
                                        }
                            
                            if metrics:
                                cohort_metadata = extract_cohort_metadata(
                                    X=X,
                                    symbols=[symbol],
                                    time_vals=time_vals,
                                    mtf_data=symbol_mtf_data,
                                    min_cs=1,
                                    max_cs_samples=max_cs_samples
                                )
                                cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
                                
                                # Compute universe_sig from FULL universe (mtf_data.keys()), NOT [symbol]
                                # This ensures consistent universe_sig across all writes in this run
                                from TRAINING.orchestration.utils.run_context import compute_universe_signature
                                from TRAINING.orchestration.utils.scope_resolution import (
                                    WriteScope, ScopePurpose, Stage
                                )
                                full_universe = list(mtf_data.keys()) if mtf_data else [symbol]
                                universe_sig = compute_universe_signature(full_universe)
                                
                                # Create WriteScope for type-safe scope handling
                                scope = WriteScope.for_symbol_specific(
                                    universe_sig=universe_sig,
                                    symbol=symbol,
                                    stage=Stage.TRAINING,
                                    purpose=ScopePurpose.FINAL
                                )
                                
                                metrics_with_cohort = {**metrics, **cohort_metrics}
                                additional_data_with_cohort = {
                                    "strategy": strategy,
                                    "n_features": len(feature_names) if feature_names else 0,
                                    "model_family": family,
                                    "route": route,  # Add route information
                                    "selected_features": feature_names[:20] if feature_names else [],  # FIX: Track top selected features from FS
                                    **cohort_additional_data
                                }
                                
                                # Use WriteScope to populate additional_data correctly
                                scope.to_additional_data(additional_data_with_cohort)
                                
                                # FIX: Compute prediction fingerprint for Training stage determinism tracking
                                prediction_fingerprint = None
                                try:
                                    if strategy_manager and hasattr(strategy_manager, 'models'):
                                        from TRAINING.common.utils.prediction_hashing import compute_prediction_fingerprint_for_model
                                        models = strategy_manager.models
                                        pred_hashes = []
                                        for model_name, model in models.items():
                                            if model is not None and hasattr(model, 'predict'):
                                                try:
                                                    # Use subset of X for efficiency
                                                    X_subset = X[:min(1000, len(X))]
                                                    preds = model.predict(X_subset)
                                                    proba = model.predict_proba(X_subset) if hasattr(model, 'predict_proba') else None
                                                    fp = compute_prediction_fingerprint_for_model(
                                                        preds=preds,
                                                        proba=proba,
                                                        model=model,
                                                        task_type="REGRESSION",  # Will be normalized inside
                                                        X=X_subset,
                                                        strict_mode=False,
                                                    )
                                                    if fp and fp.get('prediction_hash'):
                                                        pred_hashes.append(fp['prediction_hash'])
                                                except Exception as fp_e:
                                                    logger.debug(f"Prediction fingerprint failed for {model_name}: {fp_e}")
                                        if pred_hashes:
                                            import hashlib
                                            combined = hashlib.sha256('|'.join(sorted(pred_hashes)).encode()).hexdigest()
                                            prediction_fingerprint = {'prediction_hash': combined}
                                            logger.debug(f"✅ Computed prediction_fingerprint for {family}: {combined[:12]}...")
                                except Exception as pf_e:
                                    logger.debug(f"Prediction fingerprint computation failed: {pf_e}")
                                
                                tracker.log_comparison(
                                    stage=scope.stage.value,
                                    target=f"{target}:{symbol}:{family}",
                                    metrics=metrics_with_cohort,
                                    additional_data=additional_data_with_cohort,
                                    symbol=scope.symbol,
                                    view=scope.view.value,
                                    run_identity=effective_run_identity,  # SST: Pass identity for reproducibility
                                    prediction_fingerprint=prediction_fingerprint,  # FIX: Add prediction fingerprint
                                )
                        except Exception as e:
                            logger.warning(f"Reproducibility tracking failed for {family}:{target}:{symbol}: {e}")
                
                # Store symbol results
                if symbol_results:
                    if target not in results['models']:
                        results['models'][target] = {}
                    results['models'][target][symbol] = symbol_results
                    logger.info(f"  ✅ Completed {symbol}: {len(symbol_results)} models trained")
                else:
                    logger.warning(f"  ⚠️ No models trained for {target}:{symbol}")
            
            # After all symbols processed, create aggregated snapshots per model family
            try:
                from TRAINING.training_strategies.reproducibility.io import (
                    create_aggregated_training_snapshot,
                    load_training_snapshot,
                    get_training_snapshot_dir,
                    compute_cohort_id_from_metadata
                )
                from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata
                
                # Collect all per-symbol snapshots by model family
                per_family_snapshots = {}
                aggregated_cohort_id = None
                
                # Try to compute cohort_id from first symbol's metadata
                if target_feat_data and len(target_feat_data) > 0:
                    first_symbol = list(target_feat_data.keys())[0]
                    if first_symbol in mtf_data:
                        try:
                            symbol_mtf_data = {first_symbol: mtf_data[first_symbol]}
                            # Use a representative sample to compute cohort metadata
                            # We'll use the first symbol's data as a proxy
                            temp_X, temp_y, _, _, _, _, temp_time_vals, _ = prepare_training_data_cross_sectional(
                                symbol_mtf_data, target, feature_names=list(target_feat_data[first_symbol])[:10] if target_feat_data[first_symbol] else None,
                                min_cs=1, max_cs_samples=max_cs_samples, routing_decisions=routing_decisions
                            )
                            if temp_X is not None and len(temp_X) > 0:
                                # Compute cohort metadata for aggregation
                                temp_cohort_metadata = extract_cohort_metadata(
                                    X=temp_X,
                                    symbols=list(target_feat_data.keys()),
                                    time_vals=temp_time_vals,
                                    mtf_data=mtf_data,
                                    min_cs=1,
                                    max_cs_samples=max_cs_samples
                                )
                                aggregated_cohort_id = compute_cohort_id_from_metadata(temp_cohort_metadata, view="SYMBOL_SPECIFIC")
                        except Exception as e:
                            logger.debug(f"Failed to compute aggregated cohort_id: {e}")
                
                # Load all per-symbol snapshots
                for symbol in target_feat_data.keys():
                    # Find snapshots for this symbol
                    symbol_snapshot_dir = get_training_snapshot_dir(
                        output_dir=base_run_dir,
                        target=target,
                        view="SYMBOL_SPECIFIC",
                        symbol=symbol,
                        stage="TRAINING",
                    )
                    
                    # Look for cohort directories
                    if symbol_snapshot_dir.exists():
                        for cohort_dir in symbol_snapshot_dir.iterdir():
                            if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                                snapshot_path = cohort_dir / "training_snapshot.json"
                                if snapshot_path.exists():
                                    snapshot = load_training_snapshot(snapshot_path)
                                    if snapshot:
                                        family = snapshot.model_family
                                        if family not in per_family_snapshots:
                                            per_family_snapshots[family] = []
                                        per_family_snapshots[family].append(snapshot)
                                        # Use cohort_id from first snapshot if not computed
                                        if aggregated_cohort_id is None:
                                            aggregated_cohort_id = cohort_dir.name.replace("cohort=", "")
                
                # Create aggregated snapshots for each model family
                for family, snapshots in per_family_snapshots.items():
                    if len(snapshots) > 0:
                        try:
                            aggregated = create_aggregated_training_snapshot(
                                snapshots=snapshots,
                                target=target,
                                model_family=family,
                                output_dir=base_run_dir,
                                view="SYMBOL_SPECIFIC",
                                cohort_id=aggregated_cohort_id,
                            )
                            if aggregated:
                                logger.info(f"✅ Created aggregated snapshot for {target}/{family} ({len(snapshots)} symbols)")
                        except Exception as e:
                            logger.debug(f"Failed to create aggregated snapshot for {target}/{family}: {e}")
            except Exception as e:
                logger.debug(f"Failed to create aggregated snapshots for {target}: {e}")
            
            # Skip the cross-sectional training path for SYMBOL_SPECIFIC
            continue
        
        # For BOTH route, check if we need symbol-specific training after CS training
        # Store original target_feat_data for BOTH route to restore symbol-specific features later
        both_route_symbol_features = None
        if route == 'BOTH' and isinstance(target_feat_data, dict) and 'symbol_specific' in target_feat_data:
            symbol_specific_features = target_feat_data.get('symbol_specific', {})
            if symbol_specific_features and isinstance(symbol_specific_features, dict) and len(symbol_specific_features) > 0:
                # Check routing plan to see which symbols need symbol-specific training
                winner_symbols = route_info.get('winner_symbols', [])
                if not winner_symbols:
                    # If no winner_symbols specified, check symbol_specific section of routing plan
                    sym_info = route_info.get('symbol_specific', {})
                    if isinstance(sym_info, dict):
                        # Get all symbols that have symbol-specific routing enabled
                        winner_symbols = [sym for sym, sym_data in sym_info.items() 
                                         if isinstance(sym_data, dict) and sym_data.get('route') == 'ENABLED']
                
                if winner_symbols:
                    # Filter symbol-specific features to only those symbols that need training
                    filtered_symbol_features = {sym: symbol_specific_features[sym] 
                                               for sym in winner_symbols 
                                               if sym in symbol_specific_features}
                    if filtered_symbol_features:
                        both_route_symbol_features = filtered_symbol_features
                        logger.info(f"BOTH route: Will train symbol-specific models for {len(filtered_symbol_features)} symbols after CS training: {list(filtered_symbol_features.keys())}")
        
        # Cross-sectional training (for CROSS_SECTIONAL or BOTH routes)
        # Note: BLOCKED targets are skipped earlier in the loop
        # Apply runtime quarantine (dominance quarantine confirmed features)
        if selected_features and output_dir:
            try:
                from TRAINING.ranking.utils.dominance_quarantine import load_confirmed_quarantine
                from pathlib import Path
                # Determine view from route
                route_info = routing_decisions.get(target, {}) if routing_decisions else {}
                route = route_info.get('route', 'CROSS_SECTIONAL')
                view_for_quarantine = "CROSS_SECTIONAL" if route in ['CROSS_SECTIONAL', 'BOTH'] else "SYMBOL_SPECIFIC"
                
                output_dir_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
                runtime_quarantine = load_confirmed_quarantine(
                    output_dir=output_dir_path,
                    target=target,
                    view=view_for_quarantine,
                    symbol=None  # For CROSS_SECTIONAL, symbol is None
                )
                if runtime_quarantine:
                    selected_features = [f for f in selected_features if f not in runtime_quarantine]
                    logger.info(f"🔒 Applied runtime quarantine for {target}: Removed {len(runtime_quarantine)} confirmed leaky features ({len(selected_features)} remaining)")
            except Exception as e:
                logger.debug(f"Could not load runtime quarantine for {target}: {e}")
        
        X, y, feature_names, symbols, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
            mtf_data, target, feature_names=selected_features, min_cs=min_cs, max_cs_samples=max_cs_samples, routing_decisions=routing_decisions
        )
        prep_elapsed = _t.time() - prep_start
        print(f"✅ Data preparation completed in {prep_elapsed:.2f}s")  # Debug print
        
        if X is None:
            logger.error(f"❌ Failed to prepare data for target {target}")
            results['failed_targets'].append(target)
            results['failed_reasons'][target] = "Data preparation returned None (likely all features became NaN after coercion)"
            continue
        
        # CRITICAL: Validate feature count collapse (requested vs allowed vs used)
        requested_count = len(selected_features) if selected_features else 0
        allowed_count = len(feature_names) if feature_names else 0
        used_count = X.shape[1] if X is not None else 0
        
        # Load threshold from config (default: 0.5 = 50%)
        try:
            from CONFIG.config_loader import get_cfg
            collapse_threshold = get_cfg("training.feature_collapse_threshold", default=0.5, config_name="training_config")
        except Exception:
            collapse_threshold = 0.5  # Default: 50%
        
        if requested_count > 0 and used_count < requested_count * collapse_threshold:
            # Calculate breakdown of where features were dropped
            dropped_by_registry = max(0, requested_count - allowed_count)
            dropped_after_allowed = max(0, allowed_count - used_count)
            
            # Try to get more detailed breakdown from feature audit report if available
            audit_details = None
            try:
                from pathlib import Path
                audit_dir = Path(output_dir) / "artifacts" / "feature_audits"
                audit_file = audit_dir / f"{target}_audit_summary.csv"
                if audit_file.exists():
                    import pandas as pd
                    audit_df = pd.read_csv(audit_file)
                    # Get counts by stage
                    stage_counts = {}
                    for _, row in audit_df.iterrows():
                        stage = row.get('stage', 'unknown')
                        count = row.get('count', 0)
                        if stage not in stage_counts:
                            stage_counts[stage] = 0
                        stage_counts[stage] = count
                    
                    audit_details = {
                        'dropped_by_registry': stage_counts.get('registry_filter', 0),
                        'dropped_by_dtype_nan': stage_counts.get('nan_drop', 0) + stage_counts.get('non_numeric_drop', 0),
                        'dropped_after_kept': stage_counts.get('final_filter', 0),
                        'audit_report': str(audit_file)
                    }
            except Exception as e:
                logger.debug(f"Could not load feature audit report for detailed breakdown: {e}")
            
            # Build detailed warning message
            retained_pct = used_count/requested_count*100 if requested_count > 0 else 0
            warning_msg = (
                f"⚠️ Feature count collapse for {target}: "
                f"requested={requested_count} → allowed={allowed_count} → used={used_count} "
                f"({retained_pct:.1f}% retained, threshold={collapse_threshold*100:.0f}%)."
            )
            
            # Add breakdown details
            if audit_details:
                warning_msg += (
                    f"\n   Breakdown: dropped_by_registry={audit_details['dropped_by_registry']}, "
                    f"dropped_by_dtype/nan={audit_details['dropped_by_dtype_nan']}, "
                    f"dropped_after_kept={audit_details['dropped_after_kept']}"
                )
                warning_msg += f"\n   Detailed audit report: {audit_details['audit_report']}"
            else:
                # Fallback to calculated breakdown
                warning_msg += (
                    f"\n   Breakdown (estimated): dropped_by_registry={dropped_by_registry}, "
                    f"dropped_after_allowed={dropped_after_allowed} "
                    f"(includes dtype/nan filtering and other drops)"
                )
            
            warning_msg += "\n   This may indicate data quality issues or overly aggressive filtering."
            
            logger.warning(warning_msg)
        
        logger.info(
            f"📊 Feature pipeline for {target}: "
            f"requested={requested_count} allowed={allowed_count} used={used_count} "
            f"(shape: X={X.shape if X is not None else 'None'})"
        )
        
        # Extract routing info (now in slot 7)
        if isinstance(routing_meta, dict) and 'spec' in routing_meta:
            logger.info(f"[Routing] Using task spec: {routing_meta['spec']}")
        else:
            # Fallback: old code path without routing
            routing_meta = {
                'target': target,
                'spec': TaskSpec('regression', 'regression', ['rmse', 'mae']),
                'sample_weights': None,
                'group_sizes': None
            }
        
        # Apply row cap to prevent OOM
        if max_rows_train and len(X) > max_rows_train:
            # Use deterministic seed from determinism system
            from TRAINING.common.determinism import BASE_SEED, stable_seed_from
            # Generate seed based on target for deterministic downsampling
            downsample_seed = stable_seed_from([target, 'downsample']) if target else (BASE_SEED if BASE_SEED is not None else 42)
            rng = np.random.RandomState(downsample_seed)
            idx = rng.choice(len(X), max_rows_train, replace=False)
            X, y = X[idx], y[idx]
            if time_vals is not None: time_vals = time_vals[idx]
            if symbols is not None: symbols = symbols[idx]
            logger.info(f"✂️ Downsampled to max_rows_train={max_rows_train}")
        
        # Store cohort metadata context for later use in reproducibility tracking
        # Store AFTER downsampling (if any) so we track the actual training cohort
        # These will be used to extract cohort metadata at the end of training
        cohort_context = {
            'X': X,  # This is the actual training data (may be downsampled)
            'y': y,
            'time_vals': time_vals,
            'symbols': symbols,  # This is the actual training symbols (may be downsampled)
            'mtf_data': mtf_data,  # Keep original mtf_data for date range extraction
            'min_cs': min_cs,
            'max_cs_samples': max_cs_samples
        }
        
        target_results = {}
        
        # CRITICAL: Order families to prevent cross-lib thread pollution
        # Run CPU-GBDT families FIRST, then TF/XGB families
        FAMILY_ORDER = [
            "lightgbm", "quantile_lightgbm", "reward_based", "xgboost",  # CPU tree learners first (normalized)
            "mlp", "neural_network", "ensemble", "change_point", "ngboost", "gmm_regime", "ftrl_proximal", "vae", "gan", "meta_learning", "multi_task"  # Others (normalized)
        ]
        
        # Reorder validated families to prevent thread pollution
        ordered_families = []
        for priority_family in FAMILY_ORDER:
            if priority_family in validated_families:
                ordered_families.append(priority_family)
        # Add any remaining families not in the priority list
        for family in validated_families:
            if family not in ordered_families:
                ordered_families.append(family)
        
        logger.info(f"🔄 Reordered families to prevent thread pollution: {ordered_families}")
        print(f"🔄 Reordered families to prevent thread pollution: {ordered_families}")
        
        # Track training results per family
        family_results = {
            'trained_ok': [],
            'failed': [],
            'skipped': []
        }
        
        # Track family status for diagnostics
        family_status = {family: {"attempted": False, "saved": False, "error": None} for family in ordered_families}
        logger.info(f"🎯 Attempting to train {len(ordered_families)} model families: {ordered_families}")
        
        for i, family in enumerate(ordered_families, 1):
            logger.info(f"🎯 [{i}/{len(ordered_families)}] Training {family} for {target}")
            logger.info(f"📊 Data shape: X={X.shape}, y={y.shape}")
            logger.info(f"🔧 Strategy: {strategy}")
            print(f"🎯 [{i}/{len(ordered_families)}] Training {family} for {target}")  # Also print to stdout
            print(f"DEBUG: About to call train_model_comprehensive for {family}")  # Debug print
            
            try:
                # CRITICAL: family is already normalized from preflight (validated_families)
                # No need to normalize again - use directly
                normalized_family = family
                
                # Check family capabilities
                if normalized_family not in FAMILY_CAPS:
                    logger.warning(f"Model family {family} (normalized: {normalized_family}) not in capabilities map. Skipping.")
                    family_results['skipped'].append((family, normalized_family, "not_in_capabilities_map"))
                    continue
                
                caps = FAMILY_CAPS[normalized_family]
                logger.info(f"📋 Family capabilities: {caps}")
                
                # Check task type compatibility (skip incompatible families)
                if 'routing_meta' in locals() and routing_meta and routing_meta.get('spec'):
                    from TRAINING.training_strategies.utils import is_family_compatible
                    task_type = routing_meta['spec'].task
                    compatible, skip_reason = is_family_compatible(normalized_family, task_type)
                    if not compatible:
                        logger.info(f"⏭️ Skipping {family}: {skip_reason} (task={task_type})")
                        family_results['skipped'].append((family, normalized_family, skip_reason))
                        continue
                
                # Check TensorFlow dependency (skip for torch families)
                if caps.get("backend") == "torch":
                    pass  # never gate on TF for torch families
                elif caps.get("needs_tf"):
                    # For isolated models, let child process handle TF availability
                    # For in-process models, check TF availability in parent
                    from TRAINING.common.runtime_policy import should_isolate
                    if not should_isolate(normalized_family) and not tf_available():
                        logger.warning(f"TensorFlow missing → skipping {normalized_family}")
                        family_results['skipped'].append((family, normalized_family, "tensorflow_missing"))
                        continue
                    # If isolated, child process will handle TF import/initialization
                
                # Check NGBoost dependency
                if normalized_family == "NGBoost" and not ngboost_available():
                    logger.warning(f"NGBoost missing → skipping {normalized_family}")
                    family_results['skipped'].append((family, normalized_family, "ngboost_missing"))
                    continue
                
                logger.info(f"🚀 [{family}] Starting {family} training...")
                start_time = _now()
                
                # Train model using modular system with routing metadata
                # Use normalized family name for consistency
                try:
                    model_result = train_model_comprehensive(
                        normalized_family, X, y, target, strategy, feature_names, caps, routing_meta
                    )
                    elapsed = _now() - start_time
                    logger.info(f"⏱️ [{family}] {family} training completed in {elapsed:.2f} seconds")
                    if model_result is None:
                        logger.warning(f"⚠️ [{family}] train_model_comprehensive returned None")
                except Exception as train_err:
                    elapsed = _now() - start_time
                    family_status[family]["attempted"] = True
                    family_status[family]["error"] = str(train_err)
                    logger.error(f"❌ [{normalized_family}] Training failed after {elapsed:.2f} seconds: {train_err}")
                    logger.exception(f"Full traceback for {normalized_family}:")
                    family_results['failed'].append((family, normalized_family, str(train_err)))
                    # Don't re-raise - continue with next family
                    continue
                
                if model_result is not None and model_result.get('success', False):
                    family_status[family]["attempted"] = True
                    target_results[normalized_family] = model_result
                    family_results['trained_ok'].append((family, normalized_family))
                    
                    # Track reproducibility: compare to previous training run
                    if output_dir and model_result.get('success', False):
                        try:
                            from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
                            from pathlib import Path
                            
                            # Ensure output_dir is a Path object
                            output_dir_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
                            
                            # Use module-specific directory for reproducibility log
                            # output_dir is typically: output_dir_YYYYMMDD_HHMMSS/training_results/
                            # We want to store in training_results/ subdirectory for this module
                            if output_dir_path.name == 'training_results' or (output_dir_path.parent / 'training_results').exists():
                                # Already in or can find training_results subdirectory
                                if output_dir_path.name != 'training_results':
                                    module_output_dir = output_dir_path.parent / 'training_results'
                                else:
                                    module_output_dir = output_dir_path
                            else:
                                # Fallback: use output_dir directly (for standalone runs)
                                module_output_dir = output_dir_path
                            
                            tracker = ReproducibilityTracker(
                                output_dir=module_output_dir,
                                search_previous_runs=True  # Search for previous runs in parent directories
                            )
                            
                            # Extract metrics from strategy_manager if available
                            strategy_manager = model_result.get('strategy_manager')
                            metrics = {}
                            if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
                                cv_scores = strategy_manager.cv_scores
                                if cv_scores and len(cv_scores) > 0:
                                    metrics = {
                                        "metric_name": "CV Score",
                                        "auc": float(np.mean(cv_scores)),
                                        "std_score": float(np.std(cv_scores)),
                                        "composite_score": float(np.mean(cv_scores))
                                    }
                            
                            # If we have metrics, log comparison
                            if metrics:
                                # Extract cohort metadata using unified extractor
                                from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                                
                                # Extract cohort metadata from stored context (X, symbols, time_vals, mtf_data from prepare_training_data_cross_sectional)
                                # cohort_context is defined earlier in the function after data preparation (and downsampling if any)
                                if 'cohort_context' in locals() and cohort_context:
                                    # For cohort identification, use the stored X (represents the training cohort)
                                    # X_train from CV is a subset, but we want consistent cohort_id across folds
                                    # So we use the full training X from cohort_context
                                    cohort_metadata = extract_cohort_metadata(
                                        X=cohort_context.get('X'),
                                        symbols=cohort_context.get('symbols'),
                                        time_vals=cohort_context.get('time_vals'),
                                        mtf_data=cohort_context.get('mtf_data'),
                                        min_cs=cohort_context.get('min_cs'),
                                        max_cs_samples=cohort_context.get('max_cs_samples')
                                    )
                                else:
                                    # Fallback: try to extract from function variables (shouldn't happen if cohort_context is set)
                                    cohort_metadata = extract_cohort_metadata(
                                        X=X_train if 'X_train' in locals() else None,
                                        symbols=symbols if 'symbols' in locals() else (list(mtf_data.keys()) if 'mtf_data' in locals() and mtf_data else None),
                                        mtf_data=mtf_data if 'mtf_data' in locals() else None,
                                        min_cs=min_cs if 'min_cs' in locals() else None,
                                        max_cs_samples=max_cs_samples if 'max_cs_samples' in locals() else None
                                    )
                                
                                # Format for reproducibility tracker
                                cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
                                
                                # Merge with existing metrics and additional_data
                                metrics_with_cohort = {
                                    **metrics,
                                    **cohort_metrics  # Adds n_effective_cs if available
                                }
                                
                                # Compute universe_sig from FULL universe (mtf_data.keys())
                                from TRAINING.orchestration.utils.run_context import compute_universe_signature
                                from TRAINING.orchestration.utils.scope_resolution import (
                                    WriteScope, ScopePurpose, Stage
                                )
                                
                                # Get symbols from mtf_data for universe_sig computation
                                tracking_symbols = list(mtf_data.keys()) if mtf_data else []
                                universe_sig = compute_universe_signature(tracking_symbols) if tracking_symbols else None
                                
                                # Create WriteScope for type-safe scope handling
                                # This enforces CS has no symbol and validates invariants
                                if universe_sig:
                                    scope = WriteScope.for_cross_sectional(
                                        universe_sig=universe_sig,
                                        stage=Stage.TRAINING,
                                        purpose=ScopePurpose.FINAL
                                    )
                                else:
                                    scope = None  # Fallback if no universe_sig
                                
                                additional_data_with_cohort = {
                                    "strategy": strategy,
                                    "n_features": len(feature_names) if feature_names else 0,
                                    "model_family": family,  # Add model family for routing
                                    "route": route,  # Add route information
                                    "selected_features": feature_names[:20] if feature_names else [],  # FIX: Track top selected features from FS
                                    **cohort_additional_data  # Adds n_symbols, date_range, cs_config if available
                                }
                                
                                # Use WriteScope to populate additional_data correctly
                                if scope:
                                    scope.to_additional_data(additional_data_with_cohort)
                                
                                # CRITICAL: Adapt additional_data to ensure string/Enum safety
                                # This prevents 'str' object has no attribute 'name' errors
                                from TRAINING.common.utils.sst_contract import tracker_input_adapter
                                
                                # Normalize family name for consistency
                                from TRAINING.common.utils.sst_contract import normalize_family
                                family_normalized = normalize_family(family)
                                
                                # Adapt additional_data (convert Enum-like objects to strings)
                                additional_data_adapted = {}
                                for key, value in additional_data_with_cohort.items():
                                    if key in ['task_type', 'objective', 'stage', 'view', 'family']:
                                        # These fields might be Enum-like objects
                                        additional_data_adapted[key] = tracker_input_adapter(value, key)
                                    else:
                                        additional_data_adapted[key] = value
                                
                                # FIX: Compute prediction fingerprint for Training stage determinism tracking
                                prediction_fingerprint_cs = None
                                try:
                                    if model_result and model_result.get('success'):
                                        from TRAINING.common.utils.prediction_hashing import compute_prediction_fingerprint_for_model
                                        strategy_mgr = model_result.get('strategy_manager')
                                        if strategy_mgr and hasattr(strategy_mgr, 'models'):
                                            pred_hashes = []
                                            X_for_fp = X if 'X' in locals() else None
                                            for model_name, model in strategy_mgr.models.items():
                                                if model is not None and hasattr(model, 'predict') and X_for_fp is not None:
                                                    try:
                                                        X_subset = X_for_fp[:min(1000, len(X_for_fp))]
                                                        preds = model.predict(X_subset)
                                                        proba = model.predict_proba(X_subset) if hasattr(model, 'predict_proba') else None
                                                        fp = compute_prediction_fingerprint_for_model(
                                                            preds=preds, proba=proba, model=model,
                                                            task_type="REGRESSION", X=X_subset, strict_mode=False,
                                                        )
                                                        if fp and fp.get('prediction_hash'):
                                                            pred_hashes.append(fp['prediction_hash'])
                                                    except Exception:
                                                        pass
                                            if pred_hashes:
                                                import hashlib
                                                combined = hashlib.sha256('|'.join(sorted(pred_hashes)).encode()).hexdigest()
                                                prediction_fingerprint_cs = {'prediction_hash': combined}
                                except Exception:
                                    pass
                                
                                tracker.log_comparison(
                                    stage=scope.stage.value if scope else "model_training",
                                    target=f"{target}:{family_normalized}",
                                    metrics=metrics_with_cohort,
                                    additional_data=additional_data_adapted,
                                    model_family=family_normalized,
                                    view=scope.view.value if scope else "CROSS_SECTIONAL",
                                    run_identity=effective_run_identity,  # SST: Pass identity for reproducibility
                                    prediction_fingerprint=prediction_fingerprint_cs,  # FIX: Add prediction fingerprint
                                )
                        except Exception as e:
                            logger.warning(f"Reproducibility tracking failed for {family}:{target}: {e}")
                            import traceback
                            logger.debug(f"Reproducibility tracking traceback: {traceback.format_exc()}")
                    
                    # Save model to canonical target-first structure using ArtifactPaths
                    # Ensure output_dir is a Path object
                    output_dir_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
                    
                    # Determine view from route (route is defined earlier in function scope)
                    # For CROSS_SECTIONAL or BOTH routes, use CROSS_SECTIONAL view
                    view = "CROSS_SECTIONAL"  # Default for CROSS_SECTIONAL route
                    if route in ['CROSS_SECTIONAL', 'BOTH']:
                        view = "CROSS_SECTIONAL"
                    elif route == 'SYMBOL_SPECIFIC':
                        view = "SYMBOL_SPECIFIC"
                        logger.warning(f"SYMBOL_SPECIFIC route in CROSS_SECTIONAL saving block - this should not happen")
                    
                    # Find base run directory (parent of training_results/ if output_dir is training_results/)
                    base_run_dir = output_dir_path
                    if output_dir_path.name == 'training_results':
                        base_run_dir = output_dir_path.parent
                    elif (output_dir_path.parent / 'training_results').exists():
                        base_run_dir = output_dir_path.parent
                    
                    # Use ArtifactPaths to get canonical model directory
                    from TRAINING.orchestration.utils.artifact_paths import ArtifactPaths
                    model_dir = ArtifactPaths.model_dir(
                        run_root=base_run_dir,
                        target=target,
                        view=view,
                        family=family,
                        symbol=None  # CROSS_SECTIONAL doesn't have symbol
                    )
                    model_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"💾 Saving {family} model to: {model_dir} (view={view}, route={route})")
                    
                    try:
                        # Get the trained model from strategy manager
                        strategy_manager = model_result['strategy_manager']
                        models = strategy_manager.models
                        
                        # Import model wrapper for saving compatibility
                        from common.model_wrapper import wrap_model_for_saving, get_model_saving_info
                        
                        # Save each model component (same as original)
                        for model_name, model in models.items():
                            # Wrap model for saving compatibility
                            wrapped_model = wrap_model_for_saving(model, family)
                            
                            # Get saving info
                            save_info = get_model_saving_info(wrapped_model)
                            logger.info(f"💾 Saving {family} model: {save_info}")
                            
                            # Determine file extensions based on model type
                            # Save to canonical location using ArtifactPaths
                            if save_info['is_lightgbm']:  # LightGBM
                                model_path = ArtifactPaths.model_file(model_dir, family, extension='txt')
                                wrapped_model.save_model(str(model_path))
                                logger.info(f"💾 LightGBM model saved: {model_path}")
                                
                            elif save_info['is_tensorflow']:  # TensorFlow/Keras
                                model_path = ArtifactPaths.model_file(model_dir, family, extension='keras')
                                wrapped_model.save(str(model_path))
                                logger.info(f"💾 Keras model saved: {model_path}")
                                
                            elif save_info['is_pytorch']:  # PyTorch models
                                model_path = ArtifactPaths.model_file(model_dir, family, extension='pt')
                                import torch, json
                                
                                # Extract the actual PyTorch model from wrapped_model
                                # wrapped_model should contain the PyTorch model
                                if hasattr(wrapped_model, 'core') and hasattr(wrapped_model.core, 'model'):
                                    torch_model = wrapped_model.core.model
                                elif hasattr(wrapped_model, 'model'):
                                    torch_model = wrapped_model.model
                                else:
                                    torch_model = wrapped_model
                                
                                # Save state dict + metadata
                                torch.save({
                                    "state_dict": torch_model.state_dict(),
                                    "config": getattr(wrapped_model, "config", {}),
                                    "arch": family,
                                    "input_shape": X.shape
                                }, str(model_path))
                                logger.info(f"💾 PyTorch model saved: {model_path}")
                                
                            else:  # Scikit-learn models
                                model_path = ArtifactPaths.model_file(model_dir, family, extension='joblib')
                                wrapped_model.save(str(model_path))
                                logger.info(f"💾 Scikit-learn model saved: {model_path}")
                            
                            # Save preprocessors if available
                            if wrapped_model.scaler is not None:
                                scaler_path = ArtifactPaths.scaler_file(model_dir, family)
                                joblib.dump(wrapped_model.scaler, scaler_path)
                                logger.info(f"💾 Scaler saved: {scaler_path}")
                                
                            if wrapped_model.imputer is not None:
                                imputer_path = ArtifactPaths.imputer_file(model_dir, family)
                                joblib.dump(wrapped_model.imputer, imputer_path)
                                logger.info(f"💾 Imputer saved: {imputer_path}")
                            # Note: If wrapped_model.imputer is None, no imputer was used/needed
                            
                            # CRITICAL: Define _pkg_ver BEFORE conditional blocks to avoid "referenced before assignment"
                            def _pkg_ver(pkg_name):
                                try:
                                    import importlib.metadata
                                    return importlib.metadata.version(pkg_name)
                                except:
                                    try:
                                        return __import__(pkg_name).__version__
                                    except:
                                        return "unknown"
                            
                            # Save metadata (match original format exactly)
                            if save_info['is_lightgbm']:  # LightGBM - JSON format
                                # Save metadata to canonical location using ArtifactPaths
                                meta_path = ArtifactPaths.metadata_file(model_dir)
                                import json
                                metadata = {
                                    "family": family,
                                    "target": target,
                                    "route": "CROSS_SECTIONAL",  # Add route indicator
                                    "view": "CROSS_SECTIONAL",  # Add view indicator
                                    "min_cs": min_cs,
                                    "features": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                                    "feature_names": feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names),
                                    "n_features": len(feature_names),
                                    "package_versions": {
                                        "numpy": _pkg_ver("numpy"),
                                        "pandas": _pkg_ver("pandas"),
                                        "sklearn": _pkg_ver("sklearn"),
                                        "lightgbm": _pkg_ver("lightgbm"),
                                        "xgboost": _pkg_ver("xgboost"),
                                        "tensorflow": _pkg_ver("tensorflow"),
                                        "ngboost": _pkg_ver("ngboost"),
                                    },
                                    "cli_args": {
                                        "min_cs": min_cs,
                                        "max_cs_samples": max_cs_samples,
                                        "cs_normalize": "per_ts_split",
                                        "cs_block": 32,
                                        "cs_winsor_p": 0.01,
                                        "cs_ddof": 1,
                                        "batch_id": 0,
                                        "families": [family]
                                    },
                                    "n_rows_train": len(X),
                                    "n_rows_val": 0,
                                    "train_timestamps": int(np.unique(time_vals).size) if time_vals is not None else len(X),
                                    "val_timestamps": 0,
                                    "time_col": None,
                                    "val_start_ts": None,
                                    "metrics": {
                                        "mean_IC": 0.0,
                                        "mean_RankIC": 0.0,
                                        "IC_IR": 0.0,
                                        "n_times": 0,
                                        "hit_rate": 0.0,
                                        "skipped_timestamps": 0,
                                        "total_timestamps": 0
                                    },
                                    "best": {
                                        "best_iteration": 0
                                    },
                                    "params_used": None,
                                    "learner_params": {},
                                    "cs_norm": {
                                        "mode": "per_ts_split",
                                        "p": 0.01,
                                        "ddof": 1,
                                        "method": "quantile"
                                    },
                                    "rank_method": "scipy_dense",
                                    "feature_importance": {}
                                }
                                # Save metadata to canonical location
                                with open(meta_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                logger.info(f"💾 Metadata saved: {meta_path}")
                                    
                            else:  # TensorFlow/Scikit-learn - joblib format
                                # Save metadata to canonical location using ArtifactPaths
                                meta_path = ArtifactPaths.metadata_file(model_dir)
                                metadata = {
                                    "family": family,
                                    "target": target,
                                    "features": tuple(feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names))
                                }
                                joblib.dump(metadata, meta_path)
                                logger.info(f"💾 Metadata saved: {meta_path}")
                                
                    except Exception as e:
                        family_status[family]["error"] = f"Save failed: {str(e)}"
                        logger.warning(f"Failed to save model {family}_{target}: {e}")
                    else:
                        # If no exception, mark as saved
                        family_status[family]["saved"] = True
                        logger.info(f"✅ {family} model saved successfully for {target}")
                        
                        # SST: Create and save TrainingSnapshot for full parity tracking
                        try:
                            from TRAINING.training_strategies.reproducibility import create_and_save_training_snapshot
                            
                            # Get model path that was just saved
                            saved_model_path = str(model_path) if 'model_path' in locals() else None
                            
                            # Get features used from target_features
                            training_features = None
                            if target_features and target in target_features:
                                tf = target_features[target]
                                if isinstance(tf, dict) and 'features' in tf:
                                    training_features = tf['features']
                                elif isinstance(tf, (list, tuple)):
                                    training_features = list(tf)
                            
                            # Get n_samples from X if available
                            training_n_samples = X.shape[0] if 'X' in locals() and hasattr(X, 'shape') else None
                            
                            # Get train_seed from identity or config
                            training_seed = 42
                            if effective_run_identity and hasattr(effective_run_identity, 'train_seed'):
                                training_seed = effective_run_identity.train_seed
                            elif effective_run_identity and isinstance(effective_run_identity, dict):
                                training_seed = effective_run_identity.get('train_seed', 42)
                            
                            # Merge prediction fingerprint into model_result for snapshot
                            model_result_with_fp = model_result.copy() if model_result else {}
                            if prediction_fingerprint_cs:
                                model_result_with_fp['prediction_hash'] = prediction_fingerprint_cs.get('prediction_hash')
                            
                            # Extract cohort_id from cohort_metadata
                            cohort_id = None
                            if 'cohort_metadata' in locals() and cohort_metadata:
                                try:
                                    from TRAINING.training_strategies.reproducibility.io import compute_cohort_id_from_metadata
                                    cohort_id = compute_cohort_id_from_metadata(cohort_metadata, view=view)
                                except Exception as e:
                                    logger.debug(f"Failed to compute cohort_id: {e}")
                            
                            create_and_save_training_snapshot(
                                target=target,
                                model_family=family,
                                model_result=model_result_with_fp,
                                output_dir=base_run_dir,
                                view=view,
                                symbol=None,  # CROSS_SECTIONAL doesn't have symbol
                                run_identity=effective_run_identity,
                                model_path=saved_model_path,
                                features_used=training_features,
                                n_samples=training_n_samples,
                                train_seed=training_seed,
                                cohort_id=cohort_id,
                                cohort_metadata=cohort_metadata if 'cohort_metadata' in locals() else None,
                            )
                        except Exception as ts_err:
                            logger.debug(f"Training snapshot creation failed (non-critical): {ts_err}")
                    
                    logger.info(f"✅ {family} completed for {target}")
                    
                    # Memory hygiene after each family (after saving)
                    try:
                        from common.threads import hard_cleanup_after_family
                        
                        # Delete model result to free references
                        try:
                            del model_result
                        except:
                            pass
                        
                        # Aggressive cleanup (TF, XGBoost, PyTorch, CuPy)
                        hard_cleanup_after_family(family)
                        
                    except Exception as e:
                        logger.debug(f"[Cleanup] Minor cleanup issue: {e}")
                        pass
                elif model_result is not None:
                    family_results['failed'].append((family, normalized_family, "train_model_comprehensive returned success=False"))
                    logger.warning(f"❌ {family} failed for {target} (success=False)")
                else:
                    family_results['failed'].append((family, normalized_family, "train_model_comprehensive returned None"))
                    logger.warning(f"❌ {family} failed for {target} (returned None)")
                    
            except Exception as e:
                logger.exception(f"❌ [{family}] {family} failed for {target}: {e}")
                continue
        
        results['models'][target] = target_results
        
        # For BOTH route, also train symbol-specific models if needed
        if route == 'BOTH' and both_route_symbol_features is not None and len(both_route_symbol_features) > 0:
            logger.info(f"🔄 BOTH route: Training symbol-specific models for {target} ({len(both_route_symbol_features)} symbols)")
            
            # Temporarily set target_features to symbol-specific features for symbol-specific training
            original_target_features = target_features.get(target) if target_features else None
            target_features[target] = both_route_symbol_features
            
            # Train symbol-specific models (reuse the symbol-specific training path)
            for symbol, symbol_features in both_route_symbol_features.items():
                if symbol not in mtf_data:
                    logger.warning(f"Skipping {symbol} for {target}: symbol not in mtf_data")
                    continue
                
                if not isinstance(symbol_features, (list, tuple)) or len(symbol_features) == 0:
                    logger.warning(f"Skipping {symbol} for {target}: no features available")
                    continue
                
                logger.info(f"  📊 Training {symbol} with {len(symbol_features)} symbol-specific features (BOTH route)")
                
                # Prepare data for this symbol only
                symbol_mtf_data = {symbol: mtf_data[symbol]}
                X, y, feature_names, symbols_arr, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
                    symbol_mtf_data, target, feature_names=symbol_features, min_cs=1, max_cs_samples=max_cs_samples, routing_decisions=routing_decisions
                )
                
                if X is None or len(X) == 0:
                    logger.warning(f"❌ Failed to prepare data for {target}:{symbol}")
                    continue
                
                # Apply row cap if needed
                if max_rows_train and len(X) > max_rows_train:
                    from TRAINING.common.determinism import BASE_SEED, stable_seed_from
                    downsample_seed = stable_seed_from([target, symbol, 'downsample'])
                    np.random.seed(downsample_seed)
                    indices_to_keep = np.random.choice(len(X), max_rows_train, replace=False)
                    X = X[indices_to_keep]
                    y = y[indices_to_keep]
                    if symbols_arr is not None:
                        symbols_arr = symbols_arr[indices_to_keep]
                    if indices is not None:
                        indices = indices[indices_to_keep]
                    if time_vals is not None:
                        time_vals = time_vals[indices_to_keep]
                    logger.info(f"  📉 Downsampled to {len(X)} rows for {target}:{symbol}")
                
                # Train models for this symbol (reuse existing training logic)
                symbol_results = {}
                for family in families:
                    try:
                        # train_model_comprehensive is defined in this file
                        model_result = train_model_comprehensive(
                            family=family, X=X, y=y, target=target, strategy=strategy,
                            feature_names=feature_names, caps={}, routing_meta=routing_meta
                        )
                        
                        if model_result and model_result.get('success', False):
                            symbol_results[family] = model_result
                            
                            # Save model to simple training_results/<family>/symbol=<symbol>/ structure
                            try:
                                # Ensure output_dir is a Path object
                                output_dir_path = Path(output_dir) if not isinstance(output_dir, Path) else output_dir
                                
                                # Save symbol-specific models to training_results/<family>/symbol=<symbol>/
                                family_dir = output_dir_path / family
                                symbol_dir = family_dir / f"symbol={symbol}"
                                symbol_dir.mkdir(parents=True, exist_ok=True)
                                logger.info(f"💾 Saving SYMBOL_SPECIFIC model for {symbol} (BOTH route) to: {symbol_dir}")
                                
                                # Keep target-first structure ONLY for reproducibility metadata (not for model files)
                                from TRAINING.orchestration.utils.target_first_paths import (
                                    ensure_target_structure
                                )
                                ensure_target_structure(output_dir_path, target)
                                
                                # Save model (same logic as SYMBOL_SPECIFIC route)
                                from common.model_wrapper import wrap_model_for_saving, get_model_saving_info
                                strategy_manager = model_result.get('strategy_manager')
                                if strategy_manager and hasattr(strategy_manager, 'models'):
                                    models = strategy_manager.models
                                    for model_name, model in models.items():
                                        wrapped_model = wrap_model_for_saving(model, family)
                                        save_info = get_model_saving_info(wrapped_model)
                                        
                                        # Determine file extensions based on model type
                                        if save_info['is_lightgbm']:  # LightGBM
                                            model_path = symbol_dir / f"{family.lower()}_mtf_b0.txt"
                                            wrapped_model.save_model(str(model_path))
                                            logger.info(f"  💾 LightGBM model saved: {model_path}")
                                            
                                        elif save_info['is_tensorflow']:  # TensorFlow/Keras
                                            model_path = symbol_dir / f"{family.lower()}_mtf_b0.keras"
                                            wrapped_model.save(str(model_path))
                                            logger.info(f"  💾 Keras model saved: {model_path}")
                                            
                                        elif save_info['is_pytorch']:  # PyTorch models
                                            model_path = symbol_dir / f"{family.lower()}_mtf_b0.pt"
                                            import torch
                                            
                                            # Extract the actual PyTorch model
                                            if hasattr(wrapped_model, 'core') and hasattr(wrapped_model.core, 'model'):
                                                torch_model = wrapped_model.core.model
                                            elif hasattr(wrapped_model, 'model'):
                                                torch_model = wrapped_model.model
                                            else:
                                                torch_model = wrapped_model
                                            
                                            # Save state dict + metadata
                                            torch.save({
                                                "state_dict": torch_model.state_dict(),
                                                "config": getattr(wrapped_model, "config", {}),
                                                "arch": family,
                                                "input_shape": X.shape
                                            }, str(model_path))
                                            logger.info(f"  💾 PyTorch model saved: {model_path}")
                                            
                                        else:  # Scikit-learn models
                                            model_path = symbol_dir / f"{family.lower()}_mtf_b0.joblib"
                                            wrapped_model.save(str(model_path))
                                            logger.info(f"  💾 Scikit-learn model saved: {model_path}")
                                        
                                        # Save preprocessors if available
                                        if wrapped_model.scaler is not None:
                                            scaler_path = symbol_dir / f"{family.lower()}_mtf_b0_scaler.joblib"
                                            joblib.dump(wrapped_model.scaler, scaler_path)
                                            logger.info(f"  💾 Scaler saved: {scaler_path}")
                                        
                                        if wrapped_model.imputer is not None:
                                            imputer_path = symbol_dir / f"{family.lower()}_mtf_b0_imputer.joblib"
                                            joblib.dump(wrapped_model.imputer, imputer_path)
                                            logger.info(f"  💾 Imputer saved: {imputer_path}")
                                else:
                                    # Fallback: save model directly if no strategy_manager
                                    model_path = symbol_dir / "model.joblib"
                                    joblib.dump(model_result.get('model'), model_path)
                                    logger.info(f"  ✅ Saved {family} model for {target}:{symbol} to {model_path}")
                            except Exception as e:
                                logger.warning(f"Failed to save model for {target}:{symbol}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to train {family} for {target}:{symbol}: {e}")
                        continue
                
                # Store symbol results
                if symbol_results:
                    if target not in results['models']:
                        results['models'][target] = {}
                    if symbol not in results['models'][target]:
                        results['models'][target][symbol] = {}
                    results['models'][target][symbol].update(symbol_results)
                    logger.info(f"  ✅ Completed {symbol}: {len(symbol_results)} models trained (BOTH route)")
            
            # Restore original target_features
            if original_target_features is not None:
                target_features[target] = original_target_features
            elif target in target_features:
                # Restore to original BOTH structure if it was there
                if isinstance(target_feat_data, dict) and 'cross_sectional' in target_feat_data:
                    target_features[target] = target_feat_data
        
        # Memory hygiene after each target (CRITICAL for GPU models between targets)
        try:
            from common.threads import hard_cleanup_after_family
            import gc
            
            # Clean up training data (X, y can be 2-6GB)
            try:
                del X, y, feature_names, symbols, indices, feat_cols, time_vals
                logger.info(f"[Cleanup] Released training data after target {target}")
            except:
                pass
            
            # Delete target results
            try:
                del target_results
            except:
                pass
            
            # Aggressive cleanup for ALL frameworks
            logger.info(f"[Cleanup] Hard cleanup after target {target}")
            hard_cleanup_after_family(f"target_{target}")
            
        except Exception as e:
            logger.debug(f"[Cleanup] Minor cleanup issue after target {target}: {e}")
            pass
    
    # Create global training summary aggregate
    try:
        from TRAINING.training_strategies.reproducibility.io import aggregate_training_summaries
        aggregate_training_summaries(Path(output_dir))
    except Exception as e:
        logger.debug(f"Failed to create global training summary aggregate: {e}")
    
    # Summary of family status (if family_status was defined)
    if 'family_status' in locals():
        failed_families = [f for f, s in family_status.items() if not s["saved"] and s["attempted"]]
        if failed_families:
            logger.warning(f"⚠️ {len(failed_families)} families failed to save: {failed_families}")
            for family in failed_families:
                logger.warning(f"  - {family}: {family_status[family]['error'] or 'Save failed'}")
        
        skipped_families = [f for f, s in family_status.items() if not s["attempted"]]
        if skipped_families:
            logger.info(f"ℹ️ {len(skipped_families)} families were skipped: {skipped_families}")
        
        successful_families = [f for f, s in family_status.items() if s["saved"]]
        if successful_families:
            logger.info(f"✅ {len(successful_families)} families saved successfully: {successful_families}")
    
    # Create global training summary aggregate
    try:
        from TRAINING.training_strategies.reproducibility.io import aggregate_training_summaries
        aggregate_training_summaries(Path(output_dir))
    except Exception as e:
        logger.debug(f"Failed to create global training summary aggregate: {e}")
    
    # Count and log saved models with detailed summary
    # CRITICAL: Use consistent counting logic (single source of truth)
    total_saved = 0
    total_failed = 0
    total_skipped = 0
    total_attempted = 0
    
    for target, target_results in results['models'].items():
        for family, model_result in target_results.items():
            total_attempted += 1
            if model_result and model_result.get('success', False):
                total_saved += 1
            elif model_result and model_result.get('skipped', False):
                total_skipped += 1
            else:
                total_failed += 1
    
    # Aggregate family results across all targets
    all_trained_ok = []
    all_failed = []
    all_skipped = []
    
    # Note: family_results is per-target, so we'd need to track globally
    # For now, log per-target summary and overall saved count
    
    logger.info("=" * 80)
    logger.info(f"📊 Training Summary:")
    # CRITICAL: Log consistent counts (single source of truth)
    logger.info(f"  ✅ Training summary: {total_saved} successful, {total_failed} failed, {total_skipped} skipped (total attempted: {total_attempted})")
    
    # Store in results for consistency
    results['training_summary'] = {
        'total_attempted': total_attempted,
        'total_saved': total_saved,
        'total_failed': total_failed,
        'total_skipped': total_skipped
    }
    logger.info(f"  ❌ Failed targets: {len(results.get('failed_targets', []))}")
    if results.get('failed_targets'):
        for failed_target in results['failed_targets']:
            reason = results.get('failed_reasons', {}).get(failed_target, 'unknown')
            logger.info(f"    - {failed_target}: {reason}")
    logger.info(f"📁 Models saved to: {output_dir}")
    logger.info("=" * 80)
    
    # ========================================================================
    # Write training_results_summary.json to globals/summaries/
    # ========================================================================
    try:
        from pathlib import Path
        from datetime import datetime
        import json
        from TRAINING.orchestration.utils.target_first_paths import run_root, globals_dir as get_globals_dir
        
        output_dir_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        run_root_dir = run_root(output_dir_path)
        summaries_dir = get_globals_dir(run_root_dir, "summaries")
        summaries_dir.mkdir(parents=True, exist_ok=True)
        
        # Build detailed summary by target
        by_target = {}
        for target, target_models in results.get('models', {}).items():
            by_target[target] = {}
            for family, model_result in target_models.items():
                if model_result and isinstance(model_result, dict):
                    status = "success" if model_result.get('success', False) else (
                        "skipped" if model_result.get('skipped', False) else "failed"
                    )
                    by_target[target][family] = {
                        "status": status,
                        "auc": model_result.get('auc'),
                        "path": str(model_result.get('model_path')) if model_result.get('model_path') else None,
                        "error": model_result.get('error') if status == "failed" else None
                    }
        
        training_results_summary = {
            "generated_at": datetime.now().isoformat() + "Z",
            "interval": interval,
            "strategy": strategy,
            "total_models_trained": total_saved,
            "by_target": by_target,
            "summary": {
                "success": total_saved,
                "failed": total_failed,
                "skipped": total_skipped,
                "attempted": total_attempted
            },
            "failed_targets": results.get('failed_targets', []),
            "failed_reasons": results.get('failed_reasons', {})
        }
        
        summary_path = summaries_dir / "training_results_summary.json"
        with open(summary_path, "w") as f:
            json.dump(training_results_summary, f, indent=2, default=str)
        logger.info(f"💾 Training results summary saved to: {summary_path}")
    except Exception as e:
        logger.warning(f"Failed to write training_results_summary.json: {e}")
    
    return results

# Legacy code path - kept for backwards compatibility but shouldn't be reached
def _legacy_train_fallback(family: str, X: np.ndarray, y: np.ndarray, target: str = None, strategy: str = None, feature_names: List[str] = None, **kwargs):
    """Legacy fallback - should not be reached with runtime_policy."""
    logger.warning(f"[{family}] Unexpected fallback path - check runtime_policy configuration")
    if False:  # Dead code marker
        logger.info("🔄 [%s] using isolation runner (MKL backend)…", family)
        print(f"🔄 [{family}] using isolation runner (MKL backend)...")
        # Pass None to use optimal thread planning from plan_for_family()
        model = _run_family_isolated(
            family, X, y,
            omp_threads=None,  # Use optimal planning
            mkl_threads=None,  # Use optimal planning
            trainer_kwargs={}
        )
        manager = SingleTaskStrategy({'family': family})
        manager.models[family] = model
        return {
            'model': model,
            'trainer': None, 'test_predictions': None, 'success': True,
            'family': family, 'target': target, 'strategy': strategy,
            'strategy_manager': manager
        }
    
    # Route PyTorch sequential families for better performance
    # SEQ_BACKEND is only used in dead code path - default to 'torch' if not provided
    SEQ_BACKEND = kwargs.get('seq_backend', 'torch')  # Default for legacy code
    if family in TORCH_SEQ_FAMILIES and SEQ_BACKEND == 'torch':
        logger.info("🔥 [%s] using PyTorch implementation for better performance…", family)
        print(f"🔥 [{family}] using PyTorch implementation for better performance...")
        
        # Configure PyTorch threading
        try:
            import torch
            torch.set_num_threads(1 if not CPU_ONLY else THREADS)
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        
        # Convert to sequential format if needed
        if len(X.shape) == 2:  # (N, F) -> (N, T, F)
            # Load lookback from config if available
            lookback = None  # Will use config default in function
            X_seq = build_sequences_from_features(X, lookback=lookback)
        else:
            X_seq = X  # Already sequential
        
        # Import and use PyTorch trainer
        if family == 'CNN1D':
            from model_fun.cnn1d_trainer_torch import CNN1DTrainerTorch
            trainer = CNN1DTrainerTorch(config={"num_threads": THREADS})
        elif family == 'LSTM':
            from model_fun.lstm_trainer_torch import LSTMTrainerTorch
            trainer = LSTMTrainerTorch(config={"num_threads": THREADS})
        elif family == 'Transformer':
            from model_fun.transformer_trainer_torch import TransformerTrainerTorch
            trainer = TransformerTrainerTorch(config={"num_threads": THREADS})
        elif family == 'TabCNN':
            from model_fun.tabcnn_trainer_torch import TabCNNTrainerTorch
            trainer = TabCNNTrainerTorch(config={"num_threads": THREADS})
        elif family == 'TabLSTM':
            from model_fun.tablstm_trainer_torch import TabLSTMTrainerTorch
            trainer = TabLSTMTrainerTorch(config={"num_threads": THREADS})
        elif family == 'TabTransformer':
            from model_fun.tabtransformer_trainer_torch import TabTransformerTrainerTorch
            trainer = TabTransformerTrainerTorch(config={"num_threads": THREADS})
        
        # Train the model
        model = trainer.train(X_seq, y)
        
        manager = SingleTaskStrategy({'family': family})
        manager.models[family] = model
        return {
            'model': model,
            'trainer': trainer, 'test_predictions': None, 'success': True,
            'family': family, 'target': target, 'strategy': strategy,
            'strategy_manager': manager
        }
    
    # Route TensorFlow sequential families (fallback)
    if family in TORCH_SEQ_FAMILIES and SEQ_BACKEND == 'tf':
        logger.info("⚠️ [%s] using TensorFlow fallback (consider --seq-backend torch for better performance)…", family)
        print(f"⚠️ [{family}] using TensorFlow fallback (consider --seq-backend torch for better performance)...")
        
        # Import TensorFlow trainers
        if family == 'CNN1D':
            from model_fun.cnn1d_trainer import CNN1DTrainer
            trainer = CNN1DTrainer()
        elif family == 'LSTM':
            from model_fun.lstm_trainer import LSTMTrainer
            trainer = LSTMTrainer()
        elif family == 'Transformer':
            from model_fun.transformer_trainer import TransformerTrainer
            trainer = TransformerTrainer()
        elif family == 'TabCNN':
            from model_fun.tabcnn_trainer import TabCNNTrainer
            trainer = TabCNNTrainer()
        elif family == 'TabLSTM':
            from model_fun.tablstm_trainer import TabLSTMTrainer
            trainer = TabLSTMTrainer()
        elif family == 'TabTransformer':
            from model_fun.tabtransformer_trainer import TabTransformerTrainer
            trainer = TabTransformerTrainer()
        
        # Configure TF threading
        try:
            import tensorflow as tf
            if CPU_ONLY:
                tf.config.threading.set_intra_op_parallelism_threads(THREADS)
                tf.config.threading.set_inter_op_parallelism_threads(THREADS)
            else:
                tf.config.threading.set_intra_op_parallelism_threads(1)
                tf.config.threading.set_inter_op_parallelism_threads(1)
        except Exception:
            pass
        
        # Train the model
        model = trainer.train(X, y, feature_names=feature_names)
        
        manager = SingleTaskStrategy({'family': family})
        manager.models[family] = model
        return {
            'model': model,
            'trainer': trainer, 'test_predictions': None, 'success': True,
            'family': family, 'target': target, 'strategy': strategy,
            'strategy_manager': manager
        }
    
    # Import modular trainers for in-process families
    try:
        if family == 'LightGBM':
            from model_fun.lightgbm_trainer import LightGBMTrainer
            trainer = LightGBMTrainer(config={"num_threads": THREADS})
            logger.info("[%s] params: %s", family, getattr(trainer, "config", {}))
        elif family == 'XGBoost':
            from model_fun.xgboost_trainer import XGBoostTrainer
            xgb_conf = {"nthread": THREADS}
            try:
                import xgboost as xgb
                if not CPU_ONLY:
                    # Check GPU availability properly (XGBoostTrainer will do the real check)
                    # Just set CPU defaults here, let the trainer decide
                    xgb_conf.update({"tree_method": "hist"})  # Default to CPU, trainer will upgrade if GPU works
                else:
                    xgb_conf.update({"tree_method": "hist"})
            except Exception:
                xgb_conf.update({"tree_method": "hist"})
            trainer = XGBoostTrainer(config=xgb_conf)
            logger.info("[%s] params: %s", family, getattr(trainer, "config", {}))
        elif family == 'MLP':
            from model_fun.mlp_trainer import MLPTrainer
            trainer = MLPTrainer()
            # Configure TF threading based on CPU_ONLY
            try:
                import tensorflow as tf
                if CPU_ONLY:   # no GPU ⇒ let TF use cores
                    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
                    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
                else:          # GPU ⇒ keep CPU light
                    tf.config.threading.set_intra_op_parallelism_threads(1)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
        elif family == 'CNN1D':
            from model_fun.cnn1d_trainer import CNN1DTrainer
            trainer = CNN1DTrainer()
            # Configure TF threading based on CPU_ONLY
            try:
                import tensorflow as tf
                if CPU_ONLY:   # no GPU ⇒ let TF use cores
                    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
                    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
                else:          # GPU ⇒ keep CPU light
                    tf.config.threading.set_intra_op_parallelism_threads(1)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
        elif family == 'LSTM':
            from model_fun.lstm_trainer import LSTMTrainer
            trainer = LSTMTrainer()
            # Configure TF threading based on CPU_ONLY
            try:
                import tensorflow as tf
                if CPU_ONLY:   # no GPU ⇒ let TF use cores
                    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
                    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
                else:          # GPU ⇒ keep CPU light
                    tf.config.threading.set_intra_op_parallelism_threads(1)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
        elif family == 'Transformer':
            from model_fun.transformer_trainer import TransformerTrainer
            trainer = TransformerTrainer()
            # Configure TF threading based on CPU_ONLY
            try:
                import tensorflow as tf
                if CPU_ONLY:   # no GPU ⇒ let TF use cores
                    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
                    tf.config.threading.set_inter_op_parallelism_threads(THREADS)
                else:          # GPU ⇒ keep CPU light
                    tf.config.threading.set_intra_op_parallelism_threads(1)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
            except Exception:
                pass
        elif family == 'RewardBased':
            from model_fun.reward_based_trainer import RewardBasedTrainer
            trainer = RewardBasedTrainer()
        elif family == 'Ensemble':
            from model_fun.ensemble_trainer import EnsembleTrainer
            trainer = EnsembleTrainer()
        elif family == 'ChangePoint':
            from model_fun.change_point_trainer import ChangePointTrainer
            trainer = ChangePointTrainer()
        elif family == 'QuantileLightGBM':
            from model_fun.quantile_lightgbm_trainer import QuantileLightGBMTrainer
            trainer = QuantileLightGBMTrainer(config={"num_threads": THREADS, "keepalive_every": 200})
            logger.info("[%s] params: %s", family, getattr(trainer, "config", {}))
        elif family == 'NGBoost':
            from model_fun.ngboost_trainer import NGBoostTrainer
            trainer = NGBoostTrainer()
        elif family == 'GMMRegime':
            from model_fun.gmm_regime_trainer import GMMRegimeTrainer
            trainer = GMMRegimeTrainer()
        elif family == 'FTRLProximal':
            from model_fun.ftrl_proximal_trainer import FTRLProximalTrainer
            trainer = FTRLProximalTrainer()
        elif family == 'VAE':
            from model_fun.vae_trainer import VAETrainer
            trainer = VAETrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        elif family == 'GAN':
            from model_fun.gan_trainer import GANTrainer
            trainer = GANTrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        elif family == 'MetaLearning':
            from model_fun.meta_learning_trainer import MetaLearningTrainer
            trainer = MetaLearningTrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        elif family == 'MultiTask':
            from model_fun.multi_task_trainer import MultiTaskTrainer
            trainer = MultiTaskTrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        elif family == 'TabCNN':
            from model_fun.tabcnn_trainer import TabCNNTrainer
            trainer = TabCNNTrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        elif family == 'TabLSTM':
            from model_fun.tablstm_trainer import TabLSTMTrainer
            trainer = TabLSTMTrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        elif family == 'TabTransformer':
            from model_fun.tabtransformer_trainer import TabTransformerTrainer
            trainer = TabTransformerTrainer()
            # Configure TF threading for CPU learners
            import tensorflow as tf
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        else:
            logger.warning(f"Family {family} not implemented in modular system")
            return None
            
    except ImportError as e:
        logger.error(f"Failed to import {family} trainer: {e}")
        return None
    
    # ANTI-DEADLOCK: Per-family environment setup before training
    try:
        # Set per-family environment variables
        if family in {"LightGBM","QuantileLightGBM","XGBoost","RewardBased","FTRLProximal","NGBoost"}:
            _env_guard(THREADS, mkl_threads=1)
        elif family in {"GMMRegime","ChangePoint"}:
            _env_guard(1, mkl_threads=THREADS)
        else:
            # TF / GPU families: keep CPU light unless CPU_ONLY
            _env_guard(1 if not CPU_ONLY else THREADS, mkl_threads=1 if not CPU_ONLY else THREADS)
        
        logger.info("🚀 [%s] Starting %s training on %d samples…", family, family, len(X))
        print(f"🚀 [{family}] Starting {family} training on {len(X)} samples...")
        print(f"DEBUG: About to call trainer.train() for {family}")
        logger.info("Threads → OMP=%s, MKL=%s, TF(cpu)=%s/%s",
                    os.getenv("OMP_NUM_THREADS"), os.getenv("MKL_NUM_THREADS"),
                    "auto" if CPU_ONLY else "1", "auto" if CPU_ONLY else "1")
        t0 = _now()
        
        # All families now use isolation runner, so this path is no longer needed
        model = trainer.train(X, y)
        print(f"DEBUG: trainer.train() completed for {family} in {safe_duration(t0)}")

        if model is None:
            logger.warning("❌ [%s] training returned None", family)
            return None

        # (optional) quick sanity prediction
        try: test_predictions = trainer.predict(X[:100])
        except Exception: test_predictions = None

        logger.info("✅ [%s] training completed successfully", family)
        print(f"✅ [{family}] {family} training completed successfully")
        
        # Use real strategy manager instead of mock
        # Using top-level imports: SingleTaskStrategy, MultiTaskStrategy, CascadeStrategy
        
        # Create appropriate strategy manager based on strategy type
        if strategy == "single_task":
            strategy_manager = SingleTaskStrategy({'family': family})
        elif strategy == "multi_task":
            strategy_manager = MultiTaskStrategy({'family': family})
        elif strategy == "cascade":
            strategy_manager = CascadeStrategy({'family': family})
        else:
            # Default to single task for unknown strategies
            strategy_manager = SingleTaskStrategy({'family': family})
        
        # Store the trained model in the strategy manager
        strategy_manager.models[family] = model
        strategy_manager.trainer = trainer
        
        return {
            'model': model,
            'trainer': trainer,
            'test_predictions': test_predictions,
            'success': True,
            'family': family,
            'target': target,
            'strategy': strategy,
            'strategy_manager': strategy_manager
        }
        
    except Exception as e:
        logger.error(f"❌ [{family}] Error training {family}: {e}")
        return None

