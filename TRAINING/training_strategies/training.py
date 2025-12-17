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

# Additional safety: ensure the path is in sys.path for child processes
def _ensure_project_path():
    """Ensure project path is available for child processes."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# Call it immediately
_ensure_project_path()

# Set global numeric guards for stability
# Add TRAINING to path for local imports
_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Also add current directory for relative imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _PROJECT_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Import config loader
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
import atexit
# Set persistent temp folder for joblib memmapping
# Load from config if available, otherwise use default
if _CONFIG_AVAILABLE:
    joblib_temp = get_cfg("system.paths.joblib_temp", config_name="system_config")
    if joblib_temp:
        _JOBLIB_TMP = Path(joblib_temp)
    else:
        _JOBLIB_TMP = Path.home() / "trainer_tmp" / "joblib"
else:
    _JOBLIB_TMP = Path.home() / "trainer_tmp" / "joblib"
_JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_JOBLIB_TMP))

# Force clean loky worker shutdown at exit to prevent semlock/file leaks
try:
    from joblib.externals.loky import get_reusable_executor
    @atexit.register
    def _loky_shutdown():
        try:
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass
except Exception:
    pass

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

# Family classifications
TF_FAMS = {"MLP", "VAE", "GAN", "MetaLearning", "MultiTask"}
TORCH_FAMS = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}
CPU_FAMS = {"LightGBM", "QuantileLightGBM", "RewardBased", "NGBoost", "GMMRegime", "ChangePoint", "FTRLProximal", "Ensemble"}


"""Core training functions."""

# Import dependencies
from TRAINING.training_strategies.family_runners import _run_family_inproc, _run_family_isolated
from TRAINING.training_strategies.data_preparation import prepare_training_data_cross_sectional
from TRAINING.training_strategies.utils import (
    FAMILY_CAPS, ALL_FAMILIES, tf_available, ngboost_available,
    _now, _pkg_ver, THREADS, CPU_ONLY,
    TORCH_SEQ_FAMILIES, build_sequences_from_features, _env_guard, safe_duration
)
# train_model_comprehensive is defined in this file, not in utils
from TRAINING.target_router import TaskSpec
from TRAINING.strategies.single_task import SingleTaskStrategy
from TRAINING.strategies.multi_task import MultiTaskStrategy
from TRAINING.strategies.cascade import CascadeStrategy

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
                                           routing_decisions: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
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
        
        # Prepare training data with cross-sectional sampling
        print(f"🔄 Preparing training data for target: {target}")  # Debug print
        prep_start = _t.time()
        
        # Use selected features for this target if provided
        # Handle different structures: list, dict (symbol-specific), or structured dict (BOTH route)
        selected_features = None
        route_info = routing_decisions.get(target, {}) if routing_decisions else {}
        route = route_info.get('route', 'CROSS_SECTIONAL')
        
        if target_features and target in target_features:
            target_feat_data = target_features[target]
            
            # Handle different structures based on route
            if route == 'BLOCKED':
                # BLOCKED targets should be skipped entirely
                logger.warning(f"Skipping {target} (BLOCKED: {route_info.get('reason', 'suspicious score')})")
                results['failed_targets'].append(target)
                results['failed_reasons'][target] = f"BLOCKED: {route_info.get('reason', 'suspicious score')}"
                continue
            elif route == 'CROSS_SECTIONAL':
                # Simple list of features
                selected_features = target_feat_data
                if selected_features is not None:
                    if not isinstance(selected_features, (list, tuple)):
                        logger.warning(f"selected_features for {target} is not a list/tuple (type: {type(selected_features)}), converting...")
                        try:
                            selected_features = list(selected_features)
                        except Exception as e:
                            logger.error(f"Failed to convert selected_features to list: {e}")
                            selected_features = None
                    elif len(selected_features) == 0:
                        logger.warning(f"selected_features for {target} is empty, will auto-discover features")
                        selected_features = None
                    else:
                        logger.info(f"Using {len(selected_features)} cross-sectional features for {target}")
            elif route == 'SYMBOL_SPECIFIC':
                # Dict mapping symbol -> list of features
                # Train separate models per symbol (not cross-sectional)
                if isinstance(target_feat_data, dict):
                    # Will handle per-symbol training below
                    logger.info(f"SYMBOL_SPECIFIC route: will train {len(target_feat_data)} separate models (one per symbol)")
                else:
                    logger.warning(f"Expected dict for SYMBOL_SPECIFIC route, got {type(target_feat_data)}")
                    # Fallback: skip this target
                    results['failed_targets'].append(target)
                    results['failed_reasons'][target] = f"SYMBOL_SPECIFIC route requires dict, got {type(target_feat_data)}"
                    continue
            elif route == 'BOTH':
                # Structured dict: {'cross_sectional': [...], 'symbol_specific': {symbol: [...]}}
                if isinstance(target_feat_data, dict) and 'cross_sectional' in target_feat_data:
                    # Use cross-sectional features for now (can extend to ensemble later)
                    selected_features = target_feat_data['cross_sectional']
                    if selected_features and isinstance(selected_features, (list, tuple)):
                        logger.info(f"Using {len(selected_features)} cross-sectional features for {target} (BOTH route - using CS for now)")
                    else:
                        logger.warning(f"Cross-sectional features not found in BOTH structure for {target}")
                        selected_features = None
                else:
                    # Fallback: treat as simple list
                    selected_features = target_feat_data
                    if selected_features and not isinstance(selected_features, (list, tuple)):
                        logger.warning(f"BOTH route but unexpected structure, treating as list")
                        try:
                            selected_features = list(selected_features)
                        except Exception:
                            selected_features = None
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
                
                # Prepare data for this symbol only
                symbol_mtf_data = {symbol: mtf_data[symbol]}
                X, y, feature_names, symbols_arr, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
                    symbol_mtf_data, target, feature_names=symbol_features, min_cs=1, max_cs_samples=max_cs_samples
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
                    from TRAINING.target_router import route_target
                    route_info = route_target(target)
                    routing_meta = {
                        'target_name': target,
                        'spec': route_info['spec'],
                        'sample_weights': None,
                        'group_sizes': None
                    }
                
                # Train models for this symbol
                symbol_results = {}
                for family in target_families:
                    try:
                        logger.info(f"  🤖 Training {family} for {target}:{symbol}")
                        model_result = train_model_comprehensive(
                            family, X, y, target, strategy, feature_names,
                            caps={}, routing_meta=routing_meta
                        )
                        
                        if model_result is not None and model_result.get('success', False):
                            symbol_results[family] = model_result
                            
                            # Save model with symbol-specific path
                            family_dir = Path(output_dir) / family
                            symbol_target_dir = family_dir / target / symbol
                            symbol_target_dir.mkdir(parents=True, exist_ok=True)
                            
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
                                    if save_info['is_lightgbm']:  # LightGBM
                                        model_path = symbol_target_dir / f"{family.lower()}_mtf_b0.txt"
                                        wrapped_model.save_model(str(model_path))
                                        logger.info(f"  💾 LightGBM model saved: {model_path}")
                                        
                                    elif save_info['is_tensorflow']:  # TensorFlow/Keras
                                        model_path = symbol_target_dir / f"{family.lower()}_mtf_b0.keras"
                                        wrapped_model.save(str(model_path))
                                        logger.info(f"  💾 Keras model saved: {model_path}")
                                        
                                    elif save_info['is_pytorch']:  # PyTorch models
                                        model_path = symbol_target_dir / f"{family.lower()}_mtf_b0.pt"
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
                                        model_path = symbol_target_dir / f"{family.lower()}_mtf_b0.joblib"
                                        wrapped_model.save(str(model_path))
                                        logger.info(f"  💾 Scikit-learn model saved: {model_path}")
                                    
                                    # Save preprocessors if available
                                    if wrapped_model.scaler is not None:
                                        scaler_path = symbol_target_dir / f"{family.lower()}_mtf_b0_scaler.joblib"
                                        joblib.dump(wrapped_model.scaler, scaler_path)
                                        logger.info(f"  💾 Scaler saved: {scaler_path}")
                                    
                                    if wrapped_model.imputer is not None:
                                        imputer_path = symbol_target_dir / f"{family.lower()}_mtf_b0_imputer.joblib"
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
                                        meta_path = symbol_target_dir / "meta_b0.json"
                                        import json
                                        metadata = {
                                            "family": family,
                                            "target": target,
                                            "symbol": symbol,
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
                                        
                                        with open(meta_path, 'w') as f:
                                            json.dump(metadata, f, indent=2)
                                        logger.info(f"  💾 Metadata saved: {meta_path}")
                                    
                                    else:  # Other model types - save as JSON too
                                        meta_path = symbol_target_dir / "meta_b0.json"
                                        import json
                                        metadata = {
                                            "family": family,
                                            "target": target,
                                            "symbol": symbol,
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
                                        
                                        with open(meta_path, 'w') as f:
                                            json.dump(metadata, f, indent=2)
                                        logger.info(f"  💾 Metadata saved: {meta_path}")
                            else:
                                # Fallback: save model directly if no strategy_manager
                                model_path = symbol_target_dir / "model.joblib"
                                # joblib already imported at top of file (line 176)
                                joblib.dump(model_result.get('model'), model_path)
                                logger.info(f"  ✅ Saved {family} model for {target}:{symbol} to {model_path}")
                                
                                # Save basic metadata
                                meta_path = symbol_target_dir / "meta_b0.json"
                                import json
                                metadata = {
                                    "family": family,
                                    "target": target,
                                    "symbol": symbol,
                                    "n_features": len(feature_names) if feature_names else 0,
                                    "n_rows_train": len(X),
                                    "routing": {
                                        "route": "SYMBOL_SPECIFIC",
                                        "symbol": symbol,
                                        "view": "SYMBOL_SPECIFIC"
                                    }
                                }
                                with open(meta_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                logger.info(f"  💾 Basic metadata saved: {meta_path}")
                            
                            # Track reproducibility for symbol-specific model
                            if output_dir:
                                try:
                                    from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
                                    from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                                    
                                    module_output_dir = Path(output_dir)
                                    if module_output_dir.name != 'training_results':
                                        module_output_dir = module_output_dir.parent / 'training_results'
                                    
                                    tracker = ReproducibilityTracker(
                                        output_dir=module_output_dir,
                                        search_previous_runs=True
                                    )
                                    
                                    # Extract metrics
                                    strategy_manager = model_result.get('strategy_manager')
                                    metrics = {}
                                    if strategy_manager and hasattr(strategy_manager, 'cv_scores'):
                                        cv_scores = strategy_manager.cv_scores
                                        if cv_scores and len(cv_scores) > 0:
                                            metrics = {
                                                "metric_name": "CV Score",
                                                "mean_score": float(np.mean(cv_scores)),
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
                                        
                                        metrics_with_cohort = {**metrics, **cohort_metrics}
                                        additional_data_with_cohort = {
                                            "strategy": strategy,
                                            "n_features": len(feature_names) if feature_names else 0,
                                            "model_family": family,
                                            "symbol": symbol,
                                            **cohort_additional_data
                                        }
                                        
                                        tracker.log_comparison(
                                            stage="model_training",
                                            item_name=f"{target}:{symbol}:{family}",
                                            metrics=metrics_with_cohort,
                                            additional_data=additional_data_with_cohort,
                                            symbol=symbol
                                        )
                                except Exception as e:
                                    logger.warning(f"Reproducibility tracking failed for {family}:{target}:{symbol}: {e}")
                    except Exception as e:
                        logger.error(f"❌ Failed to train {family} for {target}:{symbol}: {e}")
                        import traceback
                        logger.debug(f"Full traceback for {family}:{symbol}:\n{traceback.format_exc()}")
                
                # Store symbol results
                if symbol_results:
                    if target not in results['models']:
                        results['models'][target] = {}
                    results['models'][target][symbol] = symbol_results
                    logger.info(f"  ✅ Completed {symbol}: {len(symbol_results)} models trained")
                else:
                    logger.warning(f"  ⚠️ No models trained for {target}:{symbol}")
            
            # Skip the cross-sectional training path for SYMBOL_SPECIFIC
            continue
        
        # Cross-sectional training (for CROSS_SECTIONAL or BOTH routes)
        # Note: BLOCKED targets are skipped earlier in the loop
        X, y, feature_names, symbols, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
            mtf_data, target, feature_names=selected_features, min_cs=min_cs, max_cs_samples=max_cs_samples
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
        
        if requested_count > 0 and used_count < requested_count * 0.5:
            logger.warning(
                f"⚠️ Feature count collapse for {target}: "
                f"requested={requested_count} → allowed={allowed_count} → used={used_count} "
                f"({used_count/requested_count*100:.1f}% retained). "
                f"This may indicate data quality issues or overly aggressive filtering."
            )
        
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
                'target_name': target,
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
            "QuantileLightGBM", "LightGBM", "RewardBased", "XGBoost",  # CPU tree learners first
            "MLP", "Ensemble", "ChangePoint", "NGBoost", "GMMRegime", "FTRLProximal", "VAE", "GAN", "MetaLearning", "MultiTask"  # Others
        ]
        
        # PREFLIGHT: Validate families exist in trainer registry before training starts
        from TRAINING.training_strategies.utils import normalize_family_name
        from common.isolation_runner import TRAINER_MODULE_MAP
        
        # Get canonical family set from registries (must match MODMAP in family_runners.py)
        MODMAP_KEYS = {
            "LightGBM", "QuantileLightGBM", "XGBoost", "RewardBased", "GMMRegime",
            "ChangePoint", "NGBoost", "Ensemble", "FTRLProximal", "MLP", "VAE",
            "GAN", "MetaLearning", "MultiTask"
        }
        REGISTRY_KEYS = MODMAP_KEYS | set(TRAINER_MODULE_MAP.keys())
        
        # Normalize and validate requested families
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
            logger.warning(f"⚠️ Preflight: {len(invalid_families)} families not in trainer registry:")
            for raw, norm in invalid_families:
                logger.warning(f"  - {raw} (normalized: {norm}) → SKIP")
        if skipped_families:
            logger.info(f"ℹ️ Preflight: {len(skipped_families)} families are selectors (not trainers):")
            for raw, norm, reason in skipped_families:
                logger.info(f"  - {raw} (normalized: {norm}) → SKIP ({reason})")
        
        if not validated_families:
            logger.error(f"❌ No valid trainer families after preflight validation. Requested: {target_families}")
            results['failed_targets'].append(target)
            results['failed_reasons'][target] = f"No valid trainer families (invalid: {[f[0] for f in invalid_families]}, selectors: {[f[0] for f in skipped_families]})"
            continue
        
        logger.info(f"✅ Preflight: {len(validated_families)} valid trainer families, {len(skipped_families)} selectors skipped, {len(invalid_families)} invalid")
        
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
        
        for i, family in enumerate(ordered_families, 1):
            logger.info(f"🎯 [{i}/{len(ordered_families)}] Training {family} for {target}")
            logger.info(f"📊 Data shape: X={X.shape}, y={y.shape}")
            logger.info(f"🔧 Strategy: {strategy}")
            print(f"🎯 [{i}/{len(ordered_families)}] Training {family} for {target}")  # Also print to stdout
            print(f"DEBUG: About to call train_model_comprehensive for {family}")  # Debug print
            
            try:
                # Normalize family name for capabilities map lookup
                from TRAINING.training_strategies.utils import normalize_family_name
                normalized_family = normalize_family_name(family)
                
                # Check family capabilities
                if normalized_family not in FAMILY_CAPS:
                    logger.warning(f"Model family {family} (normalized: {normalized_family}) not in capabilities map. Skipping.")
                    family_results['skipped'].append((family, normalized_family, "not_in_capabilities_map"))
                    continue
                
                caps = FAMILY_CAPS[normalized_family]
                logger.info(f"📋 Family capabilities: {caps}")
                
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
                    logger.error(f"❌ [{normalized_family}] Training failed after {elapsed:.2f} seconds: {train_err}")
                    logger.exception(f"Full traceback for {normalized_family}:")
                    family_results['failed'].append((family, normalized_family, str(train_err)))
                    # Don't re-raise - continue with next family
                    continue
                
                if model_result is not None and model_result.get('success', False):
                    target_results[normalized_family] = model_result
                    family_results['trained_ok'].append((family, normalized_family))
                    
                    # Track reproducibility: compare to previous training run
                    if output_dir and model_result.get('success', False):
                        try:
                            from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
                            # Use module-specific directory for reproducibility log
                            # output_dir is typically: output_dir_YYYYMMDD_HHMMSS/training_results/
                            # We want to store in training_results/ subdirectory for this module
                            if output_dir.name == 'training_results' or (output_dir.parent / 'training_results').exists():
                                # Already in or can find training_results subdirectory
                                if output_dir.name != 'training_results':
                                    module_output_dir = output_dir.parent / 'training_results'
                                else:
                                    module_output_dir = output_dir
                            else:
                                # Fallback: use output_dir directly (for standalone runs)
                                module_output_dir = output_dir
                            
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
                                        "mean_score": float(np.mean(cv_scores)),
                                        "std_score": float(np.std(cv_scores)),
                                        "composite_score": float(np.mean(cv_scores))
                                    }
                            
                            # If we have metrics, log comparison
                            if metrics:
                                # Extract cohort metadata using unified extractor
                                from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                                
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
                                    **cohort_metrics  # Adds N_effective_cs if available
                                }
                                
                                additional_data_with_cohort = {
                                    "strategy": strategy,
                                    "n_features": len(feature_names) if feature_names else 0,
                                    "model_family": family,  # Add model family for routing
                                    **cohort_additional_data  # Adds n_symbols, date_range, cs_config if available
                                }
                                
                                tracker.log_comparison(
                                    stage="model_training",
                                    item_name=f"{target}:{family}",
                                    metrics=metrics_with_cohort,
                                    additional_data=additional_data_with_cohort
                                )
                        except Exception as e:
                            logger.warning(f"Reproducibility tracking failed for {family}:{target}: {e}")
                            import traceback
                            logger.debug(f"Reproducibility tracking traceback: {traceback.format_exc()}")
                    
                    # Save model using original structure: FamilyName/target_name/model_files
                    family_dir = Path(output_dir) / family
                    target_dir = family_dir / target
                    target_dir.mkdir(parents=True, exist_ok=True)
                    
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
                            if save_info['is_lightgbm']:  # LightGBM
                                model_path = target_dir / f"{family.lower()}_mtf_b0.txt"
                                wrapped_model.save_model(str(model_path))
                                logger.info(f"💾 LightGBM model saved: {model_path}")
                                
                            elif save_info['is_tensorflow']:  # TensorFlow/Keras
                                model_path = target_dir / f"{family.lower()}_mtf_b0.keras"
                                wrapped_model.save(str(model_path))
                                logger.info(f"💾 Keras model saved: {model_path}")
                                
                            elif save_info['is_pytorch']:  # PyTorch models
                                model_path = target_dir / f"{family.lower()}_mtf_b0.pt"
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
                                model_path = target_dir / f"{family.lower()}_mtf_b0.joblib"
                                wrapped_model.save(str(model_path))
                                logger.info(f"💾 Scikit-learn model saved: {model_path}")
                            
                            # Save preprocessors if available
                            if wrapped_model.scaler is not None:
                                scaler_path = target_dir / f"{family.lower()}_mtf_b0_scaler.joblib"
                                joblib.dump(wrapped_model.scaler, scaler_path)
                                logger.info(f"💾 Scaler saved: {scaler_path}")
                                
                            if wrapped_model.imputer is not None:
                                imputer_path = target_dir / f"{family.lower()}_mtf_b0_imputer.joblib"
                                joblib.dump(wrapped_model.imputer, imputer_path)
                                logger.info(f"💾 Imputer saved: {imputer_path}")
                            # Note: If wrapped_model.imputer is None, no imputer was used/needed
                            
                            # Save metadata (match original format exactly)
                            if save_info['is_lightgbm']:  # LightGBM - JSON format
                                meta_path = target_dir / "meta_b0.json"
                                import json
                                metadata = {
                                    "family": family,
                                    "target": target,
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
                                with open(meta_path, 'w') as f:
                                    json.dump(metadata, f, indent=2)
                                    
                            else:  # TensorFlow/Scikit-learn - joblib format
                                meta_path = target_dir / f"{family.lower()}_mtf_b0.meta.joblib"
                                metadata = {
                                    "family": family,
                                    "target": target,
                                    "features": tuple(feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names))
                                }
                                joblib.dump(metadata, meta_path)
                                
                    except Exception as e:
                        logger.warning(f"Failed to save model {family}_{target}: {e}")
                    
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
    
    # Count and log saved models with detailed summary
    total_saved = 0
    total_failed = 0
    total_skipped = 0
    
    for target, target_results in results['models'].items():
        for family, model_result in target_results.items():
            if model_result and model_result.get('success', False):
                total_saved += 1
    
    # Aggregate family results across all targets
    all_trained_ok = []
    all_failed = []
    all_skipped = []
    
    # Note: family_results is per-target, so we'd need to track globally
    # For now, log per-target summary and overall saved count
    
    logger.info("=" * 80)
    logger.info(f"📊 Training Summary:")
    logger.info(f"  ✅ Trained successfully: {total_saved} models")
    logger.info(f"  ❌ Failed targets: {len(results.get('failed_targets', []))}")
    if results.get('failed_targets'):
        for failed_target in results['failed_targets']:
            reason = results.get('failed_reasons', {}).get(failed_target, 'unknown')
            logger.info(f"    - {failed_target}: {reason}")
    logger.info(f"📁 Models saved to: {output_dir}")
    logger.info("=" * 80)
    
    return results

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
    
    spec = routing_meta.get('spec')
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
    from target_router import get_objective_for_family
    
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

