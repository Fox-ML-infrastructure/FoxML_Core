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

# Standard library imports
from typing import Dict, List, Any, Optional, Tuple
import logging

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
                                           target_features: Dict[str, List[str]] = None) -> Dict[str, Any]:
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
        
        # Prepare training data with cross-sectional sampling
        print(f"🔄 Preparing training data for target: {target}")  # Debug print
        prep_start = _t.time()
        
        # Use selected features for this target if provided
        selected_features = None
        if target_features and target in target_features:
            selected_features = target_features[target]
            logger.info(f"Using {len(selected_features)} selected features for {target}")
        
        X, y, feature_names, symbols, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
            mtf_data, target, feature_names=selected_features, min_cs=min_cs, max_cs_samples=max_cs_samples
        )
        prep_elapsed = time.time() - prep_start
        print(f"✅ Data preparation completed in {prep_elapsed:.2f}s")  # Debug print
        
        if X is None:
            logger.error(f"❌ Failed to prepare data for target {target}")
            results['failed_targets'].append(target)
            results['failed_reasons'][target] = "Data preparation returned None (likely all features became NaN after coercion)"
            continue
        
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
            rng = np.random.RandomState(42)
            idx = rng.choice(len(X), max_rows_train, replace=False)
            X, y = X[idx], y[idx]
            if time_vals is not None: time_vals = time_vals[idx]
            if symbols is not None: symbols = symbols[idx]
            logger.info(f"✂️ Downsampled to max_rows_train={max_rows_train}")
        
        target_results = {}
        
        # CRITICAL: Order families to prevent cross-lib thread pollution
        # Run CPU-GBDT families FIRST, then TF/XGB families
        FAMILY_ORDER = [
            "QuantileLightGBM", "LightGBM", "RewardBased", "XGBoost",  # CPU tree learners first
            "MLP", "Ensemble", "ChangePoint", "NGBoost", "GMMRegime", "FTRLProximal", "VAE", "GAN", "MetaLearning", "MultiTask"  # Others
        ]
        
        # Reorder families to prevent thread pollution
        ordered_families = []
        for priority_family in FAMILY_ORDER:
            if priority_family in families:
                ordered_families.append(priority_family)
        # Add any remaining families not in the priority list
        for family in families:
            if family not in ordered_families:
                ordered_families.append(family)
        
        logger.info(f"🔄 Reordered families to prevent thread pollution: {ordered_families}")
        print(f"🔄 Reordered families to prevent thread pollution: {ordered_families}")
        
        for i, family in enumerate(ordered_families, 1):
            logger.info(f"🎯 [{i}/{len(ordered_families)}] Training {family} for {target}")
            logger.info(f"📊 Data shape: X={X.shape}, y={y.shape}")
            logger.info(f"🔧 Strategy: {strategy}")
            print(f"🎯 [{i}/{len(ordered_families)}] Training {family} for {target}")  # Also print to stdout
            print(f"DEBUG: About to call train_model_comprehensive for {family}")  # Debug print
            
            try:
                # Check family capabilities
                if family not in FAMILY_CAPS:
                    logger.warning(f"Model family {family} not in capabilities map. Skipping.")
                    continue
                
                caps = FAMILY_CAPS[family]
                logger.info(f"📋 Family capabilities: {caps}")
                
                # Check TensorFlow dependency (skip for torch families)
                if caps.get("backend") == "torch":
                    pass  # never gate on TF for torch families
                elif caps.get("needs_tf"):
                    # For isolated models, let child process handle TF availability
                    # For in-process models, check TF availability in parent
                    from TRAINING.common.runtime_policy import should_isolate
                    if not should_isolate(family) and not tf_available():
                        logger.warning(f"TensorFlow missing → skipping {family}")
                        continue
                    # If isolated, child process will handle TF import/initialization
                
                # Check NGBoost dependency
                if family == "NGBoost" and not ngboost_available():
                    logger.warning(f"NGBoost missing → skipping {family}")
                    continue
                
                logger.info(f"🚀 [{family}] Starting {family} training...")
                start_time = _now()
                
                # Train model using modular system with routing metadata
                try:
                    model_result = train_model_comprehensive(
                        family, X, y, target, strategy, feature_names, caps, routing_meta
                    )
                    elapsed = _now() - start_time
                    logger.info(f"⏱️ [{family}] {family} training completed in {elapsed:.2f} seconds")
                    if model_result is None:
                        logger.warning(f"⚠️ [{family}] train_model_comprehensive returned None")
                except Exception as train_err:
                    elapsed = _now() - start_time
                    logger.error(f"❌ [{family}] Training failed after {elapsed:.2f} seconds: {train_err}")
                    logger.exception(f"Full traceback for {family}:")
                    raise  # Re-raise to be caught by outer exception handler
                
                if model_result is not None:
                    target_results[family] = model_result
                    
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
                                
                                # Extract the actual PyTorch model from trainer
                                if hasattr(trainer, 'core') and hasattr(trainer.core, 'model'):
                                    torch_model = trainer.core.model
                                else:
                                    torch_model = wrapped_model
                                
                                # Save state dict + metadata
                                torch.save({
                                    "state_dict": torch_model.state_dict(),
                                    "config": getattr(trainer, "config", {}),
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
                else:
                    logger.warning(f"❌ {family} failed for {target}")
                    
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
                del X, y, feature_names, symbols, indices, feat_cols, time_vals, imputer
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
    
    # Count and log saved models
    total_saved = 0
    for target, target_results in results['models'].items():
        for family, model_result in target_results.items():
            if model_result and model_result.get('success', False):
                total_saved += 1
    
    logger.info(f"💾 Total models saved: {total_saved}")
    logger.info(f"📁 Models saved to: {output_dir}")
    
    return results

def train_model_comprehensive(family: str, X: np.ndarray, y: np.ndarray, 
                            target: str, strategy: str, feature_names: List[str],
                            caps: Dict[str, Any], routing_meta: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train model using modular trainers directly - enforces runtime policy and routing."""
    
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
def _legacy_train_fallback(family: str, X: np.ndarray, y: np.ndarray, **kwargs):
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

