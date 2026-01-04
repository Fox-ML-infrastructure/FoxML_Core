#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

from __future__ import annotations
import os
import random
import hashlib
import logging
from functools import lru_cache
from typing import Iterable, Optional, Dict, Any, Union

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORT-TIME STRICT ASSERTION
# ============================================================================
# If strict mode (from env) and bootstrap not complete, fail IMMEDIATELY
# This catches "someone imported determinism after importing numpy"
_strict_from_env = os.environ.get("REPRO_MODE", "best_effort").lower() == "strict"
if _strict_from_env:
    if os.environ.get("_REPRO_BOOTSTRAP_COMPLETE") != "1":
        raise RuntimeError(
            "🚨 STRICT MODE: determinism.py imported but repro_bootstrap was not run first!\n"
            "   This means numeric libs may have been imported before thread env vars were set.\n"
            "   Fix: Add 'import TRAINING.common.repro_bootstrap' as the FIRST import in your entrypoint."
        )

# Import numpy only after the assertion (allows this file to fail-fast in strict mode)
import numpy as np

# Global base seed (set by set_global_determinism)
BASE_SEED = None

def _export_env(env: Dict[str, str]) -> None:
    """Set environment variables for reproducibility."""
    for k, v in env.items():
        os.environ.setdefault(k, str(v))

def stable_seed_from(parts: Iterable[str|int], modulo: int = 2**31-1) -> int:
    """
    Generate a stable seed from multiple parts using SHA256.
    
    Args:
        parts: Iterable of strings/ints to combine
        modulo: Modulo to keep seed in int32 range
        
    Returns:
        Stable integer seed
    """
    h = hashlib.sha256(("::".join(map(str, parts))).encode("utf-8")).hexdigest()
    return int(h[:12], 16) % modulo  # 12 hex ~ 48 bits → int32 range

def set_global_determinism(
    base_seed: int = 42,
    threads: int = None,  # Auto-detect optimal threads
    deterministic_algorithms: bool = False,  # Allow parallel execution
    prefer_cpu_tree_train: bool = False,
    tf_on: bool = False,
    strict_mode: bool = False,  # Allow optimizations
) -> int:
    """
    Set global reproducibility configuration for all ML libraries.
    
    Call this BEFORE importing torch/tensorflow/xgboost/lightgbm.
    
    Args:
        base_seed: Base seed for all random operations
        threads: Number of threads (1 for enhanced reproducibility)
        deterministic_algorithms: Enable deterministic algorithms where possible
        prefer_cpu_tree_train: Use CPU for tree training (improves reproducibility)
        tf_on: Enable TensorFlow reproducibility settings
        strict_mode: Enable strict reproducibility mode (disables some optimizations)
        
    Returns:
        The normalized base seed used
    """
    # Auto-detect optimal thread count if not specified
    if threads is None:
        import os
        threads = max(1, (os.cpu_count() or 4) - 1)  # Use all cores except 1
    
    global BASE_SEED
    s = int(base_seed) % (2**31 - 1)
    BASE_SEED = s
    
    logger.info(f"🔒 Setting global determinism: seed={s}, threads={threads}, deterministic={deterministic_algorithms}")

    # Python & OS environment
    # Ensure conda CUDA libraries are in LD_LIBRARY_PATH for TensorFlow
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        conda_targets_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
        current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        # Add conda lib paths if not already present
        new_paths = []
        if conda_lib not in current_ld_path:
            new_paths.append(conda_lib)
        if conda_targets_lib not in current_ld_path:
            new_paths.append(conda_targets_lib)
        if new_paths:
            updated_ld_path = ":".join(new_paths + [current_ld_path] if current_ld_path else new_paths)
            os.environ["LD_LIBRARY_PATH"] = updated_ld_path
    
    _export_env({
        "PYTHONHASHSEED": str(s),
        # Threading & BLAS – fewer threads → less nondeterminism
        "OMP_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "VECLIB_MAXIMUM_THREADS": str(threads),
        "NUMEXPR_NUM_THREADS": str(threads),
        # Force MKL to use GNU OpenMP (libgomp) instead of Intel OpenMP (libiomp5)
        # This prevents conflicts with LightGBM/XGBoost which use libgomp
        "MKL_THREADING_LAYER": "GNU",
        # Intel MKL bitwise compatibility (helps cross-run stability on CPU)
        "MKL_CBWR": "COMPATIBLE",
        "MKL_CBWR_CONDITIONAL": "1",
        # TensorFlow determinism (only read if set pre-import)
        "TF_DETERMINISTIC_OPS": "1" if tf_on else os.environ.get("TF_DETERMINISTIC_OPS", "1"),
        "TF_CUDNN_DETERMINISTIC": "1" if tf_on else os.environ.get("TF_CUDNN_DETERMINISTIC", "1"),
        # Show TensorFlow warnings so user knows if GPU isn't working
        # "TF_CPP_MIN_LOG_LEVEL": "3",  # Removed - show warnings
        "TF_ENABLE_ONEDNN_OPTS": "0",  # More deterministic kernels
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",  # Allow memory growth
        # "TF_LOGGING_VERBOSITY": "ERROR",  # Removed - show warnings
        # GPU memory limits
        "CUDA_VISIBLE_DEVICES": "0",  # Use only GPU 0
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",  # Allow memory growth
        # Suppress XGBoost warnings
        "XGBOOST_VERBOSE": "0",
        # Suppress sklearn warnings
        "SKLEARN_WARN_ON_IMPORT": "0"
    })

    # Suppress warnings globally
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*gpu_id.*")
    warnings.filterwarnings("ignore", message=".*tree method.*")
    warnings.filterwarnings("ignore", message=".*Parameters.*not used.*")
    warnings.filterwarnings("ignore", message=".*Skipping features.*")
    warnings.filterwarnings("ignore", message=".*Early stopping.*")
    warnings.filterwarnings("ignore", message=".*Learning rate reduction.*")
    
    # Set Python random seeds
    random.seed(s)

    # NumPy
    try:
        import numpy as np
        np.random.seed(s)
        logger.info("✅ NumPy seed set")
    except Exception as e:
        logger.warning(f"NumPy seed setting failed: {e}")

    # PyTorch (optional)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        if deterministic_algorithms:
            # Stronger setting (may raise on non-deterministic ops)
            torch.use_deterministic_algorithms(True, warn_only=False)
        # CUDNN flags
        try:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
        except Exception:
            pass
        logger.info("✅ PyTorch determinism set")
    except Exception:
        logger.info("PyTorch not available")

    # TensorFlow (optional) - skip in child processes if requested
    if tf_on and os.getenv("TRAINER_CHILD_NO_TF", "0") != "1":
        try:
            import tensorflow as tf
            tf.random.set_seed(s)
            
            # Configure GPU memory growth for 8GB+ GPUs
            try:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        # Set memory limit to 8GB (8192 MB) - full utilization
                        tf.config.experimental.set_virtual_device_configuration(
                            gpu,
                            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)]
                        )
                        logger.info("✅ TensorFlow GPU memory configured (8GB limit)")
                else:
                    logger.info("✅ TensorFlow CPU mode")
            except Exception as e:
                logger.warning(f"TensorFlow GPU config failed: {e}")
            
            logger.info("✅ TensorFlow seed set")
        except Exception:
            logger.info("TensorFlow not available")

    # Tree learners default to CPU for strict reproducibility if requested
    if prefer_cpu_tree_train:
        os.environ.setdefault("XGBOOST_TREE_METHOD", "hist")
        os.environ.setdefault("XGB_USE_GPU", "0")
        os.environ.setdefault("LGBM_USE_GPU", "0")
        logger.info("✅ CPU-only tree training enabled")

    return s

def seed_for(target: str, fold: int, symbol_group: Optional[str] = None) -> int:
    """
    Generate a stable seed for a specific target/fold combination.
    
    Args:
        target: Target name (e.g., "fwd_ret_5m")
        fold: Fold number
        symbol_group: Optional symbol group identifier
        
    Returns:
        Stable seed for this target/fold combination
    """
    if BASE_SEED is None:
        raise RuntimeError("set_global_determinism() must be called first")
    
    parts = [BASE_SEED, target, f"fold={fold}"]
    if symbol_group:
        parts.append(f"group={symbol_group}")
    
    seed = stable_seed_from(parts)
    logger.debug(f"Seed lineage: base={BASE_SEED} target={target} fold={fold} → seed={seed}")
    return seed

def get_deterministic_params(library: str, seed: int, **kwargs) -> Dict[str, Any]:
    """
    Get deterministic parameters for specific ML libraries.
    
    Args:
        library: Library name ("lightgbm", "xgboost", "sklearn", "torch", "tf")
        seed: Seed to use
        **kwargs: Additional parameters
        
    Returns:
        Dictionary of deterministic parameters
    """
    if library == "lightgbm":
        return {
            "objective": kwargs.get("objective", "regression"),
            "metric": kwargs.get("metric", "mae"),
            "deterministic": True,
            "seed": seed,
            "feature_fraction_seed": seed + 1,
            "bagging_seed": seed + 2,
            "data_random_seed": seed + 3,
            "num_threads": 1,
            "bagging_freq": 0,  # Disable stochastic bagging
            "verbose": -1,
            **kwargs
        }
    
    elif library == "xgboost":
        return {
            "objective": kwargs.get("objective", "reg:squarederror"),
            "seed": seed,
            "seed": seed,
            "seed_per_iteration": True,
            "nthread": 1,
            "tree_method": os.getenv("XGBOOST_TREE_METHOD", "hist"),
            "verbose": 0,
            **kwargs
        }
    
    elif library == "sklearn":
        return {
            "seed": seed,
            **kwargs
        }
    
    elif library == "torch":
        return {
            "generator": f"torch.Generator().manual_seed({seed})",
            "worker_init_fn": f"lambda worker_id: _worker_init_fn({seed}, worker_id)",
            **kwargs
        }
    
    else:
        return kwargs

def _worker_init_fn(seed: int, worker_id: int) -> None:
    """Worker initialization function for PyTorch DataLoader."""
    import numpy as np
    import random
    import torch
    
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def reproducibility_test(train_fn, data, target: str, fold: int, **kwargs) -> bool:
    """
    Run a reproducibility test to verify deterministic training.
    
    Args:
        train_fn: Training function that returns (predictions, model_dump)
        data: Training data
        target: Target name
        fold: Fold number
        **kwargs: Additional arguments for train_fn
        
    Returns:
        True if test passes (identical results), False otherwise
    """
    import numpy as np
    from copy import deepcopy
    
    logger.info(f"🧪 Running reproducibility test for {target} fold {fold}")
    
    # Get seed for this target/fold
    seed = seed_for(target, fold)
    
    # First run
    preds1, dump1 = train_fn(seed=seed, data=data, **kwargs)
    
    # Second run with same seed
    preds2, dump2 = train_fn(seed=seed, data=deepcopy(data), **kwargs)
    
    # Check if results are identical
    preds_equal = np.array_equal(preds1, preds2) if preds1 is not None and preds2 is not None else True
    dump_equal = dump1 == dump2 if dump1 is not None and dump2 is not None else True
    
    if preds_equal and dump_equal:
        logger.info("✅ Reproducibility test PASSED")
        return True
    else:
        logger.error("❌ Reproducibility test FAILED")
        logger.error(f"Predictions equal: {preds_equal}")
        logger.error(f"Model dumps equal: {dump_equal}")
        return False

def log_determinism_info():
    """Log current reproducibility settings and library versions."""
    logger.info("🔒 Reproducibility Configuration:")
    logger.info(f"  Base seed: {BASE_SEED}")
    logger.info(f"  Python hash seed: {os.environ.get('PYTHONHASHSEED')}")
    logger.info(f"  Threads: {os.environ.get('OMP_NUM_THREADS')}")
    logger.info(f"  TF deterministic: {os.environ.get('TF_DETERMINISTIC_OPS')}")
    
    # Log library versions
    try:
        import numpy as np
        logger.info(f"  NumPy: {np.__version__}")
    except:
        pass
    
    try:
        import torch
        logger.info(f"  PyTorch: {torch.__version__}")
    except:
        pass
    
    try:
        import tensorflow as tf
        logger.info(f"  TensorFlow: {tf.__version__}")
    except:
        pass
    
    try:
        import lightgbm as lgb
        logger.info(f"  LightGBM: {lgb.__version__}")
    except:
        pass
    
    try:
        import xgboost as xgb
        logger.info(f"  XGBoost: {xgb.__version__}")
    except:
        pass

def verify_determinism_setup() -> bool:
    """Verify that reproducibility is properly configured."""
    logger.info("🔍 Verifying reproducibility setup...")
    
    checks = []
    
    # Check environment variables
    env_vars = [
        "PYTHONHASHSEED", "OMP_NUM_THREADS", "TF_DETERMINISTIC_OPS"
    ]
    
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            logger.info(f"  ✅ {var}={value}")
            checks.append(True)
        else:
            logger.warning(f"  ⚠️  {var} not set")
            checks.append(False)
    
    # Check if BASE_SEED is set
    if BASE_SEED is not None:
        logger.info(f"  ✅ BASE_SEED={BASE_SEED}")
        checks.append(True)
    else:
        logger.error("  ❌ BASE_SEED not set")
        checks.append(False)
    
    success = all(checks)
    if success:
        logger.info("✅ Determinism setup verified")
    else:
        logger.error("❌ Determinism setup incomplete")
    
    return success

def create_deterministic_test_data(n_samples: int = 100, n_features: int = 10, seed: int = None) -> tuple:
    """Create deterministic test data for reproducibility testing."""
    import numpy as np
    
    if seed is None:
        seed = BASE_SEED or 42
    
    # Set numpy seed for deterministic data generation
    np.random.seed(seed)
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    return X, y

def test_model_reproducibility(model_class, X, y, target: str = "test", fold: int = 0, **kwargs) -> bool:
    """Test if a model class produces reproducible results."""
    logger.info(f"🧪 Testing reproducibility for {model_class.__name__}")
    
    try:
        # Get seed for this target/fold
        seed = seed_for(target, fold)
        
        # First run
        model1 = model_class()
        if hasattr(model1, 'train'):
            model1.train(X, y, seed=seed, **kwargs)
            preds1 = model1.predict(X) if hasattr(model1, 'predict') else None
        else:
            model1.fit(X, y)
            preds1 = model1.predict(X)
        
        # Second run with same seed
        model2 = model_class()
        if hasattr(model2, 'train'):
            model2.train(X, y, seed=seed, **kwargs)
            preds2 = model2.predict(X) if hasattr(model2, 'predict') else None
        else:
            model2.fit(X, y)
            preds2 = model2.predict(X)
        
        # Check if results are identical
        if preds1 is not None and preds2 is not None:
            identical = np.array_equal(preds1, preds2)
            if identical:
                logger.info(f"✅ {model_class.__name__}: Reproducible")
                return True
            else:
                logger.error(f"❌ {model_class.__name__}: Not reproducible")
                return False
        else:
            logger.warning(f"⚠️  {model_class.__name__}: No predictions to compare")
            return True
            
    except Exception as e:
        logger.error(f"❌ {model_class.__name__}: Error during testing - {e}")
        return False


# ============================================================================
# UNIFIED CONFIG LOADING (ENV > YAML)
# ============================================================================

@lru_cache(maxsize=1)
def load_reproducibility_config() -> Dict[str, Any]:
    """
    Load reproducibility config (cached).
    
    SINGLE SOURCE OF TRUTH:
    1. Environment variables override YAML (for launcher/CI control)
    2. YAML provides defaults
    3. Log if YAML disagrees with env (diagnostic)
    """
    try:
        from CONFIG.config_loader import get_cfg
        
        # Read YAML values
        yaml_mode = get_cfg("reproducibility.mode", default="best_effort", config_name="reproducibility")
        yaml_seed = get_cfg("reproducibility.seed", default=42, config_name="reproducibility")
        yaml_version = get_cfg("reproducibility.version", default="v1", config_name="reproducibility")
        
        # Read subsettings (these are NOT dead config - we use them)
        yaml_require_env = get_cfg("reproducibility.strict.require_env_vars", default=True, config_name="reproducibility")
        yaml_disable_gpu = get_cfg("reproducibility.strict.disable_gpu_tree_models", default=True, config_name="reproducibility")
        yaml_single_thread = get_cfg("reproducibility.strict.force_single_thread", default=True, config_name="reproducibility")
        yaml_stable_ordering = get_cfg("reproducibility.strict.enforce_stable_ordering", default=True, config_name="reproducibility")
        
        # ENV OVERRIDES YAML (single source of truth)
        env_mode = os.environ.get("REPRO_MODE")
        env_seed = os.environ.get("REPRO_SEED")
        env_version = os.environ.get("REPRO_VERSION")
        
        final_mode = env_mode.lower() if env_mode else yaml_mode
        final_seed = int(env_seed) if env_seed else yaml_seed
        final_version = env_version if env_version else yaml_version
        
        # Log if disagreement (diagnostic)
        if env_mode and env_mode.lower() != yaml_mode:
            logger.info(f"🔧 REPRO_MODE env ({env_mode}) overrides YAML ({yaml_mode})")
        
        return {
            "mode": final_mode,
            "seed": final_seed,
            "version": final_version,
            # Subsettings (used by strict mode)
            "require_env_vars": yaml_require_env,
            "disable_gpu_tree_models": yaml_disable_gpu,
            "force_single_thread": yaml_single_thread,
            "enforce_stable_ordering": yaml_stable_ordering,
        }
    except Exception:
        # Fallback to env var + conservative defaults
        mode = os.environ.get("REPRO_MODE", "best_effort").lower()
        return {
            "mode": mode,
            "seed": 42,
            "version": "v1",
            "require_env_vars": True,
            "disable_gpu_tree_models": True,
            "force_single_thread": True,
            "enforce_stable_ordering": True,
        }


def is_strict_mode() -> bool:
    """Check if strict mode is enabled."""
    return load_reproducibility_config()["mode"] == "strict"


# ============================================================================
# SEED UTILITIES
# ============================================================================

def normalize_seed(seed: int) -> int:
    """
    Normalize seed to safe range [1, 2^31-2].
    
    Prevents seed=0 (some libraries treat as random) and overflow.
    """
    seed = seed % (2**31 - 1)
    if seed == 0:
        seed = 1
    return seed


def resolve_seed(
    global_seed: int,
    phase: str,
    target: Optional[str] = None,
    fold: int = 0,
    symbol: Optional[str] = None,
    view: Optional[str] = None,
    version: Optional[str] = None,
) -> int:
    """
    Derive a stable, versioned seed for a specific context.
    
    Uses SHA256 for stable derivation (not Python's hash()).
    
    Args:
        global_seed: Base seed
        phase: Phase name (e.g., "target_ranking", "feature_selection", "model_training")
        target: Target name
        fold: Fold number
        symbol: Symbol name
        view: View name
        version: Seed version (for invalidating old seeds)
    
    Returns:
        Normalized seed for this context
    """
    if version is None:
        version = load_reproducibility_config()["version"]
    
    payload = f"{global_seed}|{phase}|{target}|{fold}|{symbol}|{view}|{version}"
    h = hashlib.sha256(payload.encode()).hexdigest()
    raw_seed = int(h[:8], 16)
    return normalize_seed(raw_seed)


def seed_all(seed: Optional[int] = None) -> int:
    """
    Set all random seeds (Python, NumPy, Torch).
    
    Args:
        seed: Seed to use (defaults to config seed)
    
    Returns:
        The seed used
    """
    if seed is None:
        seed = load_reproducibility_config()["seed"]
    
    seed = normalize_seed(seed)
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if is_strict_mode():
            torch.use_deterministic_algorithms(True, warn_only=True)
            try:
                import torch.backends.cudnn as cudnn
                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass
    except ImportError:
        pass
    
    logger.info(f"🎲 seed_all: seed={seed}, mode={'strict' if is_strict_mode() else 'best_effort'}")
    return seed


# ============================================================================
# STABLE ORDERING
# ============================================================================

def stable_sort(items: list, key=None) -> list:
    """
    Sort items for deterministic ordering.
    
    Sorts if: strict mode AND enforce_stable_ordering=True (from config)
    Otherwise: returns as-is.
    
    Use at KEY BOUNDARIES only:
    - Feature names before model training
    - Target names before job creation
    - Fold indices before CV loop
    - Symbol names before iteration
    
    Args:
        items: List to sort
        key: Optional key function for sorting (default: str)
    
    Returns:
        Sorted list if enforced, original list otherwise
    """
    config = load_reproducibility_config()
    if is_strict_mode() and config.get("enforce_stable_ordering", True):
        return sorted(items, key=key or str)
    return items


def stable_sort_dict_keys(d: dict) -> list:
    """Get dict keys in stable order (for strict mode)."""
    return stable_sort(list(d.keys()))


# ============================================================================
# LIBRARY-SPECIFIC PARAMETERS (sklearn API keys)
# ============================================================================

def get_library_params(library: str, seed: int, strict: bool = None) -> Dict[str, Any]:
    """
    Get deterministic parameters for specific ML libraries.
    
    Uses sklearn-facing API keys (n_jobs, random_state, etc.)
    
    Args:
        library: Library name ("lightgbm", "xgboost", "catboost", "sklearn", "randomforest")
        seed: Seed to use
        strict: Force strict mode (defaults to config)
    
    Returns:
        Dictionary of library-specific parameters
    """
    if strict is None:
        strict = is_strict_mode()
    
    if library == "lightgbm":
        params = {
            # sklearn API keys
            "n_jobs": 1 if strict else -1,
            "random_state": seed,
            # Core params (passed through)
            "deterministic": True,
            "force_row_wise": True,
            "seed": seed,
            "feature_fraction_seed": normalize_seed(seed + 1),
            "bagging_seed": normalize_seed(seed + 2),
            "data_random_seed": normalize_seed(seed + 3),
            "verbose": -1,
        }
        if strict:
            params["device_type"] = "cpu"
        return params
    
    elif library == "xgboost":
        params = {
            # sklearn API keys
            "n_jobs": 1 if strict else -1,
            "random_state": seed,
            # Core params
            "verbosity": 0,
        }
        if strict:
            params["device"] = "cpu"
            params["tree_method"] = "hist"
        return params
    
    elif library == "catboost":
        params = {
            # CatBoost uses thread_count and random_seed (not sklearn standard)
            "thread_count": 1 if strict else -1,
            "random_seed": seed,
            "verbose": False,
        }
        if strict:
            params["task_type"] = "CPU"
        return params
    
    elif library in ("randomforest", "random_forest"):
        return {
            "n_jobs": 1 if strict else -1,
            "random_state": seed,
        }
    
    elif library == "sklearn":
        return {
            "random_state": seed,
        }
    
    else:
        return {"random_state": seed}


# ============================================================================
# SINGLE CHOKE POINT: create_estimator()
# ============================================================================

def create_estimator(
    library: str,
    base_config: Dict[str, Any],
    seed: int,
    problem_kind: str = "regression",  # "regression" or "classification"
) -> Any:
    """
    THE SINGLE CHOKE POINT for all model creation.
    
    All estimator instantiation MUST go through this function.
    This ensures determinism params are always applied correctly.
    
    Args:
        library: Library name ("lightgbm", "xgboost", "catboost", "randomforest", "sklearn")
        base_config: Base configuration dict
        seed: Seed to use
        problem_kind: "regression" or "classification"
    
    Returns:
        Instantiated estimator
    
    Example:
        model = create_estimator(
            library="lightgbm",
            base_config={"n_estimators": 100},
            seed=42,
            problem_kind="regression"
        )
    """
    strict = is_strict_mode()
    
    # Get determinism params for this library
    det_params = get_library_params(library, seed, strict=strict)
    
    # Merge: determinism params override base config
    merged = {**base_config, **det_params}
    
    # NOTE: Do NOT delete verbose/verbosity - we intentionally set them above
    # LightGBM uses verbose=-1, XGBoost uses verbosity=0, CatBoost uses verbose=False
    # These don't conflict because we instantiate one library at a time
    
    # Log effective params
    mode_str = "STRICT" if strict else "BEST_EFFORT"
    logger.info(f"🔧 create_estimator({library}, {problem_kind}, {mode_str}):")
    logger.info(f"   seed={seed}, n_jobs={merged.get('n_jobs', 'N/A')}")
    if strict:
        logger.info(f"   device={merged.get('device_type', merged.get('device', merged.get('task_type', 'N/A')))}")
    
    # Instantiate
    return _instantiate_estimator(library, problem_kind, merged)


def _instantiate_estimator(library: str, problem_kind: str, params: Dict[str, Any]) -> Any:
    """
    Internal: Instantiate the actual estimator class.
    
    DO NOT call this directly - use create_estimator().
    """
    is_classifier = problem_kind == "classification"
    
    if library == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if is_classifier else lgb.LGBMRegressor
        return cls(**params)
    
    elif library == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBClassifier if is_classifier else xgb.XGBRegressor
        return cls(**params)
    
    elif library == "catboost":
        from catboost import CatBoostClassifier, CatBoostRegressor
        cls = CatBoostClassifier if is_classifier else CatBoostRegressor
        return cls(**params)
    
    elif library in ("randomforest", "random_forest"):
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        cls = RandomForestClassifier if is_classifier else RandomForestRegressor
        return cls(**params)
    
    elif library == "sklearn":
        # Generic sklearn - caller must provide the class
        raise ValueError("For sklearn, use specific library names or instantiate directly")
    
    else:
        raise ValueError(f"Unknown library: {library}. Supported: lightgbm, xgboost, catboost, randomforest")
