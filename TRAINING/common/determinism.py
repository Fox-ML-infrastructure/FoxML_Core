#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

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

Deterministic Training System
============================

One switch to rule them all: global determinism for reproducible ML training.

This module must be imported FIRST in every entrypoint (before importing 
torch/tf/lightgbm/xgboost) to ensure reproducible results across all model families.

Usage:
    from common.determinism import set_global_determinism, stable_seed_from, seed_for
    
    # Set global determinism (call this FIRST)
    BASE_SEED = set_global_determinism(base_seed=1234, threads=1, deterministic_algorithms=True)
    
    # Derive per-target/fold seeds
    seed = seed_for(target_name, fold_idx, "all_symbols")
"""
from __future__ import annotations
import os
import random
import hashlib
import logging
from typing import Iterable, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Global base seed (set by set_global_determinism)
BASE_SEED = None

def _export_env(env: Dict[str, str]) -> None:
    """Set environment variables for determinism."""
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
    Set global determinism for all ML libraries.
    
    Call this BEFORE importing torch/tensorflow/xgboost/lightgbm.
    
    Args:
        base_seed: Base seed for all random operations
        threads: Number of threads (1 for strict determinism)
        deterministic_algorithms: Enable deterministic algorithms where possible
        prefer_cpu_tree_train: Use CPU for tree training (more deterministic)
        tf_on: Enable TensorFlow determinism
        strict_mode: Enable strict deterministic mode (disables some optimizations)
        
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
        "TF_CPP_MIN_LOG_LEVEL": "3",  # Suppress all TensorFlow warnings
        "TF_ENABLE_ONEDNN_OPTS": "0",  # More deterministic kernels
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
            "random_state": seed,
            "seed": seed,
            "seed_per_iteration": True,
            "nthread": 1,
            "tree_method": os.getenv("XGBOOST_TREE_METHOD", "hist"),
            "verbose": 0,
            **kwargs
        }
    
    elif library == "sklearn":
        return {
            "random_state": seed,
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
    """Log current determinism settings and library versions."""
    logger.info("🔒 Determinism Configuration:")
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
    """Verify that determinism is properly configured."""
    logger.info("🔍 Verifying determinism setup...")
    
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

def test_model_reproducibility(model_class, X, y, target_name: str = "test", fold: int = 0, **kwargs) -> bool:
    """Test if a model class produces reproducible results."""
    logger.info(f"🧪 Testing reproducibility for {model_class.__name__}")
    
    try:
        # Get seed for this target/fold
        seed = seed_for(target_name, fold)
        
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
