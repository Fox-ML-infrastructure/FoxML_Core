# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

# common/safety.py
import os, numpy as np, logging, sys
from pathlib import Path
logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded safety defaults")

def _get_safety_config(key: str, default):
    """Get safety config value, with fallback to default."""
    if _CONFIG_AVAILABLE:
        try:
            return get_cfg(f"safety.{key}", default=default, config_name="safety_config")
        except Exception as e:
            logger.debug(f"Failed to load safety config {key}: {e}")
    return default

def set_global_numeric_guards():
    """Don't crash; warn loudly on bad numerics"""
    if _CONFIG_AVAILABLE:
        try:
            error_handling = _get_safety_config("numerical.numpy_error_handling", {})
            np.seterr(
                over=error_handling.get('over', 'warn'),
                invalid=error_handling.get('invalid', 'warn'),
                divide=error_handling.get('divide', 'warn'),
                under=error_handling.get('under', 'ignore')
            )
        except Exception:
            # Fallback to defaults
            np.seterr(over='warn', invalid='warn', divide='warn', under='ignore')
    else:
        np.seterr(over='warn', invalid='warn', divide='warn', under='ignore')

def guard_features(X, clip=None):
    """Clip extreme features to prevent numerical explosions"""
    if clip is None:
        clip_config = _get_safety_config("feature_clipping", {})
        if clip_config.get('enabled', True):
            clip = clip_config.get('clip_value', 1000.0)
        else:
            clip = 1e3  # Default if disabled but function called
    X = np.asarray(X)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -clip, clip, out=X)
    return X

def guard_targets(y, cap_sigma=None):
    """Clip heavy-tailed targets using robust MAD cap"""
    if cap_sigma is None:
        cap_config = _get_safety_config("target_capping", {})
        if cap_config.get('enabled', True):
            cap_sigma = cap_config.get('cap_sigma', 15.0)
        else:
            cap_sigma = 15.0  # Default if disabled but function called
    y = np.asarray(y)
    med = float(np.nanmedian(y))
    mad = float(np.nanmedian(np.abs(y - med))) or 1e-9
    cap = cap_sigma * 1.4826 * mad
    return np.clip(y, med - cap, med + cap)

def finite_preds_or_raise(name, preds):
    """Raise if predictions are non-finite"""
    if not np.all(np.isfinite(preds)):
        raise RuntimeError(f"{name} produced non-finite predictions")

def set_thread_env(omp, mkl=1):
    """Set thread environment variables"""
    os.environ["OMP_NUM_THREADS"] = str(omp)
    os.environ["MKL_NUM_THREADS"] = str(mkl)
    os.environ["OPENBLAS_NUM_THREADS"] = str(mkl)
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

def configure_tf(cpu_only=False, intra=1, inter=1, mem_growth=True):
    """Configure TensorFlow for stability"""
    # Skip TF in child processes if requested
    if os.getenv("TRAINER_CHILD_NO_TF", "0") == "1":
        return
    try:
        import tensorflow as tf, warnings
        if mem_growth:
            try:
                for g in tf.config.list_physical_devices('GPU'):
                    tf.config.experimental.set_memory_growth(g, True)
            except Exception:
                pass
        if cpu_only:
            tf.config.threading.set_intra_op_parallelism_threads(intra)
            tf.config.threading.set_inter_op_parallelism_threads(inter)
        else:
            tf.config.threading.set_intra_op_parallelism_threads(1)
            tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception as e:
        logger.debug(f"TF config skipped: {e}")

def safe_exp(x, lo=None, hi=None):
    """Safe exponential to prevent overflow"""
    if lo is None or hi is None:
        bounds = _get_safety_config("numerical.safe_exp_bounds", {})
        lo = bounds.get('lo', -40.0) if lo is None else lo
        hi = bounds.get('hi', 40.0) if hi is None else hi
    return np.exp(np.clip(x, lo, hi))
