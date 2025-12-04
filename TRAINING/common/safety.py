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
"""

# common/safety.py
import os, numpy as np, logging
logger = logging.getLogger(__name__)

def set_global_numeric_guards():
    """Don't crash; warn loudly on bad numerics"""
    np.seterr(over='warn', invalid='warn', divide='warn', under='ignore')

def guard_features(X, clip=1e3):
    """Clip extreme features to prevent numerical explosions"""
    X = np.asarray(X)
    np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    np.clip(X, -clip, clip, out=X)
    return X

def guard_targets(y, cap_sigma=15.0):
    """Clip heavy-tailed targets using robust MAD cap"""
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

def safe_exp(x, lo=-40.0, hi=40.0):
    """Safe exponential to prevent overflow"""
    return np.exp(np.clip(x, lo, hi))
