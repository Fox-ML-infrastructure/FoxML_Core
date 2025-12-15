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

# project root: TRAINING/training_strategies/utils.py -> parents[2] = repo root
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


"""Utility functions for training strategies."""

def setup_logging(log_level: str = "INFO", logfile: str = "training_strategies.log"):
    import queue
    root = logging.getLogger()
    root.setLevel(getattr(logging, log_level.upper()))
    # remove any existing handlers
    for h in list(root.handlers): root.removeHandler(h)
    q = queue.SimpleQueue()  # lock-free
    qh = logging.handlers.QueueHandler(q)
    root.addHandler(qh)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler(logfile);      fh.setFormatter(fmt)
    listener = logging.handlers.QueueListener(q, sh, fh)
    listener.daemon = True
    listener.start()
    return listener

# --- THREAD/OMP GUARD: put this at the VERY top, once ---
import os, time as _t
def _now(): return _t.perf_counter()
def safe_duration(t0): 
    try: return f"{_t.perf_counter()-t0:.2f}s"
    except Exception: return "n/a"

# global knobs filled by main()
# Load from config if available, otherwise use defaults
if _CONFIG_AVAILABLE:
    pipeline_cfg = get_pipeline_config()
    threading_cfg = get_cfg("threading.defaults.default_threads", config_name="threading_config")
    if threading_cfg is None:
        THREADS = max(1, (os.cpu_count() or 2) - 1)
    else:
        THREADS = threading_cfg if isinstance(threading_cfg, int) else max(1, (os.cpu_count() or 2) - 1)
    MKL_THREADS_DEFAULT = get_cfg("threading.defaults.mkl_threads", default=1, config_name="threading_config")
else:
    THREADS = max(1, (os.cpu_count() or 2) - 1)
    MKL_THREADS_DEFAULT = 1
CPU_ONLY = False

def _pkg_ver(name):
    """Dynamic package version detection"""
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "missing"

def _env_guard(omp_threads: int, mkl_threads: int = 1):
    os.environ["OMP_NUM_THREADS"]   = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(mkl_threads)
    os.environ["MKL_NUM_THREADS"]      = str(mkl_threads)
    os.environ["NUMEXPR_NUM_THREADS"]  = "1"
    os.environ["KMP_AFFINITY"]         = "disabled"
    os.environ["KMP_BLOCKTIME"]        = "0"
    os.environ["MKL_THREADING_LAYER"]  = "GNU"
    os.environ["JOBLIB_START_METHOD"]  = "spawn"

_env_guard(max(1, (os.cpu_count() or 2) - 1), mkl_threads=1)

# multiprocessing start method BEFORE heavy imports

# TF env before any TF import
# Load from config if available
if _CONFIG_AVAILABLE:
    python_hash_seed = get_cfg("pipeline.determinism.python_hash_seed", default="42")
    tf_deterministic = get_cfg("pipeline.determinism.tf_deterministic_ops", default="1")
    os.environ.setdefault("PYTHONHASHSEED", python_hash_seed)
    os.environ.setdefault("TF_DETERMINISTIC_OPS", tf_deterministic)
else:
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
# TF_CPP_MIN_LOG_LEVEL already set at top of file

# CRITICAL: Import determinism FIRST before any ML libraries
from TRAINING.common.determinism import set_global_determinism, stable_seed_from, seed_for, get_deterministic_params, log_determinism_info

# Set global determinism immediately - OPTIMIZED FOR PERFORMANCE
# Load base_seed from config if available
if _CONFIG_AVAILABLE:
    base_seed = get_cfg("pipeline.determinism.base_seed", default=42)
else:
    base_seed = 42  # FALLBACK_DEFAULT_OK
BASE_SEED = set_global_determinism(
    base_seed=base_seed,
    threads=None,  # Auto-detect optimal thread count
    deterministic_algorithms=False,  # Allow parallel algorithms
    prefer_cpu_tree_train=False,  # Use GPU when available
    tf_on=True,  # Enable TensorFlow GPU
    strict_mode=False  # Allow optimizations
)

import argparse
import logging
# Removed duplicate import: numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Type alias for consistent 8-tuple returns
Eight = Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray, List[str], Optional[np.ndarray], Any]
import sys
# Removed duplicate import: os
import warnings
import joblib
from datetime import datetime
# Removed unused import: glob
import time

# Polars optimization (same as original script)
# Use global THREADS variable (already loaded from config if available)
DEFAULT_THREADS = str(THREADS)
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"
if USE_POLARS:
    try:
        os.environ.setdefault("POLARS_MAX_THREADS", DEFAULT_THREADS)
        import polars as pl
        pl.enable_string_cache()
        print("🚀 Polars optimization enabled")
    except ImportError:
        USE_POLARS = False
        print("Polars not available, falling back to pandas")

# Set up environment variables for determinism (same as original)
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Set random seeds
np.random.seed(42)
import random
random.seed(42)

# Suppress warnings
warnings.filterwarnings("ignore", message=r"Protobuf gencode version .* is exactly one major version older")

# Import modular training system
# Note: Strategy classes (SingleTaskStrategy, MultiTaskStrategy, CascadeStrategy) are not currently defined
# They may have been removed during refactoring or are defined elsewhere
# Removed unused imports: ModelFactory, DataPreprocessor, TargetResolver, ValidationUtils

# Import target router for enhanced target support
from TRAINING.target_router import route_target, spec_from_target, TaskSpec

# Import existing utilities
try:
    from target_resolver import safe_target_extraction, pick_single_target_column
    from memory_manager import aggressive_cleanup, monitor_memory
except ImportError:
    # Fallback if scripts module not available
    def safe_target_extraction(df, target):
        return df[target], target
    
    def pick_single_target_column(df, target):
        return target
    
    def aggressive_cleanup():
        import gc
        gc.collect()
    
    def monitor_memory():
        return {}

logger = logging.getLogger(__name__)

# All 20 model families from original script
ALL_FAMILIES = [
    'LightGBM', 'XGBoost', 'MLP', 'CNN1D', 'LSTM', 'Transformer',
    'TabCNN', 'TabLSTM', 'TabTransformer', 'RewardBased',
    'QuantileLightGBM', 'NGBoost', 'GMMRegime', 'ChangePoint',
    'FTRLProximal', 'VAE', 'GAN', 'Ensemble', 'MetaLearning', 'MultiTask'
]

# Model type classification for efficient training
CROSS_SECTIONAL_MODELS = [
    'LightGBM', 'XGBoost', 'MLP', 'Ensemble', 'RewardBased',
    'QuantileLightGBM', 'NGBoost', 'GMMRegime', 'ChangePoint',
    'FTRLProximal', 'VAE', 'GAN', 'MetaLearning', 'MultiTask'
]

SEQUENTIAL_MODELS = [
    'CNN1D', 'LSTM', 'Transformer', 'TabCNN', 'TabLSTM', 'TabTransformer'
]

# PyTorch sequential families (for better performance)
TORCH_SEQ_FAMILIES = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}

def normalize_family_name(family: str) -> str:
    """
    Normalize model family name to title case for capabilities map lookup.
    
    Handles common variations:
    - lightgbm -> LightGBM
    - xgboost -> XGBoost
    - random_forest -> RandomForest
    - neural_network -> NeuralNetwork
    - mutual_information -> MutualInformation
    - univariate_selection -> UnivariateSelection
    - lasso -> Lasso
    - catboost -> CatBoost
    """
    # Handle special cases first
    special_cases = {
        "lightgbm": "LightGBM",
        "xgboost": "XGBoost",
        "catboost": "CatBoost",
        "ngboost": "NGBoost",
        "random_forest": "RandomForest",
        "neural_network": "NeuralNetwork",
        "mutual_information": "MutualInformation",
        "univariate_selection": "UnivariateSelection",
        "quantilelightgbm": "QuantileLightGBM",
        "ftrlproximal": "FTRLProximal",
        "gmmregime": "GMMRegime",
        "changepoint": "ChangePoint",
        "rewardbased": "RewardBased",
        "metalearning": "MetaLearning",
        "multitask": "MultiTask",
    }
    
    family_lower = family.lower()
    if family_lower in special_cases:
        return special_cases[family_lower]
    
    # For others, try title case (handles most cases)
    # Replace underscores and title case
    return family.replace("_", "").title()


# Family capabilities map (from original script)
FAMILY_CAPS = {
    "LightGBM": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "XGBoost": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "MLP": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "CNN1D": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "LSTM": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "Transformer": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "TabCNN": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "TabLSTM": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "TabTransformer": {"nan_ok": False, "needs_tf": False, "backend": "torch", "experimental": False, "preprocess_in_family": True},
    "RewardBased": {"nan_ok": False, "needs_tf": False, "experimental": False},
    "QuantileLightGBM": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "NGBoost": {"nan_ok": False, "needs_tf": False, "experimental": True},
    "GMMRegime": {"nan_ok": False, "needs_tf": False, "experimental": True, "feature_emitter": False},
    "ChangePoint": {"nan_ok": False, "needs_tf": False, "experimental": True, "feature_emitter": False},
    "FTRLProximal": {"nan_ok": False, "needs_tf": False, "experimental": False},
    "VAE": {"nan_ok": False, "needs_tf": True, "experimental": True},
    "GAN": {"nan_ok": False, "needs_tf": True, "experimental": True},
    "Ensemble": {"nan_ok": False, "needs_tf": False, "experimental": False},
    "MetaLearning": {"nan_ok": False, "needs_tf": True, "experimental": True},
    "MultiTask": {"nan_ok": False, "needs_tf": True, "experimental": True},
    # Additional families (feature selection methods)
    "Lasso": {"nan_ok": False, "needs_tf": False, "experimental": False},
    "RandomForest": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "CatBoost": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "NeuralNetwork": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "MutualInformation": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "UnivariateSelection": {"nan_ok": True, "needs_tf": False, "experimental": False}
}


def build_sequences_from_features(X, lookback=None):
    """
    Convert 2D features (N, F) to 3D sequences (N', T, F) using rolling windows.
    
    Args:
        X: (N, F) feature matrix
        lookback: sequence length T (loads from config if None)
        
    Returns:
        X_seq: (N', T, F) where N' = N - lookback + 1
    """
    # Load lookback from config if not provided
    if lookback is None:
        if _CONFIG_AVAILABLE:
            lookback = get_cfg("pipeline.sequential.default_lookback", default=64)
        else:
            lookback = 64
    N, F = X.shape
    if N <= lookback:
        # If not enough data, pad with zeros
        X_seq = np.zeros((1, lookback, F), dtype=np.float32)
        X_seq[0, :N, :] = X
        return X_seq
    
    # Create rolling windows
    X_seq = np.zeros((N - lookback + 1, lookback, F), dtype=np.float32)
    for i in range(N - lookback + 1):
        X_seq[i] = X[i:i + lookback]
    
    return X_seq

def tf_available():
    """Check if TensorFlow is available.
    
    Note: This is a lenient check - we only verify the module can be imported.
    Full initialization happens in child processes, so we don't need to verify
    GPU availability or full library loading here.
    """
    try:
        import tensorflow as tf  # noqa: F401
        return True
    except (ImportError, Exception):
        # Catch all exceptions - TensorFlow might fail to import for various reasons
        # (library loading, CUDA issues, etc.) but child processes will handle it
        return False

def ngboost_available():
    """Check if NGBoost is available."""
    try:
        import ngboost  # noqa
        return True
    except Exception:
        return False

def pick_tf_device():
    """Dynamically detect best TensorFlow device (GPU if available, CPU otherwise)."""
    if os.getenv("CPU_ONLY", "0") == "1":
        return "/CPU:0"
    try:
        import tensorflow as tf
        gpus = tf.config.list_logical_devices("GPU")
        return "/GPU:0" if gpus else "/CPU:0"
    except Exception:
        return "/CPU:0"

TF_DEVICE = pick_tf_device()
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    logger.info(f"TF device: {TF_DEVICE} | GPUs: {len(gpus)}")
except Exception:
    logger.info(f"TF device: {TF_DEVICE}")


