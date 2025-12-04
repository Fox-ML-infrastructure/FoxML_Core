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

 # ---- PATH BOOTSTRAP: ensure project root on sys.path in parent AND children ----
import os, sys
from pathlib import Path

# project root likely: .../secure/trader (parent of TRAINING)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]

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

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
import atexit
# Set persistent temp folder for joblib memmapping
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

def _run_family_inproc(family: str, X, y, total_threads: int = 12, trainer_kwargs: dict | None = None):
    """
    Runs a family trainer in the main process with unified threading control.
    - No multiprocessing, no payload temp files, no IPC.
    - Uses plan_for_family + thread_guard to clamp pools.
    - Configures TF safely if the family uses it.
    
    Args:
        family: Model family name
        X: Training features
        y: Training targets
        total_threads: Total threads available
        trainer_kwargs: Additional trainer arguments
    
    Returns:
        Trained model
    """
    import importlib
    
    # Reset affinity and threadpools BEFORE each family to prevent inherited pinning
    from common.threads import reset_affinity, reset_threadpools
    reset_affinity(logger)
    reset_threadpools()
    
    # Module mapping
    MODMAP = {
        "LightGBM":           ("model_fun.lightgbm_trainer",        "LightGBMTrainer"),
        "QuantileLightGBM":   ("model_fun.quantile_lightgbm_trainer","QuantileLightGBMTrainer"),
        "XGBoost":            ("model_fun.xgboost_trainer",          "XGBoostTrainer"),
        "RewardBased":        ("model_fun.reward_based_trainer",     "RewardBasedTrainer"),
        "GMMRegime":          ("model_fun.gmm_regime_trainer",       "GMMRegimeTrainer"),
        "ChangePoint":        ("model_fun.change_point_trainer",     "ChangePointTrainer"),
        "NGBoost":            ("model_fun.ngboost_trainer",          "NGBoostTrainer"),
        "Ensemble":           ("model_fun.ensemble_trainer",         "EnsembleTrainer"),
        "FTRLProximal":       ("model_fun.ftrl_proximal_trainer",    "FTRLProximalTrainer"),
        "MLP":                ("model_fun.mlp_trainer",              "MLPTrainer"),
        "VAE":                ("model_fun.vae_trainer",              "VAETrainer"),
        "GAN":                ("model_fun.gan_trainer",              "GANTrainer"),
        "MetaLearning":       ("model_fun.meta_learning_trainer",    "MetaLearningTrainer"),
        "MultiTask":          ("model_fun.multi_task_trainer",       "MultiTaskTrainer"),
    }
    
    plan = plan_for_family(family, total_threads)
    omp, mkl = plan["OMP"], plan["MKL"]
    
    logger.info(f"[InProc] Training {family} with OMP={omp}, MKL={mkl}")
    
    # Best-effort CUDA visibility: keep CPU families off the GPU
    # Save original CVD to restore after CPU families
    original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    
    if family in CPU_FAMS:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # -1 avoids CUDA toolkit probing
        logger.info(f"[InProc] {family} is CPU-only, hiding GPUs")
    elif original_cvd == "-1" or original_cvd == "":
        # Restore GPU visibility if it was hidden by previous CPU family
        os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("TRAINER_GPU_IDS", "0")
        logger.info(f"[InProc] {family} is GPU-capable, restored CUDA_VISIBLE_DEVICES=0")
    
    # Configure TF (safe, idempotent)
    if family in TF_FAMS:
        try:
            tf = tf_thread_setup(intra=omp, inter=max(1, min(2, omp // 2)), allow_growth=True)
            logger.info(f"[TF] set intra_op={omp} inter_op={max(1, min(2, omp // 2))}")
        except Exception as e:
            logger.warning(f"[TF] threading setup skipped: {e}")
    
    # Clamp threadpools for this fit()
    with thread_guard(omp=omp, mkl=mkl):
        # Import and instantiate trainer
        mod_name, cls_name = MODMAP[family]
        Trainer = getattr(importlib.import_module(mod_name), cls_name)
        trainer = Trainer(**(trainer_kwargs or {}))
        
        # Push n_jobs/nthread into known estimators
        for attr in ("model", "est", "base_model", "estimator"):
            if hasattr(trainer, attr):
                try:
                    set_estimator_threads(getattr(trainer, attr), omp)
                except Exception:
                    pass
        
        # Train (wrapped in family_run_scope for clean threading)
        from common.threads import family_run_scope
        with family_run_scope(family, total_threads):
            result = trainer.train(X, y)
        logger.info(f"[InProc] {family} training completed successfully")
        return result

def _run_family_isolated(family: str, X, y, timeout_s: int = 7200,
                         omp_threads: int | None = None, mkl_threads: int | None = None,
                         trainer_kwargs: dict | None = None):
    import tempfile, joblib, multiprocessing as mp, os, time as _time, numpy as np, shutil

    MODMAP = {
        "LightGBM":           ("model_fun.lightgbm_trainer",        "LightGBMTrainer"),
        "QuantileLightGBM":   ("model_fun.quantile_lightgbm_trainer","QuantileLightGBMTrainer"),
        "XGBoost":            ("model_fun.xgboost_trainer",          "XGBoostTrainer"),
        "RewardBased":        ("model_fun.reward_based_trainer",     "RewardBasedTrainer"),
        "GMMRegime":          ("model_fun.gmm_regime_trainer",       "GMMRegimeTrainer"),
        "ChangePoint":        ("model_fun.change_point_trainer",     "ChangePointTrainer"),
        "NGBoost":            ("model_fun.ngboost_trainer",          "NGBoostTrainer"),
        "Ensemble":           ("model_fun.ensemble_trainer",         "EnsembleTrainer"),
        "FTRLProximal":       ("model_fun.ftrl_proximal_trainer",    "FTRLProximalTrainer"),
        "MLP":                ("model_fun.mlp_trainer",              "MLPTrainer"),
        "VAE":                ("model_fun.vae_trainer",              "VAETrainer"),
        "GAN":                ("model_fun.gan_trainer",              "GANTrainer"),
        "MetaLearning":       ("model_fun.meta_learning_trainer",    "MetaLearningTrainer"),
        "MultiTask":          ("model_fun.multi_task_trainer",       "MultiTaskTrainer"),
    }

    mod_name, cls_name = MODMAP[family]
    tmpdir = tempfile.mkdtemp(prefix=f"{family}_", dir=os.getenv("TRAINER_TMP", os.getenv("TRAINING_TMPDIR", "/tmp")))
    os.makedirs(tmpdir, exist_ok=True)
    payload_path = os.path.join(tmpdir, "payload.joblib")
    
    # NEW: write X/y once as .npy and pass a spec instead of raw arrays (memmap for speed)
    x_path = os.path.join(tmpdir, "X.npy")
    y_path = os.path.join(tmpdir, "y.npy")
    if not os.path.exists(x_path):
        np.save(x_path, X, allow_pickle=False)
    if not os.path.exists(y_path):
        np.save(y_path, y, allow_pickle=False)
    
    data_spec = {"mode": "memmap", "X": x_path, "y": y_path}

    # CRITICAL: Calculate optimal thread allocation for this family
    # Use CLI --threads, or env THREADS, or detect
    from common.threads import default_threads
    total_threads = int(os.getenv("THREADS", "") or default_threads())
    plan = plan_for_family(family, total_threads)
    optimal_omp, optimal_mkl = plan["OMP"], plan["MKL"]
    
    # Allow hard override for debugging (e.g., TRAINER_CHILD_FORCE_OMP=14)
    forced_omp = os.getenv("TRAINER_CHILD_FORCE_OMP")
    if forced_omp:
        logger.info("⚠️  [%s] Using forced OMP=%s (was %d)", family, forced_omp, optimal_omp)
        optimal_omp = int(forced_omp)
    
    # Use optimal threads if not explicitly provided (None = use optimal)
    if omp_threads is None:
        omp_threads = optimal_omp
    if mkl_threads is None:
        mkl_threads = optimal_mkl

    # Get optimized environment configuration
    child_env = child_env_for_family(family, total_threads, gpu_ok=True)
    
    # CRITICAL: Pass family name so child can set GPU visibility at import time
    child_env["TRAINER_CHILD_FAMILY"] = family
    
    # Set thread env vars in child env (with override support)
    child_env["OMP_NUM_THREADS"] = str(omp_threads)
    child_env["MKL_NUM_THREADS"] = str(mkl_threads)
    child_env["OPENBLAS_NUM_THREADS"] = "1"
    child_env["NUMEXPR_NUM_THREADS"] = "1"

    # Log environment configuration for debugging (including CVD for diagnostics)
    logger.info("🔧 [%s] Isolation: OMP=%d MKL=%d (plan: %s) NO_TF=%s NO_TORCH=%s CVD_parent=%s CVD_child=%s",
                family, omp_threads, mkl_threads, plan,
                child_env.get("TRAINER_CHILD_NO_TF", ""),
                child_env.get("TRAINER_CHILD_NO_TORCH", ""),
                os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
                child_env.get("CUDA_VISIBLE_DEVICES", "unset"))
    
    # Print child env summary for diagnostics
    print(f"[child-env] family={family} OMP={child_env['OMP_NUM_THREADS']} MKL={child_env['MKL_NUM_THREADS']} CVD={child_env.get('CUDA_VISIBLE_DEVICES', 'unset')}")

    ctx = mp.get_context("spawn")
    
    # CRITICAL: Set environment BEFORE spawning child
    # With spawn mode, child gets a copy of parent's os.environ at spawn time
    with temp_environ(child_env):
        # Double-check CVD is actually set in parent's os.environ
        logger.info("🔍 [%s] Parent os.environ[CUDA_VISIBLE_DEVICES]=%s just before spawn",
                    family, os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET"))
        
        p = ctx.Process(target=child_isolated, args=(payload_path, mod_name, cls_name, data_spec, None,
                                             omp_threads, mkl_threads, trainer_kwargs or {}), daemon=False)
        p.start()
        start = _time.time()
        while p.is_alive() and (_time.time() - start) < timeout_s:
            _time.sleep(5)
    if p.is_alive():
        p.terminate(); p.join(10)
        raise TimeoutError(f"{family} child timed out after {timeout_s}s")
    p.join()

    # Handle missing payload gracefully with retry for fs lag
    for retry in range(3):
        if os.path.exists(payload_path) and os.path.getsize(payload_path) > 0:
            break
        time.sleep(0.5)
    
    if not os.path.exists(payload_path):
        error_file = payload_path + ".error.txt"
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                error_content = f.read()
            raise RuntimeError(f"{family} child exited (code={p.exitcode}) with error file:\n{error_content}")
        else:
            raise RuntimeError(
                f"{family} child exited (code={p.exitcode}) without payload. "
                f"Check TRAINING logs or *.error.txt in the temp dir."
            )
    
    try:
        payload = joblib.load(payload_path)
        if "error" in payload:
            raise RuntimeError(f"{family} child error:\n{payload['error']}")
        return payload["model"]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# Ensure joblib/multiprocessing never forks after TF import
os.environ.setdefault("JOBLIB_START_METHOD", "spawn")

# Optional: see joblib decisions if anything parallelizes
os.environ.setdefault("JOBLIB_VERBOSE", "50")

# Initialize multiprocessing BEFORE importing TF/XGB and fix logging handlers
import multiprocessing as mp
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass  # already set

# Proper logging setup
import logging, logging.handlers, queue, sys

def setup_logging(log_level: str = "INFO", logfile: str = "training_strategies.log"):
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
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# CRITICAL: Import determinism FIRST before any ML libraries
from TRAINING.common.determinism import set_global_determinism, stable_seed_from, seed_for, get_deterministic_params, log_determinism_info

# Set global determinism immediately - OPTIMIZED FOR PERFORMANCE
BASE_SEED = set_global_determinism(
    base_seed=42,
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
DEFAULT_THREADS = str(max(1, (os.cpu_count() or 2) - 1))
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
from strategies import SingleTaskStrategy, MultiTaskStrategy, CascadeStrategy
# Removed unused imports: ModelFactory, DataPreprocessor, TargetResolver, ValidationUtils

# Import target router for enhanced target support
from target_router import route_target, spec_from_target, TaskSpec

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
    "MultiTask": {"nan_ok": False, "needs_tf": True, "experimental": True}
}


def build_sequences_from_features(X, lookback=64):
    """
    Convert 2D features (N, F) to 3D sequences (N', T, F) using rolling windows.
    
    Args:
        X: (N, F) feature matrix
        lookback: sequence length T
        
    Returns:
        X_seq: (N', T, F) where N' = N - lookback + 1
    """
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
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf
        return True
    except ImportError:
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


def prepare_training_data_cross_sectional(mtf_data: Dict[str, pd.DataFrame], 
                                       target: str, 
                                       feature_names: List[str] = None,
                                       min_cs: int = 10,
                                       max_cs_samples: int = None) -> Eight:
    """Prepare cross-sectional training data with polars optimization for memory efficiency."""
    
    logger.info(f"🎯 Building cross-sectional training data for target: {target}")
    if max_cs_samples:
        logger.info(f"📊 Cross-sectional sampling: max {max_cs_samples} samples per timestamp")
    else:
        # Default aggressive sampling for speed
        max_cs_samples = 1000
        logger.info(f"📊 Using default aggressive sampling: max {max_cs_samples} samples per timestamp")
    
    if USE_POLARS:
        return _prepare_training_data_polars(mtf_data, target, feature_names, min_cs, max_cs_samples)
    else:
        return _prepare_training_data_pandas(mtf_data, target, feature_names, min_cs, max_cs_samples)

def _prepare_training_data_polars(mtf_data: Dict[str, pd.DataFrame], 
                                 target: str, 
                                 feature_names: List[str] = None,
                                 min_cs: int = 10,
                                 max_cs_samples: int = None) -> Eight:
    """Polars-based data preparation for memory efficiency with cross-sectional sampling."""
    
    logger.info(f"🎯 Building cross-sectional training data (polars, memory-efficient) for target: {target}")
    
    # Harmonize schema across symbols to avoid width mismatches on concat
    import os
    align_cols = os.environ.get("CS_ALIGN_COLUMNS", "1") not in ("0", "false", "False")
    align_mode = os.environ.get("CS_ALIGN_MODE", "union").lower()
    ordered_schema = None
    if align_cols and mtf_data:
        first_df = next(iter(mtf_data.values()))
        if align_mode == "intersect":
            shared = None
            for _sym, _df in mtf_data.items():
                cols = list(_df.columns)
                shared = set(cols) if shared is None else (shared & set(cols))
            ordered_schema = [c for c in first_df.columns if c in (shared or set())]
            logger.info(f"🔧 [polars] Harmonized schema (intersect) with {len(ordered_schema)} columns")
        else:
            # union
            union = []
            seen = set()
            for c in first_df.columns:
                union.append(c); seen.add(c)
            for _sym, _df in mtf_data.items():
                for c in _df.columns:
                    if c not in seen:
                        union.append(c); seen.add(c)
            ordered_schema = union
            logger.info(f"🔧 [polars] Harmonized schema (union) with {len(ordered_schema)} columns")
    
    # Convert to polars for memory-efficient operations
    all_data_pl = []
    for symbol, df in mtf_data.items():
        if ordered_schema is not None:
            if align_mode == "intersect":
                df_use = df.loc[:, ordered_schema]
            else:
                df_use = df.reindex(columns=ordered_schema)
        else:
            df_use = df
        df_pl = pl.from_pandas(df_use)
        df_pl = df_pl.with_columns(pl.lit(symbol).alias("symbol"))
        all_data_pl.append(df_pl)
    
    # Combine using polars (memory efficient)
    combined_pl = pl.concat(all_data_pl)
    logger.info(f"Combined data shape (polars): {combined_pl.shape}")
    
    # Auto-discover features if not provided
    if feature_names is None:
        all_cols = combined_pl.columns
        feature_names = [col for col in all_cols 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', 'timestamp', 'ts']]
    
    # Normalize time column name
    ts_name = "timestamp" if "timestamp" in combined_pl.columns else ("ts" if "ts" in combined_pl.columns else None)
    
    # Enforce min_cs: filter timestamps that don't meet cross-sectional size
    if ts_name:
        combined_pl = combined_pl.filter(
            pl.len().over(ts_name) >= min_cs
        )
    
    # Apply cross-sectional sampling if specified
    if max_cs_samples and ts_name:
        logger.info(f"📊 Applying cross-sectional sampling: max {max_cs_samples} samples per timestamp")
        
        # Use deterministic per-timestamp sampling with simple approach
        combined_pl = (
            combined_pl
            .sort([ts_name])
            .group_by(ts_name, maintain_order=True)
            .head(max_cs_samples)
        )
        
        logger.info(f"Cross-sectional sampling applied")
    
    # Extract target and features using polars
    try:
        # Get target column
        target_series_pl = combined_pl.select(pl.col(target))
        y = target_series_pl.to_pandas()[target].values
        
        # Get feature columns (preserve timestamp for metadata)
        feature_cols = [target] + feature_names + ['symbol'] + ([ts_name] if ts_name else [])
        data_pl = combined_pl.select(feature_cols)
        
        # Convert to pandas for sklearn compatibility
        combined_df = data_pl.to_pandas()
        
        logger.info(f"Extracted target {target} from polars data")
        
    except Exception as e:
        logger.error(f"Error extracting target {target}: {e}")
        return (None,)*8
    
    # Continue with pandas-based processing
    return _process_combined_data_pandas(combined_df, target, feature_names)

def _prepare_training_data_pandas(mtf_data: Dict[str, pd.DataFrame], 
                                 target: str, 
                                 feature_names: List[str] = None,
                                 min_cs: int = 10,
                                 max_cs_samples: int = None) -> Eight:
    """Pandas-based data preparation (fallback)."""
    
    # Combine all symbol data
    all_data = []
    for symbol, df in mtf_data.items():
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Normalize time column name
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    
    # Enforce min_cs and apply sampling
    if time_col is not None:
        # enforce min_cs
        cs = combined_df.groupby(time_col)["symbol"].transform("size")
        combined_df = combined_df[cs >= min_cs]
        # per-timestamp deterministic sampling
        if max_cs_samples:
            combined_df["_rn"] = combined_df.groupby(time_col).cumcount()
            combined_df = (combined_df
                           .sort_values([time_col, "_rn"])
                           .groupby(time_col, group_keys=False)
                           .head(max_cs_samples)
                           .drop(columns="_rn"))
    
    # Auto-discover features
    if feature_names is None:
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', time_col]]
    
    return _process_combined_data_pandas(combined_df, target, feature_names)

def _process_combined_data_pandas(combined_df: pd.DataFrame, target: str, feature_names: List[str]) -> Eight:
    """Process combined data using pandas."""
    
    # Route target to get task specification
    route_info = route_target(target)
    spec = route_info['spec']
    logger.info(f"[Router] Target {target} → {spec.task} task (objective={spec.objective})")
    
    # Extract target using safe extraction
    try:
        target_series, actual_col = safe_target_extraction(combined_df, target)
        # Sanitize target: replace inf/-inf with NaN
        target_series = target_series.replace([np.inf, -np.inf], np.nan)
        y = target_series.values
        logger.info(f"Extracted target {target} from column {actual_col}")
    except Exception as e:
        logger.error(f"Error extracting target {target}: {e}")
        return (None,)*8
    
    # Extract feature matrix - handle non-numeric columns
    feature_df = combined_df[feature_names].copy()
    
    # Convert to numeric, coerce errors to NaN, and sanitize infinities
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Drop columns that are entirely NaN after coercion
    before_cols = feature_df.shape[1]
    feature_df = feature_df.dropna(axis=1, how='all')
    dropped_all_nan = before_cols - feature_df.shape[1]
    if dropped_all_nan:
        logger.info(f"🔧 Dropped {dropped_all_nan} all-NaN feature columns after coercion")
    
    # Ensure only numeric dtypes remain (guard against objects/arrays)
    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    if len(numeric_cols) != feature_df.shape[1]:
        non_numeric_dropped = feature_df.shape[1] - len(numeric_cols)
        feature_df = feature_df[numeric_cols]
        logger.info(f"🔧 Dropped {non_numeric_dropped} non-numeric feature columns")
    
    # Build float32 matrix safely
    X = feature_df.to_numpy(dtype=np.float32, copy=False)
    
    # Clean data - be more lenient with NaN values
    target_valid = ~np.isnan(y)
    feature_nan_ratio = np.isnan(X).mean(axis=1)
    feature_valid = feature_nan_ratio <= 0.5  # Allow up to 50% NaN in features
    
    # Treat inf in target as invalid as well
    y_is_finite = np.isfinite(y)
    valid_mask = target_valid & feature_valid & y_is_finite
    
    if not valid_mask.any():
        logger.error("No valid data after cleaning")
        return (None,)*8
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = combined_df['symbol'].values[valid_mask]
    
    # Fill remaining NaN values with median
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    X_clean = imputer.fit_transform(X_clean)
    
    logger.info(f"Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"Removed {len(X) - len(X_clean)} rows due to cleaning")
    
    # Determine time column and extract time values
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    time_vals = combined_df[time_col].values[valid_mask] if time_col else None
    
    # Apply routing-based label preparation
    y_prepared, sample_weights, group_sizes, routing_meta = route_info['prepare_fn'](y_clean, time_vals)
    
    # Store routing metadata for trainer
    routing_meta['target_name'] = target
    routing_meta['spec'] = spec
    routing_meta['sample_weights'] = sample_weights
    routing_meta['group_sizes'] = group_sizes
    
    logger.info(f"[Routing] Prepared {spec.task} task: y_shape={y_prepared.shape}, has_weights={sample_weights is not None}, has_groups={group_sizes is not None}")
    
    # Return with prepared labels instead of raw labels
    # Note: We return routing_meta in the imputer slot (slot 7) for now - trainer can extract it
    return X_clean, y_prepared, feature_names, symbols_clean, np.arange(len(X_clean)), feature_names, time_vals, routing_meta

def train_models_for_interval_comprehensive(interval: str, targets: List[str], 
                                           mtf_data: Dict[str, pd.DataFrame],
                                           families: List[str],
                                           strategy: str = 'single_task',
                                           output_dir: str = 'output',
                                           min_cs: int = 10,
                                           max_cs_samples: int = None,
                                           max_rows_train: int = None) -> Dict[str, Any]:
    """Train models for a specific interval using comprehensive approach (replicates original script)."""
    
    logger.info(f"🎯 Training models for interval: {interval}")
    
    results = {
        'interval': interval,
        'targets': targets,
        'families': families,
        'strategy': strategy,
        'models': {},
        'metrics': {}
    }
    
    for j, target in enumerate(targets, 1):
        logger.info(f"🎯 [{j}/{len(targets)}] Training models for target: {target}")
        
        # Prepare training data with cross-sectional sampling
        print(f"🔄 Preparing training data for target: {target}")  # Debug print
        prep_start = time.time()
        X, y, feature_names, symbols, indices, feat_cols, time_vals, routing_meta = prepare_training_data_cross_sectional(
            mtf_data, target, min_cs=min_cs, max_cs_samples=max_cs_samples
        )
        prep_elapsed = time.time() - prep_start
        print(f"✅ Data preparation completed in {prep_elapsed:.2f}s")  # Debug print
        
        if X is None:
            logger.error(f"Failed to prepare data for target {target}")
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
                elif caps.get("needs_tf") and not tf_available():
                    logger.warning(f"TensorFlow missing → skipping {family}")
                    continue
                
                # Check NGBoost dependency
                if family == "NGBoost" and not ngboost_available():
                    logger.warning(f"NGBoost missing → skipping {family}")
                    continue
                
                logger.info(f"🚀 [{family}] Starting {family} training...")
                start_time = _now()
                
                # Train model using modular system with routing metadata
                model_result = train_model_comprehensive(
                    family, X, y, target, strategy, feature_names, caps, routing_meta
                )
                
                elapsed = _now() - start_time
                logger.info(f"⏱️ [{family}] {family} training completed in {elapsed:.2f} seconds")
                
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
                            else:
                                # Save the imputer from data preparation
                                imputer_path = target_dir / f"{family.lower()}_mtf_b0_imputer.joblib"
                                joblib.dump(imputer, imputer_path)
                                logger.info(f"💾 Imputer saved: {imputer_path}")
                            
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
            X_seq = build_sequences_from_features(X, lookback=64)
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
                    try:
                        from xgboost.core import _has_cuda_context
                        has_cuda = bool(_has_cuda_context())
                    except Exception:
                        has_cuda = False
                    xgb_conf.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor"} if has_cuda
                                    else {"tree_method": "hist"})
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

def load_mtf_data(data_dir: str, symbols: List[str], max_rows_per_symbol: int = None) -> Dict[str, pd.DataFrame]:
    """Load MTF data for specified symbols with polars optimization (matches original script behavior)"""
    import time
    data_start = time.time()
    
    logger.info(f"Loading MTF data from {data_dir}")
    print(f"🔄 Loading MTF data from {data_dir}")  # Also print to stdout
    if max_rows_per_symbol:
        logger.info(f"📊 Limiting to {max_rows_per_symbol} most recent rows per symbol")
        print(f"📊 Limiting to {max_rows_per_symbol} most recent rows per symbol")
    else:
        logger.info("📊 Loading ALL data")
        print("📊 Loading ALL data")
    
    mtf_data = {}
    data_path = Path(data_dir)
    
    for symbol in symbols:
        # Try different possible file locations (matching original script)
        possible_paths = [
            data_path / f"symbol={symbol}" / f"{symbol}.parquet",  # New structure
            data_path / f"{symbol}.parquet",  # Direct file
            data_path / f"{symbol}_mtf.parquet",  # Legacy format
        ]
        
        symbol_file = None
        for path in possible_paths:
            if path.exists():
                symbol_file = path
                break
        
        if symbol_file and symbol_file.exists():
            try:
                if USE_POLARS:
                    # Use polars for memory-efficient loading (matching original)
                    lf = pl.scan_parquet(str(symbol_file))
                    
                    # Apply row limit if specified (most recent rows)
                    if max_rows_per_symbol:
                        lf = lf.tail(max_rows_per_symbol)
                    
                    df_pl = lf.collect(streaming=True)
                    df = df_pl.to_pandas(use_pyarrow_extension_array=False)
                    logger.info(f"Loaded {symbol} (polars): {df.shape}")
                else:
                    df = pd.read_parquet(symbol_file)
                    
                    # Apply row limit if specified (most recent rows)
                    if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                        df = df.tail(max_rows_per_symbol)
                        logger.info(f"Limited {symbol} to {max_rows_per_symbol} most recent rows")
                    
                    logger.info(f"Loaded {symbol} (pandas): {df.shape}")
                
                mtf_data[symbol] = df
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        else:
            logger.warning(f"File not found for {symbol}. Tried: {possible_paths}")
    
    data_elapsed = time.time() - data_start
    logger.info(f"✅ Data loading completed in {data_elapsed:.2f}s")
    print(f"✅ Data loading completed in {data_elapsed:.2f}s")
    
    return mtf_data

def discover_targets(mtf_data: Dict[str, pd.DataFrame], 
                   target_patterns: List[str] = None) -> List[str]:
    """Discover available targets in the data"""
    
    if target_patterns:
        return target_patterns
    
    # Auto-discover targets from first symbol
    if not mtf_data:
        return []
    
    sample_symbol = list(mtf_data.keys())[0]
    sample_df = mtf_data[sample_symbol]
    
    # Common target patterns
    target_columns = []
    for col in sample_df.columns:
        if any(col.startswith(prefix) for prefix in 
              ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_']):
            target_columns.append(col)
    
    logger.info(f"Discovered {len(target_columns)} targets: {target_columns[:10]}...")
    return target_columns

def prepare_training_data(mtf_data: Dict[str, pd.DataFrame], 
                         targets: List[str],
                         feature_names: List[str] = None) -> Dict[str, Any]:
    """Prepare training data for strategy training"""
    
    logger.info("Preparing training data...")
    
    # Optional schema harmonization: align per-symbol frames to a shared schema
    # Controls:
    #   CS_ALIGN_COLUMNS=0 to disable entirely
    #   CS_ALIGN_MODE=union|intersect (default union)
    import os
    align_cols = os.environ.get("CS_ALIGN_COLUMNS", "1") not in ("0", "false", "False")
    if align_cols and mtf_data:
        mode = os.environ.get("CS_ALIGN_MODE", "union").lower()
        first_df = next(iter(mtf_data.values()))
        if mode == "intersect":
            shared = None
            for _sym, _df in mtf_data.items():
                cols = list(_df.columns)
                shared = set(cols) if shared is None else (shared & set(cols))
            ordered = [c for c in first_df.columns if c in (shared or set())]
            for sym, df in mtf_data.items():
                if list(df.columns) != ordered:
                    mtf_data[sym] = df.loc[:, ordered]
            logger.info(f"🔧 Harmonized schema (intersect) with {len(ordered)} columns")
        else:
            # union mode: include all columns seen across symbols; fill missing as NaN
            union = []
            seen = set()
            # Start with first df order for determinism
            for c in first_df.columns:
                union.append(c); seen.add(c)
            for _sym, _df in mtf_data.items():
                for c in _df.columns:
                    if c not in seen:
                        union.append(c); seen.add(c)
            for sym, df in mtf_data.items():
                if list(df.columns) != union:
                    mtf_data[sym] = df.reindex(columns=union)
            logger.info(f"🔧 Harmonized schema (union) with {len(union)} columns")
    
    # Combine all symbol data
    all_data = []
    for symbol, df in mtf_data.items():
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Extract features and targets
    if feature_names is None:
        # Auto-discover features (exclude targets and metadata)
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', 'timestamp']]
    
    # Extract feature matrix - handle non-numeric columns
    feature_df = combined_df[feature_names].copy()
    
    # Convert to numeric, coercing errors to NaN
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    X = feature_df.values.astype(np.float32)
    
    # Extract targets
    y_dict = {}
    for target in targets:
        try:
            target_series, actual_col = safe_target_extraction(combined_df, target)
            y_dict[target] = target_series.values
            logger.info(f"Extracted target {target} from column {actual_col}")
        except Exception as e:
            logger.error(f"Error extracting target {target}: {e}")
    
    # Clean data
    valid_mask = ~np.isnan(X).any(axis=1)
    for target_name, y in y_dict.items():
        valid_mask = valid_mask & ~np.isnan(y)
    
    X_clean = X[valid_mask]
    y_clean = {name: y[valid_mask] for name, y in y_dict.items()}
    
    logger.info(f"Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features, {len(y_clean)} targets")
    
    return {
        'X': X_clean,
        'y_dict': y_clean,
        'feature_names': feature_names,
        'target_names': list(y_clean.keys())
    }

def create_strategy_config(strategy: str, targets: List[str], 
                          model_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create configuration for training strategy"""
    
    base_config = {
        'strategy': strategy,
        'targets': targets,
        'models': model_config or {}
    }
    
    if strategy == 'multi_task':
        base_config.update({
            'shared_dim': 128,
            'head_dims': {},
            'loss_weights': {},
            'batch_size': 32,
            'learning_rate': 0.001,
            'n_epochs': 100
        })
    elif strategy == 'cascade':
        base_config.update({
            'gate_threshold': 0.5,
            'calibration_method': 'isotonic',
            'gating_rules': {
                'will_peak_5m': {'action': 'reduce', 'factor': 0.5},
                'will_valley_5m': {'action': 'boost', 'factor': 1.2}
            }
        })
    
    return base_config

def train_with_strategy(strategy: str, training_data: Dict[str, Any], 
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """Train models using specified strategy"""
    
    logger.info(f"Training with strategy: {strategy}")
    
    # Create strategy manager
    if strategy == 'single_task':
        strategy_manager = SingleTaskStrategy(config)
    elif strategy == 'multi_task':
        strategy_manager = MultiTaskStrategy(config)
    elif strategy == 'cascade':
        strategy_manager = CascadeStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Train models
    results = strategy_manager.train(
        training_data['X'],
        training_data['y_dict'],
        training_data['feature_names']
    )
    
    # Test predictions
    test_predictions = strategy_manager.predict(training_data['X'][:100])
    
    return {
        'strategy_manager': strategy_manager,
        'results': results,
        'test_predictions': test_predictions,
        'success': True
    }

def compare_strategies(training_data: Dict[str, Any], 
                      strategies: List[str] = None) -> Dict[str, Any]:
    """Compare different training strategies"""
    
    if strategies is None:
        strategies = ['single_task', 'multi_task', 'cascade']
    
    logger.info(f"Comparing strategies: {strategies}")
    
    comparison_results = {}
    
    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")
        
        try:
            # Create configuration
            config = create_strategy_config(strategy, training_data['target_names'])
            
            # Train with strategy
            result = train_with_strategy(strategy, training_data, config)
            comparison_results[strategy] = result
            
            logger.info(f"✅ {strategy} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {strategy} failed: {e}")
            comparison_results[strategy] = {
                'success': False,
                'error': str(e)
            }
    
    return comparison_results

def main():
    """Main training function with comprehensive approach (replicates original script functionality)"""
    
    
    parser = argparse.ArgumentParser(description='Enhanced Training with Multiple Strategies - Full Original Functionality')
    # Core arguments
    parser.add_argument('--data-dir', required=True, help='Data directory')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to train on')
    parser.add_argument('--targets', nargs='+', help='Specific targets to train on (default: auto-discover all targets)')
    parser.add_argument('--families', nargs='+', default=ALL_FAMILIES, help='Model families to train')
    parser.add_argument('--strategy', choices=['single_task', 'multi_task', 'cascade', 'all'], 
                       default='single_task', help='Training strategy')
    parser.add_argument('--seq-backend', choices=['torch', 'tf'], default='torch', 
                       help='Backend for sequential models (default: torch)')
    parser.add_argument('--output-dir', default='modular_output', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Data size and sampling controls
    parser.add_argument('--max-symbols', type=int, help='Maximum number of symbols to process')
    parser.add_argument('--max-samples-per-symbol', type=int, default=10000, help='Maximum samples per symbol')
    parser.add_argument('--max-rows-per-symbol', type=int, help='Maximum rows per symbol to prevent OOM (default: no limit)')
    parser.add_argument('--max-rows-train', type=int, default=3000000, help='Maximum rows for training (default: 3000000)')
    parser.add_argument('--max-rows-val', type=int, default=600000, help='Maximum rows for validation (default: 600000)')
    
    # Cross-sectional parameters
    parser.add_argument('--min-cs', type=int, default=10, help='Minimum cross-sectional size per timestamp (default: 10)')
    parser.add_argument('--cs-normalize', choices=['none', 'per_ts_split'], default='per_ts_split', 
                       help='Cross-sectional normalization mode (default: per_ts_split)')
    parser.add_argument('--cs-block', type=int, default=32, help='Block size for CS transforms (default: 32)')
    parser.add_argument('--cs-winsor-p', type=float, default=0.01, help='Winsorization percentile (default: 0.01)')
    parser.add_argument('--cs-ddof', type=int, default=1, help='Degrees of freedom for standard deviation (default: 1)')
    
    # Batch processing
    parser.add_argument('--batch-size', type=int, default=50, help='Number of symbols to process per batch')
    parser.add_argument('--batch-id', type=int, default=0, help='Batch ID for this training run')
    parser.add_argument('--session-id', type=str, default=None, help='Session ID for this training run')
    
    # Model configuration
    parser.add_argument('--experimental', action='store_true', help='Include experimental models')
    parser.add_argument('--include-experimental', action='store_true', help='Include experimental/placeholder model families')
    parser.add_argument('--quantile-alpha', type=float, default=0.5, help='Alpha parameter for QuantileLightGBM (default: 0.5)')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU for all learners (LightGBM/XGBoost)')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() - 1), 
                       help=f'Number of threads for training (default: {max(1, os.cpu_count() - 1)})')
    
    # Model type selection arguments
    parser.add_argument('--model-types', choices=['cross-sectional', 'sequential', 'both'], 
                       default='both', help='Which model types to train (default: both)')
    parser.add_argument('--train-order', choices=['cross-first', 'sequential-first', 'mixed'], 
                       default='cross-first', help='Training order for model types (default: cross-first)')
    
    # Ranking and objectives
    parser.add_argument('--rank-objective', choices=['on', 'off'], default='on', 
                       help='Enable ranking objectives for LGB/XGB (default: on)')
    parser.add_argument('--rank-labels', choices=['dense', 'raw'], default='dense', 
                       help='Ranking label method: dense for dense ranks (default), raw for continuous values')
    
    # Sequence models
    parser.add_argument('--seq-lookback', type=int, default=64, 
                       help='Lookback window for temporal sequence models (default: 64)')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs for sequence models (default: 50, use 1000 for production)')
    
    # Feature management
    parser.add_argument('--feature-list', type=str, help='Path to JSON file of global feature list')
    parser.add_argument('--save-features', action='store_true', help='Save global feature list to features_all.json')
    
    # Validation and debugging
    parser.add_argument('--validate-targets', action='store_true', 
                       help='Run preflight validation checks on targets before training')
    parser.add_argument('--strict-exit', action='store_true', 
                       help='Exit with error code if any model fails (default: only exit on complete failure)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    # Memory optimization
    parser.add_argument('--use-polars', action='store_true', help='Use polars for memory optimization (default: enabled)')
    parser.add_argument('--no-polars', action='store_true', help='Disable polars, use pandas only')
    
    # Strategy configuration
    parser.add_argument('--strategy-config', type=str, help='Path to strategy configuration file')
    
    args = parser.parse_args()
    
    # Set global backend for sequential models
    global SEQ_BACKEND
    SEQ_BACKEND = args.seq_backend
    logger.info(f"Sequential backend: {SEQ_BACKEND}")
    
    # Handle polars settings
    global USE_POLARS
    if args.no_polars:
        USE_POLARS = False
        logger.info("Polars disabled by user")
    elif args.use_polars:
        USE_POLARS = True
        logger.info("Polars enabled by user")
    
    # Setup logging
    listener = setup_logging(args.log_level)
    
    # Optional: add live stack dumps for any future "quiet" periods
    try:
        import faulthandler, signal
        faulthandler.register(signal.SIGUSR2)  # run: kill -USR2 <pid> to dump all stacks
    except Exception:
        pass
    
    # Set global thread knobs from CLI
    global THREADS, MKL_THREADS_DEFAULT, CPU_ONLY
    THREADS = args.threads              # e.g., 16 on 11700K
    CPU_ONLY = args.cpu_only
    MKL_THREADS_DEFAULT = 1             # default; we'll override per-family
    
    # Apply environment guard with actual CLI values
    _env_guard(THREADS, mkl_threads=MKL_THREADS_DEFAULT)
    
    logger.info("🚀 Starting enhanced training with multiple strategies - Full Original Functionality")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Families: {args.families}")
    logger.info(f"Min cross-sectional size: {args.min_cs}")
    
    # Apply max_symbols limit if specified
    if args.max_symbols:
        args.symbols = args.symbols[:args.max_symbols]
        logger.info(f"Limited to {args.max_symbols} symbols: {args.symbols}")
    
    # Filter families based on experimental flag
    if not args.experimental:
        families = [f for f in args.families if not FAMILY_CAPS.get(f, {}).get('experimental', False)]
        logger.info(f"Filtered to non-experimental families: {families}")
    else:
        families = args.families
    
    # Filter by model type
    if args.model_types == 'cross-sectional':
        families = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        logger.info(f"🎯 Training only cross-sectional models: {len(families)} models")
    elif args.model_types == 'sequential':
        families = [f for f in families if f in SEQUENTIAL_MODELS]
        logger.info(f"🎯 Training only sequential models: {len(families)} models")
    else:  # both
        logger.info(f"🎯 Training both model types: {len(families)} models")
    
    # Sort models by training order
    if args.train_order == 'cross-first':
        # Train cross-sectional models first, then sequential
        cross_models = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        seq_models = [f for f in families if f in SEQUENTIAL_MODELS]
        families = cross_models + seq_models
        logger.info(f"📊 Training order: {len(cross_models)} cross-sectional → {len(seq_models)} sequential")
    elif args.train_order == 'sequential-first':
        # Train sequential models first, then cross-sectional
        cross_models = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        seq_models = [f for f in families if f in SEQUENTIAL_MODELS]
        families = seq_models + cross_models
        logger.info(f"📊 Training order: {len(seq_models)} sequential → {len(cross_models)} cross-sectional")
    else:  # mixed
        logger.info(f"📊 Training order: mixed (as specified)")
    
    # Create output directory with session ID (same as original)
    # Using top-level import: datetime
    session_id = f"mtf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    try:
        # Load data (with optional row limiting like original script)
        logger.info(f"📂 Loading data from {args.data_dir}")
        logger.info(f"📊 Symbols: {args.symbols}")
        logger.info(f"🔢 Max rows per symbol: {args.max_rows_per_symbol}")
        
        mtf_data = load_mtf_data(args.data_dir, args.symbols, args.max_rows_per_symbol)
        if not mtf_data:
            logger.error("No data loaded")
            return
        
        logger.info(f"✅ Loaded data for {len(mtf_data)} symbols")
        for symbol, df in mtf_data.items():
            logger.info(f"  📈 {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Discover targets
        logger.info(f"🎯 Discovering targets...")
        targets = discover_targets(mtf_data, args.targets)
        if not targets:
            logger.error("No targets found")
            return
        
        # Validate targets if requested
        if args.validate_targets:
            missing, empty = [], []
            for t in targets:
                exists_any = any(t in df.columns for df in mtf_data.values())
                if not exists_any:
                    missing.append(t); continue
                # consider empty if all-NaN across every symbol that has it
                has_any_non_nan = any((t in df.columns) and (~pd.isna(df[t])).any() for df in mtf_data.values())
                if not has_any_non_nan:
                    empty.append(t)
            if missing or empty:
                logger.error(f"Missing targets: {missing} | Empty targets: {empty}")
                if args.strict_exit: 
                    sys.exit(2)
        
        logger.info(f"✅ Found {len(targets)} targets: {targets[:5]}...")
        logger.info(f"🤖 Training {len(families)} model families: {families[:5]}...")
        logger.info(f"📋 Strategy: {args.strategy}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Memory cleanup
        aggressive_cleanup()
        
        # Train with strategy/strategies
        if args.strategy == 'all':
            # Compare all strategies using comprehensive approach
            comparison_results = {}
            for strategy in ['single_task', 'multi_task', 'cascade']:
                logger.info(f"Testing strategy: {strategy}")
                try:
                    result = train_models_for_interval_comprehensive(
                        'cross_sectional', targets, mtf_data, families,
                        strategy, str(output_dir), args.min_cs, args.max_samples_per_symbol,
                        args.max_rows_train
                    )
                    comparison_results[strategy] = result
                    logger.info(f"✅ {strategy} completed successfully")
                except Exception as e:
                    logger.error(f"❌ {strategy} failed: {e}")
                    comparison_results[strategy] = {'success': False, 'error': str(e)}
            
            # Save comparison results
            joblib.dump(comparison_results, output_dir / 'strategy_comparison.pkl')
            logger.info(f"Comparison results saved to {output_dir / 'strategy_comparison.pkl'}")
            
        else:
            # Train with single strategy using comprehensive approach
            results = train_models_for_interval_comprehensive(
                'cross_sectional', targets, mtf_data, families,
                args.strategy, str(output_dir), args.min_cs, args.max_samples_per_symbol,
                args.max_rows_train
            )
            
            # Save results
            joblib.dump(results, output_dir / f'{args.strategy}_results.pkl')
            logger.info(f"Results saved to {output_dir / f'{args.strategy}_results.pkl'}")
            
            # Print summary
            total_models = sum(len(target_results) for target_results in results['models'].values())
            logger.info(f"✅ {args.strategy} training completed: {total_models} models trained")
        
        logger.info("🎉 Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
