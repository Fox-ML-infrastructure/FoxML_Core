# MIT License - see LICENSE file

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
# Paths are already set up above

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications - import from centralized constants
from TRAINING.common.family_constants import TF_FAMS, TORCH_FAMS, CPU_FAMS, TORCH_SEQ_FAMILIES


"""Family runner functions for in-process and isolated execution."""

# Standard library imports
import logging

# Third-party imports
import numpy as np

# Setup logger
logger = logging.getLogger(__name__)

def _run_family_inproc(family: str, X, y, total_threads: int = 12, trainer_kwargs: dict | None = None):
    # CRITICAL: Normalize family name before registry lookup
    from TRAINING.training_strategies.utils import normalize_family_name
    family = normalize_family_name(family)
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
    
    # Module mapping - all keys must be canonical snake_case
    MODMAP = {
        "lightgbm":           ("model_fun.lightgbm_trainer",        "LightGBMTrainer"),
        "quantile_lightgbm":   ("model_fun.quantile_lightgbm_trainer","QuantileLightGBMTrainer"),
        "xgboost":            ("model_fun.xgboost_trainer",          "XGBoostTrainer"),
        "reward_based":        ("model_fun.reward_based_trainer",     "RewardBasedTrainer"),
        "gmm_regime":          ("model_fun.gmm_regime_trainer",       "GMMRegimeTrainer"),
        "change_point":        ("model_fun.change_point_trainer",     "ChangePointTrainer"),
        "ngboost":            ("model_fun.ngboost_trainer",          "NGBoostTrainer"),
        "ensemble":           ("model_fun.ensemble_trainer",         "EnsembleTrainer"),
        "ftrl_proximal":       ("model_fun.ftrl_proximal_trainer",    "FTRLProximalTrainer"),
        "mlp":                ("model_fun.mlp_trainer",              "MLPTrainer"),
        "neural_network":     ("model_fun.neural_network_trainer",   "NeuralNetworkTrainer"),
        "vae":                ("model_fun.vae_trainer",              "VAETrainer"),
        "gan":                ("model_fun.gan_trainer",              "GANTrainer"),
        "meta_learning":       ("model_fun.meta_learning_trainer",    "MetaLearningTrainer"),
        "multi_task":          ("model_fun.multi_task_trainer",       "MultiTaskTrainer"),
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
        # CRITICAL: family is already normalized to snake_case before this function is called
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
        try:
            with family_run_scope(family, total_threads):
                result = trainer.train(X, y)
            if result is None:
                logger.warning(f"[InProc] {family} trainer.train() returned None")
                return None
            logger.info(f"[InProc] {family} training completed successfully")
            return result
        except Exception as e:
            logger.error(f"[InProc] {family} training failed: {e}")
            logger.exception(f"Full traceback for {family} in-process training:")
            raise

def _run_family_isolated(family: str, X, y, timeout_s: int = None,
                         omp_threads: int | None = None, mkl_threads: int | None = None,
                         trainer_kwargs: dict | None = None):
    # CRITICAL: Normalize family name before registry lookup
    from TRAINING.training_strategies.utils import normalize_family_name
    family = normalize_family_name(family)
    
    # Load timeout from config if not provided
    if timeout_s is None:
        if _CONFIG_AVAILABLE:
            timeout_s = get_family_timeout(family, default=7200)
        else:
            timeout_s = 7200
    import tempfile, joblib, multiprocessing as mp, os, time as _time, numpy as np, shutil

    # Module mapping - all keys must be canonical snake_case
    MODMAP = {
        "lightgbm":           ("model_fun.lightgbm_trainer",        "LightGBMTrainer"),
        "quantile_lightgbm":   ("model_fun.quantile_lightgbm_trainer","QuantileLightGBMTrainer"),
        "xgboost":            ("model_fun.xgboost_trainer",          "XGBoostTrainer"),
        "reward_based":        ("model_fun.reward_based_trainer",     "RewardBasedTrainer"),
        "gmm_regime":          ("model_fun.gmm_regime_trainer",       "GMMRegimeTrainer"),
        "change_point":        ("model_fun.change_point_trainer",     "ChangePointTrainer"),
        "ngboost":            ("model_fun.ngboost_trainer",          "NGBoostTrainer"),
        "ensemble":           ("model_fun.ensemble_trainer",         "EnsembleTrainer"),
        "ftrl_proximal":       ("model_fun.ftrl_proximal_trainer",    "FTRLProximalTrainer"),
        "mlp":                ("model_fun.mlp_trainer",              "MLPTrainer"),
        "neural_network":     ("model_fun.neural_network_trainer",   "NeuralNetworkTrainer"),
        "vae":                ("model_fun.vae_trainer",              "VAETrainer"),
        "gan":                ("model_fun.gan_trainer",              "GANTrainer"),
        "meta_learning":       ("model_fun.meta_learning_trainer",    "MetaLearningTrainer"),
        "multi_task":          ("model_fun.multi_task_trainer",       "MultiTaskTrainer"),
    }

    # Get module mapping - check both MODMAP dictionaries
    # CRITICAL: family is already normalized to snake_case before this function is called
    if family not in MODMAP:
        # Fallback to TRAINER_MODULE_MAP from isolation_runner if not in local MODMAP
        from common.isolation_runner import TRAINER_MODULE_MAP
        if family in TRAINER_MODULE_MAP:
            mod_name, cls_name = TRAINER_MODULE_MAP[family]
        else:
            # Check if this is a selector/scorer (not a trainer)
            if family in ["mutual_information", "univariate_selection"]:
                raise ValueError(f"Family '{family}' is a feature selector/scorer, not a model trainer. Remove from training family list.")
            raise KeyError(f"Family '{family}' not found in MODMAP or TRAINER_MODULE_MAP. Available families: {sorted(set(MODMAP.keys()) | set(TRAINER_MODULE_MAP.keys()))}")
    else:
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
        _t.sleep(0.5)
    
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
        # CRITICAL: Import custom Keras layers before loading models
        # This ensures registered classes (like VAE's Sampling layer) are available
        if family in ("VAE", "GAN", "MetaLearning", "MultiTask"):
            try:
                if family == "VAE":
                    from model_fun.vae_trainer import Sampling, KLLossLayer  # noqa: F401
                elif family == "GAN":
                    from model_fun.gan_trainer import Generator, Discriminator  # noqa: F401
                # MetaLearning and MultiTask use standard layers, but import to be safe
                elif family == "MetaLearning":
                    from model_fun.meta_learning_trainer import MetaLearningTrainer  # noqa: F401
                elif family == "MultiTask":
                    from model_fun.multi_task_trainer import MultiTaskTrainer  # noqa: F401
            except ImportError as e:
                logger.warning(f"Could not import custom layers for {family}: {e}. Model loading may fail.")
        
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

