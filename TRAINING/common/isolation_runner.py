# MIT License - see LICENSE file

"""
Isolation Runner for Spawned Subprocesses

This module contains the child function for isolated model training.
It's designed to be lightweight and avoid importing the heavy orchestrator.
"""


# ---- PATH BOOTSTRAP: ensure project root on sys.path in child processes ----
import os as _os
import sys
import types
from pathlib import Path

# project root likely: .../secure/trader (parent of TRAINING)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure Python can import `common`, `model_fun`, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---- CRITICAL: Block Torch/TF in CPU-only children (prevents libiomp5/libgomp conflicts) ----
def _block_module(name: str):
    """Prevent a module from being imported by replacing it with a stub."""
    if name in sys.modules:
        del sys.modules[name]
    m = types.ModuleType(name)
    def _blocked(*a, **k):
        raise RuntimeError(f"{name} is disabled in this child process (TRAINER_CHILD_NO_TORCH/NO_TF)")
    m.__getattr__ = lambda *_: _blocked()
    sys.modules[name] = m

# ---- CRITICAL: Set GPU visibility at import time (before any CUDA init) ----
# Decide based on family hint from parent; this prevents TF/XLA from initializing with wrong CVD
_FAMILY = _os.getenv("TRAINER_CHILD_FAMILY", "")
_GPU_FAMILIES = {"MLP", "VAE", "GAN", "MetaLearning", "MultiTask", "XGBoost", 
                 "CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}
_CPU_FAMILIES = {"QuantileLightGBM", "LightGBM", "RewardBased", "Ensemble",
                 "ChangePoint", "NGBoost", "GMMRegime", "FTRLProximal"}

# ---- Fix readline library symbol lookup error ----
# Suppress readline issues that can cause "sh: undefined symbol: rl_print_keybinding"
# This happens when sh/bash was compiled against a different readline version
# Set these BEFORE any imports that might spawn subprocesses
if "SHELL" not in _os.environ:
    _os.environ["SHELL"] = "/usr/bin/bash"  # Use bash instead of sh if available
if "TERM" not in _os.environ:
    _os.environ["TERM"] = "dumb"  # Disable readline features
# Also suppress readline completely
_os.environ.setdefault("INPUTRC", "/dev/null")  # Disable readline config

if _FAMILY in _GPU_FAMILIES:
    # GPU family: expose GPU 0 (unless parent explicitly set a different value)
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    _os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "0")
    _os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
elif _FAMILY in _CPU_FAMILIES:
    # CPU-only family: hide GPUs
    _os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    _os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "none")
# else: unknown family, leave as-is

# Block Torch if NO_TORCH=1 (prevents libiomp5 from loading)
# NOTE: Do NOT touch CUDA_VISIBLE_DEVICES here - it's set above based on family
if _os.getenv("TRAINER_CHILD_NO_TORCH", "0") == "1":
    for module_name in ("torch", "torch._C", "torch.cuda", "pytorch_lightning", "torchvision"):
        _block_module(module_name)

# Show TensorFlow warnings so user knows if GPU isn't working
# _os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Removed - show warnings
# _os.environ.setdefault("TF_LOGGING_VERBOSITY", "ERROR")  # Removed - show warnings

# Ensure conda CUDA libraries are accessible to TensorFlow
_conda_prefix = _os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    _conda_lib = _os.path.join(_conda_prefix, "lib")
    _conda_targets_lib = _os.path.join(_conda_prefix, "targets", "x86_64-linux", "lib")
    _current_ld_path = _os.environ.get("LD_LIBRARY_PATH", "")
    # Add conda lib paths if not already present
    _new_paths = []
    if _conda_lib not in _current_ld_path:
        _new_paths.append(_conda_lib)
    if _conda_targets_lib not in _current_ld_path:
        _new_paths.append(_conda_targets_lib)
    if _new_paths:
        _updated_ld_path = ":".join(_new_paths + [_current_ld_path] if _current_ld_path else _new_paths)
        _os.environ["LD_LIBRARY_PATH"] = _updated_ld_path

# Block TF if NO_TF=1 (prevents TF GPU init in CPU families)

# NOTE: Do NOT touch CUDA_VISIBLE_DEVICES here - it's set above based on family
if _os.getenv("TRAINER_CHILD_NO_TF", "0") == "1":
    for module_name in ("tensorflow", "tf", "jax", "jaxlib"):
        _block_module(module_name)

import joblib
import importlib
import logging
import threading
import psutil
import time
import signal
from .safety import set_global_numeric_guards

# Suppress readline errors by redirecting stderr early
# The "sh: undefined symbol: rl_print_keybinding" error is harmless but annoying
# It comes from system calls that spawn sh, and we can't easily prevent it
# So we'll filter it from stderr output
import sys
_original_stderr = sys.stderr
class _FilteredStderr:
    """Filter stderr to suppress readline symbol lookup errors."""
    def __init__(self, original):
        self.original = original
    def write(self, s):
        if s and "symbol lookup error" in s and "rl_print_keybinding" in s:
            return  # Suppress this specific error
        return self.original.write(s)
    def flush(self):
        return self.original.flush()
    def __getattr__(self, name):
        return getattr(self.original, name)
sys.stderr = _FilteredStderr(_original_stderr)

# Set global numeric guards in child processes too
set_global_numeric_guards()

logger = logging.getLogger(__name__)

# ---- MKL GUARD FOR RISKY FAMILIES ----
# Families that segfault in scipy.linalg.solve (Ridge's Cholesky path) due to MKL/OpenMP conflicts
RISKY_MKL_FAMILIES = {"ChangePoint", "GMMRegime"}

def _apply_mkl_guard_for(family: str):
    """
    Apply MKL/BLAS guard for families that crash in scipy.linalg.solve.
    Must be called BEFORE any numpy/scipy imports.
    """
    if family not in RISKY_MKL_FAMILIES:
        return
    
    logger.info("🔒 [MKL Guard] Activating for %s (prevent scipy.linalg.solve segfault)", family)
    _os.environ["OMP_NUM_THREADS"] = "1"
    _os.environ["MKL_NUM_THREADS"] = "1"
    _os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _os.environ["NUMEXPR_NUM_THREADS"] = "1"
    _os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
    _os.environ["KMP_AFFINITY"] = "disabled"
    _os.environ["KMP_INIT_AT_FORK"] = "FALSE"
    # Optional (helps on some AMD CPUs):
    _os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    # Use safe Ridge solver (lsqr bypasses Cholesky path)
    _os.environ["SKLEARN_RIDGE_SOLVER"] = "lsqr"
    # Tell the rest of the runner not to re-force threads later
    _os.environ["THREAD_FORCE_DISABLED"] = "1"

# ---- TRAINER MODULE MAP: family → (module_path, class_name) ----
# This allows lazy imports - we only load the specific trainer needed
TRAINER_MODULE_MAP = {
    "lightgbm": ("model_fun.lightgbm_trainer", "LightGBMTrainer"),
    "quantile_lightgbm": ("model_fun.quantile_lightgbm_trainer", "QuantileLightGBMTrainer"),
    "xgboost": ("model_fun.xgboost_trainer", "XGBoostTrainer"),
    "reward_based": ("model_fun.reward_based_trainer", "RewardBasedTrainer"),
    "ngboost": ("model_fun.ngboost_trainer", "NGBoostTrainer"),
    "ensemble": ("model_fun.ensemble_trainer", "EnsembleTrainer"),
    "gmm_regime": ("model_fun.gmm_regime_trainer", "GMMRegimeTrainer"),
    "change_point": ("model_fun.change_point_trainer", "ChangePointTrainer"),
    "ftrl_proximal": ("model_fun.ftrl_proximal_trainer", "FTRLProximalTrainer"),
    "mlp": ("model_fun.mlp_trainer", "MLPTrainer"),
    "neural_network": ("model_fun.neural_network_trainer", "NeuralNetworkTrainer"),
    "cnn1d": ("model_fun.cnn1d_trainer", "CNN1DTrainer"),
    "lstm": ("model_fun.lstm_trainer", "LSTMTrainer"),
    "transformer": ("model_fun.transformer_trainer", "TransformerTrainer"),
    "tabcnn": ("model_fun.tabcnn_trainer", "TabCNNTrainer"),
    "tablstm": ("model_fun.tablstm_trainer", "TabLSTMTrainer"),
    "tabtransformer": ("model_fun.tabtransformer_trainer", "TabTransformerTrainer"),
    "vae": ("model_fun.vae_trainer", "VAETrainer"),
    "gan": ("model_fun.gan_trainer", "GANTrainer"),
    "meta_learning": ("model_fun.meta_learning_trainer", "MetaLearningTrainer"),
    "multi_task": ("model_fun.multi_task_trainer", "MultiTaskTrainer"),
}

def _bootstrap_family_runtime(family: str, logger_inst):
    """
    Centralized TF/Torch initialization - runs once per child.
    Logs GPU/CUDA info and sets threading for TF families.
    
    This prevents every trainer from duplicating GPU detection logic.
    """
    try:
        from .runtime_policy import get_policy
        
        policy = get_policy(family)
        no_tf = _os.getenv("TRAINER_CHILD_NO_TF", "0") == "1"
        no_torch = _os.getenv("TRAINER_CHILD_NO_TORCH", "0") == "1"
        
        # Visibility is already decided at import time; just read it for logging
        cvd = _os.getenv("CUDA_VISIBLE_DEVICES", "unset")
        logger_inst.info(f"[bootstrap] {family} | policy.needs_gpu={policy.needs_gpu} | CVD={cvd}")
        
        # TensorFlow families - bootstrap TF once
        if "tf" in policy.backends and not no_tf:
            # Ensure TF memory growth is enabled before import
            _os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
            # Suppress harmless version compatibility warnings (use_unbounded_threadpool)
            # These occur when graphs are created with newer TF than runtime - harmless
            import warnings
            warnings.filterwarnings("ignore", message=".*use_unbounded_threadpool.*", category=UserWarning)
            # Import TensorFlow - show warnings so user knows if GPU isn't working
            import tensorflow as tf
            # Suppress TensorFlow's own version compatibility warnings
            import logging
            tf_logger = logging.getLogger("tensorflow")
            # Filter out the specific NodeDef attribute warnings (harmless version mismatch)
            class _TFFilter(logging.Filter):
                def filter(self, record):
                    if "use_unbounded_threadpool" in record.getMessage():
                        return False
                    if "Unknown attributes will be ignored" in record.getMessage():
                        return False
                    return True
            tf_logger.addFilter(_TFFilter())
            
            # Set TF threading (read from env vars set by child_env_for_family)
            intra = int(_os.getenv("TF_NUM_INTRAOP_THREADS", "1"))
            inter = int(_os.getenv("TF_NUM_INTEROP_THREADS", "1"))
            try:
                tf.config.threading.set_intra_op_parallelism_threads(intra)
                tf.config.threading.set_inter_op_parallelism_threads(inter)
            except RuntimeError:
                pass  # Already initialized
            
            # Detect and configure GPUs
            gpus = []
            try:
                gpus = tf.config.list_physical_devices("GPU")
                for g in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except Exception:
                        pass
            except Exception:
                pass
            
            # Get TF build info for diagnostics
            try:
                build = getattr(tf.sysconfig, "get_build_info", lambda: {})()
            except Exception:
                build = {}
            
            logger_inst.info(
                "[bootstrap] TF %s | built_with_cuda=%s | cuda=%s | cudnn=%s | GPUs=%d | CVD=%s | needs_gpu=%s",
                getattr(tf, "__version__", "?"),
                getattr(tf.test, "is_built_with_cuda", lambda: "n/a")(),
                build.get("cuda_version", "n/a"),
                build.get("cudnn_version", "n/a"),
                len(gpus),
                cvd,
                policy.needs_gpu,
            )
            
            # Only warn if we expected GPU but didn't get it
            if policy.needs_gpu and len(gpus) == 0:
                logger_inst.warning("⚠️  [bootstrap] Expected GPU but TF found 0 GPUs - training on CPU")
                logger_inst.warning("   Check CUDA/cuDNN installation and TF build compatibility")
        
        # PyTorch families - bootstrap Torch once
        if family in {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"} and not no_torch:
            import torch
            logger_inst.info(
                "[bootstrap] Torch %s | cuda_available=%s | cuda=%s | cudnn=%s | CVD=%s",
                torch.__version__,
                torch.cuda.is_available(),
                getattr(torch.version, "cuda", "n/a"),
                getattr(torch.backends.cudnn, "version", lambda: "n/a")() if hasattr(torch.backends, "cudnn") else "n/a",
                cvd
            )
    except Exception:
        logger_inst.debug("bootstrap failed", exc_info=True)

# Memory watchdog globals
PEAK_GB = 0.0
STOP = False

# Configurable memory cap (0 or negative disables)
try:
    MEMCAP_GB = float(_os.getenv("TRAINER_CHILD_MEMCAP_GB", "0"))
except Exception:
    MEMCAP_GB = 0.0
DISABLE_CAP = _os.getenv("TRAINER_CHILD_MEMCAP_DISABLE", "0") == "1"

def _env_guard(omp_threads: int, mkl_threads: int = 1):
    """Set environment variables for thread control in isolated processes."""
    _os.environ.setdefault("OMP_NUM_THREADS", str(omp_threads))
    _os.environ.setdefault("MKL_NUM_THREADS", str(mkl_threads))
    _os.environ.setdefault("JOBLIB_START_METHOD", "spawn")
    _os.environ.setdefault("JOBLIB_VERBOSE", "10")

def _enforce_memcap(payload_path: str):
    """Check memory cap and write error payload if exceeded."""
    if DISABLE_CAP or MEMCAP_GB <= 0:
        return
    rss_gb = psutil.Process().memory_info().rss / 1e9
    if rss_gb > MEMCAP_GB:
        print(f"[child] Exceeding mem {rss_gb:.1f}GB (cap {MEMCAP_GB}GB) → terminating", flush=True)
        # Write an error payload so parent doesn't see a missing file
        try:
            tmp = payload_path + ".tmp"
            joblib.dump({"error": "memcap_exceeded", "rss_gb": rss_gb, "memcap_gb": MEMCAP_GB}, tmp)
            _os.replace(tmp, payload_path)
        except Exception:
            pass
        _os._exit(99)

def mem_watch():
    """Memory watchdog thread."""
    global PEAK_GB, STOP
    proc = psutil.Process(_os.getpid())
    while not STOP:
        rss = proc.memory_info().rss / (1024**3)
        PEAK_GB = max(PEAK_GB, rss)
        if not DISABLE_CAP and MEMCAP_GB > 0 and rss > MEMCAP_GB:
            logger.error(f"[child] Exceeding mem {rss:.1f}GB (cap {MEMCAP_GB}GB) → terminating")
            _os._exit(99)  # Use exit code 99 for memory cap
        time.sleep(2)

def child_isolated(payload_path, mod, cls, X_or_spec, y_unused, omp_t, mkl_t, kwargs):
    """
    Child function for isolated model training.
    
    Args:
        payload_path: Path to save the trained model
        mod: Module name (e.g., "model_fun.lightgbm_trainer")
        cls: Class name (e.g., "LightGBMTrainer")
        X_or_spec: Training features OR memmap spec dict
        y_unused: Training targets (unused if memmap spec provided)
        omp_t: OpenMP threads
        mkl_t: MKL threads
        kwargs: Additional trainer arguments
    """
    global STOP
    
    # Early logging to confirm child started
    logger.info(f"[child] STARTED for {cls}")
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Extract family name for policy lookup
    family_base = cls.replace("Trainer", "")
    
    # Log what we inherited from parent (for diagnostics)
    cvd_inherited = _os.getenv("CUDA_VISIBLE_DEVICES", "unset")
    logger.info(f"[child-early] INHERITED CUDA_VISIBLE_DEVICES={cvd_inherited}")
    
    # CRITICAL: Apply MKL guard BEFORE any numpy/scipy imports
    # Must happen before importing threads (which imports numpy/scipy)
    _apply_mkl_guard_for(family_base)
    
    # Import unified threading system
    from common.threads import (
        reset_affinity, allowed_cpus, effective_threads, 
        thread_guard, log_thread_state, set_estimator_threads
    )
    
    # CRITICAL: Reset affinity to all allowed CPUs (not inherited pinning)
    reset_affinity(logger)
    
    # Log environment for diagnostics (including CUDA_VISIBLE_DEVICES)
    logger.info(
        "[child-env] family=%s OMP=%s MKL=%s CVD=%s NO_TF=%s NO_TORCH=%s",
        cls,
        _os.getenv("OMP_NUM_THREADS"),
        _os.getenv("MKL_NUM_THREADS"),
        _os.getenv("CUDA_VISIBLE_DEVICES", "unset"),
        _os.getenv("TRAINER_CHILD_NO_TF", "?"),
        _os.getenv("TRAINER_CHILD_NO_TORCH", "?"),
    )
    
    # Diagnostic logging for resource allocation
    try:
        import psutil
        p = psutil.Process()
        aff = p.cpu_affinity() if hasattr(p, 'cpu_affinity') else allowed_cpus()
        logger.info(f"[child] CPU affinity allows {len(aff)} cores: {aff}")
    except Exception as e:
        logger.warning(f"[child] Could not get CPU affinity: {e}")
    
    logger.info(f"[child] OMP={_os.getenv('OMP_NUM_THREADS')} MKL={_os.getenv('MKL_NUM_THREADS')} "
                f"OPENBLAS={_os.getenv('OPENBLAS_NUM_THREADS')} "
                f"OMP_DYNAMIC={_os.getenv('OMP_DYNAMIC')} "
                f"OMP_PROC_BIND={_os.getenv('OMP_PROC_BIND')} OMP_PLACES={_os.getenv('OMP_PLACES')}")
    
    # Start memory watchdog
    t = threading.Thread(target=mem_watch, daemon=True)
    t.start()
    
    try:
        # Apply thread guard for the entire training block
        # This ensures OpenMP/MKL respect the planned thread counts
        # UNLESS MKL guard is active (for risky families)
        if _os.environ.get("THREAD_FORCE_DISABLED") != "1":
            context_manager = thread_guard(omp=omp_t, mkl=mkl_t)
        else:
            # MKL guard active - don't override thread settings
            logger.info("🔒 [Isolation] Thread forcing disabled for %s (MKL guard active)", cls)
            from contextlib import nullcontext
            context_manager = nullcontext()
        
        with context_manager:
            # Log thread state for diagnostics (should show full cores)
            log_thread_state(logger)
            
            # CRITICAL: Log final thread state (last-write wins)
            try:
                from threadpoolctl import threadpool_info
                pools = threadpool_info()
                pool_summary = "; ".join(
                    f"{p.get('user_api', '?')}={p.get('num_threads', '?')}"
                    for p in pools
                )
                logger.info(f"🔍 [child] FINAL threads → OMP={_os.getenv('OMP_NUM_THREADS')} "
                           f"MKL={_os.getenv('MKL_NUM_THREADS')} pools=[{pool_summary}]")
                
                # Detailed threadpool info for debugging
                pool_detail = "; ".join(
                    f"{p.get('internal_api', '?')}|{p.get('user_api', '?')}:{p.get('num_threads', '?')}"
                    for p in pools
                )
                logger.info(f"[child] Threadpools: {pool_detail}")
                
                # Check for dangerous libiomp5 presence in CPU families
                # Only warn if MKL_THREADING_LAYER is NOT set to GNU (which forces libgomp)
                mkl_layer = _os.getenv("MKL_THREADING_LAYER", "")
                for p in pools:
                    filepath = p.get('filepath', '')
                    if 'iomp' in filepath or 'libmkl_rt' in filepath:
                        if mkl_layer != "GNU":
                            logger.warning(f"⚠️  Intel OpenMP detected in CPU family: {filepath}")
                            logger.warning("    This can cause conflicts with libgomp!")
                            logger.warning(f"    Set MKL_THREADING_LAYER=GNU to use libgomp instead")
                        else:
                            logger.debug(f"Intel OpenMP detected but MKL_THREADING_LAYER=GNU is set (using libgomp): {filepath}")
            except Exception as e:
                logger.debug(f"Could not log threadpool info: {e}")
            
            # Log BLAS configuration for diagnostics (especially for risky families)
            logger.info(
                "[BLAS] OMP=%s MKL=%s OPENBLAS=%s LAYER=%s solver=%s",
                _os.getenv("OMP_NUM_THREADS"),
                _os.getenv("MKL_NUM_THREADS"),
                _os.getenv("OPENBLAS_NUM_THREADS"),
                _os.getenv("MKL_THREADING_LAYER"),
                _os.getenv("SKLEARN_RIDGE_SOLVER", "auto"),
            )
            
            # Check memory cap before starting
            _enforce_memcap(payload_path)
            
            # Load data from memmap if spec provided (much faster than copying)
            import numpy as np
            if isinstance(X_or_spec, dict) and X_or_spec.get("mode") == "memmap":
                X_ = np.load(X_or_spec["X"], mmap_mode="r")
                y_ = np.load(X_or_spec["y"], mmap_mode="r")
                logger.info(f"[child] Loaded data via memmap: X={X_.shape}, y={y_.shape}")
            else:
                X_ = X_or_spec
                y_ = y_unused
            
            # CRITICAL: Lazy import - only load the specific trainer module needed
            # This prevents CPU families from importing TF/Torch modules
            family_base_raw = cls.replace("Trainer", "")
            
            # CRITICAL: Normalize family name to snake_case before lookup
            # TRAINER_MODULE_MAP uses snake_case keys (e.g., "neural_network", not "NeuralNetwork")
            try:
                from TRAINING.common.utils.sst_contract import normalize_family
                family_base = normalize_family(family_base_raw)
            except ImportError:
                # Fallback: simple lowercase conversion if normalize_family not available
                import re
                # Convert CamelCase to snake_case
                family_base = re.sub(r'(?<!^)(?=[A-Z])', '_', family_base_raw).lower()
            
            # Look up the specific module for this family
            if family_base in TRAINER_MODULE_MAP:
                module_path, class_name = TRAINER_MODULE_MAP[family_base]
                logger.info("🔍 [child] Lazy import: %s from %s", class_name, module_path)
            else:
                # Fallback to old behavior for unknown families
                module_path = mod
                class_name = cls
                logger.warning("⚠️  [child] Family %s (normalized from %s) not in TRAINER_MODULE_MAP, using fallback", family_base, family_base_raw)
            
            # Bootstrap framework runtime (TF/Torch) BEFORE importing trainer
            # This centralizes GPU detection and avoids duplicating it in every trainer
            _bootstrap_family_runtime(family_base, logger)
            
            # Now import the specific trainer module
            Trainer = getattr(importlib.import_module(module_path), class_name)
            trainer = Trainer(**(kwargs or {}))
            
            # Train the model with proper threading
            # set_estimator_threads will push the thread count into the estimator if supported
            m = trainer.train(X_, y_)
            if m is None:
                raise RuntimeError("trainer.train() returned None")
            
            # Check memory cap after training
            _enforce_memcap(payload_path)
            
            # Save the model atomically (write to .tmp then rename)
            tmp = payload_path + ".tmp"
            logger.info(f"[child] Saving model to {payload_path}")
            joblib.dump({"model": m}, tmp, compress=3)
            # Fsync for extra safety
            with open(tmp, "rb") as f:
                _os.fsync(f.fileno())
            _os.replace(tmp, payload_path)
            logger.info(f"✅ [Isolation] {cls} training completed successfully (peak RSS: {PEAK_GB:.1f}GB)")
            logger.info(f"[child] Payload saved successfully to {payload_path}")
        
    except Exception as e:
        import traceback as tb
        error_msg = "".join(tb.format_exception(e))
        logger.error(f"❌ [Isolation] {cls} training failed: {error_msg}")
        # Always write a payload, even on error
        try:
            tmp = payload_path + ".tmp"
            joblib.dump({"error": error_msg}, tmp)
            _os.replace(tmp, payload_path)
        except Exception:
            # If we can't even write the error, write a minimal payload
            try:
                tmp = payload_path + ".tmp"
                joblib.dump({"error": "failed_to_save_error", "original_error": str(e)}, tmp)
                _os.replace(tmp, payload_path)
            except Exception:
                # As a last resort, write a text error next to payload path
                try:
                    with open(payload_path + ".error.txt", "w") as f:
                        f.write(error_msg)
                except Exception:
                    pass
        # Non-zero exit so parent can see failure quickly
        _os._exit(1)
    finally:
        # Stop memory watchdog and log peak usage
        STOP = True
        logger.info(f"[child] peak RSS: {PEAK_GB:.2f} GB")
