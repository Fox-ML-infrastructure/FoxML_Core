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

# common/threads.py
"""
Unified threading & runtime control for all trainers.
- Consistent OMP/MKL/BLAS threading across families
- Prevents TF/JAX/XLA GPU init for CPU-only families
- Avoids OMP_DYNAMIC throttling
- Provides safe TF thread setup utility
"""
from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from threadpoolctl import threadpool_limits, threadpool_info, ThreadpoolController
except Exception:
    threadpool_limits = None
    ThreadpoolController = None
    def threadpool_info():  # type: ignore
        return []


# ============= CPU Affinity Helpers =============

def allowed_cpus() -> list[int]:
    """
    Get list of CPUs this process is allowed to use (respects cgroup/taskset).
    
    Returns:
        Sorted list of CPU IDs we're allowed to use
    """
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        # Windows or older Python - assume all CPUs available
        return list(range(os.cpu_count() or 1))


def effective_threads(requested: int | None = None) -> int:
    """
    Cap requested threads by what we're actually allowed to use.
    
    Args:
        requested: Desired thread count (None = use default)
        
    Returns:
        Thread count capped by allowed CPUs
    """
    n_allowed = len(allowed_cpus())
    req = requested if (requested and requested > 0) else default_threads()
    return max(1, min(req, n_allowed))

# Family groups
CPU_FAMS = {
    "LightGBM", "QuantileLightGBM", "RewardBased",
    "NGBoost", "GMMRegime", "ChangePoint",
    "FTRLProximal", "Ensemble"
}
GPU_TF_FAMS = {"MLP", "CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer", 
               "VAE", "GAN", "MetaLearning", "MultiTask"}
GPU_TORCH   = set()  # Currently no pure PyTorch trainers (all use TF/Keras)

# Families dominated by BLAS/LAPACK linear algebra (MKL/OpenBLAS)
BLAS_HEAVY_FAMS = {
    "GMMRegime", "FTRLProximal", "ChangePoint",
    "LinearCombine", "VotingCombine"
}

# Families that are BLAS-only (no benefit from OpenMP, avoid runtime conflicts)
# These use pure linear algebra (Ridge, Lasso, etc.) via SciPy/NumPy
BLAS_ONLY_FAMS = {
    "RewardBased"  # Ridge solver via SciPy - no OpenMP needed
}

# Heuristic class name buckets (no sklearn import here to keep imports light)
OMP_HEAVY_CLASSES = {
    "RandomForestRegressor", "RandomForestClassifier",
    "ExtraTreesRegressor", "ExtraTreesClassifier",
    "HistGradientBoostingRegressor", "HistGradientBoostingClassifier",
    "LGBMRegressor", "LGBMClassifier",
    "XGBRegressor", "XGBClassifier",
}

BLAS_HEAVY_CLASSES = {
    "GaussianMixture", "BayesianGaussianMixture",
    "LinearRegression", "Ridge", "RidgeCV", "BayesianRidge",
    "Lasso", "ElasticNet", "ElasticNetCV", "SGDRegressor", "SGDClassifier",
    "PCA", "TruncatedSVD", "FactorAnalysis",
}

# Quick toggle for debugging
DISABLE_GUARD = os.getenv("TRAINER_DISABLE_THREAD_GUARD", "0") == "1"

# ============= BLAS Threading Context Manager =============

@contextmanager
def blas_threads(n: int):
    """
    Context manager to safely control BLAS/OpenMP threads for CPU-intensive operations.
    
    Uses threadpoolctl to dynamically limit threads without environment variable conflicts.
    This is the SAFE way to multithread sklearn/scipy operations.
    
    Args:
        n: Number of threads to use for BLAS/OpenMP operations
        
    Usage:
        with blas_threads(12):
            model.fit(X, y)  # Uses 12 BLAS threads
    """
    if threadpool_limits is None:
        # Fallback: set env vars (less safe, but works)
        old_omp = os.environ.get("OMP_NUM_THREADS")
        old_mkl = os.environ.get("MKL_NUM_THREADS")
        old_openblas = os.environ.get("OPENBLAS_NUM_THREADS")
        
        try:
            os.environ["OMP_NUM_THREADS"] = str(n)
            os.environ["MKL_NUM_THREADS"] = str(n)
            os.environ["OPENBLAS_NUM_THREADS"] = str(n)
            yield
        finally:
            if old_omp is not None:
                os.environ["OMP_NUM_THREADS"] = old_omp
            if old_mkl is not None:
                os.environ["MKL_NUM_THREADS"] = old_mkl
            if old_openblas is not None:
                os.environ["OPENBLAS_NUM_THREADS"] = old_openblas
    else:
        # Preferred: use threadpoolctl (dynamic, no env pollution)
        # user_api=None (default) applies to all supported libs (BLAS + OpenMP)
        with threadpool_limits(limits=n):
            yield


def set_default_env_for_cpu():
    """
    Set sensible BLAS/OpenMP defaults for CPU-intensive training.
    
    Call ONCE in main/child BEFORE importing numpy/scipy/sklearn.
    These settings prevent common threading issues while allowing parallelism.
    """
    # Allow dynamic thread adjustment (prevents oversubscription)
    os.environ.setdefault("MKL_DYNAMIC", "TRUE")
    os.environ.setdefault("OMP_DYNAMIC", "TRUE")
    
    # Optimize for throughput (don't spin-wait)
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    
    # Affinity for NUMA systems (compact = better cache locality)
    os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")
    
    # Prevent nested parallelism issues
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # Default, override with blas_threads()
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def compute_blas_threads_for_family(family: str, total_cores: int, outer_jobs: int = 1) -> int:
    """
    Compute optimal BLAS thread count for a family.
    
    Args:
        family: Family name (e.g., "ChangePoint", "GMMRegime")
        total_cores: Total available CPU cores
        outer_jobs: Number of parallel jobs at task level (e.g., CV folds)
        
    Returns:
        Recommended BLAS thread count
    """
    # If using task-level parallelism, reduce BLAS threads to avoid oversubscription
    if outer_jobs > 1:
        return max(1, total_cores // outer_jobs)
    
    # Family-specific tuning
    if family in {"NGBoost"}:
        # NGBoost doesn't parallelize well internally - use 1 BLAS thread
        # and parallelize at the CV/task level instead
        return 1
    elif family in {"ChangePoint", "GMMRegime", "FTRLProximal", "MultiTask"}:
        # BLAS-heavy families benefit from many threads
        # Leave 1-2 cores for system
        return max(1, total_cores - 2)
    else:
        # Default: use most cores
        return max(1, total_cores - 1)

# ---------------- basics ----------------

def _to_int(v, default):
    try:
        return int(v)
    except Exception:
        return default

def default_threads() -> int:
    return max(1, (os.cpu_count() or 8) - 1)

def plan_for_family(family: str, total_threads: int) -> dict[str, int]:
    """
    Decide OMP/MKL threads for a family.
    - BLAS-only families: OMP=1, MKL=1 (avoid OpenMP runtime conflicts)
    - OMP-heavy families (LGBM/XGB/RF/HGB): use full thread budget
    - GPU families: keep CPU light (OMP=1, MKL=1)
    - Never request more threads than we're allowed to use
    """
    t = effective_threads(total_threads)  # Cap by allowed CPUs
    
    # BLAS-only families (pure linear algebra, no OpenMP benefit)
    # Give MKL the threads, keep OpenMP minimal to avoid runtime conflicts
    if family in BLAS_ONLY_FAMS:
        return {"OMP": 1, "MKL": t}  # MKL gets all cores for BLAS operations
    
    # OMP-heavy families (tree/histogram models) - USE FULL THREAD BUDGET
    OMP_HEAVY = {"LightGBM", "QuantileLightGBM", "XGBoost", 
                 "RandomForest", "HistGradientBoosting"}
    if family in OMP_HEAVY:
        return {"OMP": t, "MKL": 1}
    
    # Other CPU-heavy families (may have mixed workloads)
    if family in CPU_FAMS:
        omp = min(12, t)
        mkl = 1
    # GPU families (neural networks)
    elif family in (GPU_TF_FAMS | GPU_TORCH):
        omp = 1
        mkl = 1
    # Mixed/other families
    else:
        omp = min(12, max(1, t // 2))
        mkl = 1
    return {"OMP": omp, "MKL": mkl}

# ---------------- Estimator-level planning (smart detection) ----------------

def _clsname(obj: Any) -> str:
    """Get class name safely."""
    try:
        return obj.__class__.__name__
    except Exception:
        return str(type(obj))

def is_omp_heavy_estimator(est: Any) -> bool:
    """Detect if estimator uses OpenMP parallelism (tree/histogram models)."""
    name = _clsname(est)
    return (name in OMP_HEAVY_CLASSES) or ("HistGradient" in name)

def is_blas_heavy_estimator(est: Any) -> bool:
    """Detect if estimator uses BLAS/LAPACK (linear algebra models)."""
    name = _clsname(est)
    return (name in BLAS_HEAVY_CLASSES) or any(k in name for k in ("Linear", "Ridge", "Lasso", "Elastic"))

def plan_for_estimator(family: str, est: Any, total_threads: int, phase: str = "fit") -> dict[str, int]:
    """
    Decide OMP/MKL for a specific estimator or phase.
    - RF/HGB/LGBM/XGB → OpenMP threads (MKL=1)
    - Linear solves / GMM / PCA → MKL threads (OMP=1)
    - Meta/combine phases → prefer MKL
    
    Args:
        family: Model family name
        est: Estimator object (can be None)
        total_threads: Total available threads
        phase: "fit", "predict", "meta", "linear_solve"
    
    Returns:
        Dict with OMP and MKL thread counts
    """
    t = max(1, total_threads)
    
    # Check estimator type first
    if est is not None:
        if is_omp_heavy_estimator(est):
            return {"OMP": min(12, t), "MKL": 1}
        if is_blas_heavy_estimator(est) or phase in ("meta", "linear_solve"):
            return {"OMP": 1, "MKL": min(t, 8)}
    
    # Phase-based hints
    if phase in ("meta", "linear_solve"):
        return {"OMP": 1, "MKL": min(t, 8)}
    
    # Fallback to family-level plan
    if family in BLAS_HEAVY_FAMS:
        return {"OMP": 1, "MKL": min(t, 8)}
    
    return plan_for_family(family, t)

def child_env_for_family(family: str, threads: int, gpu_ok: bool = True) -> dict[str, str]:
    """
    Build environment for an isolated child process *before* importing heavy libs.
    Applies thread policy from family_config.yaml.
    
    NOW USES: runtime_policy.needs_gpu() for GPU detection (single source of truth)
    """
    from .family_config import apply_thread_policy, get_family_info
    from .runtime_policy import needs_gpu, get_backends
    
    plan = plan_for_family(family, threads)
    family_info = get_family_info(family)
    thread_policy = family_info.get("thread_policy", "omp_heavy")
    
    # Use runtime policy to determine GPU need (includes XGBoost!)
    use_gpu = bool(gpu_ok and needs_gpu(family))
    backends = get_backends(family)
    
    # Use "-1" to hide GPUs from TF/JAX reliably in CPU families
    cvd = os.getenv("TRAINER_GPU_IDS", "0") if use_gpu else "-1"
    
    # Get allowed CPUs for GOMP_CPU_AFFINITY (tells libgomp which CPUs to use)
    cpus = allowed_cpus()
    gomp_affinity = ",".join(map(str, cpus))

    # Allow override for debugging (but NOT for BLAS-only or GPU families)
    forced_omp = os.getenv("TRAINER_CHILD_FORCE_OMP")
    if forced_omp and family not in BLAS_ONLY_FAMS and not use_gpu:
        actual_omp = int(forced_omp)
    else:
        actual_omp = plan["OMP"]
    
    env = {
        # Threading knobs
        "OMP_NUM_THREADS": str(actual_omp),
        "MKL_NUM_THREADS": str(plan["MKL"]),
        "OPENBLAS_NUM_THREADS": str(plan["MKL"]),
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_DYNAMIC": "false",     # don't silently drop to 1
        "OMP_PROC_BIND": "FALSE",   # Let OS affinity govern (was TRUE, caused 2-core collapse)
        # Don't set OMP_PLACES or KMP_AFFINITY - they can cause 2-core binding issues
        "GOMP_CPU_AFFINITY": gomp_affinity,  # Tell libgomp which CPUs to use
        "KMP_BLOCKTIME": "0",

        # TF/JAX/XLA hygiene
        "CUDA_VISIBLE_DEVICES": cvd,
        "TF_CPP_MIN_LOG_LEVEL": "3",
        # trainers should read this to *skip* importing TF in GPU families (use runtime policy)
        "TRAINER_CHILD_NO_TF": "0" if (use_gpu and "tf" in backends) else "1",
        # trainers should read this to *skip* importing Torch in GPU families (use runtime policy)
        # CRITICAL: Prevents libiomp5 (PyTorch) from conflicting with libgomp (LightGBM/scikit-learn)
        "TRAINER_CHILD_NO_TORCH": "0" if (use_gpu and "torch" in backends) else "1",
        # JAX often logs/initializes XLA even when unused
        "JAX_PLATFORMS": "gpu" if use_gpu else "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",

        # Optional: where joblib writes temp files (stable path reduces flakiness)
        "JOBLIB_TEMP_FOLDER": os.getenv("JOBLIB_TEMP_FOLDER", os.getenv("TRAINER_TMP", "/tmp")),
    }

    # TF thread env (read by some setups); TF trainers should still call `tf_thread_setup()`
    if use_gpu and "tf" in backends:
        env["TF_NUM_INTRAOP_THREADS"] = "1"
        env["TF_NUM_INTEROP_THREADS"] = "1"
        env["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        # Show CUDA load messages so we can see *why* no GPU appears
        env["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TRAINER_TF_LOG_LEVEL", "1")
        # Make the TF mask explicit (mirrors CUDA_VISIBLE_DEVICES)
        env["TF_VISIBLE_DEVICE_LIST"] = env["CUDA_VISIBLE_DEVICES"]
    else:
        env["TF_NUM_INTRAOP_THREADS"] = str(plan["OMP"])
        env["TF_NUM_INTEROP_THREADS"] = str(max(1, min(plan["OMP"] // 2, 2)))

    # CRITICAL: For BLAS-only families, remove binding knobs that can cause issues
    # Ridge/Lasso solvers don't need OpenMP binding and it can cause segfaults
    if family in BLAS_ONLY_FAMS:
        for knob in ("OMP_PLACES", "KMP_AFFINITY", "GOMP_CPU_AFFINITY"):
            env.pop(knob, None)
        env["OMP_PROC_BIND"] = "FALSE"  # Extra insurance for BLAS

    # Apply thread policy from config (overrides above settings if specified)
    env = apply_thread_policy(family, env)

    return env

@contextmanager
def temp_environ(update: dict[str, str]):
    """Temporarily set environment variables."""
    prev = {k: os.environ.get(k) for k in update}
    try:
        for k, v in update.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

def _save_env(keys):
    """Save current environment variable values."""
    return {k: os.environ.get(k) for k in keys}

def _restore_env(prev):
    """Restore environment variable values."""
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v

def _all_cpus():
    """Get list of all available CPUs."""
    try:
        return sorted(os.sched_getaffinity(0))
    except AttributeError:
        # macOS / non-Linux fallback: assume all
        return list(range(os.cpu_count() or 1))

@contextmanager
def cpu_affinity_guard(cpus: list | None = None):
    """
    Temporarily set process CPU affinity, then restore.
    Prevents trainers from getting pinned to 1-2 cores.
    
    Args:
        cpus: List of CPU IDs to allow, or None for all CPUs
    """
    try:
        prev = os.sched_getaffinity(0)
    except AttributeError:
        # No affinity support on this platform
        prev = None
    
    try:
        if prev is not None:
            target_cpus = set(cpus or _all_cpus())
            os.sched_setaffinity(0, target_cpus)
            logger.debug(f"Set CPU affinity to {len(target_cpus)} cores")
        yield
    finally:
        if prev is not None:
            os.sched_setaffinity(0, prev)
            logger.debug(f"Restored CPU affinity to {len(prev)} cores")

@contextmanager
def predict_guard(omp: int = 12, mkl: int = 1):
    """
    Guard for prediction phase - ensures multi-core prediction.
    Use with RF/HGB/XGB predictions to avoid single-core bottlenecks.
    
    Args:
        omp: OpenMP threads (default 12 for histogram learners)
        mkl: MKL/BLAS threads (default 1 to avoid oversubscription)
    
    Example:
        with predict_guard(omp=12, mkl=1):
            rf.set_params(n_jobs=12)  # parallel across trees
            y_pred = rf.predict(X)
    """
    if threadpool_limits is None:
        yield
        return
    
    with threadpool_limits(limits=omp, user_api="openmp"):
        with threadpool_limits(limits=mkl, user_api="blas"):
            yield

@contextmanager
def thread_guard(*,
                 # new-style
                 omp: int | None = None,
                 mkl: int | None = None,
                 # legacy-style from old threading_utils
                 num_threads: int | None = None,
                 blas_threads: int | None = None,
                 openmp_threads: int | None = None):
    """
    Unified thread guard - accepts BOTH new args (omp/mkl) and legacy args (num_threads/blas_threads/openmp_threads).
    Sets env vars (OMP/MKL/OPENBLAS) AND clamps via threadpoolctl.
    
    Examples:
        # New style
        with thread_guard(omp=12, mkl=1):
            ...
        
        # Legacy style (backward compatible)
        with thread_guard(num_threads=12):
            ...
    """
    # Normalize args - new style takes precedence
    if omp is None:
        omp = openmp_threads if openmp_threads is not None else (num_threads if num_threads is not None else max(1, (os.cpu_count() or 8) - 1))
    if mkl is None:
        mkl = blas_threads if blas_threads is not None else 1
    
    # Set env vars so libraries that read env (LightGBM, numpy, etc.) are aligned
    prev_env = _save_env(["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_DYNAMIC"])
    os.environ["OMP_NUM_THREADS"] = str(omp)
    os.environ["MKL_NUM_THREADS"] = str(mkl)
    os.environ["OPENBLAS_NUM_THREADS"] = str(mkl)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ["OMP_DYNAMIC"] = "false"
    
    # Clamp via threadpoolctl when available
    if DISABLE_GUARD or threadpool_limits is None:
        try:
            yield
        finally:
            _restore_env(prev_env)
        return
    
    try:
        # Limit both pools explicitly: openmp = omp, blas = mkl
        with threadpool_limits(limits=omp, user_api="openmp"):
            with threadpool_limits(limits=mkl, user_api="blas"):
                yield
    finally:
        _restore_env(prev_env)

def set_estimator_threads(est: Any, n_jobs: int) -> Any:
    """
    Recursively push threads into sklearn/xgb/lgbm estimators & pipelines.
    """
    from sklearn.pipeline import Pipeline
    if isinstance(est, Pipeline):
        for _, step in est.steps:
            set_estimator_threads(step, n_jobs)
        return est

    params = getattr(est, "get_params", lambda: {})()
    for knob in ("n_jobs", "nthread", "num_threads"):
        if hasattr(est, "set_params") and knob in params:
            try:
                est.set_params(**{knob: n_jobs})
            except Exception:
                pass
    return est

def log_thread_state(logger_inst: logging.Logger | None = None):
    """Log current thread state for diagnostics."""
    log = logger_inst or logger
    pools = "; ".join(f"{p.get('user_api')}:{p.get('num_threads')}" for p in threadpool_info())
    log.info(
        "Thread state → OMP=%s MKL=%s | pools=[%s]",
        os.getenv("OMP_NUM_THREADS"),
        os.getenv("MKL_NUM_THREADS"),
        pools or "n/a"
    )

# ---------------- TF helper (for TF trainers only) ----------------

def tf_thread_setup(intra: int, inter: int) -> None:
    """
    Call at the TOP of each TF trainer (before any TF ops/vars/keras models).
    Safe no-op if TF isn't available or already initialized.
    """
    try:
        # Respect CPU families: if parent told us to avoid TF, do nothing
        if os.getenv("TRAINER_CHILD_NO_TF", "0") == "1":
            return

        import tensorflow as tf  # local import → only in TF families
        # If a session/context already exists, setting threads raises RuntimeError.
        # Guard with a light try/except to avoid crashing.
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(intra))
            tf.config.threading.set_inter_op_parallelism_threads(int(inter))
        except RuntimeError:
            # Already initialized; nothing to do.
            pass

        # GPU hygiene if enabled
        if os.getenv("CUDA_VISIBLE_DEVICES", "-1") != "-1":
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
    except Exception:
        # Don't let TF setup blow up training
        logger.debug("tf_thread_setup: TF not available or setup skipped", exc_info=True)

# ---------------- Universal guards (use these everywhere) ----------------

from contextlib import nullcontext

@contextmanager
def guard_for_estimator(est: Any | None,
                        *,
                        family: str,
                        threads: int | None = None,
                        phase: str = "fit",
                        reset_aff: bool = True):
    """
    Universal guard to wrap .fit/.predict calls.
    Picks OMP/MKL based on estimator type and phase, sets env + clamps.
    
    This is the ONE guard you should use in BaseTrainer - it handles everything:
    - Detects if estimator is OMP-heavy (RF/HGB) or BLAS-heavy (Ridge/GMM)
    - Sets correct thread counts
    - Optionally resets CPU affinity
    - Clamps via threadpoolctl
    
    Args:
        est: The estimator object (e.g., RandomForestRegressor instance)
        family: Model family name (for fallback if est type unknown)
        threads: Total threads available (defaults to cpu_count-1)
        phase: "fit", "predict", "meta", "linear_solve"
        reset_aff: If True, reset CPU affinity to all cores
    
    Example:
        with guard_for_estimator(rf, family="Ensemble", threads=12):
            rf.fit(X, y)
    """
    t = threads or default_threads()
    plan = plan_for_estimator(family, est, t, phase)
    
    # Diagnostics (enable with TRAINER_DIAG=1)
    if os.getenv("TRAINER_DIAG", "0") == "1":
        est_name = _clsname(est) if est else "None"
        logger.info(f"🔍 guard_for_estimator: family={family} est={est_name} phase={phase} → OMP={plan['OMP']} MKL={plan['MKL']}")
        try:
            from threadpoolctl import threadpool_info
            pools_before = "; ".join(f"{p.get('user_api')}={p.get('num_threads')}" for p in threadpool_info())
            try:
                affinity_before = len(os.sched_getaffinity(0))
            except Exception:
                affinity_before = "n/a"
            logger.info(f"🔍   BEFORE: pools=[{pools_before}] affinity={affinity_before}")
        except Exception:
            pass
    
    outer = cpu_affinity_guard() if reset_aff else nullcontext()
    inner = thread_guard(omp=plan["OMP"], mkl=plan["MKL"])
    with outer:
        with inner:
            if os.getenv("TRAINER_DIAG", "0") == "1":
                try:
                    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
                    omp_eff = _openmp_effective_n_threads()
                except Exception:
                    omp_eff = "n/a"
                try:
                    affinity_during = len(os.sched_getaffinity(0))
                except Exception:
                    affinity_during = "n/a"
                logger.info(f"🔍   DURING: openmp_effective={omp_eff} affinity={affinity_during}")
            yield

@contextmanager
def family_run_scope(family: str, threads: int, *, reset_aff: bool = True):
    """
    Wrap a whole family run (trainer.train) to reset affinity and apply a base plan.
    Use this in the orchestrator around each trainer, then let guard_for_estimator
    refine for sub-steps (e.g., RF fit vs Ridge meta-fit).
    
    Args:
        family: Model family name
        threads: Total threads available
        reset_aff: If True, reset CPU affinity to all cores
    
    Example:
        with family_run_scope("Ensemble", threads=12):
            result = trainer.train(X, y)
    """
    plan = plan_for_family(family, threads)
    outer = cpu_affinity_guard() if reset_aff else nullcontext()
    inner = thread_guard(omp=plan["OMP"], mkl=plan["MKL"])
    with outer:
        with inner:
            yield

# ---------------- Reset helpers (prevent cross-family pollution) ----------------

def reset_affinity(logger_inst: logging.Logger | None = None):
    """
    Unpin the current process to all available CPUs.
    Call before each family to prevent inherited pinning from child processes/cgroups.
    """
    log = logger_inst or logger
    try:
        avail = os.sched_getaffinity(0)
        os.sched_setaffinity(0, avail)  # re-apply full set
        log.info("✅ CPU affinity reset to all cores: %s", sorted(avail))
    except AttributeError:
        log.debug("reset_affinity: sched_* not available on this platform")
    except Exception as e:
        log.warning("reset_affinity: could not reset affinity: %s", e)

def reset_threadpools():
    """
    Drop any threadpoolctl clamps so the next family starts fresh.
    Call after each family to prevent thread pollution cascade.
    """
    try:
        if ThreadpoolController is not None:
            ThreadpoolController().reset()
    except Exception:
        # best-effort; don't crash if threadpoolctl not present
        pass


# ============= GPU Guard Helpers =============

@contextmanager
def cpu_only_guard():
    """
    Context manager to hide GPUs for CPU-only families.
    Restores original GPU visibility on exit.
    
    Usage:
        with cpu_only_guard():
            # Train CPU-only model
            model.fit(X, y)
    """
    prev_cuda = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    prev_tf_visible = os.environ.get("TF_VISIBLE_DEVICE_LIST", None)
    
    try:
        # Hide GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TF_VISIBLE_DEVICE_LIST"] = ""
        logger.debug("[GPU Guard] GPUs hidden for CPU-only training")
        yield
    finally:
        # Restore original GPU visibility
        if prev_cuda is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prev_cuda
            
        if prev_tf_visible is None:
            os.environ.pop("TF_VISIBLE_DEVICE_LIST", None)
        else:
            os.environ["TF_VISIBLE_DEVICE_LIST"] = prev_tf_visible
        
        logger.debug("[GPU Guard] GPU visibility restored")


def ensure_gpu_visible(family: str = "Unknown") -> bool:
    """
    Ensure GPUs are visible to TensorFlow/CUDA.
    Call this at the start of GPU-capable families.
    
    Args:
        family: Family name for logging
        
    Returns:
        True if GPUs are visible, False otherwise
    """
    import gc
    
    # 1) Undo any prior "hide GPU" side-effects
    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_vis == "":
        # Someone hid the GPU - try to restore it
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("TRAINER_GPU_IDS", "0")
        logger.info(f"[{family}] Restored CUDA_VISIBLE_DEVICES from hidden state")
    
    # 2) Check what TensorFlow sees
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        logger.info(f"[{family}] TensorFlow sees {len(gpus)} GPU(s): {[g.name for g in gpus]}")
        
        # 3) Enable memory growth to prevent "all VRAM reserved"
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                logger.debug(f"[{family}] Enabled memory growth for {gpu.name}")
            except Exception as e:
                logger.debug(f"[{family}] Could not set memory growth: {e}")
        
        # 4) Log current CUDA env
        logger.info(f"[{family}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
        
        return len(gpus) > 0
        
    except Exception as e:
        logger.warning(f"[{family}] Could not check GPU visibility: {e}")
        return False


def force_gpu_placement(family: str = "Unknown"):
    """
    Force TensorFlow to place ops on GPU (fail if unavailable).
    Use this for families that MUST use GPU.
    
    Args:
        family: Family name for logging
        
    Raises:
        RuntimeError: If no GPU is available
    """
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError(f"[{family}] No GPU available but GPU is required")
        
        # Enable device placement logging
        tf.debugging.set_log_device_placement(True)
        logger.info(f"[{family}] Forcing GPU placement, device logging enabled")
        
        # Test GPU placement with a simple op
        with tf.device('/GPU:0'):
            _ = tf.constant(1.0)
        
        logger.info(f"[{family}] GPU placement test successful")
        
    except Exception as e:
        logger.error(f"[{family}] GPU placement failed: {e}")
        raise


def hard_cleanup_after_family(family: str = "Unknown"):
    """
    Aggressive GPU/CPU memory cleanup after training a model family.
    Releases TensorFlow, XGBoost, PyTorch, and CuPy resources.
    
    Call this after each model family completes training, BEFORE starting the next one.
    
    Args:
        family: Family name for logging
    """
    import gc
    import sys
    
    # Determine which frameworks were actually loaded
    tf_loaded = 'tensorflow' in sys.modules
    torch_loaded = 'torch' in sys.modules
    cupy_loaded = 'cupy' in sys.modules
    
    # --- TensorFlow ---
    if tf_loaded:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            logger.debug(f"[Cleanup] Cleared TensorFlow session after {family}")
            
            # Heavier hammer: reset global TF runtime (TF 2.4+)
            try:
                tf.config.experimental.reset_context()
                logger.debug(f"[Cleanup] Reset TensorFlow context after {family}")
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"[Cleanup] TF cleanup failed: {e}")
    
    # --- Python garbage collection (2 passes for CUDA refs) ---
    gc.collect()
    gc.collect()
    
    # --- CuPy / RAPIDS (only if loaded) ---
    if cupy_loaded:
        try:
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            logger.debug(f"[Cleanup] Freed CuPy memory pools after {family}")
        except Exception as e:
            logger.debug(f"[Cleanup] CuPy cleanup failed: {e}")
    
    # --- PyTorch (only if loaded AND CUDA is initialized) ---
    if torch_loaded:
        try:
            import torch
            # Only call CUDA ops if CUDA was actually initialized
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.empty_cache()
                # Also collect IPC handles (cheap, helps with multi-process)
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                logger.debug(f"[Cleanup] Cleared PyTorch CUDA cache after {family}")
            else:
                logger.debug(f"[Cleanup] PyTorch CUDA not initialized, skipping torch cleanup for {family}")
        except Exception as e:
            logger.debug(f"[Cleanup] PyTorch cleanup failed: {e}")
    
    # --- Joblib/loky executors (cleanup stray worker refs) ---
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
        logger.debug(f"[Cleanup] Shutdown joblib executor after {family}")
    except Exception:
        pass
    
    logger.info(f"[Cleanup] Hard cleanup completed after {family}")