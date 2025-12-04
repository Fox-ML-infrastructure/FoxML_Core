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

"""
Runtime Policy - Single Source of Truth for Model Family Execution

This module defines HOW each model family should run:
- Process isolation vs in-process
- GPU requirements
- Backend frameworks (TF, PyTorch, XGBoost, CuPy)
- Threading policies
- Memory limits

The harness (train_with_strategies.py) enforces these policies uniformly.
"""


from dataclasses import dataclass
from typing import FrozenSet, Dict, Optional
import logging
import sys

# Configure logger for this module (use stderr to avoid corrupting script output)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler(sys.stderr)  # Use stderr for module-level logs
    handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class RuntimePolicy:
    """
    Runtime policy for a model family.
    
    Attributes:
        run_mode: "process" (isolated child) or "inproc" (parent process)
        needs_gpu: Whether this family requires/can use GPU
        backends: Set of frameworks used: {"tf", "torch", "xgb", "cupy"}
        omp_user_api: Threading API to prioritize: "openmp", "blas", or None
        cap_vram_mb: TensorFlow VRAM cap (MB), None = no cap
        force_isolation_reason: Why this family must be isolated (for docs)
    """
    run_mode: str                           # "process" | "inproc"
    needs_gpu: bool                         # True if GPU-capable/required
    backends: FrozenSet[str]               # {"tf", "torch", "xgb", "cupy"}
    omp_user_api: Optional[str] = "openmp"  # "openmp" | "blas" | None
    cap_vram_mb: Optional[int] = None       # TF memory limit (MB)
    force_isolation_reason: str = ""        # Why isolated (for logging)


# ============================================================================
# CROSS-SECTIONAL MODEL POLICIES (Priority)
# ============================================================================

CROSS_SECTIONAL_POLICIES: Dict[str, RuntimePolicy] = {
    # ---- GPU Families (MUST isolate for memory cleanup) ----
    
    "XGBoost": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"xgb"}),
        omp_user_api="openmp",
        force_isolation_reason="GPU allocator pools don't release until process exit"
    ),
    
    "MLP": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "VAE": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "GAN": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "MultiTask": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    # ---- CPU Families (in-process for speed) ----
    
    "LightGBM": RuntimePolicy(
        run_mode="inproc",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="openmp"
    ),
    
    "QuantileLightGBM": RuntimePolicy(
        run_mode="inproc",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="openmp"
    ),
    
    "NGBoost": RuntimePolicy(
        run_mode="inproc",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="openmp"
    ),
    
    "Ensemble": RuntimePolicy(
        run_mode="inproc",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="openmp"
    ),
    
    "MetaLearning": RuntimePolicy(
        run_mode="inproc",
        needs_gpu=False,
        backends=frozenset(),  # Pure sklearn
        omp_user_api="openmp"
    ),
    
    "FTRLProximal": RuntimePolicy(
        run_mode="inproc",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="blas"
    ),
    
    # ---- BLAS-Heavy Families (isolate to prevent MKL crashes) ----
    
    "ChangePoint": RuntimePolicy(
        run_mode="process",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="blas",
        force_isolation_reason="MKL/OpenMP conflicts cause segfaults in scipy.linalg.solve"
    ),
    
    "GMMRegime": RuntimePolicy(
        run_mode="process",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="blas",
        force_isolation_reason="Large BLAS operations + MKL threading issues"
    ),
    
    "RewardBased": RuntimePolicy(
        run_mode="process",
        needs_gpu=False,
        backends=frozenset(),
        omp_user_api="blas",
        force_isolation_reason="Complex Ridge operations prone to MKL conflicts"
    ),
}


# ============================================================================
# SEQUENTIAL MODEL POLICIES
# ============================================================================

SEQUENTIAL_POLICIES: Dict[str, RuntimePolicy] = {
    # ---- GPU Sequential Families ----
    
    "CNN1D": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "LSTM": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "Transformer": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "TabLSTM": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "TabTransformer": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
    
    "TabCNN": RuntimePolicy(
        run_mode="process",
        needs_gpu=True,
        backends=frozenset({"tf"}),
        omp_user_api="openmp",
        cap_vram_mb=4096,
        force_isolation_reason="TensorFlow CUDA context persists in-process"
    ),
}


# ============================================================================
# UNIFIED POLICY REGISTRY
# ============================================================================

# Merge cross-sectional and sequential policies
POLICY: Dict[str, RuntimePolicy] = {
    **CROSS_SECTIONAL_POLICIES,
    **SEQUENTIAL_POLICIES,
}


# Default policy for unknown families (conservative: isolate, no GPU)
DEFAULT_POLICY = RuntimePolicy(
    run_mode="process",
    needs_gpu=False,
    backends=frozenset(),
    omp_user_api="openmp",
    force_isolation_reason="Unknown family - conservative isolation"
)


# ============================================================================
# PUBLIC API
# ============================================================================

def get_policy(family: str) -> RuntimePolicy:
    """
    Get runtime policy for a model family.
    
    Args:
        family: Family name (e.g., "XGBoost", "MLP", "LightGBM")
    
    Returns:
        RuntimePolicy instance with execution requirements
    """
    policy = POLICY.get(family, DEFAULT_POLICY)
    
    if family not in POLICY:
        logger.warning(
            f"[RuntimePolicy] Unknown family '{family}' - using default policy: "
            f"run_mode={DEFAULT_POLICY.run_mode}, needs_gpu={DEFAULT_POLICY.needs_gpu}"
        )
    
    return policy


def should_isolate(family: str) -> bool:
    """Check if family requires process isolation."""
    return get_policy(family).run_mode == "process"


def needs_gpu(family: str) -> bool:
    """Check if family requires/can use GPU."""
    return get_policy(family).needs_gpu


def get_backends(family: str) -> FrozenSet[str]:
    """Get set of backend frameworks used by family."""
    return get_policy(family).backends


def get_threading_api(family: str) -> Optional[str]:
    """Get preferred threading API (openmp/blas/None)."""
    return get_policy(family).omp_user_api


def get_vram_cap(family: str) -> Optional[int]:
    """Get TensorFlow VRAM cap in MB, or None for no cap."""
    return get_policy(family).cap_vram_mb


def log_policy_summary():
    """Log summary of all runtime policies (useful for debugging)."""
    logger.info("=" * 80)
    logger.info("RUNTIME POLICY SUMMARY")
    logger.info("=" * 80)
    
    isolated = [f for f, p in POLICY.items() if p.run_mode == "process"]
    inproc = [f for f, p in POLICY.items() if p.run_mode == "inproc"]
    gpu_enabled = [f for f, p in POLICY.items() if p.needs_gpu]
    
    logger.info(f"📦 Isolated (process): {len(isolated)} families")
    for fam in sorted(isolated):
        pol = POLICY[fam]
        reason = f" ({pol.force_isolation_reason})" if pol.force_isolation_reason else ""
        logger.info(f"  • {fam}{reason}")
    
    logger.info(f"⚡ In-process: {len(inproc)} families")
    for fam in sorted(inproc):
        logger.info(f"  • {fam}")
    
    logger.info(f"🎮 GPU-enabled: {len(gpu_enabled)} families")
    for fam in sorted(gpu_enabled):
        pol = POLICY[fam]
        cap = f", VRAM cap={pol.cap_vram_mb}MB" if pol.cap_vram_mb else ""
        logger.info(f"  • {fam} (backends={list(pol.backends)}{cap})")
    
    logger.info("=" * 80)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_policies():
    """
    Validate all policies for consistency.
    Raises ValueError if any policy is invalid.
    """
    errors = []
    
    for family, policy in POLICY.items():
        # Check run_mode
        if policy.run_mode not in ("process", "inproc"):
            errors.append(f"{family}: invalid run_mode '{policy.run_mode}'")
        
        # Check backends
        valid_backends = {"tf", "torch", "xgb", "cupy"}
        invalid = policy.backends - valid_backends
        if invalid:
            errors.append(f"{family}: invalid backends {invalid}")
        
        # Check omp_user_api
        if policy.omp_user_api not in ("openmp", "blas", None):
            errors.append(f"{family}: invalid omp_user_api '{policy.omp_user_api}'")
        
        # GPU families should specify backends
        if policy.needs_gpu and not policy.backends:
            errors.append(f"{family}: needs_gpu=True but backends is empty")
        
        # Process isolation should have a reason (for GPU or crash-prone families)
        if policy.run_mode == "process" and not policy.force_isolation_reason:
            errors.append(f"{family}: isolated but no force_isolation_reason")
    
    if errors:
        raise ValueError(f"Runtime policy validation failed:\n" + "\n".join(f"  • {e}" for e in errors))
    
    logger.info(f"✅ Runtime policy validation passed ({len(POLICY)} families)")


# Run validation on import
try:
    validate_policies()
except ValueError as e:
    logger.error(f"❌ Runtime policy validation failed: {e}")
    raise

