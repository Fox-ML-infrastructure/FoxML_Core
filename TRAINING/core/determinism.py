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
Global Determinism Setup - Mega Script Integration
Ensures reproducible results across all training operations.
"""


import os
import random
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def set_global_determinism(seed: int = 42) -> None:
    """Set global determinism for all random operations (mega script approach)."""
    
    # Set environment variables (must be done before imports)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # More deterministic kernels
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    
    # Set Python random seeds
    random.seed(seed)
    np.random.seed(seed)
    
    # Set TensorFlow seeds (will be called when TF is imported)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)  # TF ≥ 2.13
        tf.random.set_seed(seed)
        logger.info(f"✅ TensorFlow determinism set with seed {seed}")
    except Exception as e:
        logger.warning(f"⚠️ TensorFlow determinism setup failed: {e}")
    
    # Set threading defaults (mega script approach)
    DEFAULT_THREADS = str(max(1, (os.cpu_count() or 2) - 1))
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_var, DEFAULT_THREADS)
    
    logger.info(f"✅ Global determinism set with seed {seed}")

def ensure_deterministic_environment() -> None:
    """Ensure deterministic environment for training."""
    
    # Set threading defaults
    DEFAULT_THREADS = str(max(1, (os.cpu_count() or 2) - 1))
    for env_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_var, DEFAULT_THREADS)
    
    # Set wait policy for better performance
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    
    logger.info("✅ Deterministic environment configured")

def get_deterministic_params(seed: Optional[int] = None) -> dict:
    """Get deterministic parameters for model training."""
    if seed is None:
        seed = 42
    
    return {
        'random_state': seed,
        'seed': seed,
        'n_jobs': 1,  # Avoid threading issues
        'deterministic': True
    }

def seed_for(operation: str, base_seed: int = 42) -> int:
    """Get a deterministic seed for a specific operation."""
    # Use operation name to create deterministic but different seeds
    operation_hash = hash(operation) % 1000
    return base_seed + operation_hash
