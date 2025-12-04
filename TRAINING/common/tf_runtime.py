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
TensorFlow Runtime Management

This module provides centralized TensorFlow initialization to prevent
"cannot be modified after initialization" errors and ensure consistent
threading configuration across all TF families.
"""

import os
import logging

logger = logging.getLogger(__name__)

_TF = None  # cached module

def ensure_tf_initialized(cpu_only: bool = False, intra: int = 1, inter: int = 1, use_mixed: bool = True):
    """
    Initialize TensorFlow exactly once with proper threading configuration.
    
    Args:
        cpu_only: If True, force CPU-only mode
        intra: Number of intra-op threads
        inter: Number of inter-op threads  
        use_mixed: If True, enable mixed precision
        
    Returns:
        tensorflow module
    """
    global _TF
    if _TF is not None:
        return _TF  # already initialized

    # Respect "CPU only" families to keep CPU free for OMP learners
    if cpu_only:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    # Import AFTER env is set
    import tensorflow as tf

    # Threads must be set before any heavy TF ops occur
    try:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra))
        tf.config.threading.set_inter_op_parallelism_threads(int(inter))
        logger.info(f"TF threading configured: intra={intra}, inter={inter}")
    except RuntimeError as e:
        # If TF was already initialized, we can't change threads.
        # Don't crash—just continue.
        logger.warning(f"Could not set TF threading (already initialized): {e}")

    # Optional: safer GPU mem behavior
    try:
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Enabled memory growth for GPU: {gpu}")
    except Exception as e:
        logger.warning(f"Could not configure GPU memory growth: {e}")

    if use_mixed:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy("mixed_float16")
            logger.info("Enabled mixed precision training")
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")

    _TF = tf
    return tf

def reset_tf_runtime():
    """Reset the cached TF module (for testing)."""
    global _TF
    _TF = None
