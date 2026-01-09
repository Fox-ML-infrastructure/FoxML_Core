# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
    # Workaround for libtensorflow_cc.so.2 executable stack issue
    # This is a known issue on some Linux systems - try to import gracefully
    try:
        import tensorflow as tf
    except Exception as e:
        error_msg = str(e).lower()
        if "executable stack" in error_msg or "libtensorflow" in error_msg:
            logger.error(f"TensorFlow import failed (system library issue): {e}")
            logger.error("This is a system-level TensorFlow installation problem.")
            logger.error("Possible fixes:")
            logger.error("  1. Reinstall TensorFlow: pip install --upgrade --force-reinstall tensorflow")
            logger.error("  2. Check system security settings (SELinux, AppArmor)")
            logger.error("  3. Try: execstack -c $(python -c 'import tensorflow; print(tensorflow.__file__)')/../libtensorflow_cc.so.2")
            logger.error("TensorFlow families will be skipped.")
            raise ImportError(f"TensorFlow not available due to system library issue: {e}")
        raise

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
