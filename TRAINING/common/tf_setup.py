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

# common/tf_setup.py
"""
TensorFlow threading setup helper.
Call this BEFORE any TF ops/vars/keras models to avoid "intra op cannot be modified" errors.
"""
import logging

logger = logging.getLogger(__name__)

def tf_thread_setup(intra=1, inter=1, allow_growth=True):
    """
    Configure TensorFlow threading and GPU memory growth.
    Safe to call multiple times - won't crash if TF already initialized.
    
    Args:
        intra: Intra-op parallelism threads
        inter: Inter-op parallelism threads
        allow_growth: Enable GPU memory growth
    
    Returns:
        tensorflow module (to avoid "tf referenced before assignment" errors)
    """
    try:
        import tensorflow as tf
        
        # Try to set threading (safe if already initialized)
        try:
            tf.config.threading.set_intra_op_parallelism_threads(int(intra))
            tf.config.threading.set_inter_op_parallelism_threads(int(inter))
            logger.info(f"TF threading configured: intra={intra}, inter={inter}")
        except RuntimeError:
            # Already initialized: don't die; just continue with current settings
            logger.debug("TF threading already initialized, skipping")
        
        # GPU memory growth if enabled
        if allow_growth:
            try:
                gpus = tf.config.list_physical_devices("GPU")
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except Exception:
                        pass
            except Exception:
                pass
        
        return tf
    except ImportError:
        logger.warning("TensorFlow not available")
        return None

def apply_dataset_options(ds, private_pool=True, pool_size=None, deterministic=False):
    """
    Apply safe tf.data options that work across TF versions.
    Avoids 'use_unbounded_threadpool' warning.
    
    Args:
        ds: tf.data.Dataset
        private_pool: Use private threadpool for this dataset
        pool_size: Size of private threadpool (default: cpu_count - 1)
        deterministic: Whether to enforce deterministic execution
    
    Returns:
        Dataset with options applied
    """
    try:
        import tensorflow as tf
        import os
        
        opts = tf.data.Options()
        
        # Threading options (widely supported)
        opts.threading.max_intra_op_parallelism = 1
        if private_pool:
            # Use private pool but NOT unbounded (avoids the warning)
            opts.threading.private_threadpool_size = pool_size or max(1, (os.cpu_count() or 8) - 1)
            # Note: NOT setting use_unbounded_threadpool (it's not in all TF versions)
        
        # Performance optimizations
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.deterministic = deterministic
        
        return ds.with_options(opts)
    except Exception as e:
        logger.debug(f"Could not apply dataset options: {e}")
        return ds
