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
Environment Configuration - Mega Script Integration
Sets up the environment for optimal training performance.
"""


import os
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def setup_training_environment(config: Optional[Dict[str, Any]] = None) -> None:
    """Set up the training environment with mega script optimizations."""
    
    if config is None:
        config = {}
    
    # GPU configuration
    if config.get('cpu_only', False):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        logger.info("🔧 CPU-only mode enabled")
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
        logger.info("🔧 GPU mode enabled")
    
    # TensorFlow configuration
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    
    # Memory optimization
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("XGB_GPU_MEMORY_GROWTH", "true")
    os.environ.setdefault("LIGHTGBM_GPU_MEMORY_GROWTH", "true")
    
    # Threading configuration
    setup_threading_environment(config)
    
    # Memory management
    setup_memory_environment(config)
    
    logger.info("✅ Training environment configured")

def setup_threading_environment(config: Dict[str, Any]) -> None:
    """Set up threading environment for optimal performance."""
    
    # Get thread count from config or use default
    num_threads = config.get('num_threads', max(1, (os.cpu_count() or 2) - 1))
    
    # Set threading environment variables
    threading_vars = {
        "OMP_NUM_THREADS": str(num_threads),
        "MKL_NUM_THREADS": str(num_threads),
        "OPENBLAS_NUM_THREADS": str(num_threads),
        "VECLIB_MAXIMUM_THREADS": str(num_threads),
        "NUMEXPR_NUM_THREADS": str(num_threads),
        "OMP_WAIT_POLICY": "PASSIVE",
        "KMP_BLOCKTIME": "0"
    }
    
    for var, value in threading_vars.items():
        os.environ[var] = value
    
    logger.info(f"🔧 Threading configured: {num_threads} threads")

def setup_memory_environment(config: Dict[str, Any]) -> None:
    """Set up memory environment for optimal performance."""
    
    # Memory management settings
    memory_vars = {
        "MAX_GPU_MEMORY_FRACTION": str(config.get('max_gpu_memory_fraction', 0.70)),
        "MAX_CPU_MEMORY_FRACTION": str(config.get('max_cpu_memory_fraction', 0.80)),
        "MEMORY_CLEANUP_AGGRESSIVE": str(config.get('aggressive_cleanup', True)),
        "BATCH_MEMORY_CLEANUP": str(config.get('batch_cleanup', True)),
        "MODEL_MEMORY_CLEANUP": str(config.get('model_cleanup', True))
    }
    
    for var, value in memory_vars.items():
        os.environ[var] = value
    
    logger.info("🔧 Memory management configured")

def setup_gpu_environment() -> None:
    """Set up GPU environment for optimal performance."""
    
    try:
        import tensorflow as tf
        
        # Configure GPU memory growth
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ GPU memory growth enabled for {len(gpus)} GPU(s)")
        else:
            logger.warning("⚠️ No GPUs detected")
            
    except ImportError:
        logger.warning("⚠️ TensorFlow not available for GPU setup")
    except Exception as e:
        logger.warning(f"⚠️ GPU setup failed: {e}")

def get_environment_info() -> Dict[str, Any]:
    """Get current environment information."""
    
    info = {
        'cpu_count': os.cpu_count(),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
        'python_hashseed': os.environ.get('PYTHONHASHSEED', 'Not set'),
        'tf_deterministic_ops': os.environ.get('TF_DETERMINISTIC_OPS', 'Not set'),
        'omp_num_threads': os.environ.get('OMP_NUM_THREADS', 'Not set'),
        'mkl_num_threads': os.environ.get('MKL_NUM_THREADS', 'Not set')
    }
    
    return info
