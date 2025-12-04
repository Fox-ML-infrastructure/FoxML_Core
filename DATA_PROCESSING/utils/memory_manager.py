#!/usr/bin/env python3

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
Consolidated Memory Manager for ML Training Pipeline.
Single source of truth for memory management across all training modules.
"""


import psutil
import logging
import gc
import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Memory configuration with sensible defaults."""
    # System RAM limits
    max_memory_gb: float = 100.0
    warning_threshold: float = 0.8
    cleanup_threshold: float = 0.9
    overhead_factor: float = 1.30
    headroom: float = 0.70
    
    # GPU VRAM limits
    max_vram_gb: float = 8.0
    gpu_warning_threshold: float = 0.8
    gpu_cleanup_threshold: float = 0.9
    
    # Training-specific
    batch_size_auto: bool = True
    parallel_jobs: int = 2
    sequential_training: bool = True
    
    @classmethod
    def load(cls, config_path: Union[str, Path] = "config/memory_limits.yaml") -> "MemoryConfig":
        """Load configuration from YAML file with fallback to defaults."""
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
            
            system = data.get('system', {})
            gpu = data.get('gpu', {})
            training = data.get('training', {}).get('model_training', {})
            
            return cls(
                max_memory_gb=system.get('max_memory_gb', 100.0),
                warning_threshold=system.get('warning_threshold', 0.8),
                cleanup_threshold=system.get('cleanup_threshold', 0.9),
                overhead_factor=system.get('overhead_factor', 1.30),
                headroom=system.get('headroom', 0.70),
                max_vram_gb=gpu.get('max_vram_gb', 8.0),
                gpu_warning_threshold=gpu.get('warning_threshold', 0.8),
                gpu_cleanup_threshold=gpu.get('cleanup_threshold', 0.9),
                batch_size_auto=training.get('batch_size_auto', True),
                parallel_jobs=training.get('parallel_jobs', 2),
                sequential_training=training.get('sequential_training', True),
            )
        except Exception as e:
            logger.warning(f"Could not load memory config from {config_path}: {e}. Using defaults.")
            return cls()


class MemoryManager:
    """Consolidated memory manager with dynamic backpressure and stage-aware limits."""
    
    def __init__(self, config: Optional[MemoryConfig] = None, n_features: int = 64):
        """
        Initialize memory manager.
        
        Args:
            config: Memory configuration (loads default if None)
            n_features: Number of features for batch size computation
        """
        self.config = config or MemoryConfig.load()
        self.n_features = n_features
        
        # Apply environment variables
        self._apply_env()
        
        # Initialize memory monitoring
        self.process = psutil.Process(os.getpid())
        
        # Dynamic backpressure state
        self._backpressure_active = False
        self._original_batch_size = self._compute_batch_size()
        self._current_batch_size = self._original_batch_size
        
        # Dynamic scaling state
        self._scale_factor = 2.0
        self._backoff_threshold = 0.85
        self._last_memory_check = 0
        self._consecutive_low_usage = 0
        
        logger.info(f"Memory manager initialized:")
        logger.info(f"System RAM limit: {self.config.max_memory_gb}GB")
        logger.info(f"GPU VRAM limit: {self.config.max_vram_gb}GB")
        logger.info(f"Auto batch size: {self.config.batch_size_auto}")
        logger.info(f"Computed batch size: {self._original_batch_size} rows")
        logger.info(f"Headroom: {self.config.headroom:.1%}")
        logger.info(f"Overhead factor: {self.config.overhead_factor:.2f}")

    def _apply_env(self):
        """Apply environment variables for optimal memory usage."""
        env_vars = {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
        }
        
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value

    def _compute_batch_size(self) -> int:
        """Compute optimal batch size based on memory budget and features."""
        # Estimate memory per row: features * 4 bytes (float32) + overhead
        bytes_per_row = self.n_features * 4 * self.config.overhead_factor
        available_memory_bytes = self.config.max_memory_gb * 1024**3 * self.config.headroom
        max_rows = int(available_memory_bytes / bytes_per_row)
        
        # Ultra-aggressive bounds - start at 10M, cap at 25M
        min_batch = 10000000  # Start with 10M rows (aggressive)
        max_batch = 25000000   # Cap at 25M rows (ultra-aggressive)
        
        computed_batch = max(min_batch, min(max_rows, max_batch))
        
        logger.info(f"Batch size computation: {self.n_features} features, {bytes_per_row} bytes/row")
        logger.info(f"Available memory: {available_memory_bytes / (1024**3):.1f}GB")
        logger.info(f"Max theoretical rows: {max_rows:,}")
        logger.info(f"Computed batch size: {computed_batch:,} rows")
        
        return computed_batch

    def autotune_batch_size(self, probe_data, target_memory_gb: float = None) -> int:
        """Autotune batch size based on actual memory usage of probe data."""
        if target_memory_gb is None:
            target_memory_gb = self.config.max_memory_gb * self.config.headroom
        
        try:
            # Process a small probe (20k rows)
            probe_size = min(20000, len(probe_data))
            probe_sample = probe_data[:probe_size]
            
            # Measure memory before
            memory_before = self.process.memory_info().rss
            
            # Process the probe (this would be your actual feature computation)
            # For now, we'll estimate based on data size
            if hasattr(probe_sample, 'memory_usage'):
                memory_used = probe_sample.memory_usage(deep=True).sum()
            else:
                # Fallback estimation
                memory_used = len(probe_sample) * self.n_features * 4 * self.config.overhead_factor
            
            # Measure memory after
            memory_after = self.process.memory_info().rss
            actual_memory_delta = memory_after - memory_before
            
            # Calculate bytes per row
            bytes_per_row = max(actual_memory_delta / probe_size, memory_used / probe_size)
            
            # Calculate optimal batch size
            target_memory_bytes = target_memory_gb * 1024**3
            optimal_batch = int(target_memory_bytes / bytes_per_row)
            
            # Apply bounds
            min_batch = 250000
            max_batch = 1000000
            tuned_batch = max(min_batch, min(optimal_batch, max_batch))
            
            logger.info(f"Autotune results:")
            logger.info(f"  Probe size: {probe_size:,} rows")
            logger.info(f"  Memory delta: {actual_memory_delta / (1024**2):.1f} MB")
            logger.info(f"  Bytes per row: {bytes_per_row:.1f}")
            logger.info(f"  Optimal batch: {optimal_batch:,} rows")
            logger.info(f"  Tuned batch: {tuned_batch:,} rows")
            
            # Update current batch size
            self._current_batch_size = tuned_batch
            return tuned_batch
            
        except Exception as e:
            logger.warning(f"Autotune failed: {e}, using default batch size")
            return self._current_batch_size

    def dynamic_scale_batch_size(self) -> int:
        """Dynamically scale batch size based on current memory usage."""
        import time
        
        current_time = time.time()
        
        # Only check every 5 seconds to avoid overhead
        if current_time - self._last_memory_check < 5:
            return self._current_batch_size
            
        self._last_memory_check = current_time
        
        try:
            # Get current memory usage
            sys_usage = self.get_system_memory_usage()
            memory_percent = sys_usage["system_percent"]
            
            # Scale up if memory usage is low
            if memory_percent < 0.6:  # Less than 60% memory usage
                self._consecutive_low_usage += 1
                if self._consecutive_low_usage >= 3:  # 3 consecutive low usage checks
                    new_batch = int(self._current_batch_size * self._scale_factor)
                    max_batch = 25000000  # Cap at 25M
                    if new_batch <= max_batch:
                        self._current_batch_size = new_batch
                        self._consecutive_low_usage = 0
                        logger.info(f"Scaled batch size up to {self._current_batch_size:,} rows (memory: {memory_percent:.1%})")
            else:
                self._consecutive_low_usage = 0
                
            # Scale down if memory usage is high
            if memory_percent > self._backoff_threshold:
                new_batch = int(self._current_batch_size / self._scale_factor)
                min_batch = 10000000  # Don't go below 10M
                if new_batch >= min_batch:
                    self._current_batch_size = new_batch
                    logger.warning(f"Scaled batch size down to {self._current_batch_size:,} rows (memory: {memory_percent:.1%})")
                    
        except Exception as e:
            logger.warning(f"Dynamic scaling failed: {e}")
            
        return self._current_batch_size

    def check_memory_with_cleanup(self, stage: str = "unknown") -> bool:
        """Back-compat alias for check_memory()"""
        return self.check_memory(stage)

    def get_system_memory_usage(self) -> Dict[str, float]:
        """Get current system memory usage."""
        memory = psutil.virtual_memory()
        process_memory = self.process.memory_info()
        
        return {
            "process_gb": process_memory.rss / (1024**3),
            "system_used_gb": memory.used / (1024**3),
            "system_total_gb": memory.total / (1024**3),
            "system_available_gb": memory.available / (1024**3),
            "system_percent": memory.percent / 100.0
        }

    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        try:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / (1024**3)
                gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "gpu_used_gb": gpu_memory,
                    "gpu_total_gb": gpu_total,
                    "gpu_percent": gpu_memory / gpu_total
                }
            else:
                return {"gpu_used_gb": 0.0, "gpu_total_gb": 0.0, "gpu_percent": 0.0}
        except Exception as e:
            logger.warning(f"Could not get GPU memory usage: {e}")
            return {"gpu_used_gb": 0.0, "gpu_total_gb": 0.0, "gpu_percent": 0.0}

    def check_memory(self, stage: str = "unknown") -> bool:
        """
        Check memory usage and apply backpressure if needed.
        
        Args:
            stage: Current training stage name
            
        Returns:
            True if memory is within limits, False if backpressure applied
        """
        try:
            # Get current usage
            sys_usage = self.get_system_memory_usage()
            gpu_usage = self.get_gpu_memory_usage()
            
            # Check system memory
            sys_warn = self.config.max_memory_gb * self.config.warning_threshold
            sys_cleanup = self.config.max_memory_gb * self.config.cleanup_threshold
            
            if sys_usage["system_used_gb"] > sys_cleanup:
                raise MemoryError(
                    f"{stage}: System memory {sys_usage['system_used_gb']:.2f}GB > cleanup limit {sys_cleanup:.2f}GB"
                )
            
            # Check GPU memory if applicable
            if gpu_usage["gpu_total_gb"] > 0:
                gpu_cleanup = self.config.max_vram_gb * self.config.gpu_cleanup_threshold
                if gpu_usage["gpu_used_gb"] > gpu_cleanup:
                    raise MemoryError(
                        f"{stage}: GPU memory {gpu_usage['gpu_used_gb']:.2f}GB > cleanup limit {gpu_cleanup:.2f}GB"
                    )
            
            # Apply backpressure if needed
            if sys_usage["system_used_gb"] > sys_warn and not self._backpressure_active:
                self._apply_backpressure()
                return False
            elif sys_usage["system_used_gb"] <= sys_warn * 0.8 and self._backpressure_active:
                self._release_backpressure()
                return True
            
            return True
            
        except MemoryError as e:
            logger.error(f"Memory limit exceeded in {stage}: {e}")
            self._apply_backpressure()
            return False

    def _apply_backpressure(self):
        """Apply memory backpressure by reducing batch size and triggering cleanup."""
        if self._backpressure_active:
            return
            
        logger.warning("Applying memory backpressure...")
        
        # Reduce batch size by half
        self._current_batch_size = max(1000, self._current_batch_size // 2)
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._backpressure_active = True
        logger.warning(f"Backpressure active: batch size reduced to {self._current_batch_size}")

    def _release_backpressure(self):
        """Release memory backpressure and restore normal batch size."""
        if not self._backpressure_active:
            return
            
        logger.info("Releasing memory backpressure...")
        
        # Gradually restore batch size
        self._current_batch_size = min(
            self._original_batch_size,
            int(self._current_batch_size * 1.5)
        )
        
        self._backpressure_active = False
        logger.info(f"Backpressure released: batch size restored to {self._current_batch_size}")

    def get_batch_size(self) -> int:
        """Get current batch size (may be reduced due to backpressure)."""
        return self._current_batch_size

    def force_cleanup(self, stage: str = "cleanup"):
        """Force memory cleanup and garbage collection."""
        logger.info(f"Forcing memory cleanup at {stage}...")
        
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory usage after cleanup
        sys_usage = self.get_system_memory_usage()
        gpu_usage = self.get_gpu_memory_usage()
        
        logger.info(f"After cleanup - System: {sys_usage['system_used_gb']:.2f}GB, "
                   f"GPU: {gpu_usage['gpu_used_gb']:.2f}GB")

    def get_xgboost_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters optimized for current memory constraints."""
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "eta": 0.05,
            "max_depth": 6,
            "min_child_weight": 30,
            "subsample": 0.7,
            "colsample_bytree": 0.5,
            "lambda": 2.5,
            "alpha": 0.1,
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 1,
        }
        
        # GPU parameters if VRAM is available
        if self.config.max_vram_gb > 0:
            try:
                if torch.cuda.is_available():
                    params.update({
                        "device": "cuda",
                        "tree_method": "hist",
                        "max_bin": 256,  # Smaller bins for memory efficiency
                    })
                    logger.info("Using GPU parameters for XGBoost")
                else:
                    logger.warning("GPU requested but CUDA not available, using CPU")
            except Exception as e:
                logger.warning(f"GPU setup failed: {e}, falling back to CPU")
        
        # Adjust parameters based on memory constraints
        if self._backpressure_active:
            params.update({
                "subsample": 0.5,  # Reduce subsample under memory pressure
                "colsample_bytree": 0.3,  # Reduce column sampling
                "max_depth": 4,  # Reduce depth
            })
            logger.info("Applied memory-constrained XGBoost parameters")
        
        return params

    def get_environment_info(self) -> Dict[str, Any]:
        """Get current environment and memory information."""
        sys_usage = self.get_system_memory_usage()
        gpu_usage = self.get_gpu_memory_usage()
        
        return {
            "system": sys_usage,
            "gpu": gpu_usage,
            "config": {
                "system_limit_gb": self.config.max_memory_gb,
                "gpu_limit_gb": self.config.max_vram_gb,
                "batch_size": self.get_batch_size(),
                "backpressure_active": self._backpressure_active,
                "headroom": self.config.headroom,
                "overhead_factor": self.config.overhead_factor
            },
            "environment": {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
                "cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }


# Global memory manager instance
_global_memory_manager: Optional[MemoryManager] = None


def init_memory_manager(config_path: Union[str, Path] = "config/memory_limits.yaml", 
                       n_features: int = 64) -> MemoryManager:
    """Initialize global memory manager instance."""
    global _global_memory_manager
    _global_memory_manager = MemoryManager(MemoryConfig.load(config_path), n_features)
    return _global_memory_manager


def get_memory_manager() -> Optional[MemoryManager]:
    """Get global memory manager instance."""
    return _global_memory_manager


def memory_check(stage: str = "unknown") -> bool:
    """Check memory with global memory manager."""
    if _global_memory_manager:
        return _global_memory_manager.check_memory(stage)
    else:
        logger.warning("No memory manager available")
        return True


def memory_cleanup(stage: str = "cleanup"):
    """Force cleanup with global memory manager."""
    if _global_memory_manager:
        _global_memory_manager.force_cleanup(stage)
    else:
        logger.warning("No memory manager available")
        gc.collect()
