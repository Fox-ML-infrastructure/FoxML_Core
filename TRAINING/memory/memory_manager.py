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
Memory Management System - Mega Script Integration
Handles memory optimization, monitoring, and cleanup for large-scale training.
"""


import gc
import psutil
import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management system for large-scale training."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory_threshold = self.config.get('memory_threshold', 0.8)  # 80% memory usage
        self.chunk_size = self.config.get('chunk_size', 1000000)  # 1M rows per chunk
        self.aggressive_cleanup = self.config.get('aggressive_cleanup', True)
        self.initial_memory = self.get_memory_usage()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage information."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # Get system memory
            system_memory = psutil.virtual_memory()
            
            return {
                'rss_gb': memory_info.rss / 1024**3,  # Resident Set Size
                'vms_gb': memory_info.vms / 1024**3,  # Virtual Memory Size
                'system_total_gb': system_memory.total / 1024**3,
                'system_available_gb': system_memory.available / 1024**3,
                'system_percent': system_memory.percent
            }
        except Exception as e:
            logger.warning(f"Memory monitoring failed: {e}")
            return {'rss_gb': 0, 'vms_gb': 0, 'system_total_gb': 0, 'system_available_gb': 0, 'system_percent': 0}
    
    def log_memory_usage(self, stage: str = "Unknown") -> None:
        """Log current memory usage."""
        memory_info = self.get_memory_usage()
        logger.info(f"💾 Memory at {stage}: RSS={memory_info['rss_gb']:.1f}GB, "
                   f"System={memory_info['system_percent']:.1f}%")
        
        # Warn if memory usage is high
        if memory_info['system_percent'] > 90:
            logger.warning(f"⚠️ High system memory usage: {memory_info['system_percent']:.1f}%")
        if memory_info['rss_gb'] > 50:  # 50GB threshold
            logger.warning(f"⚠️ High process memory usage: {memory_info['rss_gb']:.1f}GB")
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        memory_info = self.get_memory_usage()
        return (memory_info['system_percent'] > self.memory_threshold * 100 or 
                memory_info['rss_gb'] > 50)  # 50GB threshold
    
    def cleanup(self, aggressive: bool = None) -> None:
        """Perform memory cleanup."""
        if aggressive is None:
            aggressive = self.aggressive_cleanup
            
        logger.info("🧹 Performing memory cleanup...")
        
        # Standard cleanup
        gc.collect()
        
        if aggressive:
            # Aggressive cleanup (mega script approach)
            self._aggressive_cleanup()
        
        # Log memory after cleanup
        self.log_memory_usage("after cleanup")
    
    def _aggressive_cleanup(self) -> None:
        """Perform aggressive memory cleanup."""
        try:
            # Clear TensorFlow sessions
            try:
                import tensorflow as tf
                tf.keras.backend.clear_session()
                tf.compat.v1.reset_default_graph()
            except Exception:
                pass
            
            # Clear PyTorch cache if available
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            
            # Force garbage collection multiple times
            for _ in range(3):
                gc.collect()
                
        except Exception as e:
            logger.warning(f"Aggressive cleanup failed: {e}")
    
    def cap_data(self, X: np.ndarray, y: np.ndarray, max_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cap data to prevent memory issues (mega script approach)."""
        if len(X) <= max_samples:
            return X, y
        
        logger.info(f"📊 Capping data from {len(X)} to {max_samples} samples")
        
        # Random sampling to maintain distribution
        indices = np.random.choice(len(X), max_samples, replace=False)
        return X[indices], y[indices]
    
    def chunk_data(self, X: np.ndarray, y: np.ndarray, chunk_size: int = None) -> list:
        """Split data into chunks for memory-efficient processing."""
        if chunk_size is None:
            chunk_size = self.chunk_size
        
        if len(X) <= chunk_size:
            return [(X, y)]
        
        logger.info(f"📦 Splitting data into chunks of {chunk_size} samples")
        
        chunks = []
        for i in range(0, len(X), chunk_size):
            end_idx = min(i + chunk_size, len(X))
            chunks.append((X[i:end_idx], y[i:end_idx]))
        
        return chunks
    
    def monitor_training(self, stage: str) -> None:
        """Monitor memory during training stages."""
        self.log_memory_usage(stage)
        
        if self.should_cleanup():
            logger.warning(f"⚠️ High memory usage at {stage}, performing cleanup")
            self.cleanup()
    
    def get_memory_recommendations(self) -> Dict[str, Any]:
        """Get memory usage recommendations."""
        memory_info = self.get_memory_usage()
        
        recommendations = {
            'current_usage_gb': memory_info['rss_gb'],
            'system_usage_percent': memory_info['system_percent'],
            'recommendations': []
        }
        
        if memory_info['system_percent'] > 90:
            recommendations['recommendations'].append("Consider reducing batch size or using chunked processing")
        
        if memory_info['rss_gb'] > 50:
            recommendations['recommendations'].append("Consider using data capping or more aggressive cleanup")
        
        if memory_info['system_available_gb'] < 10:
            recommendations['recommendations'].append("System memory is low, consider closing other applications")
        
        return recommendations
