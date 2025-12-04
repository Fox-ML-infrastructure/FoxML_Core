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
Mega Script Data Preprocessor - One-time preprocessing like mega script.
This preprocesses data once at the data level, then passes clean arrays to all models.
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Optional, Tuple, List
import gc
import psutil
import os

logger = logging.getLogger(__name__)

class MegaScriptDataPreprocessor:
    """One-time data preprocessing like mega script - preprocess once, use everywhere."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_samples = self.config.get('max_samples', 3000000)  # Mega script default: 3M rows
        self.outlier_threshold = self.config.get('outlier_threshold', 5.0)  # 5-sigma rule
        self.min_data_retention = self.config.get('min_data_retention', 0.8)  # 80% retention
        
        # Initialize preprocessing components
        self.imputer = SimpleImputer(strategy='median')
        self.scaler = StandardScaler()
        
        # Memory management
        self.memory_limit_gb = self.config.get('memory_limit_gb', 100)
        
    def log_memory(self, stage: str):
        """Log memory usage like mega script."""
        try:
            process = psutil.Process()
            memory_gb = process.memory_info().rss / 1024**3
            logger.info(f"💾 Memory at {stage}: {memory_gb:.1f} GB")
            
            # Check if we're approaching memory limits
            if memory_gb > self.memory_limit_gb * 0.9:
                logger.warning(f"⚠️ High memory usage: {memory_gb:.1f} GB (limit: {self.memory_limit_gb} GB)")
                return True
        except Exception:
            pass
        return False
        
    def preprocess_data_once(self, X: np.ndarray, y: np.ndarray, 
                           timestamps: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        MEGA SCRIPT APPROACH: Preprocess data once, then reuse clean arrays.
        This matches the mega script's one-time preprocessing approach.
        """
        logger.info(f"🔧 Starting MEGA SCRIPT one-time preprocessing on {len(X)} samples")
        self.log_memory("preprocessing_start")
        
        # 1. Data capping (like mega script)
        if len(X) > self.max_samples:
            logger.info(f"📊 Capping data from {len(X)} to {self.max_samples} samples (mega script approach)")
            indices = np.random.choice(len(X), self.max_samples, replace=False)
            X = X[indices]
            y = y[indices]
            if timestamps is not None:
                timestamps = timestamps[indices]
        
        # 2. Clean data (same as mega script)
        X_float = X.astype(np.float64, copy=False)
        y_float = y.astype(np.float64, copy=False)
        
        # Fit and transform imputer
        X_imputed = self.imputer.fit_transform(X_float)
        y_imputed = np.nan_to_num(y_float, nan=0.0).astype(np.float32)
        
        # 3. Outlier removal (mega script approach)
        target_mean = np.mean(y_imputed)
        target_std = np.std(y_imputed)
        outlier_mask = np.abs(y_imputed - target_mean) <= self.outlier_threshold * target_std
        
        if outlier_mask.sum() > len(y_imputed) * self.min_data_retention:
            X_imputed = X_imputed[outlier_mask]
            y_imputed = y_imputed[outlier_mask]
            if timestamps is not None:
                timestamps = timestamps[outlier_mask]
            logger.info(f"🗑️ Removed {len(y_imputed) - outlier_mask.sum()} extreme outliers (mega script approach)")
        
        # 4. Final cleanup (mega script approach)
        X_final = np.nan_to_num(X_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
        y_final = np.nan_to_num(y_imputed, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 5. Memory cleanup (like mega script)
        del X_float, y_float, X_imputed, y_imputed
        gc.collect()
        
        # 6. Store preprocessing state for reuse
        preprocessing_state = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'outlier_threshold': self.outlier_threshold,
            'min_data_retention': self.min_data_retention
        }
        
        self.log_memory("preprocessing_complete")
        logger.info(f"✅ MEGA SCRIPT preprocessing complete: {X_final.shape[0]} samples, {X_final.shape[1]} features")
        
        return X_final, y_final, preprocessing_state
    
    def apply_light_cleanup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Light cleanup for already preprocessed data (mega script approach).
        This is what individual trainers should use - no heavy preprocessing.
        """
        # Minimal cleanup only (like mega script)
        X_clean = np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y.astype(np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_clean, y_clean
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024**3
        except Exception:
            return 0.0
    
    def cleanup_memory(self):
        """Aggressive memory cleanup like mega script."""
        gc.collect()
        try:
            # Force garbage collection
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
