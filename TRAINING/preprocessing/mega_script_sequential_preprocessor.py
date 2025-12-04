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
Mega Script Sequential Preprocessor - Handles both cross-sectional and sequential models.
This preprocesses data once, then creates appropriate formats for different model types.
"""


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, Any, Optional, Tuple, List
import gc
import psutil

logger = logging.getLogger(__name__)

class MegaScriptSequentialPreprocessor:
    """One-time preprocessing for both cross-sectional and sequential models like mega script."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.max_samples = self.config.get('max_samples', 3000000)
        self.outlier_threshold = self.config.get('outlier_threshold', 5.0)
        self.min_data_retention = self.config.get('min_data_retention', 0.8)
        self.sequence_length = self.config.get('sequence_length', 64)
        
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
            
            if memory_gb > self.memory_limit_gb * 0.9:
                logger.warning(f"⚠️ High memory usage: {memory_gb:.1f} GB (limit: {self.memory_limit_gb} GB)")
                return True
        except Exception:
            pass
        return False
        
    def preprocess_data_once(self, X: np.ndarray, y: np.ndarray, 
                           timestamps: Optional[np.ndarray] = None,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        MEGA SCRIPT APPROACH: Preprocess data once, create formats for all model types.
        Returns both cross-sectional and sequential data formats.
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
        
        # 5. Create cross-sectional format (for RewardBased, Ensemble, etc.)
        X_cs = X_final.astype(np.float32)
        y_cs = y_final.astype(np.float32)
        
        # 6. Create sequential format (for LSTM, CNN1D, Transformer)
        X_seq, y_seq = self._create_sequential_format(X_final, y_final, timestamps)
        
        # 7. Memory cleanup
        del X_float, y_float, X_imputed, y_imputed
        gc.collect()
        
        # 8. Store preprocessing state
        preprocessing_state = {
            'imputer': self.imputer,
            'scaler': self.scaler,
            'feature_names': feature_names,
            'outlier_threshold': self.outlier_threshold,
            'min_data_retention': self.min_data_retention,
            'sequence_length': self.sequence_length
        }
        
        self.log_memory("preprocessing_complete")
        logger.info(f"✅ MEGA SCRIPT preprocessing complete:")
        logger.info(f"   📊 Cross-sectional: {X_cs.shape[0]} samples, {X_cs.shape[1]} features")
        logger.info(f"   📈 Sequential: {X_seq.shape[0]} sequences, {X_seq.shape[1]} timesteps, {X_seq.shape[2]} features")
        
        return {
            'cross_sectional': {
                'X': X_cs,
                'y': y_cs,
                'timestamps': timestamps
            },
            'sequential': {
                'X': X_seq,
                'y': y_seq,
                'timestamps': timestamps
            },
            'preprocessing_state': preprocessing_state
        }
    
    def _create_sequential_format(self, X: np.ndarray, y: np.ndarray, 
                                 timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequential format for LSTM/CNN1D/Transformer models.
        This matches the mega script's prepare_sequence_cs function.
        """
        logger.info(f"🔄 Creating sequential format with lookback={self.sequence_length}")
        
        # Sort by timestamps if available
        if timestamps is not None:
            sort_indices = np.argsort(timestamps)
            X_sorted = X[sort_indices]
            y_sorted = y[sort_indices]
            ts_sorted = timestamps[sort_indices]
        else:
            X_sorted = X
            y_sorted = y
            ts_sorted = np.arange(len(X))
        
        # Create sequences
        sequences = []
        targets = []
        
        for i in range(self.sequence_length, len(X_sorted)):
            # Extract sequence
            seq = X_sorted[i-self.sequence_length:i]
            target = y_sorted[i]
            
            sequences.append(seq)
            targets.append(target)
        
        if len(sequences) == 0:
            logger.warning("No sequences created - data too short")
            return np.array([]).reshape(0, self.sequence_length, X.shape[1]), np.array([])
        
        X_seq = np.array(sequences, dtype=np.float32)
        y_seq = np.array(targets, dtype=np.float32)
        
        logger.info(f"📈 Created {len(sequences)} sequences of length {self.sequence_length}")
        
        return X_seq, y_seq
    
    def get_cross_sectional_data(self, processed_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Get cross-sectional data for RewardBased, Ensemble, etc."""
        cs_data = processed_data['cross_sectional']
        return cs_data['X'], cs_data['y']
    
    def get_sequential_data(self, processed_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Get sequential data for LSTM, CNN1D, Transformer."""
        seq_data = processed_data['sequential']
        return seq_data['X'], seq_data['y']
    
    def apply_light_cleanup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Light cleanup for already preprocessed data (mega script approach)."""
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
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
