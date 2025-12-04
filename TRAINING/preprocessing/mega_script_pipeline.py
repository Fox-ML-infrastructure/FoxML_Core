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
Mega Script Preprocessing Pipeline
Integrates all mega script preprocessing functionality into modular system.
"""


import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Optional, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from ..memory.memory_manager import MemoryManager

logger = logging.getLogger(__name__)

class MegaScriptPreprocessor:
    """Preprocessing pipeline that matches mega script functionality."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.memory_manager = MemoryManager(config)
        self.max_samples = self.config.get('max_samples', 3000000)  # Mega script default: 3M rows
        self.outlier_threshold = self.config.get('outlier_threshold', 5.0)  # 5-sigma rule
        self.min_data_retention = self.config.get('min_data_retention', 0.8)  # 80% retention
        
    def preprocess(self, X: np.ndarray, y: np.ndarray, 
                   timestamps: Optional[np.ndarray] = None,
                   symbols: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Apply complete mega script preprocessing pipeline."""
        
        logger.info(f"🔧 Starting mega script preprocessing on {len(X)} samples")
        
        # 1. Data capping (mega script approach)
        X, y = self._cap_data(X, y)
        
        # 2. Clean data (mega script approach)
        X_clean, y_clean = self._clean_data(X, y)
        
        # 3. Outlier removal (mega script approach)
        X_clean, y_clean = self._remove_outliers(X_clean, y_clean)
        
        # 4. Final cleanup (mega script approach)
        X_clean, y_clean = self._final_cleanup(X_clean, y_clean)
        
        # 5. Memory cleanup
        self.memory_manager.cleanup()
        
        logger.info(f"✅ Preprocessing complete: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
        return X_clean, y_clean
    
    def _cap_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Cap data to prevent memory issues (mega script approach)."""
        if len(X) <= self.max_samples:
            return X, y
        
        logger.info(f"📊 Capping data from {len(X)} to {self.max_samples} samples (mega script approach)")
        
        # Random sampling to maintain distribution
        indices = np.random.choice(len(X), self.max_samples, replace=False)
        return X[indices], y[indices]
    
    def _clean_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean data using mega script approach."""
        
        # Convert to float64 (mega script approach)
        X_float = X.astype(np.float64, copy=False)
        y_float = y.astype(np.float64, copy=False)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='median')
        X_clean = imputer.fit_transform(X_float)
        y_clean = np.nan_to_num(y_float, nan=0.0).astype(np.float32)
        
        return X_clean, y_clean
    
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove outliers using mega script approach (5-sigma rule)."""
        
        # Calculate target statistics
        target_mean = np.mean(y)
        target_std = np.std(y)
        
        # Apply 5-sigma rule (mega script approach)
        outlier_mask = np.abs(y - target_mean) <= self.outlier_threshold * target_std
        
        # Check if we have enough data after outlier removal
        if outlier_mask.sum() > len(y) * self.min_data_retention:
            X_clean = X[outlier_mask]
            y_clean = y[outlier_mask]
            removed_count = len(y) - outlier_mask.sum()
            logger.info(f"🗑️ Removed {removed_count} extreme outliers (mega script approach)")
            return X_clean, y_clean
        else:
            logger.warning(f"⚠️ Outlier removal would remove too much data, keeping original")
            return X, y
    
    def _final_cleanup(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Final data cleanup (mega script approach)."""
        
        # Replace infinite values and remaining NaNs
        X_clean = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        y_clean = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return X_clean, y_clean
    
    def preprocess_with_validation(self, X_tr: np.ndarray, y_tr: np.ndarray,
                                 X_va: Optional[np.ndarray] = None,
                                 y_va: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess training and validation data."""
        
        # Preprocess training data
        X_tr_clean, y_tr_clean = self.preprocess(X_tr, y_tr)
        
        # Preprocess validation data if provided
        if X_va is not None and y_va is not None:
            X_va_clean, y_va_clean = self.preprocess(X_va, y_va)
            return X_tr_clean, y_tr_clean, X_va_clean, y_va_clean
        else:
            return X_tr_clean, y_tr_clean, None, None
    
    def get_preprocessing_stats(self, X_original: np.ndarray, y_original: np.ndarray,
                               X_processed: np.ndarray, y_processed: np.ndarray) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        
        return {
            'original_samples': len(X_original),
            'processed_samples': len(X_processed),
            'samples_removed': len(X_original) - len(X_processed),
            'retention_rate': len(X_processed) / len(X_original),
            'original_features': X_original.shape[1],
            'processed_features': X_processed.shape[1],
            'memory_reduction': (len(X_original) - len(X_processed)) / len(X_original)
        }
