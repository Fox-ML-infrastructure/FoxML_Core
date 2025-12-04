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
Data Preprocessor

Handles data preprocessing for different training strategies.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """Data preprocessing utilities for model training"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.feature_names = []
        self.target_names = []
        
    def prepare_training_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                             feature_names: List[str], strategy: str = 'single_task') -> Dict[str, Any]:
        """Prepare data for training based on strategy"""
        
        self.feature_names = feature_names
        self.target_names = list(y_dict.keys())
        
        # Validate inputs
        self._validate_inputs(X, y_dict, feature_names)
        
        # Clean data
        X_clean, y_clean = self._clean_data(X, y_dict)
        
        # Prepare based on strategy
        if strategy == 'single_task':
            return self._prepare_single_task_data(X_clean, y_clean)
        elif strategy == 'multi_task':
            return self._prepare_multi_task_data(X_clean, y_clean)
        elif strategy == 'cascade':
            return self._prepare_cascade_data(X_clean, y_clean)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _validate_inputs(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                        feature_names: List[str]):
        """Validate input data"""
        
        if X.shape[0] == 0:
            raise ValueError("X is empty")
        
        if len(feature_names) != X.shape[1]:
            raise ValueError(f"Feature names length ({len(feature_names)}) doesn't match X columns ({X.shape[1]})")
        
        if not y_dict:
            raise ValueError("y_dict is empty")
        
        # Check all targets have same length
        target_lengths = [len(y) for y in y_dict.values()]
        if len(set(target_lengths)) > 1:
            raise ValueError("All targets must have same length")
        
        if target_lengths[0] != X.shape[0]:
            raise ValueError("X and y dimensions don't match")
        
        logger.info(f"Data validation passed: {X.shape[0]} samples, {X.shape[1]} features, {len(y_dict)} targets")
    
    def _clean_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Clean data by removing NaN values and outliers"""
        
        # Find valid samples (no NaN in X or any y)
        valid_mask = ~np.isnan(X).any(axis=1)
        
        for target_name, y in y_dict.items():
            target_valid = ~np.isnan(y)
            valid_mask = valid_mask & target_valid
        
        # Apply mask
        X_clean = X[valid_mask]
        y_clean = {name: y[valid_mask] for name, y in y_dict.items()}
        
        n_removed = len(X) - len(X_clean)
        if n_removed > 0:
            logger.warning(f"Removed {n_removed} samples with NaN values")
        
        # Remove outliers if configured
        if self.config.get('remove_outliers', False):
            X_clean, y_clean = self._remove_outliers(X_clean, y_clean)
        
        logger.info(f"Cleaned data: {len(X_clean)} samples remaining")
        return X_clean, y_clean
    
    def _remove_outliers(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Remove outliers using IQR method"""
        
        # Calculate IQR for each feature
        Q1 = np.percentile(X, 25, axis=0)
        Q3 = np.percentile(X, 75, axis=0)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Find valid samples
        valid_mask = np.all((X >= lower_bound) & (X <= upper_bound), axis=1)
        
        X_clean = X[valid_mask]
        y_clean = {name: y[valid_mask] for name, y in y_dict.items()}
        
        n_removed = len(X) - len(X_clean)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} outliers")
        
        return X_clean, y_clean
    
    def _prepare_single_task_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare data for single-task training"""
        
        return {
            'X': X,
            'y_dict': y_dict,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'strategy': 'single_task'
        }
    
    def _prepare_multi_task_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare data for multi-task training"""
        
        # Ensure all targets have same length
        target_lengths = [len(y) for y in y_dict.values()]
        if len(set(target_lengths)) > 1:
            raise ValueError("All targets must have same length for multi-task learning")
        
        return {
            'X': X,
            'y_dict': y_dict,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'strategy': 'multi_task',
            'n_targets': len(y_dict)
        }
    
    def _prepare_cascade_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Prepare data for cascade training"""
        
        # Separate targets by type
        barrier_targets = []
        fwd_ret_targets = []
        
        for target_name, y in y_dict.items():
            if target_name.startswith('fwd_ret_'):
                fwd_ret_targets.append(target_name)
            elif any(target_name.startswith(prefix) for prefix in 
                    ['will_peak', 'will_valley', 'mdd', 'mfe', 'y_will_']):
                barrier_targets.append(target_name)
            else:
                # Default to regression for unknown targets
                fwd_ret_targets.append(target_name)
        
        return {
            'X': X,
            'y_dict': y_dict,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'strategy': 'cascade',
            'barrier_targets': barrier_targets,
            'fwd_ret_targets': fwd_ret_targets
        }
    
    def get_data_summary(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Get summary statistics of the data"""
        
        summary = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_targets': len(y_dict),
            'feature_stats': {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0),
                'min': np.min(X, axis=0),
                'max': np.max(X, axis=0)
            },
            'target_stats': {}
        }
        
        # Target statistics
        for target_name, y in y_dict.items():
            summary['target_stats'][target_name] = {
                'mean': float(np.mean(y)),
                'std': float(np.std(y)),
                'min': float(np.min(y)),
                'max': float(np.max(y)),
                'n_unique': len(np.unique(y)),
                'n_nan': int(np.sum(np.isnan(y)))
            }
        
        return summary
    
    def create_train_test_split(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                               test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """Create train/test split"""
        
        from sklearn.model_selection import train_test_split
        
        # Split X
        X_train, X_test, indices_train, indices_test = train_test_split(
            X, np.arange(len(X)), test_size=test_size, random_state=random_state
        )
        
        # Split y_dict
        y_train = {}
        y_test = {}
        
        for target_name, y in y_dict.items():
            y_train[target_name] = y[indices_train]
            y_test[target_name] = y[indices_test]
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'indices_train': indices_train,
            'indices_test': indices_test
        }
