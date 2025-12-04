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
Validation Utilities

Validation functions for model training and data quality.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ValidationUtils:
    """Validation utilities for model training"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
    def validate_training_data(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                             feature_names: List[str]) -> Dict[str, Any]:
        """Validate training data quality"""
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check X
        x_validation = self._validate_features(X, feature_names)
        validation_results['stats']['features'] = x_validation
        
        if not x_validation['valid']:
            validation_results['valid'] = False
            validation_results['errors'].extend(x_validation['errors'])
        
        # Check y_dict
        for target_name, y in y_dict.items():
            y_validation = self._validate_target(y, target_name)
            validation_results['stats'][target_name] = y_validation
            
            if not y_validation['valid']:
                validation_results['valid'] = False
                validation_results['errors'].extend(y_validation['errors'])
            else:
                validation_results['warnings'].extend(y_validation['warnings'])
        
        return validation_results
    
    def _validate_features(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Validate feature matrix"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check dimensions
        if X.shape[0] == 0:
            validation['valid'] = False
            validation['errors'].append("X is empty")
            return validation
        
        if X.shape[1] != len(feature_names):
            validation['valid'] = False
            validation['errors'].append(f"Feature count mismatch: X has {X.shape[1]} columns, {len(feature_names)} names")
            return validation
        
        # Check for NaN values
        n_nan = np.sum(np.isnan(X))
        n_total = X.size
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        validation['stats']['n_samples'] = X.shape[0]
        validation['stats']['n_features'] = X.shape[1]
        validation['stats']['n_nan'] = n_nan
        validation['stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > 0.1:  # More than 10% NaN
            validation['warnings'].append(f"High NaN ratio: {nan_ratio:.2%}")
        
        # Check for constant features
        constant_features = []
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            valid_data = feature_data[~np.isnan(feature_data)]
            if len(valid_data) > 0 and len(np.unique(valid_data)) <= 1:
                constant_features.append(feature_names[i])
        
        if constant_features:
            validation['warnings'].append(f"Constant features: {constant_features}")
        
        # Check for infinite values
        n_inf = np.sum(np.isinf(X))
        if n_inf > 0:
            validation['warnings'].append(f"Infinite values found: {n_inf}")
        
        return validation
    
    def _validate_target(self, y: np.ndarray, target_name: str) -> Dict[str, Any]:
        """Validate target data"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for empty target
        if len(y) == 0:
            validation['valid'] = False
            validation['errors'].append(f"Target {target_name} is empty")
            return validation
        
        # Check for NaN values
        n_nan = np.sum(np.isnan(y))
        n_total = len(y)
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        validation['stats']['n_samples'] = n_total
        validation['stats']['n_nan'] = n_nan
        validation['stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > 0.5:  # More than 50% NaN
            validation['valid'] = False
            validation['errors'].append(f"Target {target_name} has too many NaN values: {nan_ratio:.2%}")
        elif nan_ratio > 0.1:  # More than 10% NaN
            validation['warnings'].append(f"Target {target_name} has high NaN ratio: {nan_ratio:.2%}")
        
        # Check for constant values
        valid_data = y[~np.isnan(y)]
        if len(valid_data) > 0:
            n_unique = len(np.unique(valid_data))
            validation['stats']['n_unique'] = n_unique
            
            if n_unique <= 1:
                validation['valid'] = False
                validation['errors'].append(f"Target {target_name} is constant")
            elif n_unique <= 2:
                validation['warnings'].append(f"Target {target_name} has only {n_unique} unique values")
            
            # Check data range
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            validation['stats']['range'] = (data_min, data_max)
            
            # Check for infinite values
            n_inf = np.sum(np.isinf(valid_data))
            if n_inf > 0:
                validation['warnings'].append(f"Target {target_name} has infinite values: {n_inf}")
            
            # Check for extreme values
            if target_name.startswith('fwd_ret_'):
                # Forward returns should be reasonable
                if abs(data_max) > 1.0:  # More than 100% return
                    validation['warnings'].append(f"Target {target_name} has extreme values: max={data_max:.3f}")
        else:
            validation['valid'] = False
            validation['errors'].append(f"Target {target_name} has no valid data")
        
        return validation
    
    def validate_model_predictions(self, predictions: Dict[str, np.ndarray], 
                                 expected_targets: List[str]) -> Dict[str, Any]:
        """Validate model predictions"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check if all expected targets are present
        missing_targets = set(expected_targets) - set(predictions.keys())
        if missing_targets:
            validation['valid'] = False
            validation['errors'].append(f"Missing predictions for targets: {missing_targets}")
        
        # Check each prediction
        for target_name, pred in predictions.items():
            pred_validation = self._validate_prediction(pred, target_name)
            validation['stats'][target_name] = pred_validation
            
            if not pred_validation['valid']:
                validation['valid'] = False
                validation['errors'].extend(pred_validation['errors'])
            else:
                validation['warnings'].extend(pred_validation['warnings'])
        
        return validation
    
    def _validate_prediction(self, pred: np.ndarray, target_name: str) -> Dict[str, Any]:
        """Validate a single prediction"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check for NaN values
        n_nan = np.sum(np.isnan(pred))
        n_total = len(pred)
        nan_ratio = n_nan / n_total if n_total > 0 else 0
        
        validation['stats']['n_samples'] = n_total
        validation['stats']['n_nan'] = n_nan
        validation['stats']['nan_ratio'] = nan_ratio
        
        if nan_ratio > 0.1:  # More than 10% NaN
            validation['warnings'].append(f"Prediction {target_name} has high NaN ratio: {nan_ratio:.2%}")
        
        # Check for infinite values
        n_inf = np.sum(np.isinf(pred))
        if n_inf > 0:
            validation['warnings'].append(f"Prediction {target_name} has infinite values: {n_inf}")
        
        # Check data range
        valid_data = pred[~np.isnan(pred) & ~np.isinf(pred)]
        if len(valid_data) > 0:
            data_min = np.min(valid_data)
            data_max = np.max(valid_data)
            validation['stats']['range'] = (data_min, data_max)
            
            # Check for reasonable ranges based on target type
            if target_name.startswith('fwd_ret_'):
                # Forward returns should be reasonable
                if abs(data_max) > 2.0:  # More than 200% return
                    validation['warnings'].append(f"Prediction {target_name} has extreme values: max={data_max:.3f}")
            elif any(target_name.startswith(prefix) for prefix in 
                    ['will_peak', 'will_valley', 'mdd', 'mfe']):
                # Probability-like targets should be in [0, 1]
                if data_min < 0 or data_max > 1:
                    validation['warnings'].append(f"Prediction {target_name} outside [0,1] range: [{data_min:.3f}, {data_max:.3f}]")
        
        return validation
    
    def validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration"""
        
        validation = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        # Check required fields
        required_fields = ['strategy', 'targets']
        for field in required_fields:
            if field not in config:
                validation['valid'] = False
                validation['errors'].append(f"Missing required field: {field}")
        
        # Check strategy
        if 'strategy' in config:
            valid_strategies = ['single_task', 'multi_task', 'cascade']
            if config['strategy'] not in valid_strategies:
                validation['valid'] = False
                validation['errors'].append(f"Invalid strategy: {config['strategy']}. Must be one of {valid_strategies}")
        
        # Check targets
        if 'targets' in config:
            if not isinstance(config['targets'], list) or len(config['targets']) == 0:
                validation['valid'] = False
                validation['errors'].append("Targets must be a non-empty list")
        
        # Check model configuration
        if 'models' in config:
            model_config = config['models']
            if not isinstance(model_config, dict):
                validation['warnings'].append("Model configuration should be a dictionary")
        
        return validation
    
    def create_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Create a human-readable validation report"""
        
        report = []
        report.append("=== Validation Report ===")
        
        if validation_results['valid']:
            report.append("✅ Validation PASSED")
        else:
            report.append("❌ Validation FAILED")
        
        if validation_results['errors']:
            report.append("\n🚨 Errors:")
            for error in validation_results['errors']:
                report.append(f"  - {error}")
        
        if validation_results['warnings']:
            report.append("\n⚠️  Warnings:")
            for warning in validation_results['warnings']:
                report.append(f"  - {warning}")
        
        # Add statistics
        if 'stats' in validation_results:
            report.append("\n📊 Statistics:")
            for key, value in validation_results['stats'].items():
                if isinstance(value, dict):
                    report.append(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        report.append(f"    {sub_key}: {sub_value}")
                else:
                    report.append(f"  {key}: {value}")
        
        return "\n".join(report)
