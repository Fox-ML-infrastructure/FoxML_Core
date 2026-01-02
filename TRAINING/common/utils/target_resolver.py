# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Target Resolver

Resolves target names to actual columns and handles target extraction.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class TargetResolver:
    """Resolves target names and extracts target data"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.target_mappings = {}
        
    def resolve_targets(self, df: pd.DataFrame, targets: List[str]) -> Dict[str, str]:
        """Resolve target names to actual column names"""
        
        resolved = {}
        
        for target in targets:
            actual_col = self._resolve_single_target(df, target)
            if actual_col:
                resolved[target] = actual_col
                self.target_mappings[target] = actual_col
            else:
                logger.warning(f"Could not resolve target: {target}")
        
        return resolved
    
    def _resolve_single_target(self, df: pd.DataFrame, target: str) -> Optional[str]:
        """Resolve a single target name to column name"""
        
        # Exact match first
        if target in df.columns:
            return target
        
        # Pattern matching for common target types
        patterns = self._get_target_patterns(target)
        
        for pattern in patterns:
            matches = [col for col in df.columns if col.startswith(pattern)]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                logger.warning(f"Multiple matches for {target}: {matches}")
                # Return the shortest match (most specific)
                return min(matches, key=len)
        
        return None
    
    def _get_target_patterns(self, target: str) -> List[str]:
        """Get search patterns for a target name"""
        
        # Direct patterns
        patterns = [target]
        
        # Common target type patterns
        if target.startswith('fwd_ret_'):
            # Forward return patterns
            base = target.replace('fwd_ret_', '')
            patterns.extend([
                f'fwd_ret_{base}',
                f'forward_return_{base}',
                f'ret_{base}'
            ])
        elif target.startswith('will_peak'):
            # Peak patterns
            patterns.extend([
                'will_peak',
                'peak_prob',
                'peak_indicator'
            ])
        elif target.startswith('will_valley'):
            # Valley patterns
            patterns.extend([
                'will_valley',
                'valley_prob',
                'valley_indicator'
            ])
        elif target.startswith('mdd_'):
            # MDD patterns
            patterns.extend([
                'mdd_',
                'max_drawdown',
                'drawdown'
            ])
        elif target.startswith('mfe_'):
            # MFE patterns
            patterns.extend([
                'mfe_',
                'max_favorable',
                'favorable_excursion'
            ])
        
        return patterns
    
    def extract_targets(self, df: pd.DataFrame, targets: List[str]) -> Dict[str, np.ndarray]:
        """Extract target data from DataFrame"""
        
        targets = {}
        
        for target in targets:
            try:
                # Get actual column name
                actual_col = self.target_mappings.get(target, target)
                
                if actual_col not in df.columns:
                    logger.error(f"Target column {actual_col} not found in DataFrame")
                    continue
                
                # Extract target data
                target_data = df[actual_col].values
                
                # Convert to appropriate type
                if target.startswith('fwd_ret_'):
                    # Forward returns should be float
                    target_data = pd.to_numeric(target_data, errors='coerce').astype(np.float32)
                elif any(target.startswith(prefix) for prefix in 
                        ['will_peak', 'will_valley', 'mdd', 'mfe', 'y_will_']):
                    # Barrier targets should be binary
                    target_data = pd.to_numeric(target_data, errors='coerce').astype(np.float32)
                    # Ensure binary values
                    target_data = np.where(target_data > 0.5, 1.0, 0.0)
                
                targets[target] = target_data
                
            except Exception as e:
                logger.error(f"Error extracting target {target}: {e}")
                continue
        
        return targets
    
    def validate_targets(self, df: pd.DataFrame, targets: List[str]) -> Dict[str, Any]:
        """Validate target data quality"""
        
        validation_results = {}
        
        for target in targets:
            try:
                actual_col = self.target_mappings.get(target, target)
                
                if actual_col not in df.columns:
                    validation_results[target] = {
                        'valid': False,
                        'error': f"Column {actual_col} not found"
                    }
                    continue
                
                target_data = df[actual_col].values
                
                # Check for NaN values
                n_nan = np.sum(np.isnan(target_data))
                n_total = len(target_data)
                nan_ratio = n_nan / n_total if n_total > 0 else 0
                
                # Check for constant values
                unique_values = np.unique(target_data[~np.isnan(target_data)])
                is_constant = len(unique_values) <= 1
                
                # Check data range
                if not np.isnan(target_data).all():
                    data_min = np.nanmin(target_data)
                    data_max = np.nanmax(target_data)
                else:
                    data_min = data_max = np.nan
                
                validation_results[target] = {
                    'valid': nan_ratio < 0.5 and not is_constant,
                    'n_samples': n_total,
                    'n_nan': n_nan,
                    'nan_ratio': nan_ratio,
                    'is_constant': is_constant,
                    'data_range': (data_min, data_max),
                    'n_unique': len(unique_values)
                }
                
            except Exception as e:
                validation_results[target] = {
                    'valid': False,
                    'error': str(e)
                }
        
        return validation_results
    
    def get_target_info(self, targets: List[str]) -> Dict[str, Any]:
        """Get information about targets"""
        
        info = {}
        
        for target in targets:
            target_info = {
                'name': target,
                'type': self._classify_target_type(target),
                'mapped_column': self.target_mappings.get(target, target)
            }
            
            # Add specific info based on target type
            if target.startswith('fwd_ret_'):
                target_info.update({
                    'description': 'Forward return target',
                    'expected_range': '(-inf, +inf)',
                    'data_type': 'continuous'
                })
            elif target.startswith('will_peak'):
                target_info.update({
                    'description': 'Peak probability target',
                    'expected_range': '[0, 1]',
                    'data_type': 'binary'
                })
            elif target.startswith('will_valley'):
                target_info.update({
                    'description': 'Valley probability target',
                    'expected_range': '[0, 1]',
                    'data_type': 'binary'
                })
            elif target.startswith('mdd_'):
                target_info.update({
                    'description': 'Maximum drawdown target',
                    'expected_range': '[0, 1]',
                    'data_type': 'continuous'
                })
            elif target.startswith('mfe_'):
                target_info.update({
                    'description': 'Maximum favorable excursion target',
                    'expected_range': '[0, 1]',
                    'data_type': 'continuous'
                })
            
            info[target] = target_info
        
        return info
    
    def _classify_target_type(self, target: str) -> str:
        """Classify target type based on name"""
        
        if target.startswith('fwd_ret_'):
            return 'regression'
        elif any(target.startswith(prefix) for prefix in 
                ['will_peak', 'will_valley']):
            return 'classification'
        elif any(target.startswith(prefix) for prefix in 
                ['mdd_', 'mfe_']):
            return 'regression'  # Can be treated as regression for continuous values
        else:
            return 'unknown'
    
    def create_target_summary(self, df: pd.DataFrame, targets: List[str]) -> Dict[str, Any]:
        """Create summary of target data"""
        
        summary = {
            'targets': {},
            'overall_stats': {
                'n_targets': len(targets),
                'n_samples': len(df),
                'n_features': len(df.columns)
            }
        }
        
        for target in targets:
            try:
                actual_col = self.target_mappings.get(target, target)
                
                if actual_col not in df.columns:
                    summary['targets'][target] = {
                        'available': False,
                        'error': f"Column {actual_col} not found"
                    }
                    continue
                
                target_data = df[actual_col].values
                
                # Basic statistics
                n_nan = np.sum(np.isnan(target_data))
                valid_data = target_data[~np.isnan(target_data)]
                
                if len(valid_data) > 0:
                    stats = {
                        'mean': float(np.mean(valid_data)),
                        'std': float(np.std(valid_data)),
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'median': float(np.median(valid_data))
                    }
                else:
                    stats = {
                        'mean': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'median': np.nan
                    }
                
                summary['targets'][target] = {
                    'available': True,
                    'column': actual_col,
                    'n_valid': len(valid_data),
                    'n_nan': n_nan,
                    'nan_ratio': n_nan / len(target_data),
                    'stats': stats,
                    'type': self._classify_target_type(target)
                }
                
            except Exception as e:
                summary['targets'][target] = {
                    'available': False,
                    'error': str(e)
                }
        
        return summary
