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
Cascade Strategy

Option C: Train separate models + apply gating/stacking logic.
Best for: When you want to combine barrier signals with return predictions.
"""


import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from .base import BaseTrainingStrategy

logger = logging.getLogger(__name__)

class CascadeStrategy(BaseTrainingStrategy):
    """Cascade training with separate models + gating logic"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.barrier_models = {}
        self.fwd_ret_models = {}
        self.calibrators = {}
        self.target_types = {}
        
    def train(self, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
              feature_names: List[str], **kwargs) -> Dict[str, Any]:
        """Train cascade models"""
        logger.info("🔗 Training cascade models (Option C)")
        
        self.validate_data(X, y_dict)
        
        # Separate targets by type
        barrier_targets = []
        fwd_ret_targets = []
        
        for target_name, y in y_dict.items():
            target_type = self._determine_target_type(target_name, y)
            self.target_types[target_name] = target_type
            
            if target_type == 'classification':
                barrier_targets.append(target_name)
            else:
                fwd_ret_targets.append(target_name)
        
        logger.info(f"Barrier targets: {barrier_targets}")
        logger.info(f"Forward return targets: {fwd_ret_targets}")
        
        # Train barrier models (classifiers)
        for target_name in barrier_targets:
            logger.info(f"Training barrier model: {target_name}")
            model = self._create_classification_model(target_name)
            model.fit(X, y_dict[target_name])
            
            # Calibrate if needed
            calibrator = self._create_calibrator(model, X, y_dict[target_name])
            
            self.barrier_models[target_name] = model
            if calibrator:
                self.calibrators[target_name] = calibrator
        
        # Train forward return models (regressors)
        for target_name in fwd_ret_targets:
            logger.info(f"Training fwd_ret model: {target_name}")
            model = self._create_regression_model(target_name)
            model.fit(X, y_dict[target_name])
            self.fwd_ret_models[target_name] = model
        
        # Store all models in main models dict for compatibility
        self.models.update(self.barrier_models)
        self.models.update(self.fwd_ret_models)
        
        results = {
            'barrier_models': self.barrier_models,
            'fwd_ret_models': self.fwd_ret_models,
            'calibrators': self.calibrators,
            'target_types': self.target_types,
            'feature_names': feature_names,
            'n_features': X.shape[1],
            'n_samples': len(X),
            'gate_config': self._get_gate_config()
        }
        
        return results
    
    def predict(self, X: np.ndarray, **kwargs) -> Dict[str, np.ndarray]:
        """Make predictions using cascade approach with gating"""
        predictions = {}
        
        # Get barrier predictions (gates)
        barrier_preds = {}
        for target_name, model in self.barrier_models.items():
            try:
                # Get raw predictions
                if hasattr(model, 'predict_proba'):
                    raw_pred = model.predict_proba(X)[:, 1]
                else:
                    raw_pred = model.predict(X)
                
                # Apply calibration if available
                if target_name in self.calibrators:
                    calibrator = self.calibrators[target_name]
                    calibrated_pred = calibrator.predict(raw_pred.reshape(-1, 1))
                else:
                    calibrated_pred = raw_pred
                
                barrier_preds[target_name] = calibrated_pred
                
            except Exception as e:
                logger.error(f"Error predicting barrier {target_name}: {e}")
                barrier_preds[target_name] = np.zeros(len(X))
        
        # Get forward return predictions
        fwd_ret_preds = {}
        for target_name, model in self.fwd_ret_models.items():
            try:
                pred = model.predict(X)
                fwd_ret_preds[target_name] = pred
            except Exception as e:
                logger.error(f"Error predicting fwd_ret {target_name}: {e}")
                fwd_ret_preds[target_name] = np.zeros(len(X))
        
        # Apply gating logic
        gate_config = self._get_gate_config()
        
        for target_name, pred in fwd_ret_preds.items():
            # Apply barrier gates
            gated_pred = self._apply_gating(pred, barrier_preds, gate_config)
            predictions[target_name] = gated_pred
        
        # Also return barrier predictions
        predictions.update(barrier_preds)
        
        return predictions
    
    def get_target_types(self) -> Dict[str, str]:
        """Return target types for each target"""
        return self.target_types.copy()
    
    def _determine_target_type(self, target_name: str, y: np.ndarray) -> str:
        """Determine if target is regression or classification"""
        
        # Check target name patterns
        if target_name.startswith('fwd_ret_'):
            return 'regression'
        elif any(target_name.startswith(prefix) for prefix in 
                ['will_peak', 'will_valley', 'mdd', 'mfe', 'y_will_']):
            return 'classification'
        
        # Check data characteristics
        unique_values = np.unique(y[~np.isnan(y)])
        
        if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
            return 'classification'
        else:
            return 'regression'
    
    def _create_classification_model(self, target_name: str):
        """Create classification model"""
        from sklearn.ensemble import RandomForestClassifier
        
        model_config = self.config.get('models', {}).get('classification', {})
        model_type = model_config.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', None),
                random_state=42
            )
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _create_regression_model(self, target_name: str):
        """Create regression model"""
        from sklearn.ensemble import RandomForestRegressor
        
        model_config = self.config.get('models', {}).get('regression', {})
        model_type = model_config.get('type', 'random_forest')
        
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', None),
                random_state=42
            )
        else:
            return RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _create_calibrator(self, model, X: np.ndarray, y: np.ndarray):
        """Create probability calibrator"""
        calibration_method = self.config.get('calibration_method', 'none')
        
        if calibration_method == 'none':
            return None
        
        try:
            if calibration_method == 'isotonic':
                from sklearn.isotonic import IsotonicRegression
                # Get predictions for calibration
                if hasattr(model, 'predict_proba'):
                    pred_proba = model.predict_proba(X)[:, 1]
                else:
                    pred_proba = model.predict(X)
                
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(pred_proba, y)
                return calibrator
                
            elif calibration_method == 'platt':
                from sklearn.calibration import CalibratedClassifierCV
                calibrator = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                calibrator.fit(X, y)
                return calibrator
                
            else:
                logger.warning(f"Unknown calibration method: {calibration_method}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating calibrator: {e}")
            return None
    
    def _get_gate_config(self) -> Dict[str, Any]:
        """Get gating configuration"""
        return {
            'threshold': self.config.get('gate_threshold', 0.5),
            'calibration_method': self.config.get('calibration_method', 'none'),
            'gating_rules': self.config.get('gating_rules', self._get_default_gating_rules())
        }
    
    def _get_default_gating_rules(self) -> Dict[str, Any]:
        """Get default gating rules"""
        return {
            'will_peak_5m': {
                'action': 'reduce',
                'factor': 0.5,
                'description': 'Reduce signal when peak likely'
            },
            'will_valley_5m': {
                'action': 'boost',
                'factor': 1.2,
                'description': 'Boost signal when valley likely'
            },
            'mdd_5m_0.001': {
                'action': 'block',
                'threshold': 0.7,
                'description': 'Block trades when high MDD probability'
            }
        }
    
    def _apply_gating(self, fwd_ret_pred: np.ndarray, barrier_preds: Dict[str, np.ndarray], 
                     gate_config: Dict[str, Any]) -> np.ndarray:
        """Apply gating logic to forward return predictions"""
        gated_pred = fwd_ret_pred.copy()
        threshold = gate_config['threshold']
        gating_rules = gate_config['gating_rules']
        
        # Apply each gating rule
        for barrier_name, rule in gating_rules.items():
            if barrier_name in barrier_preds:
                barrier_prob = barrier_preds[barrier_name]
                
                if rule['action'] == 'reduce':
                    # Reduce signal when barrier probability is high
                    mask = barrier_prob > threshold
                    gated_pred[mask] *= rule['factor']
                    
                elif rule['action'] == 'boost':
                    # Boost signal when barrier probability is high
                    mask = barrier_prob > threshold
                    gated_pred[mask] *= rule['factor']
                    
                elif rule['action'] == 'block':
                    # Block trades when barrier probability is very high
                    block_threshold = rule.get('threshold', 0.8)
                    mask = barrier_prob > block_threshold
                    gated_pred[mask] = 0  # Block the signal
                    
                logger.debug(f"Applied {rule['action']} rule for {barrier_name}: "
                           f"{mask.sum()} samples affected")
        
        return gated_pred
    
    def get_gating_summary(self, X: np.ndarray) -> Dict[str, Any]:
        """Get summary of gating effects"""
        predictions = self.predict(X)
        
        summary = {
            'barrier_predictions': {},
            'gating_effects': {}
        }
        
        # Analyze barrier predictions
        for target_name, pred in predictions.items():
            if target_name in self.barrier_models:
                summary['barrier_predictions'][target_name] = {
                    'mean_prob': float(np.mean(pred)),
                    'std_prob': float(np.std(pred)),
                    'high_prob_count': int(np.sum(pred > 0.7))
                }
        
        # Analyze gating effects on forward returns
        for target_name, pred in predictions.items():
            if target_name in self.fwd_ret_models:
                summary['gating_effects'][target_name] = {
                    'mean_signal': float(np.mean(pred)),
                    'std_signal': float(np.std(pred)),
                    'blocked_count': int(np.sum(pred == 0))
                }
        
        return summary
    
    def _create_classification_model(self, target_name: str):
        """Create classification model"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        model_config = self.config.get('models', {}).get('classification', {})
        model_type = model_config.get('type', 'RandomForest')
        
        if model_type == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'LogisticRegression':
            return LogisticRegression(
                C=model_config.get('C', 1.0),
                random_state=42
            )
        else:
            return RandomForestClassifier(random_state=42)
    
    def _create_regression_model(self, target_name: str):
        """Create regression model"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        model_config = self.config.get('models', {}).get('regression', {})
        model_type = model_config.get('type', 'RandomForest')
        
        if model_type == 'RandomForest':
            return RandomForestRegressor(
                n_estimators=model_config.get('n_estimators', 100),
                max_depth=model_config.get('max_depth', 10),
                random_state=42
            )
        elif model_type == 'Ridge':
            return Ridge(
                alpha=model_config.get('alpha', 1.0)
            )
        else:
            return RandomForestRegressor(random_state=42)
    
    def _create_calibrator(self, model, X: np.ndarray, y: np.ndarray):
        """Create calibrator for classification model"""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            
            # Only calibrate if it's a classification model
            if hasattr(model, 'predict_proba'):
                return CalibratedClassifierCV(model, method='isotonic', cv=3)
            return None
        except ImportError:
            logger.warning("Calibration not available")
            return None