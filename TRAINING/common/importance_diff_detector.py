# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Feature Importance Diff Detector

Compares feature importances between models trained with all features
vs. models trained with only "safe" features to detect potential leakage.

If a feature has high importance in the full model but low in the safe model,
it's suspicious and may be encoding future information.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SuspiciousFeature:
    """A feature flagged as potentially leaky based on importance diff."""
    feature_name: str
    importance_full: float
    importance_safe: float
    importance_diff: float
    relative_diff: float  # diff / max(importance_full, 1e-6)
    reason: str


class ImportanceDiffDetector:
    """
    Detects potential leakage by comparing feature importances.
    
    Compares models trained with:
    - Full feature set (may include leaky features)
    - Safe feature set (only registry-allowed features)
    
    Features with high importance in full but low in safe are flagged as suspicious.
    """
    
    def __init__(
        self,
        diff_threshold: Optional[float] = None,
        relative_diff_threshold: Optional[float] = None,
        min_importance_full: Optional[float] = None
    ):
        """
        Initialize importance diff detector (all thresholds config-driven via stability_config.yaml).
        
        Args:
            diff_threshold: Absolute difference threshold (if None, loads from config)
            relative_diff_threshold: Relative difference threshold (if None, loads from config)
            min_importance_full: Minimum importance in full model (if None, loads from config)
        """
        # Load config (SST: Single Source of Truth)
        try:
            from CONFIG.config_loader import get_cfg
            self.diff_threshold = diff_threshold if diff_threshold is not None else get_cfg('stability.importance_diff.diff_threshold', default=0.1, config_name='stability_config')
            self.relative_diff_threshold = relative_diff_threshold if relative_diff_threshold is not None else get_cfg('stability.importance_diff.relative_diff_threshold', default=0.5, config_name='stability_config')
            self.min_importance_full = min_importance_full if min_importance_full is not None else get_cfg('stability.importance_diff.min_importance_full', default=0.01, config_name='stability_config')
        except Exception as e:
            logger.warning(f"Failed to load stability_config, using defaults: {e}")
            self.diff_threshold = diff_threshold if diff_threshold is not None else 0.1
            self.relative_diff_threshold = relative_diff_threshold if relative_diff_threshold is not None else 0.5
            self.min_importance_full = min_importance_full if min_importance_full is not None else 0.01
    
    def get_feature_importance(
        self,
        model: Any,
        feature_names: List[str],
        importance_method: str = 'auto'
    ) -> pd.Series:
        """
        Extract feature importance from a model.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            importance_method: Method to use ('auto', 'native', 'shap', 'permutation')
        
        Returns:
            Series with feature names as index and importance as values
        """
        # Try native importance first
        if importance_method in ('auto', 'native'):
            try:
                if hasattr(model, 'feature_importances_'):
                    # sklearn models
                    importance = model.feature_importances_
                elif hasattr(model, 'feature_importance'):
                    # LightGBM
                    importance = model.feature_importance(importance_type='gain')
                elif hasattr(model, 'get_feature_importance'):
                    # CatBoost
                    importance = model.get_feature_importance()
                elif hasattr(model, 'coef_'):
                    # Linear models
                    importance = np.abs(model.coef_)
                    if len(importance.shape) > 1:
                        importance = importance[0]  # Take first row for multi-output
                elif hasattr(model, 'get_score'):
                    # XGBoost native API
                    score_dict = model.get_score(importance_type='gain')
                    importance = np.array([score_dict.get(f, 0.0) for f in feature_names])
                else:
                    raise ValueError("Model does not have native feature importance")
                
                # Ensure importance matches feature_names length
                if len(importance) != len(feature_names):
                    logger.warning(f"Importance length ({len(importance)}) doesn't match features ({len(feature_names)})")
                    # Pad or truncate if needed
                    if len(importance) < len(feature_names):
                        importance = np.pad(importance, (0, len(feature_names) - len(importance)), 'constant')
                    else:
                        importance = importance[:len(feature_names)]
                
                # Normalize to 0-1 range
                if importance.max() > 0:
                    importance = importance / importance.max()
                
                return pd.Series(importance, index=feature_names)
            
            except Exception as e:
                logger.debug(f"Native importance extraction failed: {e}")
                if importance_method == 'native':
                    raise
        
        # Fallback: return zero importance
        logger.warning(f"Could not extract importance from model, returning zeros")
        return pd.Series(0.0, index=feature_names)
    
    def detect_suspicious_features(
        self,
        model_full: Any,
        model_safe: Any,
        feature_names_full: List[str],
        feature_names_safe: List[str],
        importance_method: str = 'auto'
    ) -> List[SuspiciousFeature]:
        """
        Compare feature importances to detect potentially leaky features.
        
        Args:
            model_full: Model trained with all features (including potentially leaky)
            model_safe: Model trained with only safe features (registry-validated)
            feature_names_full: Feature names for full model
            feature_names_safe: Feature names for safe model
            importance_method: Method to extract importance
        
        Returns:
            List of SuspiciousFeature objects
        """
        # Extract importances
        importance_full = self.get_feature_importance(model_full, feature_names_full, importance_method)
        importance_safe = self.get_feature_importance(model_safe, feature_names_safe, importance_method)
        
        # Create index for safe features (may be subset of full)
        importance_safe_indexed = pd.Series(0.0, index=feature_names_full)
        for feat in feature_names_safe:
            if feat in importance_safe.index:
                importance_safe_indexed[feat] = importance_safe[feat]
        
        # Calculate differences
        suspicious = []
        
        for feat in feature_names_full:
            imp_full = importance_full.get(feat, 0.0)
            imp_safe = importance_safe_indexed.get(feat, 0.0)
            
            # Skip if importance in full model is too low (noise)
            if imp_full < self.min_importance_full:
                continue
            
            # Calculate absolute and relative differences
            diff = imp_full - imp_safe
            relative_diff = diff / max(imp_full, 1e-6)
            
            # Check thresholds
            is_suspicious = (
                diff > self.diff_threshold or
                relative_diff > self.relative_diff_threshold
            )
            
            if is_suspicious:
                reason = []
                if diff > self.diff_threshold:
                    reason.append(f"absolute_diff={diff:.3f} > {self.diff_threshold}")
                if relative_diff > self.relative_diff_threshold:
                    reason.append(f"relative_diff={relative_diff:.1%} > {self.relative_diff_threshold:.1%}")
                
                suspicious.append(SuspiciousFeature(
                    feature_name=feat,
                    importance_full=float(imp_full),
                    importance_safe=float(imp_safe),
                    importance_diff=float(diff),
                    relative_diff=float(relative_diff),
                    reason="; ".join(reason)
                ))
        
        # Sort by difference (most suspicious first)
        suspicious.sort(key=lambda x: x.importance_diff, reverse=True)
        
        return suspicious
    
    def detect_and_report(
        self,
        model_full: Any,
        model_safe: Any,
        feature_names_full: List[str],
        feature_names_safe: List[str],
        importance_method: str = 'auto',
        top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Detect suspicious features and generate a report.
        
        Args:
            model_full: Model trained with all features
            model_safe: Model trained with only safe features
            feature_names_full: Feature names for full model
            feature_names_safe: Feature names for safe model
            importance_method: Method to extract importance
            top_n: Number of top suspicious features to include in summary
        
        Returns:
            Dictionary with detection results and summary
        """
        suspicious = self.detect_suspicious_features(
            model_full, model_safe,
            feature_names_full, feature_names_safe,
            importance_method
        )
        
        # Generate report
        if suspicious:
            logger.warning(
                f"🚨 SUSPECTED LEAKS: {len(suspicious)} features have high importance "
                f"in full model but low in safe model"
            )
            
            # Log top N suspicious features
            for i, feat in enumerate(suspicious[:top_n], 1):
                logger.warning(
                    f"  {i}. {feat.feature_name}: "
                    f"full={feat.importance_full:.3f}, "
                    f"safe={feat.importance_safe:.3f}, "
                    f"diff={feat.importance_diff:.3f} ({feat.reason})"
                )
            
            if len(suspicious) > top_n:
                logger.warning(f"  ... and {len(suspicious) - top_n} more suspicious features")
        else:
            logger.info("✅ No suspicious features detected (importance diff test passed)")
        
        return {
            'n_suspicious': len(suspicious),
            'suspicious_features': [
                {
                    'feature_name': f.feature_name,
                    'importance_full': f.importance_full,
                    'importance_safe': f.importance_safe,
                    'importance_diff': f.importance_diff,
                    'relative_diff': f.relative_diff,
                    'reason': f.reason
                }
                for f in suspicious
            ],
            'top_n': top_n
        }

