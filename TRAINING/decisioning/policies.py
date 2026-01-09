# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Decision Policies

Define thresholds and heuristics for decision-making.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DecisionPolicy:
    """A decision policy that evaluates conditions and triggers actions."""
    
    def __init__(
        self,
        name: str,
        condition_fn,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        level: int = 1  # 0=no action, 1=warning, 2=recommendation, 3=action
    ):
        """
        Initialize policy.
        
        Args:
            name: Policy name
            condition_fn: Function(cohort_data, latest_run) -> bool
            action: Action code if triggered (e.g., "freeze_features")
            reason: Reason code if triggered (e.g., "jaccard_collapse")
            level: Decision level (0-3)
        """
        self.name = name
        self.condition_fn = condition_fn
        self.action = action
        self.reason = reason
        self.level = level
    
    @staticmethod
    def get_default_policies() -> List['DecisionPolicy']:
        """Get default decision policies (all thresholds config-driven via decision_policies.yaml)."""
        # Load config (SST: Single Source of Truth)
        try:
            from CONFIG.config_loader import get_cfg
            policy_cfg = {
                'feature_instability': {
                    'jaccard_threshold': get_cfg('decision_policies.feature_instability.jaccard_threshold', default=0.5, config_name='decision_policies'),
                    'jaccard_collapse_ratio': get_cfg('decision_policies.feature_instability.jaccard_collapse_ratio', default=0.8, config_name='decision_policies'),
                    'min_runs': get_cfg('decision_policies.feature_instability.min_runs', default=3, config_name='decision_policies'),
                    'min_recent_runs': get_cfg('decision_policies.feature_instability.min_recent_runs', default=2, config_name='decision_policies')
                },
                'route_instability': {
                    'entropy_threshold': get_cfg('decision_policies.route_instability.entropy_threshold', default=1.5, config_name='decision_policies'),
                    'change_threshold': get_cfg('decision_policies.route_instability.change_threshold', default=3, config_name='decision_policies'),
                    'change_window': get_cfg('decision_policies.route_instability.change_window', default=5, config_name='decision_policies'),
                    'min_runs': get_cfg('decision_policies.route_instability.min_runs', default=3, config_name='decision_policies')
                },
                'feature_explosion_decline': {
                    'auc_decline_threshold': get_cfg('decision_policies.feature_explosion_decline.auc_decline_threshold', default=-0.01, config_name='decision_policies'),
                    'feature_increase_threshold': get_cfg('decision_policies.feature_explosion_decline.feature_increase_threshold', default=10, config_name='decision_policies'),
                    'min_runs': get_cfg('decision_policies.feature_explosion_decline.min_runs', default=3, config_name='decision_policies')
                },
                'class_balance_drift': {
                    'drift_threshold': get_cfg('decision_policies.class_balance_drift.drift_threshold', default=0.1, config_name='decision_policies'),
                    'min_runs': get_cfg('decision_policies.class_balance_drift.min_runs', default=3, config_name='decision_policies'),
                    'min_recent_runs': get_cfg('decision_policies.class_balance_drift.min_recent_runs', default=2, config_name='decision_policies')
                }
            }
        except Exception as e:
            logger.warning(f"Failed to load decision_policies config, using defaults: {e}")
            policy_cfg = {
                'feature_instability': {'jaccard_threshold': 0.5, 'jaccard_collapse_ratio': 0.8, 'min_runs': 3, 'min_recent_runs': 2},
                'route_instability': {'entropy_threshold': 1.5, 'change_threshold': 3, 'change_window': 5, 'min_runs': 3},
                'feature_explosion_decline': {'auc_decline_threshold': -0.01, 'feature_increase_threshold': 10, 'min_runs': 3},
                'class_balance_drift': {'drift_threshold': 0.1, 'min_runs': 3, 'min_recent_runs': 2}
            }
        
        policies = []
        
        # Policy 1: Feature instability (jaccard collapse)
        fi_cfg = policy_cfg['feature_instability']
        def jaccard_collapse(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < fi_cfg['min_runs']:
                return False
            if 'jaccard_topK' not in cohort_data.columns:
                return False
            recent = cohort_data['jaccard_topK'].tail(fi_cfg['min_runs']).dropna()
            if len(recent) < fi_cfg['min_recent_runs']:
                return False
            return recent.iloc[-1] < fi_cfg['jaccard_threshold'] and recent.iloc[-1] < recent.iloc[-2] * fi_cfg['jaccard_collapse_ratio']
        
        policies.append(DecisionPolicy(
            name="feature_instability",
            condition_fn=jaccard_collapse,
            action="freeze_features",
            reason="jaccard_collapse",
            level=2
        ))
        
        # Policy 2: Route instability (high entropy or frequent changes)
        ri_cfg = policy_cfg['route_instability']
        def route_instability(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < ri_cfg['min_runs']:
                return False
            if 'route_entropy' in cohort_data.columns:
                recent_entropy = cohort_data['route_entropy'].tail(ri_cfg['min_runs']).dropna()
                if len(recent_entropy) > 0:
                    return recent_entropy.iloc[-1] > ri_cfg['entropy_threshold']
            if 'route_changed' in cohort_data.columns:
                recent_changes = cohort_data['route_changed'].tail(ri_cfg['change_window']).sum()
                return recent_changes >= ri_cfg['change_threshold']
            return False
        
        policies.append(DecisionPolicy(
            name="route_instability",
            condition_fn=route_instability,
            action="tighten_routing",
            reason="route_instability",
            level=2
        ))
        
        # Policy 3: Performance decline with feature explosion
        fed_cfg = policy_cfg['feature_explosion_decline']
        def feature_explosion_decline(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < fed_cfg['min_runs']:
                return False
            if 'auc' not in cohort_data.columns or 'n_features_selected' not in cohort_data.columns:
                return False
            recent = cohort_data.tail(fed_cfg['min_runs'])
            auc_trend = recent['auc'].diff().tail(2)
            feature_trend = recent['n_features_selected'].diff().tail(2)
            # AUC declining while features increasing
            return (auc_trend.iloc[-1] < fed_cfg['auc_decline_threshold'] and feature_trend.iloc[-1] > fed_cfg['feature_increase_threshold']) if len(auc_trend) > 0 and len(feature_trend) > 0 else False
        
        policies.append(DecisionPolicy(
            name="feature_explosion_decline",
            condition_fn=feature_explosion_decline,
            action="cap_features",
            reason="feature_explosion_decline",
            level=2
        ))
        
        # Policy 4: Class balance drift
        cbd_cfg = policy_cfg['class_balance_drift']
        def class_balance_drift(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < cbd_cfg['min_runs']:
                return False
            if 'pos_rate' not in cohort_data.columns:
                return False
            recent = cohort_data['pos_rate'].tail(cbd_cfg['min_runs']).dropna()
            if len(recent) < cbd_cfg['min_recent_runs']:
                return False
            drift = abs(recent.iloc[-1] - recent.iloc[0])
            return drift > cbd_cfg['drift_threshold']
        
        policies.append(DecisionPolicy(
            name="class_balance_drift",
            condition_fn=class_balance_drift,
            action="retune_class_weights",
            reason="pos_rate_drift",
            level=1  # Warning only
        ))
        
        return policies


def evaluate_policies(
    cohort_data: pd.DataFrame,
    latest_run: pd.Series,
    policies: List[DecisionPolicy]
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate all policies.
    
    Args:
        cohort_data: Historical data for cohort
        latest_run: Latest run data
        policies: List of policies to evaluate
    
    Returns:
        Dict mapping policy_name -> {triggered: bool, level: int, action: str, reason: str}
    """
    results = {}
    
    for policy in policies:
        try:
            triggered = policy.condition_fn(cohort_data, latest_run)
            results[policy.name] = {
                'triggered': triggered,
                'level': policy.level if triggered else 0,
                'action': policy.action if triggered else None,
                'reason': policy.reason if triggered else None
            }
        except Exception as e:
            logger.warning(f"Policy {policy.name} evaluation failed: {e}")
            results[policy.name] = {
                'triggered': False,
                'level': 0,
                'action': None,
                'reason': None
            }
    
    return results


# Hard clamps on patch actions (prevent unbounded changes)
PATCH_CLAMPS = {
    'n_features_selected': {'max_change_pct': 20},  # Max ±20%
    'auc_threshold': {'max_change_pct': 20},  # Max ±20%
    'frac_symbols_good_threshold': {'max_change_pct': 20},  # Max ±20%
    'max_features': {'max_change_pct': 20},  # Max ±20%
}


def apply_decision_patch(
    resolved_config: Dict[str, Any],
    decision_result: Any  # DecisionResult (avoid circular import)
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str]]:
    """
    Apply decision patch to resolved config with hard clamps.
    
    **SAFETY: Only applies ONE policy at a time (first action in list).**
    **SAFETY: All changes are clamped to prevent unbounded modifications.**
    
    Args:
        resolved_config: Current resolved config
        decision_result: Decision result
    
    Returns:
        (new_config, patch_dict, warnings) - new config, patch that was applied, and any warnings
    """
    new_config = resolved_config.copy()
    patch = {}
    warnings = []
    
    actions = decision_result.decision_action_mask or []
    
    # SAFETY: Apply only ONE policy at a time (first action)
    if len(actions) > 1:
        warnings.append(f"Multiple actions detected: {actions}. Applying only first: {actions[0]}")
        actions = [actions[0]]
    
    if not actions:
        return new_config, patch, warnings
    
    action = actions[0]
    
    # Action: freeze_features
    if action == "freeze_features":
        # Set feature selection to use cached/previous selection
        if 'feature_selection' not in new_config:
            new_config['feature_selection'] = {}
        new_config['feature_selection']['use_cached'] = True
        patch['feature_selection.use_cached'] = True
    
    # Action: tighten_routing
    elif action == "tighten_routing":
        # Increase routing thresholds (clamped to max 20% increase)
        if 'target_routing' not in new_config:
            new_config['target_routing'] = {}
        if 'routing' not in new_config['target_routing']:
            new_config['target_routing']['routing'] = {}
        routing = new_config['target_routing']['routing']
        
        # Clamp auc_threshold (max 20% increase)
        old_cs_threshold = routing.get('auc_threshold', 0.65)
        new_cs_threshold = min(old_cs_threshold * 1.2, old_cs_threshold * 1.2)  # Max 20% increase
        if new_cs_threshold > old_cs_threshold * 1.2:
            new_cs_threshold = old_cs_threshold * 1.2
            warnings.append(f"auc_threshold clamped to max 20% increase: {old_cs_threshold} → {new_cs_threshold}")
        routing['auc_threshold'] = new_cs_threshold
        patch['target_routing.routing.auc_threshold'] = new_cs_threshold
        
        # Clamp frac_symbols_good_threshold (max 20% increase)
        old_frac_threshold = routing.get('frac_symbols_good_threshold', 0.5)
        new_frac_threshold = min(old_frac_threshold * 1.2, old_frac_threshold * 1.2)  # Max 20% increase
        if new_frac_threshold > old_frac_threshold * 1.2:
            new_frac_threshold = old_frac_threshold * 1.2
            warnings.append(f"frac_symbols_good_threshold clamped to max 20% increase: {old_frac_threshold} → {new_frac_threshold}")
        routing['frac_symbols_good_threshold'] = new_frac_threshold
        patch['target_routing.routing.frac_symbols_good_threshold'] = new_frac_threshold
    
    # Action: cap_features
    elif action == "cap_features":
        # Add feature cap (clamped: max 20% reduction from current)
        if 'feature_selection' not in new_config:
            new_config['feature_selection'] = {}
        
        # Get current max_features if set
        current_max = new_config['feature_selection'].get('max_features')
        if current_max is None:
            # Estimate from top_m_features if available
            current_max = resolved_config.get('top_m_features', 100)
        
        # Clamp: reduce by max 20%
        new_max = max(int(current_max * 0.8), 10)  # At least 10 features
        if new_max < current_max * 0.8:
            warnings.append(f"max_features clamped to max 20% reduction: {current_max} → {new_max}")
        
        new_config['feature_selection']['max_features'] = new_max
        patch['feature_selection.max_features'] = new_max
    
    # Action: retune_class_weights
    elif action == "retune_class_weights":
        # Flag for class weight retuning (doesn't auto-apply, just flags)
        if 'training' not in new_config:
            new_config['training'] = {}
        new_config['training']['retune_class_weights'] = True
        patch['training.retune_class_weights'] = True
    
    else:
        warnings.append(f"Unknown action: {action}. Skipping.")
    
    return new_config, patch, warnings
