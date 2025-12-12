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
        """Get default decision policies."""
        policies = []
        
        # Policy 1: Feature instability (jaccard collapse)
        def jaccard_collapse(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < 3:
                return False
            if 'jaccard_topK' not in cohort_data.columns:
                return False
            recent = cohort_data['jaccard_topK'].tail(3).dropna()
            if len(recent) < 2:
                return False
            return recent.iloc[-1] < 0.5 and recent.iloc[-1] < recent.iloc[-2] * 0.8
        
        policies.append(DecisionPolicy(
            name="feature_instability",
            condition_fn=jaccard_collapse,
            action="freeze_features",
            reason="jaccard_collapse",
            level=2
        ))
        
        # Policy 2: Route instability (high entropy or frequent changes)
        def route_instability(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < 3:
                return False
            if 'route_entropy' in cohort_data.columns:
                recent_entropy = cohort_data['route_entropy'].tail(3).dropna()
                if len(recent_entropy) > 0:
                    return recent_entropy.iloc[-1] > 1.5  # High entropy = unstable routing
            if 'route_changed' in cohort_data.columns:
                recent_changes = cohort_data['route_changed'].tail(5).sum()
                return recent_changes >= 3  # 3+ changes in last 5 runs
            return False
        
        policies.append(DecisionPolicy(
            name="route_instability",
            condition_fn=route_instability,
            action="tighten_routing",
            reason="route_instability",
            level=2
        ))
        
        # Policy 3: Performance decline with feature explosion
        def feature_explosion_decline(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < 3:
                return False
            if 'cs_auc' not in cohort_data.columns or 'n_features_selected' not in cohort_data.columns:
                return False
            recent = cohort_data.tail(3)
            auc_trend = recent['cs_auc'].diff().tail(2)
            feature_trend = recent['n_features_selected'].diff().tail(2)
            # AUC declining while features increasing
            return (auc_trend.iloc[-1] < -0.01 and feature_trend.iloc[-1] > 10) if len(auc_trend) > 0 and len(feature_trend) > 0 else False
        
        policies.append(DecisionPolicy(
            name="feature_explosion_decline",
            condition_fn=feature_explosion_decline,
            action="cap_features",
            reason="feature_explosion_decline",
            level=2
        ))
        
        # Policy 4: Class balance drift
        def class_balance_drift(cohort_data: pd.DataFrame, latest: pd.Series) -> bool:
            if len(cohort_data) < 3:
                return False
            if 'pos_rate' not in cohort_data.columns:
                return False
            recent = cohort_data['pos_rate'].tail(3).dropna()
            if len(recent) < 2:
                return False
            drift = abs(recent.iloc[-1] - recent.iloc[0])
            return drift > 0.1  # 10% drift
        
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
    'cs_auc_threshold': {'max_change_pct': 20},  # Max ±20%
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
        
        # Clamp cs_auc_threshold (max 20% increase)
        old_cs_threshold = routing.get('cs_auc_threshold', 0.65)
        new_cs_threshold = min(old_cs_threshold * 1.2, old_cs_threshold * 1.2)  # Max 20% increase
        if new_cs_threshold > old_cs_threshold * 1.2:
            new_cs_threshold = old_cs_threshold * 1.2
            warnings.append(f"cs_auc_threshold clamped to max 20% increase: {old_cs_threshold} → {new_cs_threshold}")
        routing['cs_auc_threshold'] = new_cs_threshold
        patch['target_routing.routing.cs_auc_threshold'] = new_cs_threshold
        
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
