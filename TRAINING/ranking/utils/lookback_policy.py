"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
Centralized Lookback Policy Resolution

Single source of truth for lookback enforcement policy.
Prevents "strict implies drop" footgun by resolving policy once from config.
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LookbackPolicy:
    """
    Resolved lookback enforcement policy.
    
    This is the single source of truth for policy behavior.
    All enforcement stages must use the same resolved policy.
    """
    cap_minutes: Optional[float]  # Lookback cap (None = no cap, "auto" = computed)
    over_budget_action: str  # "hard_stop" | "drop" | "warn"
    unknown_lookback_action: str  # "hard_stop" | "drop" | "warn"
    unknown_policy: str  # "drop" (inf) | "conservative" (1440m) - for inference only
    policy_mode: str  # "strict" | "permissive" (logging label only)
    
    def should_quarantine_unknown(self) -> bool:
        """Should unknown lookback features be quarantined?"""
        return self.unknown_lookback_action in ("hard_stop", "drop")
    
    def should_quarantine_over_cap(self) -> bool:
        """Should over-cap features be quarantined?"""
        return self.over_budget_action in ("hard_stop", "drop")
    
    def should_hard_stop(self) -> bool:
        """Should violations cause hard-stop (raise exception)?"""
        return self.over_budget_action == "hard_stop" or self.unknown_lookback_action == "hard_stop"


def resolve_lookback_policy(
    resolved_config: Optional[Any] = None,
    config_name: str = "safety_config"
) -> LookbackPolicy:
    """
    Resolve lookback enforcement policy from config.
    
    This is the SINGLE SOURCE OF TRUTH for policy resolution.
    All enforcement stages must call this and use the same resolved policy.
    
    Args:
        resolved_config: Optional ResolvedConfig (for purge-derived cap)
        config_name: Config name to load from
    
    Returns:
        LookbackPolicy with resolved actions
    """
    from CONFIG.config_loader import get_cfg
    
    # Load policy_mode (logging label only)
    policy_mode = get_cfg("safety.leakage_detection.policy", default="strict", config_name=config_name)
    
    # Load over_budget_action (the REAL behavior)
    over_budget_action = get_cfg(
        "safety.leakage_detection.over_budget_action",
        default="drop",
        config_name=config_name
    )
    
    # Load unknown_lookback_action (NEW: explicit control)
    unknown_lookback_action = get_cfg(
        "safety.leakage_detection.unknown_lookback_action",
        default=over_budget_action,  # Default to same as over_budget_action
        config_name=config_name
    )
    
    # Load cap
    cap_minutes = None
    budget_cap_raw = get_cfg(
        "safety.leakage_detection.lookback_budget_minutes",
        default="auto",
        config_name=config_name
    )
    if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
        cap_minutes = float(budget_cap_raw)
    elif resolved_config and resolved_config.purge_minutes:
        # Fallback to purge-derived (with 1% buffer)
        cap_minutes = resolved_config.purge_minutes * 0.99
    
    # Determine unknown_policy for inference
    # In strict mode or if unknown_lookback_action is hard_stop/drop, use "drop" (inf)
    # Otherwise use "conservative" (1440m) for backward compatibility
    if policy_mode == "strict" or unknown_lookback_action in ("hard_stop", "drop"):
        unknown_policy = "drop"  # inf = unsafe
    else:
        unknown_policy = "conservative"  # 1440m = safe default
    
    return LookbackPolicy(
        cap_minutes=cap_minutes,
        over_budget_action=over_budget_action,
        unknown_lookback_action=unknown_lookback_action,
        unknown_policy=unknown_policy,
        policy_mode=policy_mode
    )


def assert_featureset_fingerprint(
    label: str,
    expected: 'EnforcedFeatureSet',
    actual_features: list[str],
    logger_instance: Optional[logging.Logger] = None,
    allow_reorder: bool = False  # If True, only check set equality (not order)
) -> None:
    """
    Reusable invariant check: verify featureset matches expected with exact list equality.
    
    Call this at EVERY boundary where split-brain can happen:
    - After cleaning (SAFE_CANDIDATES)
    - After leak removal (AFTER_LEAK_REMOVAL)
    - After gatekeeper (POST_GATEKEEPER)
    - After pruning (POST_PRUNE)
    - Before model training (MODEL_TRAIN_INPUT)
    - FS pre/post enforcement
    
    Args:
        label: Stage label for error message
        expected: Expected EnforcedFeatureSet (the truth)
        actual_features: Actual feature list to validate
        logger_instance: Optional logger (uses module logger if None)
        allow_reorder: If True, only check set equality (not order). Default: False (strict order check)
    
    Raises:
        RuntimeError: If featureset mismatch detected (with actionable diff)
    """
    if logger_instance is None:
        logger_instance = logger
    
    from TRAINING.ranking.utils.cross_sectional_data import _compute_feature_fingerprint
    
    # CRITICAL: Check exact list equality first (not just hash)
    if actual_features == expected.features:
        # Exact match - pass
        logger_instance.debug(
            f"✅ INVARIANT CHECK PASSED ({label}): "
            f"exact list match, n_features={len(actual_features)}"
        )
        return
    
    # Compute fingerprints for logging
    actual_fp_set, actual_fp_ordered = _compute_feature_fingerprint(actual_features, set_invariant=True)
    
    # Check set equality (members)
    expected_set = set(expected.features)
    actual_set = set(actual_features)
    added = actual_set - expected_set
    removed = expected_set - actual_set
    
    # Check order divergence (if not allowing reorder)
    order_divergence_idx = None
    order_divergence_window = None
    if not allow_reorder:
        # Find first index where order diverges
        min_len = min(len(expected.features), len(actual_features))
        for i in range(min_len):
            if expected.features[i] != actual_features[i]:
                order_divergence_idx = i
                # Show window around divergence (5 before, 5 after)
                start = max(0, i - 5)
                end = min(len(expected.features), len(actual_features), i + 6)
                order_divergence_window = {
                    "index": i,
                    "expected": expected.features[start:end],
                    "actual": actual_features[start:end]
                }
                break
    
    # Build actionable error message
    error_parts = [
        f"🚨 FEATURESET MIS-WIRE ({label}):",
        f"   Stage: {expected.stage} → {label}",
        f"   Cap: {expected.cap_minutes:.1f}m" if expected.cap_minutes else "   Cap: None",
        f"   Expected n_features={len(expected.features)}, actual n_features={len(actual_features)}"
    ]
    
    if added:
        error_parts.append(f"   Added features ({len(added)}): {list(added)[:10]}")
    if removed:
        error_parts.append(f"   Removed features ({len(removed)}): {list(removed)[:10]}")
    
    if order_divergence_idx is not None:
        error_parts.append(
            f"   Order divergence at index {order_divergence_idx}: "
            f"expected={order_divergence_window['expected']}, "
            f"actual={order_divergence_window['actual']}"
        )
    
    # Fingerprint info
    if actual_fp_set != expected.fingerprint_set:
        error_parts.append(
            f"   Set fingerprint mismatch: expected={expected.fingerprint_set[:16]}, "
            f"actual={actual_fp_set[:16]}"
        )
    if not allow_reorder and actual_fp_ordered != expected.fingerprint_ordered:
        error_parts.append(
            f"   Order fingerprint mismatch: expected={expected.fingerprint_ordered[:16]}, "
            f"actual={actual_fp_ordered[:16]}"
        )
    
    error_msg = "\n".join(error_parts)
    logger_instance.error(error_msg)
    
    raise RuntimeError(
        f"FEATURESET MIS-WIRE ({label}): "
        f"Feature list differs from {expected.stage}. "
        f"Expected n_features={len(expected.features)}, actual n_features={len(actual_features)}. "
        f"{'Order divergence detected. ' if order_divergence_idx is not None else ''}"
        f"See logs for detailed diff."
    )
