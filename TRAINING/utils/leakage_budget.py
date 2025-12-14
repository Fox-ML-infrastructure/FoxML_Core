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
Unified Leakage Budget Calculator

Single source of truth for feature lookback calculation.
Used by audit, gatekeeper, and CV to ensure consistency.

CRITICAL: All lookback calculations must use this module to prevent
structural contradictions where audit and gatekeeper report different values.
"""

import re
import logging
import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


def _compute_feature_fingerprint(feature_names: Iterable[str], set_invariant: bool = True) -> Tuple[str, str]:
    """
    Compute feature set fingerprints (set-invariant and order-sensitive).
    
    Args:
        feature_names: Iterable of feature names
        set_invariant: If True, compute set-invariant fingerprint (sorted). If False, preserve order.
    
    Returns:
        (set_fingerprint, order_fingerprint) tuple:
        - set_fingerprint: Set-invariant fingerprint (sorted, for set equality checks)
        - order_fingerprint: Order-sensitive fingerprint (for order-change detection)
    """
    feature_list = list(feature_names)
    
    # Set-invariant fingerprint (sorted, for set equality)
    sorted_features = sorted(feature_list)
    set_str = "\n".join(sorted_features)
    set_fingerprint = hashlib.sha1(set_str.encode()).hexdigest()[:8]
    
    # Order-sensitive fingerprint (for order-change detection)
    order_str = "\n".join(feature_list)
    order_fingerprint = hashlib.sha1(order_str.encode()).hexdigest()[:8]
    
    return set_fingerprint, order_fingerprint

# Calendar/exogenous features that have 0m lookback (not rolling windows)
CALENDAR_FEATURES = {
    "day_of_week",
    "trading_day_of_month",
    "trading_day_of_quarter",
    "holiday_dummy",
    "pre_holiday_dummy",
    "post_holiday_dummy",
    "_weekday",
    "weekday",
    "is_weekend",
    "is_month_end",
    "is_quarter_end",
    "is_year_end",
}


@dataclass(frozen=True)
class LeakageBudget:
    """
    Leakage budget for a feature set.
    
    Attributes:
        interval_minutes: Data bar interval in minutes
        horizon_minutes: Target prediction horizon in minutes
        max_feature_lookback_minutes: Maximum feature lookback in minutes
    
    Properties:
        required_gap_minutes: Conservative gap required between train/test
                            (max_feature_lookback_minutes + horizon_minutes)
    """
    interval_minutes: float
    horizon_minutes: float
    max_feature_lookback_minutes: float

    @property
    def required_gap_minutes(self) -> float:
        """
        Conservative gap required: features use past up to lookback,
        label uses future up to horizon.
        """
        return self.max_feature_lookback_minutes + self.horizon_minutes


@dataclass(frozen=True)
class LookbackResult:
    """
    Result of lookback computation with fingerprint validation.
    
    Attributes:
        max_minutes: Maximum feature lookback in minutes (None if cannot compute)
        top_offenders: List of (feature_name, lookback_minutes) tuples for top offenders
        fingerprint: Feature set fingerprint (set-invariant, sorted)
        order_fingerprint: Order-sensitive fingerprint (for order-change detection)
    """
    max_minutes: Optional[float]
    top_offenders: List[Tuple[str, float]]
    fingerprint: str
    order_fingerprint: str


def infer_lookback_minutes(
    feature_name: str,
    interval_minutes: float,
    spec_lookback_minutes: Optional[float] = None,
    registry: Optional[Any] = None,
    unknown_policy: str = "conservative",  # "conservative" or "drop"
) -> float:
    """
    Infer feature lookback in minutes from feature name and metadata.
    
    Precedence order (highest to lowest):
    1. Explicit spec_lookback_minutes (from registry/schema metadata)
    2. Calendar features whitelist (0m lookback)
    3. Explicit time suffixes (_15m, _24h, _1d)
    4. Bar-based patterns (ret_288, sma_20, etc.)
    5. Keyword heuristics (daily patterns, etc.)
    6. Unknown policy (conservative default or drop)
    
    Args:
        feature_name: Feature name to analyze
        interval_minutes: Data bar interval in minutes
        spec_lookback_minutes: Explicit lookback from registry/schema (highest priority)
        registry: Optional feature registry for metadata lookup
        unknown_policy: "conservative" (default 1440m) or "drop" (return inf)
    
    Returns:
        Lookback in minutes (float('inf') if unknown_policy="drop" and cannot infer)
    """
    # 1) Schema/registry metadata wins (highest priority)
    if spec_lookback_minutes is not None:
        return float(spec_lookback_minutes)
    
    # Try registry if available
    if registry is not None:
        try:
            metadata = registry.get_feature_metadata(feature_name)
            lag_bars = metadata.get('lag_bars')
            if lag_bars is not None and lag_bars >= 0:
                return float(lag_bars * interval_minutes)
        except Exception:
            pass  # Fall through to pattern matching
    
    # 2) True "calendar/exogenous" features should be 0 lookback
    if feature_name in CALENDAR_FEATURES:
        return 0.0
    
    # Check for calendar feature patterns (before keyword heuristics)
    if any(cal in feature_name.lower() for cal in ['day_of_week', 'holiday', 'trading_day', 'weekday', 'is_weekend', 'is_month_end']):
        return 0.0
    
    # 3) Parse explicit time suffix patterns (most reliable)
    # Minute-based patterns (e.g., _15m, _30m, _1440m) - CHECK FIRST
    minutes_match = re.search(r'_(\d+(?:\.\d+)?)(m|M)$', feature_name)
    if minutes_match:
        val = float(minutes_match.group(1))
        return val
    
    # Hour-based patterns (e.g., _12h, _24h)
    hours_match = re.search(r'_(\d+(?:\.\d+)?)(h|H)(?!\d)', feature_name)
    if hours_match:
        val = float(hours_match.group(1))
        return val * 60.0
    
    # Day-based patterns (e.g., _1d, _3d)
    days_match = re.search(r'_(\d+(?:\.\d+)?)(d|D)(?!\d)', feature_name)
    if days_match:
        val = float(days_match.group(1))
        return val * 1440.0
    
    # 4) Parse "bars" style (e.g., ret_288, sma_20) as bars*interval
    # Only treat as bars if it's plausibly a window size (avoid ticker/enum collisions)
    bar_patterns = [
        r'^(ret|sma|ema|rsi|macd|bb|atr|adx|mom|vol|std|var|sma|ema|rsi)_(\d+)$',
        r'_(\d+)$',  # Generic numeric suffix (only if >= 2 to avoid false positives)
    ]
    
    for pattern in bar_patterns:
        match = re.match(pattern, feature_name) if '^' in pattern else re.search(pattern, feature_name)
        if match:
            bars = int(match.group(2) if len(match.groups()) > 1 else match.group(1))
            # Only treat as bars if it's plausibly a window size (>= 2)
            if bars >= 2:
                return bars * float(interval_minutes)
    
    # 5) Keyword heuristics (fallback only if no explicit suffix found)
    # Explicit daily patterns (ends with _1d, _24h, starts with daily_, etc.)
    if (re.search(r'_1d$|_1D$|_24h$|_24H$|^daily_|_daily$|_1440m|1440(?!\d)', feature_name, re.I) or
        re.search(r'rolling.*daily|daily.*high|daily.*low', feature_name, re.I) or
        re.search(r'volatility.*day|vol.*day|volume.*day', feature_name, re.I)):
        # Explicit daily patterns
        return 1440.0
    
    # Calendar features (monthly, quarterly, yearly) - these are NOT rolling windows
    if re.search(r'monthly|quarterly|yearly', feature_name, re.I):
        # These are calendar features, not rolling windows - should be 0m
        # But if they're used as rolling aggregations, they need lookback
        # For now, be conservative and return 0m (they're exogenous)
        return 0.0
    
    # 6) Unknown feature policy
    if unknown_policy == "drop":
        return float("inf")  # Caller will drop
    
    # Conservative default (1440m = 1 day)
    return 1440.0


def compute_budget(
    final_feature_names: Iterable[str],
    interval_minutes: float,
    horizon_minutes: float,
    registry: Optional[Any] = None,
    max_lookback_cap_minutes: Optional[float] = None,
    unknown_policy: str = "conservative",
    expected_fingerprint: Optional[str] = None,
    stage: str = "unknown"
) -> Tuple[LeakageBudget, str, str]:
    """
    Compute leakage budget from final feature list.
    
    This is the SINGLE SOURCE OF TRUTH for lookback calculation.
    Audit, gatekeeper, and CV must all call this function with the SAME
    final_feature_names list to ensure consistency.
    
    Args:
        final_feature_names: Final feature names used in training (post gatekeeper + pruning)
        interval_minutes: Data bar interval in minutes
        horizon_minutes: Target prediction horizon in minutes
        registry: Optional feature registry for metadata lookup
        max_lookback_cap_minutes: Optional cap for ranking mode (e.g., 240m = 4 hours)
        unknown_policy: "conservative" (default 1440m) or "drop" (return inf)
        expected_fingerprint: Optional expected fingerprint for validation
        stage: Stage name for logging (e.g., "post_gatekeeper", "post_pruning")
    
    Returns:
        (LeakageBudget, set_fingerprint, order_fingerprint) tuple
    """
    feature_list = list(final_feature_names) if final_feature_names else []
    set_fingerprint, order_fingerprint = _compute_feature_fingerprint(feature_list, set_invariant=True)
    
    # Validate fingerprint if expected (use set-invariant for comparison)
    if expected_fingerprint is not None and set_fingerprint != expected_fingerprint:
        logger.error(
            f"🚨 FINGERPRINT MISMATCH at stage={stage}: "
            f"expected={expected_fingerprint}, actual={set_fingerprint}. "
            f"This indicates lookback computed on different feature set than enforcement."
        )
    
    if not feature_list or interval_minutes <= 0:
        return (
            LeakageBudget(
                interval_minutes=interval_minutes,
                horizon_minutes=horizon_minutes,
                max_feature_lookback_minutes=0.0
            ),
            set_fingerprint,
            order_fingerprint
        )
    
    # Get registry if not provided
    if registry is None:
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
        except Exception:
            registry = None
    
    # Compute lookback for each feature
    lookbacks = []
    for feat_name in final_feature_names:
        # Try registry first for explicit metadata
        spec_lookback = None
        if registry is not None:
            try:
                metadata = registry.get_feature_metadata(feat_name)
                lag_bars = metadata.get('lag_bars')
                if lag_bars is not None and lag_bars >= 0:
                    spec_lookback = float(lag_bars * interval_minutes)
            except Exception:
                pass
        
        # Infer lookback
        lookback = infer_lookback_minutes(
            feat_name,
            interval_minutes,
            spec_lookback_minutes=spec_lookback,
            registry=registry,
            unknown_policy=unknown_policy
        )
        
        # Skip if unknown_policy="drop" and lookback is inf
        if lookback == float("inf"):
            continue
        
        lookbacks.append(lookback)
    
    # Compute max lookback
    max_lookback = max(lookbacks) if lookbacks else 0.0
    
    # Apply cap if provided
    if max_lookback_cap_minutes is not None and max_lookback > max_lookback_cap_minutes:
        max_lookback = max_lookback_cap_minutes
    
    # Log fingerprint with lookback computation
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"📊 compute_budget({stage}): max_lookback={max_lookback:.1f}m, "
            f"n_features={len(feature_list)}, fingerprint={set_fingerprint}"
        )
    
    return (
        LeakageBudget(
            interval_minutes=interval_minutes,
            horizon_minutes=horizon_minutes,
            max_feature_lookback_minutes=max_lookback
        ),
        set_fingerprint,
        order_fingerprint
    )


def compute_feature_lookback_max(
    feature_names: List[str],
    interval_minutes: Optional[float] = None,
    max_lookback_cap_minutes: Optional[float] = None,
    horizon_minutes: Optional[float] = None,
    registry: Optional[Any] = None,
    expected_fingerprint: Optional[str] = None,
    stage: str = "unknown"
) -> LookbackResult:
    """
    Legacy wrapper for compute_budget() to maintain backward compatibility.
    
    This function is DEPRECATED. New code should use compute_budget() directly.
    
    Returns:
        LookbackResult dataclass
    """
    if not feature_names or interval_minutes is None or interval_minutes <= 0:
        set_fp, order_fp = _compute_feature_fingerprint([], set_invariant=True)
        return LookbackResult(
            max_minutes=None,
            top_offenders=[],
            fingerprint=set_fp,
            order_fingerprint=order_fp
        )
    
    # Use default horizon if not provided
    if horizon_minutes is None:
        horizon_minutes = 60.0  # Default 1 hour
    
    # Compute budget (returns tuple now)
    budget, set_fingerprint, order_fingerprint = compute_budget(
        feature_names,
        interval_minutes,
        horizon_minutes,
        registry=registry,
        max_lookback_cap_minutes=max_lookback_cap_minutes,
        expected_fingerprint=expected_fingerprint,
        stage=stage
    )
    
    # Build top offenders list (for backward compatibility)
    # CRITICAL: Only include features from the ACTUAL feature_names list (current feature set)
    # STRICT: Build lookback dict ONLY for features in feature_names (no registry-wide fallback)
    top_offenders = []
    max_lookback = budget.max_feature_lookback_minutes if budget.max_feature_lookback_minutes > 0 else None
    
    # Build lookback for ALL features in the current feature set
    # CRITICAL: Only iterate over feature_names (the passed list), not any global registry
    feature_lookbacks = []
    for feat_name in feature_names:
        spec_lookback = None
        if registry is not None:
            try:
                metadata = registry.get_feature_metadata(feat_name)
                lag_bars = metadata.get('lag_bars')
                if lag_bars is not None and lag_bars >= 0:
                    spec_lookback = float(lag_bars * interval_minutes)
            except Exception:
                pass
        
        lookback = infer_lookback_minutes(
            feat_name,
            interval_minutes,
            spec_lookback_minutes=spec_lookback,
            registry=registry
        )
        
        if lookback != float("inf"):
            feature_lookbacks.append((feat_name, lookback))
    
    # Sort by lookback (descending)
    feature_lookbacks.sort(key=lambda x: x[1], reverse=True)
    
    # SANITY CHECK: Verify max_lookback matches actual max from feature set
    # CRITICAL: Only warn about mismatch if expected_fingerprint is provided (invariant-checked stage)
    # For earlier stages (pre-filter), mismatch is expected and not an error
    if feature_lookbacks and max_lookback is not None:
        actual_max = feature_lookbacks[0][1]
        # Apply cap if provided (to match what was applied to max_lookback)
        if max_lookback_cap_minutes is not None and actual_max > max_lookback_cap_minutes:
            actual_max = max_lookback_cap_minutes
        
        # Only warn if this is an invariant-checked stage (expected_fingerprint provided)
        # AND there's a real mismatch (not just floating point noise)
        if expected_fingerprint is not None and abs(actual_max - max_lookback) > 1.0:
            logger.error(
                f"🚨 Lookback mismatch (invariant violation): reported max={max_lookback:.1f}m but actual max from features={actual_max:.1f}m. "
                f"This indicates lookback computed on different feature set than expected (stage={stage})."
            )
    
    # Build top_offenders STRICTLY from feature_names (no registry-wide fallback)
    # CRITICAL: Only include features that are in the passed feature_names list
    # This ensures top_offenders reflects the ACTUAL feature set, not a global registry
    effective_cap = max_lookback_cap_minutes if max_lookback_cap_minutes is not None else float("inf")
    feature_names_set = set(feature_names)  # For fast lookup
    
    # STRICT: Build top_offenders only from features in feature_names
    # feature_lookbacks is already built from feature_names, but add explicit check for safety
    for feat_name, lookback in feature_lookbacks:
        # STRICT: Only include if feature is in the passed feature_names list
        # This is redundant since feature_lookbacks is built from feature_names, but ensures correctness
        if feat_name not in feature_names_set:
            continue  # Skip features not in current feature set (should never happen, but safety check)
        
        # Only include if within effective cap
        if lookback <= effective_cap:
            # If we have a max_lookback, only show features that are close to it (within 10% or top 10)
            if max_lookback is None or lookback >= max_lookback * 0.9 or len(top_offenders) < 10:
                top_offenders.append((feat_name, lookback))
    
    # Final sanity check: Verify all top_offenders are in feature_names
    top_feature_names = {f for f, _ in top_offenders}
    if not top_feature_names.issubset(feature_names_set):
        missing = top_feature_names - feature_names_set
        logger.error(
            f"🚨 CRITICAL: top_offenders contains features not in feature_names: {missing}. "
            f"This indicates a bug in top_offenders construction."
        )
        # Filter out invalid features
        top_offenders = [(f, l) for f, l in top_offenders if f in feature_names_set]
    
    # Log fingerprint with lookback computation
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"📊 compute_feature_lookback_max({stage}): max_lookback={max_lookback:.1f}m, "
            f"n_features={len(feature_names)}, fingerprint={set_fingerprint}"
        )
    
    # Return LookbackResult dataclass
    return LookbackResult(
        max_minutes=max_lookback,
        top_offenders=top_offenders[:10],
        fingerprint=set_fingerprint,
        order_fingerprint=order_fingerprint
    )
