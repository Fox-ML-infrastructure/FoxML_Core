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
from dataclasses import dataclass, replace
from typing import Iterable, Optional, Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

# Warning deduplication: track (feature_prefix, stage, warning_code) pairs to avoid spam
# Module-level cache persists across runs in same process - use reset_warning_cache() to clear between runs
_warning_cache: Dict[Tuple[str, str, str], bool] = {}

# Aggregated warning tracking: collect features with registry_zero warnings per stage
_registry_zero_aggregate: Dict[str, List[str]] = {}  # stage -> list of feature names

# Canonical lookback map cache: keyed by (featureset_fingerprint, interval_minutes, policy_flags)
_canonical_lookback_cache: Dict[Tuple[str, float, str], Dict[str, float]] = {}

# Budget cache: keyed by (featureset_fingerprint, interval_minutes, horizon_minutes, cap_minutes, stage)
# Reduces log noise by caching budget computation results
_budget_cache: Dict[Tuple[str, float, float, Optional[float], str], Tuple[Any, str, str]] = {}


def _get_feature_prefix(feature_name: str) -> str:
    """Extract feature prefix for warning deduplication (e.g., 'rsi' from 'rsi_30' or 'bb_upper' from 'bb_upper_20').
    
    Handles:
    - Simple: rsi_30 -> rsi
    - Compound (known): bb_upper_20 -> bb_upper, stoch_k_21 -> stoch_k
    - Compound (generic): rsi_wilder_14 -> rsi_wilder, bb_percent_b_20 -> bb_percent_b
    """
    parts = feature_name.split('_')
    if len(parts) < 2:
        return feature_name
    
    # Known compound patterns: bb_*, macd_*, stoch_*
    if len(parts) >= 3:
        if parts[0] in ['bb', 'macd', 'stoch']:
            # For bb_percent_b_20, we want bb_percent_b (first 3 parts)
            # For bb_upper_20, we want bb_upper (first 2 parts)
            # Check if it's a 4-part name (bb_percent_b_20)
            if len(parts) >= 4 and parts[0] == 'bb' and parts[1] == 'percent':
                return f"{parts[0]}_{parts[1]}_{parts[2]}"
            # Otherwise use first 2 parts
            return f"{parts[0]}_{parts[1]}"
        # Generic compound: rsi_wilder_14 -> rsi_wilder
        # Extract everything except the last numeric part
        if len(parts) >= 3 and parts[-1].isdigit():
            return '_'.join(parts[:-1])
    
    # Simple: rsi_30 -> rsi
    return parts[0]


def reset_warning_cache():
    """Reset the warning cache. Call this between runs/experiments to avoid suppressing warnings across runs.
    
    CRITICAL: Call at the OUTERMOST unit of work, NOT in inner loops.
    
    Good places (outermost scope):
    - Start of target evaluation (per target) - e.g., in evaluate_target_predictability()
    - Start of feature selection run - e.g., in select_features_for_target()
    - Start of rank_targets() for each target
    
    Bad places (inner loops - will re-spam):
    - Inside per-symbol loops
    - Inside per-timestamp loops
    - Inside per-fold loops
    
    This ensures warnings from one run don't suppress distinct issues in the next run.
    Module-level cache persists across runs in the same process.
    """
    global _warning_cache, _registry_zero_aggregate, _canonical_lookback_cache
    _warning_cache.clear()
    _registry_zero_aggregate.clear()
    _canonical_lookback_cache.clear()


def _log_warning_once(feature_name: str, stage: str, warning_code: str, message: str):
    """Log a warning once per (feature_prefix, stage, warning_code) pair to avoid spam.
    
    For 'registry_zero' warnings, aggregates features and logs once per stage with sample.
    
    Args:
        feature_name: Full feature name
        stage: Stage name (e.g., 'POST_GATEKEEPER', 'canonical_map')
        warning_code: Type of warning ('registry_zero', 'spec_zero', 'pattern_mismatch') - REQUIRED for granularity
        message: Warning message to log
    
    Note: warning_code is mandatory (not optional) to ensure proper deduplication granularity.
    Different warning codes for the same feature prefix are logged separately.
    """
    global _registry_zero_aggregate
    
    # Special handling for registry_zero: aggregate and log once per stage
    if warning_code == 'registry_zero':
        if stage not in _registry_zero_aggregate:
            _registry_zero_aggregate[stage] = []
        if feature_name not in _registry_zero_aggregate[stage]:
            _registry_zero_aggregate[stage].append(feature_name)
        
        # Log aggregated warning once per stage (after collecting all features)
        # Use a special key to track if we've logged the aggregate for this stage
        aggregate_key = (f"__aggregate__{stage}", stage, warning_code)
        if aggregate_key not in _warning_cache:
            _warning_cache[aggregate_key] = True
            n_features = len(_registry_zero_aggregate[stage])
            sample = _registry_zero_aggregate[stage][:5]
            sample_str = ', '.join(sample)
            if n_features > 5:
                sample_str += f", ... ({n_features - 5} more)"
            logger.warning(
                f"⚠️ Registry lag_bars=0 for {n_features} indicator-period features in stage '{stage}'; "
                f"falling back to inference. Sample: [{sample_str}]"
            )
        return  # Don't log individual message for registry_zero
    
    # For other warning codes, use standard deduplication
    prefix = _get_feature_prefix(feature_name)
    key = (prefix, stage, warning_code)
    if key not in _warning_cache:
        _warning_cache[key] = True
        logger.warning(message)


def _feat_key(f) -> str:
    """
    Normalize feature key to canonical string representation.
    
    This ensures consistent key matching across different feature representations
    (strings, objects with .name attribute, etc.).
    """
    if hasattr(f, "name"):
        return str(f.name)
    return str(f)


def _is_indicator_period_feature(name: str) -> bool:
    """Check if feature name matches indicator-period pattern (e.g., rsi_30, stoch_d_21, bb_upper_20).
    
    This is used to detect when registry lag_bars=0 is likely incorrect metadata
    and should be ignored in favor of pattern-based inference.
    
    Handles both simple patterns (rsi_30) and compound patterns (bb_upper_20, stoch_k_fast_21).
    """
    # Simple indicator-period patterns: rsi_30, cci_30, stoch_d_21, roc_50, etc.
    simple_patterns = [
        r'^(stoch_d|stoch_k|williams_r)_(\d+)$',
        r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var|roc)_(\d+)$',
        r'^(ret|sma|ema|vol)_(\d+)$',
    ]
    for pattern in simple_patterns:
        if re.match(pattern, name, re.I):
            return True
    
    # Compound indicator patterns: bb_upper_20, bb_lower_20, bb_width_20, bb_percent_b_20
    # Pattern: indicator_component_period where component is optional
    compound_patterns = [
        r'^bb_(upper|lower|width|percent_b|middle)_(\d+)$',
        r'^macd_(signal|hist|diff)_(\d+)$',  # For future variants like macd_signal_12
        r'^(stoch_k|stoch_d|rsi|cci|mfi|atr|adx|mom|std|var)_(fast|slow|wilder|smooth|upper|lower|width)_(\d+)$',
    ]
    for pattern in compound_patterns:
        if re.match(pattern, name, re.I):
            return True
    
    # Fallback: Check if name ends with _<digits> and contains known indicator prefixes
    # This catches variants like rsi_wilder_14, stoch_k_fast_21, etc.
    if re.search(r'_(rsi|cci|mfi|atr|adx|macd|bb|stoch|williams|mom|std|var|sma|ema|ret|vol)', name, re.I):
        if re.search(r'_(\d+)$', name):
            return True
    
    return False


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

# OHLCV base columns - should have 1 bar lookback (current bar only)
OHLCV_BASE_COLUMNS = {
    "open", "high", "low", "close", "volume", "vwap",
    "adj_open", "adj_high", "adj_low", "adj_close", "adj_volume",
    "o", "h", "l", "c", "v",  # Short names
}

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
    "hour_of_day",  # Time-of-day feature (0 lookback - known at time t)
    "minute_of_hour",  # Minute-of-hour feature (0 lookback)
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
        max_feature_lookback_minutes: Actual maximum feature lookback in minutes (uncapped)
        cap_max_lookback_minutes: Optional config cap (e.g., 100m) - separate from actual
        allowed_max_lookback_minutes: Derived from purge (purge - buffer) - separate from actual
    
    Properties:
        required_gap_minutes: Conservative gap required between train/test
                            (max_feature_lookback_minutes + horizon_minutes)
    """
    interval_minutes: float
    horizon_minutes: float
    max_feature_lookback_minutes: float  # Actual max from features (uncapped)
    cap_max_lookback_minutes: Optional[float] = None  # Optional config cap (e.g., 100m)
    allowed_max_lookback_minutes: Optional[float] = None  # Derived from purge: purge - buffer

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
    canonical_lookback_map: Optional[Dict[str, float]] = None  # NEW: Store canonical map for reuse


def infer_lookback_minutes(
    feature_name: str,
    interval_minutes: float,
    spec_lookback_minutes: Optional[float] = None,
    registry: Optional[Any] = None,
    unknown_policy: str = "conservative",  # "conservative" or "drop"
    feature_time_meta: Optional[Any] = None  # NEW: Optional FeatureTimeMeta for per-feature interval
) -> float:
    """
    Infer feature lookback in minutes from feature name and metadata.
    
    Precedence order (highest to lowest):
    1. Explicit spec_lookback_minutes (from registry/schema metadata)
    2. FeatureTimeMeta.lookback_minutes or lookback_bars (if provided)
    3. Calendar features whitelist (0m lookback)
    4. Explicit time suffixes (_15m, _24h, _1d)
    5. Bar-based patterns (ret_288, sma_20, etc.) - uses per-feature interval if available
    6. Keyword heuristics (daily patterns, etc.)
    7. Unknown policy (conservative default or drop)
    
    Args:
        feature_name: Feature name to analyze
        interval_minutes: Data bar interval in minutes (fallback if feature_time_meta not provided)
        spec_lookback_minutes: Explicit lookback from registry/schema (highest priority)
        registry: Optional feature registry for metadata lookup
        unknown_policy: "conservative" (default 1440m) or "drop" (return inf)
        feature_time_meta: Optional FeatureTimeMeta for per-feature interval and lookback
    
    Returns:
        Lookback in minutes (float('inf') if unknown_policy="drop" and cannot infer)
    
    Note: If feature_time_meta is provided, uses feature_time_meta.native_interval_minutes
    for scaling bar-based patterns. Otherwise uses interval_minutes (backward compatible).
    """
    # NEW: Use per-feature interval from FeatureTimeMeta if available
    # This enables multi-interval support (features with different native intervals)
    effective_interval_minutes = interval_minutes
    if feature_time_meta is not None:
        if feature_time_meta.native_interval_minutes is not None:
            effective_interval_minutes = feature_time_meta.native_interval_minutes
        # If FeatureTimeMeta has explicit lookback, use it (highest priority after spec_lookback_minutes)
        if spec_lookback_minutes is None:
            if feature_time_meta.lookback_minutes is not None:
                spec_lookback_minutes = feature_time_meta.lookback_minutes
            elif feature_time_meta.lookback_bars is not None:
                spec_lookback_minutes = feature_time_meta.lookback_bars * effective_interval_minutes
    
    # DEBUG: Track execution path for known offenders (only if log_mode=debug)
    # Check log_mode config to determine if we should log per-feature details
    log_mode = "summary"  # Default
    try:
        from CONFIG.config_loader import get_cfg
        log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
    except Exception:
        pass
    
    debug_offenders = ['cci_30', 'rsi_30', 'rsi_21', 'stoch_d_21', 'stoch_k_21', 'mfi_21', 'williams_r_21']
    debug_mode = (log_mode == "debug") and (feature_name in debug_offenders)
    
    # 1) Schema/registry metadata wins (highest priority)
    # CRITICAL: If spec_lookback is 0.0 for an indicator-period feature or _Xd feature, this is likely incorrect metadata
    # Fall through to pattern matching instead of returning 0.0
    # Aggregate spec_zero warnings (same as registry_zero)
    if spec_lookback_minutes is not None:
        is_xd_feature = bool(re.search(r'_\d+d$', feature_name, re.I))
        if spec_lookback_minutes == 0.0 and (_is_indicator_period_feature(feature_name) or is_xd_feature):
            feature_type = "_Xd feature" if is_xd_feature else "indicator-period feature"
            # Use aggregated warning (same bucket as registry_zero)
            _log_warning_once(
                feature_name, 'infer_lookback', 'registry_zero',  # Use same warning_code for aggregation
                f"⚠️ infer_lookback_minutes({feature_name}): spec_lookback_minutes=0.0 for {feature_type}. "
                f"This is likely incorrect metadata. Falling through to pattern matching."
            )
            # Don't return 0.0 - fall through to pattern matching
        else:
            if debug_mode:
                logger.debug(f"   infer_lookback_minutes({feature_name}): using spec_lookback_minutes={spec_lookback_minutes}")
            return float(spec_lookback_minutes)
    
    # Try registry if available
    if registry is not None:
        try:
            metadata = registry.get_feature_metadata(feature_name)
            lag_bars = metadata.get('lag_bars')
            if lag_bars is not None and lag_bars >= 0:
                result = float(lag_bars * interval_minutes)
                # GUARD: If registry returns 0.0 for an indicator-period feature or _Xd feature, this is suspicious
                # Don't return 0.0 - fall through to pattern matching instead
                is_xd_feature = bool(re.search(r'_\d+d$', feature_name, re.I))
                if result == 0.0 and (_is_indicator_period_feature(feature_name) or is_xd_feature):
                    feature_type = "_Xd feature" if is_xd_feature else "indicator-period feature"
                    _log_warning_once(
                        feature_name, 'infer_lookback', 'registry_zero',
                        f"⚠️ infer_lookback_minutes({feature_name}): Registry returned lag_bars=0 → result=0.0m. "
                        f"This is likely incorrect metadata for {feature_type}. "
                        f"Falling through to pattern matching."
                    )
                    # Don't return 0.0 - fall through to pattern matching
                elif result > 0:
                    if debug_mode:
                        logger.info(f"   infer_lookback_minutes({feature_name}): using registry lag_bars={lag_bars} → {result}m")
                    return result
                # If result is 0.0 and NOT an indicator-period or _Xd feature, return it (might be correct for some features)
                else:
                    if debug_mode:
                        logger.debug(f"   infer_lookback_minutes({feature_name}): registry returned 0.0m (not indicator-period/_Xd, might be correct)")
                    return result
        except Exception as e:
            if debug_mode:
                logger.debug(f"   infer_lookback_minutes({feature_name}): registry lookup failed: {e}, falling through to pattern matching")
            pass  # Fall through to pattern matching
    
    # 2a) OHLCV base columns - should have 1 bar lookback (current bar only)
    if feature_name.lower() in OHLCV_BASE_COLUMNS:
        return float(interval_minutes)  # 1 bar = interval_minutes
    
    # 2b) True "calendar/exogenous" features should be 0 lookback
    # CRITICAL: Check calendar patterns BEFORE any daily/24h heuristics to prevent misclassification
    if feature_name in CALENDAR_FEATURES:
        return 0.0
    
    # Calendar/seasonality feature patterns (comprehensive list)
    # These are deterministic from timestamp and have 0m lookback
    # CRITICAL: Must come BEFORE daily/24h heuristics to prevent misclassification
    CALENDAR_ZERO_PATTERNS = [
        r'^day_of_week$',
        r'^_weekday$',
        r'^wd_\d+$',  # weekday dummies: wd_0, wd_1, etc.
        r'^day_of_month$',
        r'^trading_day_of_month$',
        r'^trading_day_of_quarter$',
        r'^month_of_year$',
        r'^quarter$',
        r'^year$',
        r'^hour_of_day$',  # hour_of_day (time-of-day feature, 0 lookback)
        r'^minute_of_hour$',  # minute_of_hour (time-of-day feature, 0 lookback)
        r'^_(day|month|quarter|year|hour)$',  # _day, _month, _quarter, _year, _hour
        r'^weekly_seasonality_.*$',  # weekly_seasonality_friday, etc.
        r'^quarterly_seasonality_.*$',
        r'^(pre_|post_)?holiday_dummy$',
        r'^(pre_|post_)?holiday_effect$',  # If they're dummies (check if they need explicit metadata)
        r'^hour_x_.*$',  # hour_x_volume, hour_x_* (hour-based features)
        r'^.*_year$',  # *_year (ends with _year)
        r'^.*_month$',  # *_month (ends with _month, but not indicator-period like rsi_30)
        r'^.*_hour$',  # *_hour (ends with _hour)
    ]
    for pattern in CALENDAR_ZERO_PATTERNS:
        if re.match(pattern, feature_name, re.I):
            # Additional guard: don't match if it's an indicator-period feature (e.g., rsi_30_monthly_avg)
            # But do match pure calendar features (day_of_week, _year, etc.)
            # If pattern ends with _year/_month/_hour, only match if NOT an indicator-period feature
            if pattern.endswith('_year$') or pattern.endswith('_month$') or pattern.endswith('_hour$'):
                # Check if it's an indicator-period feature first
                if _is_indicator_period_feature(feature_name):
                    continue  # Skip - it's an indicator, not a calendar feature
            if debug_mode:
                logger.debug(f"   infer_lookback_minutes({feature_name}): matched calendar pattern {pattern} → 0.0m")
            return 0.0
    
    # Additional calendar keywords (before keyword heuristics)
    # CRITICAL: Don't match indicator-period features (e.g., cci_30 should NOT match "is_month_end")
    # Only match if it's actually a calendar feature name, not an indicator with period
    calendar_keywords = ['day_of_week', 'holiday', 'trading_day', 'weekday', 'is_weekend', 'is_month_end']
    if any(cal in feature_name.lower() for cal in calendar_keywords) and not re.search(r'_\d+$', feature_name):
        if debug_mode:
            logger.debug(f"   infer_lookback_minutes({feature_name}): matched calendar keyword → 0.0m")
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
    
    # Day-based patterns (e.g., _1d, _3d, _60d, _20d)
    # CRITICAL: This must match ANY feature ending with _Nd (e.g., price_momentum_60d, volatility_20d)
    days_match = re.search(r'_(\d+(?:\.\d+)?)(d|D)(?!\d)', feature_name)
    if days_match:
        val = float(days_match.group(1))
        lookback = val * 1440.0
        if debug_mode:
            logger.debug(
                f"   infer_lookback_minutes({feature_name}): matched pattern _{int(val)}d → {lookback:.0f}m"
            )
        return lookback
    
    # 4) Parse indicator-period patterns (e.g., rsi_30, cci_30, stoch_d_21, roc_50) as bars*interval
    # CRITICAL: This must come before generic numeric suffix to catch indicator-period features
    # Known indicator families with period suffixes (period = bars, not minutes)
    # Pattern handles both simple (rsi_30) and compound (stoch_d_21, williams_r_14) indicator names
    indicator_patterns = [
        # Compound indicators (with underscore in name): stoch_d_21, stoch_k_21, williams_r_14
        r'^(stoch_d|stoch_k|williams_r)_(\d+)$',
        # Simple indicators: rsi_30, cci_30, mfi_21, roc_50
        r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var|roc)_(\d+)$',
        # Other common patterns: ret_288, sma_20, ema_12, vol_30
        r'^(ret|sma|ema|vol)_(\d+)$',
    ]
    
    for pattern in indicator_patterns:
        match = re.match(pattern, feature_name, re.I)  # Case-insensitive
        if match:
            bars = int(match.group(2))
            # Only treat as bars if it's plausibly a window size (>= 2)
            if bars >= 2:
                lookback = bars * float(effective_interval_minutes)  # Use per-feature interval
                if lookback > 0:  # Ensure we got a valid result
                    # DEBUG: Log successful indicator-period match for known offenders
                    if debug_mode:
                        logger.debug(
                            f"   infer_lookback_minutes({feature_name}): matched pattern {pattern} → "
                            f"{bars} bars * {interval_minutes}m = {lookback}m"
                        )
                    return lookback
                else:
                    # This shouldn't happen, but log if it does
                    if debug_mode:
                        logger.error(
                            f"🚨 BUG: {feature_name} matched pattern but lookback=0 (bars={bars}, interval={interval_minutes})"
                        )
        elif debug_mode:
            logger.debug(f"   infer_lookback_minutes({feature_name}): pattern {pattern} did not match")
    
    # 4b) Generic numeric suffix fallback (only if no indicator pattern matched)
    # This is less reliable, so we only use it if the number is >= 2
    generic_suffix_match = re.search(r'_(\d+)$', feature_name)
    if generic_suffix_match:
        bars = int(generic_suffix_match.group(1))
        # Only treat as bars if it's plausibly a window size (>= 2)
        if bars >= 2:
            lookback = bars * float(effective_interval_minutes)  # Use per-feature interval
            if lookback > 0:  # Ensure we got a valid result
                return lookback
    
    # 5) Keyword heuristics (fallback only if no explicit suffix found)
    # CRITICAL: Check calendar patterns FIRST to prevent misclassification
    # Calendar features should have already been caught above, but double-check here
    # before falling into daily/24h heuristic
    calendar_check_patterns = [
        r'^day_of_week$', r'^_weekday$', r'^wd_\d+$',
        r'^day_of_month$', r'^trading_day_of_month$', r'^trading_day_of_quarter$',
        r'^month_of_year$', r'^quarter$', r'^year$',
        r'^_(day|month|quarter|year|hour)$',
        r'^weekly_seasonality_.*$', r'^quarterly_seasonality_.*$',
        r'^(pre_|post_)?holiday_dummy$', r'^(pre_|post_)?holiday_effect$',
        r'^hour_x_.*$', r'^.*_year$', r'^.*_month$', r'^.*_hour$',
    ]
    for cal_pattern in calendar_check_patterns:
        if re.match(cal_pattern, feature_name, re.I):
            # Additional guard: don't match if it's an indicator-period feature
            if cal_pattern.endswith('_year$') or cal_pattern.endswith('_month$') or cal_pattern.endswith('_hour$'):
                if _is_indicator_period_feature(feature_name):
                    break  # Skip - it's an indicator, not a calendar feature
            if debug_mode:
                logger.debug(f"   infer_lookback_minutes({feature_name}): matched calendar pattern {cal_pattern} → 0.0m (before daily heuristic)")
            return 0.0
    
    # Explicit daily patterns (ends with _1d, _24h, starts with daily_, etc.)
    # Only match if NOT a calendar feature (calendar features already returned 0.0 above)
    if (re.search(r'_1d$|_1D$|_24h$|_24H$|^daily_|_daily$|_1440m|1440(?!\d)', feature_name, re.I) or
        re.search(r'rolling.*daily|daily.*high|daily.*low', feature_name, re.I) or
        re.search(r'volatility.*day|vol.*day|volume.*day', feature_name, re.I)):
        # Explicit daily patterns (NOT calendar features - those already returned 0.0)
        return 1440.0
    
    # Calendar features (monthly, quarterly, yearly) - these are NOT rolling windows
    # CRITICAL: Don't match indicator-period features (e.g., cci_30 should NOT match "quarterly")
    # Only match if it's actually a calendar feature name, not an indicator with period
    if re.search(r'monthly|quarterly|yearly', feature_name, re.I) and not re.search(r'_\d+$', feature_name):
        # These are calendar features, not rolling windows - should be 0m
        # But if they're used as rolling aggregations, they need lookback
        # For now, be conservative and return 0m (they're exogenous)
        if debug_mode:
            logger.debug(f"   infer_lookback_minutes({feature_name}): matched calendar keyword → 0.0m")
        return 0.0
    
    # 6) Unknown feature policy
    # CRITICAL: If we can't infer lookback and unknown_policy is "drop", return inf
    # Otherwise, we should NOT silently return 0.0 for unknown features - that's dangerous
    if unknown_policy == "drop":
        return float("inf")  # Caller will drop
    
    # If we get here, we couldn't infer lookback and unknown_policy is "conservative"
    # For safety, we should NOT return 0.0 (that would allow leakage)
    # Instead, return conservative default (1440m = 1 day) OR raise if in strict mode
    # For now, return conservative default but log a warning for known offenders
    if debug_mode:
        logger.error(
            f"🚨 CRITICAL: Known indicator-period feature {feature_name} could not be resolved. "
            f"interval_minutes={interval_minutes}. This indicates a bug in pattern matching. "
            f"All patterns were tried but none matched."
        )
        # For known offenders, try to extract period from name as last resort
        # Pattern: indicator_period or indicator_compound_period
        last_underscore_match = re.search(r'_(\d+)$', feature_name)
        if last_underscore_match and effective_interval_minutes > 0:
            bars = int(last_underscore_match.group(1))
            if bars >= 2:
                lookback = bars * float(effective_interval_minutes)  # Use per-feature interval
                logger.warning(
                    f"   Fallback: Using last numeric suffix as period: {feature_name} → {bars} bars * {effective_interval_minutes}m = {lookback}m"
                )
                return lookback
        # If fallback also fails, this is a critical bug
        logger.error(
            f"🚨 CRITICAL: Fallback also failed for {feature_name}. Cannot compute lookback. "
            f"This will cause incorrect budget computation."
        )
    
    # HARD-FAIL: If this is an indicator-period feature but we computed 0.0, raise immediately
    # This prevents silent degradation if a new naming variant slips through
    if _is_indicator_period_feature(feature_name):
        # Get the patterns that were tried for error message
        all_patterns = [
            r'^(stoch_d|stoch_k|williams_r)_(\d+)$',
            r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var)_(\d+)$',
            r'^(ret|sma|ema|vol)_(\d+)$',
        ]
        patterns_tried = ', '.join([p.replace('^', '').replace('$', '') for p in all_patterns])
        raise ValueError(
                    f"🚨 HARD-FAIL: Indicator-period feature '{feature_name}' computed lookback=0.0m. "
                    f"This indicates a pattern matching failure. "
                    f"Feature matches _is_indicator_period_feature() but no pattern resolved it. "
                    f"Patterns tried: {patterns_tried}. "
                    f"interval_minutes={effective_interval_minutes} (effective, from feature_time_meta if provided). "
                    f"This is a bug - pattern detection must be expanded or feature name normalized."
        )
    
    # Conservative default (1440m = 1 day)
    # But for known offenders, we should have caught them earlier - log if we get here
    if debug_mode:
        logger.error(
            f"🚨 CRITICAL: {feature_name} fell through to conservative default (1440m). "
            f"This should not happen for known indicator-period features. "
            f"Pattern matching should have caught it."
        )
    return 1440.0


def compute_budget(
    final_feature_names: Iterable[str],
    interval_minutes: float,
    horizon_minutes: float,
    registry: Optional[Any] = None,
    max_lookback_cap_minutes: Optional[float] = None,
    unknown_policy: str = "conservative",
    expected_fingerprint: Optional[str] = None,
    stage: str = "unknown",
    canonical_lookback_map: Optional[Dict[str, float]] = None,  # Optional pre-computed canonical map (the truth)
    feature_time_meta_map: Optional[Dict[str, Any]] = None,  # NEW: Optional map of feature_name -> FeatureTimeMeta
    base_interval_minutes: Optional[float] = None  # NEW: Base training grid interval (for defaulting native_interval)
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
    
    # CRITICAL: Check budget cache first (reduce log noise from repeated compute_budget calls)
    # Cache key includes fingerprint, interval, horizon, cap, and stage
    # This prevents recomputing budget for the same featureset at the same stage
    cache_key = (set_fingerprint, interval_minutes, horizon_minutes, max_lookback_cap_minutes, stage)
    global _budget_cache
    if cache_key in _budget_cache and canonical_lookback_map is None:
        # Cache hit - reuse budget (only if no canonical map passed, as that might be different)
        cached_budget, cached_fp, cached_order_fp = _budget_cache[cache_key]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"📋 Budget cache hit ({stage}): fingerprint={set_fingerprint[:8]}, "
                f"max_lookback={cached_budget.max_feature_lookback_minutes:.1f}m"
            )
        return cached_budget, cached_fp, cached_order_fp
    
    # Validate fingerprint if expected (use set-invariant for comparison)
    if expected_fingerprint is not None and set_fingerprint != expected_fingerprint:
        logger.error(
            f"🚨 FINGERPRINT MISMATCH at stage={stage}: "
            f"expected={expected_fingerprint}, actual={set_fingerprint}. "
            f"This indicates lookback computed on different feature set than enforcement."
        )
    
    if not feature_list or interval_minutes <= 0:
        # If canonical map provided but empty, use it; otherwise empty dict
        if canonical_lookback_map is None:
            canonical_lookback_map = {}
        return (
            LeakageBudget(
                interval_minutes=interval_minutes,
                horizon_minutes=horizon_minutes,
                max_feature_lookback_minutes=0.0,
                cap_max_lookback_minutes=None,
                allowed_max_lookback_minutes=None
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
    
    # CRITICAL: Normalize feature keys to canonical string representation
    # Normalize feature list to canonical keys
    feature_list_normalized = [_feat_key(f) for f in final_feature_names]
    feature_set_normalized = set(feature_list_normalized)
    
    # CRITICAL: Use canonical lookback map (compute once, use everywhere)
    # If provided, use it directly (ensures no drift between budget and actual feature lookbacks)
    # Otherwise, compute it using the canonical function (same as compute_feature_lookback_max)
    map_source = "recomputed"
    if canonical_lookback_map is not None:
        # Use provided canonical map (the truth)
        # Normalize map keys to match feature keys
        feature_lookback_map = {_feat_key(k): v for k, v in canonical_lookback_map.items()}
        map_source = "passed"
        
        # VALIDATION: Check if canonical map covers all features in current list
        missing_keys = feature_set_normalized - set(feature_lookback_map.keys())
        if missing_keys:
            missing_sample = list(missing_keys)[:5]
            logger.warning(
                f"⚠️ CANONICAL MAP INCOMPLETE ({stage}): canonical map missing {len(missing_keys)} feature keys. "
                f"Sample missing: {missing_sample}. Recomputing canonical map for missing features."
            )
            # Recompute missing features
            for feat_name in missing_keys:
                spec_lookback = None
                if registry is not None:
                    try:
                        metadata = registry.get_feature_metadata(feat_name)
                        lag_bars = metadata.get('lag_bars')
                        if lag_bars is not None and lag_bars >= 0:
                            computed_spec = float(lag_bars * interval_minutes)
                            # CRITICAL: If registry returns 0.0 for indicator-period features or _Xd features, ignore it and use pattern matching
                            # Note: re is imported at module level
                            is_xd_feature = bool(re.search(r'_\d+d$', feat_name, re.I))
                            if computed_spec == 0.0 and (_is_indicator_period_feature(feat_name) or is_xd_feature):
                                feature_type = "_Xd feature" if is_xd_feature else "indicator-period feature"
                                _log_warning_once(
                                    feat_name, stage, 'registry_zero',
                                    f"⚠️ Registry returned lag_bars=0 for {feature_type} {feat_name}. "
                                    f"Ignoring and using pattern matching instead."
                                )
                                spec_lookback = None  # Don't pass 0.0 - let pattern matching handle it
                            else:
                                spec_lookback = computed_spec
                    except Exception:
                        pass
                
                lookback = infer_lookback_minutes(
                    feat_name,
                    interval_minutes,
                    spec_lookback_minutes=spec_lookback,
                    registry=registry
                )
                
                if lookback != float("inf"):
                    feature_lookback_map[feat_name] = lookback
            map_source = "passed+recomputed"
        
        # CRITICAL: Compute lookback for ALL features in final_feature_names
        # If a feature is missing from feature_lookback_map, we MUST compute it (don't default to 0.0)
        # Missing features should be treated as unknown (inf), not safe (0.0)
        lookbacks = []
        for f in final_feature_names:
            feat_key = _feat_key(f)
            if feat_key in feature_lookback_map:
                lookback = feature_lookback_map[feat_key]
                if lookback != float("inf"):
                    lookbacks.append(lookback)
            else:
                # Feature missing from canonical map - compute it now
                # This should not happen if canonical map was built correctly, but handle gracefully
                logger.warning(
                    f"⚠️ compute_budget({stage}): Feature '{f}' missing from canonical map. "
                    f"Computing lookback now."
                )
                spec_lookback = None
                if registry is not None:
                    try:
                        metadata = registry.get_feature_metadata(feat_key)
                        lag_bars = metadata.get('lag_bars')
                        if lag_bars is not None and lag_bars >= 0:
                            computed_spec = float(lag_bars * interval_minutes)
                            if computed_spec == 0.0 and _is_indicator_period_feature(feat_key):
                                spec_lookback = None
                            else:
                                spec_lookback = computed_spec
                    except Exception:
                        pass
                
                feat_meta = feature_time_meta_map.get(feat_key) if feature_time_meta_map else None
                lookback = infer_lookback_minutes(
                    feat_key,
                    interval_minutes,
                    spec_lookback_minutes=spec_lookback,
                    registry=registry,
                    feature_time_meta=feat_meta
                )
                
                if feat_meta is not None:
                    from TRAINING.ranking.utils.feature_time_meta import effective_lookback_minutes
                    effective_base = base_interval_minutes or interval_minutes
                    lookback = effective_lookback_minutes(feat_meta, effective_base, inferred_lookback_minutes=lookback)
                
                # Store in map for future use (ALL features, even if inf)
                feature_lookback_map[feat_key] = lookback
                
                if lookback != float("inf"):
                    lookbacks.append(lookback)
    else:
        # Compute canonical lookback map using the EXACT same logic as compute_feature_lookback_max
        # CRITICAL: Require valid interval_minutes for recompute path
        if interval_minutes is None or interval_minutes <= 0:
            logger.error(
                f"🚨 RECOMPUTE PATH INVALID ({stage}): interval_minutes={interval_minutes} is invalid. "
                f"Cannot recompute canonical map. This indicates a call site bug (missing/zero interval)."
            )
            raise ValueError(
                f"Cannot recompute canonical lookback map: interval_minutes={interval_minutes} is invalid. "
                f"Call site must provide valid interval_minutes or pass canonical_lookback_map."
            )
        
        # DEBUG: Log interval and feature type for recompute path
        sample_feat = list(final_feature_names)[0] if final_feature_names else None
        logger.debug(
            f"   RECOMPUTE PATH ({stage}): interval_minutes={interval_minutes}, "
            f"n_features={len(feature_list)}, sample_feat_type={type(sample_feat).__name__}"
        )
        
        lookbacks = []
        feature_lookback_map = {}
        
        for feat in final_feature_names:
            feat_name = _feat_key(feat)  # Normalize key
            # Try registry first for explicit metadata (same logic as compute_feature_lookback_max)
            spec_lookback = None
            if registry is not None:
                try:
                    metadata = registry.get_feature_metadata(feat_name)
                    lag_bars = metadata.get('lag_bars')
                    if lag_bars is not None and lag_bars >= 0:
                        computed_spec = float(lag_bars * interval_minutes)
                        # CRITICAL: If registry returns 0.0 for indicator-period features, ignore it and use pattern matching
                        # Also ignore 0.0 for _Xd features (day-suffix) - they should use pattern matching
                        # Note: re is imported at module level
                        is_xd_feature = bool(re.search(r'_\d+d$', feat_name, re.I))
                        if computed_spec == 0.0 and (_is_indicator_period_feature(feat_name) or is_xd_feature):
                            feature_type = "_Xd feature" if is_xd_feature else "indicator-period feature"
                            _log_warning_once(
                                feat_name, stage, 'registry_zero',
                                f"⚠️ RECOMPUTE PATH: Registry returned lag_bars=0 for {feature_type} {feat_name}. "
                                f"Ignoring and using pattern matching instead."
                            )
                            spec_lookback = None  # Don't pass 0.0 - let pattern matching handle it
                        else:
                            spec_lookback = computed_spec
                except Exception:
                    pass
            
            # Infer lookback using canonical function (same as compute_feature_lookback_max)
            # CRITICAL: Use the EXACT same call signature as compute_feature_lookback_max to ensure consistency
            # (compute_feature_lookback_max doesn't pass unknown_policy, so it uses default "conservative")
            # NEW: Get FeatureTimeMeta for this feature if available
            feat_meta = feature_time_meta_map.get(feat_name) if feature_time_meta_map else None
            lookback = infer_lookback_minutes(
                feat_name,
                interval_minutes,  # Fallback interval (used if feat_meta.native_interval_minutes is None)
                spec_lookback_minutes=spec_lookback,
                registry=registry,
                feature_time_meta=feat_meta  # NEW: Pass per-feature metadata
                # NOTE: Don't pass unknown_policy here - use default "conservative" to match compute_feature_lookback_max
                # The unknown_policy parameter in compute_budget() is only for filtering (drop vs keep), not for lookback calculation
            )
            
            # NEW: If FeatureTimeMeta is available, compute effective lookback (lookback + embargo)
            if feat_meta is not None:
                from TRAINING.ranking.utils.feature_time_meta import effective_lookback_minutes
                effective_base = base_interval_minutes or interval_minutes
                lookback = effective_lookback_minutes(feat_meta, effective_base, inferred_lookback_minutes=lookback)
            
            # CRITICAL: Store lookback for ALL features (even if inf), so canonical map is complete
            # This ensures every feature has an entry (inf for unknown, not missing)
            feature_lookback_map[feat_name] = lookback
            
            # Only add to lookbacks list if not inf (for max calculation)
            # But feature_lookback_map must have ALL features for gatekeeper to check
            if lookback != float("inf"):
                lookbacks.append(lookback)
            
            # DEBUG: Log computed lookback for _Xd features and known offenders (only in debug mode)
            # Check log_mode config to determine if we should log per-feature details
            log_mode = "summary"  # Default
            try:
                from CONFIG.config_loader import get_cfg
                log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
            except Exception:
                pass
            
            is_xd_feature = bool(re.search(r'_\d+d$', feat_name, re.I))
            if log_mode == "debug" and (is_xd_feature or feat_name in ['rsi_30', 'cci_30', 'rsi_21', 'stoch_d_21', 'stoch_k_21', 'mfi_21', 'williams_r_21']):
                logger.debug(
                    f"   RECOMPUTE ({stage}): {feat_name} → lookback={lookback:.1f}m "
                    f"(interval={interval_minutes}, spec_lookback={spec_lookback})"
                )
                # If lookback is 0.0 for _Xd feature, this is a bug - log details
                if lookback == 0.0 and is_xd_feature:
                    logger.error(
                        f"🚨 RECOMPUTE BUG: {feat_name} is _Xd feature but computed lookback=0.0m. "
                        f"interval_minutes={interval_minutes}, spec_lookback={spec_lookback}. "
                        f"This indicates pattern matching failed."
                    )
    
    # DEBUG: Log lookback for known offenders to diagnose mismatch
    # Always log these (not just DEBUG level) to catch the 100m vs 150m mismatch
    # Use a per-stage cache to avoid spammy repeated logs
    if not hasattr(compute_budget, '_offender_log_cache'):
        compute_budget._offender_log_cache = set()
    
    # DEBUG: Log offenders watchlist (only for present features, only if not per-feature stage, only once per stage)
    # Skip logging for per-feature stages to avoid spam
    if not stage.endswith("_per_feature"):
        cache_key = f"{stage}:{set_fingerprint}"
        should_log_offenders = cache_key not in compute_budget._offender_log_cache
        
        offenders = ['cci_30', 'rsi_30', 'rsi_21', 'stoch_d_21', 'stoch_k_21', 'mfi_21', 'williams_r_21']
        found_offenders = []
        for offender in offenders:
            offender_normalized = _feat_key(offender)
            present = offender_normalized in feature_set_normalized
            # Only process if feature is actually present (don't compute/infer for absent features)
            if not present:
                continue  # Skip absent features entirely
            
            value = feature_lookback_map.get(offender_normalized, None)  # Use None, not 0.0
            if value is not None:
                # GUARD: If feature is present but lookback is 0.0, this is a bug
                if value == 0.0:
                    logger.error(
                        f"🚨 BUG: Feature {offender} is present but lookback=0.0m (map={map_source}, stage={stage}). "
                        f"This indicates a bug in lookback computation. interval_minutes={interval_minutes}."
                    )
                    # Force recompute using known-good path if we're in recompute mode
                    if map_source == "recomputed":
                        raise ValueError(
                            f"Feature {offender} present but lookback=0.0m in recompute path. "
                            f"This indicates invalid interval_minutes={interval_minutes} or lookback computation bug."
                        )
                found_offenders.append(f"{offender}={value:.1f}m(present,map={map_source})")
            else:
                found_offenders.append(f"{offender}=MISSING(present,map={map_source})")
        
        if found_offenders and should_log_offenders:
            logger.info(f"   DEBUG compute_budget({stage}) offenders: {', '.join(found_offenders)}")
            compute_budget._offender_log_cache.add(cache_key)
    
    # Compute ACTUAL max lookback (uncapped - this is the truth)
    # CRITICAL: Unknown lookback (inf) = unsafe
    # Rule: If unknowns exist, they should have been dropped/quarantined BEFORE calling compute_budget
    # This function should only be called on SAFE features (no unknowns)
    finite_lookbacks = [lb for lb in lookbacks if lb != float("inf") and lb is not None]
    
    # Check for unknown features (they should have been handled by gatekeeper/sanitizer)
    unknown_features = [f for f, lb in feature_lookback_map.items() if lb == float("inf")]
    if unknown_features:
        # CRITICAL: Stage-aware handling
        # Pre-enforcement stages (create_resolved_config, etc.) are allowed to see unknowns
        # Post-enforcement stages (POST_PRUNE, POST_GATEKEEPER, etc.) should never see unknowns
        is_pre_enforcement_stage = any(
            pre_stage in stage.lower() 
            for pre_stage in ["create_resolved_config", "pre_", "initial", "baseline"]
        )
        is_post_enforcement_stage = any(
            post_stage in stage.lower()
            for post_stage in ["post_prune", "post_gatekeeper", "post_", "gatekeeper_budget", "enforced"]
        )
        
        if is_post_enforcement_stage:
            # Post-enforcement: This is a bug - unknowns should not reach compute_budget
            policy = "strict"
            try:
                from CONFIG.config_loader import get_cfg
                policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
            except Exception:
                pass
            
            error_msg = (
                f"🚨 compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
                f"This indicates a bug: compute_budget() was called on features that should have been quarantined. "
                f"Enforcement (gatekeeper/sanitizer) should have dropped these BEFORE calling compute_budget(). "
                f"Sample: {unknown_features[:5]}"
            )
            
            if policy == "strict":
                logger.error(error_msg)
                # Hard-fail: unknowns should not reach compute_budget in post-enforcement stages
                raise RuntimeError(
                    f"{error_msg} "
                    f"(policy: strict - training blocked. Fix enforcement to quarantine unknowns before calling compute_budget)"
                )
            else:
                logger.warning(error_msg)
        elif is_pre_enforcement_stage:
            # Pre-enforcement: This is a diagnostic-only stage
            # In strict mode, we should still log at INFO (not DEBUG) to make contract visible
            # But don't hard-fail - enforcement will handle later
            policy = "strict"
            try:
                from CONFIG.config_loader import get_cfg
                policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
            except Exception:
                pass
            
            if policy == "strict":
                # In strict mode, log at INFO to make contract visible (not hidden in DEBUG)
                # This is expected in pre-enforcement, but we want visibility that unknowns exist
                logger.info(
                    f"📊 compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
                    f"This is expected in pre-enforcement stages. Enforcement (gatekeeper/sanitizer) will quarantine these. "
                    f"Sample: {unknown_features[:5]}"
                )
            else:
                # Non-strict: log at DEBUG (less noise)
                logger.debug(
                    f"📊 compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
                    f"This is expected in pre-enforcement stages. Enforcement will handle later."
                )
        else:
            # Unknown stage - be conservative and log warning (but don't hard-fail)
            logger.warning(
                f"⚠️ compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
                f"Stage '{stage}' is not clearly pre- or post-enforcement. "
                f"Enforcement (gatekeeper/sanitizer) should drop these. Sample: {unknown_features[:5]}"
            )
    
    actual_max_lookback = max(finite_lookbacks) if finite_lookbacks else 0.0
    
    # Store cap separately (don't modify actual_max)
    cap_max_lookback = max_lookback_cap_minutes
    
    # For backward compatibility, max_feature_lookback_minutes is the actual (not capped)
    # The cap is stored separately for policy decisions
    max_lookback = actual_max_lookback
    
    # SAFETY GUARD: Log if we detect potential lookback underestimation
    # This helps catch cases where lookback inference might be inconsistent
    if len(lookbacks) > 0:
        # Check for suspicious patterns (e.g., all lookbacks are exactly 100m = 20 bars * 5m)
        # This might indicate a default/fallback being used incorrectly
        # Only warn if we have enough features to be suspicious (n >= 10)
        unique_lookbacks = sorted(set(lookbacks), reverse=True)
        if len(lookbacks) >= 10 and len(unique_lookbacks) == 1 and unique_lookbacks[0] == 100.0:
            logger.warning(
                f"⚠️ SUSPICIOUS: All {len(lookbacks)} features have identical lookback=100m. "
                f"This might indicate a default/fallback being used incorrectly. "
                f"Sample features: {list(final_feature_names)[:5]}"
            )
    
    # CRITICAL: Log ALL features with lookback > 240m (or cap if set) for gatekeeper/sanitizer stages
    # This must happen AFTER feature_lookback_map is fully populated and max is computed
    # This diagnostic is essential to catch split-brain early - shows what gatekeeper/sanitizer see
    # NOTE: feature_lookback_map is guaranteed to exist here (defined in both if/else paths above)
    if stage in ["GATEKEEPER", "feature_sanitizer", "POST_GATEKEEPER_sanity_check"]:
        cap_for_logging = max_lookback_cap_minutes if max_lookback_cap_minutes is not None else 240.0
        offenders_for_logging = []
        for feat_name in final_feature_names:
            feat_key = _feat_key(feat_name)
            lookback = feature_lookback_map.get(feat_key)
            if lookback is not None and lookback != float("inf") and lookback > cap_for_logging:
                offenders_for_logging.append((feat_name, lookback))
        
        if offenders_for_logging:
            offenders_for_logging.sort(key=lambda x: x[1], reverse=True)
            logger.info(
                f"🔍 {stage} DIAGNOSTIC: {len(offenders_for_logging)} features exceed cap ({cap_for_logging:.1f}m): "
                f"{', '.join([f'{f}({l:.0f}m)' for f, l in offenders_for_logging[:10]])}"
            )
        else:
            logger.debug(
                f"🔍 {stage} DIAGNOSTIC: No features exceed cap ({cap_for_logging:.1f}m) - all features safe"
            )
    
    # Create LeakageBudget object
    budget = LeakageBudget(
        interval_minutes=interval_minutes,
        horizon_minutes=horizon_minutes,
        max_feature_lookback_minutes=max_lookback,  # Actual (uncapped) from canonical map
        cap_max_lookback_minutes=cap_max_lookback,  # Optional cap
        allowed_max_lookback_minutes=None  # Will be set by caller if purge-derived
    )
    
    # Cache budget result (for log noise reduction)
    # Only cache if no canonical map passed (canonical map might be different)
    if canonical_lookback_map is None:
        _budget_cache[cache_key] = (budget, set_fingerprint, order_fingerprint)
    
    # Log one-line summary (not per-feature details)
    # Only log if not a cache hit (cache hits logged above at DEBUG)
    # Note: Check cache BEFORE we added to it (was_cached = before this call)
    was_cached_before = cache_key in _budget_cache
    if not was_cached_before:
        # One-line summary at INFO (reduces noise)
        logger.info(
            f"📊 compute_budget({stage}): max_lookback={max_lookback:.1f}m, "
            f"n_features={len(feature_list)}, fingerprint={set_fingerprint[:8]}"
        )
    
    return budget, set_fingerprint, order_fingerprint


def compute_feature_lookback_max(
    feature_names: List[str],
    interval_minutes: Optional[float] = None,
    max_lookback_cap_minutes: Optional[float] = None,
    horizon_minutes: Optional[float] = None,
    registry: Optional[Any] = None,
    expected_fingerprint: Optional[str] = None,
    stage: str = "unknown",
    feature_time_meta_map: Optional[Dict[str, Any]] = None,  # NEW: Optional map of feature_name -> FeatureTimeMeta
    base_interval_minutes: Optional[float] = None,  # NEW: Base training grid interval (for defaulting native_interval)
    canonical_lookback_map: Optional[Dict[str, float]] = None  # NEW: Optional pre-computed canonical map (SST reuse)
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
            order_fingerprint=order_fp,
            canonical_lookback_map=None
        )
    
    # Use default horizon if not provided
    if horizon_minutes is None:
        horizon_minutes = 60.0  # Default 1 hour
    
    # CRITICAL: Compute canonical lookback map FIRST (the truth)
    # Then pass it to compute_budget() to ensure both use the same map
    # Normalize all keys to canonical string representation
    # CRITICAL: Require valid interval_minutes
    if interval_minutes is None or interval_minutes <= 0:
        logger.error(
            f"🚨 CANONICAL MAP BUILDER INVALID ({stage}): interval_minutes={interval_minutes} is invalid. "
            f"Cannot build canonical map."
        )
        raise ValueError(
            f"Cannot build canonical lookback map: interval_minutes={interval_minutes} is invalid."
        )
    
    # CRITICAL: Use passed canonical_lookback_map if provided (SST reuse)
    # Otherwise, build it from cache or compute it
    set_fp, _ = _compute_feature_fingerprint(feature_names, set_invariant=True)
    
    if canonical_lookback_map is not None:
        # Use passed map (the truth) - no cache lookup or building needed
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"📋 Using passed canonical lookback map: fingerprint={set_fp[:8]}, n_features={len(canonical_lookback_map)}")
    else:
        # Check cache first (keyed by featureset fingerprint, interval, and policy flags)
        # Policy flags: include feature_time_meta_map presence and base_interval_minutes
        policy_flags = f"meta={feature_time_meta_map is not None},base={base_interval_minutes}"
        cache_key = (set_fp, interval_minutes, policy_flags)
        
        global _canonical_lookback_cache
        if cache_key in _canonical_lookback_cache:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📋 Cache hit for canonical lookback map: fingerprint={set_fp[:8]}, interval={interval_minutes}m")
            canonical_lookback_map = _canonical_lookback_cache[cache_key].copy()
        else:
            canonical_lookback_map = {}
            
            # CRITICAL: Determine unknown_policy based on safety policy (for consistency)
            # In strict mode, unknown features should be inf (unsafe), not 1440m (conservative default)
            # This ensures gatekeeper and strict check use the same unknown policy
            unknown_policy_for_canonical = "conservative"  # Default
            try:
                from CONFIG.config_loader import get_cfg
                safety_policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                # In strict mode, use "drop" to get inf for unknown features (treat as unsafe)
                # In other modes, use "conservative" to get 1440m (backward compatibility)
                if safety_policy == "strict":
                    unknown_policy_for_canonical = "drop"  # inf = unsafe, will be quarantined
            except Exception:
                pass  # Use default "conservative" if config lookup fails
            
            # Build canonical map from scratch
            for feat in feature_names:
                feat_name = _feat_key(feat)  # Normalize key
                spec_lookback = None
                if registry is not None:
                    try:
                        metadata = registry.get_feature_metadata(feat_name)
                        lag_bars = metadata.get('lag_bars')
                        if lag_bars is not None and lag_bars >= 0:
                            computed_spec = float(lag_bars * interval_minutes)
                            # CRITICAL: If registry returns 0.0 for indicator-period features or _Xd features, ignore it and use pattern matching
                            is_xd_feature = bool(re.search(r'_\d+d$', feat_name, re.I))
                            if computed_spec == 0.0 and (_is_indicator_period_feature(feat_name) or is_xd_feature):
                                feature_type = "_Xd feature" if is_xd_feature else "indicator-period feature"
                                _log_warning_once(
                                    feat_name, stage, 'registry_zero',
                                    f"⚠️ CANONICAL MAP BUILDER: Registry returned lag_bars=0 for {feature_type} {feat_name}. "
                                    f"Ignoring and using pattern matching instead."
                                )
                                spec_lookback = None  # Don't pass 0.0 - let pattern matching handle it
                            else:
                                spec_lookback = computed_spec
                    except Exception:
                        pass
                
                # NEW: Get FeatureTimeMeta for this feature if available
                feat_meta = feature_time_meta_map.get(feat_name) if feature_time_meta_map else None
                lookback = infer_lookback_minutes(
                    feat_name,
                    interval_minutes,  # Fallback interval (used if feat_meta.native_interval_minutes is None)
                    spec_lookback_minutes=spec_lookback,
                    registry=registry,
                    unknown_policy=unknown_policy_for_canonical,  # CRITICAL: Use consistent policy
                    feature_time_meta=feat_meta  # NEW: Pass per-feature metadata
                )
                
                # NEW: If FeatureTimeMeta is available, compute effective lookback (lookback + embargo)
                if feat_meta is not None:
                    from TRAINING.ranking.utils.feature_time_meta import effective_lookback_minutes
                    effective_base = base_interval_minutes or interval_minutes
                    lookback = effective_lookback_minutes(feat_meta, effective_base, inferred_lookback_minutes=lookback)
                
                # CRITICAL: Store lookback for ALL features (even if inf), so canonical map is complete
                # This ensures every feature has an entry (None/inf for unknown, not missing)
                canonical_lookback_map[feat_name] = lookback
                
                # HARD-FAIL: If lookback is 0.0 for an indicator-period feature, raise immediately
                if lookback == 0.0 and _is_indicator_period_feature(feat_name):
                    # Get the patterns that were tried for error message
                    all_patterns = [
                        r'^(stoch_d|stoch_k|williams_r)_(\d+)$',
                        r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var)_(\d+)$',
                        r'^(ret|sma|ema|vol)_(\d+)$',
                    ]
                    patterns_tried = ', '.join([p.replace('^', '').replace('$', '') for p in all_patterns])
                    raise ValueError(
                        f"🚨 HARD-FAIL: Indicator-period feature '{feat_name}' computed lookback=0.0m. "
                        f"This indicates a pattern matching failure. "
                        f"Feature matches _is_indicator_period_feature() but no pattern resolved it. "
                        f"Patterns tried: {patterns_tried}. "
                        f"interval_minutes={interval_minutes}, spec_lookback={spec_lookback}. "
                        f"This is a bug - pattern detection must be expanded or feature name normalized."
                    )
            
            # Store in cache for future use (only if we built it, not if it was passed)
            policy_flags = f"meta={feature_time_meta_map is not None},base={base_interval_minutes}"
            cache_key = (set_fp, interval_minutes, policy_flags)
            if cache_key not in _canonical_lookback_cache:
                _canonical_lookback_cache[cache_key] = canonical_lookback_map.copy()
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"📋 Cached canonical lookback map: fingerprint={set_fp[:8]}, interval={interval_minutes}m, n_features={len(canonical_lookback_map)}")
    
    # Compute budget using the canonical map (ensures no drift)
    budget, set_fingerprint, order_fingerprint = compute_budget(
        feature_names,
        interval_minutes,
        horizon_minutes,
        registry=registry,
        max_lookback_cap_minutes=max_lookback_cap_minutes,
        expected_fingerprint=expected_fingerprint,
        stage=stage,
        canonical_lookback_map=canonical_lookback_map,  # Pass canonical map (the truth)
        feature_time_meta_map=feature_time_meta_map,  # NEW: Pass per-feature metadata
        base_interval_minutes=base_interval_minutes  # NEW: Pass base interval
    )
    
    # Log budget computation for debugging
    logger.debug(
        f"📊 compute_feature_lookback_max({stage}): budget.actual_max={budget.max_feature_lookback_minutes:.1f}m, "
        f"n_features={len(feature_names)}, fingerprint={set_fingerprint}"
    )
    
    # Build top offenders list (for backward compatibility)
    # CRITICAL: max_lookback and top_offenders MUST be derived from the EXACT same canonical map
    # Use the canonical map that was already built (single source of truth)
    # NO recomputation - use the canonical map directly
    top_offenders = []
    
    # Build lookback list from canonical map (the truth - already computed above)
    # CRITICAL: Use canonical_lookback_map, not recompute
    feature_lookbacks = []
    for feat_name in feature_names:
        feat_key = _feat_key(feat_name)  # Normalize key
        lookback = canonical_lookback_map.get(feat_key)
        
        if lookback is None:
            # Feature missing from canonical map - this should not happen
            logger.error(
                f"🚨 compute_feature_lookback_max({stage}): Feature '{feat_name}' missing from canonical map. "
                f"This indicates a bug in canonical map construction."
            )
            # Treat as unknown (inf) - unsafe
            lookback = float("inf")
        
        # Include ALL features in feature_lookbacks (even if inf) for accurate reporting
        feature_lookbacks.append((feat_name, lookback))
    
    # Sort by lookback (descending), but put inf at the end (they're unknown/unsafe)
    feature_lookbacks.sort(key=lambda x: (x[1] == float("inf"), -x[1] if x[1] != float("inf") else 0))
    
    # Compute ACTUAL max from feature_lookbacks (uncapped - this is the truth)
    # Exclude inf (unknown) from max calculation, but log warning if they exist
    finite_lookbacks = [(f, l) for f, l in feature_lookbacks if l != float("inf")]
    unknown_features = [f for f, l in feature_lookbacks if l == float("inf")]
    
    if unknown_features:
        logger.warning(
            f"⚠️ compute_feature_lookback_max({stage}): {len(unknown_features)} features have unknown lookback (inf). "
            f"These should have been dropped/quarantined. Sample: {unknown_features[:5]}"
        )
    
    actual_max_uncapped = finite_lookbacks[0][1] if finite_lookbacks else 0.0
    
    # Use the ACTUAL uncapped max for reporting (not the capped budget value)
    # The cap is for gatekeeper logic, not for reporting
    max_lookback = actual_max_uncapped if actual_max_uncapped > 0 else None
    
    # CRITICAL INVARIANT CHECK: budget.max_feature_lookback_minutes MUST match actual_max_uncapped
    # Both are computed from the SAME canonical map, so they MUST agree
    # This is the single source of truth validation
    if feature_lookbacks and budget.max_feature_lookback_minutes is not None:
        budget_actual_max = budget.max_feature_lookback_minutes  # This is the actual (uncapped) max
        budget_cap = budget.cap_max_lookback_minutes  # Optional cap
        
        # HARD-FAIL on mismatch (split-brain detection)
        if abs(actual_max_uncapped - budget_actual_max) > 1.0:
            logger.error(
                f"🚨 SPLIT-BRAIN DETECTED (compute_feature_lookback_max): "
                f"budget.max={budget_actual_max:.1f}m vs actual_max_uncapped={actual_max_uncapped:.1f}m. "
                f"Both should use the same canonical map. This indicates a bug."
            )
            # In strict mode, this is a hard-stop
            policy = "strict"
            try:
                from CONFIG.config_loader import get_cfg
                policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
            except Exception:
                pass
            
            if policy == "strict":
                raise RuntimeError(
                    f"🚨 SPLIT-BRAIN DETECTED (compute_feature_lookback_max): "
                    f"budget.max={budget_actual_max:.1f}m vs actual_max_uncapped={actual_max_uncapped:.1f}m. "
                    f"Both should use the same canonical map. Training blocked until this is fixed."
                )
        
        # Check for cap violation (actual > cap)
        if budget_cap is not None and actual_max_uncapped > budget_cap:
            exceeding_features = [(f, l) for f, l in feature_lookbacks if l > budget_cap + 1.0]
            exceeding_count = len(exceeding_features)
            
            # CRITICAL: In strict mode, this is a hard-stop (safety bug)
            # Gatekeeper/sanitizer should have dropped these features
            policy = "strict"  # Default
            try:
                from CONFIG.config_loader import get_cfg
                policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
            except Exception:
                pass
            
            error_msg = (
                f"🚨 CAP VIOLATION: actual_max={actual_max_uncapped:.1f}m > cap={budget_cap:.1f}m. "
                f"Feature set contains {exceeding_count} features exceeding cap. "
                f"Gatekeeper/sanitizer should have dropped these features. "
                f"Top offenders: {', '.join([f'{f}({l:.0f}m)' for f, l in exceeding_features[:10]])}"
            )
            
            if policy == "strict":
                raise RuntimeError(error_msg + " (policy: strict - training blocked)")
            else:
                logger.error(error_msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
        
        # Check for fingerprint/invariant violation (computed on different feature set)
        if expected_fingerprint is not None and abs(actual_max_uncapped - budget_actual_max) > 1.0:
            # This is a real mismatch - budget was computed on different features
            exceeding_features = [(f, l) for f, l in feature_lookbacks if l > budget_actual_max + 1.0]
            exceeding_features.sort(key=lambda x: x[1], reverse=True)  # Sort by lookback descending
            
            # SAFETY GUARD: Underestimating lookback can under-purge and reintroduce leakage risk
            # Use the max of both values to be safe (conservative approach)
            if actual_max_uncapped > budget_actual_max:
                logger.error(
                    f"🚨 Lookback mismatch (invariant violation): budget.actual_max={budget_actual_max:.1f}m but actual max from features={actual_max_uncapped:.1f}m. "
                    f"This indicates lookback computed on different feature set than expected (stage={stage}). "
                    f"Feature set contains {len(exceeding_features)} features with lookback > budget.actual_max."
                )
                
                # Log the top features that exceed budget.actual_max (helps debug which features are causing the mismatch)
                if exceeding_features:
                    top_exceeding = exceeding_features[:10]  # Top 10
                    logger.error(
                        f"   Top features exceeding budget.actual_max ({budget_actual_max:.1f}m): "
                        f"{', '.join([f'{f}({l:.0f}m)' for f, l in top_exceeding])}"
                    )
                
                # SAFETY: Update budget to use the higher value to prevent under-purge
                # This is a conservative safety measure - underestimating lookback is dangerous
                # Use dataclasses.replace() since budget is frozen
                logger.warning(
                    f"⚠️ SAFETY GUARD: Updating budget.actual_max from {budget_actual_max:.1f}m to {actual_max_uncapped:.1f}m "
                    f"to prevent under-purge (underestimating lookback reintroduces leakage risk)."
                )
                budget = replace(budget, max_feature_lookback_minutes=actual_max_uncapped)
            else:
                # Budget is higher than actual - this is less dangerous but still indicates inconsistency
                logger.warning(
                    f"⚠️ Lookback mismatch (budget higher): budget.actual_max={budget_actual_max:.1f}m > actual max from features={actual_max_uncapped:.1f}m. "
                    f"This might indicate stale budget or different feature set."
                )
    
    # Build top_offenders STRICTLY from feature_lookbacks (which is built from feature_names)
    # CRITICAL: max_lookback and top_offenders MUST come from the same feature_lookbacks list
    # NO filtering by cap here - show the actual top offenders from the actual feature set
    # The cap is for gatekeeper logic (dropping features), not for reporting
    feature_names_set = set(feature_names)  # For fast lookup
    
    # STRICT: Build top_offenders only from feature_lookbacks (which is from feature_names)
    # Show top 10 features by lookback, regardless of cap
    # This ensures max_lookback and top_offenders are from the same source
    for feat_name, lookback in feature_lookbacks:
        # STRICT: Only include if feature is in the passed feature_names list
        # This is redundant since feature_lookbacks is built from feature_names, but ensures correctness
        if feat_name not in feature_names_set:
            continue  # Skip features not in current feature set (should never happen, but safety check)
        
        # Include top features by lookback (no cap filtering - show reality)
        # If we have a max_lookback, show features that are close to it (within 10% or top 10)
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
    
    # Return LookbackResult dataclass with corrected budget if it was updated
    # NOTE: The corrected budget (if any) is stored in the budget variable after replace()
    # We return the actual_max_uncapped as max_minutes (the truth), not budget's potentially underestimated value
    return LookbackResult(
        max_minutes=max_lookback,  # Use actual_max_uncapped (the truth from canonical computation)
        top_offenders=top_offenders[:10],
        fingerprint=set_fingerprint,
        order_fingerprint=order_fingerprint,
        canonical_lookback_map=canonical_lookback_map.copy() if canonical_lookback_map else None  # NEW: Include canonical map for reuse
    )
