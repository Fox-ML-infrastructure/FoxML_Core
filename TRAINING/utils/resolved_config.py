"""
Resolved Configuration Object

Centralizes computation of "requested" vs "effective" values and ensures
consistent logging and reproducibility tracking.

This module provides a single source of truth for:
- min_cs (requested vs effective)
- purge/embargo derivation (single formula)
- feature counts (safe → dropped_nan → final)
- interval/horizon resolution
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import logging

from TRAINING.utils.duration_parser import (
    parse_duration,
    enforce_purge_audit_rule,
    format_duration,
    Duration,
    DurationLike
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedConfig:
    """
    Single resolved configuration object for a target evaluation.
    
    All values are computed once and logged consistently.
    
    **Duration Type Conversion Boundary:**
    Internally, all duration comparisons and computations use Duration objects
    (canonical representation). The float minutes stored here are converted at the
    boundary for backward compatibility with existing code that expects float minutes.
    
    For new code, prefer using Duration objects directly from duration_parser.
    """
    # Cross-sectional sampling
    requested_min_cs: int
    n_symbols_available: int
    effective_min_cs: int
    max_cs_samples: Optional[int]
    
    # Data configuration
    # @deprecated: Use Duration objects directly in new code. These float fields are for backward compatibility.
    interval_minutes: Optional[float]  # Converted from Duration at boundary (backward compat)
    horizon_minutes: Optional[float]  # Converted from Duration at boundary (backward compat)
    
    # Purge/embargo (single source of truth)
    # @deprecated: Use Duration objects directly in new code. These float fields are for backward compatibility.
    # NOTE: These are converted from Duration to float minutes at the boundary.
    # Internally, all comparisons use Duration objects (see enforce_purge_audit_rule).
    purge_minutes: float  # @deprecated: Use Duration objects in new code
    embargo_minutes: float  # @deprecated: Use Duration objects in new code
    purge_buffer_bars: int = 5
    purge_buffer_minutes: Optional[float] = None
    
    # Feature counts
    features_safe: int = 0
    features_dropped_nan: int = 0
    features_final: int = 0
    
    # Additional metadata
    view: str = "CROSS_SECTIONAL"
    symbol: Optional[str] = None
    
    # Time contract metadata (for reproducibility and validation)
    decision_time: str = "bar_close"  # When prediction happens
    label_starts_at: str = "t+1"  # When label window starts (t+1 = never includes bar t)
    prices: str = "unknown"  # Price adjustment: unknown/unadjusted/adjusted
    
    # Feature lookback (for audit validation)
    feature_lookback_max_minutes: Optional[float] = None  # Maximum feature lookback in minutes
    
    def __post_init__(self):
        """Compute derived values after initialization."""
        # Compute effective_min_cs if not set
        if not hasattr(self, 'effective_min_cs') or self.effective_min_cs is None:
            self.effective_min_cs = min(self.requested_min_cs, self.n_symbols_available)
        
        # Compute purge_buffer_minutes if not set
        if self.purge_buffer_minutes is None and self.interval_minutes is not None:
            self.purge_buffer_minutes = self.purge_buffer_bars * self.interval_minutes
    
    def log_summary(self, logger_instance: Optional[logging.Logger] = None) -> None:
        """
        Log a single authoritative summary line.
        
        This is the ONE place where all resolved values are logged together.
        """
        log = logger_instance or logger
        
        # Cross-sectional sampling
        min_cs_reason = f"only_{self.n_symbols_available}_symbols_loaded" if self.effective_min_cs < self.requested_min_cs else "requested"
        log.info(
            f"📊 Cross-sectional sampling: "
            f"requested_min_cs={self.requested_min_cs} → effective_min_cs={self.effective_min_cs} "
            f"(reason={min_cs_reason}, n_symbols={self.n_symbols_available}), "
            f"max_cs_samples={self.max_cs_samples}"
        )
        
        # Purge/embargo
        log.info(
            f"⏱️  Temporal safety: "
            f"horizon={self.horizon_minutes:.1f}m, "
            f"purge={self.purge_minutes:.1f}m, "
            f"embargo={self.embargo_minutes:.1f}m "
            f"(buffer={self.purge_buffer_minutes:.1f}m from {self.purge_buffer_bars} bars)"
        )
        
        # Feature counts
        if self.features_dropped_nan > 0:
            log.info(
                f"🔧 Features: "
                f"safe={self.features_safe} → "
                f"drop_all_nan={self.features_dropped_nan} → "
                f"final={self.features_final}"
            )
        else:
            log.info(
                f"🔧 Features: safe={self.features_safe} → final={self.features_final}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reproducibility tracking."""
        return {
            "requested_min_cs": self.requested_min_cs,
            "n_symbols_available": self.n_symbols_available,
            "effective_min_cs": self.effective_min_cs,
            "max_cs_samples": self.max_cs_samples,
            "interval_minutes": self.interval_minutes,
            "horizon_minutes": self.horizon_minutes,
            "purge_minutes": self.purge_minutes,
            "embargo_minutes": self.embargo_minutes,
            "purge_buffer_bars": self.purge_buffer_bars,
            "purge_buffer_minutes": self.purge_buffer_minutes,
            "features_safe": self.features_safe,
            "features_dropped_nan": self.features_dropped_nan,
            "features_final": self.features_final,
            "view": self.view,
            "symbol": self.symbol,
        }


def derive_purge_embargo(
    horizon_minutes: Optional[Union[float, DurationLike]],
    interval_minutes: Optional[Union[float, DurationLike]] = None,
    feature_lookback_max_minutes: Optional[Union[float, DurationLike]] = None,
    purge_buffer_bars: int = 5,
    default_purge_minutes: Optional[Union[float, DurationLike]] = None  # If None, loads from safety_config.yaml (SST)
) -> tuple[float, float]:
    """
    CENTRALIZED purge/embargo derivation function.
    
    This is the SINGLE source of truth for purge/embargo computation.
    Use this everywhere instead of local derivations.
    
    Formula:
        base = horizon_minutes (feature lookback is NOT included - it's historical and safe)
        buffer = purge_buffer_bars * interval_minutes
        purge = embargo = base + buffer
    
    Note: feature_lookback_max_minutes is accepted for API compatibility but NOT used in calculation.
    Feature lookback is historical data that doesn't need purging - only the target's future window does.
    
    Args:
        horizon_minutes: Target horizon (float minutes, Duration, or duration string like "60m")
        interval_minutes: Data interval (float minutes, Duration, or duration string like "5m")
        feature_lookback_max_minutes: Maximum feature lookback (accepted but not used)
        purge_buffer_bars: Number of bars to add as buffer
        default_purge_minutes: Default if horizon cannot be determined (if None, loads from safety_config.yaml)
    
    Returns:
        (purge_minutes, embargo_minutes) tuple as floats (for backward compatibility)
    """
    # Parse interval (for buffer calculation)
    if interval_minutes is not None:
        if isinstance(interval_minutes, (int, float)):
            interval_d = Duration.from_seconds(interval_minutes * 60.0)
        else:
            interval_d = parse_duration(interval_minutes)
        buffer_d = interval_d * purge_buffer_bars
    else:
        # Fallback: assume 5m bars if interval unknown
        interval_d = Duration.from_seconds(5.0 * 60.0)
        buffer_d = interval_d * purge_buffer_bars
    
    # Base purge/embargo = horizon (feature lookback is separate concern)
    # Feature lookback doesn't need to be purged - it's historical data that's safe to use
    # Purge is only needed to prevent leakage from the target's future window
    
    # NUCLEAR TEST MODE: Force 24-hour purge to test feature leak vs target leak
    # If default_purge_minutes >= 1500, use it regardless of horizon (diagnostic test)
    # TEST COMPLETE (2025-12-12): Score dropped from 0.99 to 0.763 with 24h purge
    # This confirmed feature leak is fixed, but 0.763 is still suspicious (target repainting likely)
    # Final Gatekeeper now handles feature leaks autonomously - nuclear test mode kept for future diagnostics
    force_nuclear_test = False
    if default_purge_minutes is None:
        try:
            from CONFIG.config_loader import get_cfg
            default_purge_minutes = get_cfg('safety.temporal.default_purge_minutes', default=85.0, config_name='safety_config')
        except Exception:
            default_purge_minutes = 85.0  # Final fallback
    
    # Parse default_purge_minutes
    if isinstance(default_purge_minutes, (int, float)):
        default_purge_d = Duration.from_seconds(default_purge_minutes * 60.0)
    else:
        default_purge_d = parse_duration(default_purge_minutes)
    
    if default_purge_d.to_minutes() >= 1500.0:
        # Nuclear test mode: Force 24-hour purge to test if leak is in features or target
        force_nuclear_test = True
        base_d = default_purge_d
    elif horizon_minutes is not None:
        # Parse horizon
        if isinstance(horizon_minutes, (int, float)):
            base_d = Duration.from_seconds(horizon_minutes * 60.0)
        else:
            base_d = parse_duration(horizon_minutes)
    else:
        base_d = default_purge_d
    
    # Add buffer
    purge_embargo_d = base_d + buffer_d
    
    # Convert back to minutes (float) for backward compatibility
    return purge_embargo_d.to_minutes(), purge_embargo_d.to_minutes()


def compute_feature_lookback_max(
    feature_names: List[str],
    interval_minutes: Optional[float] = None,
    max_lookback_cap_minutes: Optional[float] = None
) -> Tuple[Optional[float], List[Tuple[str, float]]]:
    """
    Compute maximum feature lookback from actual feature names.
    
    Uses feature registry to get lag_bars for each feature, then converts to minutes.
    
    Args:
        feature_names: List of feature names to analyze
        interval_minutes: Data interval in minutes (for conversion)
        max_lookback_cap_minutes: Optional cap for ranking mode (e.g., 240m = 4 hours)
    
    Returns:
        (max_lookback_minutes, top_lookback_features) tuple
        - max_lookback_minutes: Maximum lookback in minutes (None if cannot compute)
        - top_lookback_features: List of (feature_name, lookback_minutes) for top offenders
    """
    if not feature_names or interval_minutes is None or interval_minutes <= 0:
        return None, []
    
    try:
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
    except Exception:
        # Fallback: use pattern matching if registry unavailable
        registry = None
    
    feature_lookbacks = []
    
    for feat_name in feature_names:
        lag_bars = None
        
        # Try registry first
        if registry:
            try:
                metadata = registry.get_feature_metadata(feat_name)
                lag_bars = metadata.get('lag_bars')
            except Exception:
                pass
        
        # Fallback: pattern matching for common patterns
        if lag_bars is None:
            import re
            # PRECEDENCE ORDER (critical for accuracy):
            # 1. Explicit time suffixes (most reliable) - check FIRST
            # 2. Keyword heuristics (less reliable) - only as fallback
            
            # PRIORITY 1: Explicit time-based suffixes (most reliable)
            # These take precedence over keyword heuristics to avoid false positives
            # Example: "intraday_seasonality_15m" should use 15m, not "day" keyword
            
            # Minute-based patterns (e.g., _15m, _30m, _1440m) - CHECK FIRST
            minutes_match = re.search(r'_(\d+)m$', feat_name, re.I)
            if minutes_match:
                minutes = int(minutes_match.group(1))
                # Convert minutes to bars
                lag_bars = int(minutes / interval_minutes) if interval_minutes > 0 else minutes // 5
            
            # Hour-based patterns (e.g., _12h, _24h) - CHECK SECOND
            elif re.search(r'_(\d+)h', feat_name, re.I):
                hours_match = re.search(r'_(\d+)h', feat_name, re.I)
                hours = int(hours_match.group(1))
                # Convert hours to bars (assume 12 bars/hour for 5m data)
                lag_bars = hours * 12
            
            # Day-based patterns (e.g., _1d, _3d) - CHECK THIRD
            elif re.search(r'_(\d+)d', feat_name, re.I):
                days_match = re.search(r'_(\d+)d', feat_name, re.I)
                days = int(days_match.group(1))
                # Convert days to bars (assume 288 bars/day for 5m data)
                lag_bars = days * 288
            
            # PRIORITY 2: Keyword heuristics (fallback only if no explicit suffix)
            # Only use keyword patterns if no explicit time suffix was found
            # This prevents false positives like "intraday_seasonality_15m" being tagged as 1440m
            elif (re.search(r'_1d$|_1D$|_24h$|_24H$|^daily_|_daily$|_1440m|1440(?!\d)', feat_name, re.I) or
                  re.search(r'rolling.*daily|daily.*high|daily.*low', feat_name, re.I) or
                  re.search(r'volatility.*day|vol.*day|volume.*day', feat_name, re.I)):
                # Explicit daily patterns (ends with _1d, _24h, starts with daily_, etc.)
                # 1 day = 1440 minutes
                if interval_minutes > 0:
                    lag_bars = int(1440 / interval_minutes)  # 1 day in bars
                else:
                    lag_bars = 288  # Fallback: assume 5m bars (1440 / 5 = 288)
            
            # Last resort: very aggressive "day" keyword (only if no explicit suffix)
            # This is less reliable but catches features like "atr_day", "vol_day" that have no suffix
            elif re.search(r'.*day.*', feat_name, re.I):
                # Very aggressive: catch "day" anywhere (but only if no explicit suffix found)
                # Convert to bars based on interval
                if interval_minutes > 0:
                    lag_bars = int(1440 / interval_minutes)  # 1 day in bars
                else:
                    lag_bars = 288  # Fallback: assume 5m bars (1440 / 5 = 288)
            # Bar-based patterns (ret_N, sma_N, ema_N, rsi_N, etc.)
            elif re.match(r'^(ret|sma|ema|rsi|macd|bb|atr|adx|mom|vol|std|var)_(\d+)', feat_name):
                match = re.match(r'^(ret|sma|ema|rsi|macd|bb|atr|adx|mom|vol|std|var)_(\d+)', feat_name)
                lag_bars = int(match.group(2))
            # Calendar features (monthly, quarterly, yearly)
            elif re.search(r'monthly|quarterly|yearly', feat_name, re.I):
                # Calendar features - assume 1 month = 30 days
                lag_bars = 30 * 288  # Very long lookback
            else:
                # Unknown feature - assume minimal lookback (1 bar)
                lag_bars = 1
        
        if lag_bars is not None and lag_bars > 0:
            lookback_minutes = lag_bars * interval_minutes
            feature_lookbacks.append((feat_name, lookback_minutes))
    
    if not feature_lookbacks:
        return None, []
    
    # Sort by lookback (descending)
    feature_lookbacks.sort(key=lambda x: x[1], reverse=True)
    max_lookback = feature_lookbacks[0][1]
    
    # Apply cap if provided
    if max_lookback_cap_minutes is not None and max_lookback > max_lookback_cap_minutes:
        max_lookback = max_lookback_cap_minutes
    
    # Return top 10 offenders
    top_offenders = feature_lookbacks[:10]
    
    return max_lookback, top_offenders


def create_resolved_config(
    requested_min_cs: int,
    n_symbols_available: int,
    max_cs_samples: Optional[int],
    interval_minutes: Optional[float],
    horizon_minutes: Optional[float],
    feature_lookback_max_minutes: Optional[float] = None,
    purge_buffer_bars: int = 5,
    default_purge_minutes: float = 85.0,
    features_safe: int = 0,
    features_dropped_nan: int = 0,
    features_final: int = 0,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    feature_names: Optional[List[str]] = None,  # NEW: actual feature names for lookback computation
    recompute_lookback: bool = False  # NEW: if True, recompute from feature_names
) -> ResolvedConfig:
    """
    Create a ResolvedConfig object with all values computed consistently.
    
    This is the factory function that ensures purge/embargo are derived
    using the centralized function.
    """
    # Compute effective_min_cs
    effective_min_cs = min(requested_min_cs, n_symbols_available)
    
    # Recompute feature_lookback_max from actual features if requested
    if recompute_lookback and feature_names and interval_minutes:
        # Load ranking mode cap from config
        max_lookback_cap = None
        try:
            from CONFIG.config_loader import get_cfg
            max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
            if max_lookback_cap is not None:
                max_lookback_cap = float(max_lookback_cap)
        except Exception:
            pass
        
        computed_lookback, top_offenders = compute_feature_lookback_max(
            feature_names, interval_minutes, max_lookback_cap_minutes=max_lookback_cap
        )
        
        if computed_lookback is not None:
            # Log top offenders
            if top_offenders and top_offenders[0][1] > 240:  # Only log if > 4 hours
                logger.info(f"  📊 Feature lookback analysis: max={computed_lookback:.1f}m")
                logger.info(f"    Top lookback features: {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
            
            feature_lookback_max_minutes = computed_lookback
    
    # Compute purge/embargo using centralized function
    purge_minutes, embargo_base = derive_purge_embargo(
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes,
        feature_lookback_max_minutes=None,  # Don't pass to derive (it's separate)
        purge_buffer_bars=purge_buffer_bars,
        default_purge_minutes=default_purge_minutes
    )
    
    # CRITICAL FIX: Separate purge and embargo
    # - purge: max(horizon+buffer, feature_lookback_max) - prevents rolling window leakage
    # - embargo: horizon+buffer only - prevents label/horizon overlap (NOT tied to feature lookback)
    embargo_minutes = embargo_base  # Embargo is NOT affected by feature lookback
    
    # AUDIT VIOLATION FIX: If feature lookback > purge, increase purge to satisfy audit rule
    # This prevents "ROLLING WINDOW LEAKAGE RISK" violations
    # NOTE: Only purge is affected, NOT embargo
    # Can be disabled via config if features are strictly causal (only use past data)
    purge_include_feature_lookback = True  # Default: conservative (include feature lookback)
    try:
        from CONFIG.config_loader import get_cfg
        purge_include_feature_lookback = get_cfg("safety.leakage_detection.purge_include_feature_lookback", default=True, config_name="safety_config")
    except Exception:
        pass
    
    if purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        # Use generalized duration-aware audit rule enforcement
        # All inputs are currently floats (minutes), but we support DurationLike for future extensibility
        purge_in = purge_minutes  # Already a float (minutes)
        lookback_in = feature_lookback_max_minutes  # Already a float (minutes)
        interval_for_rule = interval_minutes  # Already a float (minutes) or None
        
        # Enforce audit rule with duration-aware comparison
        # parse_duration will handle float inputs (interpreted as seconds by default)
        # But we want minutes, so convert: float minutes -> Duration
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            purge_in * 60.0,  # Convert minutes to seconds for parse_duration
            lookback_in * 60.0,  # Convert minutes to seconds
            interval=interval_for_rule * 60.0 if interval_for_rule is not None else None,  # Convert to seconds
            buffer_frac=0.01,  # 1% safety buffer
            strict_greater=True
        )
        
        if changed:
            purge_minutes = purge_out.to_minutes()
            purge_in_str = f"{purge_in:.1f}m"
            lookback_in_str = f"{lookback_in:.1f}m"
            # Only log once per unique (purge, lookback, interval) combination
            # Use a simple cache key to avoid duplicate warnings
            cache_key = f"purge_bump_{purge_in:.1f}_{lookback_in:.1f}_{interval_for_rule or 0:.1f}"
            if not hasattr(create_resolved_config, '_logged_warnings'):
                create_resolved_config._logged_warnings = set()
            
            if cache_key not in create_resolved_config._logged_warnings:
                logger.warning(
                    f"⚠️  Audit violation prevention: purge ({purge_in_str}) < "
                    f"feature_lookback_max ({lookback_in_str}). "
                    f"Increasing purge to {format_duration(purge_out)} (min required: {format_duration(min_purge)}) "
                    f"to satisfy audit rule. Embargo remains {embargo_minutes:.1f}m (horizon-based, not feature lookback)."
                )
                create_resolved_config._logged_warnings.add(cache_key)
        # embargo_minutes stays at embargo_base (NOT increased)
    elif not purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        # Format lookback for logging (handle both float and DurationLike)
        if isinstance(feature_lookback_max_minutes, (int, float)):
            lookback_str = f"{feature_lookback_max_minutes:.1f}m"
        else:
            lookback_str = format_duration(parse_duration(feature_lookback_max_minutes))
        
        logger.info(
            f"ℹ️  Feature lookback ({lookback_str}) detected, but purge_include_feature_lookback=false. "
            f"Using horizon-based purge only ({purge_minutes:.1f}m). "
            f"Note: This assumes features are strictly causal (only use past data)."
        )
    
    # Compute buffer minutes
    if interval_minutes is not None:
        purge_buffer_minutes = purge_buffer_bars * interval_minutes
    else:
        purge_buffer_minutes = purge_buffer_bars * 5.0
    
    return ResolvedConfig(
        requested_min_cs=requested_min_cs,
        n_symbols_available=n_symbols_available,
        effective_min_cs=effective_min_cs,
        max_cs_samples=max_cs_samples,
        interval_minutes=interval_minutes,
        horizon_minutes=horizon_minutes,
        purge_minutes=purge_minutes,
        embargo_minutes=embargo_minutes,
        purge_buffer_bars=purge_buffer_bars,
        purge_buffer_minutes=purge_buffer_minutes,
        features_safe=features_safe,
        features_dropped_nan=features_dropped_nan,
        features_final=features_final,
        view=view,
        symbol=symbol,
        feature_lookback_max_minutes=feature_lookback_max_minutes
    )
