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

from TRAINING.common.utils.duration_parser import (
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
    
    # Purge/embargo (single source of truth) - REQUIRED FIELDS (must come before defaults)
    # @deprecated: Use Duration objects directly in new code. These float fields are for backward compatibility.
    # NOTE: These are converted from Duration to float minutes at the boundary.
    # Internally, all comparisons use Duration objects (see enforce_purge_audit_rule).
    purge_minutes: float  # @deprecated: Use Duration objects in new code
    embargo_minutes: float  # @deprecated: Use Duration objects in new code
    
    # NEW: Base interval for training grid (multi-interval support) - OPTIONAL FIELDS (with defaults)
    base_interval_minutes: Optional[float] = None  # Base training grid interval (from config or auto-detected)
    base_interval_source: str = "auto"  # "config" or "auto" (for config trace)
    default_embargo_minutes: float = 0.0  # Default embargo for features without explicit metadata
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
    
    # NEW: Multi-interval feature metadata (for as-of alignment)
    feature_time_meta_map: Optional[Dict[str, Any]] = None  # Map of feature_name -> FeatureTimeMeta
    
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
            # NEW: Base interval for replayability (auto-detected or from config)
            "base_interval_minutes": self.base_interval_minutes,
            "base_interval_source": self.base_interval_source,
        }


def derive_purge_embargo(
    horizon_minutes: Optional[Union[float, DurationLike]],
    interval_minutes: Optional[Union[float, DurationLike]] = None,
    feature_lookback_max_minutes: Optional[Union[float, DurationLike]] = None,
    purge_buffer_bars: int = 5,
    embargo_extra_bars: Optional[int] = None,  # If None, loads from safety_config.yaml
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
    
    # Add buffer for purge
    purge_d = base_d + buffer_d
    
    # Embargo uses separate extra_bars (if configured) or same buffer
    if embargo_extra_bars is None:
        # Load from config
        try:
            from CONFIG.config_loader import get_cfg
            embargo_extra_bars = int(get_cfg('safety.leakage_detection.cv.embargo_extra_bars', default=purge_buffer_bars, config_name='safety_config'))
        except Exception:
            embargo_extra_bars = purge_buffer_bars  # Fallback: use same as purge buffer
    
    # Compute embargo buffer separately
    embargo_buffer_d = interval_d * embargo_extra_bars
    embargo_d = base_d + embargo_buffer_d
    
    # Convert back to minutes (float) for backward compatibility
    return purge_d.to_minutes(), embargo_d.to_minutes()


def compute_feature_lookback_max(
    feature_names: List[str],
    interval_minutes: Optional[float] = None,
    max_lookback_cap_minutes: Optional[float] = None,
    horizon_minutes: Optional[float] = None,
    registry: Optional[Any] = None,
    expected_fingerprint: Optional[str] = None,
    stage: str = "unknown",
    feature_time_meta_map: Optional[Dict[str, Any]] = None,  # NEW: Optional map of feature_name -> FeatureTimeMeta
    base_interval_minutes: Optional[float] = None,  # NEW: Base training grid interval
    canonical_lookback_map: Optional[Dict[str, float]] = None  # NEW: Optional pre-computed canonical map (SST reuse)
) -> Tuple[Optional[float], List[Tuple[str, float]]]:
    """
    Compute maximum feature lookback from actual feature names.
    
    DEPRECATED: This is now a thin wrapper around leakage_budget.compute_feature_lookback_max()
    to maintain backward compatibility. New code should use leakage_budget.compute_budget() directly.
    
    Uses unified leakage budget calculator to ensure consistency with audit and gatekeeper.
    
    Args:
        feature_names: List of feature names to analyze
        interval_minutes: Data interval in minutes (for conversion)
        max_lookback_cap_minutes: Optional cap for ranking mode (e.g., 240m = 4 hours)
        horizon_minutes: Optional target horizon (for budget calculation)
        registry: Optional feature registry (will be loaded if None)
        feature_time_meta_map: Optional map of feature_name -> FeatureTimeMeta (for multi-interval support)
        base_interval_minutes: Optional base training grid interval (for defaulting native_interval)
    
    Returns:
        LookbackResult dataclass (or tuple for backward compatibility if needed)
        - max_minutes: Maximum lookback in minutes (None if cannot compute)
        - top_offenders: List of (feature_name, lookback_minutes) for top offenders
        - fingerprint: Set-invariant fingerprint
        - order_fingerprint: Order-sensitive fingerprint
    """
    # Delegate to unified leakage budget calculator
    # Use the legacy wrapper function from leakage_budget module
    from TRAINING.utils import leakage_budget
    
    result = leakage_budget.compute_feature_lookback_max(
        feature_names=feature_names,
        interval_minutes=interval_minutes,
        max_lookback_cap_minutes=max_lookback_cap_minutes,
        horizon_minutes=horizon_minutes,
        registry=registry,
        expected_fingerprint=expected_fingerprint,
        stage=stage if stage != "unknown" else "resolved_config_wrapper",
        feature_time_meta_map=feature_time_meta_map,  # NEW: Pass per-feature metadata
        base_interval_minutes=base_interval_minutes,  # NEW: Pass base interval
        canonical_lookback_map=canonical_lookback_map  # NEW: Pass canonical map for SST reuse
    )
    # Return the LookbackResult dataclass directly (not a tuple)
    # This maintains compatibility with new code that expects dataclass
    # Old code expecting tuple will need to be updated
    return result


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
    recompute_lookback: bool = False,  # NEW: if True, recompute from feature_names
    experiment_config: Optional[Any] = None  # NEW: Optional ExperimentConfig for base_interval_minutes
) -> ResolvedConfig:
    """
    Create a ResolvedConfig object with all values computed consistently.
    
    This is the factory function that ensures purge/embargo are derived
    using the centralized function.
    """
    # Compute effective_min_cs
    effective_min_cs = min(requested_min_cs, n_symbols_available)
    
    # NEW: Resolve base_interval_minutes from experiment config or auto-detection
    base_interval_minutes = None
    base_interval_source = "auto"
    default_embargo_minutes = 0.0
    
    if experiment_config is not None:
        # Try to get base_interval_minutes from experiment config
        if hasattr(experiment_config, 'data') and hasattr(experiment_config.data, 'base_interval_minutes'):
            base_interval_minutes = experiment_config.data.base_interval_minutes
            base_interval_source = experiment_config.data.base_interval_source or "config"
            default_embargo_minutes = experiment_config.data.default_embargo_minutes or 0.0
            
            # Config trace logging
            config_path = None
            try:
                from CONFIG.config_loader import get_config_path
                # Try to get experiment config path
                config_path = get_config_path("experiment_config") if hasattr(experiment_config, 'name') else None
            except Exception:
                pass
            
            provenance = f"experiment_config.data.base_interval_minutes = {base_interval_minutes if base_interval_minutes is not None else 'None'}"
            if config_path:
                provenance += f" (from {config_path})"
            else:
                provenance += " (from ExperimentConfig object)"
            
            value_str = f"{base_interval_minutes}m" if base_interval_minutes is not None else "None"
            logger.info(f"📋 CONFIG TRACE (base_interval_minutes): source={base_interval_source}, value={value_str}, {provenance}")
            
            # If base_interval_source='config' but base_interval_minutes is None, hard-fail
            if base_interval_source == 'config' and base_interval_minutes is None:
                raise ValueError(
                    f"base_interval_source='config' requires base_interval_minutes to be set in experiment config. "
                    f"Either set base_interval_minutes or use base_interval_source='auto'."
                )
    
    # If base_interval_minutes not set from config, use auto-detected interval_minutes
    if base_interval_minutes is None:
        base_interval_minutes = interval_minutes
        base_interval_source = "auto"
        value_str = f"{base_interval_minutes}m" if base_interval_minutes is not None else "None"
        logger.info(
            f"📋 CONFIG TRACE (base_interval_minutes): source=auto, value={value_str} "
            f"(from auto-detected interval_minutes)"
        )
    
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
        
        lookback_result = compute_feature_lookback_max(
            feature_names, interval_minutes, max_lookback_cap_minutes=max_lookback_cap,
            stage="create_resolved_config",
            feature_time_meta_map=None,  # TODO: Pass feature_time_meta_map when available
            base_interval_minutes=base_interval_minutes
        )
        # Handle dataclass return
        if hasattr(lookback_result, 'max_minutes'):
            computed_lookback = lookback_result.max_minutes
            top_offenders = lookback_result.top_offenders
        else:
            # Tuple return (backward compatibility)
            computed_lookback, top_offenders = lookback_result
        
        if computed_lookback is not None:
            # Log top offenders
            if top_offenders and top_offenders[0][1] > 240:  # Only log if > 4 hours
                logger.info(f"  📊 Feature lookback analysis: max={computed_lookback:.1f}m")
                logger.info(f"    Top lookback features: {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
            
            feature_lookback_max_minutes = computed_lookback
    
    # NEW: Build feature_time_meta_map from registry or config (for multi-interval alignment)
    feature_time_meta_map = None
    if feature_names and base_interval_minutes is not None:
        from TRAINING.ranking.utils.feature_time_meta import FeatureTimeMeta
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
        except Exception:
            registry = None
        
        feature_time_meta_map = {}
        for feat_name in feature_names:
            # Skip target columns explicitly
            if any(feat_name.startswith(prefix) for prefix in 
                   ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_',
                    'tth_', 'p_', 'barrier_', 'hit_']):
                continue
            
            # Try to get metadata from registry
            native_interval = None
            embargo = default_embargo_minutes
            publish_offset = 0.0
            max_staleness = None
            lookback_bars = None
            lookback_minutes = None
            
            if registry is not None:
                try:
                    metadata = registry.get_feature_metadata(feat_name)
                    # Extract time semantics from registry if available
                    # Registry may have fields like: native_interval_minutes, embargo_minutes, etc.
                    if 'native_interval_minutes' in metadata:
                        native_interval = metadata['native_interval_minutes']
                    if 'embargo_minutes' in metadata:
                        embargo = metadata['embargo_minutes']
                    if 'publish_offset_minutes' in metadata:
                        publish_offset = metadata['publish_offset_minutes']
                    if 'max_staleness_minutes' in metadata:
                        max_staleness = metadata['max_staleness_minutes']
                    if 'lag_bars' in metadata and metadata['lag_bars'] is not None:
                        lookback_bars = metadata['lag_bars']
                except Exception:
                    pass
            
            # Override with experiment config if available (experiment overrides registry)
            if experiment_config is not None:
                # TODO: If experiment config has per-feature time metadata, use it here
                # For now, use defaults from experiment config
                if hasattr(experiment_config, 'data'):
                    if native_interval is None:
                        native_interval = base_interval_minutes  # Default to base interval
                    if embargo == default_embargo_minutes:
                        embargo = experiment_config.data.default_embargo_minutes or default_embargo_minutes
            
            # Only create metadata if feature needs alignment (different interval, embargo, or publish_offset)
            # OR if explicitly requested (for now, create for all features to enable alignment)
            # In future, we can optimize to only create for features that need it
            if native_interval is None:
                native_interval = base_interval_minutes  # Default to base interval
            
            # Create FeatureTimeMeta (only if different from base or has embargo/publish_offset)
            if (native_interval != base_interval_minutes or 
                embargo != 0.0 or 
                publish_offset != 0.0 or
                max_staleness is not None):
                feature_time_meta_map[feat_name] = FeatureTimeMeta(
                    name=feat_name,
                    native_interval_minutes=native_interval,
                    embargo_minutes=embargo,
                    publish_offset_minutes=publish_offset,
                    max_staleness_minutes=max_staleness,
                    lookback_bars=lookback_bars,
                    lookback_minutes=lookback_minutes
                )
        
        if feature_time_meta_map:
            logger.info(
                f"🔧 Multi-interval metadata: Built {len(feature_time_meta_map)} FeatureTimeMeta entries "
                f"(from registry + config, experiment overrides registry)"
            )
        else:
            logger.debug("Multi-interval metadata: No features require alignment (all use base interval with zero embargo)")
    
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
    # NOTE: This purge computation may be overridden by CV splitter (which uses final post-prune featureset)
    # The CV splitter is the authoritative source for purge in train_and_evaluate_models()
    embargo_minutes = embargo_base  # Embargo is NOT affected by feature lookback
    
    # AUDIT VIOLATION FIX: If feature lookback > purge, increase purge to satisfy audit rule
    # This prevents "ROLLING WINDOW LEAKAGE RISK" violations
    # NOTE: Only purge is affected, NOT embargo
    # Can be disabled via config if features are strictly causal (only use past data)
    purge_include_feature_lookback = True  # Default: conservative (include feature lookback)
    purge_include_provenance = None
    try:
        from CONFIG.config_loader import get_cfg, get_config_path
        purge_include_feature_lookback = get_cfg("safety.leakage_detection.purge_include_feature_lookback", default=True, config_name="safety_config")
        config_path = get_config_path("safety_config")
        purge_include_provenance = f"safety_config.yaml:{config_path} → safety.leakage_detection.purge_include_feature_lookback = {purge_include_feature_lookback} (default=True)"
    except Exception as e:
        purge_include_provenance = f"config lookup failed: {e}"
    
    # Log config trace for purge_include_feature_lookback
    logger.info(f"📋 CONFIG TRACE (purge_include_feature_lookback): {purge_include_provenance}")
    
    if purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        # CRITICAL: In pre-enforcement stages, cap lookback used for purge bump
        # Pre-enforcement lookback includes long-lookback features that will be dropped by gatekeeper
        # Don't inflate purge based on pre-enforcement max - cap it to lookback budget cap
        lookback_budget_cap = None
        try:
            from CONFIG.config_loader import get_cfg
            # Check for lookback budget cap (ranking mode)
            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                lookback_budget_cap = float(budget_cap_raw)
            # Fallback to legacy ranking_mode_max_lookback_minutes
            if lookback_budget_cap is None:
                lookback_budget_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
                if lookback_budget_cap is not None:
                    lookback_budget_cap = float(lookback_budget_cap)
        except Exception:
            pass
        
        # If we have a cap and lookback exceeds it, cap lookback for purge bump
        # This prevents pre-enforcement purge inflation (gatekeeper will drop long-lookback features)
        lookback_for_purge = feature_lookback_max_minutes
        if lookback_budget_cap is not None and feature_lookback_max_minutes > lookback_budget_cap:
            lookback_for_purge = lookback_budget_cap
            logger.debug(
                f"📊 Pre-enforcement purge guard: feature_lookback_max={feature_lookback_max_minutes:.1f}m > "
                f"cap={lookback_budget_cap:.1f}m. Capping lookback used for purge bump to {lookback_budget_cap:.1f}m "
                f"(gatekeeper will drop long-lookback features, final purge will be recomputed at POST_PRUNE)."
            )
        
        # Use generalized duration-aware audit rule enforcement
        # All inputs are currently floats (minutes), but we support DurationLike for future extensibility
        purge_in = purge_minutes  # Already a float (minutes)
        lookback_in = lookback_for_purge  # Capped lookback (if pre-enforcement) or original
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
        feature_lookback_max_minutes=feature_lookback_max_minutes,
        base_interval_minutes=base_interval_minutes,  # NEW
        base_interval_source=base_interval_source,  # NEW
        default_embargo_minutes=default_embargo_minutes,  # NEW
        feature_time_meta_map=feature_time_meta_map  # NEW: Multi-interval alignment metadata
    )
