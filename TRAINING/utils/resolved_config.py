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
from typing import Optional, Dict, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class ResolvedConfig:
    """
    Single resolved configuration object for a target evaluation.
    
    All values are computed once and logged consistently.
    """
    # Cross-sectional sampling
    requested_min_cs: int
    n_symbols_available: int
    effective_min_cs: int
    max_cs_samples: Optional[int]
    
    # Data configuration
    interval_minutes: Optional[float]
    horizon_minutes: Optional[float]
    
    # Purge/embargo (single source of truth)
    purge_minutes: float
    embargo_minutes: float
    purge_buffer_bars: int = 5
    purge_buffer_minutes: Optional[float] = None
    
    # Feature counts
    features_safe: int = 0
    features_dropped_nan: int = 0
    features_final: int = 0
    
    # Additional metadata
    view: str = "CROSS_SECTIONAL"
    symbol: Optional[str] = None
    
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
    horizon_minutes: Optional[float],
    interval_minutes: Optional[float] = None,
    feature_lookback_max_minutes: Optional[float] = None,
    purge_buffer_bars: int = 5,
    default_purge_minutes: float = 85.0
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
        horizon_minutes: Target horizon in minutes
        interval_minutes: Data interval in minutes (for buffer calculation)
        feature_lookback_max_minutes: Maximum feature lookback in minutes (accepted but not used)
        purge_buffer_bars: Number of bars to add as buffer
        default_purge_minutes: Default if horizon cannot be determined
    
    Returns:
        (purge_minutes, embargo_minutes) tuple
    """
    # Compute buffer
    if interval_minutes is not None:
        buffer_minutes = purge_buffer_bars * interval_minutes
    else:
        # Fallback: assume 5m bars if interval unknown
        buffer_minutes = purge_buffer_bars * 5.0
    
    # Base purge/embargo = horizon (feature lookback is separate concern)
    # Feature lookback doesn't need to be purged - it's historical data that's safe to use
    # Purge is only needed to prevent leakage from the target's future window
    if horizon_minutes is not None:
        base_minutes = horizon_minutes
    else:
        # Fallback to default if horizon unknown
        base_minutes = default_purge_minutes
    
    # Add buffer
    purge_embargo_minutes = base_minutes + buffer_minutes
    
    return purge_embargo_minutes, purge_embargo_minutes


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
    symbol: Optional[str] = None
) -> ResolvedConfig:
    """
    Create a ResolvedConfig object with all values computed consistently.
    
    This is the factory function that ensures purge/embargo are derived
    using the centralized function.
    """
    # Compute effective_min_cs
    effective_min_cs = min(requested_min_cs, n_symbols_available)
    
    # Compute purge/embargo using centralized function
    purge_minutes, embargo_minutes = derive_purge_embargo(
        horizon_minutes=horizon_minutes,
        interval_minutes=interval_minutes,
        feature_lookback_max_minutes=feature_lookback_max_minutes,
        purge_buffer_bars=purge_buffer_bars,
        default_purge_minutes=default_purge_minutes
    )
    
    # AUDIT VIOLATION FIX: If feature lookback > purge, increase purge to satisfy audit rule
    # This prevents "ROLLING WINDOW LEAKAGE RISK" violations
    # Note: This is a conservative fix - ideally we'd filter features, but we don't have per-feature lookback metadata
    if feature_lookback_max_minutes is not None and purge_minutes < feature_lookback_max_minutes:
        logger.warning(
            f"⚠️  Audit violation prevention: purge ({purge_minutes:.1f}m) < feature_lookback_max ({feature_lookback_max_minutes:.1f}m). "
            f"Increasing purge to {feature_lookback_max_minutes:.1f}m to satisfy audit rule."
        )
        purge_minutes = feature_lookback_max_minutes
        embargo_minutes = feature_lookback_max_minutes
    
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
        symbol=symbol
    )
