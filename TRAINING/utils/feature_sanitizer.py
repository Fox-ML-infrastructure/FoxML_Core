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

Active Feature Sanitization

Automatically quarantines features that violate lookback rules before training starts.
This prevents "ghost feature" discrepancies where audit and auto-fix see different lookback values.

This is the "Ghost Buster" - it proactively removes problematic features instead of
just detecting them after the fact.
"""

import re
import logging
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Add CONFIG directory to path for centralized config loading
_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded defaults")


def auto_quarantine_long_lookback_features(
    feature_names: List[str],
    interval_minutes: Optional[float] = None,
    max_safe_lookback_minutes: Optional[float] = None,
    enabled: Optional[bool] = None
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Active sanitization: automatically quarantine features with excessive lookback.
    
    This function scans features for patterns that indicate long lookback windows
    (daily/24h/1440m features) and removes them before training starts. This prevents
    "ghost feature" discrepancies where audit and auto-fix see different lookback values.
    
    Args:
        feature_names: List of feature names to scan
        interval_minutes: Data interval in minutes (for lookback calculation)
        max_safe_lookback_minutes: Maximum safe lookback in minutes (if None, loads from config)
        enabled: Whether to enable active sanitization (if None, loads from config)
    
    Returns:
        (safe_features, quarantined_features, quarantine_report) tuple where:
        - safe_features: Features that passed sanitization
        - quarantined_features: Features that were quarantined (excluded)
        - quarantine_report: Dict with details about what was quarantined and why
    """
    # Load config
    if enabled is None:
        if _CONFIG_AVAILABLE:
            try:
                enabled = get_cfg("safety.leakage_detection.active_sanitization.enabled", default=True, config_name="safety_config")
            except Exception:
                enabled = True  # Default: enabled
        else:
            enabled = True  # Default: enabled
    
    if not enabled:
        logger.debug("Active sanitization disabled - all features passed through")
        return feature_names, [], {"enabled": False, "quarantined": []}
    
    # Load max safe lookback threshold
    if max_safe_lookback_minutes is None:
        if _CONFIG_AVAILABLE:
            try:
                max_safe_lookback_minutes = get_cfg("safety.leakage_detection.active_sanitization.max_safe_lookback_minutes", default=240.0, config_name="safety_config")
            except Exception:
                max_safe_lookback_minutes = 240.0  # Default: 4 hours
        else:
            max_safe_lookback_minutes = 240.0  # Default: 4 hours
    
    if not feature_names:
        return [], [], {"enabled": True, "quarantined": [], "reason": "no_features"}
    
    # Compute lookback for all features at once (more efficient)
    from TRAINING.utils.resolved_config import compute_feature_lookback_max
    
    # Get lookback for all features in one call
    max_lookback_all, top_offenders_all = compute_feature_lookback_max(
        feature_names,
        interval_minutes=interval_minutes,
        max_lookback_cap_minutes=None  # Don't cap - we want the real value
    )
    
    # Build lookup dict from top_offenders (contains features with significant lookback)
    # Note: top_offenders_all only contains top 10, so we need to check all features
    # Compute lookback for each feature to catch any that exceed threshold
    feature_lookbacks = []
    for feat_name in feature_names:
        # Compute lookback for this feature
        max_lookback, _ = compute_feature_lookback_max(
            [feat_name],
            interval_minutes=interval_minutes,
            max_lookback_cap_minutes=None  # Don't cap - we want the real value
        )
        
        if max_lookback is not None:
            feature_lookbacks.append((feat_name, max_lookback))
        else:
            # Unknown feature - assume safe (minimal lookback)
            feature_lookbacks.append((feat_name, 0.0))
    
    # Separate safe and problematic features
    safe_features = []
    quarantined_features = []
    quarantine_reasons = {}
    
    for feat_name, lookback_minutes in feature_lookbacks:
        if lookback_minutes > max_safe_lookback_minutes:
            quarantined_features.append(feat_name)
            quarantine_reasons[feat_name] = {
                "lookback_minutes": lookback_minutes,
                "max_safe_lookback_minutes": max_safe_lookback_minutes,
                "reason": f"lookback ({lookback_minutes:.1f}m) exceeds safe threshold ({max_safe_lookback_minutes:.1f}m)"
            }
        else:
            safe_features.append(feat_name)
    
    # Build quarantine report
    quarantine_report = {
        "enabled": True,
        "max_safe_lookback_minutes": max_safe_lookback_minutes,
        "quarantined_count": len(quarantined_features),
        "safe_count": len(safe_features),
        "quarantined": quarantined_features,
        "reasons": quarantine_reasons
    }
    
    # Log results
    if quarantined_features:
        logger.warning(
            f"👻 ACTIVE SANITIZATION: Quarantined {len(quarantined_features)} feature(s) "
            f"with lookback > {max_safe_lookback_minutes:.1f}m to prevent audit violations"
        )
        for feat_name in quarantined_features:
            reason = quarantine_reasons[feat_name]
            logger.warning(
                f"   🚫 {feat_name}: {reason['reason']}"
            )
        logger.info(f"   ✅ {len(safe_features)} safe features remaining")
    else:
        logger.debug(f"✅ Active sanitization: All {len(safe_features)} features passed (lookback <= {max_safe_lookback_minutes:.1f}m)")
    
    return safe_features, quarantined_features, quarantine_report


def quarantine_by_pattern(
    feature_names: List[str],
    patterns: Optional[List[str]] = None,
    enabled: Optional[bool] = None
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Quarantine features by regex patterns (for specific problematic patterns).
    
    This is a more aggressive approach that quarantines features based on naming
    patterns rather than computed lookback. Useful for known problematic feature types.
    
    Args:
        feature_names: List of feature names to scan
        patterns: List of regex patterns to match (if None, loads from config)
        enabled: Whether to enable pattern-based quarantine (if None, loads from config)
    
    Returns:
        (safe_features, quarantined_features, quarantine_report) tuple
    """
    # Load config
    if enabled is None:
        if _CONFIG_AVAILABLE:
            try:
                enabled = get_cfg("safety.leakage_detection.active_sanitization.pattern_quarantine.enabled", default=False, config_name="safety_config")
            except Exception:
                enabled = False  # Default: disabled (more aggressive)
        else:
            enabled = False
    
    if not enabled:
        return feature_names, [], {"enabled": False, "quarantined": []}
    
    # Load patterns from config if not provided
    if patterns is None:
        if _CONFIG_AVAILABLE:
            try:
                patterns = get_cfg("safety.leakage_detection.active_sanitization.pattern_quarantine.patterns", default=[], config_name="safety_config")
            except Exception:
                patterns = []
        else:
            patterns = []
    
    if not patterns:
        return feature_names, [], {"enabled": True, "quarantined": [], "reason": "no_patterns"}
    
    # Compile patterns
    compiled_patterns = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    # Scan features
    safe_features = []
    quarantined_features = []
    quarantine_reasons = {}
    
    for feat_name in feature_names:
        matched = False
        matched_pattern = None
        
        for pattern in compiled_patterns:
            if pattern.search(feat_name):
                matched = True
                matched_pattern = pattern.pattern
                break
        
        if matched:
            quarantined_features.append(feat_name)
            quarantine_reasons[feat_name] = {
                "reason": f"matched pattern: {matched_pattern}",
                "pattern": matched_pattern
            }
        else:
            safe_features.append(feat_name)
    
    # Build report
    quarantine_report = {
        "enabled": True,
        "patterns": patterns,
        "quarantined_count": len(quarantined_features),
        "safe_count": len(safe_features),
        "quarantined": quarantined_features,
        "reasons": quarantine_reasons
    }
    
    # Log results
    if quarantined_features:
        logger.warning(
            f"👻 PATTERN QUARANTINE: Quarantined {len(quarantined_features)} feature(s) "
            f"matching {len(patterns)} pattern(s)"
        )
        for feat_name in quarantined_features:
            reason = quarantine_reasons[feat_name]
            logger.warning(f"   🚫 {feat_name}: {reason['reason']}")
        logger.info(f"   ✅ {len(safe_features)} safe features remaining")
    
    return safe_features, quarantined_features, quarantine_report
