# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
    
    # Load sanitization mode and threshold from config
    sanitize_mode = "budget_cap"  # Default: use budget cap
    if _CONFIG_AVAILABLE:
        try:
            sanitize_mode = get_cfg("safety.leakage_detection.active_sanitization.mode", default="budget_cap", config_name="safety_config")
        except Exception:
            pass
    
    # Determine threshold based on mode
    if max_safe_lookback_minutes is None:
        if sanitize_mode == "budget_cap":
            # Use budget cap from config (if set)
            if _CONFIG_AVAILABLE:
                try:
                    budget_cap = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                    if budget_cap != "auto" and isinstance(budget_cap, (int, float)):
                        max_safe_lookback_minutes = float(budget_cap)
                    else:
                        # No cap set - disable sanitization or use purge-derived
                        max_safe_lookback_minutes = None  # Will be handled below
                except Exception:
                    max_safe_lookback_minutes = None
            else:
                max_safe_lookback_minutes = None
        elif sanitize_mode == "fixed":
            # Use fixed threshold from config
            if _CONFIG_AVAILABLE:
                try:
                    max_safe_lookback_minutes = get_cfg("safety.leakage_detection.active_sanitization.fixed_threshold_minutes", default=240.0, config_name="safety_config")
                except Exception:
                    max_safe_lookback_minutes = 240.0  # Default: 4 hours
            else:
                max_safe_lookback_minutes = 240.0  # Default: 4 hours
        elif sanitize_mode == "purge_allowance":
            # Will be computed from purge - buffer (not available here, skip sanitization)
            logger.debug("Sanitization mode 'purge_allowance' requires purge context - skipping sanitization")
            return feature_names, [], {"enabled": True, "quarantined": [], "reason": "purge_allowance_mode_requires_context"}
        else:
            # Unknown mode - use fixed default
            max_safe_lookback_minutes = 240.0  # Default: 4 hours
    
    # If no threshold determined and mode is budget_cap, disable sanitization
    if max_safe_lookback_minutes is None and sanitize_mode == "budget_cap":
        logger.debug("Sanitization mode 'budget_cap' but no cap set (lookback_budget_minutes=auto) - disabling sanitization")
        return feature_names, [], {"enabled": True, "quarantined": [], "reason": "no_cap_set"}
    
    if not feature_names:
        return [], [], {"enabled": True, "quarantined": [], "reason": "no_features"}
    
    # CRITICAL: Use apply_lookback_cap() to follow the same structure as all other phases
    # This ensures consistency: same canonical map, same quarantine logic, same invariants
    from TRAINING.ranking.utils.lookback_cap_enforcement import apply_lookback_cap
    from CONFIG.config_loader import get_cfg
    from TRAINING.common.feature_registry import get_registry
    
    # Load policy and log_mode from config
    policy = "drop"  # Sanitizer uses "drop" (quarantine, don't hard-fail)
    try:
        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
    except Exception:
        pass
    
    log_mode = "summary"
    try:
        log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
    except Exception:
        pass
    
    # Get registry
    registry = None
    try:
        registry = get_registry()
    except Exception:
        pass
    
    # Use apply_lookback_cap() - follows standard 6-step structure
    cap_result = apply_lookback_cap(
        features=feature_names,
        interval_minutes=interval_minutes if interval_minutes is not None else 5.0,
        cap_minutes=max_safe_lookback_minutes,
        policy=policy,
        stage="feature_sanitizer",
        registry=registry,
        log_mode=log_mode
    )
    
    # Extract results (follows same structure as apply_lookback_cap)
    safe_features = cap_result.safe_features
    quarantined_features = cap_result.quarantined_features
    
    # Build quarantine_report from result metadata (preserve existing API)
    quarantine_reasons = {}
    for feat_name in quarantined_features:
        # Get lookback from canonical map
        from TRAINING.ranking.utils.leakage_budget import _feat_key
        feat_key = _feat_key(feat_name)
        lookback = cap_result.canonical_map.get(feat_key)
        
        if lookback is None or lookback == float("inf"):
            quarantine_reasons[feat_name] = {
                "lookback_minutes": None,  # Unknown
                "max_safe_lookback_minutes": max_safe_lookback_minutes,
                "reason": "unknown lookback (cannot infer - treated as unsafe)"
            }
        else:
            quarantine_reasons[feat_name] = {
                "lookback_minutes": lookback,
                "max_safe_lookback_minutes": max_safe_lookback_minutes,
                "reason": f"lookback ({lookback:.1f}m) exceeds safe threshold ({max_safe_lookback_minutes:.1f}m)"
            }
    
    # Build quarantine report
    quarantine_report = {
        "enabled": True,
        "max_safe_lookback_minutes": max_safe_lookback_minutes,
        "quarantined_count": len(quarantined_features),
        "safe_count": len(safe_features),
        "quarantined": quarantined_features,
        "reasons": quarantine_reasons
    }
    
    # Log results (aggregated summary + small sample)
    if quarantined_features:
        sample_size = min(5, len(quarantined_features))
        sample = quarantined_features[:sample_size]
        sample_str = ', '.join(sample)
        if len(quarantined_features) > sample_size:
            sample_str += f", ... ({len(quarantined_features) - sample_size} more)"
        
        logger.warning(
            f"👻 ACTIVE SANITIZATION: Quarantined {len(quarantined_features)} feature(s) "
            f"with lookback > {max_safe_lookback_minutes:.1f}m to prevent audit violations. "
            f"Sample: [{sample_str}]"
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
