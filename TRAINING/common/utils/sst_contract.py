"""
SST (Single Source of Truth) Contract Module

Centralized normalization and validation functions for:
- Family name normalization
- Target horizon resolution
- Tracker input adaptation (string/Enum-safe)
- Feature drop reason tracking

This ensures consistency across all pipeline layers.
"""

import re
import logging
from typing import Optional, Dict, Any, Union, List
from enum import Enum

logger = logging.getLogger(__name__)

# ============================================================================
# Family Name Normalization
# ============================================================================

def normalize_family(family: Union[str, None]) -> str:
    """
    Canonicalize model family name to snake_case lowercase.
    
    This is the SINGLE SOURCE OF TRUTH for family name normalization.
    All registries (MODMAP, TRAINER_MODULE_MAP, runtime_policy, FAMILY_CAPS)
    MUST use this function.
    
    Args:
        family: Family name (can be any case/variant)
    
    Returns:
        Normalized family name in snake_case lowercase
    
    Examples:
        "LightGBM" -> "lightgbm"
        "XGBoost" -> "xgboost"
        "x_g_boost" -> "xgboost"
        "RandomForest" -> "random_forest"
        "random_forest" -> "random_forest"
    """
    if not family or not isinstance(family, str):
        return str(family).lower() if family else ""
    
    # Normalize input: strip, replace hyphens/spaces with underscores
    family_clean = family.strip().replace("-", "_").replace(" ", "_")
    
    # Special cases for common variants (CRITICAL: handle before other normalization)
    special_cases = {
        "x_g_boost": "xgboost",  # Fix: x_g_boost -> xgboost alias
        "xgb": "xgboost",
        "lgb": "lightgbm",
        "lgbm": "lightgbm",
        "xgboost": "xgboost",  # Ensure XGBoost -> xgboost (not x_g_boost)
    }
    family_lower = family_clean.lower()
    if family_lower in special_cases:
        return special_cases[family_lower]
    
    # Also check if input is "XGBoost" (TitleCase) - normalize before special cases
    if family_clean == "XGBoost":
        return "xgboost"
    
    # If already snake_case (has underscores), just lowercase
    if "_" in family_clean:
        return family_clean.lower().replace("__", "_")
    
    # Convert TitleCase/CamelCase to snake_case
    # Split on capital letters: "LightGBM" -> ["", "Light", "GBM"]
    parts = re.split(r'(?=[A-Z])', family_clean)
    parts = [p for p in parts if p]  # Remove empty strings
    
    if len(parts) == 1:
        # Single word, just lowercase
        return parts[0].lower()
    
    # Join parts with underscores, all lowercase
    result = "_".join(p.lower() for p in parts)
    
    # Clean up: remove double underscores
    result = result.replace("__", "_")
    
    return result


# ============================================================================
# Target Horizon Resolution
# ============================================================================

def resolve_target_horizon_minutes(target_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Resolve target horizon in minutes from target name.
    
    This is the SINGLE SOURCE OF TRUTH for horizon extraction.
    Handles special cases like *_same_day, *_5d, etc.
    
    Args:
        target_name: Target column name (e.g., 'fwd_ret_oc_same_day', 'fwd_ret_5d', 'y_will_peak_60m_0.8')
        config: Optional config dict with horizon_extraction patterns
    
    Returns:
        Horizon in minutes, or None if cannot be determined (should NOT default silently)
    
    Special Cases:
        - *_same_day: Returns 390 minutes (6.5 hours = trading session)
        - *_oc_same_day: Returns 390 minutes (open-to-close same day)
        - fwd_ret_5d: Returns 5 * 1440 = 7200 minutes
        - fwd_ret_1d: Returns 1440 minutes
    """
    if not target_name or not isinstance(target_name, str):
        return None
    
    target_lower = target_name.lower()
    
    # Special cases for same-day targets
    if "same_day" in target_lower or "oc_same_day" in target_lower:
        # Same-day open-to-close: ~6.5 hours = 390 minutes
        return 390
    
    # Load config if not provided
    if config is None:
        try:
            from TRAINING.ranking.utils.leakage_filtering import _load_leakage_config
            config = _load_leakage_config()
        except Exception:
            pass
    
    # Default patterns (from excluded_features.yaml structure)
    # CRITICAL FIX: Use trading days calendar for day-based horizons
    # Trading session = 6.5 hours = 390 minutes (9:30 AM - 4:00 PM ET)
    # This matches the calendar used by target labels (trading days, not calendar days)
    patterns = [
        {'regex': r'(\d+)m', 'multiplier': 1},      # 60m -> 60
        {'regex': r'(\d+)h', 'multiplier': 60},     # 2h -> 120
        {'regex': r'(\d+)d', 'multiplier': 390},   # 1d -> 390 (trading session), 5d -> 1950 (5 trading sessions)
    ]
    
    if config and 'horizon_extraction' in config:
        patterns = config['horizon_extraction'].get('patterns', patterns)
    
    for pattern_config in patterns:
        regex = pattern_config.get('regex')
        multiplier = pattern_config.get('multiplier', 1)
        
        if regex:
            match = re.search(regex, target_name, re.IGNORECASE)
            if match:
                value = int(match.group(1))
                return value * multiplier
    
    # No match found - return None (do NOT default silently)
    return None


# ============================================================================
# Tracker Input Adapter (String/Enum-Safe)
# ============================================================================

def tracker_input_adapter(value: Any, field_name: str = "value") -> str:
    """
    Adapt value for tracker input (handles both strings and Enum-like objects).
    
    This prevents 'str' object has no attribute 'name' errors.
    
    Args:
        value: Value to adapt (can be str, Enum, or object with .name/.value attribute)
        field_name: Name of field (for error messages)
    
    Returns:
        String representation of value
    
    Examples:
        "CROSS_SECTIONAL" -> "CROSS_SECTIONAL"
        Enum.CROSS_SECTIONAL -> "CROSS_SECTIONAL" (if has .name)
        TaskSpec(...) -> "regression" (if has .task attribute)
    """
    if value is None:
        return None
    
    # If already a string, return as-is
    if isinstance(value, str):
        return value
    
    # Try .name attribute (for Enums)
    if hasattr(value, 'name'):
        try:
            return str(value.name)
        except Exception:
            pass
    
    # Try .value attribute (for Enums with values)
    if hasattr(value, 'value'):
        try:
            return str(value.value)
        except Exception:
            pass
    
    # Try common attribute names (task, objective, etc.)
    for attr in ['task', 'objective', 'stage', 'route_type', 'family']:
        if hasattr(value, attr):
            try:
                attr_value = getattr(value, attr)
                # Recursively adapt if it's not a string
                if isinstance(attr_value, str):
                    return attr_value
                return tracker_input_adapter(attr_value, attr)
            except Exception:
                pass
    
    # Fallback: convert to string
    return str(value)


# ============================================================================
# Feature Drop Reason Tracking
# ============================================================================

class FeatureDropReason:
    """Reason codes for feature drops"""
    MISSING_FROM_POLARS = "missing_from_polars"
    DROPPED_BY_DTYPE = "dropped_by_dtype"
    DROPPED_BY_NAN = "dropped_by_nan"
    DROPPED_BY_REGISTRY = "dropped_by_registry"
    DROPPED_BY_LOOKBACK = "dropped_by_lookback"
    DROPPED_BY_TARGET_CONDITIONAL = "dropped_by_target_conditional"
    UNKNOWN = "unknown"


def track_feature_drops(
    requested: List[str],
    allowed: List[str],
    kept: List[str],
    used: List[str],
    drop_reasons: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Track feature drops through the pipeline.
    
    Args:
        requested: Features requested (from feature selection)
        allowed: Features allowed by registry/filtering
        kept: Features kept after dtype/nan filtering
        used: Features actually used in training
        drop_reasons: Optional dict mapping feature -> reason code
    
    Returns:
        Dict with drop statistics and lists
    """
    requested_set = set(requested) if requested else set()
    allowed_set = set(allowed) if allowed else set()
    kept_set = set(kept) if kept else set()
    used_set = set(used) if used else set()
    
    stats = {
        "requested": len(requested_set),
        "allowed": len(allowed_set),
        "kept": len(kept_set),
        "used": len(used_set),
        "dropped_by_registry": list(requested_set - allowed_set),
        "dropped_by_dtype_nan": list(allowed_set - kept_set),
        "dropped_after_kept": list(kept_set - used_set),
        "drop_reasons": drop_reasons or {}
    }
    
    # Calculate ratios
    if stats["requested"] > 0:
        stats["allowed_ratio"] = stats["allowed"] / stats["requested"]
        stats["kept_ratio"] = stats["kept"] / stats["requested"]
        stats["used_ratio"] = stats["used"] / stats["requested"]
    else:
        stats["allowed_ratio"] = 0.0
        stats["kept_ratio"] = 0.0
        stats["used_ratio"] = 0.0
    
    return stats


def validate_feature_drops(stats: Dict[str, Any], threshold: float = 0.5, target: str = "unknown") -> bool:
    """
    Validate that feature drops are within acceptable threshold.
    
    Args:
        stats: Stats from track_feature_drops
        threshold: Minimum ratio of used/requested (default: 0.5 = 50%)
        target: Target name (for error messages)
    
    Returns:
        True if valid, False if excessive drops
    
    Raises:
        ValueError: If drops exceed threshold and strict mode
    """
    if stats["requested"] == 0:
        logger.warning(f"[{target}] No features requested - cannot validate drops")
        return False
    
    used_ratio = stats["used_ratio"]
    
    if used_ratio < threshold:
        error_msg = (
            f"🚨 CRITICAL [{target}]: Excessive feature drops detected. "
            f"Requested={stats['requested']}, Used={stats['used']} "
            f"(ratio={used_ratio:.1%} < threshold={threshold:.1%}). "
            f"Dropped by registry: {len(stats['dropped_by_registry'])}, "
            f"Dropped by dtype/nan: {len(stats['dropped_by_dtype_nan'])}, "
            f"Dropped after kept: {len(stats['dropped_after_kept'])}. "
            f"This indicates a pipeline bug."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return True

