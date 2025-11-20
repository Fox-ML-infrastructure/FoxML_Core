"""
Target-Aware Leakage Filtering Utilities

Provides reusable functions for filtering features based on target horizon
to prevent temporal overlap leakage.
"""

import re
from pathlib import Path
from typing import List, Set, Optional, Dict
import yaml
import logging

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def load_exclusion_config(config_path: Path = None) -> dict:
    """Load feature exclusion configuration"""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "excluded_features.yaml"
    
    if not config_path.exists():
        logger.warning(f"Exclusion config not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def extract_target_horizon(target_name: str) -> Optional[Dict[str, any]]:
    """
    Extract horizon information from target name.
    
    Examples:
        fwd_ret_20d -> {'value': 20, 'unit': 'd', 'minutes': 20*1440}
        fwd_ret_60m -> {'value': 60, 'unit': 'm', 'minutes': 60}
        fwd_ret_oc_same_day -> {'value': 1, 'unit': 'd', 'minutes': 1440}
        peak_60m_0.8 -> {'value': 60, 'unit': 'm', 'minutes': 60}
        y_will_peak_60m_0.8 -> {'value': 60, 'unit': 'm', 'minutes': 60}
        y_will_peak_mfe_10m_0.001 -> {'value': 10, 'unit': 'm', 'minutes': 10}
    
    Returns:
        Dict with 'value', 'unit', 'minutes' or None if can't parse
    """
    # Special case: open-to-close same day (treat as 1 day)
    if target_name == 'fwd_ret_oc_same_day':
        return {'value': 1, 'unit': 'd', 'minutes': 1440}
    
    # Pattern for fwd_ret_XXd or fwd_ret_XXm
    fwd_ret_match = re.match(r'fwd_ret_(\d+)([dm])', target_name)
    if fwd_ret_match:
        value = int(fwd_ret_match.group(1))
        unit = fwd_ret_match.group(2)
        minutes = value * 1440 if unit == 'd' else value
        return {'value': value, 'unit': unit, 'minutes': minutes}
    
    # Pattern for XXm in target name (e.g., peak_60m, valley_30m, y_will_peak_mfe_10m_0.001)
    # This will match the first occurrence of XXm in the name
    minutes_match = re.search(r'(\d+)m', target_name)
    if minutes_match:
        value = int(minutes_match.group(1))
        return {'value': value, 'unit': 'm', 'minutes': value}
    
    # Pattern for XXd in target name
    days_match = re.search(r'(\d+)d', target_name)
    if days_match:
        value = int(days_match.group(1))
        return {'value': value, 'unit': 'd', 'minutes': value * 1440}
    
    return None


def get_temporal_overlap_features(
    horizon_minutes: int, 
    all_feature_names: List[str] = None,
    config: dict = None
) -> Set[str]:
    """
    Get features that would create temporal overlap with target horizon.
    
    For a target with horizon H minutes, exclude features with windows:
    - Exactly H minutes (direct overlap)
    - H/2 to H*1.5 minutes (strong autocorrelation)
    - For daily targets: exclude features with matching day windows
    
    Args:
        horizon_minutes: Target prediction horizon in minutes
        config: Exclusion config (loads from file if None)
    
    Returns:
        Set of feature names to exclude
    """
    if config is None:
        config = load_exclusion_config()
    
    excluded = set()
    
    # Get all temporal overlap patterns from config
    temporal_patterns = config.get('temporal_overlap_30m_plus', [])
    
    # For minute-based horizons, exclude features with matching windows
    if horizon_minutes <= 1440:  # <= 1 day
        # Exclude features with windows in range [horizon/2, horizon*1.5]
        min_window = max(1, horizon_minutes // 2)
        max_window = int(horizon_minutes * 1.5)
        
        # Pattern to match: ret_XXm, vol_XXm, etc.
        for pattern in temporal_patterns:
            # Extract window size if it's a minute-based feature
            match = re.search(r'(\d+)m', pattern)
            if match:
                window_minutes = int(match.group(1))
                if min_window <= window_minutes <= max_window:
                    excluded.add(pattern)
    
    # If we have actual feature names, dynamically scan for matching windows
    if all_feature_names:
        # For minute-based horizons
        if horizon_minutes <= 1440:  # <= 1 day
            min_window = max(1, horizon_minutes // 2)
            max_window = int(horizon_minutes * 1.5)
            for feature_name in all_feature_names:
                matches = re.finditer(r'(\d+)m', feature_name)
                for match in matches:
                    window_minutes = int(match.group(1))
                    if (min_window <= window_minutes <= max_window) or (window_minutes >= horizon_minutes):
                        excluded.add(feature_name)
                        break
        
        # For day-based horizons
        elif horizon_minutes > 1440:  # > 1 day
            horizon_days = horizon_minutes / 1440
            min_window_days = max(1, horizon_days / 2)
            max_window_days = horizon_days * 1.5
            
            for feature_name in all_feature_names:
                # Pattern 1: XXd
                matches = re.finditer(r'(\d+)d', feature_name)
                for match in matches:
                    window_days = int(match.group(1))
                    if (min_window_days <= window_days <= max_window_days) or (window_days >= horizon_days):
                        excluded.add(feature_name)
                        break
                
                # Pattern 2: _XX (no suffix)
                matches = re.finditer(r'_(\d+)(?:_|$)', feature_name)
                for match in matches:
                    window_value = int(match.group(1))
                    if 1 <= window_value <= 365:
                        window_days = window_value
                        if (min_window_days <= window_days <= max_window_days) or (window_days >= horizon_days):
                            excluded.add(feature_name)
                            break
            
            # Minute-based features that overlap with day horizon
            for feature_name in all_feature_names:
                matches = re.finditer(r'(\d+)m', feature_name)
                for match in matches:
                    window_minutes = int(match.group(1))
                    window_days = window_minutes / 1440
                    if (min_window_days <= window_days <= max_window_days) or (window_days >= horizon_days):
                        excluded.add(feature_name)
                        break
    else:
        # Fallback: use config patterns only (old behavior)
        if horizon_minutes <= 1440:
            min_window = max(1, horizon_minutes // 2)
            max_window = int(horizon_minutes * 1.5)
            # If we have actual feature names, dynamically scan
            if all_feature_names:
                for feature_name in all_feature_names:
                    # Pattern 1: XXd
                    matches = re.finditer(r'(\d+)d', feature_name)
                    for match in matches:
                        window_days = int(match.group(1))
                        if (min_window_days <= window_days <= max_window_days) or (window_days >= horizon_days):
                            excluded.add(feature_name)
                            break
                    # Pattern 2: _XX (no suffix)
                    matches = re.finditer(r'_(\d+)(?:_|$)', feature_name)
                    for match in matches:
                        window_value = int(match.group(1))
                        if 1 <= window_value <= 365:
                            window_days = window_value
                            if (min_window_days <= window_days <= max_window_days) or (window_days >= horizon_days):
                                excluded.add(feature_name)
                                break
                # Minute-based features that overlap
                for feature_name in all_feature_names:
                    matches = re.finditer(r'(\d+)m', feature_name)
                    for match in matches:
                        window_minutes = int(match.group(1))
                        window_days = window_minutes / 1440
                        if (min_window_days <= window_days <= max_window_days) or (window_days >= horizon_days):
                            excluded.add(feature_name)
                            break
                    if min_window <= window_minutes <= max_window:
                        excluded.add(pattern)
        elif horizon_minutes > 1440:
            horizon_days = horizon_minutes / 1440
            min_window_days = max(1, horizon_days / 2)
            max_window_days = horizon_days * 1.5
            for pattern in temporal_patterns:
                match = re.search(r'(\d+)d', pattern)
                if match:
                    window_days = int(match.group(1))
                    if min_window_days <= window_days <= max_window_days:
                        excluded.add(pattern)
                match = re.search(r'(\d+)m', pattern)
                if match:
                    window_minutes = int(match.group(1))
                    window_days = window_minutes / 1440
                    if min_window_days <= window_days <= max_window_days:
                        excluded.add(pattern)
    
    return excluded


def get_excluded_features_for_target(
    target_name: str,
    all_feature_names: List[str] = None,
    config: dict = None,
    exclude_definite_leaks: bool = True,
    exclude_temporal_overlap: bool = True
) -> Set[str]:
    """
    Get set of features to exclude for a specific target.
    
    This is target-aware filtering that:
    1. Always excludes definite leaks
    2. Excludes temporal overlap features based on target horizon
    3. Excludes metadata and target columns
    
    Args:
        target_name: Name of the target (e.g., 'fwd_ret_20d', 'peak_60m_0.8')
        config: Exclusion config (loads from file if None)
        exclude_definite_leaks: Whether to exclude definite leaks
        exclude_temporal_overlap: Whether to exclude temporal overlap features
    
    Returns:
        Set of feature names to exclude
    """
    if config is None:
        config = load_exclusion_config()
    
    excluded = set()
    
    # Always exclude definite leaks
    if exclude_definite_leaks and config.get('exclude_definite_leaks', True):
        excluded.update(config.get('definite_leaks', []))
    
    # Exclude temporal overlap features based on target horizon
    if exclude_temporal_overlap:
        horizon_info = extract_target_horizon(target_name)
        if horizon_info:
            temporal_overlap = get_temporal_overlap_features(
                horizon_info['minutes'], 
                all_feature_names=all_feature_names,
                config=config
            )
            excluded.update(temporal_overlap)
            
            # SPECIAL CASE: For forward return targets, exclude ALL return/volatility/momentum
            # features with ANY day-based window (they're inherently autocorrelated)
            if target_name.startswith('fwd_ret_') and all_feature_names:
                import re
                # Patterns that indicate return/volatility/momentum features
                leaky_patterns = [
                    r'ret.*\d+[dm]',      # returns_5d, ret_20d, etc.
                    r'vol.*\d+[dm]',      # volatility_5d, vol_20d, etc.
                    r'mom.*\d+[dm]',      # momentum_5d, mom_20d, etc.
                    r'returns_\d+[dm]',   # returns_5d, returns_20d
                    r'volatility_\d+[dm]', # volatility_5d, volatility_20d
                    r'price_momentum_\d+[dm]',  # price_momentum_5d
                    r'sector_momentum_\d+[dm]', # sector_momentum_5d
                    r'volume_momentum_\d+[dm]',  # volume_momentum_5d
                    r'relative_performance_\d+[dm]', # relative_performance_5d
                    r'ret_x_mom_\d+[dm]', # ret_x_mom_5d
                    r'vol_x_mom_\d+[dm]', # vol_x_mom_5d
                ]
                
                for feature_name in all_feature_names:
                    # Check if feature matches any leaky pattern
                    for pattern in leaky_patterns:
                        if re.search(pattern, feature_name, re.IGNORECASE):
                            excluded.add(feature_name)
                            break
                    
                    # Also check for _XX patterns (no suffix) that might be days
                    # Only for return/volatility/momentum features
                    if any(term in feature_name.lower() for term in ['ret', 'vol', 'mom', 'momentum']):
                        matches = list(re.finditer(r'_(\d+)(?:_|$)', feature_name))
                        for match in matches:
                            value = int(match.group(1))
                            # If it's a reasonable day value (1-365) and feature is return/vol/mom related
                            if 1 <= value <= 365:
                                excluded.add(feature_name)
                                break
            
            logger.debug(
                f"Target {target_name} (horizon: {horizon_info['value']}{horizon_info['unit']}): "
                f"Excluding {len(temporal_overlap)} temporal overlap features"
            )
        else:
            # Fallback: use default temporal overlap (30m+ for 60m targets)
            if config.get('exclude_temporal_overlap', True):
                excluded.update(config.get('temporal_overlap_30m_plus', []))
    
    # Exclude metadata columns
    if config.get('exclude_metadata', True):
        excluded.update(config.get('metadata_columns', []))
    
    # Exclude target columns (patterns)
    if config.get('exclude_targets', True):
        target_patterns = config.get('target_patterns', [])
        # Don't exclude the current target itself (it's needed)
        for pattern in target_patterns:
            if not re.match(pattern, target_name):
                # This pattern doesn't match our target, so exclude features matching it
                # (We'll handle target exclusion separately in filter_features)
                pass
    
    return excluded


def filter_features_for_target(
    all_columns: List[str],
    target_name: str,
    config: dict = None,
    verbose: bool = True
) -> List[str]:
    """
    Filter feature list for a specific target, excluding:
    1. Definite leaks
    2. Temporal overlap features (target-aware)
    3. Metadata columns
    4. Target columns (except the current target)
    
    Args:
        all_columns: List of all column names in dataset
        target_name: Name of the target being predicted
        config: Exclusion config (loads from file if None)
        verbose: Whether to log filtering stats
    
    Returns:
        List of safe feature columns for this target
    """
    if config is None:
        config = load_exclusion_config()
    
    # Get excluded features for this target
    excluded_set = get_excluded_features_for_target(
        target_name, 
        all_feature_names=all_columns,
        config=config
    )
    
    # Get target patterns (to exclude other targets)
    target_patterns = config.get('target_patterns', []) if config.get('exclude_targets', True) else []
    
    # Filter columns
    safe_features = []
    excluded_by_name = []
    excluded_by_pattern = []
    
    for col in all_columns:
        # Skip the target itself (it's not a feature)
        if col == target_name:
            continue
        
        # Check if in exclusion set
        if col in excluded_set:
            excluded_by_name.append(col)
            continue
        
        # Check if matches target pattern (but not the current target)
        if matches_target_pattern(col, target_patterns):
            excluded_by_pattern.append(col)
            continue
        
        # Safe feature
        safe_features.append(col)
    
    # Log stats
    if verbose:
        horizon_info = extract_target_horizon(target_name)
        horizon_str = f" (horizon: {horizon_info['value']}{horizon_info['unit']})" if horizon_info else ""
        logger.info(f"Feature filtering for {target_name}{horizon_str}:")
        logger.info(f"  Total columns: {len(all_columns)}")
        logger.info(f"  Safe features: {len(safe_features)}")
        logger.info(f"  Excluded by name: {len(excluded_by_name)}")
        logger.info(f"  Excluded by pattern: {len(excluded_by_pattern)}")
    
    return safe_features


def matches_target_pattern(column_name: str, patterns: List[str]) -> bool:
    """Check if column matches any target pattern (regex)"""
    for pattern in patterns:
        if re.match(pattern, column_name):
            return True
    return False


# Backward compatibility: re-export functions from filter_leaking_features
def get_excluded_features(config: dict = None) -> Set[str]:
    """Backward compatibility wrapper"""
    return get_excluded_features_for_target("", config, exclude_temporal_overlap=False)


def filter_features(
    all_columns: List[str],
    config: dict = None,
    verbose: bool = True
) -> List[str]:
    """Backward compatibility wrapper (target-agnostic filtering)"""
    if config is None:
        config = load_exclusion_config()
    
    excluded_set = get_excluded_features(config)
    target_patterns = config.get('target_patterns', []) if config.get('exclude_targets', True) else []
    
    safe_features = []
    for col in all_columns:
        if col in excluded_set:
            continue
        if matches_target_pattern(col, target_patterns):
            continue
        safe_features.append(col)
    
    if verbose:
        logger.info(f"Feature filtering (target-agnostic):")
        logger.info(f"  Total columns: {len(all_columns)}")
        logger.info(f"  Safe features: {len(safe_features)}")
    
    return safe_features

