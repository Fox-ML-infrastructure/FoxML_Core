"""
Copyright (c) 2025 Fox ML Infrastructure

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
Target-Aware Leakage Filtering

Filters out features that would leak information about the target being predicted.
Uses temporal awareness: features computed at time t cannot use information from
time t+horizon or later.

All exclusion patterns are loaded from CONFIG/excluded_features.yaml - no hardcoded patterns.
"""


import re
import yaml
from typing import List, Set, Optional, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Cache for loaded config
_LEAKAGE_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "CONFIG" / "excluded_features.yaml"
_CONFIG_MTIME: Optional[float] = None  # Track file modification time for cache invalidation


def _load_leakage_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load leakage filtering configuration from YAML file (cached).
    
    Args:
        force_reload: If True, reload config even if cached
    
    Returns:
        Config dictionary
    """
    global _LEAKAGE_CONFIG, _CONFIG_MTIME
    
    # Check if config file was modified (cache invalidation)
    if _LEAKAGE_CONFIG is not None and not force_reload:
        if _CONFIG_PATH.exists():
            current_mtime = _CONFIG_PATH.stat().st_mtime
            if _CONFIG_MTIME is not None and current_mtime > _CONFIG_MTIME:
                # File was modified, clear cache
                logger.info(f"Config file modified, reloading from {_CONFIG_PATH}")
                _LEAKAGE_CONFIG = None
                _CONFIG_MTIME = None
        elif _CONFIG_MTIME is not None:
            # File was deleted, clear cache
            logger.warning(f"Config file deleted, clearing cache")
            _LEAKAGE_CONFIG = None
            _CONFIG_MTIME = None
    
    if _LEAKAGE_CONFIG is not None:
        return _LEAKAGE_CONFIG
    
    if not _CONFIG_PATH.exists():
        logger.warning(f"Leakage config not found: {_CONFIG_PATH}, using empty config")
        _LEAKAGE_CONFIG = {
            'always_exclude': {'regex_patterns': [], 'prefix_patterns': [], 'keyword_patterns': [], 'exact_patterns': []},
            'target_type_rules': {},
            'target_classification': {},
            'horizon_extraction': {'patterns': []},
            'metadata_columns': [],
            'config': {}
        }
        return _LEAKAGE_CONFIG
    
    try:
        with open(_CONFIG_PATH, 'r') as f:
            _LEAKAGE_CONFIG = yaml.safe_load(f) or {}
        
        # Store modification time for cache invalidation
        _CONFIG_MTIME = _CONFIG_PATH.stat().st_mtime
        
        # Ensure all required keys exist with defaults
        defaults = {
            'always_exclude': {'regex_patterns': [], 'prefix_patterns': [], 'keyword_patterns': [], 'exact_patterns': []},
            'target_type_rules': {},
            'target_classification': {},
            'horizon_extraction': {'patterns': []},
            'metadata_columns': [],
            'config': {}
        }
        
        for key, default_value in defaults.items():
            if key not in _LEAKAGE_CONFIG:
                _LEAKAGE_CONFIG[key] = default_value
            elif isinstance(default_value, dict):
                for subkey, subdefault in default_value.items():
                    if subkey not in _LEAKAGE_CONFIG[key]:
                        _LEAKAGE_CONFIG[key][subkey] = subdefault
        
        # Validate that we actually loaded patterns (not empty config)
        always_exclude = _LEAKAGE_CONFIG.get('always_exclude', {})
        total_patterns = (
            len(always_exclude.get('regex_patterns', [])) +
            len(always_exclude.get('prefix_patterns', [])) +
            len(always_exclude.get('keyword_patterns', [])) +
            len(always_exclude.get('exact_patterns', []))
        )
        
        if total_patterns == 0:
            logger.warning(
                f"⚠️  WARNING: Config loaded but has ZERO exclusion patterns! "
                f"This will allow all features (including leaks). "
                f"Check {_CONFIG_PATH}"
            )
        else:
            logger.debug(f"Loaded leakage config from {_CONFIG_PATH} ({total_patterns} patterns)")
        
        return _LEAKAGE_CONFIG
    except Exception as e:
        logger.error(f"Failed to load leakage config: {e}, using empty config")
        _LEAKAGE_CONFIG = {
            'always_exclude': {'regex_patterns': [], 'prefix_patterns': [], 'keyword_patterns': [], 'exact_patterns': []},
            'target_type_rules': {},
            'target_classification': {},
            'horizon_extraction': {'patterns': []},
            'metadata_columns': [],
            'config': {}
        }
        return _LEAKAGE_CONFIG


def filter_features_for_target(
    all_columns: List[str],
    target_column: str,
    verbose: bool = False
) -> List[str]:
    """
    Filter features that would leak information about the target.
    
    All exclusion patterns are loaded from CONFIG/excluded_features.yaml.
    Works with any dataset, features, and targets - fully configurable.
    
    Args:
        all_columns: List of all column names in the dataset
        target_column: Name of the target column being predicted
        verbose: If True, log excluded features
    
    Returns:
        List of safe feature column names
    """
    config = _load_leakage_config()
    
    # Start with all columns except the target itself
    safe_columns = [c for c in all_columns if c != target_column]
    
    # Exclude metadata columns if configured
    if config.get('config', {}).get('exclude_metadata', True):
        metadata = config.get('metadata_columns', [])
        excluded_metadata = [c for c in safe_columns if c in metadata]
        safe_columns = [c for c in safe_columns if c not in metadata]
        if excluded_metadata and verbose:
            logger.info(f"  Excluded {len(excluded_metadata)} metadata columns")
    
    # Get target metadata
    target_type = _classify_target_type(target_column, config)
    target_horizon = _extract_horizon(target_column, config)
    
    # Apply always-exclude patterns (regardless of target type)
    always_exclude = config.get('always_exclude', {})
    excluded_always = _apply_exclusion_patterns(safe_columns, always_exclude, "always-exclude")
    safe_columns = [c for c in safe_columns if c not in excluded_always]
    if excluded_always and verbose:
        logger.info(f"  Excluded {len(excluded_always)} always-excluded features")
    
    # Apply target-specific filtering rules
    if target_type == 'forward_return':
        safe_columns = _filter_for_forward_return_target(
            safe_columns, target_column, target_horizon, config, verbose
        )
    elif target_type == 'barrier':
        safe_columns = _filter_for_barrier_target(
            safe_columns, target_column, target_horizon, config, verbose
        )
    elif target_type == 'first_touch':
        # First touch targets use barrier rules if configured
        first_touch_rules = config.get('target_type_rules', {}).get('first_touch', {})
        if first_touch_rules.get('use_barrier_rules', True):
            safe_columns = _filter_for_barrier_target(
                safe_columns, target_column, target_horizon, config, verbose
            )
        else:
            # Apply first_touch specific rules if defined
            first_touch_exclude = _get_target_type_exclude_patterns('first_touch', config)
            excluded_ft = _apply_exclusion_patterns(safe_columns, first_touch_exclude, "first_touch")
            safe_columns = [c for c in safe_columns if c not in excluded_ft]
            if excluded_ft and verbose:
                logger.info(f"  Excluded {len(excluded_ft)} features for first_touch target")
    
    return safe_columns


def _classify_target_type(target_column: str, config: Dict[str, Any]) -> str:
    """Classify target type from column name using config rules."""
    classification = config.get('target_classification', {})
    
    # Check forward_return
    fr_config = classification.get('forward_return', {})
    if fr_config.get('prefix') and target_column.startswith(fr_config['prefix']):
        return 'forward_return'
    
    # Check barrier
    barrier_config = classification.get('barrier', {})
    if barrier_config.get('prefix') and target_column.startswith(barrier_config['prefix']):
        return 'barrier'
    
    # Check first_touch
    ft_config = classification.get('first_touch', {})
    if ft_config.get('keyword') and ft_config['keyword'] in target_column:
        if ft_config.get('prefix') and target_column.startswith(ft_config['prefix']):
            return 'first_touch'
    
    return 'unknown'


def _extract_horizon(target_column: str, config: Dict[str, Any]) -> Optional[int]:
    """
    Extract horizon from target column name (in minutes) using config patterns.
    
    Examples:
        fwd_ret_60m -> 60
        y_will_peak_15m_0.8 -> 15
        fwd_ret_1d -> 1440 (assuming 1d = 1440 minutes)
    """
    horizon_config = config.get('horizon_extraction', {})
    patterns = horizon_config.get('patterns', [])
    
    for pattern_config in patterns:
        regex = pattern_config.get('regex')
        multiplier = pattern_config.get('multiplier', 1)
        
        if regex:
            match = re.search(regex, target_column)
            if match:
                value = int(match.group(1))
                return value * multiplier
    
    return None


def _get_target_type_exclude_patterns(target_type: str, config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Get exclusion patterns for a specific target type."""
    target_rules = config.get('target_type_rules', {}).get(target_type, {})
    
    return {
        'regex_patterns': target_rules.get('regex_patterns', []),
        'prefix_patterns': target_rules.get('prefix_patterns', []),
        'keyword_patterns': target_rules.get('keyword_patterns', []),
        'exact_patterns': target_rules.get('exact_patterns', [])
    }


def _apply_exclusion_patterns(
    columns: List[str],
    patterns: Dict[str, List[str]],
    pattern_type: str = ""
) -> List[str]:
    """
    Apply exclusion patterns to a list of columns.
    
    Args:
        columns: List of column names to filter
        patterns: Dict with keys: regex_patterns, prefix_patterns, keyword_patterns, exact_patterns
        pattern_type: Label for logging (optional)
    
    Returns:
        List of excluded column names
    """
    excluded = []
    
    # Apply regex patterns
    for pattern in patterns.get('regex_patterns', []):
        try:
            regex = re.compile(pattern)
            for col in columns:
                if col not in excluded and regex.match(col):
                    excluded.append(col)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{pattern}' in {pattern_type}: {e}")
    
    # Apply prefix patterns
    for prefix in patterns.get('prefix_patterns', []):
        for col in columns:
            if col not in excluded and col.startswith(prefix):
                excluded.append(col)
    
    # Apply keyword patterns (substring match, case-insensitive)
    for keyword in patterns.get('keyword_patterns', []):
        keyword_lower = keyword.lower()
        for col in columns:
            if col not in excluded and keyword_lower in col.lower():
                excluded.append(col)
    
    # Apply exact patterns
    exact_set = set(patterns.get('exact_patterns', []))
    for col in columns:
        if col not in excluded and col in exact_set:
            excluded.append(col)
    
    return excluded


def _filter_for_forward_return_target(
    columns: List[str],
    target_column: str,
    target_horizon: Optional[int],
    config: Dict[str, Any],
    verbose: bool
) -> List[str]:
    """
    Filter features for forward return targets using config rules.
    """
    excluded = []
    safe = []
    
    target_rules = config.get('target_type_rules', {}).get('forward_return', {})
    horizon_overlap = target_rules.get('horizon_overlap', {})
    
    for col in columns:
        should_exclude = False
        reason = None
        
        # Check if we should exclude ALL forward returns
        if horizon_overlap.get('exclude_all', False):
            if col.startswith('fwd_ret_'):
                should_exclude = True
                reason = "forward return (excluded for all targets)"
        # Check horizon overlap if enabled (and not excluding all)
        elif horizon_overlap.get('enabled', True) and target_horizon is not None:
            if col.startswith('fwd_ret_'):
                col_horizon = _extract_horizon(col, config)
                if col_horizon is not None:
                    exclude_if_ge = horizon_overlap.get('exclude_if_ge', True)
                    if exclude_if_ge and col_horizon >= target_horizon:
                        should_exclude = True
                        reason = "overlapping forward return"
        
        # Apply target-type-specific exclusion patterns
        fr_exclude = _get_target_type_exclude_patterns('forward_return', config)
        if col in _apply_exclusion_patterns([col], fr_exclude, "forward_return"):
            should_exclude = True
            reason = reason or "forward_return exclusion pattern"
        
        if not should_exclude:
            safe.append(col)
        elif verbose and reason:
            excluded.append((col, reason))
    
    if verbose and excluded:
        logger.info(f"  Excluded {len(excluded)} features for forward return target:")
        for col, reason in excluded[:10]:
            logger.info(f"    - {col}: {reason}")
        if len(excluded) > 10:
            logger.info(f"    ... and {len(excluded) - 10} more")
    
    return safe


def _filter_for_barrier_target(
    columns: List[str],
    target_column: str,
    target_horizon: Optional[int],
    config: Dict[str, Any],
    verbose: bool
) -> List[str]:
    """
    Filter features for barrier targets using config rules.
    
    Target-aware filtering:
    - Peak targets: exclude zigzag_high (but keep zigzag_low)
    - Valley targets: exclude zigzag_low (but keep zigzag_high)
    - CRITICAL: Exclude features with matching horizon (temporal overlap)
    """
    excluded = []
    safe = []
    
    # Determine if this is a peak or valley target
    is_peak_target = 'peak' in target_column.lower()
    is_valley_target = 'valley' in target_column.lower()
    
    # Get barrier-specific config
    barrier_rules = config.get('target_type_rules', {}).get('barrier', {})
    horizon_overlap = barrier_rules.get('horizon_overlap', {})
    exclude_matching_horizon = horizon_overlap.get('exclude_matching_horizon', True)
    exclude_overlapping_horizon = horizon_overlap.get('exclude_overlapping_horizon', True)
    
    for col in columns:
        should_exclude = False
        reason = None
        
        # CRITICAL: Exclude features with matching horizon ONLY if they're forward-looking
        # Past features (volatility_15m, rsi_15m) are VALID even if they match target horizon (fwd_ret_15m)
        # Only forward-looking features (fwd_ret_15m, next_15m_high) leak information
        if exclude_matching_horizon and target_horizon is not None:
            col_horizon = _extract_horizon(col, config)
            if col_horizon is not None:
                # CRITICAL FIX: Only exclude if feature is forward-looking (uses future data)
                # Past features with matching horizon are valid predictors (e.g., volatility_15m for fwd_ret_15m)
                is_forward_looking = (
                    col.startswith('fwd_ret_') or
                    col.startswith('fwd_') or
                    col.startswith('y_') or
                    col.startswith('p_') or
                    col.startswith('barrier_') or
                    col.startswith('next_') or
                    col.startswith('future_') or
                    'forward' in col.lower() or
                    'future' in col.lower()
                )
                
                if col_horizon == target_horizon and is_forward_looking:
                    should_exclude = True
                    reason = f"temporal overlap (forward-looking feature horizon {col_horizon}m matches target horizon {target_horizon}m)"
                # Only exclude overlapping horizons for forward-looking features (fwd_ret_*, y_*, etc.)
                # Standard technical indicators (RSI, MA, volatility) computed on past data are safe
                # regardless of horizon - they represent causality, not leakage
                elif exclude_overlapping_horizon and col_horizon >= target_horizon / 4:
                    # Check if this is a forward-looking feature (target-like feature)
                    is_forward_looking = (
                        col.startswith('fwd_ret_') or
                        col.startswith('y_') or
                        col.startswith('p_') or
                        col.startswith('barrier_') or
                        'forward' in col.lower() or
                        'future' in col.lower()
                    )
                    if is_forward_looking:
                        should_exclude = True
                        reason = f"overlapping horizon (forward-looking feature {col_horizon}m >= target {target_horizon}m/4)"
        
        # CRITICAL: Exclude HIGH-based features for peak targets ONLY if they're forward-looking
        # Past data (bollinger_upper, rolling_max, daily_high) is VALID - it represents historical resistance
        # Future data (fwd_ret_*, next_10_candles_high) is LEAKAGE - it uses future information
        if is_peak_target and not should_exclude:
            col_lower = col.lower()
            # Only exclude if it's a forward-looking feature (starts with fwd_, future_, next_, etc.)
            is_forward_looking = (
                col_lower.startswith('fwd_') or
                col_lower.startswith('future_') or
                col_lower.startswith('next_') or
                'forward' in col_lower or
                'future' in col_lower
            )
            
            if is_forward_looking and any(kw in col_lower for kw in ['high', 'upper', 'max', 'top', 'ceiling']):
                should_exclude = True
                reason = "Forward-looking HIGH-based feature (excluded for peak targets - uses future information)"
        
        # CRITICAL: Exclude LOW-based features for valley targets ONLY if they're forward-looking
        # Past data (bollinger_lower, rolling_min, daily_low) is VALID - it represents historical support
        # Future data (fwd_ret_*, next_10_candles_low) is LEAKAGE - it uses future information
        if is_valley_target and not should_exclude:
            col_lower = col.lower()
            # Only exclude if it's a forward-looking feature (starts with fwd_, future_, next_, etc.)
            is_forward_looking = (
                col_lower.startswith('fwd_') or
                col_lower.startswith('future_') or
                col_lower.startswith('next_') or
                'forward' in col_lower or
                'future' in col_lower
            )
            
            if is_forward_looking and any(kw in col_lower for kw in ['low', 'lower', 'min', 'bottom', 'floor']):
                should_exclude = True
                reason = "Forward-looking LOW-based feature (excluded for valley targets - uses future information)"
        
        # Apply target-type-specific exclusion patterns
        if not should_exclude:
            barrier_exclude = _get_target_type_exclude_patterns('barrier', config)
            
            # Get keyword patterns
            keyword_patterns = barrier_exclude.get('keyword_patterns', [])
            
            # Apply keyword patterns with target-aware logic for zigzag features
            for keyword in keyword_patterns:
                keyword_lower = keyword.lower()
                if keyword_lower in col.lower():
                    # Special handling for zigzag features
                    if keyword_lower == 'zigzag_high':
                        # Only exclude zigzag_high for peak targets
                        if is_peak_target:
                            should_exclude = True
                            reason = "zigzag_high (excluded for peak targets)"
                        # Keep zigzag_high for valley targets
                    elif keyword_lower == 'zigzag_low':
                        # Only exclude zigzag_low for valley targets
                        if is_valley_target:
                            should_exclude = True
                            reason = "zigzag_low (excluded for valley targets)"
                        # Keep zigzag_low for peak targets
                    else:
                        # For other keywords (peak, valley, swing, first_touch), apply normally
                        should_exclude = True
                        reason = f"barrier keyword pattern: {keyword}"
                    break
            
            # Apply other exclusion patterns (regex, prefix, exact)
            if not should_exclude:
                other_patterns = {
                    'regex_patterns': barrier_exclude.get('regex_patterns', []),
                    'prefix_patterns': barrier_exclude.get('prefix_patterns', []),
                    'exact_patterns': barrier_exclude.get('exact_patterns', [])
                }
                if col in _apply_exclusion_patterns([col], other_patterns, "barrier"):
                    should_exclude = True
                    reason = "barrier exclusion pattern"
        
        # Special case: exclude other barrier targets (but allow the current target)
        # This is handled by the always-exclude y_* pattern, but we check here for clarity
        if col.startswith('y_will_') and col != target_column:
            should_exclude = True
            reason = "other barrier target"
        
        # Special case: keyword patterns should not exclude the target itself
        if should_exclude and col == target_column:
            should_exclude = False
            reason = None
        
        if not should_exclude:
            safe.append(col)
        elif verbose and reason:
            excluded.append((col, reason))
    
    if verbose and excluded:
        logger.info(f"  Excluded {len(excluded)} features for barrier target:")
        for col, reason in excluded[:10]:
            logger.info(f"    - {col}: {reason}")
        if len(excluded) > 10:
            logger.info(f"    ... and {len(excluded) - 10} more")
    
    return safe
