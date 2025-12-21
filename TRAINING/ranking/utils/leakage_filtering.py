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
_CONFIG_PATH_CACHE: Optional[Path] = None
_SCHEMA_CONFIG: Optional[Dict[str, Any]] = None
_SCHEMA_CONFIG_PATH_CACHE: Optional[Path] = None

def _load_schema_config(force_reload: bool = False) -> Dict[str, Any]:
    """
    Load feature/target schema configuration.
    
    This defines the explicit schema: what's metadata, what's a target, what's a feature.
    """
    global _SCHEMA_CONFIG, _SCHEMA_CONFIG_PATH_CACHE
    
    # Try to get schema path from config first
    schema_path = None
    if _CONFIG_AVAILABLE:
        try:
            system_cfg = get_system_config()
            config_path = system_cfg.get('system', {}).get('paths', {})
            schema_path_str = config_path.get('feature_target_schema')
            if schema_path_str:
                schema_path = Path(schema_path_str)
                if not schema_path.is_absolute():
                    script_file = Path(__file__).resolve()
                    repo_root = script_file.parents[2]
                    schema_path = repo_root / schema_path_str
            else:
                # Use config_dir from config
                config_dir = config_path.get('config_dir', 'CONFIG')
                script_file = Path(__file__).resolve()
                repo_root = script_file.parents[2]
                # Try new location first (data/), then old (root)
                schema_path = repo_root / config_dir / "data" / "feature_target_schema.yaml"
                if not schema_path.exists():
                    schema_path = repo_root / config_dir / "feature_target_schema.yaml"
        except Exception:
            pass  # Fall through to default
    
    # Fallback to default location - try new location first (data/), then old (root)
    if schema_path is None or not schema_path.exists():
        excluded_path = _find_config_path()
        # Try data/ subdirectory first
        schema_path = excluded_path.parent / "data" / "feature_target_schema.yaml"
        if not schema_path.exists():
            schema_path = excluded_path.parent / "feature_target_schema.yaml"
    
    # Use cache if available and path matches
    if not force_reload and _SCHEMA_CONFIG is not None and _SCHEMA_CONFIG_PATH_CACHE == schema_path:
        return _SCHEMA_CONFIG
    
    if not schema_path.exists():
        logger.warning(f"Schema config not found at {schema_path}, using defaults")
        return {
            'metadata_columns': ['symbol', 'interval', 'source', 'ts', 'timestamp'],
            'target_patterns': ['^y_', '^fwd_ret_', '^barrier_'],
            'feature_families': {},
            'modes': {
                'ranking': {'default_action': 'allow'},
                'training': {'default_action': 'allow'}
            }
        }
    
    try:
        with open(schema_path, 'r') as f:
            _SCHEMA_CONFIG = yaml.safe_load(f) or {}
        _SCHEMA_CONFIG_PATH_CACHE = schema_path
        logger.debug(f"Loaded schema config from {schema_path}")
        return _SCHEMA_CONFIG
    except Exception as e:
        logger.warning(f"Failed to load schema config from {schema_path}: {e}, using defaults")
        return {
            'metadata_columns': ['symbol', 'interval', 'source', 'ts', 'timestamp'],
            'target_patterns': ['^y_', '^fwd_ret_', '^barrier_'],
            'feature_families': {},
            'modes': {
                'ranking': {'default_action': 'allow'},
                'training': {'default_action': 'allow'}
            }
        }

def _is_feature_in_schema_family(feature_name: str, schema_config: Dict[str, Any], mode: str = 'ranking') -> bool:
    """
    Check if a feature matches any allowed feature family in the schema.
    
    Args:
        feature_name: Name of the feature
        schema_config: Schema configuration dict
        mode: 'ranking' or 'training'
    
    Returns:
        True if feature matches an allowed family, False otherwise
    """
    feature_lower = feature_name.lower()
    families = schema_config.get('feature_families', {})
    mode_config = schema_config.get('modes', {}).get(mode, {})
    allowed_families = mode_config.get('allow_families', [])
    
    # Check each allowed family
    for family_name in allowed_families:
        if family_name not in families:
            continue
        
        family = families[family_name]
        patterns = family.get('patterns', [])
        
        for pattern in patterns:
            # Handle exact match (ends with $)
            if pattern.endswith('$'):
                pattern_regex = pattern
            # Handle prefix match
            elif pattern.startswith('^'):
                pattern_regex = pattern
            else:
                # Convert to regex
                pattern_regex = f"^{pattern}"
            
            try:
                if re.match(pattern_regex, feature_name, re.IGNORECASE):
                    return True
            except re.error:
                # Fallback to simple string matching
                if pattern.endswith('$'):
                    if feature_name.lower() == pattern[:-1].lower():
                        return True
                elif pattern.startswith('^'):
                    if feature_lower.startswith(pattern[1:].lower()):
                        return True
                else:
                    if feature_lower.startswith(pattern.lower()):
                        return True
    
    return False

# Minimal safe feature families for ranking (always allowed, even if registry/config excludes them)
# These are baseline features that should be available for target ranking evaluation
_RANKING_SAFE_FEATURE_PATTERNS = {
    # OHLCV - core market data
    'ohlcv_exact': ['open', 'high', 'low', 'close', 'volume'],
    'ohlcv_prefixes': ['open_', 'high_', 'low_', 'close_', 'volume_'],
    
    # Returns - backward-looking only
    'returns_prefixes': ['ret_', 'returns_'],  # ret_1, ret_5, returns_1d, etc.
    
    # Volatility - backward-looking
    'volatility_prefixes': ['vol_', 'volatility_', 'atr_'],  # vol_5m, volatility_20d, atr_14
    
    # Moving averages - all are backward-looking
    'ma_prefixes': ['sma_', 'ema_', 'hma_', 'wma_', 'vwma_', 'kama_', 'tema_', 'dema_', 'hull_ma_'],
    
    # Oscillators - backward-looking
    'oscillator_prefixes': ['rsi_', 'macd', 'stoch_', 'williams_r', 'cci_', 'roc_'],
    'oscillator_exact': ['macd', 'macd_signal', 'macd_hist', 'williams_r'],
    
    # Bollinger Bands - backward-looking
    'bollinger_prefixes': ['bollinger_', 'bb_'],
    
    # Momentum - backward-looking
    'momentum_prefixes': ['mom_', 'momentum_', 'price_momentum_'],
    
    # Volume indicators - backward-looking
    'volume_prefixes': ['volume_', 'dollar_volume', 'turnover_', 'vwap', 'obv'],
    'volume_exact': ['volume', 'dollar_volume', 'vwap', 'obv', 'obv_ema'],
    
    # Trend indicators - backward-looking
    'trend_prefixes': ['adx', 'plus_di', 'minus_di', 'aroon_', 'psar'],
    'trend_exact': ['adx', 'adx_14', 'plus_di', 'minus_di', 'psar'],
    
    # Support/Resistance - backward-looking (past highs/lows)
    'support_resistance_prefixes': ['rolling_max_', 'rolling_min_', 'daily_high', 'daily_low'],
}

def _is_ranking_safe_feature(feature_name: str) -> bool:
    """
    Check if a feature is in the minimal safe feature family for ranking.
    
    These features are always allowed in ranking mode, even if registry/config excludes them.
    They represent baseline OHLCV + backward-looking TA that should be available for evaluation.
    
    Args:
        feature_name: Name of the feature to check
    
    Returns:
        True if feature is in the safe family, False otherwise
    """
    feature_lower = feature_name.lower()
    
    # Check exact matches first
    if feature_name in _RANKING_SAFE_FEATURE_PATTERNS.get('ohlcv_exact', []):
        return True
    if feature_name in _RANKING_SAFE_FEATURE_PATTERNS.get('oscillator_exact', []):
        return True
    if feature_name in _RANKING_SAFE_FEATURE_PATTERNS.get('volume_exact', []):
        return True
    if feature_name in _RANKING_SAFE_FEATURE_PATTERNS.get('trend_exact', []):
        return True
    
    # Check prefix patterns (more specific first)
    # Returns
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('returns_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    # Volatility
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('volatility_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    # Moving averages
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('ma_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    # Oscillators (handle both prefixes and exact matches like 'macd')
    for pattern in _RANKING_SAFE_FEATURE_PATTERNS.get('oscillator_prefixes', []):
        if pattern.endswith('_'):
            if feature_lower.startswith(pattern):
                return True
        else:
            # Exact match or starts with pattern
            if feature_lower == pattern or feature_lower.startswith(pattern + '_'):
                return True
    
    # Bollinger Bands
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('bollinger_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    # Momentum
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('momentum_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    # Volume (handle both prefixes and exact matches like 'vwap', 'obv')
    for pattern in _RANKING_SAFE_FEATURE_PATTERNS.get('volume_prefixes', []):
        if pattern.endswith('_'):
            if feature_lower.startswith(pattern):
                return True
        else:
            # Exact match or starts with pattern
            if feature_lower == pattern or feature_lower.startswith(pattern + '_'):
                return True
    
    # Trend indicators (handle both prefixes and exact matches like 'adx', 'psar')
    for pattern in _RANKING_SAFE_FEATURE_PATTERNS.get('trend_prefixes', []):
        if pattern.endswith('_'):
            if feature_lower.startswith(pattern):
                return True
        else:
            # Exact match or starts with pattern
            if feature_lower == pattern or feature_lower.startswith(pattern + '_'):
                return True
    
    # Support/Resistance
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('support_resistance_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    # OHLCV prefixes (check last, as they're more general)
    for prefix in _RANKING_SAFE_FEATURE_PATTERNS.get('ohlcv_prefixes', []):
        if feature_lower.startswith(prefix):
            return True
    
    return False

# Try to import config loader for path configuration
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass

# Robust path resolution: try multiple possible locations
def _find_config_path() -> Path:
    """Find the excluded_features.yaml config file using multiple strategies."""
    # Strategy 0: Use centralized config loader if available
    if _CONFIG_AVAILABLE:
        try:
            from CONFIG.config_loader import get_config_path
            config_file = get_config_path("excluded_features")
            if config_file.exists():
                return config_file
        except (ImportError, AttributeError):
            pass  # Fall through to fallback strategies
        except Exception:
            pass  # Fall through to fallback strategies
        
        # Fallback: Try system config
        try:
            system_cfg = get_system_config()
            config_path = system_cfg.get('system', {}).get('paths', {})
            excluded_path = config_path.get('excluded_features')
            if excluded_path:
                config_file = Path(excluded_path)
                if config_file.is_absolute():
                    if config_file.exists():
                        return config_file
                else:
                    # Relative path - find repo root
                    script_file = Path(__file__).resolve()
                    repo_root = script_file.parents[2]
                    config_file = repo_root / excluded_path
                    if config_file.exists():
                        return config_file
            else:
                # Use config_dir from config
                config_dir = config_path.get('config_dir', 'CONFIG')
                script_file = Path(__file__).resolve()
                repo_root = script_file.parents[2]
                # Try new location first (data/), then old (root)
                config_file = repo_root / config_dir / "data" / "excluded_features.yaml"
                if not config_file.exists():
                    config_file = repo_root / config_dir / "excluded_features.yaml"
                if config_file.exists():
                    return config_file
        except Exception:
            pass  # Fall through to fallback strategies
    
    # Strategy 1: Relative to this file (TRAINING/utils/leakage_filtering.py -> repo root)
    # Go up: TRAINING/utils/ -> TRAINING/ -> repo_root/
    script_file = Path(__file__).resolve()
    # Try new location first (data/), then old (root)
    repo_root_via_script = script_file.parents[2] / "CONFIG" / "data" / "excluded_features.yaml"
    if not repo_root_via_script.exists():
        repo_root_via_script = script_file.parents[2] / "CONFIG" / "excluded_features.yaml"
    if repo_root_via_script.exists():
        return repo_root_via_script
    
    # Strategy 2: Look for CONFIG directory in current working directory
    cwd_config = Path.cwd() / "CONFIG" / "data" / "excluded_features.yaml"
    if not cwd_config.exists():
        cwd_config = Path.cwd() / "CONFIG" / "excluded_features.yaml"
    if cwd_config.exists():
        return cwd_config
    
    # Strategy 3: Try to find repo root by looking for .git or CONFIG directory
    # Start from current working directory and walk up
    current = Path.cwd()
    for _ in range(10):  # Search up to 10 levels (more generous)
        config_path = current / "CONFIG" / "excluded_features.yaml"
        if config_path.exists():
            return config_path
        # Also check if we're at repo root (has .git or CONFIG dir)
        if (current / ".git").exists() or (current / "CONFIG").is_dir():
            if config_path.exists():
                return config_path
        current = current.parent
        if current == current.parent:  # Reached filesystem root
            break
    
    # Strategy 4: Try relative to script file again, but also check parent directories
    # Sometimes __file__ resolves differently depending on how module is imported
    script_dir = script_file.parent
    for level in range(1, 6):  # Try going up 1-5 levels from script
        potential_root = script_dir
        for _ in range(level):
            potential_root = potential_root.parent
        config_path = potential_root / "CONFIG" / "excluded_features.yaml"
        if config_path.exists():
            return config_path
    
    # Strategy 5: Walk up from script file to find .git or CONFIG marker
    # This is more reliable when CWD is wrong
    script_parent = script_file.parent
    for _ in range(10):  # Walk up from script location
        potential_config = script_parent / "CONFIG" / "excluded_features.yaml"
        if potential_config.exists():
            return potential_config
        # Check if we found repo root marker
        if (script_parent / ".git").exists() or (script_parent / "CONFIG").is_dir():
            if potential_config.exists():
                return potential_config
        script_parent = script_parent.parent
        if script_parent == script_parent.parent:  # Reached root
            break
    
    # Fallback: return the most likely path (will show error if not found)
    return repo_root_via_script

def _get_config_path() -> Path:
    """Get config path, with caching and re-evaluation if needed."""
    global _CONFIG_PATH_CACHE
    
    # Re-evaluate if cache is None or if cached path doesn't exist
    if _CONFIG_PATH_CACHE is None or not _CONFIG_PATH_CACHE.exists():
        _CONFIG_PATH_CACHE = _find_config_path()
    
    return _CONFIG_PATH_CACHE

_CONFIG_MTIME: Optional[float] = None  # Track file modification time for cache invalidation


def reload_feature_configs() -> None:
    """
    Reload feature configs (leakage config and schema config).
    
    This is useful after auto-fixer modifies configs and you want to re-evaluate
    targets with the updated configuration.
    """
    _load_leakage_config(force_reload=True)
    _load_schema_config(force_reload=True)
    logger.info("Reloaded feature configs (excluded_features.yaml, feature_registry.yaml, feature_target_schema.yaml)")


def _load_leakage_config(force_reload: bool = False) -> Dict[str, Any]:
    """Load leakage filtering configuration from YAML file (cached).
    
    Args:
        force_reload: If True, reload config even if cached
    
    Returns:
        Config dictionary
    """
    global _LEAKAGE_CONFIG, _CONFIG_MTIME, _CONFIG_PATH_CACHE
    
    # Get config path (lazy evaluation)
    config_path = _get_config_path()
    
    # Final fallback: try CWD/CONFIG if path doesn't exist
    if not config_path.exists():
        cwd_config = Path.cwd() / "CONFIG" / "excluded_features.yaml"
        if cwd_config.exists():
            logger.debug(f"Using config from CWD: {cwd_config}")
            config_path = cwd_config
            # Update cache
            _CONFIG_PATH_CACHE = cwd_config
    
    # Check if config file was modified (cache invalidation)
    if _LEAKAGE_CONFIG is not None and not force_reload:
        if config_path.exists():
            current_mtime = config_path.stat().st_mtime
            if _CONFIG_MTIME is not None and current_mtime > _CONFIG_MTIME:
                # File was modified, clear cache
                logger.info(f"Config file modified, reloading from {config_path}")
                _LEAKAGE_CONFIG = None
                _CONFIG_MTIME = None
        elif _CONFIG_MTIME is not None:
            # File was deleted, clear cache
            logger.warning(f"Config file deleted, clearing cache")
            _LEAKAGE_CONFIG = None
            _CONFIG_MTIME = None
    
    if _LEAKAGE_CONFIG is not None:
        return _LEAKAGE_CONFIG
    
    if not config_path.exists():
        # Try one more aggressive search from script file location
        script_file = Path(__file__).resolve()
        script_parent = script_file.parent
        for _ in range(10):  # Walk up from script location
            potential_config = script_parent / "CONFIG" / "excluded_features.yaml"
            if potential_config.exists():
                logger.info(f"Found config via script path walk: {potential_config}")
                config_path = potential_config
                _CONFIG_PATH_CACHE = potential_config
                break
            if (script_parent / ".git").exists() or (script_parent / "CONFIG").is_dir():
                if potential_config.exists():
                    logger.info(f"Found config via repo marker: {potential_config}")
                    config_path = potential_config
                    _CONFIG_PATH_CACHE = potential_config
                    break
            script_parent = script_parent.parent
            if script_parent == script_parent.parent:
                break
        
        if not config_path.exists():
            logger.error(
                f"⚠️  CRITICAL: Leakage config not found at {config_path}\n"
                f"   This will cause data leakage! Features like 'ts', 'p_*', and target-related features will NOT be filtered.\n"
                f"   Please ensure CONFIG/excluded_features.yaml exists in the repo root.\n"
                f"   Current working directory: {Path.cwd()}\n"
                f"   Script file location: {Path(__file__).resolve()}\n"
                f"   Using empty config (NO LEAKAGE PROTECTION)"
            )
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
        with open(config_path, 'r') as f:
            _LEAKAGE_CONFIG = yaml.safe_load(f) or {}
        
        # Store modification time for cache invalidation
        _CONFIG_MTIME = config_path.stat().st_mtime
        
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
                f"Check {config_path}"
            )
        else:
            logger.debug(f"Loaded leakage config from {config_path} ({total_patterns} patterns)")
        
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
    verbose: bool = False,
    use_registry: bool = True,
    data_interval_minutes: int = 5,
    for_ranking: bool = False,  # If True, use more permissive rules (allow basic OHLCV/TA)
    dropped_tracker: Optional[Any] = None  # NEW: Optional DroppedFeaturesTracker for telemetry
) -> List[str]:
    """
    Filter features that would leak information about the target.
    
    Uses both pattern-based filtering (excluded_features.yaml) and
    structural rules (feature_registry.yaml) if enabled.
    
    Args:
        all_columns: List of all column names in the dataset
        target_column: Name of the target column being predicted
        verbose: If True, log excluded features
        use_registry: If True, use FeatureRegistry for structural validation (default: True)
        data_interval_minutes: Data bar interval in minutes (default: 5) for horizon conversion
        for_ranking: If True, use more permissive rules for ranking step:
            - Allows basic OHLCV/TA features (sma_*, rsi_*, volume, etc.) even if in always_exclude
            - Only excludes obvious leaks (y_*, fwd_ret_*, barrier_*, etc.)
            - Registry is advisory (unknown features allowed if they pass pattern filtering)
            - For training: False (stricter rules)
    
    Returns:
        List of safe feature column names
    """
    config = _load_leakage_config()
    
    # CRITICAL: Start with all columns except the target itself
    # The target column remains in the dataset for extraction, but is excluded from features
    # Other target columns (y_*, fwd_ret_*, etc.) will be excluded by pattern matching below
    safe_columns = [c for c in all_columns if c != target_column]
    
    # CRITICAL: Always exclude known metadata columns, even if config fails
    # This is a hardcoded safety net to prevent leakage when config isn't loaded
    known_metadata = ['ts', 'timestamp', 'symbol', 'date', 'time', 'datetime', 'interval', 'source']
    excluded_metadata_hardcoded = [c for c in safe_columns if c in known_metadata]
    safe_columns = [c for c in safe_columns if c not in known_metadata]
    if excluded_metadata_hardcoded and verbose:
        logger.info(f"  Excluded {len(excluded_metadata_hardcoded)} metadata columns (hardcoded safety net): {excluded_metadata_hardcoded[:5]}")
    
    # Exclude metadata columns if configured (additional layer)
    if config.get('config', {}).get('exclude_metadata', True):
        metadata = config.get('metadata_columns', [])
        excluded_metadata = [c for c in safe_columns if c in metadata]
        safe_columns = [c for c in safe_columns if c not in metadata]
        if excluded_metadata and verbose:
            logger.info(f"  Excluded {len(excluded_metadata)} additional metadata columns from config")
    
    # Get target metadata
    target_type = _classify_target_type(target_column, config)
    target_horizon_minutes = _extract_horizon(target_column, config)
    
    # Convert horizon from minutes to bars for registry
    target_horizon_bars = None
    if target_horizon_minutes is not None and data_interval_minutes > 0:
        target_horizon_bars = target_horizon_minutes // data_interval_minutes
        if verbose:
            logger.debug(f"  Target horizon: {target_horizon_minutes}m = {target_horizon_bars} bars (interval={data_interval_minutes}m)")
    
    # Apply feature registry filtering if enabled
    # Registry can filter metadata columns even without horizon (use horizon=1 as default)
    if use_registry:
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
            
            # Use target_horizon_bars if available, otherwise use default (1 bar)
            # This allows registry to filter metadata columns even when horizon extraction fails
            registry_horizon = target_horizon_bars if target_horizon_bars is not None else 1
            
            # For ranking: be more permissive - allow unknown features that pass pattern filtering
            # For training: use registry strictly
            if for_ranking:
                # In ranking mode, registry is advisory - we allow unknown features through
                registry_allowed = registry.get_allowed_features(safe_columns, registry_horizon, verbose=verbose)
                registry_allowed_set = set(registry_allowed)
                if verbose:
                    if target_horizon_minutes is not None and target_horizon_bars is not None:
                        logger.info(f"  Feature registry (ranking mode): {len(registry_allowed)} features explicitly allowed for horizon_minutes={target_horizon_minutes:.1f}m, horizon_bars={target_horizon_bars} bars @ interval={data_interval_minutes:.1f}m (unknown features will be allowed if they pass pattern filtering)")
                    elif target_horizon_bars is not None:
                        logger.info(f"  Feature registry (ranking mode): {len(registry_allowed)} features explicitly allowed for horizon_bars={target_horizon_bars} bars @ interval={data_interval_minutes:.1f}m (unknown features will be allowed if they pass pattern filtering)")
                    else:
                        logger.info(f"  Feature registry (ranking mode): {len(registry_allowed)} features explicitly allowed (unknown features will be allowed if they pass pattern filtering)")
            else:
                # In training mode, use registry strictly
                registry_allowed = registry.get_allowed_features(safe_columns, registry_horizon, verbose=verbose)
                registry_allowed_set = set(registry_allowed)
                if verbose:
                    if target_horizon_minutes is not None and target_horizon_bars is not None:
                        logger.info(f"  Feature registry: {len(registry_allowed)} features explicitly allowed for horizon_minutes={target_horizon_minutes:.1f}m, horizon_bars={target_horizon_bars} bars @ interval={data_interval_minutes:.1f}m")
                    elif target_horizon_bars is not None:
                        logger.info(f"  Feature registry: {len(registry_allowed)} features explicitly allowed for horizon_bars={target_horizon_bars} bars @ interval={data_interval_minutes:.1f}m")
                    else:
                        # CRITICAL: Do NOT silently default to horizon=1
                        # This invalidates safety contracts (purge/embargo, lookback caps)
                        logger.warning(
                            f"  ⚠️  Feature registry: Horizon extraction failed for target '{target_column}'. "
                            f"Using default horizon=1 bar for registry filtering ONLY. "
                            f"This may be incorrect for targets like 'fwd_ret_oc_same_day' or 'fwd_ret_5d'. "
                            f"Target should be marked as 'unresolved_horizon' and quarantined from production."
                        )
                        # Still use horizon=1 for registry filtering (permissive), but log warning
                        logger.info(f"  Feature registry: {len(registry_allowed)} features explicitly allowed (horizon extraction failed, using default horizon=1 for registry ONLY)")
        except Exception as e:
            logger.warning(f"  Feature registry not available: {e}. Using pattern-based filtering only.")
            registry_allowed_set = None
    else:
        registry_allowed_set = None
    
    # Continue with existing pattern-based filtering (as additional safety layer)
    target_horizon = target_horizon_minutes  # Keep original for pattern-based filtering
    
    # CRITICAL: Hardcoded always-exclude patterns as safety net (even if config fails)
    # These are known leaky patterns that should NEVER be used as features
    # IMPORTANT: These exclude TARGET columns (y_*, fwd_ret_*, ret_zscore_*, etc.) from being features,
    # but the target columns themselves remain in the dataset for evaluation
    hardcoded_leaky_patterns = {
        'prefix_patterns': ['p_', 'y_', 'fwd_ret_', 'ret_zscore_', 'tth_', 'mfe_', 'mdd_', 'barrier_', 'next_', 'future_'],
        'exact_patterns': ['ts', 'timestamp', 'symbol', 'date', 'time']
    }
    excluded_hardcoded = _apply_exclusion_patterns(safe_columns, hardcoded_leaky_patterns, "hardcoded-safety-net")
    safe_columns = [c for c in safe_columns if c not in excluded_hardcoded]
    if excluded_hardcoded and verbose:
        # INFO: Count + sample prefixes only (readable)
        sample_prefixes = set()
        for col in excluded_hardcoded[:20]:  # Sample first 20 to identify patterns
            for prefix in ['y_', 'fwd_ret_', 'ret_zscore_', 'barrier_', 'p_', 'tth_', 'mfe_', 'mdd_']:
                if col.startswith(prefix):
                    sample_prefixes.add(prefix)
                    break
        prefix_str = ', '.join(sorted(sample_prefixes)) if sample_prefixes else 'various'
        logger.info(f"  Excluded {len(excluded_hardcoded)} target/label columns (patterns: {prefix_str}, ...)")
        # DEBUG: Full list for detailed analysis
        logger.debug(f"  Full excluded target/label list ({len(excluded_hardcoded)}): {excluded_hardcoded}")
    
    # Apply always-exclude patterns from config (additional layer)
    # For ranking: only exclude obvious leaks, not basic OHLCV/TA features
    always_exclude = config.get('always_exclude', {})
    if for_ranking:
        # For ranking, only apply prefix/keyword patterns (obvious leaks like y_*, fwd_ret_*, barrier_*)
        # Skip exact_patterns which may include basic TA indicators that are safe for ranking
        ranking_exclude = {
            'prefix_patterns': always_exclude.get('prefix_patterns', []),
            'regex_patterns': always_exclude.get('regex_patterns', []),
            'keyword_patterns': always_exclude.get('keyword_patterns', []),
            # Skip exact_patterns - these may include safe TA features
            'exact_patterns': []
        }
        excluded_always = _apply_exclusion_patterns(safe_columns, ranking_exclude, "always-exclude (ranking mode)")
        if verbose:
            logger.info(f"  Ranking mode: Only excluding obvious leaks (y_*, fwd_ret_*, barrier_*, etc.), allowing basic OHLCV/TA features")
    else:
        # For training: apply all exclusion patterns (stricter)
        excluded_always = _apply_exclusion_patterns(safe_columns, always_exclude, "always-exclude")
    
    safe_columns = [c for c in safe_columns if c not in excluded_always]
    if excluded_always and verbose:
        logger.info(f"  Excluded {len(excluded_always)} always-excluded features from config")
    
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
    
    # Apply registry filtering as final step (if enabled)
    # Allow: (1) features explicitly allowed by registry, OR (2) unknown features that passed pattern filtering
    if use_registry and registry_allowed_set is not None:
        # Get metadata for all remaining features to check if they're explicitly rejected
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
        final_safe = []
        for feature in safe_columns:
            # Allow if: (1) explicitly allowed by registry, OR (2) unknown (not in registry, not explicitly rejected)
            if feature in registry_allowed_set:
                final_safe.append(feature)  # Explicitly allowed
            else:
                # Check if feature is explicitly rejected
                metadata = registry.get_feature_metadata(feature)
                if not metadata.get('rejected', False):
                    # Unknown feature that's not explicitly rejected - allow it (passed pattern filtering)
                    final_safe.append(feature)
                # If rejected=True, exclude it
        
        if verbose and len(final_safe) != len(safe_columns):
            excluded_by_registry = len(safe_columns) - len(final_safe)
            logger.info(f"  Registry final filter: {len(final_safe)} features allowed ({excluded_by_registry} explicitly rejected by registry)")
        
        safe_columns = final_safe

    # CRITICAL: For ranking mode, always include minimal safe feature family
    # This ensures ranking has a baseline feature set (OHLCV + basic TA) even if registry/config excludes them
    if for_ranking:
        # Load schema config to get explicit feature families
        schema_config = _load_schema_config()
        mode = 'ranking'
        mode_config = schema_config.get('modes', {}).get(mode, {})
        default_action = mode_config.get('default_action', 'allow')
        
        # Find all available columns (excluding target and metadata)
        all_available = set(all_columns) - {target_column}
        metadata_cols = set(schema_config.get('metadata_columns', []))
        all_available = all_available - metadata_cols
        
        # Get target patterns from schema
        target_patterns = schema_config.get('target_patterns', [])
        
        # Filter out targets
        for pattern in target_patterns:
            pattern_regex = pattern if pattern.startswith('^') else f"^{pattern}"
            try:
                all_available = {f for f in all_available if not re.match(pattern_regex, f, re.IGNORECASE)}
            except re.error:
                # Fallback to simple prefix check
                prefix = pattern.replace('^', '').replace('$', '')
                all_available = {f for f in all_available if not f.startswith(prefix)}
        
        # Find features that match schema families OR use hardcoded patterns as fallback
        schema_safe_features = [f for f in all_available if _is_feature_in_schema_family(f, schema_config, mode)]
        hardcoded_safe_features = [f for f in all_available if _is_ranking_safe_feature(f)]
        
        # Combine: schema-based + hardcoded fallback
        ranking_safe_features = list(set(schema_safe_features) | set(hardcoded_safe_features))
        
        # If default_action is 'allow', also include unknown features that don't match leak patterns
        # These are features that passed earlier filtering (not targets, not metadata, not obvious leaks)
        # but weren't explicitly in schema families - we allow them for ranking
        if default_action == 'allow':
            # Features that are in all_available (passed basic filtering) but not in schema/hardcoded patterns
            # These are "unknown but safe" features - allow them in ranking mode
            unknown_safe = [f for f in all_available 
                          if f not in ranking_safe_features]
            ranking_safe_features.extend(unknown_safe)
        
        # Merge: keep current safe_columns + add any ranking-safe features that were excluded
        safe_set = set(safe_columns)
        added_safe = []
        for feature in ranking_safe_features:
            if feature not in safe_set:
                safe_set.add(feature)
                added_safe.append(feature)
        
        if added_safe and verbose:
            logger.info(f"  Ranking mode: Added {len(added_safe)} safe features from schema (OHLCV/TA families)")
            logger.debug(f"    Added features: {added_safe[:20]}{'...' if len(added_safe) > 20 else ''}")
        
        safe_columns = list(safe_set)
        
        if verbose:
            # Count features by source, accounting for overlap
            schema_only = [f for f in safe_columns if _is_feature_in_schema_family(f, schema_config, mode) and not _is_ranking_safe_feature(f)]
            hardcoded_only = [f for f in safe_columns if _is_ranking_safe_feature(f) and not _is_feature_in_schema_family(f, schema_config, mode)]
            overlap = [f for f in safe_columns if _is_feature_in_schema_family(f, schema_config, mode) and _is_ranking_safe_feature(f)]
            schema_family_hits = len(schema_only) + len(overlap)
            pattern_hits = len(hardcoded_only) + len(overlap)
            union_hits = len(schema_only) + len(hardcoded_only) + len(overlap)
            final_total = len(safe_columns)
            
            # Log with explicit breakdown: union = schema_only + hardcoded_only + overlap
            # final_total may include additional sources (registry-allowed, etc.) beyond these two buckets
            if overlap:
                logger.info(
                    f"  Ranking mode feature composition: "
                    f"schema_family_hits={schema_family_hits}, pattern_hits={pattern_hits}, "
                    f"overlap={len(overlap)}, union_hits={union_hits}, final_total={final_total}"
                )
            else:
                logger.info(
                    f"  Ranking mode feature composition: "
                    f"schema_family_hits={schema_family_hits}, pattern_hits={pattern_hits}, "
                    f"union_hits={union_hits}, final_total={final_total}"
                )
    
    # ACTIVE SANITIZATION: Quarantine features with excessive lookback AFTER all merging is complete
    # This prevents "ghost feature" discrepancies where audit and auto-fix see different lookback values
    # CRITICAL: Must run AFTER ranking mode schema merge (line 833) to catch ghost features that sneak in
    try:
        from TRAINING.ranking.utils.feature_sanitizer import auto_quarantine_long_lookback_features
        
        sanitized_features, quarantined_features, quarantine_report = auto_quarantine_long_lookback_features(
            feature_names=safe_columns,
            interval_minutes=data_interval_minutes,
            max_safe_lookback_minutes=None,  # Loads from config
            enabled=None  # Loads from config
        )
        
        if quarantined_features:
            # Update safe_columns with sanitized features
            safe_columns = sanitized_features
            if verbose:
                logger.info(
                    f"  👻 Active sanitization: Quarantined {len(quarantined_features)} feature(s) "
                    f"with lookback > {quarantine_report.get('max_safe_lookback_minutes', 'unknown')}m"
                )
                logger.debug(f"    Quarantined features: {quarantined_features}")
            
            # NEW: Track sanitizer quarantines for telemetry (if tracker provided)
            if 'dropped_tracker' in locals() and dropped_tracker is not None:
                from TRAINING.ranking.utils.dropped_features_tracker import DropReason
                
                # Create structured reasons from quarantine_report
                structured_reasons = {}
                max_safe = quarantine_report.get('max_safe_lookback_minutes', 0.0)
                reasons_dict = quarantine_report.get('reasons', {})
                
                for feat_name in quarantined_features:
                    reason_info = reasons_dict.get(feat_name, {})
                    measured_value = reason_info.get('lookback_minutes')
                    threshold_value = reason_info.get('max_safe_lookback_minutes', max_safe)
                    human_reason = reason_info.get('reason', f"lookback ({measured_value:.1f}m) exceeds safe threshold ({threshold_value:.1f}m)")
                    
                    config_provenance = f"max_safe_lookback_minutes={threshold_value:.1f}m (from config)"
                    
                    structured_reasons[feat_name] = DropReason(
                        reason_code="QUARANTINED_LOOKBACK",
                        stage="sanitizer",
                        human_reason=human_reason,
                        measured_value=measured_value,
                        threshold_value=threshold_value,
                        config_provenance=config_provenance
                    )
                
                config_provenance_dict = {
                    "max_safe_lookback_minutes": max_safe,
                    "enabled": quarantine_report.get('enabled', True)
                }
                
                dropped_tracker.add_sanitizer_quarantines(
                    quarantined_features,
                    structured_reasons,
                    input_features=safe_columns,  # Before sanitization
                    output_features=sanitized_features,  # After sanitization
                    config_provenance=config_provenance_dict
                )
    except Exception as e:
        # Don't fail if sanitization unavailable - just log and continue
        logger.debug(f"Active sanitization unavailable: {e}")
    
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
    
    Uses SST contract function for consistency.
    
    Examples:
        fwd_ret_60m -> 60
        y_will_peak_15m_0.8 -> 15
        fwd_ret_1d -> 1440 (assuming 1d = 1440 minutes)
        fwd_ret_oc_same_day -> 390 (special case: trading session)
    """
    # Use SST contract function for consistency
    try:
        from TRAINING.common.utils.sst_contract import resolve_target_horizon_minutes
        return resolve_target_horizon_minutes(target_column, config)
    except ImportError:
        # Fallback to original logic if SST contract not available
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
    
    IMPORTANT: This excludes OTHER target columns (fwd_ret_*) from being features,
    but the target_column itself should already be excluded by the caller.
    All target columns remain in the dataset - they're just not used as features.
    """
    excluded = []
    safe = []
    
    target_rules = config.get('target_type_rules', {}).get('forward_return', {})
    horizon_overlap = target_rules.get('horizon_overlap', {})
    
    for col in columns:
        should_exclude = False
        reason = None
        
        # CRITICAL: Exclude ALL forward return columns from features (they're targets, not features)
        # This includes the current target_column (already excluded by caller) and all other fwd_ret_* columns
        # Targets remain in the dataset for extraction, but are never used as features
        if col.startswith('fwd_ret_'):
            should_exclude = True
            reason = "forward return target column (targets are not features - excluded from feature set)"
        # Legacy: Check horizon overlap if enabled (but we already exclude all fwd_ret_* above)
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
    Filter features for barrier targets (peak/valley) using config rules.
    
    IMPORTANT: This excludes OTHER target columns (y_will_peak_*, y_will_valley_*, etc.) from being features,
    but the target_column itself should already be excluded by the caller.
    All target columns remain in the dataset - they're just not used as features.
    """
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
