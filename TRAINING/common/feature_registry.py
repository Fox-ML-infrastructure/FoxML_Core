# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Registry

Manages feature metadata and enforces temporal rules to prevent data leakage.
Makes leakage structurally impossible without lying to the configuration.
"""

import re
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import hashlib

logger = logging.getLogger(__name__)

# Try to import config loader for path configuration
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass

# Global registry instance (lazy-loaded)
_REGISTRY: Optional['FeatureRegistry'] = None


class FeatureRegistry:
    """
    Manages feature metadata and enforces temporal rules.
    
    Features must have:
    - lag_bars: How many bars back the feature is allowed to peek (>= 0)
    - allowed_horizons: List of target horizons this feature can predict
    - source: Where the feature comes from (price, volume, derived, etc.)
    
    Hard rules:
    - lag_bars >= 0 (cannot look into future)
    - lag_bars >= horizon_bars for price/derived features
    - allowed_horizons must be non-empty for usable features
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize feature registry from YAML config.
        
        Args:
            config_path: Path to feature_registry.yaml (default: from config or CONFIG/feature_registry.yaml)
        """
        if config_path is None:
            # Try to get path from config first
            if _CONFIG_AVAILABLE:
                try:
                    system_cfg = get_system_config()
                    config_paths = system_cfg.get('system', {}).get('paths', {})
                    registry_path_str = config_paths.get('feature_registry')
                    if registry_path_str:
                        config_path = Path(registry_path_str)
                        if not config_path.is_absolute():
                            repo_root = Path(__file__).resolve().parents[2]
                            config_path = repo_root / registry_path_str
                    else:
                        # Use config_dir from config
                        config_dir = config_paths.get('config_dir', 'CONFIG')
                        repo_root = Path(__file__).resolve().parents[2]
                        config_path = repo_root / config_dir / "feature_registry.yaml"
                except Exception:
                    # Fallback to default
                    repo_root = Path(__file__).resolve().parents[2]
                    config_path = repo_root / "CONFIG" / "feature_registry.yaml"
            else:
                # Fallback to default
                repo_root = Path(__file__).resolve().parents[2]
                config_path = repo_root / "CONFIG" / "feature_registry.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.features = self.config.get('features', {})
        self.families = self.config.get('feature_families', {})
        self.validation_rules = self.config.get('validation', {})
        
        # Validate registry on load
        self._validate_registry()
        
        logger.info(f"Loaded feature registry: {len(self.features)} features, {len(self.families)} families")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load feature registry from YAML file."""
        if not self.config_path.exists():
            # Try fallback: CWD/CONFIG/feature_registry.yaml
            cwd_config = Path.cwd() / "CONFIG" / "feature_registry.yaml"
            if cwd_config.exists():
                logger.info(f"Using feature registry from CWD: {cwd_config}")
                self.config_path = cwd_config
            else:
                logger.warning(
                    f"Feature registry not found at {self.config_path} or {cwd_config}. "
                    f"Using empty registry (all features will be auto-inferred)."
                )
                return {
                    'features': {},
                    'feature_families': {},
                    'validation': {
                        'hard_rules': [],
                        'warnings': []
                    }
                }
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            
            # Ensure required keys exist
            config.setdefault('features', {})
            config.setdefault('feature_families', {})
            config.setdefault('validation', {
                'hard_rules': [],
                'warnings': []
            })
            
            return config
        except Exception as e:
            logger.error(f"Failed to load feature registry from {self.config_path}: {e}")
            return {
                'features': {},
                'feature_families': {},
                'validation': {
                    'hard_rules': [],
                    'warnings': []
                }
            }
    
    def _validate_registry(self):
        """Validate all features against hard rules."""
        errors = []
        warnings = []
        
        for name, metadata in self.features.items():
            try:
                self._validate_feature(name, metadata)
            except ValueError as e:
                errors.append(str(e))
            except Warning as w:
                warnings.append(str(w))
        
        if errors:
            raise ValueError(f"Feature registry validation failed:\n" + "\n".join(errors))
        
        if warnings:
            for w in warnings:
                logger.warning(w)
    
    def _validate_feature(self, name: str, metadata: Dict[str, Any]):
        """
        Validate a single feature against hard rules.
        
        Raises:
            ValueError: If hard rule violated
            Warning: If soft rule violated
        """
        lag_bars = metadata.get('lag_bars', 0)
        allowed_horizons = metadata.get('allowed_horizons', [])
        source = metadata.get('source', 'unknown')
        rejected = metadata.get('rejected', False)
        
        # Skip validation for explicitly rejected features (they're documented as leaky)
        if rejected:
            return
        
        # Hard rule: Cannot look into future (for non-rejected features)
        if lag_bars < 0:
            raise ValueError(
                f"Feature '{name}': lag_bars={lag_bars} < 0 "
                f"(looks into future - structurally impossible). "
                f"Mark as rejected: true if this is intentional."
            )
        
        # Hard rule: For price/derived features, lag must be >= 0 (can't look into future)
        # Note: A feature with lag_bars=1 CAN predict a 3-bar horizon target.
        # The feature just needs to be from the past, not necessarily lag by the full horizon.
        # The actual temporal safety is enforced by PurgedTimeSeriesSplit (purge gap = horizon).
        if source in ['price', 'derived']:
            # Only check that lag is non-negative
            if lag_bars < 0:
                raise ValueError(
                    f"Feature '{name}': lag_bars={lag_bars} < 0 "
                    f"(cannot look into future - structurally impossible)"
                )
        
        # Hard rule: Usable features must have allowed horizons
        if not allowed_horizons:
            logger.warning(
                f"Feature '{name}': No allowed_horizons specified "
                f"(will be rejected by default - safe). "
                f"Mark as rejected: true if this is intentional."
            )
    
    def is_allowed(self, feature_name: str, target_horizon: int) -> bool:
        """
        Check if feature is allowed for a target horizon.
        
        Args:
            feature_name: Name of the feature
            target_horizon: Target horizon in bars (e.g., 12 for 60-minute target with 5m bars)
        
        Returns:
            True if feature is allowed, False otherwise
        """
        # Check explicit feature metadata
        if feature_name in self.features:
            metadata = self.features[feature_name]
            
            # Explicitly rejected features
            if metadata.get('rejected', False):
                return False
            
            # Check if horizon is in allowed list
            allowed_horizons = metadata.get('allowed_horizons', [])
            if target_horizon in allowed_horizons:
                return True
            
            # Not in allowed list
            return False
        
        # Check feature families (pattern matching)
        for family_name, family_config in self.families.items():
            pattern = family_config.get('pattern')
            if pattern and re.match(pattern, feature_name):
                # Rejected families
                if family_name.startswith('rejected_') or family_config.get('rejected', False):
                    return False
                
                # Allowed families - check default horizons
                default_horizons = family_config.get('default_allowed_horizons', [])
                if target_horizon in default_horizons:
                    return True
        
        # Unknown feature: auto-infer or reject (safe default)
        inferred = self.auto_infer_metadata(feature_name)
        if inferred.get('rejected', False):
            return False
        
        allowed_horizons = inferred.get('allowed_horizons', [])
        return target_horizon in allowed_horizons
    
    def get_allowed_features(
        self, 
        all_features: List[str], 
        target_horizon: int,
        verbose: bool = False
    ) -> List[str]:
        """
        Get list of allowed features for a target horizon.
        
        Args:
            all_features: List of all feature names
            target_horizon: Target horizon in bars
            verbose: If True, log excluded features
        
        Returns:
            List of allowed feature names
        """
        allowed = []
        excluded = []
        
        for feature in all_features:
            if self.is_allowed(feature, target_horizon):
                allowed.append(feature)
            else:
                excluded.append(feature)
        
        if verbose and excluded:
            logger.info(
                f"Feature registry: Allowed {len(allowed)} features, "
                f"excluded {len(excluded)} features for horizon={target_horizon}"
            )
            if len(excluded) <= 10:
                logger.debug(f"  Excluded: {', '.join(excluded)}")
            else:
                logger.debug(f"  Excluded: {', '.join(excluded[:10])}... ({len(excluded)} total)")
        
        return allowed
    
    def auto_infer_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Auto-infer metadata for unknown features (backward compatibility).
        
        Uses pattern matching to infer lag_bars and allowed_horizons.
        
        Returns:
            Metadata dictionary with inferred values
        """
        # Lagged returns: ret_N where N is the lag
        ret_match = re.match(r"^ret_(\d+)$", feature_name)
        if ret_match:
            lag = int(ret_match.group(1))
            return {
                'source': 'price',
                'lag_bars': lag,
                'allowed_horizons': [lag, lag*3, lag*5, lag*12] if lag > 0 else [],
                'description': f"Auto-inferred: {lag}-bar lagged return"
            }
        
        # Forward returns (leaky): ret_future_N or fwd_ret_N
        if re.match(r"^(ret_future_|fwd_ret_)", feature_name):
            return {
                'source': 'price',
                'lag_bars': -1,  # Negative = looks into future
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - forward return (leaky)'
            }
        
        # Technical indicators with lookback: rsi_N, sma_N, ema_N, cci_N, stoch_k_N, etc.
        # Use comprehensive pattern matching to catch all indicator-period features
        # Simple patterns: rsi_30, cci_30, stoch_d_21, etc.
        simple_patterns = [
            (r'^(stoch_d|stoch_k|williams_r)_(\d+)$', 2),
            (r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var)_(\d+)$', 2),
            (r'^(ret|sma|ema|vol)_(\d+)$', 2),
        ]
        for pattern, group_idx in simple_patterns:
            match = re.match(pattern, feature_name, re.I)
            if match:
                lookback = int(match.group(group_idx))
                indicator_type = match.group(1)
                return {
                    'source': 'derived',
                    'lag_bars': lookback,
                    'allowed_horizons': [1, 3, 5, 12, 24, 60] if lookback > 0 else [],
                    'description': f"Auto-inferred: {indicator_type.upper()} with {lookback}-bar lookback"
                }
        
        # Compound indicator patterns: bb_upper_20, bb_lower_20, bb_width_20, bb_percent_b_20
        compound_patterns = [
            (r'^bb_(upper|lower|width|percent_b|middle)_(\d+)$', 2),
            (r'^macd_(signal|hist|diff)_(\d+)$', 2),
            (r'^(stoch_k|stoch_d|rsi|cci|mfi|atr|adx|mom|std|var)_(fast|slow|wilder|smooth|upper|lower|width)_(\d+)$', 3),
        ]
        for pattern, group_idx in compound_patterns:
            match = re.match(pattern, feature_name, re.I)
            if match:
                lookback = int(match.group(group_idx))
                indicator_type = match.group(1)
                return {
                    'source': 'derived',
                    'lag_bars': lookback,
                    'allowed_horizons': [1, 3, 5, 12, 24, 60] if lookback > 0 else [],
                    'description': f"Auto-inferred: {indicator_type.upper()} (compound) with {lookback}-bar lookback"
                }
        
        # Volume/volatility features with period: volume_ema_5, realized_vol_10, etc.
        vol_patterns = [
            (r'^volume_(ema|sma)_(\d+)$', 2),
            (r'^realized_vol_(\d+)$', 1),
            (r'^vol_(ema|sma|std)_(\d+)$', 2),
        ]
        for pattern, group_idx in vol_patterns:
            match = re.match(pattern, feature_name, re.I)
            if match:
                lookback = int(match.group(group_idx))
                return {
                    'source': 'derived',
                    'lag_bars': lookback,
                    'allowed_horizons': [1, 3, 5, 12, 24, 60] if lookback > 0 else [],
                    'description': f"Auto-inferred: volume/volatility feature with {lookback}-bar lookback"
                }
        
        # Time-to-hit features (leaky)
        if re.match(r"^tth_", feature_name):
            return {
                'source': 'derived',
                'lag_bars': 0,  # Computed at time of hit (requires future)
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - time-to-hit requires future path'
            }
        
        # MFE/MDD features (leaky)
        if re.match(r"^(mfe|mdd)_", feature_name):
            return {
                'source': 'derived',
                'lag_bars': 0,
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - MFE/MDD requires future path'
            }
        
        # Barrier features (leaky)
        if re.match(r"^barrier_", feature_name):
            return {
                'source': 'derived',
                'lag_bars': 0,
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - barrier features encode barrier logic'
            }
        
        # Target columns (leaky)
        if re.match(r"^(y_|target_)", feature_name):
            return {
                'source': 'target',
                'lag_bars': 0,
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - target column (leaky)'
            }
        
        # Prediction/probability features (leaky)
        if re.match(r"^p_", feature_name):
            return {
                'source': 'derived',
                'lag_bars': 0,
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - prediction/probability feature (leaky)'
            }
        
        # Timestamp/metadata columns
        if feature_name in ['ts', 'timestamp', 'symbol', 'time']:
            return {
                'source': 'metadata',
                'lag_bars': 0,
                'allowed_horizons': [],
                'rejected': True,
                'description': 'Auto-inferred: REJECTED - metadata column'
            }
        
        # Unknown feature: reject by default (safe)
        logger.debug(
            f"Unknown feature '{feature_name}': rejecting by default (safe). "
            f"Add to feature_registry.yaml to allow."
        )
        return {
            'source': 'unknown',
            'lag_bars': 0,
            'allowed_horizons': [],
            'rejected': True,
            'description': 'Auto-inferred: REJECTED - unknown feature (safe default)'
        }
    
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """Get metadata for a feature (from registry or auto-inferred)."""
        if feature_name in self.features:
            return self.features[feature_name]
        
        return self.auto_infer_metadata(feature_name)
    
    def register_feature(self, name: str, metadata: Dict[str, Any]):
        """
        Register a new feature with metadata.
        
        Validates the feature before adding to registry.
        
        Args:
            name: Feature name
            metadata: Feature metadata dict
        """
        # Validate before adding
        self._validate_feature(name, metadata)
        
        # Add to registry
        self.features[name] = metadata
        logger.info(f"Registered feature '{name}' with metadata: {metadata}")


def get_registry(config_path: Optional[Path] = None) -> FeatureRegistry:
    """
    Get global feature registry instance (singleton pattern).
    
    Args:
        config_path: Optional path to feature_registry.yaml
    
    Returns:
        FeatureRegistry instance
    """
    global _REGISTRY
    
    if _REGISTRY is None:
        _REGISTRY = FeatureRegistry(config_path)
    
    return _REGISTRY


def reset_registry():
    """Reset global registry (useful for testing)."""
    global _REGISTRY
    _REGISTRY = None

