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

Target-Conditional Feature Exclusion

Generates per-target exclusion lists based on:
- Target horizon (short-term vs long-term safety rules)
- Target semantics (peak/valley, volatility, direction, etc.)
- Feature lookback limits relative to target horizon

This implements "Target-Conditional Feature Selection" - a preprocessing step
that tailors the feature set to the specific physics of each target before ranking.
"""

import logging
import yaml
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    import sys
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "CONFIG"
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using defaults")


def extract_target_horizon_minutes(target_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """
    Extract target horizon in minutes from target name.
    
    Args:
        target_name: Target column name (e.g., 'y_will_peak_60m_0.8', 'fwd_ret_15m')
        config: Optional config dict with horizon_extraction patterns
    
    Returns:
        Horizon in minutes, or None if cannot be determined
    """
    if config is None and _CONFIG_AVAILABLE:
        try:
            from TRAINING.utils.leakage_filtering import _load_leakage_config
            config = _load_leakage_config()
        except Exception:
            pass
    
    # Default patterns (from excluded_features.yaml structure)
    patterns = [
        {'regex': r'(\d+)m', 'multiplier': 1},      # 60m -> 60
        {'regex': r'(\d+)h', 'multiplier': 60},     # 2h -> 120
        {'regex': r'(\d+)d', 'multiplier': 1440},  # 1d -> 1440
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
    
    return None


def classify_target_semantics(target_name: str) -> Dict[str, bool]:
    """
    Classify target semantics to determine exclusion rules.
    
    Returns:
        Dict with semantic flags: peak, valley, volatility, direction, etc.
    """
    target_lower = target_name.lower()
    
    semantics = {
        'peak': bool(re.search(r'peak|high|swing.*high|zigzag.*high', target_lower)),
        'valley': bool(re.search(r'valley|low|swing.*low|zigzag.*low', target_lower)),
        'volatility': bool(re.search(r'vol|volatility|variance|std', target_lower)),
        'direction': bool(re.search(r'direction|trend|momentum|ret', target_lower)),
        'barrier': bool(re.search(r'barrier|touch|first_touch', target_lower)),
        'forward_return': bool(re.search(r'fwd_ret|forward.*ret', target_lower)),
    }
    
    return semantics


def compute_feature_lookback_minutes(
    feature_name: str,
    interval_minutes: float = 5.0,
    registry: Optional[Any] = None
) -> Optional[float]:
    """
    Compute feature lookback in minutes.
    
    Uses same logic as resolved_config.compute_feature_lookback_max().
    """
    # Try registry first
    if registry:
        try:
            metadata = registry.get_feature_metadata(feature_name)
            lag_bars = metadata.get('lag_bars')
            if lag_bars is not None:
                return lag_bars * interval_minutes
        except Exception:
            pass
    
    # Fallback: pattern matching
    # PRECEDENCE ORDER (same as compute_feature_lookback_max):
    # 1. Explicit time suffixes (most reliable) - check FIRST
    # 2. Keyword heuristics (less reliable) - only as fallback
    
    # PRIORITY 1: Explicit time-based suffixes (most reliable)
    # Minute-based patterns (e.g., _15m, _30m, _1440m) - CHECK FIRST
    minutes_match = re.search(r'_(\d+)m$', feature_name, re.I)
    if minutes_match:
        minutes = int(minutes_match.group(1))
        return float(minutes)
    
    # Hour-based patterns (e.g., _12h, _24h) - CHECK SECOND
    hours_match = re.search(r'_(\d+)h', feature_name, re.I)
    if hours_match:
        hours = int(hours_match.group(1))
        return hours * 60.0
    
    # Day-based patterns (e.g., _1d, _3d) - CHECK THIRD
    days_match = re.search(r'_(\d+)d', feature_name, re.I)
    if days_match:
        days = int(days_match.group(1))
        return days * 1440.0
    
    # PRIORITY 2: Keyword heuristics (fallback only if no explicit suffix)
    # Explicit daily patterns (ends with _1d, _24h, starts with daily_, etc.)
    if (re.search(r'_1d$|_1D$|_24h$|_24H$|^daily_|_daily$|_1440m|1440(?!\d)', feature_name, re.I) or
        re.search(r'rolling.*daily|daily.*high|daily.*low', feature_name, re.I) or
        re.search(r'volatility.*day|vol.*day|volume.*day', feature_name, re.I)):
        return 1440.0
    
    # Last resort: very aggressive "day" keyword (only if no explicit suffix)
    if re.search(r'.*day.*', feature_name, re.I):
        return 1440.0
    
    # Bar-based patterns (sma_200, rsi_14, etc.)
    bar_match = re.match(r'^(ret|sma|ema|rsi|macd|bb|atr|adx|mom|vol|std|var)_(\d+)', feature_name)
    if bar_match:
        bars = int(bar_match.group(2))
        return bars * interval_minutes
    
    return None


def generate_target_exclusion_list(
    target_name: str,
    all_features: List[str],
    interval_minutes: float = 5.0,
    output_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
    registry: Optional[Any] = None
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Generate per-target exclusion list based on target characteristics.
    
    This implements "Target-Conditional Feature Selection" - tailoring the
    feature set to the specific physics of each target.
    
    Args:
        target_name: Target column name
        all_features: List of all available feature names
        interval_minutes: Data interval in minutes
        output_dir: Directory to save exclusion list (optional)
        config: Optional config dict
        registry: Optional feature registry
    
    Returns:
        (excluded_features, exclusion_metadata) tuple
    """
    exclusions = []
    exclusion_reasons = {}
    metadata = {
        'target_name': target_name,
        'target_horizon_minutes': None,
        'target_semantics': {},
        'exclusion_rules_applied': [],
        'total_features': len(all_features),
        'excluded_count': 0,
    }
    
    # Extract target characteristics
    target_horizon = extract_target_horizon_minutes(target_name, config)
    target_semantics = classify_target_semantics(target_name)
    
    metadata['target_horizon_minutes'] = target_horizon
    metadata['target_semantics'] = target_semantics
    
    # Load config for exclusion rules
    if config is None and _CONFIG_AVAILABLE:
        try:
            # Try to load target-conditional exclusion config
            try:
                config = {}
                config['horizon_safety_multiplier'] = get_cfg(
                    "target_conditional_exclusions.horizon_safety_multiplier",
                    default=4.0,
                    config_name="training_config"
                )
                config['enable_semantic_rules'] = get_cfg(
                    "target_conditional_exclusions.enable_semantic_rules",
                    default=True,
                    config_name="training_config"
                )
            except Exception:
                pass
        except Exception:
            pass
    
    if config is None:
        config = {
            'horizon_safety_multiplier': 4.0,  # Default: 4x horizon
            'enable_semantic_rules': True,
        }
    
    # Rule 1: Horizon Safety (The "Ghost Buster" logic)
    # If target is 60m, don't let the model see data from too far back
    if target_horizon is not None:
        safety_multiplier = config.get('horizon_safety_multiplier', 4.0)
        safe_lookback_limit = target_horizon * safety_multiplier
        
        metadata['exclusion_rules_applied'].append({
            'rule': 'horizon_safety',
            'safe_lookback_limit_minutes': safe_lookback_limit,
            'multiplier': safety_multiplier,
        })
        
        for feature_name in all_features:
            feature_lookback = compute_feature_lookback_minutes(
                feature_name, interval_minutes, registry
            )
            
            if feature_lookback is not None and feature_lookback > safe_lookback_limit:
                if feature_name not in exclusions:
                    exclusions.append(feature_name)
                    exclusion_reasons[feature_name] = (
                        f"lookback ({feature_lookback:.1f}m) > safe_limit "
                        f"({safe_lookback_limit:.1f}m = {target_horizon}m * {safety_multiplier})"
                    )
    
    # Rule 2: Semantic Safety (Target-Specific Leaks)
    if config.get('enable_semantic_rules', True):
        # Peak/Valley targets: exclude repainting indicators
        if target_semantics.get('peak') or target_semantics.get('valley'):
            repainting_patterns = [
                r'zigzag',
                r'pivot.*high',
                r'pivot.*low',
                r'future.*high',
                r'future.*low',
                r'swing.*high',
                r'swing.*low',
            ]
            
            metadata['exclusion_rules_applied'].append({
                'rule': 'semantic_peak_valley',
                'patterns': repainting_patterns,
            })
            
            for feature_name in all_features:
                for pattern in repainting_patterns:
                    if re.search(pattern, feature_name, re.IGNORECASE):
                        if feature_name not in exclusions:
                            exclusions.append(feature_name)
                            exclusion_reasons[feature_name] = f"repainting indicator (peak/valley target)"
                        break
        
        # Volatility targets: exclude directional indicators (optional)
        if target_semantics.get('volatility'):
            # This is optional - volatility targets might benefit from directional features
            # Uncomment if you want to exclude directional indicators for volatility targets
            # directional_patterns = [r'rsi', r'macd', r'adx', r'momentum']
            pass
    
    metadata['excluded_count'] = len(exclusions)
    metadata['exclusion_reasons'] = exclusion_reasons
    
    # Save exclusion list to file
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanitize target name for filename
        safe_target_name = re.sub(r'[^\w\-_]', '_', target_name)
        exclusion_file = output_dir / f"{safe_target_name}_exclusions.yaml"
        
        exclusion_data = {
            'target_name': target_name,
            'target_horizon_minutes': target_horizon,
            'target_semantics': target_semantics,
            'excluded_features': exclusions,
            'exclusion_reasons': exclusion_reasons,
            'metadata': metadata,
        }
        
        with open(exclusion_file, 'w') as f:
            yaml.dump(exclusion_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(
            f"📋 Generated target-conditional exclusions for {target_name}: "
            f"{len(exclusions)} features excluded (saved to {exclusion_file})"
        )
    
    return exclusions, metadata


def load_target_exclusion_list(
    target_name: str,
    exclusion_dir: Path
) -> Optional[List[str]]:
    """
    Load previously generated exclusion list for a target.
    
    This function checks for existing exclusion lists in the RESULTS directory structure:
    RESULTS/{cohort}/{run}/feature_exclusions/{target}_exclusions.yaml
    
    If found, returns the excluded features list. This allows reusing exclusion lists
    across runs for the same target, improving consistency and performance.
    
    Args:
        target_name: Target column name
        exclusion_dir: Directory containing exclusion lists (typically RESULTS/{cohort}/{run}/feature_exclusions/)
    
    Returns:
        List of excluded feature names, or None if not found
    """
    exclusion_dir = Path(exclusion_dir)
    safe_target_name = re.sub(r'[^\w\-_]', '_', target_name)
    exclusion_file = exclusion_dir / f"{safe_target_name}_exclusions.yaml"
    
    if not exclusion_file.exists():
        logger.debug(f"Exclusion list not found at {exclusion_file} - will generate new one")
        return None
    
    try:
        with open(exclusion_file, 'r') as f:
            data = yaml.safe_load(f)
            excluded_features = data.get('excluded_features', [])
            if excluded_features:
                logger.debug(
                    f"✅ Loaded exclusion list from {exclusion_file}: "
                    f"{len(excluded_features)} features excluded"
                )
            return excluded_features
    except Exception as e:
        logger.warning(f"Failed to load exclusion list from {exclusion_file}: {e}")
        return None
