"""
Look-Ahead Bias Fix Configuration Utility

Provides centralized access to look-ahead bias fix configuration flags.
All fixes are disabled by default to maintain backward compatibility.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def get_lookahead_bias_fix_config() -> Dict[str, Any]:
    """
    Load look-ahead bias fix configuration flags from safety_config.
    
    Returns:
        Dictionary with fix configuration:
        - exclude_current_bar: bool - Exclude current bar from rolling windows
        - normalize_inside_cv: bool - Normalize inside CV loops
        - verify_pct_change: bool - Verify pct_change excludes current bar
        - migration_mode: str - Migration mode (off/test/warn/enforce)
    
    Defaults (if config not available):
        All fixes disabled (False/off) to maintain backward compatibility.
    """
    try:
        from CONFIG.config_loader import get_cfg
        fix_cfg = get_cfg(
            "safety.leakage_detection.lookahead_bias_fixes",
            default={},
            config_name="safety_config"
        )
        
        return {
            'exclude_current_bar': fix_cfg.get('exclude_current_bar_from_rolling', False),
            'normalize_inside_cv': fix_cfg.get('normalize_inside_cv', False),
            'verify_pct_change': fix_cfg.get('verify_pct_change_shift', False),
            'migration_mode': fix_cfg.get('migration_mode', 'off')
        }
    except Exception as e:
        # Default: all fixes disabled (maintains current behavior)
        logger.debug(f"Failed to load lookahead_bias_fixes config: {e}, using defaults (all disabled)")
        return {
            'exclude_current_bar': False,
            'normalize_inside_cv': False,
            'verify_pct_change': False,
            'migration_mode': 'off'
        }


def is_fix_enabled(fix_name: str) -> bool:
    """
    Check if a specific look-ahead bias fix is enabled.
    
    Args:
        fix_name: Name of the fix ('exclude_current_bar', 'normalize_inside_cv', 'verify_pct_change')
    
    Returns:
        True if fix is enabled, False otherwise
    """
    config = get_lookahead_bias_fix_config()
    
    fix_map = {
        'exclude_current_bar': 'exclude_current_bar',
        'normalize_inside_cv': 'normalize_inside_cv',
        'verify_pct_change': 'verify_pct_change'
    }
    
    if fix_name not in fix_map:
        logger.warning(f"Unknown fix name: {fix_name}, returning False")
        return False
    
    return config.get(fix_map[fix_name], False)


def get_migration_mode() -> str:
    """
    Get the current migration mode.
    
    Returns:
        Migration mode: 'off', 'test', 'warn', or 'enforce'
    """
    config = get_lookahead_bias_fix_config()
    return config.get('migration_mode', 'off')


def should_log_differences() -> bool:
    """
    Check if we should log differences between old and new behavior.
    
    Returns:
        True if migration_mode is 'test', 'warn', or 'enforce'
    """
    mode = get_migration_mode()
    return mode in ('test', 'warn', 'enforce')


def should_warn_on_discrepancies() -> bool:
    """
    Check if we should warn on discrepancies.
    
    Returns:
        True if migration_mode is 'warn' or 'enforce'
    """
    mode = get_migration_mode()
    return mode in ('warn', 'enforce')


def should_fail_on_discrepancies() -> bool:
    """
    Check if we should fail on discrepancies.
    
    Returns:
        True if migration_mode is 'enforce'
    """
    mode = get_migration_mode()
    return mode == 'enforce'
