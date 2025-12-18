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
Configuration Loading for Reproducibility Tracker

Functions for loading configuration values from config files.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def load_thresholds(override: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
    """Load reproducibility thresholds from config."""
    if override:
        return override
    
    try:
        from CONFIG.config_loader import get_safety_config
        safety_cfg = get_safety_config()
        safety_section = safety_cfg.get('safety', {})
        repro_cfg = safety_section.get('reproducibility', {})
        thresholds_cfg = repro_cfg.get('thresholds', {})
        
        # Default thresholds if config missing
        defaults = {
            'roc_auc': {'abs': 0.005, 'rel': 0.02, 'z_score': 1.0},
            'composite': {'abs': 0.02, 'rel': 0.05, 'z_score': 1.5},
            'importance': {'abs': 0.05, 'rel': 0.20, 'z_score': 2.0}
        }
        
        # Merge config with defaults
        thresholds = {}
        for metric in ['roc_auc', 'composite', 'importance']:
            thresholds[metric] = defaults[metric].copy()
            if metric in thresholds_cfg:
                thresholds[metric].update(thresholds_cfg[metric])
        
        return thresholds
    except Exception as e:
        logger.debug(f"Could not load reproducibility thresholds from config: {e}, using defaults")
        # Return defaults
        return {
            'roc_auc': {'abs': 0.005, 'rel': 0.02, 'z_score': 1.0},
            'composite': {'abs': 0.02, 'rel': 0.05, 'z_score': 1.5},
            'importance': {'abs': 0.05, 'rel': 0.20, 'z_score': 2.0}
        }


def load_use_z_score(override: Optional[bool] = None) -> bool:
    """Load use_z_score setting from config."""
    if override is not None:
        return override
    
    try:
        from CONFIG.config_loader import get_safety_config
        safety_cfg = get_safety_config()
        safety_section = safety_cfg.get('safety', {})
        repro_cfg = safety_section.get('reproducibility', {})
        return repro_cfg.get('use_z_score', True)
    except Exception:
        return True  # Default: use z-score


def load_audit_mode() -> str:
    """Load audit mode from config. Defaults to 'off'."""
    try:
        from CONFIG.config_loader import get_safety_config
        safety_cfg = get_safety_config()
        safety_section = safety_cfg.get('safety', {})
        repro_cfg = safety_section.get('reproducibility', {})
        return repro_cfg.get('audit_mode', 'off')
    except Exception:
        return 'off'  # Default: audit mode off


def load_cohort_aware() -> bool:
    """
    Load cohort_aware setting from config.
    
    Defaults to True (cohort-aware mode enabled) for all new installations.
    Set to False in config only if you need legacy flat-file structure.
    """
    try:
        from CONFIG.config_loader import get_safety_config
        safety_cfg = get_safety_config()
        safety_section = safety_cfg.get('safety', {})
        repro_cfg = safety_section.get('reproducibility', {})
        # Default to True (cohort-aware mode) if not specified
        return repro_cfg.get('cohort_aware', True)
    except Exception:
        return True  # Default: cohort-aware mode


def load_n_ratio_threshold() -> float:
    """Load n_ratio_threshold from config. Defaults to 0.5."""
    try:
        from CONFIG.config_loader import get_safety_config
        safety_cfg = get_safety_config()
        safety_section = safety_cfg.get('safety', {})
        repro_cfg = safety_section.get('reproducibility', {})
        return float(repro_cfg.get('n_ratio_threshold', 0.5))
    except Exception:
        return 0.5  # Default: 0.5


def load_cohort_config_keys() -> List[str]:
    """Load cohort_config_keys from config. Defaults to standard keys."""
    try:
        from CONFIG.config_loader import get_safety_config
        safety_cfg = get_safety_config()
        safety_section = safety_cfg.get('safety', {})
        repro_cfg = safety_section.get('reproducibility', {})
        keys = repro_cfg.get('cohort_config_keys', [
            'N_effective_cs',
            'n_symbols',
            'date_range',
            'cs_config'
        ])
        return keys if isinstance(keys, list) else [
            'N_effective_cs',
            'n_symbols',
            'date_range',
            'cs_config'
        ]
    except Exception:
        return [
            'N_effective_cs',
            'n_symbols',
            'date_range',
            'cs_config'
        ]

