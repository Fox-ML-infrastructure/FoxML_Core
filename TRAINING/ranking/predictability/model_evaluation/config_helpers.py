# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Configuration Helper Functions

Helper functions for loading and managing configuration.
"""

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass


def get_importance_top_fraction() -> float:
    """Get the top fraction for importance analysis from config."""
    if _CONFIG_AVAILABLE:
        try:
            # Load from feature_selection/multi_model.yaml
            fraction = float(get_cfg("aggregation.importance_top_fraction", default=0.10, config_name="multi_model"))
            return fraction
        except Exception:
            return 0.10  # FALLBACK_DEFAULT_OK
    return 0.10  # FALLBACK_DEFAULT_OK

