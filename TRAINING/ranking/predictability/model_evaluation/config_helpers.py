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

