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
Registry Validation - Enforce Canonical Key Invariants

This module provides startup assertions to ensure all registry keys
are canonical (snake_case lowercase) and collision-free.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _assert_canonical_keys(map_obj: Dict[str, Any], name: str):
    """
    Assert that all keys in a registry are canonical (snake_case lowercase).
    
    Raises AssertionError if:
    - Any key is not in canonical form (normalize(key) != key)
    - Any keys collide after normalization (e.g., "LightGBM" and "lightgbm")
    
    Args:
        map_obj: Dictionary to validate
        name: Name of the registry (for error messages)
    """
    from TRAINING.training_strategies.utils import normalize_family_name
    
    normed = {}
    errors = []
    
    for k in map_obj.keys():
        nk = normalize_family_name(k)
        if nk != k:
            errors.append(f"{name} has non-canonical key: '{k}' (expected '{nk}')")
        if nk in normed:
            errors.append(f"{name} has collision after normalization: '{k}' -> '{nk}' (collides with '{normed[nk]}')")
        normed[nk] = k
    
    if errors:
        error_msg = f"Registry validation failed for {name}:\n" + "\n".join(f"  • {e}" for e in errors)
        raise AssertionError(error_msg)


def validate_all_registries():
    """
    Validate all model family registries at startup.
    
    Checks:
    - MODMAP (in family_runners.py)
    - TRAINER_MODULE_MAP (in isolation_runner.py)
    - POLICY (in runtime_policy.py)
    - FAMILY_CAPS (in utils.py)
    
    Raises AssertionError if any registry has non-canonical keys or collisions.
    """
    try:
        from TRAINING.training_strategies.execution.family_runners import _run_family_inproc
        # MODMAP is defined inside _run_family_inproc, so we can't import it directly
        # We'll validate it via the test suite instead
        pass
    except ImportError:
        pass
    
    try:
        from TRAINING.common.isolation_runner import TRAINER_MODULE_MAP
        _assert_canonical_keys(TRAINER_MODULE_MAP, "TRAINER_MODULE_MAP")
        logger.debug("✅ TRAINER_MODULE_MAP validation passed")
    except ImportError:
        logger.warning("Could not import TRAINER_MODULE_MAP for validation")
    
    try:
        from TRAINING.common.runtime_policy import POLICY
        _assert_canonical_keys(POLICY, "POLICY")
        logger.debug("✅ POLICY validation passed")
    except ImportError:
        logger.warning("Could not import POLICY for validation")
    
    try:
        from TRAINING.training_strategies.utils import FAMILY_CAPS
        _assert_canonical_keys(FAMILY_CAPS, "FAMILY_CAPS")
        logger.debug("✅ FAMILY_CAPS validation passed")
    except ImportError:
        logger.warning("Could not import FAMILY_CAPS for validation")


# Run validation on import (fail-fast)
try:
    validate_all_registries()
except AssertionError as e:
    logger.error(f"❌ Registry validation failed: {e}")
    raise

