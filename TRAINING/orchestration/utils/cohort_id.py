# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Cohort ID Generation

Single Source of Truth for cohort ID computation.
Unifies duplicate logic from ReproducibilityTracker and training_strategies.
"""

import json
import hashlib
import logging
from typing import Dict, Any, Union
import pandas as pd

# SST: Import View enum for consistent view handling
from TRAINING.orchestration.utils.scope_resolution import View

# SST: Import universe_sig extractor for consistent access
try:
    from TRAINING.orchestration.utils.reproducibility.utils import extract_universe_sig
    _EXTRACT_UNIVERSE_SIG_AVAILABLE = True
except ImportError:
    _EXTRACT_UNIVERSE_SIG_AVAILABLE = False
    extract_universe_sig = None

logger = logging.getLogger(__name__)


def compute_cohort_id(
    cohort_metadata: Dict[str, Any],
    view: Union[str, View]
) -> str:
    """
    Single source of truth for cohort ID generation.
    
    Format: {mode_prefix}_{date_range}_{universe}_{config}_{version}_{hash}
    Example: cs_2025Q3_ef91e9db233a_min_cs3_max2000_v1_abc12345
    
    Args:
        cohort_metadata: Cohort metadata dict from extract_cohort_metadata()
        view: View enum or string (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
    
    Returns:
        Cohort ID string with prefix matching view
    
    Raises:
        ValueError: If view is invalid
    
    Examples:
        >>> cohort_metadata = {
        ...     'date_range': {'start_ts': '2025-01-01', 'end_ts': '2025-03-31'},
        ...     'cs_config': {'universe_sig': 'abc123', 'min_cs': 3, 'max_cs_samples': 2000},
        ...     'n_effective_cs': 100, 'n_symbols': 10
        ... }
        >>> compute_cohort_id(cohort_metadata, View.CROSS_SECTIONAL)
        'cs_2025Q1_abc123_min_cs3_max2000_v1_...'
    """
    # Normalize view to enum, then map to mode prefix
    view_enum = View.from_string(view) if isinstance(view, str) else view
    if view_enum == View.CROSS_SECTIONAL:
        mode_prefix = "cs"
    elif view_enum == View.SYMBOL_SPECIFIC:
        mode_prefix = "sy"
    else:
        raise ValueError(f"Invalid view: {view}. Must be View.CROSS_SECTIONAL or View.SYMBOL_SPECIFIC")
    
    # Extract date range
    date_range = cohort_metadata.get('date_range', {})
    date_start = date_range.get('start_ts', '')
    date_end = date_range.get('end_ts', '')
    
    # Convert to quarter format if possible
    date_str = ""
    if date_start:
        try:
            dt = pd.Timestamp(date_start)
            date_str = f"{dt.year}Q{(dt.month-1)//3 + 1}"
        except Exception as e:
            logger.debug(f"Failed to parse date {date_start} for cohort ID: {e}, using YYYY-MM format")
            date_str = date_start[:7] if len(date_start) >= 7 else date_start  # YYYY-MM
    
    # Extract universe/config
    cs_config = cohort_metadata.get('cs_config', {})
    
    # Use SST accessor for universe_sig if available, otherwise fallback to direct access
    if _EXTRACT_UNIVERSE_SIG_AVAILABLE and extract_universe_sig:
        universe = extract_universe_sig(cohort_metadata, cs_config) or 'default'
    else:
        universe = cs_config.get('universe_sig', 'default')
    
    min_cs = cs_config.get('min_cs', '')
    max_cs = cs_config.get('max_cs_samples', '')
    leak_ver = cs_config.get('leakage_filter_version', 'v1')
    
    # Build readable parts
    parts = [mode_prefix]
    if date_str:
        parts.append(date_str)
    if universe and universe != 'default':
        parts.append(universe)
    if min_cs:
        parts.append(f"min_cs{min_cs}")
    if max_cs and max_cs != 100000:  # Only include if non-default
        parts.append(f"max{max_cs}")
    parts.append(leak_ver.replace('.', '_'))
    
    cohort_id = "_".join(parts)
    
    # Add short hash for uniqueness
    # Create deterministic hash for final uniqueness check
    # SST: Use sha256_short helper for consistent hashing
    from TRAINING.common.utils.config_hashing import sha256_short
    hash_str = "|".join([
        str(cohort_metadata.get('n_effective_cs', '')),
        str(cohort_metadata.get('n_symbols', '')),
        date_start,
        date_end,
        json.dumps(cs_config, sort_keys=True)
    ])
    short_hash = sha256_short(hash_str, 8)
    
    return f"{cohort_id}_{short_hash}"
