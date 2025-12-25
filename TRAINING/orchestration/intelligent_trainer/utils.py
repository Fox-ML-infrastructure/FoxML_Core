# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Intelligent Trainer Utilities

Utility functions for intelligent training orchestrator.
"""

import datetime
import numpy as np
from typing import Any, Dict, Optional

# Try to import pandas for Timestamp handling
try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False


def json_default(obj: Any) -> Any:
    """
    Fallback serializer for json.dump when saving ranking cache.
    Handles pandas / numpy / datetime objects.
    """
    # Datetime-like
    if isinstance(obj, (datetime.datetime, datetime.date)):
        # ISO-8601 string is human readable and round-trippable enough for our use
        return obj.isoformat()
    
    # Pandas Timestamp (must check after datetime since pd.Timestamp is a subclass)
    if _PANDAS_AVAILABLE:
        try:
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
        except ImportError:
            pass
    
    # Numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    
    # Numpy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Anything else falls back to string representation
    return str(obj)


def get_sample_size_bin(n_effective: int) -> Dict[str, Any]:
    """
    Compute sample size bin for directory organization.
    
    Bins:
        - sample_0-5k: 0 <= N < 5,000
        - sample_5k-10k: 5,000 <= N < 10,000
        - sample_10k-25k: 10,000 <= N < 25,000
        - sample_25k-50k: 25,000 <= N < 50,000
        - sample_50k-100k: 50,000 <= N < 100,000
        - sample_100k-250k: 100,000 <= N < 250,000
        - sample_250k-500k: 250,000 <= N < 500,000
        - sample_500k-1M: 500,000 <= N < 1,000,000
        - sample_1M+: N >= 1,000,000
    
    This groups runs with similar cross-sectional sample sizes together for easy comparison.
    **Note:** Bin is for directory organization only. Trend series keys use stable identity (cohort_id, stage, target)
    and do NOT include bin_name to prevent fragmentation when binning scheme changes.
    
    Args:
        n_effective: Effective sample size
        
    Returns:
        Dict with keys: bin_name, bin_min, bin_max, binning_scheme_version
    """
    BINNING_SCHEME_VERSION = "sample_bin_v1"
    
    # Define bins with EXCLUSIVE upper bounds (bin_min <= N < bin_max)
    bins = [
        (0, 5000, "sample_0-5k"),
        (5000, 10000, "sample_5k-10k"),
        (10000, 25000, "sample_10k-25k"),
        (25000, 50000, "sample_25k-50k"),
        (50000, 100000, "sample_50k-100k"),
        (100000, 250000, "sample_100k-250k"),
        (250000, 500000, "sample_250k-500k"),
        (500000, 1000000, "sample_500k-1M"),
        (1000000, float('inf'), "sample_1M+")
    ]
    
    for bin_min, bin_max, bin_name in bins:
        if bin_min <= n_effective < bin_max:
            return {
                "bin_name": bin_name,
                "bin_min": bin_min,
                "bin_max": bin_max if bin_max != float('inf') else None,
                "binning_scheme_version": BINNING_SCHEME_VERSION
            }
    
    # Fallback (should never reach here)
    return {
        "bin_name": "sample_unknown",
        "bin_min": None,
        "bin_max": None,
        "binning_scheme_version": BINNING_SCHEME_VERSION
    }

