"""
Fingerprinting utilities for feature sets and other hashable collections.

This module provides the SINGLE SOURCE OF TRUTH for fingerprint computation.
Do not duplicate these functions elsewhere.
"""

import hashlib
from typing import Iterable, Optional, Tuple


def compute_feature_fingerprint(
    feature_names: Iterable[str],
    set_invariant: bool = True
) -> Tuple[str, str]:
    """
    Compute feature set fingerprints (set-invariant and order-sensitive).
    
    This is the canonical implementation - use this everywhere instead of
    local copies in cross_sectional_data.py or leakage_budget.py.
    
    Args:
        feature_names: Iterable of feature names
        set_invariant: If True, compute set-invariant fingerprint (sorted). 
                      If False, preserve order for the first return value.
    
    Returns:
        (set_fingerprint, order_fingerprint) tuple:
        - set_fingerprint: Set-invariant fingerprint (sorted, for set equality checks)
        - order_fingerprint: Order-sensitive fingerprint (for order-change detection)
    """
    feature_list = list(feature_names)
    
    # Set-invariant fingerprint (sorted, for set equality)
    sorted_features = sorted(feature_list)
    set_str = "\n".join(sorted_features)
    set_fingerprint = hashlib.sha1(set_str.encode()).hexdigest()[:8]
    
    # Order-sensitive fingerprint (for order-change detection)
    order_str = "\n".join(feature_list)
    order_fingerprint = hashlib.sha1(order_str.encode()).hexdigest()[:8]
    
    return set_fingerprint, order_fingerprint


# Alias for backward compatibility with existing code using underscore prefix
_compute_feature_fingerprint = compute_feature_fingerprint


def compute_data_fingerprint(
    n_symbols: Optional[int] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    min_cs: Optional[int] = None,
    max_cs_samples: Optional[int] = None,
    data_fingerprint: Optional[str] = None
) -> Optional[str]:
    """
    Compute data fingerprint from cohort metadata.
    
    SST (Single Source of Truth) for data identity fingerprinting.
    
    Args:
        n_symbols: Number of symbols in dataset
        date_start: Date range start (ISO format)
        date_end: Date range end (ISO format)
        min_cs: Minimum cross-sectional size
        max_cs_samples: Maximum cross-sectional samples
        data_fingerprint: Pre-computed data hash (if available)
    
    Returns:
        16-character hex fingerprint, or None if no data provided
    """
    data_parts = []
    
    if n_symbols is not None:
        data_parts.append(f"n_symbols={n_symbols}")
    if date_start is not None:
        data_parts.append(f"date_start={date_start}")
    if date_end is not None:
        data_parts.append(f"date_end={date_end}")
    if min_cs is not None:
        data_parts.append(f"min_cs={min_cs}")
    if max_cs_samples is not None:
        data_parts.append(f"max_cs_samples={max_cs_samples}")
    if data_fingerprint is not None:
        data_parts.append(f"data_id={data_fingerprint}")
    
    if data_parts:
        # Canonicalize: sorted keys for deterministic output
        data_str = "|".join(sorted(data_parts))
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    return None


def compute_config_fingerprint(
    min_cs: Optional[int] = None,
    max_cs_samples: Optional[int] = None,
    leakage_filter_version: Optional[str] = None,
    universe_sig: Optional[str] = None,
    cv_method: Optional[str] = None,
    folds: Optional[int] = None,
    purge_minutes: Optional[float] = None,
    embargo_minutes: Optional[float] = None,
    horizon_minutes: Optional[float] = None,
    **extra_config
) -> Optional[str]:
    """
    Compute config fingerprint from configuration parameters.
    
    SST (Single Source of Truth) for config identity fingerprinting.
    
    Args:
        min_cs: Minimum cross-sectional size
        max_cs_samples: Maximum cross-sectional samples  
        leakage_filter_version: Leakage filter version
        universe_sig: Universe signature
        cv_method: Cross-validation method
        folds: Number of CV folds
        purge_minutes: Purge window in minutes
        embargo_minutes: Embargo window in minutes
        horizon_minutes: Prediction horizon in minutes
        **extra_config: Additional config parameters to include
    
    Returns:
        16-character hex fingerprint, or None if no config provided
    """
    config_parts = []
    
    if min_cs is not None:
        config_parts.append(f"min_cs={min_cs}")
    if max_cs_samples is not None:
        config_parts.append(f"max_cs_samples={max_cs_samples}")
    if leakage_filter_version is not None:
        config_parts.append(f"leakage_filter={leakage_filter_version}")
    if universe_sig is not None:
        config_parts.append(f"universe={universe_sig}")
    if cv_method is not None:
        config_parts.append(f"cv_method={cv_method}")
    if folds is not None:
        config_parts.append(f"folds={folds}")
    if purge_minutes is not None:
        config_parts.append(f"purge={purge_minutes}")
    if embargo_minutes is not None:
        config_parts.append(f"embargo={embargo_minutes}")
    if horizon_minutes is not None:
        config_parts.append(f"horizon={horizon_minutes}")
    
    # Include extra config parameters
    for key, val in sorted(extra_config.items()):
        if val is not None:
            config_parts.append(f"{key}={val}")
    
    if config_parts:
        config_str = "|".join(sorted(config_parts))
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    return None


def compute_target_fingerprint(
    target: Optional[str] = None,
    target_column: Optional[str] = None,
    label_definition_hash: Optional[str] = None
) -> Optional[str]:
    """
    Compute target fingerprint from target parameters.
    
    SST (Single Source of Truth) for target identity fingerprinting.
    
    Args:
        target: Target name (e.g., "ret_5m_cs")
        target_column: Target column in data
        label_definition_hash: Pre-computed label definition hash
    
    Returns:
        16-character hex fingerprint, or None if no target info provided
    """
    target_parts = []
    
    if target is not None:
        target_parts.append(f"name={target}")
    if target_column is not None:
        target_parts.append(f"column={target_column}")
    if label_definition_hash is not None:
        target_parts.append(f"label_def={label_definition_hash}")
    
    if target_parts:
        target_str = "|".join(sorted(target_parts))
        return hashlib.sha256(target_str.encode()).hexdigest()[:16]
    return None
