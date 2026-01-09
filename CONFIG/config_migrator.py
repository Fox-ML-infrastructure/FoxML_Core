# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Config Migration Adapter

Migrates deprecated config keys to their canonical SST names.
This allows old configs to continue working while the codebase uses canonical names.

Usage:
    from CONFIG.config_migrator import migrate_config
    
    config = yaml.safe_load(f)
    config = migrate_config(config)  # Migrates deprecated keys, warns on use
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Deprecated key -> canonical key mapping
# Note: We only migrate top-level and well-known nested keys
DEPRECATED_KEYS: Dict[str, str] = {
    # Seed
    "random_state": "seed",
    "random_seed": "seed",
    # Folds
    "n_folds": "folds",
    "cv_folds": "folds",
    "num_folds": "folds",
    # Universe
    "universe_id": "universe_sig",
    # Output directory
    "out_dir": "output_dir",
    "results_dir": "output_dir",
    # Target
    "target_name": "target",
    "item_name": "target",
    # Sample size
    "N_effective": "n_effective",
    "N_effective_cs": "n_effective",
    # View/mode
    "resolved_mode": "view",
    "route_type": "view",
    "requested_mode": "requested_view",
    "mode_reason": "view_reason",
    # Dates
    "date_range_start": "date_start",
    "date_range_end": "date_end",
    "start_date": "date_start",
    "end_date": "date_end",
    # Metrics
    "mean_score": "auc",
    "cs_auc": "auc",
    "cs_logloss": "logloss",
    "cs_pr_auc": "pr_auc",
    # Hashes
    "cs_config_hash": "config_hash",
    "featureset_fingerprint": "featureset_hash",
}


def migrate_config(
    config: Dict[str, Any],
    warn: bool = True,
    recursive: bool = True,
    path: str = ""
) -> Dict[str, Any]:
    """
    Migrate deprecated config keys to canonical SST names.
    
    Args:
        config: Configuration dictionary to migrate
        warn: If True, log warnings for each migrated key
        recursive: If True, recursively migrate nested dicts
        path: Internal - current path for logging
        
    Returns:
        Migrated config dictionary (mutates in place and returns same object)
        
    Raises:
        ValueError: If both deprecated and canonical keys exist (ambiguous)
    """
    if not isinstance(config, dict):
        return config
    
    keys_to_migrate: List[str] = []
    
    # First pass: identify keys to migrate
    for old_key, new_key in DEPRECATED_KEYS.items():
        if old_key in config:
            if new_key in config:
                # Both exist - this is ambiguous and should error
                full_path = f"{path}.{old_key}" if path else old_key
                raise ValueError(
                    f"Ambiguous config: both deprecated key '{old_key}' and "
                    f"canonical key '{new_key}' exist at {full_path}. "
                    f"Remove the deprecated key."
                )
            keys_to_migrate.append(old_key)
    
    # Second pass: migrate keys
    for old_key in keys_to_migrate:
        new_key = DEPRECATED_KEYS[old_key]
        config[new_key] = config.pop(old_key)
        
        if warn:
            full_path = f"{path}.{old_key}" if path else old_key
            logger.warning(
                f"Config migration: '{full_path}' -> '{new_key}' "
                f"(deprecated key used, please update your config)"
            )
    
    # Recursively migrate nested dicts
    if recursive:
        for key, value in config.items():
            if isinstance(value, dict):
                child_path = f"{path}.{key}" if path else key
                migrate_config(value, warn=warn, recursive=True, path=child_path)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        child_path = f"{path}.{key}[{i}]" if path else f"{key}[{i}]"
                        migrate_config(item, warn=warn, recursive=True, path=child_path)
    
    return config


def get_deprecated_keys() -> Dict[str, str]:
    """Return the deprecated key -> canonical key mapping."""
    return DEPRECATED_KEYS.copy()
