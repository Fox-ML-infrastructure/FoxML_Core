"""
Fingerprinting utilities for feature sets and other hashable collections.

This module provides domain-specific fingerprint functions.
All functions use canonicalization from config_hashing.py (SST).

Do not duplicate fingerprint logic elsewhere.
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Import canonicalization SST from config_hashing
from TRAINING.common.utils.config_hashing import (
    canonical_json,
    sha256_full,
    sha256_short,
)


# =============================================================================
# RUN IDENTITY (SST Object)
# =============================================================================

@dataclass
class RunIdentity:
    """
    Single Source of Truth for run identity.
    
    Computed once at run initialization, passed to all downstream components.
    Persisted alongside run artifacts for future reference.
    
    Keys:
    - strict_key: Full identity including train_seed (for diff telemetry)
    - replicate_key: Identity without train_seed (for stability analysis across seeds)
    - debug_key: Human-readable key for logs (uses short hashes)
    
    All identity keys use full 64-char SHA256 hashes to avoid collisions.
    """
    # Schema version (bump if component structure changes)
    schema_version: int = 1
    
    # Component signatures (64-char SHA256)
    dataset_signature: str = ""
    split_signature: str = ""
    target_signature: str = ""
    feature_signature: str = ""
    hparams_signature: str = ""
    routing_signature: str = ""
    
    # Optional signatures
    library_versions_signature: Optional[str] = None
    
    # Training randomness
    train_seed: Optional[int] = None
    
    # Pre-computed keys (computed in __post_init__)
    strict_key: str = field(init=False, default="")
    replicate_key: str = field(init=False, default="")
    debug_key: str = field(init=False, default="")
    
    def __post_init__(self):
        """Compute keys from component signatures."""
        self.strict_key = self._compute_strict_key()
        self.replicate_key = self._compute_replicate_key()
        self.debug_key = self._compute_debug_key()
    
    def _compute_strict_key(self) -> str:
        """Compute strict identity key (includes train_seed)."""
        payload = {
            "schema": self.schema_version,
            "dataset": self.dataset_signature,
            "split": self.split_signature,
            "target": self.target_signature,
            "features": self.feature_signature,
            "hparams": self.hparams_signature,
            "routing": self.routing_signature,
            "seed": self.train_seed,
        }
        return sha256_full(canonical_json(payload))
    
    def _compute_replicate_key(self) -> str:
        """Compute replicate identity key (excludes train_seed for cross-seed stability)."""
        payload = {
            "schema": self.schema_version,
            "dataset": self.dataset_signature,
            "split": self.split_signature,
            "target": self.target_signature,
            "features": self.feature_signature,
            "hparams": self.hparams_signature,
            "routing": self.routing_signature,
            # NOTE: train_seed intentionally excluded
        }
        return sha256_full(canonical_json(payload))
    
    def _compute_debug_key(self) -> str:
        """Compute human-readable debug key (uses short hashes)."""
        parts = []
        if self.dataset_signature:
            parts.append(f"data={self.dataset_signature[:8]}")
        if self.split_signature:
            parts.append(f"split={self.split_signature[:8]}")
        if self.target_signature:
            parts.append(f"target={self.target_signature[:8]}")
        if self.feature_signature:
            parts.append(f"features={self.feature_signature[:8]}")
        if self.hparams_signature:
            parts.append(f"hparams={self.hparams_signature[:8]}")
        if self.routing_signature:
            parts.append(f"routing={self.routing_signature[:8]}")
        if self.train_seed is not None:
            parts.append(f"seed={self.train_seed}")
        return "|".join(parts) if parts else "empty"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "schema_version": self.schema_version,
            "dataset_signature": self.dataset_signature,
            "split_signature": self.split_signature,
            "target_signature": self.target_signature,
            "feature_signature": self.feature_signature,
            "hparams_signature": self.hparams_signature,
            "routing_signature": self.routing_signature,
            "library_versions_signature": self.library_versions_signature,
            "train_seed": self.train_seed,
            "strict_key": self.strict_key,
            "replicate_key": self.replicate_key,
            "debug_key": self.debug_key,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunIdentity':
        """Deserialize from persistence."""
        return cls(
            schema_version=data.get("schema_version", 1),
            dataset_signature=data.get("dataset_signature", ""),
            split_signature=data.get("split_signature", ""),
            target_signature=data.get("target_signature", ""),
            feature_signature=data.get("feature_signature", ""),
            hparams_signature=data.get("hparams_signature", ""),
            routing_signature=data.get("routing_signature", ""),
            library_versions_signature=data.get("library_versions_signature"),
            train_seed=data.get("train_seed"),
        )
    
    def is_complete(self) -> bool:
        """Check if all required signatures are present."""
        required = [
            self.dataset_signature,
            self.split_signature,
            self.target_signature,
            self.feature_signature,
            self.hparams_signature,
            self.routing_signature,
        ]
        return all(sig for sig in required)


def construct_comparison_group_key_from_dict(
    comparison_group: Dict[str, Any],
    mode: str = "debug"
) -> str:
    """
    Construct comparison group key from dict (for backward compatibility).
    
    SST (Single Source of Truth) for key construction from legacy dicts.
    Use this instead of duplicating key construction logic.
    
    Args:
        comparison_group: Dictionary with comparison group fields
        mode: "strict" (full 64-char hash), "replicate" (no seed), or "debug" (short hashes)
    
    Returns:
        Comparison group key string
    """
    if not comparison_group:
        return "default"
    
    if mode == "strict":
        # Full hash including seed
        payload = {
            "schema": 1,
            "dataset": comparison_group.get("dataset_signature"),
            "split": comparison_group.get("split_signature"),
            "target": comparison_group.get("task_signature"),  # Note: task_signature maps to target
            "features": comparison_group.get("feature_signature"),
            "hparams": comparison_group.get("hyperparameters_signature"),
            "routing": comparison_group.get("routing_signature"),
            "seed": comparison_group.get("train_seed"),
        }
        return sha256_full(canonical_json(payload))
    
    elif mode == "replicate":
        # Full hash excluding seed
        payload = {
            "schema": 1,
            "dataset": comparison_group.get("dataset_signature"),
            "split": comparison_group.get("split_signature"),
            "target": comparison_group.get("task_signature"),
            "features": comparison_group.get("feature_signature"),
            "hparams": comparison_group.get("hyperparameters_signature"),
            "routing": comparison_group.get("routing_signature"),
            # NOTE: train_seed intentionally excluded
        }
        return sha256_full(canonical_json(payload))
    
    else:  # debug mode - backward compatible short-hash format
        parts = []
        if comparison_group.get('experiment_id'):
            parts.append(f"exp={comparison_group['experiment_id']}")
        if comparison_group.get('dataset_signature'):
            parts.append(f"data={comparison_group['dataset_signature'][:8]}")
        if comparison_group.get('task_signature'):
            parts.append(f"task={comparison_group['task_signature'][:8]}")
        if comparison_group.get('routing_signature'):
            parts.append(f"route={comparison_group['routing_signature'][:8]}")
        if comparison_group.get('n_effective') is not None:
            parts.append(f"n={comparison_group['n_effective']}")
        if comparison_group.get('model_family'):
            parts.append(f"family={comparison_group['model_family']}")
        if comparison_group.get('feature_signature'):
            parts.append(f"features={comparison_group['feature_signature'][:8]}")
        if comparison_group.get('hyperparameters_signature'):
            parts.append(f"hps={comparison_group['hyperparameters_signature'][:8]}")
        if comparison_group.get('train_seed') is not None:
            parts.append(f"seed={comparison_group['train_seed']}")
        if comparison_group.get('library_versions_signature'):
            parts.append(f"libs={comparison_group['library_versions_signature'][:8]}")
        
        return "|".join(parts) if parts else "default"


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
    
    Uses canonical_json from config_hashing SST.
    
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
    payload = {
        "schema": 1,
        "n_symbols": n_symbols,
        "date_start": date_start,
        "date_end": date_end,
        "min_cs": min_cs,
        "max_cs_samples": max_cs_samples,
        "data_id": data_fingerprint,
    }
    
    # canonical_json drops None values, so empty payload becomes "{}"
    json_str = canonical_json(payload)
    if json_str == '{"schema":1}':
        return None
    
    return sha256_short(json_str, 16)


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
    
    Uses canonical_json from config_hashing SST.
    
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
    payload = {
        "schema": 1,
        "min_cs": min_cs,
        "max_cs_samples": max_cs_samples,
        "leakage_filter_version": leakage_filter_version,
        "universe_sig": universe_sig,
        "cv_method": cv_method,
        "folds": folds,
        "purge_minutes": purge_minutes,
        "embargo_minutes": embargo_minutes,
        "horizon_minutes": horizon_minutes,
        **extra_config,
    }
    
    json_str = canonical_json(payload)
    if json_str == '{"schema":1}':
        return None
    
    return sha256_short(json_str, 16)


def compute_target_fingerprint(
    target: Optional[str] = None,
    target_column: Optional[str] = None,
    label_definition_hash: Optional[str] = None,
    # Extended parameters for full target identity
    horizon_minutes: Optional[float] = None,
    objective: Optional[str] = None,
    barriers: Optional[Dict[str, Any]] = None,
    thresholds: Optional[Dict[str, Any]] = None,
    normalization: Optional[str] = None,
    winsorize_limits: Optional[Tuple[float, float]] = None,
    binning_rules: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    Compute target fingerprint from target parameters.
    
    Uses canonical_json from config_hashing SST.
    Extended to support full target parameterization (barriers, thresholds, etc.).
    
    Args:
        target: Target name (e.g., "ret_5m_cs")
        target_column: Target column in data
        label_definition_hash: Pre-computed label definition hash
        horizon_minutes: Prediction horizon in minutes
        objective: Target objective (regression, classification, ranking)
        barriers: Triple barrier config (if applicable)
        thresholds: Classification thresholds (if applicable)
        normalization: Normalization method (zscore, minmax, etc.)
        winsorize_limits: Winsorization limits (lower, upper percentiles)
        binning_rules: Binning rules for classification
    
    Returns:
        16-character hex fingerprint, or None if no target info provided
    """
    payload = {
        "schema": 1,
        "target": target,
        "target_column": target_column,
        "label_definition_hash": label_definition_hash,
        "horizon_minutes": horizon_minutes,
        "objective": objective,
        "barriers": barriers,
        "thresholds": thresholds,
        "normalization": normalization,
        "winsorize_limits": list(winsorize_limits) if winsorize_limits else None,
        "binning_rules": binning_rules,
    }
    
    json_str = canonical_json(payload)
    if json_str == '{"schema":1}':
        return None
    
    return sha256_short(json_str, 16)


# =============================================================================
# NEW FINGERPRINT FUNCTIONS (use full 64-char hashes for identity)
# =============================================================================

def compute_split_fingerprint(
    cv_method: str,
    n_folds: int,
    purge_minutes: float,
    embargo_minutes: float,
    fold_boundaries: List[Tuple[datetime, datetime]],
    split_seed: Optional[int] = None,
    boundary_inclusive: Tuple[bool, bool] = (True, False),
    fold_row_counts: Optional[List[Tuple[int, int]]] = None,
) -> str:
    """
    Compute split/CV fingerprint with timezone-stable boundary canonicalization.
    
    Uses canonical_json from config_hashing SST.
    
    Args:
        cv_method: Cross-validation method (purged_kfold, walk_forward, etc.)
        n_folds: Number of CV folds
        purge_minutes: Purge window in minutes
        embargo_minutes: Embargo window in minutes
        fold_boundaries: List of (start, end) datetime tuples per fold
        split_seed: Random seed for split (if applicable)
        boundary_inclusive: (start_inclusive, end_inclusive) tuple
        fold_row_counts: Optional list of (train_n, val_n) per fold after filtering
    
    Returns:
        64-character SHA256 fingerprint (full hash for identity)
    """
    # Canonicalize boundaries to UTC ISO format
    canonical_boundaries = []
    for start, end in fold_boundaries:
        canonical_boundaries.append({
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
            "start_inclusive": boundary_inclusive[0],
            "end_inclusive": boundary_inclusive[1],
        })
    
    payload = {
        "schema": 1,
        "cv_method": cv_method,
        "n_folds": n_folds,
        "purge_minutes": purge_minutes,
        "embargo_minutes": embargo_minutes,
        "boundaries": canonical_boundaries,
        "split_seed": split_seed,
        "fold_row_counts": fold_row_counts,
    }
    
    return sha256_full(canonical_json(payload))


def compute_hparams_fingerprint(
    model_family: str,
    params: Dict[str, Any],
    defaults: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Compute hyperparameters fingerprint with model family namespace.
    
    Uses canonical_json from config_hashing SST.
    
    Args:
        model_family: Model family name (lightgbm, catboost, xgboost, etc.)
        params: Hyperparameters dict
        defaults: Optional default params to merge (explicit defaults)
    
    Returns:
        64-character SHA256 fingerprint (full hash for identity)
    """
    # Merge defaults if provided
    if defaults:
        full_params = {**defaults, **params}
    else:
        full_params = params
    
    payload = {
        "schema": 1,
        "model_family": model_family,
        "params": full_params,
    }
    
    return sha256_full(canonical_json(payload))


def compute_feature_fingerprint_from_specs(
    resolved_specs: List[Dict[str, Any]],
) -> str:
    """
    Compute feature fingerprint from resolved FeatureSpec objects.
    
    CRITICAL: This hashes the FINAL feature set (post-selection/pruning),
    not candidate features. Each spec should include:
    - key: Canonical feature key (e.g., "price/returns/v3")
    - params: Feature parameters
    - scope: Feature scope (cross_sectional, symbol_specific)
    - version: Feature version
    - output_columns: List of output column names
    - impl_digest: Implementation digest (optional, for code changes)
    
    Sorts entries by canonical_json(entry) to handle collisions.
    
    Args:
        resolved_specs: List of resolved feature spec dicts
    
    Returns:
        64-character SHA256 fingerprint (full hash for identity)
    """
    manifest = []
    for spec in resolved_specs:
        entry = {
            "key": spec.get("key") or spec.get("name"),
            "params": spec.get("params", {}),
            "scope": spec.get("scope"),
            "version": spec.get("version"),
            "output_columns": spec.get("output_columns", []),
            "impl_digest": spec.get("impl_digest"),
        }
        manifest.append(entry)
    
    # Sort by serialized entry to handle key collisions
    manifest.sort(key=lambda x: canonical_json(x))
    
    payload = {
        "schema": 1,
        "features": manifest,
    }
    
    return sha256_full(canonical_json(payload))
