# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Fingerprinting utilities for feature sets and other hashable collections.

This module provides domain-specific fingerprint functions.
All functions use canonicalization from config_hashing.py (SST).

Do not duplicate fingerprint logic elsewhere.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Import canonicalization SST from config_hashing
from TRAINING.common.utils.config_hashing import (
    canonical_json,
    sha256_full,
    sha256_short,
)

logger = logging.getLogger(__name__)

# =============================================================================
# IDENTITY CONFIG LOADER
# =============================================================================

_IDENTITY_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_identity_config() -> Dict[str, Any]:
    """
    Load identity configuration from CONFIG/core/identity_config.yaml.
    
    Returns cached config after first load.
    
    Defaults (if config file not found):
    - identity.mode: "strict"
    - stability.filter_mode: "replicate"
    - stability.allow_legacy_snapshots: False
    - feature_identity.mode: "registry_resolved"
    """
    global _IDENTITY_CONFIG_CACHE
    
    if _IDENTITY_CONFIG_CACHE is not None:
        return _IDENTITY_CONFIG_CACHE
    
    # Default config (production-safe)
    defaults = {
        "identity": {"mode": "strict"},
        "stability": {
            "filter_mode": "replicate",
            "allow_legacy_snapshots": False,
            "min_snapshots": 2,
        },
        "feature_identity": {"mode": "registry_resolved"},
    }
    
    # Try to load from file (new location: core/identity_config.yaml)
    try:
        import yaml
        # Try new location first (core/), then fallback to root for backward compatibility
        config_path = Path(__file__).resolve().parents[3] / "CONFIG" / "core" / "identity_config.yaml"
        if not config_path.exists():
            # Fallback to old root location for backward compatibility
            config_path = Path(__file__).resolve().parents[3] / "CONFIG" / "identity_config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                loaded = yaml.safe_load(f) or {}
            # Merge with defaults
            for key in defaults:
                if key in loaded:
                    defaults[key].update(loaded[key])
            logger.debug(f"Loaded identity config from {config_path}")
        else:
            logger.debug(f"Identity config not found at {config_path}, using defaults")
    except Exception as e:
        logger.debug(f"Failed to load identity config: {e}, using defaults")
    
    _IDENTITY_CONFIG_CACHE = defaults
    return defaults


def get_identity_mode() -> str:
    """Get identity enforcement mode: 'strict', 'relaxed', or 'legacy'.
    
    SST: If reproducibility.yaml mode=strict, identity enforcement is also strict.
    This ensures full auditability when determinism is enforced.
    
    Fallback to identity_config.yaml for non-strict reproducibility mode.
    """
    # SST: Reproducibility strict implies identity strict
    try:
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            return "strict"
    except Exception:
        pass
    
    # Fallback to identity_config.yaml setting
    return get_identity_config().get("identity", {}).get("mode", "strict")


# =============================================================================
# TIMESTAMP CANONICALIZATION (Finance-Safe)
# =============================================================================

def _infer_epoch_unit(x: int) -> str:
    """
    Infer epoch unit from magnitude.
    
    2026-ish magnitudes:
      s  ~ 1e9
      ms ~ 1e12
      us ~ 1e15
      ns ~ 1e18
    """
    ax = abs(int(x))
    if ax >= 10**17:
        return "ns"
    if ax >= 10**14:
        return "us"
    if ax >= 10**11:
        return "ms"
    return "s"


def canonicalize_timestamp(ts: Any, *, assume_utc_for_naive: bool = False) -> Optional[str]:
    """
    Convert timestamp to stable UTC ISO string: 'YYYY-MM-DDTHH:MM:SSZ'
    
    Handles: datetime, pd.Timestamp, numpy datetime64, unix epoch (s/ms/us/ns).
    
    Args:
        ts: Any timestamp-like value
        assume_utc_for_naive: If False (default), raises on naive timestamps.
                             If True, treats naive as UTC (for relaxed mode).
    
    Returns:
        Canonical UTC string 'YYYY-MM-DDTHH:MM:SSZ' or None if ts is None.
    
    Raises:
        ValueError: If ts is naive and assume_utc_for_naive=False (strict mode).
    """
    if ts is None:
        return None

    import pandas as pd

    # Epoch number - 4-way unit inference
    if isinstance(ts, (int, float)) or type(ts).__name__ in ("int64", "int32", "float64", "float32"):
        unit = _infer_epoch_unit(int(ts))
        t = pd.Timestamp(int(ts), unit=unit)
    else:
        t = ts if isinstance(ts, pd.Timestamp) else pd.Timestamp(ts)

    # Timezone handling - explicit policy
    if t.tz is None:
        if not assume_utc_for_naive:
            raise ValueError(f"Naive timestamp encountered: {t!r}. Set assume_utc_for_naive=True or fix data.")
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    # Floor to seconds for stability (removes microsecond noise)
    t = t.floor("s")
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


# =============================================================================
# RUN IDENTITY (SST Object)
# =============================================================================

@dataclass
class RunIdentity:
    """
    Single Source of Truth for run identity.
    
    Two-phase construction:
    1. Create partial identity (without feature_signature) early in pipeline
    2. Call finalize(feature_signature) after features are locked to get final identity
    
    Keys are ONLY computed when is_final=True. Partial identities have no keys.
    
    Keys:
    - strict_key: Full identity including train_seed (for diff telemetry)
    - replicate_key: Identity without train_seed (for stability analysis across seeds)
    - debug_key: Human-readable key for logs (uses short hashes)
    
    All identity keys use full 64-char SHA256 hashes to avoid collisions.
    """
    # Schema version (bump if component structure changes) - INCLUDED IN KEY PAYLOAD
    schema_version: int = 1
    
    # Component signatures (64-char SHA256)
    dataset_signature: str = ""
    split_signature: str = ""
    target_signature: str = ""
    feature_signature: Optional[str] = None  # None for partial, set when finalized (alias for feature_signature_output)
    feature_signature_input: Optional[str] = None  # Candidate feature universe entering FS stage
    feature_signature_output: Optional[str] = None  # Selected features exiting FS stage
    hparams_signature: str = ""
    routing_signature: str = ""
    
    # Contracted routing payload (stored for debugging, not just the hash)
    routing_payload: Optional[Dict[str, Any]] = None
    
    # Optional signatures
    library_versions_signature: Optional[str] = None
    
    # Training randomness (required for strict_key)
    train_seed: Optional[int] = None
    
    # Finalization flag - keys only valid when True
    is_final: bool = False
    
    # Pre-computed keys (only set when is_final=True)
    strict_key: Optional[str] = field(init=False, default=None)
    replicate_key: Optional[str] = field(init=False, default=None)
    debug_key: Optional[str] = field(init=False, default=None)
    
    def __post_init__(self):
        """Compute keys only if finalized."""
        if self.is_final:
            self._validate_required_for_final()
            self.strict_key = self._compute_strict_key()
            self.replicate_key = self._compute_replicate_key()
            self.debug_key = self._compute_debug_key()
        else:
            self.strict_key = None
            self.replicate_key = None
            self.debug_key = None
    
    def _validate_required_for_final(self) -> None:
        """Validate all required signatures are present for finalization."""
        missing = []
        if not self.dataset_signature:
            missing.append("dataset_signature")
        if not self.split_signature:
            missing.append("split_signature")
        if not self.target_signature:
            missing.append("target_signature")
        if not self.feature_signature:
            missing.append("feature_signature")
        if not self.hparams_signature:
            missing.append("hparams_signature")
        if not self.routing_signature:
            missing.append("routing_signature")
        
        if missing:
            raise ValueError(
                f"Cannot finalize RunIdentity: missing required signatures: {missing}"
            )
    
    def finalize(self, feature_signature: str) -> 'RunIdentity':
        """
        Create a finalized identity with the given feature_signature.
        
        Returns a NEW RunIdentity object with is_final=True and computed keys.
        Does not mutate the original.
        
        Args:
            feature_signature: 64-char SHA256 of final resolved feature specs
            
        Returns:
            New RunIdentity with is_final=True and computed keys
            
        Raises:
            ValueError: If any required partial signatures are missing
        """
        if self.is_final:
            raise ValueError("RunIdentity is already finalized")
        
        return RunIdentity(
            schema_version=self.schema_version,
            dataset_signature=self.dataset_signature,
            split_signature=self.split_signature,
            target_signature=self.target_signature,
            feature_signature=feature_signature,
            feature_signature_input=self.feature_signature_input,
            feature_signature_output=feature_signature,  # Selected features = output
            hparams_signature=self.hparams_signature,
            routing_signature=self.routing_signature,
            routing_payload=self.routing_payload,
            library_versions_signature=self.library_versions_signature,
            train_seed=self.train_seed,
            is_final=True,
        )
    
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
        return "|".join(parts) if parts else "partial"
    
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
            "routing_payload": self.routing_payload,
            "library_versions_signature": self.library_versions_signature,
            "train_seed": self.train_seed,
            "is_final": self.is_final,
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
            feature_signature=data.get("feature_signature"),
            hparams_signature=data.get("hparams_signature", ""),
            routing_signature=data.get("routing_signature", ""),
            routing_payload=data.get("routing_payload"),
            library_versions_signature=data.get("library_versions_signature"),
            train_seed=data.get("train_seed"),
            is_final=data.get("is_final", False),
        )
    
    def is_complete(self) -> bool:
        """Check if all required signatures are present (alias for is_final check)."""
        return self.is_final


def create_stage_identity(
    stage: str,  # "TARGET_RANKING", "FEATURE_SELECTION", "TRAINING"
    symbols: List[str],
    experiment_config: Optional[Any] = None,
    data_dir: Optional[Path] = None,
) -> RunIdentity:
    """
    SST factory for creating partial RunIdentity objects.
    
    This is the Single Source of Truth for identity creation across all pipeline
    entry points. Use this instead of manually constructing RunIdentity objects.
    
    Call finalize(feature_signature) on the returned object after features are locked.
    
    Args:
        stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
        symbols: List of symbols in the universe
        experiment_config: Optional experiment config (for seed extraction)
        data_dir: Optional data directory (currently unused, reserved for future)
    
    Returns:
        Partial RunIdentity (is_final=False) ready for finalization
    
    Example:
        # Early in pipeline
        partial = create_stage_identity("TARGET_RANKING", symbols, experiment_config)
        
        # Later, after features are locked
        final = partial.finalize(feature_signature)
    """
    # Universe signature from symbols
    universe_sig = ""
    if symbols:
        try:
            from TRAINING.orchestration.utils.run_context import compute_universe_signature
            universe_sig = compute_universe_signature(symbols) or ""
        except Exception:
            # Fallback: compute manually (uses module-level hashlib import)
            symbols_str = "|".join(sorted(symbols))
            universe_sig = hashlib.sha256(symbols_str.encode()).hexdigest()
    
    # Seed derivation: prefer config seed, then derive from universe_sig for reproducibility
    # This ensures: same universe + same config = same seed
    base_seed = None
    if experiment_config and hasattr(experiment_config, 'seed'):
        base_seed = experiment_config.seed
    if base_seed is None:
        try:
            from CONFIG.config_loader import get_cfg
            base_seed = get_cfg("pipeline.determinism.base_seed", default=42)
        except Exception:
            base_seed = 42
    
    # Use base_seed directly for TR/FS/TRAINING consistency
    # Universe differentiation is already handled by universe_sig in comparison_group
    # This ensures: same config seed = same train_seed across all stages
    train_seed = base_seed
    
    return RunIdentity(
        dataset_signature=universe_sig,
        split_signature="",  # Computed after CV folds created
        target_signature="",  # Computed per-target
        feature_signature=None,  # Set via finalize()
        feature_signature_input=None,  # Set before FS stage starts
        feature_signature_output=None,  # Set after FS stage completes (same as feature_signature)
        hparams_signature="",  # Computed per-model
        routing_signature="",  # Computed per-view
        train_seed=train_seed,
        is_final=False,
    )


def construct_comparison_group_key_from_dict(
    comparison_group: Dict[str, Any],
    mode: str = "debug",
    stage: str = "TRAINING"
) -> Optional[str]:
    """
    Construct comparison group key from dict (for backward compatibility).
    
    SST (Single Source of Truth) for key construction from legacy dicts.
    Use this instead of duplicating key construction logic.
    
    Args:
        comparison_group: Dictionary with comparison group fields
        mode: "strict" (full 64-char hash), "replicate" (no seed), or "debug" (full format)
        stage: Stage name for key construction (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
    
    Returns:
        Comparison group key string, or None if invalid
    """
    if not comparison_group:
        return None  # No longer return "default" - invalid groups return None
    
    if mode == "strict":
        # Full hash including seed
        payload = {
            "schema": 1,
            "stage": stage,
            "dataset": comparison_group.get("dataset_signature"),
            "split": comparison_group.get("split_signature"),
            "target": comparison_group.get("task_signature"),  # Note: task_signature maps to target
            "features": comparison_group.get("feature_signature"),
            "hparams": comparison_group.get("hyperparameters_signature"),
            "routing": comparison_group.get("routing_signature"),
            "seed": comparison_group.get("train_seed"),
            "libs": comparison_group.get("library_versions_signature"),
        }
        return sha256_full(canonical_json(payload))
    
    elif mode == "replicate":
        # Full hash excluding seed
        payload = {
            "schema": 1,
            "stage": stage,
            "dataset": comparison_group.get("dataset_signature"),
            "split": comparison_group.get("split_signature"),
            "target": comparison_group.get("task_signature"),
            "features": comparison_group.get("feature_signature"),
            "hparams": comparison_group.get("hyperparameters_signature"),
            "routing": comparison_group.get("routing_signature"),
            # NOTE: train_seed intentionally excluded
        }
        return sha256_full(canonical_json(payload))
    
    else:  # debug mode - new format matching ComparisonGroup.to_key()
        # Import schema version from types
        try:
            from TRAINING.orchestration.utils.diff_telemetry.types import COMPARISON_GROUP_SCHEMA_VERSION
            schema_version = COMPARISON_GROUP_SCHEMA_VERSION
        except ImportError:
            schema_version = 1
        
        # Helper to escape pipe delimiters
        def escape_value(v: str) -> str:
            if v is None:
                return "<NONE>"
            return str(v).replace("|", "\\|")
        
        parts = [f"schema={schema_version}"]
        parts.append(f"stage={stage}")
        
        # Serialize all fields explicitly (no truthy filtering)
        exp_val = escape_value(comparison_group.get('experiment_id')) if comparison_group.get('experiment_id') is not None else "<NONE>"
        parts.append(f"exp={exp_val}")
        
        data_val = comparison_group.get('dataset_signature') if comparison_group.get('dataset_signature') is not None else "<NONE>"
        parts.append(f"data={data_val}")
        
        task_val = comparison_group.get('task_signature') if comparison_group.get('task_signature') is not None else "<NONE>"
        parts.append(f"task={task_val}")
        
        route_val = comparison_group.get('routing_signature') if comparison_group.get('routing_signature') is not None else "<NONE>"
        parts.append(f"route={route_val}")
        
        split_val = comparison_group.get('split_signature') if comparison_group.get('split_signature') is not None else "<NONE>"
        parts.append(f"split={split_val}")
        
        n_val = comparison_group.get('n_effective') if comparison_group.get('n_effective') is not None else "<NONE>"
        parts.append(f"n={n_val}")
        
        family_val = escape_value(comparison_group.get('model_family')) if comparison_group.get('model_family') is not None else "<NONE>"
        parts.append(f"family={family_val}")
        
        features_val = comparison_group.get('feature_signature') if comparison_group.get('feature_signature') is not None else "<NONE>"
        parts.append(f"features={features_val}")
        
        hps_val = comparison_group.get('hyperparameters_signature') if comparison_group.get('hyperparameters_signature') is not None else "<NONE>"
        parts.append(f"hps={hps_val}")
        
        seed_val = comparison_group.get('train_seed') if comparison_group.get('train_seed') is not None else "<NONE>"
        parts.append(f"seed={seed_val}")
        
        libs_val = comparison_group.get('library_versions_signature') if comparison_group.get('library_versions_signature') is not None else "<NONE>"
        parts.append(f"libs={libs_val}")
        
        universe_val = comparison_group.get('universe_sig') if comparison_group.get('universe_sig') is not None else "<NONE>"
        parts.append(f"universe={universe_val}")
        
        symbol_val = escape_value(comparison_group.get('symbol')) if comparison_group.get('symbol') is not None else "<NONE>"
        parts.append(f"symbol={symbol_val}")
        
        return "|".join(parts)


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
        64-character SHA256 fingerprint, or None if no target info provided
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

    # Use full 64-char hash for identity consistency
    return sha256_full(json_str)


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


def resolve_feature_specs_from_registry(
    feature_names: List[str],
    registry: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Registry-based best-effort feature manifest for fingerprinting.

    Guarantees:
    - Never returns empty-string placeholders.
    - Never silently returns names-only: degraded cases are encoded in-manifest.
    - Distinguishes explicit registry entries from auto-inferred.
    - Uses full SHA256 for entry digests (no truncation).

    LIMITATION:
    - Still reconstructs from names; bulletproof fix is to fingerprint actual
      resolved FeatureSpec objects at matrix materialization time.
    
    Args:
        feature_names: List of feature names to resolve
        registry: Optional FeatureRegistry instance (auto-loaded if None)
    
    Returns:
        List of spec dicts with registry metadata encoded
    """
    # Filter out None/non-string, dedupe, stable ordering
    names = sorted(set(str(n) for n in feature_names if n is not None))

    registry_available = True
    if registry is None:
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
        except Exception:
            registry_available = False
            registry = None

    specs: List[Dict[str, Any]] = []
    for name in names:
        entry: Dict[str, Any] = {
            "key": name,
            "registry_available": registry_available,
        }

        if not registry_available:
            # Degraded: can't resolve entries at all
            entry["registry_explicit"] = False
            specs.append(entry)
            continue

        # Check if feature is EXPLICITLY in registry vs auto-inferred
        # get_feature_metadata() never returns None - it auto-infers for unknowns
        explicit_features = getattr(registry, 'features', {}) or {}
        is_explicit = name in explicit_features

        entry["registry_explicit"] = is_explicit

        try:
            if is_explicit:
                metadata = explicit_features[name]
            else:
                metadata = registry.get_feature_metadata(name)  # Auto-inferred
        except Exception:
            # Shouldn't happen, but defensive
            metadata = {"source": "error", "lag_bars": 0, "allowed_horizons": []}

        # Contract to identity-relevant fields only
        entry_payload = {
            "lag_bars": metadata.get("lag_bars"),
            "source": metadata.get("source"),
            "allowed_horizons": metadata.get("allowed_horizons", []),
            "rejected": metadata.get("rejected", False),
            "scope": metadata.get("scope"),
            "version": metadata.get("version"),
        }

        # Full 64-char digest (no truncation)
        entry["registry_digest"] = sha256_full(canonical_json(entry_payload))
        specs.append(entry)

    return specs


def compute_feature_fingerprint_from_specs(
    resolved_specs: List[Dict[str, Any]],
) -> str:
    """
    Compute feature fingerprint from resolved feature specs.
    
    Supports two input formats:
    1. Registry-resolved specs (from resolve_feature_specs_from_registry)
       - Has registry_available, registry_explicit, registry_digest
    2. Legacy specs with full feature metadata
       - Has key, params, scope, version, output_columns, impl_digest
    
    Adds mode marker to indicate provenance:
    - registry_explicit: All features have explicit registry entries
    - registry_mixed: Some explicit, some auto-inferred
    - registry_inferred: All features auto-inferred from name patterns
    - names_only_degraded: Registry unavailable, names only
    - empty: No features
    - legacy: Using legacy spec format (no registry metadata)
    
    Args:
        resolved_specs: List of resolved feature spec dicts
    
    Returns:
        64-character SHA256 fingerprint (full hash for identity)
    """
    # Build manifest from specs
    manifest = []
    for spec in resolved_specs:
        # Check if this is a registry-resolved spec or legacy format
        if "registry_available" in spec:
            # Registry-resolved format - pass through as-is
            entry = {
                "key": spec.get("key"),
                "registry_available": spec.get("registry_available"),
                "registry_explicit": spec.get("registry_explicit"),
                "registry_digest": spec.get("registry_digest"),
            }
        else:
            # Legacy format - extract fields
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
    
    # Determine mode from registry state
    if not resolved_specs:
        mode = "empty"
    elif any("registry_available" in spec for spec in resolved_specs):
        # Registry-resolved format
        if all(not spec.get("registry_available", True) for spec in resolved_specs):
            mode = "names_only_degraded"
        elif all(spec.get("registry_explicit", False) for spec in resolved_specs):
            mode = "registry_explicit"
        elif any(spec.get("registry_explicit", False) for spec in resolved_specs):
            mode = "registry_mixed"
        else:
            mode = "registry_inferred"
    else:
        # Legacy format
        mode = "legacy"
    
    payload = {
        "schema": 1,
        "mode": mode,
        "features": manifest,
    }
    
    return sha256_full(canonical_json(payload))


def compute_routing_fingerprint(
    view: str,
    symbol: Optional[str] = None,
    min_symbols_threshold: Optional[int] = None,
    auto_flip_enabled: bool = False,
    gating_flags: Optional[Dict[str, bool]] = None,
    feature_availability: Optional[Dict[str, bool]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """
    Compute routing fingerprint from contracted routing payload.
    
    CRITICAL: Only hash fields that affect model behavior, not runtime noise.
    Returns both the signature AND the contracted payload for debugging.
    
    Args:
        view: Modeling view (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
        symbol: Symbol being processed (if SYMBOL_SPECIFIC)
        min_symbols_threshold: Threshold for auto-flip to symbol-specific
        auto_flip_enabled: Whether auto-flip is enabled
        gating_flags: Dict of gating flags that affect feature availability
        feature_availability: Dict of feature -> available for this routing
    
    Returns:
        Tuple of (64-char signature, contracted payload dict)
        Store the payload for debugging "why did routing_signature change?"
    """
    # Build contracted payload (only behavior-affecting fields)
    contracted_payload = {
        "schema": 1,
        "view": view,
        "symbol": symbol,
        "min_symbols_threshold": min_symbols_threshold,
        "auto_flip_enabled": auto_flip_enabled,
        "gating_flags": gating_flags,
        "feature_availability": feature_availability,
    }
    
    signature = sha256_full(canonical_json(contracted_payload))
    
    return signature, contracted_payload
