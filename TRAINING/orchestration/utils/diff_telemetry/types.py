# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Diff Telemetry Types

Data classes and enums for diff telemetry system.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional, List


class ChangeSeverity(str, Enum):
    """Severity levels for changes."""
    CRITICAL = "critical"  # Hard invariants (splits, targets, leakage)
    MAJOR = "major"  # Important but not breaking (hyperparams, versions)
    MINOR = "minor"  # Soft changes (metrics, minor config)
    NONE = "none"  # No meaningful change


class ComparabilityStatus(str, Enum):
    """Comparability status for runs."""
    COMPARABLE = "comparable"  # Same comparison group, can diff
    INCOMPARABLE = "incomparable"  # Different groups, don't diff
    PARTIAL = "partial"  # Some overlap, diff with warnings


@dataclass
class ResolvedRunContext:
    """Resolved run context (SST) - all outcome-influencing metadata resolved at source.
    
    This ensures snapshots have non-null values for all required fields, preventing
    false comparability from None values and ensuring auditability.
    """
    # Data provenance (required for all stages)
    n_symbols: Optional[int] = None
    symbols: Optional[List[str]] = None
    date_range_start: Optional[str] = None  # ISO format
    date_range_end: Optional[str] = None  # ISO format
    n_rows_total: Optional[int] = None
    n_effective: Optional[int] = None
    data_fingerprint: Optional[str] = None
    
    # Task provenance (required for all stages)
    target_name: Optional[str] = None
    labeling_impl_hash: Optional[str] = None
    horizon_minutes: Optional[int] = None
    objective: Optional[str] = None
    target_fingerprint: Optional[str] = None
    
    # Split provenance (required for all stages)
    cv_method: Optional[str] = None
    cv_folds: Optional[int] = None
    purge_minutes: Optional[float] = None
    embargo_minutes: Optional[Any] = None  # Can be dict with kind/reason
    leakage_filter_version: Optional[str] = None
    split_seed: Optional[int] = None
    fold_assignment_hash: Optional[str] = None
    split_protocol_fingerprint: Optional[str] = None
    
    # Feature provenance (required for FEATURE_SELECTION and TRAINING)
    feature_names: Optional[List[str]] = None
    feature_set_id: Optional[str] = None
    feature_pipeline_signature: Optional[str] = None
    n_features: Optional[int] = None
    feature_fingerprint: Optional[str] = None
    
    # Stage strategy (stage-specific)
    ranking_strategy: Optional[str] = None  # For TARGET_RANKING
    feature_selection_strategy: Optional[str] = None  # For FEATURE_SELECTION
    trainer_strategy: Optional[str] = None  # For TRAINING
    model_family: Optional[str] = None  # For TRAINING (required)
    
    # Config provenance
    min_cs: Optional[int] = None
    max_cs_samples: Optional[int] = None
    
    # Environment (tracked but not outcome-influencing)
    python_version: Optional[str] = None
    library_versions: Optional[Dict[str, str]] = None
    cuda_version: Optional[str] = None
    
    # Experiment tracking
    experiment_id: Optional[str] = None
    
    # View/routing
    view: Optional[str] = None
    routing_signature: Optional[str] = None


@dataclass
class ComparisonGroup:
    """Defines what makes runs comparable.
    
    CRITICAL: Only runs with EXACTLY the same outcome-influencing metadata are comparable.
    This includes:
    - Exact N_effective (sample size) - 5k runs only compare against 5k runs
    - Same dataset (universe, date range, min_cs, max_cs_samples)
    - Same task (target, horizon, objective)
    - Same routing/view configuration
    - Same model_family (different families produce different outcomes)
    - Same feature set (different features produce different outcomes)
    - Same hyperparameters (learning_rate, max_depth, etc. - CRITICAL: impact outcomes)
    - Same train_seed (CRITICAL: different seeds = different outcomes)
    - Same library versions (CRITICAL: different versions = different outcomes)
    
    Runs are stored together ONLY if they match exactly on all these dimensions.
    """
    experiment_id: Optional[str] = None
    dataset_signature: Optional[str] = None  # Hash of universe + time rules + min_cs + max_cs_samples
    task_signature: Optional[str] = None  # Hash of target + horizon + objective
    routing_signature: Optional[str] = None  # Hash of routing config
    n_effective: Optional[int] = None  # Exact sample size (CRITICAL: must match exactly)
    model_family: Optional[str] = None  # Model family (CRITICAL: different families = different outcomes)
    feature_signature: Optional[str] = None  # Hash of feature set (CRITICAL: different features = different outcomes)
    hyperparameters_signature: Optional[str] = None  # Hash of hyperparameters (CRITICAL: different HPs = different outcomes)
    train_seed: Optional[int] = None  # Training seed (CRITICAL: different seeds = different outcomes)
    library_versions_signature: Optional[str] = None  # Hash of library versions (CRITICAL: different versions = different outcomes)
    
    def to_key(self) -> str:
        """Generate comparison group key.
        
        This key is used to:
        1. Group runs in storage (runs with same key stored together)
        2. Find previous comparable runs (only runs with same key are comparable)
        3. Establish baselines (baselines are per comparison_group_key)
        """
        parts = []
        if self.experiment_id:
            parts.append(f"exp={self.experiment_id}")
        if self.dataset_signature:
            parts.append(f"data={self.dataset_signature[:8]}")
        if self.task_signature:
            parts.append(f"task={self.task_signature[:8]}")
        if self.routing_signature:
            parts.append(f"route={self.routing_signature[:8]}")
        # CRITICAL: Include exact N_effective to ensure only identical sample sizes compare
        if self.n_effective is not None:
            parts.append(f"n={self.n_effective}")
        # CRITICAL: Include model_family (different families = different outcomes)
        if self.model_family:
            parts.append(f"family={self.model_family}")
        # CRITICAL: Include feature signature (different features = different outcomes)
        if self.feature_signature:
            parts.append(f"features={self.feature_signature[:8]}")
        # CRITICAL: Include hyperparameters signature (different HPs = different outcomes)
        if self.hyperparameters_signature:
            parts.append(f"hps={self.hyperparameters_signature[:8]}")
        # CRITICAL: Include train_seed (different seeds = different outcomes)
        if self.train_seed is not None:
            parts.append(f"seed={self.train_seed}")
        # CRITICAL: Include library versions signature (different versions = different outcomes)
        if self.library_versions_signature:
            parts.append(f"libs={self.library_versions_signature[:8]}")
        return "|".join(parts) if parts else "default"
    
    def to_dir_name(self) -> str:
        """Generate compact, filesystem-safe directory name from comparison group key.
        
        Uses a single hash of the comparison group key for compactness, with
        optional human-readable components (n_effective, model_family) if available.
        
        Example: "exp=test|data=abc12345|task=def67890|n=5000|family=lightgbm|features=ghi11111"
        -> "cg-abc12345_n-5000" (compact hash + n_effective)
        """
        key = self.to_key()
        import hashlib
        
        # Generate a single compact hash from the full key
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:12]  # 12 chars for uniqueness
        
        # Extract human-readable components for readability
        parts = []
        
        # Always include n_effective if available (most important for organization)
        if self.n_effective is not None:
            parts.append(f"n-{self.n_effective}")
        
        # Optionally include model_family if available (for quick identification)
        if self.model_family:
            # Normalize family name to be filesystem-safe
            family_clean = self.model_family.lower().replace("_", "-")
            parts.append(f"fam-{family_clean}")
        
        # Build directory name: hash + optional readable parts
        if parts:
            dir_name = f"cg-{key_hash}_{'_'.join(parts)}"
        else:
            dir_name = f"cg-{key_hash}"
        
        # Sanitize for filesystem safety
        import re
        dir_name = re.sub(r'[^a-zA-Z0-9_-]', '', dir_name)
        
        # Limit length (shouldn't be needed with hash, but safety check)
        if len(dir_name) > 200:
            dir_name = f"cg-{key_hash}"
        
        return dir_name if dir_name else "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class NormalizedSnapshot:
    """Normalized snapshot for diffing (SST-compliant)."""
    # Core identifiers
    run_id: str
    timestamp: str
    stage: str  # TARGET_RANKING, FEATURE_SELECTION, TRAINING
    view: Optional[str] = None  # CROSS_SECTIONAL, SYMBOL_SPECIFIC
    target: Optional[str] = None
    symbol: Optional[str] = None
    
    # Monotonic sequence number for correct ordering (assigned at save time)
    # This ensures correct "prev run" selection regardless of mtime/timestamp quirks
    snapshot_seq: Optional[int] = None
    
    # Fingerprint schema version (for compatibility checking)
    fingerprint_schema_version: str = "1.0"  # FINGERPRINT_SCHEMA_VERSION
    
    # Fingerprints (for change detection)
    config_fingerprint: Optional[str] = None
    data_fingerprint: Optional[str] = None
    feature_fingerprint: Optional[str] = None
    target_fingerprint: Optional[str] = None
    
    # Output digests (for artifact/metric reproducibility verification)
    metrics_sha256: Optional[str] = None  # SHA256 of metrics dict (enables metric reproducibility comparison)
    artifacts_manifest_sha256: Optional[str] = None  # SHA256 of artifacts manifest (enables artifact reproducibility comparison)
    predictions_sha256: Optional[str] = None  # SHA256 of predictions (if available, enables prediction reproducibility comparison)
    
    # Fingerprint source descriptions (for auditability)
    fingerprint_sources: Dict[str, str] = field(default_factory=dict)
    # e.g., {"fold_assignment_hash": "hash over row_id→fold_id mapping"}
    
    # Inputs (what was fed to the run)
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Process (what happened during execution)
    process: Dict[str, Any] = field(default_factory=dict)
    
    # Outputs (what was produced)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Comparability
    comparison_group: Optional[ComparisonGroup] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.comparison_group:
            result['comparison_group'] = self.comparison_group.to_dict()
        return result
    
    def to_hash(self) -> str:
        """Generate hash of normalized snapshot (for deduplication)."""
        import hashlib
        import json
        import numpy as np
        # Hash only the diffable parts (exclude run_id, timestamp)
        hashable = {
            'stage': self.stage,
            'view': self.view,
            'target': self.target,
            'symbol': self.symbol,
            'config_fingerprint': self.config_fingerprint,
            'data_fingerprint': self.data_fingerprint,
            'feature_fingerprint': self.feature_fingerprint,
            'target_fingerprint': self.target_fingerprint,
            'inputs': self._normalize_for_hash(self.inputs),
            'process': self._normalize_for_hash(self.process),
            'outputs': self._normalize_for_hash(self.outputs)
        }
        json_str = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def _normalize_for_hash(obj: Any) -> Any:
        """Normalize object for hashing (sort, round floats, etc.)."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: NormalizedSnapshot._normalize_for_hash(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [NormalizedSnapshot._normalize_for_hash(v) for v in sorted(obj) if v is not None]
        elif isinstance(obj, float):
            # Round to 6 decimal places for stability
            return round(obj, 6) if not np.isnan(obj) and not np.isinf(obj) else None
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        else:
            return str(obj)


@dataclass
class DiffResult:
    """Result of diffing two snapshots."""
    prev_run_id: Optional[str]  # Previous run ID (None if no previous run)
    current_run_id: str
    comparable: bool
    comparability_reason: Optional[str] = None
    
    # Previous run metadata (for auditability and validation)
    prev_timestamp: Optional[str] = None  # When the previous run happened
    prev_snapshot_seq: Optional[int] = None  # Sequence number of previous snapshot
    prev_stage: Optional[str] = None  # Stage of previous run (should match current)
    prev_view: Optional[str] = None  # View of previous run (should match current)
    comparison_source: Optional[str] = None  # Where previous run was found: "same_run", "snapshot_index", "comparison_group_directory", or None if no previous run
    
    # Change detection
    changed_keys: List[str] = field(default_factory=list)  # Canonical paths
    severity: ChangeSeverity = ChangeSeverity.NONE
    severity_reason: Optional[str] = None  # CRITICAL: Explain why severity was set
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Excluded factors changed (hyperparameters, seeds, versions)
    # These are tracked but don't block comparability
    excluded_factors_changed: Dict[str, Any] = field(default_factory=dict)
    
    # Patch operations (JSON-Patch style)
    patch: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metric deltas
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Trend deltas (comparison of trend analysis between consecutive runs)
    trend_deltas: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['severity'] = self.severity.value
        return d


@dataclass
class BaselineState:
    """Baseline state for a comparison group."""
    comparison_group_key: str
    baseline_run_id: str
    baseline_timestamp: str
    baseline_metrics: Dict[str, float]
    established_at: str
    update_count: int = 0
    regression_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

