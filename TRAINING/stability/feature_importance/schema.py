# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Importance Snapshot Schema

Standardized format for storing feature importance snapshots for stability analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime
from pathlib import Path
import uuid


@dataclass
class FeatureImportanceSnapshot:
    """
    Snapshot of feature importance for a single run.
    
    Used for tracking stability across runs, universes, and methods.
    
    Identity fields (for strict/replicate grouping):
    - strict_key: Full 64-char SHA256 (includes train_seed)
    - replicate_key: Full 64-char SHA256 (excludes train_seed)
    - Component signatures: dataset, split, target, feature, hparams, routing
    """
    target: str
    method: str                      # "quick_pruner", "rfe", "boruta", "lightgbm", etc.
    universe_sig: Optional[str]       # "TOP100", "ALL", "MEGA_CAP", symbol name, or None
    run_id: str                      # UUID or timestamp-based identifier
    created_at: datetime
    features: List[str]              # Feature names (same order as importances)
    importances: List[float]         # Importance values (same order as features)
    
    # Identity keys (computed from signatures)
    strict_key: Optional[str] = None      # Full 64-char hash (includes seed)
    replicate_key: Optional[str] = None   # Full 64-char hash (excludes seed)
    
    # Component signatures (64-char SHA256 each)
    feature_signature: Optional[str] = None
    split_signature: Optional[str] = None
    target_signature: Optional[str] = None
    hparams_signature: Optional[str] = None
    dataset_signature: Optional[str] = None
    routing_signature: Optional[str] = None
    
    # Training randomness
    train_seed: Optional[int] = None
    
    # Prediction fingerprints (for determinism verification and drift detection)
    prediction_hash: Optional[str] = None          # Strict bitwise hash
    prediction_hash_live: Optional[str] = None     # Quantized hash for drift detection
    prediction_row_ids_hash: Optional[str] = None  # Hash of row identifiers
    prediction_classes_hash: Optional[str] = None  # Hash of class order (classification)
    prediction_kind: Optional[str] = None          # "regression", "binary_proba", etc.
    
    @classmethod
    def from_dict_series(
        cls,
        target: str,
        method: str,
        importance_dict: Dict[str, float],
        universe_sig: Optional[str] = None,
        run_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        run_identity: Optional[Dict] = None,
    ) -> 'FeatureImportanceSnapshot':
        """
        Create snapshot from dictionary of feature -> importance.
        
        Args:
            target: Target name (e.g., "peak_60m_0.8")
            method: Method name (e.g., "lightgbm", "quick_pruner")
            importance_dict: Dictionary mapping feature names to importance values
            universe_sig: Optional universe identifier
            run_id: Optional run ID (generates UUID if not provided)
            created_at: Optional creation timestamp (uses now if not provided)
            run_identity: Optional RunIdentity.to_dict() for identity signatures
        
        Returns:
            FeatureImportanceSnapshot instance
        """
        # Sort features by importance (descending) for consistent ordering
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        features = [f for f, _ in sorted_items]
        importances = [imp for _, imp in sorted_items]
        
        if run_id is None:
            run_id = str(uuid.uuid4())
        
        if created_at is None:
            created_at = datetime.utcnow()
        
        # Extract identity fields from run_identity if provided
        identity = run_identity or {}
        
        return cls(
            target=target,
            method=method,
            universe_sig=universe_sig,
            run_id=run_id,
            created_at=created_at,
            features=features,
            importances=importances,
            strict_key=identity.get("strict_key"),
            replicate_key=identity.get("replicate_key"),
            feature_signature=identity.get("feature_signature"),
            split_signature=identity.get("split_signature"),
            target_signature=identity.get("target_signature"),
            hparams_signature=identity.get("hparams_signature"),
            dataset_signature=identity.get("dataset_signature"),
            routing_signature=identity.get("routing_signature"),
            train_seed=identity.get("train_seed"),
            # Prediction fingerprints
            prediction_hash=identity.get("prediction_hash"),
            prediction_hash_live=identity.get("prediction_hash_live"),
            prediction_row_ids_hash=identity.get("prediction_row_ids_hash"),
            prediction_classes_hash=identity.get("prediction_classes_hash"),
            prediction_kind=identity.get("prediction_kind"),
        )
    
    @classmethod
    def from_series(
        cls,
        target: str,
        method: str,
        importance_series,  # pd.Series with feature names as index
        universe_sig: Optional[str] = None,
        run_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        run_identity: Optional[Dict] = None,
    ) -> 'FeatureImportanceSnapshot':
        """
        Create snapshot from pandas Series.
        
        Args:
            target: Target name
            method: Method name
            importance_series: pandas Series with feature names as index
            universe_sig: Optional universe identifier
            run_id: Optional run ID
            created_at: Optional creation timestamp
            run_identity: Optional RunIdentity.to_dict() for identity signatures
        
        Returns:
            FeatureImportanceSnapshot instance
        """
        # Convert Series to dict, then use from_dict_series
        importance_dict = importance_series.to_dict()
        return cls.from_dict_series(
            target=target,
            method=method,
            importance_dict=importance_dict,
            universe_sig=universe_sig,
            run_id=run_id,
            created_at=created_at,
            run_identity=run_identity,
        )
    
    def to_dict(self) -> Dict:
        """Convert snapshot to dictionary for JSON serialization."""
        result = {
            "target": self.target,
            "method": self.method,
            "universe_sig": self.universe_sig,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "features": self.features,
            "importances": self.importances,
        }
        # Add identity fields if present
        if self.strict_key:
            result["strict_key"] = self.strict_key
        if self.replicate_key:
            result["replicate_key"] = self.replicate_key
        if self.feature_signature:
            result["feature_signature"] = self.feature_signature
        if self.split_signature:
            result["split_signature"] = self.split_signature
        if self.target_signature:
            result["target_signature"] = self.target_signature
        if self.hparams_signature:
            result["hparams_signature"] = self.hparams_signature
        if self.dataset_signature:
            result["dataset_signature"] = self.dataset_signature
        if self.routing_signature:
            result["routing_signature"] = self.routing_signature
        if self.train_seed is not None:
            result["train_seed"] = self.train_seed
        # Prediction fingerprints
        if self.prediction_hash:
            result["prediction_hash"] = self.prediction_hash
        if self.prediction_hash_live:
            result["prediction_hash_live"] = self.prediction_hash_live
        if self.prediction_row_ids_hash:
            result["prediction_row_ids_hash"] = self.prediction_row_ids_hash
        if self.prediction_classes_hash:
            result["prediction_classes_hash"] = self.prediction_classes_hash
        if self.prediction_kind:
            result["prediction_kind"] = self.prediction_kind
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FeatureImportanceSnapshot':
        """Create snapshot from dictionary (loaded from JSON)."""
        from datetime import datetime
        return cls(
            target=data["target"],
            method=data["method"],
            universe_sig=data.get("universe_sig"),
            run_id=data["run_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            features=data["features"],
            importances=data["importances"],
            # Identity fields
            strict_key=data.get("strict_key"),
            replicate_key=data.get("replicate_key"),
            feature_signature=data.get("feature_signature"),
            split_signature=data.get("split_signature"),
            target_signature=data.get("target_signature"),
            hparams_signature=data.get("hparams_signature"),
            dataset_signature=data.get("dataset_signature"),
            routing_signature=data.get("routing_signature"),
            train_seed=data.get("train_seed"),
            # Prediction fingerprints
            prediction_hash=data.get("prediction_hash"),
            prediction_hash_live=data.get("prediction_hash_live"),
            prediction_row_ids_hash=data.get("prediction_row_ids_hash"),
            prediction_classes_hash=data.get("prediction_classes_hash"),
            prediction_kind=data.get("prediction_kind"),
        )


@dataclass
class FeatureSelectionSnapshot:
    """
    Full snapshot for feature selection stage - mirrors TARGET_RANKING snapshot structure.
    
    Written to:
    - Per-cohort: targets/{target}/reproducibility/{view}/cohort=.../fs_snapshot.json
    - Global index: globals/fs_snapshot_index.json
    
    Structure mirrors snapshot_index.json entries for TARGET_RANKING but with
    stage="FEATURE_SELECTION" and includes method (model family).
    """
    # Identity
    run_id: str
    timestamp: str
    stage: str = "FEATURE_SELECTION"
    view: str = "CROSS_SECTIONAL"  # "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    target: str = ""
    symbol: Optional[str] = None  # For SYMBOL_SPECIFIC views
    method: str = ""  # Model family: "xgboost", "lightgbm", "multi_model_aggregated", etc.
    snapshot_seq: int = 0  # Sequence number for this run (matches TARGET_RANKING)
    
    # Fingerprints (for determinism verification)
    fingerprint_schema_version: str = "1.0"
    metrics_schema_version: str = "1.1"  # Bump when metrics structure changes (added 2026-01)
    scoring_schema_version: str = "1.1"  # Phase 3.1: SE-based stability, skill-gating, classification centering
    config_fingerprint: Optional[str] = None  # Full hash (includes run_id/timestamp) - for metadata
    deterministic_config_fingerprint: Optional[str] = None  # Deterministic hash (excludes run_id/timestamp) - for comparison
    data_fingerprint: Optional[str] = None
    feature_fingerprint: Optional[str] = None  # Alias for feature_fingerprint_output (selected features)
    feature_fingerprint_input: Optional[str] = None  # Candidate feature universe entering FS
    feature_fingerprint_output: Optional[str] = None  # Selected features exiting FS
    target_fingerprint: Optional[str] = None
    metrics_sha256: Optional[str] = None  # Hash of outputs.metrics for drift detection
    artifacts_manifest_sha256: Optional[str] = None  # Hash of output artifacts for tampering detection
    predictions_sha256: Optional[str] = None  # Aggregated prediction hash
    selection_signature: Optional[str] = None  # SHA256 hash of selection parameters for determinism
    
    # Selection mode (P0 correctness: clarify whether actual selection happened)
    # "rank_only" = full ranking, no selection (n_selected == n_candidates)
    # "top_k" = selected top k features by importance
    # "threshold" = selected features above importance threshold
    # "importance_cutoff" = selected features above cumulative importance cutoff
    selection_mode: str = "rank_only"
    n_candidates: int = 0  # Number of candidate features entering selection
    n_selected: int = 0  # Number of features after selection
    selection_params: Dict[str, Any] = field(default_factory=dict)  # {"k": 50} or {"threshold": 0.01}
    
    # Fingerprint source documentation (what each fingerprint means)
    fingerprint_sources: Dict[str, str] = field(default_factory=lambda: {
        "config_fingerprint": "hash of FS model hyperparameters and config sections",
        "data_fingerprint": "hash of n_samples, symbols list, and date_range (start/end)",
        "feature_fingerprint": "hash of sorted feature spec list resolved from registry",
        "feature_fingerprint_input": "hash of candidate features before selection",
        "feature_fingerprint_output": "hash of selected features after selection",
        "target_fingerprint": "hash of target name, view, horizon_minutes, and labeling_impl_hash",
        "selection_signature": "hash of selection parameters (mode, params, aggregation config) for determinism",
    })
    
    # Inputs (mirrors TARGET_RANKING)
    inputs: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "config": {"min_cs": 3, "max_cs_samples": 2000},
    #   "data": {"n_symbols": 10, "date_start": "...", "date_end": "..."},
    #   "target": {"target": "fwd_ret_10m", "view": "CROSS_SECTIONAL", "horizon_minutes": 10},
    #   "selected_targets": ["fwd_ret_10m", "fwd_ret_30m"],  # From TARGET_RANKING stage
    #   "candidate_features": ["low_vol_frac", "ret_zscore_15m", ...],  # Input feature universe
    # }
    
    # Process
    process: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "split": {"cv_method": "purged_kfold", "purge_minutes": 245.0, "split_seed": 42}
    # }
    
    # Outputs
    outputs: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "metrics": {"n_features_selected": 15, "mean_importance": 0.26},
    #   "top_features": ["low_vol_frac", "ret_zscore_15m", ...]
    # }
    
    # Comparison group (for cross-run matching) - full parity with TARGET_RANKING
    comparison_group: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "dataset_signature": "...",
    #   "split_signature": "...",
    #   "target_signature": "...",
    #   "routing_signature": "...",
    #   "hyperparameters_signature": "...",  # FS model hyperparameters
    #   "train_seed": 42,
    #   "universe_sig": "ef91e9db233a",
    #   "n_effective": 1887790,  # Effective sample count from FS
    #   "feature_registry_hash": "...",  # Hash of feature registry used
    #   "comparable_key": "..."  # Pre-computed comparison key
    # }
    
    # Path to this snapshot (for global index)
    path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "stage": self.stage,
            "view": self.view,
            "target": self.target,
            "symbol": self.symbol,
            "method": self.method,
            "snapshot_seq": self.snapshot_seq,
            "fingerprint_schema_version": self.fingerprint_schema_version,
            "metrics_schema_version": self.metrics_schema_version,
            "scoring_schema_version": self.scoring_schema_version,
            "config_fingerprint": self.config_fingerprint,
            "deterministic_config_fingerprint": self.deterministic_config_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "feature_fingerprint_input": self.feature_fingerprint_input,
            "feature_fingerprint_output": self.feature_fingerprint_output,
            "target_fingerprint": self.target_fingerprint,
            "metrics_sha256": self.metrics_sha256,
            "artifacts_manifest_sha256": self.artifacts_manifest_sha256,
            "predictions_sha256": self.predictions_sha256,
            "selection_signature": self.selection_signature,  # Determinism hash of selection parameters
            "fingerprint_sources": self.fingerprint_sources,
            "inputs": self.inputs,
            "process": self.process,
            "outputs": self.outputs,
            "comparison_group": self.comparison_group,
            "path": self.path,
            # Selection mode fields (P0 correctness)
            "selection_mode": self.selection_mode,
            "n_candidates": self.n_candidates,
            "n_selected": self.n_selected,
            "selection_params": self.selection_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSelectionSnapshot':
        """Create from dictionary (loaded from JSON)."""
        return cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            stage=data.get("stage", "FEATURE_SELECTION"),
            view=data.get("view", "CROSS_SECTIONAL"),
            target=data.get("target", ""),
            symbol=data.get("symbol"),
            method=data.get("method", ""),
            snapshot_seq=data.get("snapshot_seq", 0),
            fingerprint_schema_version=data.get("fingerprint_schema_version", "1.0"),
            metrics_schema_version=data.get("metrics_schema_version", "1.0"),  # Default to 1.0 for old snapshots
            scoring_schema_version=data.get("scoring_schema_version", "1.0"),
            config_fingerprint=data.get("config_fingerprint"),
            deterministic_config_fingerprint=data.get("deterministic_config_fingerprint"),
            data_fingerprint=data.get("data_fingerprint"),
            feature_fingerprint=data.get("feature_fingerprint"),
            feature_fingerprint_input=data.get("feature_fingerprint_input"),
            feature_fingerprint_output=data.get("feature_fingerprint_output"),
            target_fingerprint=data.get("target_fingerprint"),
            metrics_sha256=data.get("metrics_sha256"),
            artifacts_manifest_sha256=data.get("artifacts_manifest_sha256"),
            predictions_sha256=data.get("predictions_sha256"),
            selection_signature=data.get("selection_signature"),  # Determinism hash of selection parameters
            fingerprint_sources=data.get("fingerprint_sources", {}),
            inputs=data.get("inputs", {}),
            process=data.get("process", {}),
            outputs=data.get("outputs", {}),
            comparison_group=data.get("comparison_group", {}),
            path=data.get("path"),
            # Selection mode fields (P0 correctness) - defaults for backward compat
            selection_mode=data.get("selection_mode", "rank_only"),
            n_candidates=data.get("n_candidates", 0),
            n_selected=data.get("n_selected", 0),
            selection_params=data.get("selection_params", {}),
        )
    
    @classmethod
    def from_importance_snapshot(
        cls,
        importance_snapshot: FeatureImportanceSnapshot,
        view: str = "CROSS_SECTIONAL",
        symbol: Optional[str] = None,
        inputs: Optional[Dict[str, Any]] = None,
        process: Optional[Dict[str, Any]] = None,
        outputs: Optional[Dict[str, Any]] = None,
        stage: str = "FEATURE_SELECTION",  # Allow caller to specify stage
        snapshot_seq: int = 0,  # Sequence number for this run
        n_effective: Optional[int] = None,  # Effective sample count from FS
        feature_registry_hash: Optional[str] = None,  # Hash of feature registry
        comparable_key: Optional[str] = None,  # Pre-computed comparison key
        output_dir: Optional[Path] = None,  # Output directory for loading config.resolved.json
        # P0 correctness: selection mode fields
        selection_mode: Optional[str] = None,  # "rank_only" | "top_k" | "threshold" | "importance_cutoff"
        n_candidates: Optional[int] = None,  # Number of candidate features entering selection
        n_selected: Optional[int] = None,  # Number of features after selection
        selection_params: Optional[Dict[str, Any]] = None,  # {"k": 50} or {"threshold": 0.01}
    ) -> 'FeatureSelectionSnapshot':
        """
        Create from a FeatureImportanceSnapshot with additional context.
        
        This bridges the existing snapshot system with the new full structure.
        
        Args:
            stage: Pipeline stage - "FEATURE_SELECTION" (default) or "TARGET_RANKING"
            snapshot_seq: Sequence number for this run (matches TARGET_RANKING)
            n_effective: Effective sample count from FS stage
            feature_registry_hash: Hash of feature registry used
            comparable_key: Pre-computed comparison key from cohort metadata
            selection_mode: P0 - How selection was performed ("rank_only", "top_k", etc.)
            n_candidates: P0 - Number of candidate features before selection
            n_selected: P0 - Number of features after selection
            selection_params: P0 - Parameters for selection (e.g., {"k": 50})
        """
        import hashlib
        import json
        
        # Build comparison group from importance snapshot signatures (full parity with TR)
        comparison_group = {}
        # experiment_id (if available from identity/context)
        if hasattr(importance_snapshot, 'experiment_id') and importance_snapshot.experiment_id:
            comparison_group["experiment_id"] = importance_snapshot.experiment_id
        if importance_snapshot.dataset_signature:
            comparison_group["dataset_signature"] = importance_snapshot.dataset_signature
        if importance_snapshot.split_signature:
            comparison_group["split_signature"] = importance_snapshot.split_signature
        if importance_snapshot.target_signature:
            comparison_group["target_signature"] = importance_snapshot.target_signature
            comparison_group["task_signature"] = importance_snapshot.target_signature  # Alias for parity
        if importance_snapshot.routing_signature:
            comparison_group["routing_signature"] = importance_snapshot.routing_signature
        if importance_snapshot.hparams_signature:
            comparison_group["hyperparameters_signature"] = importance_snapshot.hparams_signature
        if importance_snapshot.train_seed is not None:
            comparison_group["train_seed"] = importance_snapshot.train_seed
        if importance_snapshot.universe_sig:
            comparison_group["universe_sig"] = importance_snapshot.universe_sig
        if importance_snapshot.feature_signature:
            comparison_group["feature_signature"] = importance_snapshot.feature_signature
        # library_versions_signature (if available from identity/context)
        if hasattr(importance_snapshot, 'library_versions_signature') and importance_snapshot.library_versions_signature:
            comparison_group["library_versions_signature"] = importance_snapshot.library_versions_signature
        # Additional fields for full parity
        if n_effective is not None:
            comparison_group["n_effective"] = n_effective
        if feature_registry_hash:
            comparison_group["feature_registry_hash"] = feature_registry_hash
        if comparable_key:
            comparison_group["comparable_key"] = comparable_key
        
        # Build outputs from importance data
        default_outputs = {
            "metrics": {
                "n_features": len(importance_snapshot.features),
                "mean_importance": sum(importance_snapshot.importances) / len(importance_snapshot.importances) if importance_snapshot.importances else 0,
            },
            "top_features": importance_snapshot.features[:10],  # Top 10 features
        }
        final_outputs = outputs or default_outputs
        
        # Compute metrics_sha256 for drift detection
        metrics_sha256 = None
        if final_outputs.get("metrics"):
            try:
                metrics_json = json.dumps(final_outputs["metrics"], sort_keys=True)
                metrics_sha256 = hashlib.sha256(metrics_json.encode()).hexdigest()
            except Exception:
                pass
        
        # Extract feature_fingerprint_input from inputs if provided
        feature_input_hash = inputs.get("feature_fingerprint_input") if inputs else None
        
        # CRITICAL: Extract config fingerprints from inputs or load from config.resolved.json
        # This ensures deterministic_config_fingerprint is populated for run hash computation
        config_fp = importance_snapshot.hparams_signature  # Model config hash (fallback)
        deterministic_config_fp = None
        
        # Try to get both fingerprints from inputs (passed from diff_telemetry or reproducibility_tracker)
        if inputs:
            deterministic_config_fp = inputs.get("deterministic_config_fingerprint")
            if inputs.get("config_fingerprint") and not config_fp:
                config_fp = inputs.get("config_fingerprint")
        
        # If deterministic fingerprint not in inputs, try to load from config.resolved.json
        # FIX: Walk up to find run root (output_dir may be target_repro_dir, not run root)
        if not deterministic_config_fp and output_dir:
            try:
                import json
                # Walk up to find run root with globals/ directory
                from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                base_dir = get_run_root(Path(output_dir))
                globals_dir = base_dir / "globals" if (base_dir / "globals").exists() else None
                
                if globals_dir:
                    resolved_config_path = globals_dir / "config.resolved.json"
                    if resolved_config_path.exists():
                        with open(resolved_config_path, 'r') as f:
                            resolved_config = json.load(f)
                        deterministic_config_fp = resolved_config.get('deterministic_config_fingerprint')
                        if not config_fp:
                            config_fp = resolved_config.get('config_fingerprint')
            except Exception as e:
                # Log at debug level - this is a fallback path
                import logging
                logger = logging.getLogger(__name__)
                logger.debug(f"Failed to load deterministic_config_fingerprint from config.resolved.json: {e}")
                pass  # Fallback to hparams_signature only
        
        # P0 correctness: Determine selection mode if not explicitly provided
        # Infer n_candidates from inputs.candidate_features if available
        actual_n_candidates = n_candidates
        if actual_n_candidates is None and inputs:
            candidate_features = inputs.get("candidate_features", [])
            if candidate_features:
                actual_n_candidates = len(candidate_features)
        if actual_n_candidates is None:
            actual_n_candidates = 0
        
        # Infer n_selected from importance_snapshot.features
        actual_n_selected = n_selected
        if actual_n_selected is None:
            actual_n_selected = len(importance_snapshot.features)
        
        # Infer selection_mode if not explicitly provided
        actual_selection_mode = selection_mode
        if actual_selection_mode is None:
            # If n_selected == n_candidates (or n_candidates not known), it's rank_only
            if actual_n_candidates == 0 or actual_n_selected == actual_n_candidates:
                actual_selection_mode = "rank_only"
            elif selection_params and "k" in selection_params:
                actual_selection_mode = "top_k"
            elif selection_params and "threshold" in selection_params:
                actual_selection_mode = "threshold"
            else:
                # Default: assume some selection happened if n_selected < n_candidates
                actual_selection_mode = "top_k" if actual_n_selected < actual_n_candidates else "rank_only"
        
        # Compute selection_signature (hash of selection parameters for determinism)
        # Similar to scoring_signature for TARGET_RANKING composite scoring
        selection_signature = None
        try:
            # Extract aggregation config from inputs if available
            aggregation_config = {}
            if inputs:
                # Try to get aggregation config from inputs.config or inputs.aggregation
                config_section = inputs.get("config", {})
                if "aggregation" in config_section:
                    aggregation_config = config_section["aggregation"]
                elif "aggregation" in inputs:
                    aggregation_config = inputs["aggregation"]
            
            # Build selection params dict (all parameters that affect selection outcome)
            selection_params_dict = {
                "selection_mode": actual_selection_mode,
                "selection_params": selection_params or {},
                "aggregation": aggregation_config,  # Include aggregation method/config if available
                "version": "1.0",  # Version for future schema changes
            }
            # Canonical JSON (sorted keys) for deterministic hashing
            selection_params_json = json.dumps(selection_params_dict, sort_keys=True, separators=(',', ':'))
            selection_signature = hashlib.sha256(selection_params_json.encode()).hexdigest()
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Failed to compute selection_signature: {e}")
        
        return cls(
            run_id=importance_snapshot.run_id,
            timestamp=importance_snapshot.created_at.isoformat(),
            stage=stage,  # Use caller-provided stage
            view=view,
            target=importance_snapshot.target,
            symbol=symbol,
            method=importance_snapshot.method,
            snapshot_seq=snapshot_seq,
            # Fingerprint mappings from FeatureImportanceSnapshot
            config_fingerprint=config_fp,  # Full config fingerprint (includes run_id/timestamp) - for metadata
            deterministic_config_fingerprint=deterministic_config_fp,  # Deterministic fingerprint (excludes run_id/timestamp) - for comparison
            data_fingerprint=importance_snapshot.dataset_signature,  # Dataset signature
            feature_fingerprint=importance_snapshot.feature_signature,  # Feature set signature
            feature_fingerprint_input=feature_input_hash,  # Candidate features before selection
            feature_fingerprint_output=importance_snapshot.feature_signature,  # Selected features
            target_fingerprint=importance_snapshot.target_signature,  # Target definition hash
            metrics_sha256=metrics_sha256,  # Hash of outputs.metrics
            predictions_sha256=importance_snapshot.prediction_hash,  # Prediction determinism hash
            artifacts_manifest_sha256=None,  # Feature selection typically doesn't produce model artifacts (no saved models)
            inputs=inputs or {},
            process=process or {},
            outputs=final_outputs,
            comparison_group=comparison_group,
            # P0 correctness: selection mode fields
            selection_mode=actual_selection_mode,
            n_candidates=actual_n_candidates,
            n_selected=actual_n_selected,
            selection_params=selection_params or {},
            selection_signature=selection_signature,  # Determinism hash of selection parameters
        )
    
    def get_index_key(self) -> str:
        """
        Generate index key for globals/fs_snapshot_index.json.
        
        Format: {timestamp}:{stage}:{target}:{view}:{method}:{symbol_or_NONE}
        """
        symbol_part = self.symbol if self.symbol else "NONE"
        return f"{self.timestamp}:{self.stage}:{self.target}:{self.view}:{self.method}:{symbol_part}"
