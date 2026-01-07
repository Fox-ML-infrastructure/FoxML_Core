# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Feature Importance Snapshot Schema

Standardized format for storing feature importance snapshots for stability analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any
from datetime import datetime
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
    
    # Fingerprints (for determinism verification)
    fingerprint_schema_version: str = "1.0"
    config_fingerprint: Optional[str] = None
    data_fingerprint: Optional[str] = None
    feature_fingerprint: Optional[str] = None  # Alias for feature_fingerprint_output (selected features)
    feature_fingerprint_input: Optional[str] = None  # Candidate feature universe entering FS
    feature_fingerprint_output: Optional[str] = None  # Selected features exiting FS
    target_fingerprint: Optional[str] = None
    predictions_sha256: Optional[str] = None  # Aggregated prediction hash
    
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
    
    # Comparison group (for cross-run matching)
    comparison_group: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "dataset_signature": "...",
    #   "split_signature": "...",
    #   "train_seed": 42,
    #   "universe_sig": "ef91e9db233a"
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
            "fingerprint_schema_version": self.fingerprint_schema_version,
            "config_fingerprint": self.config_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "feature_fingerprint_input": self.feature_fingerprint_input,
            "feature_fingerprint_output": self.feature_fingerprint_output,
            "target_fingerprint": self.target_fingerprint,
            "predictions_sha256": self.predictions_sha256,
            "inputs": self.inputs,
            "process": self.process,
            "outputs": self.outputs,
            "comparison_group": self.comparison_group,
            "path": self.path,
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
            fingerprint_schema_version=data.get("fingerprint_schema_version", "1.0"),
            config_fingerprint=data.get("config_fingerprint"),
            data_fingerprint=data.get("data_fingerprint"),
            feature_fingerprint=data.get("feature_fingerprint"),
            feature_fingerprint_input=data.get("feature_fingerprint_input"),
            feature_fingerprint_output=data.get("feature_fingerprint_output"),
            target_fingerprint=data.get("target_fingerprint"),
            predictions_sha256=data.get("predictions_sha256"),
            inputs=data.get("inputs", {}),
            process=data.get("process", {}),
            outputs=data.get("outputs", {}),
            comparison_group=data.get("comparison_group", {}),
            path=data.get("path"),
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
    ) -> 'FeatureSelectionSnapshot':
        """
        Create from a FeatureImportanceSnapshot with additional context.
        
        This bridges the existing snapshot system with the new full structure.
        
        Args:
            stage: Pipeline stage - "FEATURE_SELECTION" (default) or "TARGET_RANKING"
        """
        # Build comparison group from importance snapshot signatures
        comparison_group = {}
        if importance_snapshot.dataset_signature:
            comparison_group["dataset_signature"] = importance_snapshot.dataset_signature
        if importance_snapshot.split_signature:
            comparison_group["split_signature"] = importance_snapshot.split_signature
        if importance_snapshot.target_signature:
            comparison_group["target_signature"] = importance_snapshot.target_signature
        if importance_snapshot.routing_signature:
            comparison_group["routing_signature"] = importance_snapshot.routing_signature
        if importance_snapshot.train_seed is not None:
            comparison_group["train_seed"] = importance_snapshot.train_seed
        if importance_snapshot.universe_sig:
            comparison_group["universe_sig"] = importance_snapshot.universe_sig
        
        # Build outputs from importance data
        default_outputs = {
            "metrics": {
                "n_features": len(importance_snapshot.features),
                "mean_importance": sum(importance_snapshot.importances) / len(importance_snapshot.importances) if importance_snapshot.importances else 0,
            },
            "top_features": importance_snapshot.features[:10],  # Top 10 features
        }
        
        return cls(
            run_id=importance_snapshot.run_id,
            timestamp=importance_snapshot.created_at.isoformat(),
            stage=stage,  # Use caller-provided stage
            view=view,
            target=importance_snapshot.target,
            symbol=symbol,
            method=importance_snapshot.method,
            # Fingerprint mappings from FeatureImportanceSnapshot
            config_fingerprint=importance_snapshot.hparams_signature,  # Model config hash
            data_fingerprint=importance_snapshot.dataset_signature,  # Dataset signature
            feature_fingerprint=importance_snapshot.feature_signature,  # Feature set signature
            feature_fingerprint_output=importance_snapshot.feature_signature,  # Selected features
            target_fingerprint=importance_snapshot.target_signature,  # Target definition hash
            predictions_sha256=importance_snapshot.prediction_hash,  # Prediction determinism hash
            inputs=inputs or {},
            process=process or {},
            outputs=outputs or default_outputs,
            comparison_group=comparison_group,
        )
    
    def get_index_key(self) -> str:
        """
        Generate index key for globals/fs_snapshot_index.json.
        
        Format: {timestamp}:{stage}:{target}:{view}:{method}:{symbol_or_NONE}
        """
        symbol_part = self.symbol if self.symbol else "NONE"
        return f"{self.timestamp}:{self.stage}:{self.target}:{self.view}:{self.method}:{symbol_part}"
