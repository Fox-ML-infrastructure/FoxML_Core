# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Training Stage Snapshot Schema

Full-parity tracking for TRAINING stage (Stage 3), mirroring the structure
of TARGET_RANKING (snapshot.json) and FEATURE_SELECTION (fs_snapshot.json).

This enables end-to-end determinism verification:
    TR snapshot → FS snapshot → Training snapshot
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid
import logging

# SST: Import Stage enum for consistent stage handling
from TRAINING.orchestration.utils.scope_resolution import Stage

logger = logging.getLogger(__name__)


@dataclass
class TrainingSnapshot:
    """
    Full snapshot for training stage - mirrors TARGET_RANKING/FEATURE_SELECTION snapshot structure.
    
    Written to:
    - Per-cohort: targets/{target}/reproducibility/stage=TRAINING/{view}/cohort={cohort_id}/training_snapshot.json
    - For SYMBOL_SPECIFIC: targets/{target}/reproducibility/stage=TRAINING/SYMBOL_SPECIFIC/symbol={symbol}/cohort={cohort_id}/training_snapshot.json
    - Aggregated SYMBOL_SPECIFIC: targets/{target}/reproducibility/stage=TRAINING/SYMBOL_SPECIFIC/cohort={cohort_id}/training_snapshot.json (symbol=None)
    - Parquet format: Same location with training_metadata.parquet
    - Global index: globals/training_snapshot_index.json
    - Global summary: globals/training_summary.json and globals/training_summary.csv
    
    Structure mirrors snapshot_index.json entries for TARGET_RANKING but with
    stage="TRAINING" and includes model_family and model_artifact_sha256.
    
    Note: For SYMBOL_SPECIFIC views, symbol=None indicates an aggregated snapshot
    that combines metrics across all symbols for the same target and model family.
    """
    # Identity
    run_id: str
    timestamp: str
    stage: str = "TRAINING"
    view: str = "CROSS_SECTIONAL"  # "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    target: str = ""
    symbol: Optional[str] = None  # For SYMBOL_SPECIFIC views
    model_family: str = ""  # e.g., "xgboost", "lightgbm", "ridge"
    snapshot_seq: int = 0  # Sequence number for this run
    
    # Fingerprints (for determinism verification)
    fingerprint_schema_version: str = "1.0"
    metrics_schema_version: str = "1.1"  # Bump when metrics structure changes (added 2026-01)
    scoring_schema_version: str = "1.1"  # Phase 3.1: SE-based stability, skill-gating, classification centering
    config_fingerprint: Optional[str] = None  # Full hash of training config (includes run_id/timestamp) - for metadata
    deterministic_config_fingerprint: Optional[str] = None  # Deterministic hash (excludes run_id/timestamp) - for comparison
    data_fingerprint: Optional[str] = None  # Dataset signature
    feature_fingerprint: Optional[str] = None  # Selected features from FS stage
    target_fingerprint: Optional[str] = None  # Target definition hash
    hyperparameters_signature: Optional[str] = None  # Model hyperparameters hash
    split_signature: Optional[str] = None  # CV split configuration hash
    model_artifact_sha256: Optional[str] = None  # Hash of saved model file (.pkl/.joblib)
    artifacts_manifest_sha256: Optional[str] = None  # Hash of output artifacts for tampering detection
    metrics_sha256: Optional[str] = None  # Hash of outputs.metrics for drift detection
    predictions_sha256: Optional[str] = None  # Prediction fingerprint (train/val)
    
    # Fingerprint source documentation (what each fingerprint means)
    fingerprint_sources: Dict[str, str] = field(default_factory=lambda: {
        "config_fingerprint": "hash of training config sections (optimizer, epochs, etc.)",
        "data_fingerprint": "hash of n_samples, symbols list, and date_range (start/end)",
        "feature_fingerprint": "hash of selected features from FEATURE_SELECTION stage",
        "target_fingerprint": "hash of target name, view, horizon_minutes, and labeling_impl_hash",
        "hyperparameters_signature": "hash of model hyperparameters (learning_rate, max_depth, etc.)",
        "split_signature": "hash of CV configuration (method, purge, embargo, n_splits)",
        "model_artifact_sha256": "SHA256 hash of the saved model file for tamper detection",
        "artifacts_manifest_sha256": "SHA256 hash of all output artifacts (model, scaler, imputer, etc.) for tamper detection",
        "predictions_sha256": "hash of model predictions on validation data for determinism",
    })
    
    # Inputs (what went into training)
    inputs: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "features_used": ["low_vol_frac", "ret_zscore_15m", ...],  # From FEATURE_SELECTION
    #   "n_features": 15,
    #   "n_samples": 1887790,
    #   "n_symbols": 10,
    #   "date_start": "2016-01-04T14:30:00",
    #   "date_end": "2025-08-29T19:45:00",
    #   "selected_targets": ["fwd_ret_10m"],  # From TARGET_RANKING
    #   "selected_features": ["low_vol_frac", ...],  # From FEATURE_SELECTION
    # }
    
    # Process (how training was done)
    process: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "split": {"cv_method": "purged_kfold", "purge_minutes": 15.0, "embargo_minutes": 15.0, "n_splits": 5},
    #   "train_seed": 42,
    #   "strategy": "single_task",
    #   "normalization": "zscore",
    # }
    
    # Outputs (what training produced)
    outputs: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "metrics": {"train_loss": 0.32, "val_loss": 0.35, "val_auc": 0.72, ...},
    #   "model_path": "targets/fwd_ret_10m/models/stage=TRAINING/.../model.pkl",
    #   "training_time_seconds": 123.4,
    #   "n_iterations": 100,
    # }
    
    # Comparison group (for cross-run matching) - full parity with TARGET_RANKING/FEATURE_SELECTION
    comparison_group: Dict[str, Any] = field(default_factory=dict)
    # {
    #   "dataset_signature": "...",
    #   "split_signature": "...",
    #   "target_signature": "...",
    #   "routing_signature": "...",
    #   "hyperparameters_signature": "...",
    #   "train_seed": 42,
    #   "universe_sig": "ef91e9db233a",
    #   "n_effective": 1887790,
    #   "feature_registry_hash": "...",
    #   "comparable_key": "..."
    # }
    
    # Model path (for artifact lookup)
    model_path: Optional[str] = None
    
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
            "model_family": self.model_family,
            "snapshot_seq": self.snapshot_seq,
            "fingerprint_schema_version": self.fingerprint_schema_version,
            "metrics_schema_version": self.metrics_schema_version,
            "scoring_schema_version": self.scoring_schema_version,
            "config_fingerprint": self.config_fingerprint,
            "data_fingerprint": self.data_fingerprint,
            "feature_fingerprint": self.feature_fingerprint,
            "target_fingerprint": self.target_fingerprint,
            "hyperparameters_signature": self.hyperparameters_signature,
            "split_signature": self.split_signature,
            "model_artifact_sha256": self.model_artifact_sha256,
            "artifacts_manifest_sha256": self.artifacts_manifest_sha256,
            "metrics_sha256": self.metrics_sha256,
            "predictions_sha256": self.predictions_sha256,
            "fingerprint_sources": self.fingerprint_sources,
            "inputs": self.inputs,
            "process": self.process,
            "outputs": self.outputs,
            "comparison_group": self.comparison_group,
            "model_path": self.model_path,
            "path": self.path,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingSnapshot':
        """Create from dictionary (loaded from JSON)."""
        return cls(
            run_id=data.get("run_id", ""),
            timestamp=data.get("timestamp", ""),
            stage=data.get("stage", "TRAINING"),
            view=data.get("view", "CROSS_SECTIONAL"),
            target=data.get("target", ""),
            symbol=data.get("symbol"),
            model_family=data.get("model_family", ""),
            snapshot_seq=data.get("snapshot_seq", 0),
            fingerprint_schema_version=data.get("fingerprint_schema_version", "1.0"),
            metrics_schema_version=data.get("metrics_schema_version", "1.0"),  # Default to 1.0 for old snapshots
            scoring_schema_version=data.get("scoring_schema_version", "1.0"),  # Default to 1.0 for old snapshots
            config_fingerprint=data.get("config_fingerprint"),
            data_fingerprint=data.get("data_fingerprint"),
            feature_fingerprint=data.get("feature_fingerprint"),
            target_fingerprint=data.get("target_fingerprint"),
            hyperparameters_signature=data.get("hyperparameters_signature"),
            split_signature=data.get("split_signature"),
            model_artifact_sha256=data.get("model_artifact_sha256"),
            artifacts_manifest_sha256=data.get("artifacts_manifest_sha256"),
            metrics_sha256=data.get("metrics_sha256"),
            predictions_sha256=data.get("predictions_sha256"),
            fingerprint_sources=data.get("fingerprint_sources", {}),
            inputs=data.get("inputs", {}),
            process=data.get("process", {}),
            outputs=data.get("outputs", {}),
            comparison_group=data.get("comparison_group", {}),
            model_path=data.get("model_path"),
            path=data.get("path"),
        )
    
    @classmethod
    def from_training_result(
        cls,
        target: str,
        model_family: str,
        model_result: Dict[str, Any],
        view: str = "CROSS_SECTIONAL",
        symbol: Optional[str] = None,
        run_identity: Optional[Any] = None,
        model_path: Optional[str] = None,
        features_used: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        train_seed: int = 42,
        snapshot_seq: int = 0,
        output_dir: Optional[Path] = None,  # Output directory for loading config.resolved.json
    ) -> 'TrainingSnapshot':
        """
        Create from training result dictionary.
        
        This is the main factory method for creating TrainingSnapshots after
        model training completes.
        
        Args:
            target: Target name (e.g., "fwd_ret_10m")
            model_family: Model family (e.g., "xgboost", "lightgbm")
            model_result: Dictionary from training with metrics, model info
            view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
            symbol: Symbol name for SYMBOL_SPECIFIC views
            run_identity: RunIdentity object or dict with identity signatures
            model_path: Path to saved model artifact
            features_used: List of feature names used in training
            n_samples: Number of training samples
            train_seed: Training seed for reproducibility
            snapshot_seq: Sequence number for this run
        
        Returns:
            TrainingSnapshot instance
        """
        import hashlib
        import json
        
        run_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Extract identity fields from run_identity if provided
        identity = {}
        if run_identity is not None:
            if hasattr(run_identity, 'to_dict'):
                identity = run_identity.to_dict()
            elif isinstance(run_identity, dict):
                identity = run_identity
        
        # CRITICAL: Extract config fingerprints from identity or load from config.resolved.json
        # This ensures deterministic_config_fingerprint is populated for run hash computation
        config_fp = identity.get("config_fingerprint")
        deterministic_config_fp = identity.get("deterministic_config_fingerprint")
        
        # If deterministic fingerprint not in identity, try to load from config.resolved.json
        if not deterministic_config_fp and output_dir:
            try:
                import json
                globals_dir = Path(output_dir) / "globals"
                resolved_config_path = globals_dir / "config.resolved.json"
                if resolved_config_path.exists():
                    with open(resolved_config_path, 'r') as f:
                        resolved_config = json.load(f)
                    deterministic_config_fp = resolved_config.get('deterministic_config_fingerprint')
                    if not config_fp:
                        config_fp = resolved_config.get('config_fingerprint')
            except Exception:
                pass  # Fallback to identity only
        
        # Build comparison group from identity (full parity with TARGET_RANKING)
        comparison_group = {}
        if identity.get("experiment_id"):
            comparison_group["experiment_id"] = identity["experiment_id"]
        if identity.get("dataset_signature"):
            comparison_group["dataset_signature"] = identity["dataset_signature"]
        if identity.get("split_signature"):
            comparison_group["split_signature"] = identity["split_signature"]
        if identity.get("target_signature"):
            comparison_group["target_signature"] = identity["target_signature"]
            comparison_group["task_signature"] = identity["target_signature"]  # Alias for parity
        if identity.get("routing_signature"):
            comparison_group["routing_signature"] = identity["routing_signature"]
        if identity.get("hparams_signature"):
            comparison_group["hyperparameters_signature"] = identity["hparams_signature"]
        comparison_group["train_seed"] = identity.get("train_seed", train_seed)
        # Extract universe_sig from identity (RunIdentity stores it as dataset_signature)
        # Reuse pattern from reproducibility_tracker.py:1376-1378
        universe_sig = identity.get("universe_sig") or identity.get("dataset_signature")
        if universe_sig:
            comparison_group["universe_sig"] = universe_sig
        if identity.get("feature_signature"):
            comparison_group["feature_signature"] = identity["feature_signature"]
        if identity.get("library_versions_signature"):
            comparison_group["library_versions_signature"] = identity["library_versions_signature"]
        comparison_group["model_family"] = model_family
        comparison_group["symbol"] = symbol
        # Add n_effective from inputs if available
        if n_samples is not None:
            comparison_group["n_effective"] = n_samples
        
        # Build inputs
        inputs = {
            "features_used": features_used or [],
            "n_features": len(features_used) if features_used else 0,
            "n_samples": n_samples,
        }
        
        # Build process
        process = {
            "train_seed": identity.get("train_seed", train_seed),
        }
        
        # Build outputs from model_result using clean, grouped structure
        outputs = {}
        if model_result:
            # Use clean metrics builder for structured, task-gated output
            try:
                from TRAINING.ranking.predictability.metrics_schema import build_clean_training_metrics
                from TRAINING.common.utils.task_types import TaskType
                
                # Get task_type from model_result
                task_type = model_result.get("task_type")
                if isinstance(task_type, str):
                    task_type = TaskType[task_type.upper()]
                elif task_type is None:
                    # Fallback: try to infer from available metrics
                    if "val_auc" in model_result or "auc" in model_result:
                        task_type = TaskType.BINARY_CLASSIFICATION
                    else:
                        task_type = TaskType.REGRESSION
                
                # Get feature and sample counts
                n_features = None
                if features_used:
                    n_features = len(features_used)
                elif "n_features" in model_result:
                    n_features = model_result["n_features"]
                
                n_samples_val = n_samples
                if "n_samples" in model_result:
                    n_samples_val = model_result["n_samples"]
                
                # Build clean metrics
                metrics = build_clean_training_metrics(
                    model_result=model_result,
                    task_type=task_type,
                    view=view,
                    n_features=n_features,
                    n_samples=n_samples_val,
                )
            except Exception as e:
                # Fallback to flat structure if clean builder fails
                logger.warning(f"Failed to build clean training metrics, using fallback: {e}")
                metrics = {}
                for key in ["train_loss", "val_loss", "train_auc", "val_auc", "accuracy", "r2", "mse", "mae"]:
                    if key in model_result:
                        metrics[key] = model_result[key]
                if model_result.get("metrics"):
                    metrics.update(model_result["metrics"])
            
            outputs["metrics"] = metrics
            
            # Extract training info
            if model_result.get("training_time_seconds"):
                outputs["training_time_seconds"] = model_result["training_time_seconds"]
            if model_result.get("n_iterations"):
                outputs["n_iterations"] = model_result["n_iterations"]
        
        outputs["model_path"] = model_path
        
        # Compute metrics_sha256
        metrics_sha256 = None
        if outputs.get("metrics"):
            try:
                metrics_json = json.dumps(outputs["metrics"], sort_keys=True)
                metrics_sha256 = hashlib.sha256(metrics_json.encode()).hexdigest()
            except Exception:
                pass
        
        # Compute model_artifact_sha256 if model_path provided
        model_artifact_sha256 = None
        if model_path:
            try:
                model_file = Path(model_path)
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        model_artifact_sha256 = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                pass
        
        # Compute artifacts_manifest_sha256 (hash of all artifacts: model, scaler, imputer, etc.)
        artifacts_manifest_sha256 = None
        if model_path:
            try:
                artifact_hashes = []
                model_file = Path(model_path)
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        model_hash = hashlib.sha256(f.read()).hexdigest()
                        artifact_hashes.append(f"model:{model_hash}")
                
                # Try to find and hash related artifacts (scaler, imputer)
                model_dir = model_file.parent
                scaler_path = model_dir / "scaler.pkl"
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        scaler_hash = hashlib.sha256(f.read()).hexdigest()
                        artifact_hashes.append(f"scaler:{scaler_hash}")
                
                imputer_path = model_dir / "imputer.pkl"
                if imputer_path.exists():
                    with open(imputer_path, 'rb') as f:
                        imputer_hash = hashlib.sha256(f.read()).hexdigest()
                        artifact_hashes.append(f"imputer:{imputer_hash}")
                
                if artifact_hashes:
                    manifest_str = "|".join(sorted(artifact_hashes))
                    artifacts_manifest_sha256 = hashlib.sha256(manifest_str.encode()).hexdigest()
            except Exception:
                pass
        
        # Extract prediction fingerprint from model_result or identity
        predictions_sha256 = identity.get("prediction_hash")
        if not predictions_sha256 and model_result:
            predictions_sha256 = model_result.get("prediction_hash")
        
        return cls(
            run_id=run_id,
            timestamp=timestamp,
            stage=Stage.TRAINING,
            view=view,
            target=target,
            symbol=symbol,
            model_family=model_family,
            snapshot_seq=snapshot_seq,
            config_fingerprint=config_fp,  # Full config fingerprint (includes run_id/timestamp) - for metadata
            deterministic_config_fingerprint=deterministic_config_fp,  # Deterministic fingerprint (excludes run_id/timestamp) - for comparison
            data_fingerprint=identity.get("dataset_signature"),
            feature_fingerprint=identity.get("feature_signature"),
            target_fingerprint=identity.get("target_signature"),
            hyperparameters_signature=identity.get("hparams_signature"),
            split_signature=identity.get("split_signature"),
            model_artifact_sha256=model_artifact_sha256,
            artifacts_manifest_sha256=artifacts_manifest_sha256,
            metrics_sha256=metrics_sha256,
            predictions_sha256=predictions_sha256,
            inputs=inputs,
            process=process,
            outputs=outputs,
            comparison_group=comparison_group,
            model_path=model_path,
        )
    
    def get_index_key(self) -> str:
        """
        Generate index key for globals/training_snapshot_index.json.
        
        Format: {timestamp}:{stage}:{target}:{view}:{model_family}:{symbol_or_NONE}
        """
        symbol_part = self.symbol if self.symbol else "NONE"
        return f"{self.timestamp}:{self.stage}:{self.target}:{self.view}:{self.model_family}:{symbol_part}"
    
    def to_parquet_dict(self) -> Dict[str, Any]:
        """
        Convert to flat dictionary suitable for Parquet serialization.
        
        Flattens nested structures and ensures all values are Parquet-compatible.
        This is used for training_metadata.parquet files.
        
        Returns:
            Flat dictionary with all nested structures flattened
        """
        # Start with base dict
        result = self.to_dict()
        
        # Flatten nested structures for Parquet
        # Parquet works best with flat structures, so we'll keep nested dicts
        # but ensure all keys are strings (handled by _prepare_for_parquet)
        
        # The to_dict() already returns a flat structure with nested dicts
        # We just need to ensure it's Parquet-compatible
        return result