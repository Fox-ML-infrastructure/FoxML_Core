# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Feature Importance Snapshot Schema

Standardized format for storing feature importance snapshots for stability analysis.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
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
        )
