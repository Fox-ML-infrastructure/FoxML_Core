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
    """
    target: str
    method: str                      # "quick_pruner", "rfe", "boruta", "lightgbm", etc.
    universe_sig: Optional[str]       # "TOP100", "ALL", "MEGA_CAP", symbol name, or None
    run_id: str                      # UUID or timestamp-based identifier
    created_at: datetime
    features: List[str]              # Feature names (same order as importances)
    importances: List[float]         # Importance values (same order as features)
    
    @classmethod
    def from_dict_series(
        cls,
        target: str,
        method: str,
        importance_dict: Dict[str, float],
        universe_sig: Optional[str] = None,
        run_id: Optional[str] = None,
        created_at: Optional[datetime] = None
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
        
        return cls(
            target=target,
            method=method,
            universe_sig=universe_sig,
            run_id=run_id,
            created_at=created_at,
            features=features,
            importances=importances
        )
    
    @classmethod
    def from_series(
        cls,
        target: str,
        method: str,
        importance_series,  # pd.Series with feature names as index
        universe_sig: Optional[str] = None,
        run_id: Optional[str] = None,
        created_at: Optional[datetime] = None
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
            created_at=created_at
        )
    
    def to_dict(self) -> Dict:
        """Convert snapshot to dictionary for JSON serialization."""
        return {
            "target": self.target,
            "method": self.method,
            "universe_sig": self.universe_sig,
            "run_id": self.run_id,
            "created_at": self.created_at.isoformat(),
            "features": self.features,
            "importances": self.importances,
        }
    
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
        )
