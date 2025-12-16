"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
FeatureSet Artifact (Phase 1 + Phase 2)

Single canonical artifact per stage for debugging and reproducibility.
Phase 1: Persist to disk for debugging
Phase 2: Pass through pipeline to eliminate recomputation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FeatureSetArtifact:
    """
    FeatureSet artifact for persistence, debugging, and pipeline integration.
    
    Phase 1: Persist to disk for debugging
    Phase 2: Pass through pipeline to eliminate recomputation
    """
    features: List[str]  # Final feature list
    fingerprint_set: str  # Set-invariant fingerprint
    fingerprint_ordered: str  # Order-sensitive fingerprint
    canonical_lookback_map: Dict[str, float]  # Feature → lookback (minutes)
    actual_max_lookback_minutes: float  # Actual max from safe features
    cap_minutes: Optional[float]  # Cap that was enforced (if any)
    stage: str  # Stage name (e.g., "POST_GATEKEEPER", "POST_PRUNE")
    removal_reasons: Dict[str, str]  # Feature → reason (quarantined, pruned, etc.)
    timestamp: str  # ISO timestamp when artifact was created
    # Phase 2: Optional budget and config (for full integration)
    budget: Optional[Any] = None  # LeakageBudget object (optional, for full integration)
    resolved_config: Optional[Any] = None  # ResolvedConfig object (optional, for full integration)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Note: budget and resolved_config are not serialized (they're complex objects).
        They should be recomputed from features if needed after loading.
        """
        data = asdict(self)
        # Remove non-serializable fields
        data.pop('budget', None)
        data.pop('resolved_config', None)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSetArtifact':
        """Create from dictionary (for loading from JSON)."""
        return cls(**data)
    
    def save(self, output_dir: Path, filename: Optional[str] = None) -> Path:
        """
        Save artifact to disk for debugging.
        
        Args:
            output_dir: Directory to save artifact
            filename: Optional filename (default: featureset_{stage}.json)
        
        Returns:
            Path to saved file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f"featureset_{self.stage.lower()}.json"
        
        artifact_path = output_dir / filename
        
        try:
            with open(artifact_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.debug(f"  📄 FeatureSet artifact saved: {artifact_path}")
            return artifact_path
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to save FeatureSet artifact: {e}")
            raise
    
    @classmethod
    def load(cls, artifact_path: Path) -> 'FeatureSetArtifact':
        """
        Load artifact from disk.
        
        Args:
            artifact_path: Path to JSON file
        
        Returns:
            FeatureSetArtifact instance
        """
        try:
            with open(artifact_path, 'r') as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"  ❌ Failed to load FeatureSet artifact from {artifact_path}: {e}")
            raise
    
    def validate(self) -> bool:
        """
        Validate artifact invariants.
        
        Returns:
            True if valid, raises RuntimeError if invalid
        """
        # Check that actual_max <= cap (if cap is set)
        if self.cap_minutes is not None:
            if self.actual_max_lookback_minutes > self.cap_minutes:
                raise RuntimeError(
                    f"Invalid artifact: actual_max={self.actual_max_lookback_minutes:.1f}m > "
                    f"cap={self.cap_minutes:.1f}m"
                )
        
        # Check that all features in canonical_map are in features list or removal_reasons
        all_tracked = set(self.features) | set(self.removal_reasons.keys())
        canonical_keys = set(self.canonical_lookback_map.keys())
        untracked = canonical_keys - all_tracked
        if untracked:
            logger.warning(
                f"  ⚠️  Artifact validation: {len(untracked)} features in canonical_map "
                f"not in features or removal_reasons. Sample: {list(untracked)[:5]}"
            )
        
        return True


def create_artifact_from_enforced(
    enforced: Any,  # EnforcedFeatureSet or LookbackCapResult
    stage: str,
    removal_reasons: Optional[Dict[str, str]] = None
) -> FeatureSetArtifact:
    """
    Create FeatureSetArtifact from EnforcedFeatureSet or LookbackCapResult.
    
    Args:
        enforced: EnforcedFeatureSet or LookbackCapResult
        stage: Stage name
        removal_reasons: Optional dict of feature → reason for removed features
    
    Returns:
        FeatureSetArtifact
    """
    from TRAINING.utils.cross_sectional_data import _compute_feature_fingerprint
    
    # Handle both EnforcedFeatureSet and LookbackCapResult
    if hasattr(enforced, 'features'):
        features = enforced.features
        fingerprint_set = enforced.fingerprint_set if hasattr(enforced, 'fingerprint_set') else enforced.fingerprint
        fingerprint_ordered = enforced.fingerprint_ordered if hasattr(enforced, 'fingerprint_ordered') else fingerprint_set
        canonical_map = enforced.canonical_map
        actual_max = enforced.actual_max_minutes if hasattr(enforced, 'actual_max_minutes') else enforced.actual_max_lookback
        cap_minutes = enforced.cap_minutes if hasattr(enforced, 'cap_minutes') else None
    else:
        # LookbackCapResult
        features = enforced.safe_features
        fingerprint_set = enforced.fingerprint
        fingerprint_ordered = fingerprint_set  # LookbackCapResult doesn't have ordered fingerprint
        canonical_map = enforced.canonical_map
        actual_max = enforced.actual_max_lookback
        cap_minutes = None  # LookbackCapResult doesn't store cap
    
    # Compute ordered fingerprint if not available
    if fingerprint_ordered == fingerprint_set:
        _, fingerprint_ordered = _compute_feature_fingerprint(features, set_invariant=True)
    
    # Build removal_reasons from quarantined features
    if removal_reasons is None:
        removal_reasons = {}
    
    # Add quarantined features to removal_reasons
    if hasattr(enforced, 'quarantined'):
        for feat_name, lookback in enforced.quarantined.items():
            if feat_name not in removal_reasons:
                if lookback == float("inf"):
                    removal_reasons[feat_name] = "unknown lookback (quarantined)"
                else:
                    removal_reasons[feat_name] = f"lookback ({lookback:.1f}m) > cap (quarantined)"
    
    if hasattr(enforced, 'quarantined_features'):
        for feat_name in enforced.quarantined_features:
            if feat_name not in removal_reasons:
                lookback = canonical_map.get(feat_name, float("inf"))
                if lookback == float("inf"):
                    removal_reasons[feat_name] = "unknown lookback (quarantined)"
                else:
                    removal_reasons[feat_name] = f"lookback ({lookback:.1f}m) > cap (quarantined)"
    
    return FeatureSetArtifact(
        features=features,
        fingerprint_set=fingerprint_set,
        fingerprint_ordered=fingerprint_ordered,
        canonical_lookback_map=canonical_map,
        actual_max_lookback_minutes=actual_max,
        cap_minutes=cap_minutes,
        stage=stage,
        removal_reasons=removal_reasons,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )

