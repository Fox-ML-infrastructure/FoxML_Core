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
Feature Importance Snapshot I/O

Save and load feature importance snapshots for stability analysis.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from .schema import FeatureImportanceSnapshot

logger = logging.getLogger(__name__)


def save_importance_snapshot(
    snapshot: FeatureImportanceSnapshot,
    base_dir: Path,
) -> Path:
    """
    Save feature importance snapshot to disk.
    
    Directory structure:
        {base_dir}/{target_name}/{method}/{run_id}.json
    
    Args:
        snapshot: FeatureImportanceSnapshot to save
        base_dir: Base directory for snapshots (e.g., "artifacts/feature_importance")
    
    Returns:
        Path to saved snapshot file
    """
    # Create directory structure
    target_dir = base_dir / snapshot.target_name / snapshot.method
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    path = target_dir / f"{snapshot.run_id}.json"
    
    try:
        with path.open("w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        logger.debug(f"Saved importance snapshot: {path}")
    except Exception as e:
        logger.error(f"Failed to save importance snapshot to {path}: {e}")
        raise
    
    return path


def load_snapshots(
    base_dir: Path,
    target_name: str,
    method: str,
    min_timestamp: Optional[datetime] = None,
    max_timestamp: Optional[datetime] = None,
) -> List[FeatureImportanceSnapshot]:
    """
    Load all snapshots for a target and method.
    
    Args:
        base_dir: Base directory for snapshots
        target_name: Target name to load
        method: Method name to load
        min_timestamp: Optional minimum timestamp filter
        max_timestamp: Optional maximum timestamp filter
    
    Returns:
        List of FeatureImportanceSnapshot instances, sorted by created_at (oldest first)
    """
    target_dir = base_dir / target_name / method
    
    if not target_dir.exists():
        logger.debug(f"No snapshots directory found: {target_dir}")
        return []
    
    snapshots = []
    for path in sorted(target_dir.glob("*.json")):
        try:
            with path.open("r") as f:
                data = json.load(f)
            
            snapshot = FeatureImportanceSnapshot.from_dict(data)
            
            # Apply timestamp filters if provided
            if min_timestamp and snapshot.created_at < min_timestamp:
                continue
            if max_timestamp and snapshot.created_at > max_timestamp:
                continue
            
            snapshots.append(snapshot)
        except Exception as e:
            logger.warning(f"Failed to load snapshot {path}: {e}")
            continue
    
    # Sort by creation time (oldest first)
    snapshots.sort(key=lambda s: s.created_at)
    
    return snapshots


def get_snapshot_base_dir(output_dir: Optional[Path] = None) -> Path:
    """
    Get base directory for snapshots.
    
    Uses output_dir if provided, otherwise defaults to artifacts/feature_importance.
    
    Args:
        output_dir: Optional output directory (snapshots go in {output_dir}/feature_importance_snapshots)
    
    Returns:
        Path to base snapshot directory
    """
    if output_dir is not None:
        return output_dir / "feature_importance_snapshots"
    else:
        # Default: artifacts/feature_importance
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[4]  # TRAINING/stability/feature_importance/io.py -> repo root
        return repo_root / "artifacts" / "feature_importance"
