# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

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
    use_hash_path: bool = False,
) -> Path:
    """
    Save feature importance snapshot to disk.
    
    Directory structure:
        Hash-based (preferred): {base_dir}/replicate/{replicate_key}/{strict_key}.json
        Legacy: {base_dir}/{target}/{method}/{run_id}.json
    
    Args:
        snapshot: FeatureImportanceSnapshot to save
        base_dir: Base directory for snapshots (e.g., "artifacts/feature_importance")
        use_hash_path: If True, use hash-based path (replicate_key/strict_key)
    
    Returns:
        Path to saved snapshot file
    """
    if use_hash_path:
        # Hash-based path: replicate/<replicate_key>/<strict_key>.json
        replicate_key = snapshot.replicate_key
        strict_key = snapshot.strict_key
        
        if not replicate_key or not strict_key:
            raise ValueError(
                "Cannot use hash-based path without replicate_key and strict_key. "
                "Ensure run_identity is finalized before saving."
            )
        
        # Create directory structure
        replicate_dir = base_dir / "replicate" / replicate_key
        replicate_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        path = replicate_dir / f"{strict_key}.json"
        
        logger.debug(f"Saving snapshot with hash-based path: {path}")
    else:
        # Legacy path: target/method/run_id.json
        target_dir = base_dir / snapshot.target / snapshot.method
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        path = target_dir / f"{snapshot.run_id}.json"
        
        logger.debug(f"Saving snapshot with legacy path: {path}")
    
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
    target: Optional[str] = None,
    method: Optional[str] = None,
    min_timestamp: Optional[datetime] = None,
    max_timestamp: Optional[datetime] = None,
    replicate_key: Optional[str] = None,
    strict_key: Optional[str] = None,
    allow_legacy: bool = False,
) -> List[FeatureImportanceSnapshot]:
    """
    Load snapshots with hash-based or legacy paths.
    
    PREFERRED: Use replicate_key to load all snapshots in a replicate group.
    LEGACY: Use target/method with allow_legacy=True.
    
    Args:
        base_dir: Base directory for snapshots
        target: Target name (legacy mode only)
        method: Method name (legacy mode only)
        min_timestamp: Optional minimum timestamp filter
        max_timestamp: Optional maximum timestamp filter
        replicate_key: Load all snapshots with this replicate_key (hash-based)
        strict_key: Load single snapshot with this strict_key (requires replicate_key)
        allow_legacy: If True, allow loading from legacy target/method paths
    
    Returns:
        List of FeatureImportanceSnapshot instances, sorted by created_at (oldest first)
    """
    snapshots = []
    
    # Hash-based loading (preferred)
    if replicate_key:
        replicate_dir = base_dir / "replicate" / replicate_key
        
        if not replicate_dir.exists():
            logger.debug(f"No replicate directory found: {replicate_dir}")
            return []
        
        if strict_key:
            # Load single snapshot
            path = replicate_dir / f"{strict_key}.json"
            if path.exists():
                try:
                    with path.open("r") as f:
                        data = json.load(f)
                    snapshots.append(FeatureImportanceSnapshot.from_dict(data))
                except Exception as e:
                    logger.warning(f"Failed to load snapshot {path}: {e}")
            return snapshots
        
        # Load all snapshots in replicate group
        for path in sorted(replicate_dir.glob("*.json")):
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
    
    # Legacy loading (requires explicit opt-in)
    if target and method:
        if not allow_legacy:
            logger.warning(
                f"Legacy snapshot loading requested for {target}/{method} but allow_legacy=False. "
                "Use replicate_key for hash-based loading, or set allow_legacy=True."
            )
            return []
        
        target_dir = base_dir / target / method
        
        if not target_dir.exists():
            logger.debug(f"No snapshots directory found: {target_dir}")
            return []
        
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
    
    # No valid loading mode specified
    logger.warning(
        "load_snapshots called without replicate_key or target/method. "
        "Use replicate_key for hash-based loading."
    )
    return []


def get_snapshot_base_dir(output_dir: Optional[Path] = None, target: Optional[str] = None) -> Path:
    """
    Get base directory for snapshots.
    
    Uses target-first structure if output_dir and target are provided.
    Never creates root-level feature_importance_snapshots directory.
    
    Args:
        output_dir: Optional output directory (snapshots go in target-first structure)
        target: Optional target name for target-first structure
    
    Returns:
        Path to base snapshot directory
    """
    if output_dir is not None:
        # REQUIRE target when output_dir is provided (saving case)
        # This ensures snapshots are written to the SST directory structure that aggregators scan
        if not target:
            raise ValueError(
                "target is required for snapshot base directory when output_dir is provided. "
                "This ensures snapshots are written to the SST directory structure that aggregators scan. "
                f"output_dir={output_dir}"
            )
        
        # Try to use target-first structure
        # Find base run directory
        # Only stop if we find a run directory (has targets/, globals/, or cache/)
        # Don't stop at RESULTS/ - continue to find actual run directory
        base_output_dir = output_dir
        for _ in range(10):
            if (base_output_dir / "targets").exists() or (base_output_dir / "globals").exists() or (base_output_dir / "cache").exists():
                break
            if not base_output_dir.parent.exists():
                break
            base_output_dir = base_output_dir.parent
        
        if base_output_dir.exists() and (base_output_dir / "targets").exists():
            try:
                from TRAINING.orchestration.utils.target_first_paths import (
                    get_target_reproducibility_dir, ensure_target_structure
                )
                target_clean = target.replace('/', '_').replace('\\', '_')
                ensure_target_structure(base_output_dir, target_clean)
                target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_clean)
                return target_repro_dir / "feature_importance_snapshots"
            except Exception as e:
                # If target-first structure fails, raise error (no fallback to artifacts)
                raise RuntimeError(
                    f"Failed to use target-first structure for snapshots with target={target}. "
                    f"output_dir={output_dir}, base_output_dir={base_output_dir}. "
                    f"Error: {e}. This is required for SST compliance."
                ) from e
        
        # If we can't find a valid run directory, raise error
        raise RuntimeError(
            f"Could not find valid run directory (with targets/ or globals/) from output_dir={output_dir}. "
            f"target={target}. This is required for SST compliance."
        )
    else:
        # Default: artifacts/feature_importance
        from pathlib import Path
        repo_root = Path(__file__).resolve().parents[4]  # TRAINING/stability/feature_importance/io.py -> repo root
        return repo_root / "artifacts" / "feature_importance"
