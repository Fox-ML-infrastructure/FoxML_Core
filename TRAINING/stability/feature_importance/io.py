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

from .schema import FeatureImportanceSnapshot, FeatureSelectionSnapshot

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
        
        # Write per-directory manifest for human readability (hash-based path only)
        if use_hash_path:
            _update_directory_manifest(replicate_dir, snapshot, strict_key)
            # Also update global manifest for easy method-to-directory lookup
            update_global_importance_manifest(base_dir, snapshot)
    except Exception as e:
        logger.error(f"Failed to save importance snapshot to {path}: {e}")
        raise
    
    return path


def _update_directory_manifest(
    replicate_dir: Path,
    snapshot: FeatureImportanceSnapshot,
    strict_key: str,
) -> None:
    """
    Update manifest.json in a replicate directory for human readability.
    
    The manifest maps hash-based filenames to human-readable metadata.
    """
    manifest_path = replicate_dir / "manifest.json"
    
    try:
        # Load existing manifest or create new
        if manifest_path.exists():
            with manifest_path.open("r") as f:
                manifest = json.load(f)
        else:
            manifest = {
                "target": snapshot.target,
                "method": snapshot.method,
                "view": getattr(snapshot, 'view', 'CROSS_SECTIONAL'),
                "replicate_key": snapshot.replicate_key,
                "snapshots": []
            }
        
        # Add this snapshot if not already present
        snapshot_entry = {
            "file": f"{strict_key}.json",
            "timestamp": snapshot.created_at.isoformat() if hasattr(snapshot.created_at, 'isoformat') else str(snapshot.created_at),
            "run_id": snapshot.run_id,
            "n_features": len(snapshot.features) if snapshot.features else 0,
        }
        
        # Check if already in manifest
        existing_files = [s.get('file') for s in manifest.get('snapshots', [])]
        if snapshot_entry['file'] not in existing_files:
            manifest.setdefault('snapshots', []).append(snapshot_entry)
        
        # Update metadata
        manifest['target'] = snapshot.target
        manifest['method'] = snapshot.method
        manifest['last_updated'] = datetime.utcnow().isoformat()
        
        # Write manifest
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.debug(f"Updated manifest at {manifest_path}")
    except Exception as e:
        logger.debug(f"Failed to update manifest (non-critical): {e}")


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
            # Skip fs_snapshot.json (different schema, not a FeatureImportanceSnapshot)
            if path.name == "fs_snapshot.json":
                continue
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
            logger.debug(
                f"Legacy snapshot loading skipped for {target}/{method} (allow_legacy=False). "
                "Use replicate_key for hash-based loading."
            )
            return []
        
        target_dir = base_dir / target / method
        
        if not target_dir.exists():
            logger.debug(f"No snapshots directory found: {target_dir}")
            return []
        
        for path in sorted(target_dir.glob("*.json")):
            # Skip fs_snapshot.json (different schema, not a FeatureImportanceSnapshot)
            if path.name == "fs_snapshot.json":
                continue
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


def get_snapshot_base_dir(
    output_dir: Optional[Path] = None,
    target: Optional[str] = None,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
) -> Path:
    """
    Get base directory for snapshots.
    
    Uses target-first structure scoped by view if output_dir and target are provided.
    Never creates root-level feature_importance_snapshots directory.
    
    Args:
        output_dir: Optional output directory (snapshots go in target-first structure)
        target: Optional target name for target-first structure
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" for scoping
        symbol: Symbol name for SYMBOL_SPECIFIC view
    
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
                    ensure_scoped_artifact_dir, ensure_target_structure
                )
                target_clean = target.replace('/', '_').replace('\\', '_')
                ensure_target_structure(base_output_dir, target_clean)
                return ensure_scoped_artifact_dir(
                    base_output_dir, target_clean, "feature_importance_snapshots",
                    view=view, symbol=symbol
                )
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


def save_fs_snapshot(
    snapshot: 'FeatureSelectionSnapshot',
    cohort_dir: Path,
) -> Path:
    """
    Save FeatureSelectionSnapshot to fs_snapshot.json in cohort directory.
    
    Mirrors the snapshot.json structure used by TARGET_RANKING.
    
    Args:
        snapshot: FeatureSelectionSnapshot to save
        cohort_dir: Cohort directory (e.g., targets/fwd_ret_10m/reproducibility/CROSS_SECTIONAL/cohort=.../
    
    Returns:
        Path to saved fs_snapshot.json
    """
    from .schema import FeatureSelectionSnapshot
    
    cohort_dir = Path(cohort_dir)
    cohort_dir.mkdir(parents=True, exist_ok=True)
    
    path = cohort_dir / "fs_snapshot.json"
    
    try:
        with path.open("w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)
        logger.debug(f"Saved fs_snapshot.json: {path}")
    except Exception as e:
        logger.error(f"Failed to save fs_snapshot.json to {path}: {e}")
        raise
    
    return path


def update_fs_snapshot_index(
    snapshot: 'FeatureSelectionSnapshot',
    output_dir: Path,
) -> Optional[Path]:
    """
    Update globals/fs_snapshot_index.json with new snapshot entry.
    
    Mirrors the snapshot_index.json structure used by TARGET_RANKING.
    
    Args:
        snapshot: FeatureSelectionSnapshot to add to index
        output_dir: Run output directory (containing globals/)
    
    Returns:
        Path to updated fs_snapshot_index.json, or None on failure
    """
    from .schema import FeatureSelectionSnapshot
    
    output_dir = Path(output_dir)
    
    # Find globals directory
    globals_dir = None
    # Try to find run root with globals/
    base_dir = output_dir
    for _ in range(10):
        if (base_dir / "globals").exists():
            globals_dir = base_dir / "globals"
            break
        if not base_dir.parent.exists():
            break
        base_dir = base_dir.parent
    
    if globals_dir is None:
        # Create globals in output_dir
        globals_dir = output_dir / "globals"
        globals_dir.mkdir(parents=True, exist_ok=True)
    
    index_path = globals_dir / "fs_snapshot_index.json"
    
    # Load existing index or create new
    index = {}
    if index_path.exists():
        try:
            with index_path.open("r") as f:
                index = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load existing fs_snapshot_index.json: {e}")
            index = {}
    
    # Add/update entry
    key = snapshot.get_index_key()
    index[key] = snapshot.to_dict()
    
    # Write updated index
    try:
        with index_path.open("w") as f:
            json.dump(index, f, indent=2, default=str)
        logger.debug(f"Updated fs_snapshot_index.json with key: {key}")
        return index_path
    except Exception as e:
        logger.error(f"Failed to update fs_snapshot_index.json: {e}")
        return None


def create_fs_snapshot_from_importance(
    importance_snapshot: FeatureImportanceSnapshot,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    cohort_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    inputs: Optional[Dict] = None,
    process: Optional[Dict] = None,
    stage: str = "FEATURE_SELECTION",  # Allow caller to specify stage
) -> Optional['FeatureSelectionSnapshot']:
    """
    Create and save FeatureSelectionSnapshot from existing FeatureImportanceSnapshot.
    
    This bridges the existing snapshot system with the new full structure,
    writing fs_snapshot.json and updating fs_snapshot_index.json.
    
    Args:
        importance_snapshot: Existing FeatureImportanceSnapshot
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name for SYMBOL_SPECIFIC views
        cohort_dir: Optional cohort directory for fs_snapshot.json
        output_dir: Optional output directory for fs_snapshot_index.json
        inputs: Optional inputs dict (config, data, target info)
        process: Optional process dict (split info)
        stage: Pipeline stage - "FEATURE_SELECTION" (default) or "TARGET_RANKING"
    
    Returns:
        FeatureSelectionSnapshot if created successfully, None otherwise
    """
    from .schema import FeatureSelectionSnapshot
    
    try:
        # Create snapshot from importance snapshot
        fs_snapshot = FeatureSelectionSnapshot.from_importance_snapshot(
            importance_snapshot=importance_snapshot,
            view=view,
            symbol=symbol,
            inputs=inputs,
            process=process,
            stage=stage,  # Pass stage to schema
        )
        
        # Set path relative to targets/
        if cohort_dir:
            cohort_path = Path(cohort_dir)
            # Try to extract relative path from targets/
            try:
                path_parts = cohort_path.parts
                if 'targets' in path_parts:
                    targets_idx = path_parts.index('targets')
                    relative_path = '/'.join(path_parts[targets_idx:])
                    fs_snapshot.path = f"{relative_path}/fs_snapshot.json"
            except Exception:
                fs_snapshot.path = str(cohort_path / "fs_snapshot.json")
        
        # Save to cohort directory if provided
        if cohort_dir:
            save_fs_snapshot(fs_snapshot, cohort_dir)
        
        # Update global index if output_dir provided
        if output_dir:
            update_fs_snapshot_index(fs_snapshot, output_dir)
        
        return fs_snapshot
    except Exception as e:
        logger.warning(f"Failed to create FeatureSelectionSnapshot: {e}")
        return None


def update_global_importance_manifest(
    base_dir: Path,
    snapshot: FeatureImportanceSnapshot,
) -> None:
    """
    Update global manifest.json in feature_importance_snapshots/ directory.
    
    Maps method names to their replicate directories for human navigation.
    
    Structure:
    {
        "target": "fwd_ret_10m",
        "last_updated": "2026-01-06T...",
        "methods": {
            "xgboost": {
                "replicate_dir": "replicate/abc123.../",
                "last_run_id": "...",
                "n_snapshots": 5
            },
            ...
        }
    }
    """
    manifest_path = base_dir / "manifest.json"
    
    try:
        # Load existing manifest or create new
        if manifest_path.exists():
            with manifest_path.open("r") as f:
                manifest = json.load(f)
        else:
            manifest = {
                "target": snapshot.target,
                "methods": {},
            }
        
        # Update method entry
        method = snapshot.method
        replicate_key = snapshot.replicate_key
        
        if method and replicate_key:
            if method not in manifest.get('methods', {}):
                manifest.setdefault('methods', {})[method] = {
                    "replicate_dir": f"replicate/{replicate_key}/",
                    "last_run_id": snapshot.run_id,
                    "n_snapshots": 1,
                }
            else:
                # Update existing entry
                entry = manifest['methods'][method]
                entry['last_run_id'] = snapshot.run_id
                entry['n_snapshots'] = entry.get('n_snapshots', 0) + 1
                # Update replicate_dir if different (new replicate group)
                if entry.get('replicate_dir') != f"replicate/{replicate_key}/":
                    entry['replicate_dir'] = f"replicate/{replicate_key}/"
        
        manifest['target'] = snapshot.target
        manifest['last_updated'] = datetime.utcnow().isoformat()
        
        # Write manifest
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        
        logger.debug(f"Updated global manifest at {manifest_path}")
    except Exception as e:
        logger.debug(f"Failed to update global manifest (non-critical): {e}")
