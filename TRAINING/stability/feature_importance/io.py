# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Importance Snapshot I/O

Save and load feature importance snapshots for stability analysis.
"""

import json
import logging
import fcntl
import time
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
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
    # Ensure base_dir is a Path object (Path is imported at module level)
    if not isinstance(base_dir, Path):
        base_dir = Path(base_dir)
    
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
            # Skip files with different schemas (not FeatureImportanceSnapshot)
            # - fs_snapshot.json: FeatureSelectionSnapshot schema
            # - manifest.json: Per-replicate manifest with nested run_id (no top-level run_id)
            if path.name in ("fs_snapshot.json", "manifest.json"):
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
            # Skip files with different schemas (not FeatureImportanceSnapshot)
            # - fs_snapshot.json: FeatureSelectionSnapshot schema
            # - manifest.json: Per-replicate manifest with nested run_id (no top-level run_id)
            if path.name in ("fs_snapshot.json", "manifest.json"):
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
    universe_sig: Optional[str] = None,
    stage: Optional[str] = None,
    ensure_exists: bool = True,
) -> Path:
    """
    Get base directory for snapshots.

    Uses target-first structure scoped by stage, view and universe if output_dir and target are provided.
    Never creates root-level feature_importance_snapshots directory.

    Args:
        output_dir: Optional output directory (snapshots go in target-first structure)
        target: Optional target name for target-first structure
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC" for scoping
        symbol: Symbol name for SYMBOL_SPECIFIC view
        universe_sig: Universe signature for additional scoping within view
        stage: Optional stage name (TARGET_RANKING, FEATURE_SELECTION, TRAINING) - uses SST if not provided
        ensure_exists: If True (default), create the directory if it doesn't exist.
                       Set to False when reading to avoid creating empty directories.

    Returns:
        Path to base snapshot directory
    """
    # Convert string to Path if needed
    # Path is imported at module level, so it's always available
    if output_dir is not None and not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    
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
        # Don't stop at RESULTS/ - continue to find actual run directory using SST helper
        from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
        base_output_dir = get_run_root(output_dir)
        
        if base_output_dir.exists() and (base_output_dir / "targets").exists():
            try:
                from TRAINING.orchestration.utils.target_first_paths import (
                    ensure_scoped_artifact_dir, get_scoped_artifact_dir, ensure_target_structure
                )
                from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                target_clean = normalize_target_name(target)
                
                if ensure_exists:
                    # Writing: create directories
                    ensure_target_structure(base_output_dir, target_clean)
                    return ensure_scoped_artifact_dir(
                        base_output_dir, target_clean, "feature_importance_snapshots",
                        view=view, symbol=symbol, universe_sig=universe_sig, stage=stage
                    )
                else:
                    # Reading: don't create directories (prevents empty dir pollution)
                    return get_scoped_artifact_dir(
                        base_output_dir, target_clean, "feature_importance_snapshots",
                        view=view, symbol=symbol, universe_sig=universe_sig, stage=stage
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
        # Path is already imported at module level
        repo_root = Path(__file__).resolve().parents[4]  # TRAINING/stability/feature_importance/io.py -> repo root
        return repo_root / "artifacts" / "feature_importance"


def _write_atomic_json_with_lock(
    file_path: Path,
    data: Dict[str, Any],
    lock_timeout: float = 30.0
) -> None:
    """
    Write JSON file atomically with file locking to prevent race conditions.
    
    Uses fcntl.flock with LOCK_EX to ensure exclusive access during write.
    This prevents concurrent writes from multiple processes/threads.
    
    Args:
        file_path: Target file path
        data: Data to write (will use default=str for serialization)
        lock_timeout: Maximum time to wait for lock (seconds)
    
    Raises:
        IOError: If write fails or lock cannot be acquired
    """
    # Create lock file (same directory, .lock extension)
    lock_file = file_path.with_suffix('.lock')
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to temp file first
    temp_file = file_path.with_suffix('.tmp')
    
    start_time = time.time()
    lock_acquired = False
    
    try:
        # Try to acquire lock with timeout
        with open(lock_file, 'w') as lock_f:
            # Non-blocking attempt first
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_acquired = True
            except BlockingIOError:
                # Lock is held, wait for it with timeout
                elapsed = 0
                while elapsed < lock_timeout:
                    try:
                        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        lock_acquired = True
                        break
                    except BlockingIOError:
                        time.sleep(0.1)  # Wait 100ms before retry
                        elapsed = time.time() - start_time
                
                if not lock_acquired:
                    raise IOError(f"Could not acquire lock for {file_path} within {lock_timeout}s")
            
            # Lock acquired - perform atomic write
            try:
                with open(temp_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Atomic rename
                os.replace(temp_file, file_path)
                
                # Sync directory entry
                try:
                    dir_fd = os.open(file_path.parent, os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except (OSError, AttributeError):
                    pass
            except Exception as e:
                # Cleanup temp file on failure
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception:
                        pass
                raise
            
            # Lock is automatically released when file is closed
    except Exception as e:
        if lock_acquired:
            # Release lock on error (if we had it)
            try:
                with open(lock_file, 'w') as lock_f:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        raise IOError(f"Failed to write locked JSON to {file_path}: {e}") from e


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
        # Use atomic write with locking for safety (prevents race conditions)
        _write_atomic_json_with_lock(path, snapshot.to_dict())
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
    # Try to find run root with globals/ using SST helper
    from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
    base_dir = get_run_root(output_dir)
    globals_dir = base_dir / "globals" if (base_dir / "globals").exists() else None
    
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
    
    # SST: Sanitize index data to normalize enums to strings before JSON serialization
    from enum import Enum
    import pandas as pd
    def _sanitize_for_json(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [_sanitize_for_json(v) for v in obj]
        else:
            return obj
    sanitized_index = _sanitize_for_json(index)
    
    # Write updated index
    try:
        with index_path.open("w") as f:
            json.dump(sanitized_index, f, indent=2, default=str)
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
    snapshot_seq: int = 0,  # Sequence number for this run
    n_effective: Optional[int] = None,  # Effective sample count from FS
    feature_registry_hash: Optional[str] = None,  # Hash of feature registry
    comparable_key: Optional[str] = None,  # Pre-computed comparison key
    # P0 correctness: selection mode fields
    selection_mode: Optional[str] = None,  # "rank_only" | "top_k" | "threshold" | "importance_cutoff"
    n_candidates: Optional[int] = None,  # Number of candidate features entering selection
    n_selected: Optional[int] = None,  # Number of features after selection
    selection_params: Optional[Dict] = None,  # {"k": 50} or {"threshold": 0.01}
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
        snapshot_seq: Sequence number for this run (matches TARGET_RANKING)
        n_effective: Effective sample count from FS stage
        feature_registry_hash: Hash of feature registry used
        comparable_key: Pre-computed comparison key from cohort metadata
    
    Returns:
        FeatureSelectionSnapshot if created successfully, None otherwise
    """
    from .schema import FeatureSelectionSnapshot
    
    try:
        # Create snapshot from importance snapshot with full parity fields
        fs_snapshot = FeatureSelectionSnapshot.from_importance_snapshot(
            importance_snapshot=importance_snapshot,
            view=view,
            symbol=symbol,
            inputs=inputs,
            process=process,
            stage=stage,  # Pass stage to schema
            snapshot_seq=snapshot_seq,
            n_effective=n_effective,
            feature_registry_hash=feature_registry_hash,
            comparable_key=comparable_key,
            output_dir=output_dir,  # Pass output_dir for config.resolved.json loading
            # P0 correctness: selection mode fields
            selection_mode=selection_mode,
            n_candidates=n_candidates,
            n_selected=n_selected,
            selection_params=selection_params,
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
            index_result = update_fs_snapshot_index(fs_snapshot, output_dir)
            if index_result is None:
                logger.warning(
                    f"Failed to update fs_snapshot_index for {importance_snapshot.method} "
                    f"(target={importance_snapshot.target}, view={view}). "
                    f"Per-model snapshot may not be indexed."
                )
        
        return fs_snapshot
    except Exception as e:
        logger.warning(
            f"Failed to create FeatureSelectionSnapshot for {importance_snapshot.method if hasattr(importance_snapshot, 'method') else 'unknown'}: {e}"
        )
        import traceback
        logger.debug(f"FeatureSelectionSnapshot creation traceback: {traceback.format_exc()}")
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
