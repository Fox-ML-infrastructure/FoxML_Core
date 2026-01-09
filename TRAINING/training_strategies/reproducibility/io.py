# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Training Snapshot I/O Operations

Save/load TrainingSnapshots and manage global training_snapshot_index.json.
Reuses SST patterns from feature_importance/io.py for consistency.
"""

import json
import logging
import fcntl
import hashlib
import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Any, List

from .schema import TrainingSnapshot

logger = logging.getLogger(__name__)


def compute_cohort_id_from_metadata(
    cohort_metadata: Dict[str, Any],
    view: str = "CROSS_SECTIONAL",
) -> str:
    """
    Compute cohort ID from cohort metadata (mirrors ReproducibilityTracker._compute_cohort_id).
    
    Args:
        cohort_metadata: Cohort metadata dict from extract_cohort_metadata()
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
    
    Returns:
        Cohort ID string (e.g., "cs_2025Q3_ef91e9db233a_min_cs3_max2000_v1_abc12345")
    """
    # Map view to mode prefix
    view_upper = (view or "").upper()
    if view_upper == "CROSS_SECTIONAL":
        mode_prefix = "cs"
    elif view_upper == "SYMBOL_SPECIFIC":
        mode_prefix = "sy"
    else:
        raise ValueError(f"Invalid view: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
    
    # Extract date range
    date_range = cohort_metadata.get('date_range', {})
    date_start = date_range.get('start_ts', '')
    date_end = date_range.get('end_ts', '')
    
    # Convert to quarter format if possible
    date_str = ""
    if date_start:
        try:
            dt = pd.Timestamp(date_start)
            date_str = f"{dt.year}Q{(dt.month-1)//3 + 1}"
        except Exception as e:
            logger.debug(f"Failed to parse date {date_start} for cohort ID: {e}, using YYYY-MM format")
            date_str = date_start[:7] if len(date_start) >= 7 else date_start  # YYYY-MM
    
    # Extract universe/config
    cs_config = cohort_metadata.get('cs_config', {})
    universe = cs_config.get('universe_sig', 'default')
    min_cs = cs_config.get('min_cs', '')
    max_cs = cs_config.get('max_cs_samples', '')
    leak_ver = cs_config.get('leakage_filter_version', 'v1')
    
    # Build readable parts
    parts = [mode_prefix]
    if date_str:
        parts.append(date_str)
    if universe and universe != 'default':
        parts.append(universe)
    if min_cs:
        parts.append(f"min_cs{min_cs}")
    if max_cs and max_cs != 100000:  # Only include if non-default
        parts.append(f"max{max_cs}")
    parts.append(leak_ver.replace('.', '_'))
    
    cohort_id = "_".join(parts)
    
    # Add short hash for uniqueness
    hash_str = "|".join([
        str(cohort_metadata.get('n_effective_cs', '')),
        str(cohort_metadata.get('n_symbols', '')),
        date_start,
        date_end,
        json.dumps(cs_config, sort_keys=True)
    ])
    short_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:8]
    
    return f"{cohort_id}_{short_hash}"


def get_training_snapshot_dir(
    output_dir: Path,
    target: str,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    stage: str = "TRAINING",
    cohort_id: Optional[str] = None,
) -> Path:
    """
    Get the directory for training snapshots using stage-scoped paths with cohort support.
    
    Structure: targets/{target}/reproducibility/stage=TRAINING/{view}/cohort={cohort_id}/
    For SYMBOL_SPECIFIC: targets/{target}/reproducibility/stage=TRAINING/SYMBOL_SPECIFIC/symbol={symbol}/cohort={cohort_id}/
    
    Args:
        output_dir: Base output directory
        target: Target name (e.g., "fwd_ret_10m")
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name for SYMBOL_SPECIFIC views
        stage: Pipeline stage (default: "TRAINING")
        cohort_id: Cohort identifier (if None, returns base directory without cohort)
    
    Returns:
        Path to training snapshot directory
    """
    base_path = output_dir / "targets" / target / "reproducibility" / f"stage={stage}" / view
    
    if view == "SYMBOL_SPECIFIC" and symbol:
        base_path = base_path / f"symbol={symbol}"
    
    if cohort_id:
        base_path = base_path / f"cohort={cohort_id}"
    
    return base_path


def _prepare_for_parquet(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts dictionary keys to strings for Parquet compatibility.
    
    Parquet cannot serialize dictionaries with non-string keys (e.g., integers).
    This function ensures all keys are strings while preserving the structure.
    """
    prepared_data = {}
    for k, v in data.items():
        if isinstance(v, dict):
            prepared_data[str(k)] = _prepare_for_parquet(v)
        elif isinstance(v, list):
            prepared_data[str(k)] = [
                _prepare_for_parquet(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            prepared_data[str(k)] = v
    return prepared_data


def _write_atomic_json(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON file atomically using temp file + rename with full durability.
    
    This ensures crash consistency AND power-loss safety:
    1. Write to temp file
    2. fsync(tempfile) - ensure data is on disk
    3. os.replace() - atomic rename (POSIX: atomic, Windows: best-effort)
    4. fsync(directory) - ensure directory entry is on disk
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = file_path.with_suffix('.tmp')
    
    try:
        # Write to temp file
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()  # Ensure immediate write
            os.fsync(f.fileno())  # Force write to disk (durability)
        
        # Atomic rename (POSIX: atomic, Windows: best-effort)
        os.replace(temp_file, file_path)
        
        # Sync directory entry to ensure rename is durable
        try:
            dir_fd = os.open(file_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)  # Sync directory entry
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            # Fallback: some systems don't support directory fsync
            pass
    except Exception as e:
        # Cleanup temp file on failure
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise IOError(f"Failed to write atomic JSON to {file_path}: {e}") from e


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
    import time
    
    # Create lock file (same directory, .lock extension)
    lock_file = file_path.with_suffix('.lock')
    
    # Ensure parent directory exists
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
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
            
            # Lock acquired - perform write
            _write_atomic_json(file_path, data)
            
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


def save_training_snapshot(
    snapshot: TrainingSnapshot,
    output_dir: Path,
    filename: str = "training_snapshot.json",
) -> Optional[Path]:
    """
    Save TrainingSnapshot to stage-scoped path.
    
    Args:
        snapshot: TrainingSnapshot to save
        output_dir: Directory to save in (should be cohort or model directory)
        filename: Filename for snapshot (default: training_snapshot.json)
    
    Returns:
        Path to saved snapshot, or None if failed
    """
    try:
        # Ensure output_dir is a Path object (Path is imported at module level)
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)
        output_path = output_dir / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use atomic write with locking for safety (prevents race conditions)
        _write_atomic_json_with_lock(output_path, snapshot.to_dict())
        
        logger.debug(f"Saved training snapshot: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"Failed to save training snapshot: {e}")
        return None


def save_training_metadata_parquet(
    snapshot: TrainingSnapshot,
    output_dir: Path,
    filename: str = "training_metadata.parquet",
) -> Optional[Path]:
    """
    Save TrainingSnapshot to Parquet format for querying.
    
    Args:
        snapshot: TrainingSnapshot to save
        output_dir: Directory to save in (should be cohort directory)
        filename: Filename for Parquet file (default: training_metadata.parquet)
    
    Returns:
        Path to saved Parquet file, or None if failed
    """
    try:
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert snapshot to flat dict for Parquet
        parquet_dict = snapshot.to_parquet_dict()
        
        # Prepare for Parquet (stringify all keys)
        prepared_dict = _prepare_for_parquet(parquet_dict)
        
        # Create DataFrame (single row)
        df = pd.DataFrame([prepared_dict])
        
        # Write to Parquet
        df.to_parquet(
            output_path,
            index=False,
            engine='pyarrow',
            compression='snappy'
        )
        
        logger.debug(f"Saved training metadata Parquet: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"Failed to save training metadata Parquet: {e}")
        return None


def load_training_snapshot(snapshot_path: Path) -> Optional[TrainingSnapshot]:
    """
    Load TrainingSnapshot from JSON file.
    
    Args:
        snapshot_path: Path to training_snapshot.json
    
    Returns:
        TrainingSnapshot if loaded successfully, None otherwise
    """
    try:
        with open(snapshot_path, 'r') as f:
            data = json.load(f)
        return TrainingSnapshot.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load training snapshot from {snapshot_path}: {e}")
        return None


def create_aggregated_training_snapshot(
    snapshots: List[TrainingSnapshot],
    target: str,
    model_family: str,
    output_dir: Path,
    view: str = "SYMBOL_SPECIFIC",
    cohort_id: Optional[str] = None,
) -> Optional[TrainingSnapshot]:
    """
    Create aggregated training snapshot from multiple per-symbol snapshots.
    
    Aggregates metrics across symbols (mean, std, min, max) and creates
    a single cohort-level snapshot with symbol=None.
    
    Args:
        snapshots: List of per-symbol TrainingSnapshot objects
        target: Target name
        model_family: Model family
        output_dir: Base output directory
        view: View type (should be "SYMBOL_SPECIFIC" for aggregated)
        cohort_id: Cohort identifier
    
    Returns:
        Aggregated TrainingSnapshot if created successfully, None otherwise
    """
    if not snapshots:
        logger.warning("Cannot create aggregated snapshot: no snapshots provided")
        return None
    
    try:
        import numpy as np
        
        # Use first snapshot as template (fingerprints should be consistent)
        template = snapshots[0]
        
        # Aggregate metrics across symbols
        all_metrics = []
        all_n_samples = []
        all_training_times = []
        
        for snap in snapshots:
            if snap.outputs.get("metrics"):
                all_metrics.append(snap.outputs["metrics"])
            if snap.inputs.get("n_samples"):
                all_n_samples.append(snap.inputs["n_samples"])
            if snap.outputs.get("training_time_seconds"):
                all_training_times.append(snap.outputs["training_time_seconds"])
        
        # Aggregate metrics (mean, std, min, max for numeric values)
        aggregated_metrics = {}
        if all_metrics:
            # Get all unique metric keys
            all_keys = set()
            for metrics in all_metrics:
                all_keys.update(metrics.keys())
            
            for key in all_keys:
                values = [m.get(key) for m in all_metrics if key in m and isinstance(m[key], (int, float))]
                if values:
                    aggregated_metrics[f"{key}_mean"] = float(np.mean(values))
                    aggregated_metrics[f"{key}_std"] = float(np.std(values)) if len(values) > 1 else 0.0
                    aggregated_metrics[f"{key}_min"] = float(np.min(values))
                    aggregated_metrics[f"{key}_max"] = float(np.max(values))
                    # Also keep original key with mean value
                    aggregated_metrics[key] = aggregated_metrics[f"{key}_mean"]
        
        # Aggregate inputs
        aggregated_inputs = template.inputs.copy()
        if all_n_samples:
            aggregated_inputs["n_samples"] = int(np.sum(all_n_samples))
            aggregated_inputs["n_samples_mean"] = float(np.mean(all_n_samples))
            aggregated_inputs["n_samples_std"] = float(np.std(all_n_samples)) if len(all_n_samples) > 1 else 0.0
        
        # Aggregate outputs
        aggregated_outputs = template.outputs.copy()
        aggregated_outputs["metrics"] = aggregated_metrics
        if all_training_times:
            aggregated_outputs["training_time_seconds"] = float(np.sum(all_training_times))
            aggregated_outputs["training_time_mean"] = float(np.mean(all_training_times))
        
        # Create aggregated snapshot
        aggregated_snapshot = TrainingSnapshot(
            run_id=template.run_id,
            timestamp=template.timestamp,
            stage=template.stage,
            view=view,
            target=target,
            symbol=None,  # None indicates aggregated snapshot
            model_family=model_family,
            snapshot_seq=template.snapshot_seq,
            fingerprint_schema_version=template.fingerprint_schema_version,
            config_fingerprint=template.config_fingerprint,
            data_fingerprint=template.data_fingerprint,
            feature_fingerprint=template.feature_fingerprint,
            target_fingerprint=template.target_fingerprint,
            hyperparameters_signature=template.hyperparameters_signature,
            split_signature=template.split_signature,
            model_artifact_sha256=None,  # Aggregated snapshot doesn't have single model artifact
            metrics_sha256=None,  # Will be recomputed
            predictions_sha256=None,  # Aggregated snapshot doesn't have single prediction hash
            fingerprint_sources=template.fingerprint_sources,
            inputs=aggregated_inputs,
            process=template.process,
            outputs=aggregated_outputs,
            comparison_group=template.comparison_group,
            model_path=None,  # Aggregated snapshot doesn't have single model path
        )
        
        # Recompute metrics_sha256
        if aggregated_outputs.get("metrics"):
            try:
                metrics_json = json.dumps(aggregated_outputs["metrics"], sort_keys=True)
                aggregated_snapshot.metrics_sha256 = hashlib.sha256(metrics_json.encode()).hexdigest()
            except Exception:
                pass
        
        # Save aggregated snapshot
        snapshot_dir = get_training_snapshot_dir(
            output_dir=output_dir,
            target=target,
            view=view,
            symbol=None,  # Aggregated snapshot has no symbol
            stage="TRAINING",
            cohort_id=cohort_id,
        )
        
        saved_path = save_training_snapshot(aggregated_snapshot, snapshot_dir)
        
        # Also save Parquet
        try:
            save_training_metadata_parquet(aggregated_snapshot, snapshot_dir)
        except Exception as e:
            logger.debug(f"Failed to save aggregated training metadata Parquet (non-critical): {e}")
        
        if saved_path:
            update_training_snapshot_index(aggregated_snapshot, output_dir)
            logger.info(f"Created aggregated training snapshot for {target}/{model_family} (cohort={cohort_id}): {saved_path}")
            return aggregated_snapshot
        
        return None
    except Exception as e:
        logger.warning(f"Failed to create aggregated training snapshot: {e}")
        return None


def update_training_snapshot_index(
    snapshot: TrainingSnapshot,
    output_dir: Path,
    index_filename: str = "training_snapshot_index.json",
) -> Optional[Path]:
    """
    Update global training_snapshot_index.json with new snapshot entry.
    
    Uses file locking for safe concurrent access (reuses pattern from fs_snapshot_index).
    
    Args:
        snapshot: TrainingSnapshot to add to index
        output_dir: Base output directory containing globals/
        index_filename: Name of index file (default: training_snapshot_index.json)
    
    Returns:
        Path to index file if updated successfully, None otherwise
    """
    try:
        globals_dir = Path(output_dir) / "globals"
        globals_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = globals_dir / index_filename
        
        # Load existing index or create new
        existing_index = {}
        if index_path.exists():
            try:
                with open(index_path, 'r') as f:
                    existing_index = json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted index file, starting fresh: {index_path}")
                existing_index = {}
        
        # Add new entry
        key = snapshot.get_index_key()
        existing_index[key] = snapshot.to_dict()
        
        # Write with file locking for concurrent safety
        with open(index_path, 'w') as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                json.dump(existing_index, f, indent=2, default=str)
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except BlockingIOError:
                # Another process has the lock, try without exclusive lock
                json.dump(existing_index, f, indent=2, default=str)
        
        logger.debug(f"Updated training snapshot index: {index_path} (key={key})")
        return index_path
    except Exception as e:
        logger.warning(f"Failed to update training snapshot index: {e}")
        return None


def create_and_save_training_snapshot(
    target: str,
    model_family: str,
    model_result: Dict[str, Any],
    output_dir: Path,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    run_identity: Optional[Any] = None,
    model_path: Optional[str] = None,
    features_used: Optional[list] = None,
    n_samples: Optional[int] = None,
    train_seed: int = 42,
    snapshot_seq: int = 0,
    cohort_id: Optional[str] = None,
    cohort_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[TrainingSnapshot]:
    """
    Create TrainingSnapshot from training result and save to disk.
    
    This is the main entry point for saving training snapshots after model training.
    
    Args:
        target: Target name (e.g., "fwd_ret_10m")
        model_family: Model family (e.g., "xgboost", "lightgbm")
        model_result: Dictionary from training with metrics, model info
        output_dir: Base output directory
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name for SYMBOL_SPECIFIC views
        run_identity: RunIdentity object or dict with identity signatures
        model_path: Path to saved model artifact
        features_used: List of feature names used in training
        n_samples: Number of training samples
        train_seed: Training seed for reproducibility
        snapshot_seq: Sequence number for this run
        cohort_id: Cohort identifier (if None and cohort_metadata provided, will be computed)
        cohort_metadata: Cohort metadata dict (used to compute cohort_id if not provided)
    
    Returns:
        TrainingSnapshot if created and saved successfully, None otherwise
    """
    try:
        # Compute cohort_id if not provided but cohort_metadata is available
        if cohort_id is None and cohort_metadata is not None:
            try:
                cohort_id = compute_cohort_id_from_metadata(cohort_metadata, view=view)
            except Exception as e:
                logger.debug(f"Failed to compute cohort_id from metadata: {e}")
        
        # Create snapshot from training result
        snapshot = TrainingSnapshot.from_training_result(
            target=target,
            model_family=model_family,
            model_result=model_result,
            view=view,
            symbol=symbol,
            run_identity=run_identity,
            model_path=model_path,
            features_used=features_used,
            n_samples=n_samples,
            train_seed=train_seed,
            snapshot_seq=snapshot_seq,
            output_dir=output_dir,  # Pass output_dir for config.resolved.json loading
        )
        
        # Get snapshot directory (now with cohort support)
        snapshot_dir = get_training_snapshot_dir(
            output_dir=output_dir,
            target=target,
            view=view,
            symbol=symbol,
            stage="TRAINING",
            cohort_id=cohort_id,
        )
        
        # Save snapshot directly to cohort directory (no model-family subdirectory)
        # Set path for global index
        try:
            path_parts = snapshot_dir.parts
            if 'targets' in path_parts:
                targets_idx = path_parts.index('targets')
                relative_path = '/'.join(path_parts[targets_idx:])
                snapshot.path = f"{relative_path}/training_snapshot.json"
        except Exception:
            snapshot.path = str(snapshot_dir / "training_snapshot.json")
        
        # Save snapshot to cohort directory (JSON)
        saved_path = save_training_snapshot(snapshot, snapshot_dir)
        
        # Also save Parquet format for querying
        try:
            save_training_metadata_parquet(snapshot, snapshot_dir)
        except Exception as e:
            logger.debug(f"Failed to save training metadata Parquet (non-critical): {e}")
        
        if saved_path:
            # Update global index
            update_training_snapshot_index(snapshot, output_dir)
            logger.info(f"Created training snapshot for {target}/{model_family} (cohort={cohort_id}): {saved_path}")
            return snapshot
        
        return None
    except Exception as e:
        logger.warning(f"Failed to create training snapshot for {target}/{model_family}: {e}")
        return None


def aggregate_training_summaries(output_dir: Path) -> None:
    """
    Aggregate training summaries from all targets into globals/ directory.
    
    Collects all training snapshots from targets/*/reproducibility/stage=TRAINING/
    and creates aggregated summaries in globals/:
    - globals/training_summary.json (JSON format, human-readable)
    - globals/training_summary.csv (CSV format, easy inspection)
    
    Args:
        output_dir: Base run output directory
    """
    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
    from datetime import datetime
    
    globals_dir = get_globals_dir(output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    
    targets_dir = output_dir / "targets"
    if not targets_dir.exists():
        logger.debug("No targets directory found, skipping training summary aggregation")
        return
    
    # Collect all training snapshots
    all_snapshots = []
    by_target = {}
    summary_stats = {
        "success": 0,
        "failed": 0,
        "by_view": {"CROSS_SECTIONAL": 0, "SYMBOL_SPECIFIC": 0},
        "by_family": {}
    }
    
    # Walk through all target directories
    for target_dir in targets_dir.iterdir():
        if not target_dir.is_dir():
            continue
        
        target = target_dir.name
        repro_dir = target_dir / "reproducibility" / "stage=TRAINING"
        
        if not repro_dir.exists():
            continue
        
        by_target[target] = {
            "CROSS_SECTIONAL": {},
            "SYMBOL_SPECIFIC": {
                "aggregated": {},
                "by_symbol": {}
            }
        }
        
        # Check CROSS_SECTIONAL view
        cs_dir = repro_dir / "CROSS_SECTIONAL"
        if cs_dir.exists():
            for cohort_dir in cs_dir.iterdir():
                if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                    snapshot_path = cohort_dir / "training_snapshot.json"
                    if snapshot_path.exists():
                        snapshot = load_training_snapshot(snapshot_path)
                        if snapshot:
                            all_snapshots.append(snapshot)
                            family = snapshot.model_family
                            if family not in by_target[target]["CROSS_SECTIONAL"]:
                                by_target[target]["CROSS_SECTIONAL"][family] = snapshot.to_dict()
                            summary_stats["by_view"]["CROSS_SECTIONAL"] += 1
                            summary_stats["by_family"][family] = summary_stats["by_family"].get(family, 0) + 1
                            summary_stats["success"] += 1
        
        # Check SYMBOL_SPECIFIC view
        sym_dir = repro_dir / "SYMBOL_SPECIFIC"
        if sym_dir.exists():
            # Check for aggregated snapshots (symbol=None)
            for cohort_dir in sym_dir.iterdir():
                if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                    snapshot_path = cohort_dir / "training_snapshot.json"
                    if snapshot_path.exists():
                        snapshot = load_training_snapshot(snapshot_path)
                        if snapshot and snapshot.symbol is None:
                            # This is an aggregated snapshot
                            family = snapshot.model_family
                            by_target[target]["SYMBOL_SPECIFIC"]["aggregated"][family] = snapshot.to_dict()
            
            # Check per-symbol snapshots
            for symbol_dir in sym_dir.iterdir():
                if symbol_dir.is_dir() and symbol_dir.name.startswith("symbol="):
                    symbol = symbol_dir.name.replace("symbol=", "")
                    for cohort_dir in symbol_dir.iterdir():
                        if cohort_dir.is_dir() and cohort_dir.name.startswith("cohort="):
                            snapshot_path = cohort_dir / "training_snapshot.json"
                            if snapshot_path.exists():
                                snapshot = load_training_snapshot(snapshot_path)
                                if snapshot:
                                    all_snapshots.append(snapshot)
                                    family = snapshot.model_family
                                    if symbol not in by_target[target]["SYMBOL_SPECIFIC"]["by_symbol"]:
                                        by_target[target]["SYMBOL_SPECIFIC"]["by_symbol"][symbol] = {}
                                    by_target[target]["SYMBOL_SPECIFIC"]["by_symbol"][symbol][family] = snapshot.to_dict()
                                    summary_stats["by_view"]["SYMBOL_SPECIFIC"] += 1
                                    summary_stats["by_family"][family] = summary_stats["by_family"].get(family, 0) + 1
                                    summary_stats["success"] += 1
    
    # Create summary structure
    training_summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_models_trained": len(all_snapshots),
        "by_target": by_target,
        "summary": summary_stats
    }
    
    # Write JSON summary
    summary_json_path = globals_dir / "training_summary.json"
    try:
        with open(summary_json_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        logger.info(f"✅ Saved training summary JSON: {summary_json_path}")
    except Exception as e:
        logger.warning(f"Failed to write training_summary.json: {e}")
    
    # Write CSV summary for easy inspection
    summary_csv_path = globals_dir / "training_summary.csv"
    try:
        csv_rows = []
        for snapshot in all_snapshots:
            metrics = snapshot.outputs.get("metrics", {})
            primary_metric = None
            # Try to find primary metric (val_auc, r2, etc.)
            for key in ["val_auc", "r2", "spearman_ic__cs__mean", "roc_auc__cs__mean"]:
                if key in metrics:
                    primary_metric = metrics[key]
                    break
            
            csv_rows.append({
                "target": snapshot.target,
                "view": snapshot.view,
                "symbol": snapshot.symbol or "",
                "model_family": snapshot.model_family,
                "n_samples": snapshot.inputs.get("n_samples", ""),
                "n_features": snapshot.inputs.get("n_features", ""),
                "training_time_seconds": snapshot.outputs.get("training_time_seconds", ""),
                "primary_metric": primary_metric,
                "run_id": snapshot.run_id,
                "timestamp": snapshot.timestamp,
            })
        
        if csv_rows:
            df = pd.DataFrame(csv_rows)
            df.to_csv(summary_csv_path, index=False)
            logger.info(f"✅ Saved training summary CSV: {summary_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write training_summary.csv: {e}")
