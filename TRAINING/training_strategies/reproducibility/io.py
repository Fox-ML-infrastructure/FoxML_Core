# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Training Snapshot I/O Operations

Save/load TrainingSnapshots and manage global training_snapshot_index.json.
Reuses SST patterns from feature_importance/io.py for consistency.
"""

import json
import logging
import fcntl
from pathlib import Path
from typing import Dict, Optional, Any

from .schema import TrainingSnapshot

logger = logging.getLogger(__name__)


def get_training_snapshot_dir(
    output_dir: Path,
    target: str,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    stage: str = "TRAINING",
) -> Path:
    """
    Get the directory for training snapshots using stage-scoped paths.
    
    Structure: targets/{target}/reproducibility/stage=TRAINING/{view}/[symbol=X/]
    
    Args:
        output_dir: Base output directory
        target: Target name (e.g., "fwd_ret_10m")
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name for SYMBOL_SPECIFIC views
        stage: Pipeline stage (default: "TRAINING")
    
    Returns:
        Path to training snapshot directory
    """
    base_path = output_dir / "targets" / target / "reproducibility" / f"stage={stage}" / view
    
    if view == "SYMBOL_SPECIFIC" and symbol:
        base_path = base_path / f"symbol={symbol}"
    
    return base_path


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
        output_path = Path(output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)
        
        logger.debug(f"Saved training snapshot: {output_path}")
        return output_path
    except Exception as e:
        logger.warning(f"Failed to save training snapshot: {e}")
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
    
    Returns:
        TrainingSnapshot if created and saved successfully, None otherwise
    """
    try:
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
        )
        
        # Get snapshot directory
        snapshot_dir = get_training_snapshot_dir(
            output_dir=output_dir,
            target=target,
            view=view,
            symbol=symbol,
            stage="TRAINING",
        )
        
        # Create model-specific subdirectory
        model_snapshot_dir = snapshot_dir / model_family
        
        # Set path for global index
        try:
            path_parts = model_snapshot_dir.parts
            if 'targets' in path_parts:
                targets_idx = path_parts.index('targets')
                relative_path = '/'.join(path_parts[targets_idx:])
                snapshot.path = f"{relative_path}/training_snapshot.json"
        except Exception:
            snapshot.path = str(model_snapshot_dir / "training_snapshot.json")
        
        # Save snapshot
        saved_path = save_training_snapshot(snapshot, model_snapshot_dir)
        
        if saved_path:
            # Update global index
            update_training_snapshot_index(snapshot, output_dir)
            logger.info(f"Created training snapshot for {target}/{model_family}: {saved_path}")
            return snapshot
        
        return None
    except Exception as e:
        logger.warning(f"Failed to create training snapshot for {target}/{model_family}: {e}")
        return None
