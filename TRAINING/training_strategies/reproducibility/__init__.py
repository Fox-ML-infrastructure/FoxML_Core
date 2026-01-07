# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Training Stage Reproducibility Module

Provides full-parity tracking for TRAINING stage (Stage 3) to match
TARGET_RANKING and FEATURE_SELECTION stages.

Key components:
- TrainingSnapshot: Dataclass for training run snapshots
- save_training_snapshot: Save snapshot to stage-scoped path
- update_training_snapshot_index: Update global index
"""

from .schema import TrainingSnapshot
from .io import (
    save_training_snapshot,
    load_training_snapshot,
    update_training_snapshot_index,
)

__all__ = [
    "TrainingSnapshot",
    "save_training_snapshot",
    "load_training_snapshot",
    "update_training_snapshot_index",
]
