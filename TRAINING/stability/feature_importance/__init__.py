# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Feature Importance Stability Tracking

Non-invasive hooks for tracking and analyzing feature importance stability across runs.
"""

from .schema import FeatureImportanceSnapshot
from .io import save_importance_snapshot, load_snapshots, get_snapshot_base_dir
from .analysis import analyze_stability_auto, compute_stability_metrics, save_stability_report
from .hooks import save_snapshot_hook, save_snapshot_from_series_hook, analyze_all_stability_hook

__all__ = [
    'FeatureImportanceSnapshot',
    'save_importance_snapshot',
    'load_snapshots',
    'get_snapshot_base_dir',
    'analyze_stability_auto',
    'compute_stability_metrics',
    'save_stability_report',
    'save_snapshot_hook',
    'save_snapshot_from_series_hook',
    'analyze_all_stability_hook',
]
