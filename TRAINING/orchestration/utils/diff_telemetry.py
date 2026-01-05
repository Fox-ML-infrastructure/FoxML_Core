# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Diff Telemetry System

First-class telemetry with strict SST (Stable, Sortable, Typed) rules for tracking
changes across runs. Provides:
- Normalized snapshots for diffing
- Delta tracking (prev vs baseline)
- Comparison groups and comparability checks
- Blame assignment for drift
- Regression detection

Key principle: Only diff things that are canonically normalized and hash-addressed.
"""

import json
import logging
import hashlib
import fcntl
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Fingerprint schema version - increment when fingerprint computation changes
FINGERPRINT_SCHEMA_VERSION = "1.0"


# Import from modular components
from TRAINING.orchestration.utils.diff_telemetry.types import (
    ChangeSeverity,
    ComparabilityStatus,
    ResolvedRunContext,
    ComparisonGroup,
    NormalizedSnapshot,
    DiffResult,
    BaselineState
)

# Keep FINGERPRINT_SCHEMA_VERSION constant for backward compatibility
FINGERPRINT_SCHEMA_VERSION = "1.0"

# All dataclasses and enums are now in diff_telemetry/types.py
# Import them above (lines 52-60)

# Import atomic JSON write from centralized utilities
from TRAINING.common.utils.file_utils import write_atomic_json as _write_atomic_json

# DiffTelemetry class definition starts here
class DiffTelemetry:
    """
    First-class diff telemetry system with SST rules.
    
    Tracks:
    - Normalized snapshots per run
    - Diffs against previous comparable runs
    - Diffs against baseline (regression point)
    - Comparison groups and comparability
    - Blame assignment for drift
    """
    
    def __init__(
        self,
        output_dir: Path,
        min_runs_for_baseline: int = 5,
        baseline_window_size: int = 10
    ):
        """
        Initialize diff telemetry.
        
        Args:
            output_dir: Base output directory (RESULTS/ or RESULTS/{run}/)
            min_runs_for_baseline: Minimum runs before establishing baseline
            baseline_window_size: Rolling window size for baseline
        """
        self.output_dir = Path(output_dir)
        self.min_runs_for_baseline = min_runs_for_baseline
        self.baseline_window_size = baseline_window_size
        
        # Find RESULTS directory and determine structure
        results_dir = self.output_dir
        run_dir = None
        bin_dir = None
        
        # Walk up to find RESULTS directory and identify run/bin structure
        temp_dir = self.output_dir
        for _ in range(10):  # Limit depth
            if temp_dir.name == "RESULTS":
                results_dir = temp_dir
                break
            # Check if we're in a run directory (has REPRODUCIBILITY subdirectory)
            if (temp_dir / "REPRODUCIBILITY").exists():
                run_dir = temp_dir
            # Check if we're in a sample size bin directory
            if temp_dir.name.startswith("sample_") and temp_dir.parent.name == "RESULTS":
                bin_dir = temp_dir
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        # If we couldn't find RESULTS, try to infer from output_dir
        if results_dir.name != "RESULTS":
            # Try to find RESULTS by looking for sample_* directories
            temp_dir = self.output_dir
            for _ in range(10):
                if any((temp_dir / d).is_dir() and d.startswith("sample_") for d in temp_dir.iterdir() if d.is_dir()):
                    results_dir = temp_dir
                    break
                if not temp_dir.parent.exists():
                    break
                temp_dir = temp_dir.parent
        
        # Run-specific snapshot index: stored in run's globals/ (target-first structure)
        # Find run directory (where targets/ or globals/ exists)
        run_dir_for_globals = None
        temp_dir = self.output_dir
        for _ in range(10):  # Limit depth
            if (temp_dir / "targets").exists() or (temp_dir / "globals").exists() or temp_dir.name in ["RESULTS", "intelligent_output"]:
                run_dir_for_globals = temp_dir
                break
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        if run_dir_for_globals:
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(run_dir_for_globals)
            globals_dir.mkdir(parents=True, exist_ok=True)
            self.run_metrics_dir = globals_dir
            self.snapshot_index = self.run_metrics_dir / "snapshot_index.json"
        else:
            # Fallback: use output_dir/globals/ if we can't find run directory
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(self.output_dir)
            globals_dir.mkdir(parents=True, exist_ok=True)
            self.run_metrics_dir = globals_dir
            self.snapshot_index = self.run_metrics_dir / "snapshot_index.json"
        
        # Baselines are stored per-cohort (in cohort directory), not in a global index
        # This ensures only exactly the same runs (same cohort, stage, target, etc.) share baselines
        # We'll load baselines on-demand from cohort directories when needed
        
        # Store run_dir for later use when saving baselines
        self.run_dir = run_dir if run_dir else self.output_dir
        self.results_dir = results_dir
        
        # Track if run has been moved to comparison group directory
        self._run_moved = False
        
        # Load existing indices
        self._snapshots: Dict[str, NormalizedSnapshot] = {}
        self._baselines: Dict[str, BaselineState] = {}  # Cache for current session, but saved per-cohort
    
    def _load_indices(self):
        """
        Load snapshot index (baselines loaded on-demand from cohort directories).
        
        Handles both old format (run_id key) and new format (run_id:stage key) for backwards compatibility.
        """
        if self.snapshot_index.exists():
            try:
                with open(self.snapshot_index) as f:
                    data = json.load(f)
                    for key, snap_data in data.items():
                        snap = self._deserialize_snapshot(snap_data)
                        # Handle old formats:
                        # - run_id:stage (legacy, 1 colon)
                        # - run_id:stage:target:view (previous fix, 3 colons)
                        # - run_id:stage:target:view:symbol (current format, 4 colons)
                        if key.count(':') >= 4:
                            # Current format: run_id:stage:target:view:symbol
                            self._snapshots[key] = snap
                        elif key.count(':') >= 3:
                            # Previous format: run_id:stage:target:view - build new key with symbol
                            target_clean = (snap.target or "unknown").replace('/', '_').replace('\\', '_')
                            view_clean = snap.view or "UNKNOWN"
                            symbol_clean = (snap.symbol or "NONE").replace('/', '_').replace('\\', '_')
                            new_key = f"{snap.run_id}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}"
                            self._snapshots[new_key] = snap
                        else:
                            # Legacy format: run_id or run_id:stage - build new key with target, view, symbol
                            target_clean = (snap.target or "unknown").replace('/', '_').replace('\\', '_')
                            view_clean = snap.view or "UNKNOWN"
                            symbol_clean = (snap.symbol or "NONE").replace('/', '_').replace('\\', '_')
                            new_key = f"{snap.run_id}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}"
                            self._snapshots[new_key] = snap
            except Exception as e:
                logger.warning(f"Failed to load snapshot index: {e}")
        
        # Baselines are loaded on-demand from cohort directories (see _load_baseline_from_cohort)
    
    def _save_indices(self):
        """
        Save snapshot index per-run (not one mega file).
        
        CRITICAL: Uses (run_id, stage, target, view, symbol) as key to prevent overwrites.
        A single run_id can produce multiple stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING),
        and each stage can process multiple targets and symbols, so we need all five components in the key.
        
        Index is stored per-run in: {run_dir}/globals/snapshot_index.json (target-first structure)
        This keeps indices correlated by run and prevents one mega file from growing unbounded.
        
        Merges with existing snapshots in the file to prevent overwriting snapshots from other targets/symbols.
        """
        if not self.run_dir or not self.snapshot_index:
            return
        
        try:
            # Load existing snapshots from file to merge with new ones
            existing_snapshots = {}
            if self.snapshot_index.exists():
                try:
                    with open(self.snapshot_index) as f:
                        existing_data = json.load(f)
                        # Handle old formats:
                        # - run_id:stage (legacy)
                        # - run_id:stage:target:view (previous fix)
                        # - run_id:stage:target:view:symbol (current format)
                        for key, snap_data in existing_data.items():
                            existing_snapshots[key] = snap_data
                except Exception as e:
                    logger.debug(f"Failed to load existing snapshot index for merge: {e}")
            
            # Merge existing snapshots with new ones (new ones take precedence)
            run_snapshots = existing_snapshots.copy()
            for snapshot_key, snap in self._snapshots.items():
                # Build key: run_id:stage:target:view:symbol
                target_clean = (snap.target or "unknown").replace('/', '_').replace('\\', '_')
                view_clean = snap.view or "UNKNOWN"
                symbol_clean = (snap.symbol or "NONE").replace('/', '_').replace('\\', '_')
                key = f"{snap.run_id}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}"
                run_snapshots[key] = snap.to_dict()
            
            _write_atomic_json(self.snapshot_index, run_snapshots)
        except Exception as e:
            logger.warning(f"Failed to save snapshot index: {e}")
    
    def _load_baseline_from_cohort(self, cohort_dir: Path, comparison_group_key: str) -> Optional[BaselineState]:
        """Load baseline from cohort directory."""
        baseline_file = Path(cohort_dir) / "baseline.json"
        if baseline_file.exists():
            try:
                with open(baseline_file) as f:
                    data = json.load(f)
                    # Verify it matches the comparison group
                    if data.get('comparison_group_key') == comparison_group_key:
                        return BaselineState(**data)
            except Exception as e:
                logger.debug(f"Failed to load baseline from {baseline_file}: {e}")
        return None
    
    def _save_baseline_to_cohort(self, cohort_dir: Path, baseline: BaselineState):
        """Save baseline to cohort directory."""
        cohort_dir = Path(cohort_dir)
        # NEVER create REPRODUCIBILITY directories - only use target-first structure
        if "REPRODUCIBILITY" in str(cohort_dir):
            logger.warning(f"⚠️ Skipping baseline save to legacy REPRODUCIBILITY path: {cohort_dir}")
            return
        cohort_dir.mkdir(parents=True, exist_ok=True)
        baseline_file = cohort_dir / "baseline.json"
        try:
            with open(baseline_file, 'w') as f:
                json.dump(baseline.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save baseline to {baseline_file}: {e}")
    
    def _deserialize_snapshot(self, data: Dict[str, Any]) -> NormalizedSnapshot:
        """Deserialize snapshot from dict."""
        comp_group = None
        if 'comparison_group' in data:
            comp_group_data = data['comparison_group']
            # Handle backward compatibility: old snapshots might not have new fields
            if 'n_effective' not in comp_group_data:
                comp_group_data['n_effective'] = None
            if 'model_family' not in comp_group_data:
                comp_group_data['model_family'] = None
            if 'feature_signature' not in comp_group_data:
                comp_group_data['feature_signature'] = None
            comp_group = ComparisonGroup(**comp_group_data)
        
        return NormalizedSnapshot(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            stage=data['stage'],
            view=data.get('view'),
            target=data.get('target'),
            symbol=data.get('symbol'),
            snapshot_seq=data.get('snapshot_seq'),  # May be None for old snapshots
            config_fingerprint=data.get('config_fingerprint'),
            data_fingerprint=data.get('data_fingerprint'),
            feature_fingerprint=data.get('feature_fingerprint'),
            target_fingerprint=data.get('target_fingerprint'),
            metrics_sha256=data.get('metrics_sha256'),  # May be None for old snapshots
            artifacts_manifest_sha256=data.get('artifacts_manifest_sha256'),  # May be None for old snapshots
            predictions_sha256=data.get('predictions_sha256'),  # May be None for old snapshots
            fingerprint_sources=data.get('fingerprint_sources', {}),
            inputs=data.get('inputs', {}),
            process=data.get('process', {}),
            outputs=data.get('outputs', {}),
            comparison_group=comp_group
        )
    
    def _build_resolved_context(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]],
        cohort_dir: Optional[Path] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None
    ) -> ResolvedRunContext:
        """Build resolved run context from available data sources.
        
        CRITICAL: This resolves all outcome-influencing metadata from the actual sources
        to ensure no nulls for required fields.
        
        Priority order (for SST consistency):
        1. resolved_metadata (in-memory, most authoritative - same data that will be written)
        2. metadata.json from filesystem (only if resolved_metadata not provided, and verify run_id matches)
        3. cohort_metadata (fallback)
        4. additional_data (fallback)
        
        Args:
            stage: Pipeline stage
            run_data: Run data dict
            cohort_metadata: Cohort metadata (fallback)
            additional_data: Additional data dict
            cohort_dir: Cohort directory (fallback - only used if resolved_metadata not provided)
            resolved_metadata: In-memory metadata dict (SST - preferred source)
        
        Returns:
            ResolvedRunContext with all resolved values
        """
        ctx = ResolvedRunContext()
        
        # Priority 1: Use resolved_metadata if provided (SST consistency)
        # CRITICAL: Use shallow copy to prevent mutation of caller's dict
        metadata = {}
        if resolved_metadata:
            metadata = dict(resolved_metadata)  # Shallow copy to prevent mutation
            logger.debug("Using resolved_metadata for SST consistency (in-memory dict, shallow copy)")
        else:
            # Priority 2: Try to load metadata.json from filesystem (only if resolved_metadata not provided)
            if cohort_dir:
                metadata_file = Path(cohort_dir) / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            file_metadata = json.load(f)
                        # CRITICAL: Verify run_id AND stage match to avoid stale file hazard and cross-stage weirdness
                        current_run_id = run_data.get('run_id') or run_data.get('timestamp')
                        file_run_id = file_metadata.get('run_id')
                        file_stage = file_metadata.get('stage')
                        current_stage = stage
                        
                        if file_run_id == current_run_id and file_stage == current_stage:
                            metadata = file_metadata
                            logger.debug(f"Using metadata.json from filesystem (run_id={current_run_id}, stage={current_stage} match)")
                        else:
                            mismatch_reasons = []
                            if file_run_id != current_run_id:
                                mismatch_reasons.append(f"run_id mismatch: file={file_run_id}, current={current_run_id}")
                            if file_stage != current_stage:
                                mismatch_reasons.append(f"stage mismatch: file={file_stage}, current={current_stage}")
                            logger.debug(f"Skipping stale metadata.json ({', '.join(mismatch_reasons)})")
                    except Exception as e:
                        logger.debug(f"Could not load metadata.json from {metadata_file}: {e}")
        
        # Data provenance (from metadata.json or cohort_metadata)
        ctx.n_symbols = (
            metadata.get('n_symbols') or
            cohort_metadata.get('n_symbols') if cohort_metadata else None
        )
        ctx.symbols = (
            metadata.get('symbols') or
            cohort_metadata.get('symbols') if cohort_metadata else None
        )
        # Date range - use SST accessor
        from TRAINING.orchestration.utils.reproducibility.utils import extract_date_range
        ctx.date_start, ctx.date_end = extract_date_range(metadata, cohort_metadata)
        ctx.n_effective = (
            metadata.get('n_effective') or
            cohort_metadata.get('n_effective_cs') if cohort_metadata else None
        )
        ctx.min_cs = (
            metadata.get('min_cs') or
            cohort_metadata.get('min_cs') if cohort_metadata else None
        )
        ctx.max_cs_samples = (
            metadata.get('max_cs_samples') or
            cohort_metadata.get('max_cs_samples') if cohort_metadata else None
        )
        
        # Task provenance - extract target from multiple sources
        # CRITICAL: Check resolved_metadata first (SST consistency), then fallback to other sources
        ctx.target = (
            (resolved_metadata.get('target') if resolved_metadata else None) or
            (resolved_metadata.get('target') if resolved_metadata else None) or
            metadata.get('target') or  # Primary: metadata.json uses 'target'
            metadata.get('target') or  # Fallback: some sources use 'target'
            (additional_data.get('target') if additional_data else None) or
            (additional_data.get('target') if additional_data else None) or
            (run_data.get('target')) or
            (run_data.get('target')) or
            (run_data.get('target'))  # Fallback: target often contains target
        )
        
        # If still None, try to parse from cohort_dir path as last resort
        if not ctx.target and cohort_dir:
            try:
                # Path format: .../TARGET_RANKING/CROSS_SECTIONAL/{target}/cohort=...
                parts = str(cohort_dir).split('/')
                for i, part in enumerate(parts):
                    if part in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO'] and i + 1 < len(parts):
                        ctx.target = parts[i + 1]
                        break
            except Exception:
                pass
        cv_details = metadata.get('cv_details', {})
        ctx.horizon_minutes = (
            cv_details.get('horizon_minutes') or
            additional_data.get('horizon_minutes') if additional_data else None
        )
        ctx.labeling_impl_hash = (
            cv_details.get('label_definition_hash') or
            additional_data.get('labeling_hash') if additional_data else None
        )
        
        # Split provenance
        ctx.cv_method = cv_details.get('cv_method')
        ctx.purge_minutes = cv_details.get('purge_minutes')
        ctx.embargo_minutes = cv_details.get('embargo_minutes')
        ctx.leakage_filter_version = (
            metadata.get('leakage_filter_version') or
            additional_data.get('leakage_filter_version') if additional_data else None
        )
        ctx.split_seed = (
            metadata.get('seed') or
            additional_data.get('split_seed') if additional_data else None
        )
        ctx.fold_assignment_hash = (
            additional_data.get('fold_assignment_hash') if additional_data else None
        )
        
        # Feature provenance
        # Check multiple sources in priority order:
        # 1. resolved_metadata['evaluation']['n_features'] (nested - where it's actually stored)
        # 2. resolved_metadata['n_features'] (top-level fallback)
        # 3. metadata['evaluation']['n_features'] (from filesystem, nested)
        # 4. metadata['n_features'] (from filesystem, top-level)
        # 5. additional_data['n_features'] (direct pass-through)
        # Also check n_features_selected as alternative key name
        ctx.n_features = (
            (resolved_metadata.get('evaluation', {}).get('n_features') if resolved_metadata else None) or
            (resolved_metadata.get('n_features') if resolved_metadata else None) or
            (resolved_metadata.get('evaluation', {}).get('n_features_selected') if resolved_metadata else None) or
            (resolved_metadata.get('n_features_selected') if resolved_metadata else None) or
            metadata.get('evaluation', {}).get('n_features') or
            metadata.get('n_features') or
            metadata.get('evaluation', {}).get('n_features_selected') or
            metadata.get('n_features_selected') or
            (additional_data.get('n_features') if additional_data else None) or
            (additional_data.get('n_features_selected') if additional_data else None)
        )
        ctx.feature_names = (
            additional_data.get('feature_names') if additional_data else None
        )
        
        # Stage strategy
        ctx.view = (
            metadata.get('view') or
            additional_data.get('view') if additional_data else None
        )
        ctx.model_family = (
            additional_data.get('model_family') if additional_data else None
        )
        ctx.trainer_strategy = (
            additional_data.get('strategy') if additional_data else None
        )
        
        # Environment
        ctx.python_version = (
            additional_data.get('python_version') if additional_data else None
        )
        ctx.library_versions = (
            additional_data.get('library_versions') if additional_data else None
        )
        
        # Experiment tracking
        ctx.experiment_id = (
            metadata.get('experiment_id') or
            additional_data.get('experiment_id') if additional_data else None
        )
        
        return ctx
    
    def _get_required_fields_for_stage(self, stage: str) -> List[str]:
        """
        Get list of required fields for a given stage.
        
        These fields must be present and non-null in resolved_metadata before finalize_run().
        
        Args:
            stage: Pipeline stage
        
        Returns:
            List of required field names
        """
        base_required = ['stage', 'run_id', 'cohort_id']
        
        if stage == 'TARGET_RANKING':
            return base_required + [
                'date_start', 'date_end', 'n_symbols', 'n_effective',
                'target', 'view', 'min_cs', 'max_cs_samples'
            ]
        elif stage == 'FEATURE_SELECTION':
            return base_required + [
                'date_start', 'date_end', 'n_symbols', 'n_effective',
                'target', 'view', 'min_cs', 'max_cs_samples'
            ]
        elif stage == 'TRAINING':
            return base_required + [
                'date_start', 'date_end', 'n_symbols', 'n_effective',
                'target', 'view', 'model_family', 'min_cs', 'max_cs_samples'
            ]
        else:
            # Unknown stage - require base fields only
            return base_required
    
    def _validate_stage_schema(
        self,
        stage: str,
        ctx: ResolvedRunContext
    ) -> Tuple[bool, Optional[str]]:
        """Validate that required fields for stage are present and non-null.
        
        Returns:
            (is_valid, reason) - if not valid, reason explains what's missing
        """
        missing = []
        
        # All stages require:
        if ctx.n_effective is None:
            missing.append("n_effective")
        if ctx.target is None:
            missing.append("target")
        if ctx.view is None:
            missing.append("view")
        if ctx.date_start is None:
            missing.append("date_start")
        if ctx.date_end is None:
            missing.append("date_end")
        
        # Stage-specific requirements
        if stage == "TARGET_RANKING":
            # TARGET_RANKING does NOT require model_family or feature_signature
            pass
        elif stage == "FEATURE_SELECTION":
            # FEATURE_SELECTION requires feature pipeline info
            if ctx.n_features is None:
                missing.append("n_features")
        elif stage == "TRAINING":
            # TRAINING requires model_family and feature info
            if ctx.model_family is None:
                missing.append("model_family")
            if ctx.n_features is None:
                missing.append("n_features")
        
        if missing:
            return False, f"Missing required fields for {stage}: {', '.join(missing)}"
        return True, None
    
    def normalize_snapshot(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        cohort_dir: Optional[Path] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None,
        run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object
        prediction_fingerprint: Optional[Dict] = None,  # NEW: Prediction fingerprint dict
    ) -> NormalizedSnapshot:
        """
        Create normalized snapshot from run data.
        
        CRITICAL: This now uses ResolvedRunContext to ensure all required fields are
        non-null. Missing required fields will cause validation failure.
        
        For SST consistency, prefer `resolved_metadata` (in-memory metadata dict) over
        reading from filesystem. This ensures snapshot computation uses the exact same
        data that will be persisted.
        
        Args:
            stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
            run_data: Run data dict (from reproducibility tracker)
            cohort_metadata: Cohort metadata (fallback)
            additional_data: Additional data dict
            cohort_dir: Cohort directory (fallback - only used if resolved_metadata not provided)
            resolved_metadata: In-memory metadata dict (SST - preferred source)
            run_identity: RunIdentity SST object with authoritative signatures
            prediction_fingerprint: Prediction fingerprint dict for predictions_sha256
        
        Returns:
            NormalizedSnapshot
        
        Raises:
            ValueError: If required fields for stage are missing
        """
        # Build resolved context from all available sources (prefer resolved_metadata for SST)
        ctx = self._build_resolved_context(
            stage, run_data, cohort_metadata, additional_data, cohort_dir, resolved_metadata
        )
        
        # Validate stage-specific schema
        is_valid, reason = self._validate_stage_schema(stage, ctx)
        if not is_valid:
            raise ValueError(f"Cannot create snapshot for {stage}: {reason}")
        
        # Extract core identifiers
        run_id = run_data.get('run_id') or run_data.get('timestamp', datetime.now().isoformat())
        timestamp = run_data.get('timestamp', datetime.now().isoformat())
        view = ctx.view
        target = ctx.target
        symbol = additional_data.get('symbol') if additional_data else None
        
        # Extract universe_sig for comparison scoping (CRITICAL for CS runs)
        # Check top-level first, then cs_config for backward compatibility
        universe_sig = None
        if additional_data:
            universe_sig = additional_data.get('universe_sig')
            if not universe_sig and 'cs_config' in additional_data:
                universe_sig = additional_data['cs_config'].get('universe_sig')
        
        # Build fingerprints (using resolved context)
        config_fp = self._compute_config_fingerprint_from_context(ctx, additional_data)
        data_fp = self._compute_data_fingerprint_from_context(ctx)
        feature_fp = self._compute_feature_fingerprint_from_context(ctx)
        target_fp = self._compute_target_fingerprint_from_context(ctx)
        
        # Store fingerprints in context for comparison group
        ctx.data_fingerprint = data_fp
        ctx.target_fingerprint = target_fp
        ctx.feature_fingerprint = feature_fp
        
        # Extract hyperparameters and train_seed from process data (before normalization)
        # We need these for the comparison group, so extract them early
        hyperparameters_signature = None
        train_seed = None
        
        # Try to get hyperparameters from resolved_metadata (metadata.json), additional_data, or run_data
        training_data = {}
        if resolved_metadata and 'training' in resolved_metadata:
            training_data = resolved_metadata['training']
        elif additional_data and 'training' in additional_data:
            training_data = additional_data['training']
        elif run_data.get('additional_data') and 'training' in run_data.get('additional_data', {}):
            training_data = run_data['additional_data']['training']
        elif run_data.get('training'):
            training_data = run_data['training']
        
        # Extract hyperparameters (exclude model_family, strategy, seeds - those are handled separately)
        hyperparameters = {}
        excluded_keys = {'model_family', 'strategy', 'split_seed', 'train_seed', 'seed'}
        for key, value in training_data.items():
            if key not in excluded_keys and value is not None:
                hyperparameters[key] = value
        
        # Compute hyperparameters signature if we have any
        if hyperparameters:
            # Sort keys for stable hash
            hp_str = "|".join(f"{k}={v}" for k, v in sorted(hyperparameters.items()))
            hyperparameters_signature = hashlib.sha256(hp_str.encode()).hexdigest()[:16]
        
        # Extract train_seed
        train_seed = (
            training_data.get('train_seed') or
            training_data.get('seed') or
            (additional_data.get('train_seed') if additional_data else None) or
            (additional_data.get('seed') if additional_data else None) or
            (run_data.get('train_seed')) or
            (run_data.get('seed'))
        )
        if train_seed is not None:
            try:
                train_seed = int(train_seed)
            except (ValueError, TypeError):
                train_seed = None
        
        # Extract library versions and compute signature (CRITICAL: different versions = different outcomes)
        library_versions_signature = None
        library_versions = ctx.library_versions
        if not library_versions:
            # Try to get from resolved_metadata (metadata.json), additional_data, or run_data
            if resolved_metadata and 'environment' in resolved_metadata and 'library_versions' in resolved_metadata['environment']:
                library_versions = resolved_metadata['environment']['library_versions']
            elif additional_data and 'library_versions' in additional_data:
                library_versions = additional_data['library_versions']
            elif run_data.get('additional_data') and 'library_versions' in run_data.get('additional_data', {}):
                library_versions = run_data['additional_data']['library_versions']
            elif run_data.get('library_versions'):
                library_versions = run_data['library_versions']
        
        if library_versions and isinstance(library_versions, dict):
            # Sort keys for stable hash, include python_version if available
            lib_parts = []
            if ctx.python_version:
                lib_parts.append(f"python={ctx.python_version}")
            # Sort library versions for stable hash
            for key in sorted(library_versions.keys()):
                lib_parts.append(f"{key}={library_versions[key]}")
            if lib_parts:
                lib_str = "|".join(lib_parts)
                library_versions_signature = hashlib.sha256(lib_str.encode()).hexdigest()[:16]
        
        # Build comparison group (using resolved context, stage-aware)
        # CRITICAL: Pass symbol and universe_sig for proper comparison scoping
        # NEW: Pass run_identity for authoritative signatures
        comparison_group = self._build_comparison_group_from_context(
            stage, ctx, config_fp, data_fp, target_fp, hyperparameters_signature, train_seed, library_versions_signature,
            symbol=symbol, universe_sig=universe_sig, run_identity=run_identity
        )
        
        # Normalize inputs (using resolved context - no nulls for required fields)
        inputs = self._normalize_inputs_from_context(stage, ctx)
        
        # Normalize process (using resolved context)
        process = self._normalize_process_from_context(ctx)
        
        # Normalize outputs
        outputs = self._normalize_outputs(run_data, additional_data, cohort_dir, resolved_metadata)
        
        # CRITICAL: Compute output digests for artifact/metric reproducibility verification
        # These enable comparison of outputs across reruns for reproducibility tracking
        metrics_sha256 = self._compute_metrics_digest(outputs, resolved_metadata, cohort_dir)
        artifacts_manifest_sha256 = self._compute_artifacts_manifest_digest(cohort_dir, stage)
        # NEW: Pass prediction_fingerprint for authoritative prediction hash
        predictions_sha256 = self._compute_predictions_digest(cohort_dir, stage, prediction_fingerprint)
        
        # Build fingerprint source descriptions (for auditability)
        fingerprint_sources = {}
        if ctx.fold_assignment_hash:
            fingerprint_sources['fold_assignment_hash'] = (
                (additional_data.get('fold_assignment_hash_source') if additional_data else None) or
                "hash over row_id→fold_id mapping"
            )
        
        return NormalizedSnapshot(
            run_id=run_id,
            timestamp=timestamp,
            stage=stage,
            view=view,
            target=target,
            symbol=symbol,
            config_fingerprint=config_fp,
            data_fingerprint=data_fp,
            feature_fingerprint=feature_fp,
            target_fingerprint=target_fp,
            metrics_sha256=metrics_sha256,
            artifacts_manifest_sha256=artifacts_manifest_sha256,
            predictions_sha256=predictions_sha256,
            fingerprint_sources=fingerprint_sources,
            inputs=inputs,
            process=process,
            outputs=outputs,
            comparison_group=comparison_group
        )
    
    def _compute_config_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext,
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute config fingerprint from resolved context."""
        config_parts = []
        
        # Strategy and model_family (if applicable)
        if ctx.trainer_strategy:
            config_parts.append(f"strategy={ctx.trainer_strategy}")
        if ctx.model_family:
            config_parts.append(f"model_family={ctx.model_family}")
        if ctx.n_features is not None:
            config_parts.append(f"n_features={ctx.n_features}")
        if ctx.min_cs is not None:
            config_parts.append(f"min_cs={ctx.min_cs}")
        if ctx.max_cs_samples is not None:
            config_parts.append(f"max_cs_samples={ctx.max_cs_samples}")
        
        # Split protocol signature
        split_parts = []
        if ctx.cv_method:
            split_parts.append(f"cv_method={ctx.cv_method}")
        if ctx.folds is not None:
            split_parts.append(f"folds={ctx.folds}")
        if ctx.purge_minutes is not None:
            split_parts.append(f"purge_minutes={ctx.purge_minutes}")
        if ctx.embargo_minutes is not None:
            normalized_embargo = self._normalize_value_for_hash(ctx.embargo_minutes)
            split_parts.append(f"embargo_minutes={normalized_embargo}")
        if ctx.leakage_filter_version:
            split_parts.append(f"leakage_filter_version={ctx.leakage_filter_version}")
        if ctx.horizon_minutes is not None:
            split_parts.append(f"horizon_minutes={ctx.horizon_minutes}")
        if ctx.split_seed is not None:
            split_parts.append(f"split_seed={ctx.split_seed}")
        if ctx.fold_assignment_hash:
            split_parts.append(f"fold_assignment_hash={ctx.fold_assignment_hash}")
        
        if split_parts:
            split_str = "|".join(sorted(split_parts))
            config_parts.append(f"split={hashlib.sha256(split_str.encode()).hexdigest()[:8]}")
        
        if config_parts:
            config_str = "|".join(sorted(config_parts))
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_config_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute config fingerprint.
        
        Includes:
        - Strategy, model_family, n_features, min_cs, max_cs_samples
        - Split protocol signature (CV scheme, folds, purge/embargo, leakage guards)
        """
        config_parts = []
        
        # Extract config-relevant fields
        if additional_data:
            for key in ['strategy', 'model_family', 'n_features', 'min_cs', 'max_cs_samples']:
                if key in additional_data:
                    config_parts.append(f"{key}={additional_data[key]}")
            
            # Split protocol signature (CV scheme, folds, purge/embargo, leakage guards)
            # CRITICAL: Include split_seed (if fold assignment depends on seed) but NOT train_seed
            # CRITICAL: Include fold_assignment_hash (from actual fold IDs, not just seed)
            #           This ensures fold logic changes break comparability even if seed stays same
            split_parts = []
            for key in ['cv_method', 'folds', 'purge_minutes', 'embargo_minutes', 
                       'leakage_filter_version', 'horizon_minutes', 'split_seed', 'fold_assignment_hash']:
                if key in additional_data:
                    val = additional_data[key]
                    if val is not None:
                        # Normalize value for stable hashing
                        normalized_val = self._normalize_value_for_hash(val)
                        split_parts.append(f"{key}={normalized_val}")
            
            if split_parts:
                split_str = "|".join(sorted(split_parts))
                config_parts.append(f"split={hashlib.sha256(split_str.encode()).hexdigest()[:8]}")
        
        if run_data.get('additional_data'):
            for key in ['strategy', 'model_family']:
                if key in run_data['additional_data']:
                    config_parts.append(f"{key}={run_data['additional_data'][key]}")
        
        if config_parts:
            # Canonicalize: sorted keys, normalized values
            config_str = "|".join(sorted(config_parts))
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return None
    
    def _normalize_value_for_hash(self, val: Any) -> str:
        """Normalize value for stable hashing.
        
        Ensures:
        - Dicts are sorted by key
        - Lists preserve order (only sort when explicitly unordered, e.g., feature sets)
        - Floats use repr() for exact representation (avoids 1e-7 vs 0.0 collapse)
        - NaN/inf/-0.0 handled reproducibly
        - None/null are handled consistently
        """
        if val is None:
            return "None"
        elif isinstance(val, (int, str, bool)):
            return str(val)
        elif isinstance(val, float):
            # Use repr() for exact representation (avoids precision loss)
            # Handle special cases reproducibly
            if np.isnan(val):
                return "nan"
            elif np.isinf(val):
                return "inf" if val > 0 else "-inf"
            elif val == 0.0:
                # Distinguish -0.0 from 0.0
                return "0.0" if str(val) == "0.0" else "-0.0"
            else:
                # Use repr() to preserve exact value (e.g., 1e-7 stays 1e-7, not 0.0)
                return repr(val)
        elif isinstance(val, (list, tuple)):
            # CRITICAL: Preserve order by default (order may be semantic)
            # Only sort when explicitly marked as unordered (e.g., feature sets)
            # For now, preserve order - caller should sort feature lists before hashing
            normalized = [self._normalize_value_for_hash(v) for v in val]
            return "[" + ",".join(normalized) + "]"
        elif isinstance(val, dict):
            # Sort by key (dicts are unordered by definition)
            normalized = {k: self._normalize_value_for_hash(v) for k, v in val.items()}
            sorted_items = sorted(normalized.items())
            return "{" + ",".join(f"{k}:{v}" for k, v in sorted_items) + "}"
        else:
            return str(val)
    
    def _compute_data_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Optional[str]:
        """Compute data fingerprint from resolved context."""
        data_parts = []
        
        # Data parameters (all required, so should be non-null)
        if ctx.n_symbols is not None:
            data_parts.append(f"n_symbols={ctx.n_symbols}")
        if ctx.date_start:
            data_parts.append(f"date_start={ctx.date_start}")
        if ctx.date_end:
            data_parts.append(f"date_end={ctx.date_end}")
        if ctx.min_cs is not None:
            data_parts.append(f"min_cs={ctx.min_cs}")
        if ctx.max_cs_samples is not None:
            data_parts.append(f"max_cs_samples={ctx.max_cs_samples}")
        
        # Data identity (if available)
        if ctx.data_fingerprint:
            data_parts.append(f"data_id={ctx.data_fingerprint}")
        
        if data_parts:
            data_str = "|".join(sorted(data_parts))
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_data_fingerprint(
        self,
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute data fingerprint.
        
        Includes:
        - Data parameters (n_symbols, date_range, min_cs, max_cs_samples)
        - Data identity (if available: row IDs hash, file manifest, parquet metadata)
        """
        data_parts = []
        
        if cohort_metadata:
            # Extract data-relevant fields
            for key in ['n_symbols', 'date_start', 'date_end', 'min_cs', 'max_cs_samples']:
                if key in cohort_metadata:
                    val = cohort_metadata[key]
                    if val is not None:
                        data_parts.append(f"{key}={val}")
            
            # Data identity (actual data fingerprint if available)
            if 'data_fingerprint' in cohort_metadata:
                data_parts.append(f"data_id={cohort_metadata['data_fingerprint']}")
            elif 'data_hash' in cohort_metadata:
                data_parts.append(f"data_id={cohort_metadata['data_hash']}")
        
        if additional_data:
            for key in ['n_symbols', 'date_range']:
                if key in additional_data:
                    val = additional_data[key]
                    if val is not None:
                        data_parts.append(f"{key}={val}")
            
            # Data identity from additional_data
            if 'data_fingerprint' in additional_data:
                data_parts.append(f"data_id={additional_data['data_fingerprint']}")
        
        if data_parts:
            # Canonicalize: sorted keys, normalized values
            data_str = "|".join(sorted(data_parts))
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_feature_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Optional[str]:
        """Compute feature fingerprint from resolved context."""
        feature_parts = []
        
        # Feature count (if available)
        if ctx.n_features is not None:
            feature_parts.append(f"n_features={ctx.n_features}")
        
        # Feature names (if available)
        if ctx.feature_names:
            # Sort feature names for stable hash (features are unordered set)
            feature_list_str = "|".join(sorted(ctx.feature_names))
            feature_parts.append(f"names_hash={hashlib.sha256(feature_list_str.encode()).hexdigest()[:8]}")
        
        # Feature pipeline signature
        if ctx.feature_pipeline_signature:
            feature_parts.append(f"pipeline={ctx.feature_pipeline_signature}")
        
        if feature_parts:
            features_str = "|".join(sorted(feature_parts))
            return hashlib.sha256(features_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_feature_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute feature fingerprint.
        
        Includes:
        - Feature count and names (if available)
        - Feature pipeline signature (transforms, lookbacks, normalization, winsorization, missing-value policy)
        """
        feature_parts = []
        
        # Feature count and names
        if additional_data:
            if 'n_features' in additional_data:
                feature_parts.append(f"count={additional_data['n_features']}")
            if 'feature_names' in additional_data:
                # Hash actual feature list for precise matching
                feature_list = additional_data['feature_names']
                if isinstance(feature_list, list):
                    feature_list_str = "|".join(sorted(feature_list))
                    feature_parts.append(f"names_hash={hashlib.sha256(feature_list_str.encode()).hexdigest()[:8]}")
            # Feature pipeline signature
            if 'feature_pipeline_hash' in additional_data:
                feature_parts.append(f"pipeline={additional_data['feature_pipeline_hash']}")
            elif 'feature_transforms' in additional_data:
                # Hash transform config (normalization, winsorization, missing-value policy, lookbacks)
                transforms = additional_data['feature_transforms']
                if isinstance(transforms, dict):
                    transforms_str = "|".join(f"{k}={v}" for k, v in sorted(transforms.items()))
                    feature_parts.append(f"transforms={hashlib.sha256(transforms_str.encode()).hexdigest()[:8]}")
        
        if run_data.get('additional_data'):
            if 'n_features' in run_data['additional_data']:
                feature_parts.append(f"count={run_data['additional_data']['n_features']}")
            if 'feature_names' in run_data['additional_data']:
                feature_list = run_data['additional_data']['feature_names']
                if isinstance(feature_list, list):
                    feature_list_str = "|".join(sorted(feature_list))
                    feature_parts.append(f"names_hash={hashlib.sha256(feature_list_str.encode()).hexdigest()[:8]}")
        
        if feature_parts:
            # Canonicalize: sorted keys, normalized values
            features_str = "|".join(sorted(feature_parts))
            return hashlib.sha256(features_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_target_fingerprint_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Optional[str]:
        """Compute target fingerprint from resolved context."""
        target_parts = []
        
        # Target name (from resolved context)
        if ctx.target:
            target_parts.append(f"target={ctx.target}")
        
        # Labeling implementation signature (from resolved context)
        if ctx.labeling_impl_hash:
            target_parts.append(f"labeling={ctx.labeling_impl_hash}")
        
        # Horizon and objective (from resolved context)
        if ctx.horizon_minutes is not None:
            target_parts.append(f"horizon={ctx.horizon_minutes}")
        if ctx.objective:
            target_parts.append(f"objective={ctx.objective}")
        
        if target_parts:
            # Canonicalize: sorted keys, normalized values
            target_str = "|".join(sorted(target_parts))
            return hashlib.sha256(target_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_target_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute target fingerprint.
        
        Includes:
        - Target name
        - Labeling implementation signature (labeling code/config hash, not just target name)
        """
        target_parts = []
        
        # Target name
        target = None
        if additional_data and 'target' in additional_data:
            target = additional_data['target']
        elif run_data.get('target'):
            # Extract target from target (format: "target:family" or "target:symbol:family")
            parts = run_data['target'].split(':')
            target = parts[0] if parts else None
        
        if target:
            target_parts.append(f"target={target}")
        
        # Labeling implementation signature (code/config hash)
        if additional_data:
            if 'labeling_hash' in additional_data:
                target_parts.append(f"labeling={additional_data['labeling_hash']}")
            elif 'labeling_config' in additional_data:
                # Hash labeling config (horizon, labeling rules, etc.)
                labeling_config = additional_data['labeling_config']
                if isinstance(labeling_config, dict):
                    labeling_str = "|".join(f"{k}={v}" for k, v in sorted(labeling_config.items()))
                    target_parts.append(f"labeling={hashlib.sha256(labeling_str.encode()).hexdigest()[:8]}")
            # Horizon and objective
            if 'horizon_minutes' in additional_data:
                target_parts.append(f"horizon={additional_data['horizon_minutes']}")
            if 'objective' in additional_data:
                target_parts.append(f"objective={additional_data['objective']}")
        
        if target_parts:
            # Canonicalize: sorted keys, normalized values
            target_str = "|".join(sorted(target_parts))
            return hashlib.sha256(target_str.encode()).hexdigest()[:16]
        return None
    
    def _build_comparison_group_from_context(
        self,
        stage: str,
        ctx: ResolvedRunContext,
        config_fp: Optional[str],
        data_fp: Optional[str],
        task_fp: Optional[str],
        hyperparameters_signature: Optional[str] = None,
        train_seed: Optional[int] = None,
        library_versions_signature: Optional[str] = None,
        symbol: Optional[str] = None,
        universe_sig: Optional[str] = None,
        run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object
    ) -> ComparisonGroup:
        """Build comparison group from resolved context (stage-aware).

        CRITICAL: Only includes fields that are relevant for the stage.
        - TARGET_RANKING: Does NOT include model_family or feature_signature (not applicable)
        - FEATURE_SELECTION: Includes feature_signature but NOT model_family
        - TRAINING: Includes both model_family and feature_signature

        CRITICAL: symbol and universe_sig are required for proper comparison scoping:
        - symbol: For SS runs, ensures AAPL only compares to AAPL (not AVGO)
        - universe_sig: For CS runs, ensures same symbol set comparisons
        
        NEW: If run_identity is provided (RunIdentity SST object), use its signatures
        instead of the fallback values from ResolvedRunContext.

        This prevents storing null placeholders for fields that aren't stage-relevant.
        """
        # Routing signature from view (SST) - use view if available, fallback to view
        routing_signature = None
        view_for_fingerprint = None
        try:
            from TRAINING.orchestration.utils.run_context import get_view
            if hasattr(self, '_repro_base_dir'):
                view_for_fingerprint = get_view(self._repro_base_dir)
        except Exception:
            pass
        
        # Use view (SST) if available, otherwise fallback to view
        mode_for_signature = view_for_fingerprint if view_for_fingerprint else ctx.view
        if mode_for_signature:
            routing_signature = hashlib.sha256(mode_for_signature.encode()).hexdigest()[:16]
        
        # Stage-specific fields
        model_family = None
        feature_signature = None

        # Stage-specific fields
        hp_sig = None
        seed = None
        lib_sig = None  # Initialize for all stages

        # NEW: If run_identity is provided, use its signatures (SST source of truth)
        if run_identity is not None:
            # Extract signatures from RunIdentity if available
            if hasattr(run_identity, 'feature_signature') and run_identity.feature_signature:
                feature_signature = run_identity.feature_signature
            if hasattr(run_identity, 'hparams_signature') and run_identity.hparams_signature:
                hp_sig = run_identity.hparams_signature
            if hasattr(run_identity, 'train_seed') and run_identity.train_seed is not None:
                seed = run_identity.train_seed
            if hasattr(run_identity, 'dataset_signature') and run_identity.dataset_signature:
                # Override data_fp with the authoritative dataset_signature
                data_fp = run_identity.dataset_signature
            if hasattr(run_identity, 'target_signature') and run_identity.target_signature:
                task_fp = run_identity.target_signature
            if hasattr(run_identity, 'routing_signature') and run_identity.routing_signature:
                routing_signature = run_identity.routing_signature
            # Model family from context (not in RunIdentity)
            if stage == "TRAINING":
                model_family = ctx.model_family
        else:
            # Fallback to old logic when run_identity not provided
            if stage == "TARGET_RANKING":
                # TARGET_RANKING: model_family, feature_signature, hyperparameters, train_seed, and library_versions are NOT applicable
                # Don't include them (not None, just omit from ComparisonGroup)
                pass
            elif stage == "FEATURE_SELECTION":
                # FEATURE_SELECTION: feature_signature, hyperparameters, train_seed, and library_versions are required
                # CRITICAL: Different hyperparameters/seeds/versions in feature selection models = different features selected
                feature_signature = ctx.feature_fingerprint
                hp_sig = hyperparameters_signature
                seed = train_seed
                lib_sig = library_versions_signature
            elif stage == "TRAINING":
                # TRAINING: model_family, feature_signature, hyperparameters, train_seed, and library_versions are all required
                model_family = ctx.model_family
                feature_signature = ctx.feature_fingerprint
                hp_sig = hyperparameters_signature
                seed = train_seed
                lib_sig = library_versions_signature
        
        # Extract split_signature from run_identity or context
        split_sig = None
        if run_identity is not None and hasattr(run_identity, 'split_signature') and run_identity.split_signature:
            split_sig = run_identity.split_signature
        elif ctx.split_protocol_fingerprint:
            split_sig = ctx.split_protocol_fingerprint
        elif ctx.fold_assignment_hash:
            # Fallback: use fold_assignment_hash as split signature
            split_sig = ctx.fold_assignment_hash
        
        comparison_group = ComparisonGroup(
            experiment_id=ctx.experiment_id,  # Can be None if not tracked
            dataset_signature=data_fp,
            task_signature=task_fp,
            routing_signature=routing_signature,
            split_signature=split_sig,  # CRITICAL: CV split identity (required for all stages)
            n_effective=ctx.n_effective,  # CRITICAL: Exact sample size (required, non-null)
            model_family=model_family,  # Stage-specific: None for TARGET_RANKING/FEATURE_SELECTION
            feature_signature=feature_signature,  # Stage-specific: None for TARGET_RANKING
            hyperparameters_signature=hp_sig,  # Stage-specific: Only for FEATURE_SELECTION and TRAINING
            train_seed=seed,  # Stage-specific: Only for FEATURE_SELECTION and TRAINING
            library_versions_signature=lib_sig,  # Stage-specific: Only for FEATURE_SELECTION and TRAINING
            universe_sig=universe_sig,  # CRITICAL: For CS, ensures same symbol set comparisons
            symbol=symbol  # CRITICAL: For SS, ensures AAPL only compares to AAPL
        )
        
        # Validate in strict mode
        from TRAINING.common.determinism import is_strict_mode
        if is_strict_mode():
            comparison_group.validate(stage, strict=True)  # Will raise if invalid
        
        return comparison_group
    
    def _build_comparison_group(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]],
        config_fp: Optional[str],
        data_fp: Optional[str],
        task_fp: Optional[str]
    ) -> ComparisonGroup:
        """Build comparison group.
        
        CRITICAL: Includes ALL outcome-influencing metadata to ensure only runs with
        EXACTLY the same configuration are compared and stored together.
        
        Outcome-influencing metadata includes:
        - Exact n_effective (sample size) - 5k vs 5k, not 5k vs 10k
        - Dataset (universe, date range, min_cs, max_cs_samples)
        - Task (target, horizon, objective)
        - Routing/view configuration
        - Model family (different families = different outcomes)
        - Feature set (different features = different outcomes)
        - Hyperparameters (different HPs = different outcomes)
        - Train seed (different seeds = different outcomes)
        """
        # Extract experiment_id if available
        experiment_id = None
        if additional_data and 'experiment_id' in additional_data:
            experiment_id = additional_data['experiment_id']
        elif run_data.get('additional_data') and 'experiment_id' in run_data['additional_data']:
            experiment_id = run_data['additional_data']['experiment_id']
        
        # Routing signature from view
        routing_signature = None
        if additional_data and 'view' in additional_data:
            routing_signature = hashlib.sha256(additional_data['view'].encode()).hexdigest()[:16]
        
        # Extract exact n_effective (CRITICAL: must match exactly for comparison)
        n_effective = None
        if cohort_metadata and 'n_effective_cs' in cohort_metadata:
            n_effective = int(cohort_metadata['n_effective_cs'])
        elif additional_data and 'n_effective_cs' in additional_data:
            n_effective = int(additional_data['n_effective_cs'])
        elif run_data.get('metrics') and 'n_effective_cs' in run_data['metrics']:
            n_effective = int(run_data['metrics']['n_effective_cs'])
        elif run_data.get('additional_data') and 'n_effective_cs' in run_data['additional_data']:
            n_effective = int(run_data['additional_data']['n_effective_cs'])
        
        # Extract model_family (CRITICAL: different families = different outcomes)
        model_family = None
        if additional_data and 'model_family' in additional_data:
            model_family = additional_data['model_family']
        elif run_data.get('additional_data') and 'model_family' in run_data['additional_data']:
            model_family = run_data['additional_data']['model_family']
        elif run_data.get('target'):
            # Extract from target (format: "target:family" or "target:symbol:family")
            parts = run_data['target'].split(':')
            if len(parts) >= 2:
                model_family = parts[-1]  # Last part is usually the family
        
        # Extract feature signature (CRITICAL: different features = different outcomes)
        feature_signature = None
        if additional_data and 'n_features' in additional_data:
            # Use feature fingerprint if available, otherwise hash n_features
            feature_fp = self._compute_feature_fingerprint(run_data, additional_data)
            if feature_fp:
                feature_signature = feature_fp
        elif run_data.get('additional_data') and 'n_features' in run_data['additional_data']:
            feature_fp = self._compute_feature_fingerprint(run_data, run_data.get('additional_data'))
            if feature_fp:
                feature_signature = feature_fp
        
        # Extract hyperparameters and train_seed (CRITICAL: different HPs/seeds = different outcomes)
        hyperparameters_signature = None
        train_seed = None
        
        # Try to get hyperparameters from additional_data or run_data
        training_data = {}
        if additional_data and 'training' in additional_data:
            training_data = additional_data['training']
        elif run_data.get('additional_data') and 'training' in run_data.get('additional_data', {}):
            training_data = run_data['additional_data']['training']
        elif run_data.get('training'):
            training_data = run_data['training']
        
        # Extract hyperparameters (exclude model_family, strategy, seeds - those are handled separately)
        hyperparameters = {}
        excluded_keys = {'model_family', 'strategy', 'split_seed', 'train_seed', 'seed'}
        for key, value in training_data.items():
            if key not in excluded_keys and value is not None:
                hyperparameters[key] = value
        
        # Compute hyperparameters signature if we have any
        if hyperparameters:
            # Sort keys for stable hash
            hp_str = "|".join(f"{k}={v}" for k, v in sorted(hyperparameters.items()))
            hyperparameters_signature = hashlib.sha256(hp_str.encode()).hexdigest()[:16]
        
        # Extract train_seed
        train_seed = (
            training_data.get('train_seed') or
            training_data.get('seed') or
            (additional_data.get('train_seed') if additional_data else None) or
            (additional_data.get('seed') if additional_data else None) or
            (run_data.get('train_seed')) or
            (run_data.get('seed'))
        )
        if train_seed is not None:
            try:
                train_seed = int(train_seed)
            except (ValueError, TypeError):
                train_seed = None
        
        # Extract library versions and compute signature (CRITICAL: different versions = different outcomes)
        library_versions_signature = None
        library_versions = None
        if additional_data and 'library_versions' in additional_data:
            library_versions = additional_data['library_versions']
        elif run_data.get('additional_data') and 'library_versions' in run_data.get('additional_data', {}):
            library_versions = run_data['additional_data']['library_versions']
        elif run_data.get('library_versions'):
            library_versions = run_data['library_versions']
        
        if library_versions and isinstance(library_versions, dict):
            # Sort keys for stable hash, include python_version if available
            lib_parts = []
            python_version = (
                (additional_data.get('python_version') if additional_data else None) or
                (run_data.get('python_version'))
            )
            if python_version:
                lib_parts.append(f"python={python_version}")
            # Sort library versions for stable hash
            for key in sorted(library_versions.keys()):
                lib_parts.append(f"{key}={library_versions[key]}")
            if lib_parts:
                lib_str = "|".join(lib_parts)
                library_versions_signature = hashlib.sha256(lib_str.encode()).hexdigest()[:16]
        
        return ComparisonGroup(
            experiment_id=experiment_id,
            dataset_signature=data_fp,
            task_signature=task_fp,
            routing_signature=routing_signature,
            n_effective=n_effective,  # CRITICAL: Exact sample size
            model_family=model_family,  # CRITICAL: Model family
            feature_signature=feature_signature,  # CRITICAL: Feature set
            hyperparameters_signature=hyperparameters_signature,  # CRITICAL: Different HPs = different outcomes
            train_seed=train_seed,  # CRITICAL: Different seeds = different outcomes
            library_versions_signature=library_versions_signature  # CRITICAL: Different versions = different outcomes
        )
    
    def _normalize_inputs_from_context(
        self,
        stage: str,
        ctx: ResolvedRunContext
    ) -> Dict[str, Any]:
        """Normalize inputs section from resolved context (no nulls for required fields)."""
        inputs = {}
        
        # Config (stage-specific - only include applicable fields)
        config = {}
        if ctx.min_cs is not None:
            config['min_cs'] = ctx.min_cs
        if ctx.max_cs_samples is not None:
            config['max_cs_samples'] = ctx.max_cs_samples
        if stage in ["FEATURE_SELECTION", "TRAINING"]:
            if ctx.n_features is not None:
                config['n_features'] = ctx.n_features
        if stage == "TRAINING":
            if ctx.model_family:
                config['model_family'] = ctx.model_family
            if ctx.trainer_strategy:
                config['strategy'] = ctx.trainer_strategy
        if config:
            inputs['config'] = config
        
        # Data (all required fields, should be non-null)
        inputs['data'] = {
            'n_symbols': ctx.n_symbols,  # Required, validated
            'date_start': ctx.date_start,  # Required, validated
            'date_end': ctx.date_end,  # Required, validated
        }
        if ctx.n_rows_total is not None:
            inputs['data']['n_rows_total'] = ctx.n_rows_total
        
        # Target (required, should be non-null)
        inputs['target'] = {
            'target': ctx.target,  # Required, validated
            'view': ctx.view,  # Required, validated
        }
        if ctx.horizon_minutes is not None:
            inputs['target']['horizon_minutes'] = ctx.horizon_minutes
        if ctx.labeling_impl_hash:
            inputs['target']['labeling_impl_hash'] = ctx.labeling_impl_hash
        
        # Feature set (stage-specific)
        if stage in ["FEATURE_SELECTION", "TRAINING"]:
            features = {}
            if ctx.n_features is not None:
                features['n_features'] = ctx.n_features
            if ctx.feature_fingerprint:
                features['feature_fingerprint'] = ctx.feature_fingerprint
            if ctx.feature_names:
                features['feature_names'] = ctx.feature_names
            if features:
                inputs['features'] = features
        
        return inputs
    
    def _normalize_inputs(
        self,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize inputs section."""
        inputs = {}
        
        # Config fingerprint tree
        if additional_data:
            inputs['config'] = {
                'strategy': additional_data.get('strategy'),
                'model_family': additional_data.get('model_family'),
                'n_features': additional_data.get('n_features'),
                'min_cs': additional_data.get('min_cs'),
                'max_cs_samples': additional_data.get('max_cs_samples')
            }
        
        # Data fingerprint tree
        if cohort_metadata:
            inputs['data'] = {
                'n_symbols': cohort_metadata.get('n_symbols'),
                'date_start': cohort_metadata.get('date_start'),
                'date_end': cohort_metadata.get('date_end'),
                'n_samples': cohort_metadata.get('n_samples')
            }
        
        # Target provenance
        target = None
        if additional_data and 'target' in additional_data:
            target = additional_data['target']
        elif run_data.get('target'):
            parts = run_data['target'].split(':')
            target = parts[0] if parts else None
        
        if target:
            inputs['target'] = {
                'target': target,
                'view': additional_data.get('view') if additional_data else None
            }
        
        # Feature set provenance
        if additional_data and 'n_features' in additional_data:
            inputs['features'] = {
                'n_features': additional_data['n_features'],
                'feature_fingerprint': self._compute_feature_fingerprint(run_data, additional_data)
            }
        
        return inputs
    
    def _normalize_process_from_context(
        self,
        ctx: ResolvedRunContext
    ) -> Dict[str, Any]:
        """Normalize process section from resolved context."""
        process = {}
        
        # Split integrity (required fields, should be non-null)
        split = {}
        if ctx.min_cs is not None:
            split['min_cs'] = ctx.min_cs
        if ctx.max_cs_samples is not None:
            split['max_cs_samples'] = ctx.max_cs_samples
        if ctx.cv_method:
            split['cv_method'] = ctx.cv_method
        if ctx.purge_minutes is not None:
            split['purge_minutes'] = ctx.purge_minutes
        if ctx.embargo_minutes is not None:
            split['embargo_minutes'] = ctx.embargo_minutes
        if ctx.split_seed is not None:
            split['split_seed'] = ctx.split_seed
        if ctx.fold_assignment_hash:
            split['fold_assignment_hash'] = ctx.fold_assignment_hash
        if split:
            process['split'] = split
        
        # Training regime (only for TRAINING stage)
        if ctx.trainer_strategy or ctx.model_family:
            training = {}
            if ctx.trainer_strategy:
                training['strategy'] = ctx.trainer_strategy
            if ctx.model_family:
                training['model_family'] = ctx.model_family
            if training:
                process['training'] = training
        
        # Environment (tracked but not outcome-influencing)
        environment = {}
        if ctx.python_version:
            environment['python_version'] = ctx.python_version
        if ctx.library_versions:
            environment['library_versions'] = ctx.library_versions
        if ctx.cuda_version:
            environment['cuda_version'] = ctx.cuda_version
        if environment:
            process['environment'] = environment
        
        return process
    
    def _normalize_process(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize process section."""
        process = {}
        
        # Split integrity
        if additional_data:
            process['split'] = {
                'min_cs': additional_data.get('min_cs'),
                'max_cs_samples': additional_data.get('max_cs_samples')
            }
        
        # Training regime
        if additional_data:
            process['training'] = {
                'strategy': additional_data.get('strategy'),
                'model_family': additional_data.get('model_family')
            }
        
        # Compute environment (if available)
        process['environment'] = {
            'python_version': None,  # Could extract from sys.version
            'library_versions': {}  # Could extract from package versions
        }
        
        return process
    
    def _normalize_outputs(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]],
        cohort_dir: Optional[Path] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize outputs section.
        
        Extracts all numeric metrics from:
        1. run_data['metrics'] (if available)
        2. resolved_metadata['metrics'] (if available)
        3. metrics.json/parquet files in cohort_dir (most common for TARGET_RANKING/FEATURE_SELECTION)
        
        This ensures we capture all metrics for proper delta computation.
        """
        outputs = {}
        metrics_data = {}
        
        # Try run_data first
        if run_data.get('metrics'):
            metrics_data = run_data['metrics']
        
        # Check resolved_metadata if run_data doesn't have metrics
        if not metrics_data and resolved_metadata:
            metrics_data = resolved_metadata.get('metrics', {})
        
        # SST Architecture: Read metrics from canonical location (cohort_dir) first
        # This is the most common case for TARGET_RANKING/FEATURE_SELECTION stages
        if not metrics_data and cohort_dir:
            cohort_path = Path(cohort_dir)
            
            # 1. Try canonical metrics.parquet in cohort directory (SST)
            metrics_parquet = cohort_path / "metrics.parquet"
            if metrics_parquet.exists():
                try:
                    import pandas as pd
                    df_metrics = pd.read_parquet(metrics_parquet)
                    if len(df_metrics) > 0:
                        metrics_dict = df_metrics.iloc[0].to_dict()
                        # Extract all numeric metrics (exclude metadata fields)
                        metrics_data = {
                            k: v for k, v in metrics_dict.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode', 
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                            and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                        }
                        logger.debug(f"✅ Loaded metrics from canonical parquet: {metrics_parquet}")
                except Exception as e:
                    logger.debug(f"Failed to read metrics.parquet from {cohort_dir}: {e}")
            
            # 2. Fall back to metrics.json in cohort directory (debug export)
            if not metrics_data:
                metrics_json_file = cohort_path / "metrics.json"
                if metrics_json_file.exists():
                    try:
                        with open(metrics_json_file, 'r') as f:
                            metrics_json = json.load(f)
                        # Extract all numeric metrics (exclude metadata fields)
                        metrics_data = {
                            k: v for k, v in metrics_json.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                            and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                        }
                        logger.debug(f"✅ Loaded metrics from JSON: {metrics_json_file}")
                    except Exception as e:
                        logger.debug(f"Failed to read metrics.json from {cohort_dir}: {e}")
            
            # Fallback: Try target-first structure if metrics not found in cohort_dir
            if not metrics_data:
                try:
                    # Try to extract target from cohort_dir path or resolved_metadata
                    target = None
                    if resolved_metadata:
                        target = resolved_metadata.get('target')
                    
                    # If we can't get target from metadata, try to extract from path
                    if not target:
                        # Walk up from cohort_dir to find target
                        current = cohort_path
                        for _ in range(10):
                            if current.name.startswith('cohort='):
                                # Parent should be target directory
                                if current.parent.name not in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'FEATURE_SELECTION', 'TARGET_RANKING']:
                                    target = current.parent.name
                                    break
                            if not current.parent.exists():
                                break
                            current = current.parent
                    
                    if target:
                        # Find run directory
                        run_dir = cohort_path
                        for _ in range(10):
                            if (run_dir / "targets").exists() or run_dir.name == "RESULTS":
                                break
                            if not run_dir.parent.exists():
                                break
                            run_dir = run_dir.parent
                        
                        # Try target-first metrics location
                        target_metrics_dir = run_dir / "targets" / target / "metrics"
                        if target_metrics_dir.exists():
                            # Check for view-organized metrics
                            for item in target_metrics_dir.iterdir():
                                if item.is_dir() and item.name.startswith("view="):
                                    metrics_file = item / "metrics.json"
                                    if metrics_file.exists():
                                        try:
                                            with open(metrics_file, 'r') as f:
                                                metrics_json = json.load(f)
                                            metrics_data = {
                                                k: v for k, v in metrics_json.items()
                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                           'composite_version', 'leakage', 'leakage_flag']
                                                and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                                            }
                                            if metrics_data:
                                                break
                                        except Exception:
                                            pass
                                elif item.name == "metrics.json":
                                    try:
                                        with open(item, 'r') as f:
                                            metrics_json = json.load(f)
                                        metrics_data = {
                                            k: v for k, v in metrics_json.items()
                                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                       'composite_version', 'leakage', 'leakage_flag']
                                            and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                                        }
                                        if metrics_data:
                                            break
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.debug(f"Failed to read metrics from target-first structure: {e}")
        
        # Store all extracted metrics
        if metrics_data:
            outputs['metrics'] = metrics_data
        
        # Stability metrics (if available)
        if run_data.get('additional_data'):
            if 'stability' in run_data['additional_data']:
                outputs['stability'] = run_data['additional_data']['stability']
        
        return outputs
    
    def _compute_metrics_digest(
        self,
        outputs: Dict[str, Any],
        resolved_metadata: Optional[Dict[str, Any]],
        cohort_dir: Optional[Path]
    ) -> Optional[str]:
        """
        Compute SHA256 digest of metrics for reproducibility verification.
        
        This enables comparison of metrics across reruns with same inputs/process.
        
        Metrics can be in:
        1. outputs['metrics'] (from run_data)
        2. resolved_metadata['metrics'] (from metadata.json)
        3. metrics.json file in cohort_dir (most common for TARGET_RANKING/FEATURE_SELECTION)
        """
        metrics_data = outputs.get('metrics', {})
        
        # Check resolved_metadata for metrics if outputs.metrics is empty
        if not metrics_data and resolved_metadata:
            metrics_data = resolved_metadata.get('metrics', {})
        
        # SST Architecture: Read metrics from canonical location (cohort_dir) first
        # Then fall back to reference pointers, then legacy locations
        if not metrics_data and cohort_dir:
            cohort_path = Path(cohort_dir)
            
            # 1. Try canonical metrics.parquet in cohort directory (SST)
            metrics_parquet = cohort_path / "metrics.parquet"
            if metrics_parquet.exists():
                try:
                    import pandas as pd
                    df_metrics = pd.read_parquet(metrics_parquet)
                    if len(df_metrics) > 0:
                        metrics_dict = df_metrics.iloc[0].to_dict()
                        # Extract key metrics (exclude diff_telemetry, run_id, timestamp, and other metadata for stable hash)
                        metrics_data = {
                            k: v for k, v in metrics_dict.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode', 
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                        }
                        logger.debug(f"✅ Loaded metrics from canonical parquet: {metrics_parquet}")
                except Exception as e:
                    logger.debug(f"Failed to read metrics.parquet from {cohort_dir}: {e}")
            
            # 2. Fall back to metrics.json in cohort directory (debug export)
            if not metrics_data:
                metrics_json_file = cohort_path / "metrics.json"
                if metrics_json_file.exists():
                    try:
                        with open(metrics_json_file, 'r') as f:
                            metrics_json = json.load(f)
                        metrics_data = {
                            k: v for k, v in metrics_json.items()
                            if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                       'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                       'composite_version', 'leakage', 'leakage_flag']
                        }
                        logger.debug(f"✅ Loaded metrics from JSON: {metrics_json_file}")
                    except Exception as e:
                        logger.debug(f"Failed to read metrics.json from {cohort_dir}: {e}")
            
            # 3. Fallback to reference pointer in metrics/ directory
            if not metrics_data:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import get_target_metrics_dir
                    # Try to extract target from path
                    target = None
                    current = cohort_path
                    for _ in range(10):
                        if current.name.startswith('cohort='):
                            # Walk up to find target
                            parent = current.parent
                            if parent.name in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC']:
                                parent = parent.parent
                            if parent.name not in ['reproducibility', 'CROSS_SECTIONAL', 'SYMBOL_SPECIFIC']:
                                target = parent.name
                                break
                        if not current.parent.exists():
                            break
                        current = current.parent
                    
                    if target:
                        # Find base output directory
                        base_output_dir = cohort_path
                        for _ in range(10):
                            if (base_output_dir / "targets").exists():
                                break
                            if not base_output_dir.parent.exists():
                                break
                            base_output_dir = base_output_dir.parent
                        
                        if (base_output_dir / "targets").exists():
                            target_clean = target.replace('/', '_').replace('\\', '_')
                            target_metrics_dir = get_target_metrics_dir(base_output_dir, target_clean)
                            
                            # Try to find view from path
                            view = None
                            if 'CROSS_SECTIONAL' in cohort_path.parts:
                                view = 'CROSS_SECTIONAL'
                            elif 'SYMBOL_SPECIFIC' in cohort_path.parts:
                                view = 'SYMBOL_SPECIFIC'
                            
                            if view:
                                view_metrics_dir = target_metrics_dir / f"view={view}"
                                ref_file = view_metrics_dir / "latest_ref.json"
                                if ref_file.exists():
                                    try:
                                        with open(ref_file, 'r') as f:
                                            ref_data = json.load(f)
                                        canonical_path = Path(ref_data.get("canonical_path", ""))
                                        if canonical_path.exists():
                                            from TRAINING.common.utils.metrics import MetricsWriter
                                            metrics_dict = MetricsWriter.export_metrics_json_from_parquet(canonical_path)
                                            metrics_data = {
                                                k: v for k, v in metrics_dict.items()
                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                           'composite_version', 'leakage', 'leakage_flag']
                                            }
                                            logger.debug(f"✅ Loaded metrics via reference pointer: {canonical_path}")
                                    except Exception as e:
                                        logger.debug(f"Failed to follow reference pointer: {e}")
                except Exception as e:
                    logger.debug(f"Failed to load metrics via reference: {e}")
            
            # 4. Last resort: try legacy locations for backward compatibility
            if not metrics_data:
                try:
                    from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                    metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir)
                    if metrics_dir:
                        legacy_parquet = metrics_dir / "metrics.parquet"
                        if legacy_parquet.exists():
                            import pandas as pd
                            df_metrics = pd.read_parquet(legacy_parquet)
                            if len(df_metrics) > 0:
                                metrics_dict = df_metrics.iloc[0].to_dict()
                                metrics_data = {
                                    k: v for k, v in metrics_dict.items()
                                    if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode', 
                                               'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                               'composite_version', 'leakage', 'leakage_flag']
                                }
                        elif (metrics_dir / "metrics.json").exists():
                            with open(metrics_dir / "metrics.json", 'r') as f:
                                metrics_json = json.load(f)
                                metrics_data = {
                                    k: v for k, v in metrics_json.items()
                                    if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                               'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                               'composite_version', 'leakage', 'leakage_flag']
                                }
                except Exception as e:
                    logger.debug(f"Failed to load metrics from legacy location: {e}")
        
        if not metrics_data:
            return None
        
        # Normalize metrics dict for stable hashing (sort keys, round floats)
        normalized = self._normalize_value_for_hash(metrics_data)
        json_str = json.dumps(normalized, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _compute_artifacts_manifest_digest(
        self,
        cohort_dir: Optional[Path],
        stage: str
    ) -> Optional[str]:
        """
        Compute SHA256 digest of artifacts manifest for reproducibility verification.
        
        This enables comparison of artifacts (feature_importances.parquet, etc.) across reruns.
        Creates a manifest of artifact files with their sizes and modification times.
        
        Artifacts are stored at the view level (one level up from cohort directory):
        - TARGET_RANKING: targets/{target}/reproducibility/{view}/feature_importances/, target_confidence.json
        - FEATURE_SELECTION: targets/{target}/reproducibility/{view}/feature_importances/, selected_features.txt, target_confidence.json
        - TRAINING: targets/{target}/reproducibility/{view}/cohort={cohort_id}/ (artifacts in cohort dir)
        """
        if not cohort_dir or not cohort_dir.exists():
            return None
        
        # For TARGET_RANKING and FEATURE_SELECTION, artifacts are at target level (one level up from cohort)
        # For TRAINING, artifacts are in the cohort directory itself
        if stage in ['TARGET_RANKING', 'FEATURE_SELECTION']:
            # Go up one level from cohort={cohort_id} to target level
            target_dir = cohort_dir.parent
        else:
            # TRAINING: artifacts are in cohort directory
            target_dir = cohort_dir
        
        # Define artifact file patterns by stage
        artifact_patterns = {
            'TARGET_RANKING': [
                ('target_confidence.json', target_dir),  # At target level
                ('feature_importances', target_dir / 'feature_importances')  # Directory with CSV files
            ],
            'FEATURE_SELECTION': [
                ('selected_features.txt', target_dir),  # At target level
                ('target_confidence.json', target_dir),  # At target level
                ('feature_importances', target_dir / 'feature_importances')  # Directory with CSV files
            ],
            'TRAINING': [
                ('model_hash.txt', cohort_dir),  # In cohort directory
                ('meta_*.json', cohort_dir)  # In cohort directory
            ]
        }
        
        patterns = artifact_patterns.get(stage, [])
        manifest = []
        
        for pattern, search_dir in patterns:
            if not search_dir or not search_dir.exists():
                continue
            
            if pattern == 'feature_importances':
                # Special case: hash all CSV files in feature_importances directory
                if search_dir.is_dir():
                    csv_files = sorted(search_dir.glob('*.csv'))
                    for csv_file in csv_files:
                        if csv_file.is_file():
                            # Hash file contents instead of using mtime (volatile)
                            try:
                                with open(csv_file, 'rb') as f:
                                    content_hash = hashlib.sha256(f.read()).hexdigest()
                                manifest.append({
                                    'path': f'feature_importances/{csv_file.name}',
                                    'content_sha256': content_hash
                                })
                            except Exception as e:
                                logger.debug(f"Failed to hash {csv_file}: {e}")
                                # Fallback to size only (no mtime)
                                stat = csv_file.stat()
                                manifest.append({
                                    'path': f'feature_importances/{csv_file.name}',
                                    'size': stat.st_size
                                })
            elif '*' in pattern:
                # Handle glob patterns
                matches = list(search_dir.glob(pattern))
                for match in sorted(matches):
                    if match.is_file():
                        # Hash file contents instead of using mtime (volatile)
                        try:
                            with open(match, 'rb') as f:
                                content_hash = hashlib.sha256(f.read()).hexdigest()
                            manifest.append({
                                'path': match.name,
                                'content_sha256': content_hash
                            })
                        except Exception as e:
                            logger.debug(f"Failed to hash {match}: {e}")
                            # Fallback to size only (no mtime)
                            stat = match.stat()
                            manifest.append({
                                'path': match.name,
                                'size': stat.st_size
                            })
            else:
                # Exact file match
                file_path = search_dir / pattern
                if file_path.exists() and file_path.is_file():
                    # Hash file contents instead of using mtime (volatile)
                    try:
                        with open(file_path, 'rb') as f:
                            content_hash = hashlib.sha256(f.read()).hexdigest()
                        manifest.append({
                            'path': pattern,
                            'content_sha256': content_hash
                        })
                    except Exception as e:
                        logger.debug(f"Failed to hash {file_path}: {e}")
                        # Fallback to size only (no mtime)
                        stat = file_path.stat()
                        manifest.append({
                            'path': pattern,
                            'size': stat.st_size
                        })
        
        if not manifest:
            return None
        
        # Sort manifest for stable hashing
        manifest_sorted = sorted(manifest, key=lambda x: x['path'])
        json_str = json.dumps(manifest_sorted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _compute_predictions_digest(
        self,
        cohort_dir: Optional[Path],
        stage: str,
        prediction_fingerprint: Optional[Dict] = None,  # NEW: Prediction fingerprint dict
    ) -> Optional[str]:
        """
        Compute SHA256 digest of predictions for reproducibility verification.

        This enables comparison of predictions across reruns.
        Currently, predictions may not be stored in cohort directories for all stages.
        
        NEW: If prediction_fingerprint is provided, use its prediction_hash directly.
        This is the authoritative source from the prediction hashing system.
        """
        # NEW: If prediction_fingerprint provided, use its hash
        if prediction_fingerprint and prediction_fingerprint.get('prediction_hash'):
            return prediction_fingerprint['prediction_hash']
        
        if not cohort_dir or not cohort_dir.exists():
            return None
        
        # Check for predictions files (if they exist)
        predictions_files = []
        for pattern in ['predictions.parquet', 'predictions.csv', 'predictions.json']:
            file_path = cohort_dir / pattern
            if file_path.exists() and file_path.is_file():
                # For large files, hash first N bytes + size instead of full content
                # This is a compromise for performance while still detecting changes
                stat = file_path.stat()
                if stat.st_size > 10 * 1024 * 1024:  # > 10MB
                    # Hash first 1MB + size + mtime
                    with open(file_path, 'rb') as f:
                        first_mb = f.read(1024 * 1024)
                    content_hash = hashlib.sha256(
                        first_mb + str(stat.st_size).encode() + str(stat.st_mtime).encode()
                    ).hexdigest()
                else:
                    # Hash full file for smaller files
                    with open(file_path, 'rb') as f:
                        content_hash = hashlib.sha256(f.read()).hexdigest()
                
                predictions_files.append({
                    'path': pattern,
                    'size': stat.st_size,
                    'hash': content_hash
                })
        
        if not predictions_files:
            return None
        
        # Sort for stable hashing
        predictions_sorted = sorted(predictions_files, key=lambda x: x['path'])
        json_str = json.dumps(predictions_sorted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def _normalize_value_for_hash(self, val: Any) -> Any:
        """Normalize value for hashing (round floats, sort lists, etc.)."""
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return round(val, 6)
        elif isinstance(val, (list, tuple)):
            try:
                return sorted([self._normalize_value_for_hash(v) for v in val])
            except TypeError:
                return [self._normalize_value_for_hash(v) for v in val]
        elif isinstance(val, dict):
            return {k: self._normalize_value_for_hash(v) for k, v in sorted(val.items())}
        else:
            return val
    
    def save_snapshot(
        self,
        snapshot: NormalizedSnapshot,
        cohort_dir: Path
    ) -> None:
        """
        Save normalized snapshot to cohort directory.
        
        Automatically organizes run by comparison group metadata on first snapshot.
        
        Args:
            snapshot: Normalized snapshot
            cohort_dir: Cohort directory (where metadata.json lives)
        """
        cohort_dir = Path(cohort_dir)
        
        # CRITICAL: Ensure we only write to target-first structure
        # If cohort_dir is in legacy REPRODUCIBILITY structure, find/create target-first equivalent
        target_cohort_dir = cohort_dir
        if "REPRODUCIBILITY" in str(cohort_dir):
            # Extract identifiers from cohort_dir path and create target-first path
            try:
                parts = Path(cohort_dir).parts
                stage = None
                view = None
                target = None
                cohort_id = None
                symbol_for_target = None
                
                for i, part in enumerate(parts):
                    if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING']:
                        stage = part
                        if i + 1 < len(parts) and parts[i+1] in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']:
                            view = parts[i+1]
                            if i + 2 < len(parts) and not parts[i+2].startswith('cohort='):
                                target = parts[i+2]
                        # Find cohort_id and symbol
                        for j in range(i, len(parts)):
                            if parts[j].startswith('cohort='):
                                cohort_id = parts[j].replace('cohort=', '')
                            elif parts[j].startswith('symbol='):
                                symbol_for_target = parts[j].replace('symbol=', '')
                            elif parts[j].startswith('model_family='):
                                # Skip model_family in path
                                continue
                
                # Only create target-first structure for TARGET_RANKING and FEATURE_SELECTION
                if stage in ['TARGET_RANKING', 'FEATURE_SELECTION'] and target and cohort_id:
                    # Find base output directory (run directory)
                    temp_dir = cohort_dir
                    for _ in range(10):  # Limit depth
                        if (temp_dir / "targets").exists() or (temp_dir.parent / "targets").exists():
                            if (temp_dir / "targets").exists():
                                base_output_dir = temp_dir
                            else:
                                base_output_dir = temp_dir.parent
                            break
                        if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                            break
                        temp_dir = temp_dir.parent
                    
                    if base_output_dir:
                        from TRAINING.orchestration.utils.target_first_paths import (
                            get_target_reproducibility_dir, ensure_target_structure
                        )
                        
                        # CRITICAL FIX: Prefer snapshot's view/symbol over path parsing
                        # This fixes CROSS_SECTIONAL vs SYMBOL_SPECIFIC path organization bug
                        snapshot_view = getattr(snapshot, 'view', None)
                        snapshot_symbol = getattr(snapshot, 'symbol', None)
                        
                        # Use snapshot's view if available, otherwise fallback to path-parsed view
                        view_for_target = snapshot_view if snapshot_view else view
                        
                        # Normalize view for FEATURE_SELECTION (INDIVIDUAL -> SYMBOL_SPECIFIC)
                        if stage == 'FEATURE_SELECTION' and view_for_target == 'INDIVIDUAL':
                            view_for_target = 'SYMBOL_SPECIFIC'
                        
                        # Use snapshot's symbol if available, otherwise fallback to path-parsed symbol
                        symbol_for_target_final = snapshot_symbol if snapshot_symbol else symbol_for_target
                        
                        # Ensure target structure exists
                        ensure_target_structure(base_output_dir, target)
                        
                        # Build target-first reproducibility path
                        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target)
                        if view_for_target == "SYMBOL_SPECIFIC" and symbol_for_target_final:
                            # Include symbol in path to prevent overwriting
                            target_cohort_dir = target_repro_dir / view_for_target / f"symbol={symbol_for_target_final}" / f"cohort={cohort_id}"
                        else:
                            target_cohort_dir = target_repro_dir / view_for_target / f"cohort={cohort_id}"
                        target_cohort_dir.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"✅ Created target-first cohort directory for snapshot: {target_cohort_dir}")
            except Exception as e:
                logger.warning(f"Failed to create target-first structure for snapshot: {e}")
                # Don't fall back to legacy path - fail instead to prevent REPRODUCIBILITY creation
                target_cohort_dir = None
        
        # Use target-first directory for all writes - NEVER use legacy REPRODUCIBILITY paths
        if target_cohort_dir is None:
            logger.error(f"❌ Cannot save snapshot: failed to create target-first structure and legacy paths are not allowed")
            return
        
        cohort_dir = target_cohort_dir
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # NOTE: Run organization by comparison group is now done at startup (config load time)
        # in IntelligentTrainer.__init__(). This ensures runs are organized by metadata from the start.
        # We no longer move runs here - they're already in the correct location.
        # Keeping this as a fallback for edge cases where startup organization didn't happen.
        if not self._run_moved and snapshot.comparison_group and self.run_dir and self.results_dir:
            # Only organize if run is in a sample size bin or _pending (not already organized)
            run_parent = self.run_dir.parent.name
            if run_parent.startswith("sample_") or run_parent == "_pending":
                self._organize_run_by_comparison_group(snapshot)
            else:
                # Already organized, mark as moved
                self._run_moved = True
        
        # Assign monotonic sequence number for correct ordering (concurrency-safe)
        # This ensures "prev run" selection is correct regardless of mtime/timestamp quirks
        # CRITICAL: Must be done under lock to prevent two concurrent writers from picking same seq
        if snapshot.snapshot_seq is None:
            # Use cohort-level lock file for sequence assignment
            cohort_lock_file = cohort_dir / ".snapshot_seq.lock"
            
            with open(cohort_lock_file, 'w') as lock_f:
                try:
                    # Acquire exclusive lock (blocks until available)
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                    
                    # Re-read snapshots to get latest sequence (another process may have updated it)
                    # Load from run-level snapshot index to get all snapshots for this run
                    run_snapshot_index = self.snapshot_index
                    if run_snapshot_index and run_snapshot_index.exists():
                        try:
                            with open(run_snapshot_index) as f:
                                index_data = json.load(f)
                                for key, snap_data in index_data.items():
                                    snap = self._deserialize_snapshot(snap_data)
                                    # Handle old formats:
                                    # - run_id:stage (legacy, 1 colon)
                                    # - run_id:stage:target:view (previous fix, 3 colons)
                                    # - run_id:stage:target:view:symbol (current format, 4 colons)
                                    if key.count(':') >= 4:
                                        # Current format: use key as-is
                                        if key not in self._snapshots:
                                            self._snapshots[key] = snap
                                    elif key.count(':') >= 3:
                                        # Previous format (run_id:stage:target:view): build new key with symbol
                                        target_clean = (snap.target or "unknown").replace('/', '_').replace('\\', '_')
                                        view_clean = snap.view or "UNKNOWN"
                                        symbol_clean = (snap.symbol or "NONE").replace('/', '_').replace('\\', '_')
                                        new_key = f"{snap.run_id}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}"
                                        if new_key not in self._snapshots:
                                            self._snapshots[new_key] = snap
                                    else:
                                        # Legacy format (run_id:stage): build new key with target, view, symbol
                                        target_clean = (snap.target or "unknown").replace('/', '_').replace('\\', '_')
                                        view_clean = snap.view or "UNKNOWN"
                                        symbol_clean = (snap.symbol or "NONE").replace('/', '_').replace('\\', '_')
                                        new_key = f"{snap.run_id}:{snap.stage}:{target_clean}:{view_clean}:{symbol_clean}"
                                        if new_key not in self._snapshots:
                                            self._snapshots[new_key] = snap
                        except Exception:
                            pass
                    
                    # Get next sequence number (max existing + 1, or 1 if none)
                    max_seq = 0
                    for snap in self._snapshots.values():
                        if snap.snapshot_seq and snap.snapshot_seq > max_seq:
                            max_seq = snap.snapshot_seq
                    snapshot.snapshot_seq = max_seq + 1
                    
                    # Lock is automatically released when file is closed
                except Exception as e:
                    # Release lock on error
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    raise
        
        # Save full snapshot atomically
        snapshot_file = cohort_dir / "snapshot.json"
        snapshot_dict = snapshot.to_dict()
        _write_atomic_json(snapshot_file, snapshot_dict)
        
        # Also write to target-first structure (for TARGET_RANKING and FEATURE_SELECTION stages)
        try:
            # Extract identifiers from cohort_dir path
            # Path structure: .../STAGE/VIEW/target/cohort=.../
            parts = Path(cohort_dir).parts
            stage = None
            view = None
            target = None
            cohort_id = None
            
            for i, part in enumerate(parts):
                if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING']:
                    stage = part
                    if i + 1 < len(parts) and parts[i+1] in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']:
                        view = parts[i+1]
                        if i + 2 < len(parts) and not parts[i+2].startswith('cohort='):
                            target = parts[i+2]
                    # Find cohort_id
                    for j in range(i, len(parts)):
                        if parts[j].startswith('cohort='):
                            cohort_id = parts[j].replace('cohort=', '')
                            break
                    break
            
            # Only create target-first structure for TARGET_RANKING and FEATURE_SELECTION
            if stage in ['TARGET_RANKING', 'FEATURE_SELECTION'] and target and cohort_id:
                # Find base output directory (run directory)
                temp_dir = cohort_dir
                for _ in range(10):  # Limit depth
                    if (temp_dir / "targets").exists() or (temp_dir.parent / "targets").exists():
                        # Found run directory
                        if (temp_dir / "targets").exists():
                            base_output_dir = temp_dir
                        else:
                            base_output_dir = temp_dir.parent
                        break
                    if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                        break
                    temp_dir = temp_dir.parent
                
                if base_output_dir:
                    from TRAINING.orchestration.utils.target_first_paths import (
                        get_target_reproducibility_dir, ensure_target_structure
                    )
                    
                    # CRITICAL FIX: Prefer snapshot's view/symbol over path parsing
                    snapshot_view = getattr(snapshot, 'view', None)
                    snapshot_symbol = getattr(snapshot, 'symbol', None)
                    
                    # Use snapshot's view if available, otherwise fallback to path-parsed view
                    view_for_target = snapshot_view if snapshot_view else view
                    
                    # Normalize view for FEATURE_SELECTION (INDIVIDUAL -> SYMBOL_SPECIFIC)
                    if stage == 'FEATURE_SELECTION' and view_for_target == 'INDIVIDUAL':
                        view_for_target = 'SYMBOL_SPECIFIC'
                    
                    # Use snapshot's symbol if available
                    symbol_for_target = snapshot_symbol
                    
                    # Ensure target structure exists
                    ensure_target_structure(base_output_dir, target)
                    
                    # Build target-first reproducibility path
                    target_repro_dir = get_target_reproducibility_dir(base_output_dir, target)
                    if view_for_target == "SYMBOL_SPECIFIC" and symbol_for_target:
                        # Include symbol in path to prevent overwriting
                        target_cohort_dir = target_repro_dir / view_for_target / f"symbol={symbol_for_target}" / f"cohort={cohort_id}"
                    else:
                        target_cohort_dir = target_repro_dir / view_for_target / f"cohort={cohort_id}"
                    target_cohort_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Write snapshot.json to target-first structure
                    target_snapshot_file = target_cohort_dir / "snapshot.json"
                    _write_atomic_json(target_snapshot_file, snapshot_dict)
                    logger.debug(f"✅ Also saved snapshot.json to target-first structure")
        except Exception as e:
            logger.debug(f"Failed to save snapshot.json to target-first structure (non-critical): {e}")
        
        # Update index (keyed by run_id:stage:target:view:symbol for uniqueness across targets and symbols)
        # Include target, view, and symbol in key to prevent overwrites when multiple targets/symbols are processed
        target_clean = (snapshot.target or "unknown").replace('/', '_').replace('\\', '_')
        view_clean = snapshot.view or "UNKNOWN"
        symbol_clean = (snapshot.symbol or "NONE").replace('/', '_').replace('\\', '_')
        snapshot_key = f"{snapshot.run_id}:{snapshot.stage}:{target_clean}:{view_clean}:{symbol_clean}"
        self._snapshots[snapshot_key] = snapshot
        self._save_indices()
        
        logger.debug(f"✅ Saved snapshot to {snapshot_file}")
    
    def _organize_run_by_comparison_group(self, snapshot: NormalizedSnapshot) -> None:
        """
        Move run directory to comparison group-based organization.
        
        Structure: RESULTS/{comparison_group_dir}/{run_name}/
        
        This ensures runs with exactly the same outcome-influencing metadata
        are stored together for human auditability.
        
        Args:
            snapshot: First snapshot with comparison group metadata
        """
        if not snapshot.comparison_group:
            return
        
        try:
            # Determine target directory based on comparison group
            # Structure: RESULTS/runs/{comparison_group_dir}/{run_name}/
            comparison_group_dir = snapshot.comparison_group.to_dir_name()
            runs_dir = self.results_dir / "runs"
            target_dir = runs_dir / comparison_group_dir / self.run_dir.name
            
            # Skip if already in correct location
            if self.run_dir.resolve() == target_dir.resolve():
                self._run_moved = True
                return
            
            # Skip if target already exists (another run with same comparison group)
            if target_dir.exists():
                logger.warning(f"⚠️  Target directory already exists: {target_dir}")
                logger.warning(f"   Run will remain in current location: {self.run_dir}")
                self._run_moved = True  # Mark as moved to prevent retry
                return
            
            # Create target directory parent
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            
            # Move the entire run directory
            import shutil
            logger.info(f"📁 Organizing run by comparison group metadata...")
            logger.info(f"   From: {self.run_dir}")
            logger.info(f"   To:   {target_dir}")
            logger.info(f"   Group: {snapshot.comparison_group.to_key(snapshot.stage, strict=False)}")
            
            shutil.move(str(self.run_dir), str(target_dir))
            
            # Update internal references
            self.run_dir = target_dir
            
            # Use target-first structure (globals/) instead of legacy REPRODUCIBILITY/
            # REPRODUCIBILITY should only exist within run directories, not at RESULTS root
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            globals_dir = get_globals_dir(target_dir)
            globals_dir.mkdir(parents=True, exist_ok=True)
            self.run_metrics_dir = globals_dir
            self.snapshot_index = self.run_metrics_dir / "snapshot_index.json"
            
            # Update output_dir if it was pointing to run_dir
            if self.output_dir.resolve() == self.run_dir.resolve():
                self.output_dir = target_dir
            
            self._run_moved = True
            logger.info(f"✅ Run organized by comparison group metadata")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to organize run by comparison group: {e}")
            logger.warning(f"   Run will remain in current location: {self.run_dir}")
            # Mark as moved to prevent retry loops
            self._run_moved = True
    
    def compute_diff(
        self,
        current_snapshot: NormalizedSnapshot,
        prev_snapshot: NormalizedSnapshot,
        prev_cohort_dir: Optional[Path] = None,
        curr_cohort_dir: Optional[Path] = None
    ) -> DiffResult:
        """
        Compute diff between two snapshots.
        
        Args:
            current_snapshot: Current run snapshot
            prev_snapshot: Previous run snapshot
        
        Returns:
            DiffResult
        """
        # Check comparability
        comparable, reason = self._check_comparability(current_snapshot, prev_snapshot)
        
        if not comparable:
            # CRITICAL: If not comparable, severity must be CRITICAL with reason
            return DiffResult(
                prev_run_id=prev_snapshot.run_id,
                current_run_id=current_snapshot.run_id,
                comparable=False,
                comparability_reason=reason,
                severity=ChangeSeverity.CRITICAL,
                severity_reason=f"Runs are not comparable: {reason}",
                excluded_factors_changed={},  # Empty but present (stable shape)
                summary={
                    'total_changes': 0,
                    'input_changes': 0,
                    'process_changes': 0,
                    'output_changes': 0,
                    'metric_deltas_count': 0,
                    'excluded_factors_changed': False,
                    'excluded_factors_summary': None
                }
            )
        
        # Compute changes
        changed_keys = []
        patch = []
        
        # Diff inputs
        input_changes = self._diff_dict(
            prev_snapshot.inputs,
            current_snapshot.inputs,
            prefix="inputs"
        )
        changed_keys.extend(input_changes['keys'])
        patch.extend(input_changes['patch'])
        
        # Diff process
        process_changes = self._diff_dict(
            prev_snapshot.process,
            current_snapshot.process,
            prefix="process"
        )
        changed_keys.extend(process_changes['keys'])
        patch.extend(process_changes['patch'])
        
        # Diff outputs (metrics)
        output_changes = self._diff_dict(
            prev_snapshot.outputs,
            current_snapshot.outputs,
            prefix="outputs"
        )
        changed_keys.extend(output_changes['keys'])
        patch.extend(output_changes['patch'])
        
        # Compute metric deltas
        metric_deltas, metric_deltas_total = self._compute_metric_deltas(
            prev_snapshot.outputs,
            current_snapshot.outputs
        )
        
        # Compute trend deltas (if cohort directories are provided)
        trend_deltas = {}
        if prev_cohort_dir and curr_cohort_dir:
            prev_metadata_path = Path(prev_cohort_dir) / "metadata.json"
            curr_metadata_path = Path(curr_cohort_dir) / "metadata.json"
            
            prev_trend = self._load_trend_from_metadata(prev_metadata_path)
            curr_trend = self._load_trend_from_metadata(curr_metadata_path)
            
            if prev_trend and curr_trend:
                trend_deltas = self._compute_trend_deltas(prev_trend, curr_trend)
        
        # CRITICAL: Check output digests for artifact/metric reproducibility
        # If digests differ, this indicates reproducibility variance even with same inputs/process
        output_digest_changes = []
        if prev_snapshot.metrics_sha256 != current_snapshot.metrics_sha256:
            output_digest_changes.append("metrics_sha256")
        if prev_snapshot.artifacts_manifest_sha256 != current_snapshot.artifacts_manifest_sha256:
            output_digest_changes.append("artifacts_manifest_sha256")
        if prev_snapshot.predictions_sha256 != current_snapshot.predictions_sha256:
            output_digest_changes.append("predictions_sha256")
        
        # Extract excluded factors (hyperparameters, seeds, versions) for reporting
        excluded_factors_changed = self._extract_excluded_factor_changes(
            current_snapshot, prev_snapshot
        )
        
        # Build summary with readable excluded factors summary
        excluded_summary = self._format_excluded_factors_summary(excluded_factors_changed)
        
        # Compute impact classification from metric deltas
        impact_label, top_regressions, top_improvements = self._classify_metric_impact(metric_deltas)
        
        # Count significant deltas (all entries in metric_deltas are significant by design)
        metric_deltas_significant = len(metric_deltas)
        
        # Compute trend direction change (if slope_per_day is available)
        trend_direction_change = None
        if trend_deltas and 'slope_per_day' in trend_deltas:
            prev_slope = trend_deltas['slope_per_day'].get('prev')
            curr_slope = trend_deltas['slope_per_day'].get('curr')
            if prev_slope is not None and curr_slope is not None:
                prev_improving = prev_slope > 0
                curr_improving = curr_slope > 0
                if prev_improving != curr_improving:
                    if prev_improving:
                        trend_direction_change = "improving→declining"
                    else:
                        trend_direction_change = "declining→improving"
        
        summary = {
            'total_changes': len(changed_keys),
            'input_changes': len(input_changes['keys']),
            'process_changes': len(process_changes['keys']),
            'output_changes': len(output_changes['keys']),
            'metric_deltas_total': metric_deltas_total,  # Total metrics compared
            'metric_deltas_count': metric_deltas_significant,  # Only significant deltas (backward compat)
            'metric_deltas_significant': metric_deltas_significant,  # Explicit significant count
            'excluded_factors_changed': bool(excluded_factors_changed),
            'excluded_factors_summary': excluded_summary,  # Human-readable summary
            'output_digest_changes': output_digest_changes,  # List of changed output digests
            'impact_label': impact_label,  # none|noise|minor|major
            'top_regressions': top_regressions,  # List of up to K metric keys with worst deltas (significant only)
            'top_improvements': top_improvements,  # List of up to K metric keys with best deltas (significant only)
            'trend_deltas_count': len(trend_deltas),  # Number of trend fields compared
            'trend_direction_change': trend_direction_change  # Direction change if slope sign changed
        }
        
        # CRITICAL: Determine severity purely from the report (SST-style)
        severity, severity_reason = self._determine_severity(
            changed_keys=changed_keys,
            input_changes=input_changes,
            process_changes=process_changes,
            output_changes=output_changes,
            metric_deltas=metric_deltas,
            excluded_factors_changed=excluded_factors_changed,
            excluded_factors_summary=excluded_summary,
            summary=summary
        )
        
        return DiffResult(
            prev_run_id=prev_snapshot.run_id,
            current_run_id=current_snapshot.run_id,
            comparable=True,
            prev_timestamp=prev_snapshot.timestamp,
            prev_snapshot_seq=prev_snapshot.snapshot_seq,
            prev_stage=prev_snapshot.stage,
            prev_view=prev_snapshot.view,
            comparison_source=getattr(prev_snapshot, '_comparison_source', None),  # Set by find_previous_comparable
            changed_keys=changed_keys,
            severity=severity,
            severity_reason=severity_reason,
            summary=summary,
            excluded_factors_changed=excluded_factors_changed,
            patch=patch,
            metric_deltas=metric_deltas,
            trend_deltas=trend_deltas
        )
    
    def _check_comparability(
        self,
        current: NormalizedSnapshot,
        prev: NormalizedSnapshot
    ) -> Tuple[bool, Optional[str]]:
        """Check if two snapshots are comparable."""
        # CRITICAL: Check fingerprint schema version compatibility
        if current.fingerprint_schema_version != prev.fingerprint_schema_version:
            return False, (
                f"Different fingerprint schema versions: "
                f"{current.fingerprint_schema_version} vs {prev.fingerprint_schema_version}. "
                f"Fingerprint computation changed - runs are not comparable."
            )
        
        # Must be same stage
        if current.stage != prev.stage:
            return False, f"Different stages: {current.stage} vs {prev.stage}"
        
        # Must be same view
        if current.view != prev.view:
            return False, f"Different views: {current.view} vs {prev.view}"
        
        # Must be same target (if specified)
        if current.target and prev.target and current.target != prev.target:
            return False, f"Different targets: {current.target} vs {prev.target}"
        
        # CRITICAL: For SYMBOL_SPECIFIC view, must be same symbol
        # This is defense-in-depth (comparison_group.to_key() also includes symbol now)
        # but protects against comparing older snapshots without updated comparison groups
        if current.view == "SYMBOL_SPECIFIC":
            if current.symbol != prev.symbol:
                return False, f"Different symbols: {current.symbol} vs {prev.symbol}"
        
        # CRITICAL: Must have identical comparison groups (all metadata must match exactly)
        # This is the primary check - all outcome-influencing metadata must match
        if current.comparison_group and prev.comparison_group:
            # CRITICAL: Validate both comparison groups before comparing
            # Validate current
            try:
                is_valid, missing = current.comparison_group.validate(current.stage, strict=False)
                if not is_valid:
                    return False, f"Current snapshot missing required fields: {missing}"
            except Exception as e:
                return False, f"Current snapshot validation failed: {e}"
            
            # Validate prev
            try:
                is_valid, missing = prev.comparison_group.validate(prev.stage, strict=False)
                if not is_valid:
                    return False, f"Previous snapshot missing required fields: {missing}"
            except Exception as e:
                return False, f"Previous snapshot validation failed: {e}"
            
            # Now compare keys (both should be valid)
            cg_curr = current.comparison_group.to_key(current.stage, strict=False)
            cg_prev = prev.comparison_group.to_key(prev.stage, strict=False)
            
            # CRITICAL: If either key is None, not comparable
            if cg_curr is None or cg_prev is None:
                return False, "One or both comparison groups are invalid (missing required fields)"
            
            if cg_curr != cg_prev:
                return False, f"Different comparison groups: {cg_curr} vs {cg_prev}"
        elif current.comparison_group or prev.comparison_group:
            # One has a comparison group, the other doesn't - not comparable
            return False, "One snapshot missing comparison group"
        
        return True, None
    
    def _extract_excluded_factor_changes(
        self,
        current: NormalizedSnapshot,
        prev: NormalizedSnapshot
    ) -> Dict[str, Any]:
        """Extract changes in excluded factors.
        
        NOTE: Hyperparameters, train_seed, and library_versions are now part of the comparison group
        and will prevent runs from being comparable if they differ. There are no excluded factors
        anymore - all outcome-influencing metadata is required for comparability.
        
        Returns:
            Empty dict (no excluded factors remain)
        """
        # All outcome-influencing factors are now in the comparison group
        # No excluded factors remain
        return {}
    
    def _format_excluded_factors_summary(self, excluded: Dict[str, Any]) -> Optional[str]:
        """Format excluded factors changes into readable summary.
        
        NOTE: All outcome-influencing factors (hyperparameters, train_seed, library_versions)
        are now part of the comparison group, so they won't appear here (runs with different
        values won't be comparable). No excluded factors remain.
        
        Returns:
            None (no excluded factors)
        """
        # All outcome-influencing factors are now in comparison group
        return None
    
    def _count_excluded_factors_changed(self, excluded: Dict[str, Any]) -> int:
        """Count number of excluded factors that changed.
        
        NOTE: All outcome-influencing factors (hyperparameters, train_seed, library_versions)
        are now part of the comparison group, so they won't be counted here (runs with different
        values won't be comparable). No excluded factors remain.
        
        Returns:
            0 (no excluded factors remain)
        """
        # All outcome-influencing factors are now in comparison group
        return 0
    
    def _diff_dict(
        self,
        prev: Dict[str, Any],
        current: Dict[str, Any],
        prefix: str = ""
    ) -> Dict[str, List]:
        """Diff two dictionaries, return changed keys and patch operations."""
        changed_keys = []
        patch = []
        
        all_keys = set(prev.keys()) | set(current.keys())
        
        for key in sorted(all_keys):
            path = f"{prefix}.{key}" if prefix else key
            
            prev_val = prev.get(key)
            current_val = current.get(key)
            
            if prev_val != current_val:
                changed_keys.append(path)
                
                if key not in prev:
                    # Added
                    patch.append({
                        "op": "add",
                        "path": f"/{path}",
                        "value": self._normalize_value(current_val)
                    })
                elif key not in current:
                    # Removed
                    patch.append({
                        "op": "remove",
                        "path": f"/{path}"
                    })
                else:
                    # Changed
                    patch.append({
                        "op": "replace",
                        "path": f"/{path}",
                        "value": self._normalize_value(current_val),
                        "old_value": self._normalize_value(prev_val)
                    })
        
        return {'keys': changed_keys, 'patch': patch}
    
    def _normalize_value(self, val: Any) -> Any:
        """Normalize value for diffing (round floats, sort lists, etc.)."""
        if isinstance(val, float):
            if np.isnan(val) or np.isinf(val):
                return None
            return round(val, 6)
        elif isinstance(val, (list, tuple)):
            # Sort if all elements are comparable
            try:
                return sorted([self._normalize_value(v) for v in val])
            except TypeError:
                return [self._normalize_value(v) for v in val]
        elif isinstance(val, dict):
            return {k: self._normalize_value(v) for k, v in sorted(val.items())}
        else:
            return val
    
    def _reload_snapshot_metrics(self, snapshot: NormalizedSnapshot) -> NormalizedSnapshot:
        """
        Reload metrics from actual metrics.json file for a snapshot.
        
        This ensures we have complete metrics for comparison, even if the snapshot
        was saved before we fixed _normalize_outputs to load all metrics.
        
        Args:
            snapshot: Snapshot to reload metrics for
        
        Returns:
            Snapshot with reloaded metrics (or original if reload fails)
        """
        if not snapshot.stage or not snapshot.view or not snapshot.target:
            return snapshot  # Can't determine path, return as-is
        
        # Try to find the cohort directory for this snapshot
        # We need to search for the run directory and then find the cohort
        if not hasattr(self, 'run_dir') or not self.run_dir:
            return snapshot
        
        # Find RESULTS directory
        results_dir = self.run_dir
        while results_dir.parent.exists() and results_dir.name != "RESULTS":
            results_dir = results_dir.parent
            if results_dir.name == "RESULTS":
                break
        
        if results_dir.name != "RESULTS":
            return snapshot
        
        # Search for the run directory containing this snapshot
        target_clean = snapshot.target.replace('/', '_').replace('\\', '_')
        stage_dir_name = snapshot.stage
        view_dir_name = snapshot.view
        
        # Search in runs directory
        runs_dir = results_dir / "runs"
        if not runs_dir.exists():
            return snapshot
        
        # Search all comparison group directories
        for cg_dir in runs_dir.iterdir():
            if not cg_dir.is_dir() or not cg_dir.name.startswith("cg-"):
                continue
            
            # Search for run directories
            for run_dir in cg_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                
                # Check if this run directory contains the snapshot
                snapshot_path = (
                    run_dir / "REPRODUCIBILITY" / stage_dir_name / view_dir_name / 
                    target_clean
                )
                
                if not snapshot_path.exists():
                    continue
                
                # Search for cohort directories
                for cohort_dir in snapshot_path.iterdir():
                    if not cohort_dir.is_dir() or not cohort_dir.name.startswith("cohort="):
                        continue
                    
                    # Check if this cohort has a snapshot.json with matching run_id
                    snapshot_file = cohort_dir / "snapshot.json"
                    if snapshot_file.exists():
                        try:
                            with open(snapshot_file, 'r') as f:
                                snapshot_data = json.load(f)
                                if snapshot_data.get('run_id') == snapshot.run_id:
                                    # Found the cohort directory! Reload metrics
                                    metrics_file = cohort_dir / "metrics.json"
                                    if metrics_file.exists():
                                        try:
                                            with open(metrics_file, 'r') as f:
                                                metrics_json = json.load(f)
                                            # Extract all numeric metrics (same logic as _normalize_outputs)
                                            metrics_data = {
                                                k: v for k, v in metrics_json.items()
                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                           'composite_version', 'leakage', 'leakage_flag']
                                                and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                                            }
                                            # Update snapshot's outputs.metrics
                                            if metrics_data:
                                                snapshot.outputs['metrics'] = metrics_data
                                                logger.debug(f"Reloaded metrics for snapshot {snapshot.run_id} from {metrics_file}")
                                        except Exception as e:
                                            logger.debug(f"Failed to reload metrics from {metrics_file}: {e}")
                                    return snapshot
                        except Exception:
                            continue
        
        return snapshot  # Return original if we couldn't find/reload
    
    def _compute_metric_deltas(
        self,
        prev_outputs: Dict[str, Any],
        current_outputs: Dict[str, Any]
    ) -> Tuple[Dict[str, Dict[str, Any]], int]:
        """
        Compute structured metric deltas with noise detection.
        
        Returns:
            Tuple of (deltas_dict, total_compared_count)
            - deltas_dict: Only includes significant deltas (above tolerance), keyed by metric name
            - total_compared_count: Total number of metrics compared (including zero-delta ones)
        
        Deltas include:
        - delta_abs, delta_pct: absolute and percentage changes
        - prev, curr: previous and current values
        - z_score: z-score if std_score and n_models available (for noise detection)
        - impact_label: none|noise|minor|major (based on z-score and tolerances)
        - rel_tol, abs_tol: tolerances used for classification
        - changed: True (all entries are significant)
        
        Only computes deltas for numeric metrics that exist in both prev and current.
        Zero-delta entries (like pos_rate, n_models that never change) are excluded from deltas_dict
        but counted in total_compared_count.
        """
        import math
        
        deltas = {}
        total_compared = 0
        
        prev_metrics = prev_outputs.get('metrics', {})
        current_metrics = current_outputs.get('metrics', {})
        
        # Get context for z-score computation (std_score, n_models)
        prev_std_score = prev_metrics.get('std_score')
        prev_n_models = prev_metrics.get('n_models')
        curr_std_score = current_metrics.get('std_score')
        curr_n_models = current_metrics.get('n_models')
        
        # Use average std_score and n_models for z-score computation
        avg_std_score = None
        avg_n_models = None
        if prev_std_score is not None and curr_std_score is not None:
            avg_std_score = (float(prev_std_score) + float(curr_std_score)) / 2.0
        elif prev_std_score is not None:
            avg_std_score = float(prev_std_score)
        elif curr_std_score is not None:
            avg_std_score = float(curr_std_score)
        
        if prev_n_models is not None and curr_n_models is not None:
            avg_n_models = (float(prev_n_models) + float(curr_n_models)) / 2.0
        elif prev_n_models is not None:
            avg_n_models = float(prev_n_models)
        elif curr_n_models is not None:
            avg_n_models = float(curr_n_models)
        
        # Compute standard error if we have std_score and n_models
        se = None
        if avg_std_score is not None and avg_n_models is not None and avg_n_models > 0:
            se = avg_std_score / math.sqrt(avg_n_models)
        
        # Define tolerances (metric-dependent defaults)
        # For score-like metrics, use tighter tolerances
        score_metrics = {'auc', 'composite_score', 'std_score'}
        abs_tol_default = 1e-4  # Default absolute tolerance
        rel_tol_default = 1e-4  # Default relative tolerance (0.01%)
        
        for key in set(prev_metrics.keys()) & set(current_metrics.keys()):
            prev_val = prev_metrics.get(key)
            curr_val = current_metrics.get(key)
            
            # Skip non-numeric values
            if prev_val is None or curr_val is None:
                continue
            
            try:
                prev_float = float(prev_val)
                curr_float = float(curr_val)
                
                # Skip NaN or Inf values
                if math.isnan(prev_float) or math.isinf(prev_float) or math.isnan(curr_float) or math.isinf(curr_float):
                    continue
                
                # Count this metric as compared (even if delta is zero)
                total_compared += 1
                
                delta_abs = curr_float - prev_float
                delta_pct = (delta_abs / abs(prev_float) * 100) if prev_float != 0 else 0.0
                
                # Choose tolerances based on metric type
                abs_tol = abs_tol_default
                rel_tol = rel_tol_default
                
                # For score metrics, use tighter tolerances
                if key in score_metrics:
                    abs_tol = 1e-3  # Slightly more lenient for scores
                    rel_tol = 1e-4  # 0.01%
                
                # Compute z-score if we have standard error
                z_score = None
                se_ratio = None
                noise_explanation = None
                if se is not None and se > 0:
                    z_score = delta_abs / se
                    se_ratio = abs(delta_abs) / se  # How many SEs is this delta?
                    
                    # Generate noise explanation for statistical context
                    # We'll use a placeholder that will be replaced with exact serialized value
                    # This ensures the explanation matches the exact value in delta_abs field
                    delta_abs_abs = abs(delta_abs)
                    if abs(z_score) < 0.25:
                        noise_explanation_template = (
                            "Delta ({DELTA_ABS}) is {se_ratio:.4f}× the standard error (SE≈{se:.4f}). "
                            "This is statistically insignificant - well within expected run-to-run variability."
                        )
                    elif abs(z_score) < 1.0:
                        noise_explanation_template = (
                            "Delta ({DELTA_ABS}) is {se_ratio:.4f}× the standard error (SE≈{se:.4f}). "
                            "This is a minor change but still within expected variability."
                        )
                    else:
                        noise_explanation_template = (
                            "Delta ({DELTA_ABS}) is {se_ratio:.4f}× the standard error (SE≈{se:.4f}). "
                            "This exceeds expected variability and may indicate a real change."
                        )
                    # Store template for later replacement with exact serialized value
                    noise_explanation = noise_explanation_template.format(
                        DELTA_ABS="{DELTA_ABS}",  # Placeholder
                        se_ratio=se_ratio,
                        se=se
                    )
                
                # Check if values differ (after rounding for comparison)
                # Round to 6 decimal places for comparison (same as serialization)
                prev_rounded = round(prev_float, 6)
                curr_rounded = round(curr_float, 6)
                differs = prev_rounded != curr_rounded
                
                # Check if delta is significant (above tolerance)
                # Use max(abs_tol, rel_tol * max(abs(prev), abs(curr), 1.0)) for combined tolerance
                # This ensures we use the stricter of absolute or relative tolerance
                abs_change = abs(delta_abs)
                max_value = max(abs(prev_float), abs(curr_float), 1.0)  # At least 1.0 to avoid division issues
                tol = max(abs_tol, rel_tol * max_value)
                is_significant = abs_change > tol
                
                # Classify impact
                impact_label = 'none'
                if z_score is not None:
                    # Use z-score for classification
                    if abs(z_score) < 0.25:
                        impact_label = 'noise'
                    elif abs(z_score) < 1.0:
                        impact_label = 'minor'
                    else:
                        impact_label = 'major'
                else:
                    # Fall back to tolerance-based classification using same tol calculation
                    if abs_change < tol:
                        impact_label = 'none'
                    elif abs_change < tol * 10:
                        impact_label = 'noise'
                    elif abs_change < tol * 100:
                        impact_label = 'minor'
                    else:
                        impact_label = 'major'
                
                # Only include significant deltas (above tolerance)
                # This prevents spam from zero-delta entries like pos_rate, n_models that never change
                if is_significant:
                    # Serialize delta_abs first to get exact value for explanation
                    delta_abs_serialized = round(delta_abs, 6)
                    
                    delta_entry = {
                        'delta_abs': delta_abs_serialized,
                        'delta_pct': round(delta_pct, 4),  # More precision for percentage
                        'prev': prev_rounded,
                        'curr': curr_rounded,
                        'differs': differs,  # prev != curr (after rounding)
                        'changed_tol': is_significant,  # abs(delta) > tol_used
                        'impact_label': impact_label,
                        'abs_tol': abs_tol,
                        'rel_tol': rel_tol,
                        'tol_used': round(tol, 6)  # The actual tolerance used: max(abs_tol, rel_tol*max(abs(prev),abs(curr),1.0))
                    }
                    
                    # Add statistical context if available
                    if se is not None and se > 0:
                        delta_entry['se'] = round(se, 6)  # Standard error
                        # se_ratio is preferred over z_score (this is a proxy, not a true z-score)
                        delta_entry['se_ratio'] = round(se_ratio, 4) if se_ratio is not None else None
                        # z_score kept for backward compatibility, but se_ratio is preferred
                        delta_entry['z_score'] = round(z_score, 4) if z_score is not None else None
                        delta_entry['z_score_basis'] = "auc_se_proxy"  # Using std_score/sqrt(n_models) as proxy for all metrics
                        if noise_explanation:
                            # Use exact serialized value in explanation to avoid formatting bugs
                            # Replace placeholder with exact serialized value
                            delta_entry['noise_explanation'] = noise_explanation.replace(
                                "{DELTA_ABS}",
                                str(delta_abs_serialized)
                            )
                    else:
                        # No SE available
                        delta_entry['se_ratio'] = None
                        delta_entry['z_score'] = None  # Deprecated: use se_ratio
                        delta_entry['z_score_basis'] = None
                    
                    deltas[key] = delta_entry
            except (ValueError, TypeError, ZeroDivisionError):
                # Skip non-numeric or problematic values
                continue
        
        return deltas, total_compared
    
    def _load_trend_from_metadata(self, metadata_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load trend data from metadata.json file.
        
        Args:
            metadata_path: Path to metadata.json file
            
        Returns:
            Trend dict from metadata.json['trend'] if available, None otherwise
        """
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            trend = metadata.get('trend')
            if trend and isinstance(trend, dict):
                return trend
        except Exception as e:
            logger.debug(f"Failed to load trend from {metadata_path}: {e}")
        
        return None
    
    def _compute_trend_deltas(
        self,
        prev_trend: Optional[Dict[str, Any]],
        curr_trend: Optional[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute trend deltas between previous and current runs.
        
        Compares all trend fields: slope_per_day, current_estimate, ewma_value, 
        status, n_runs, residual_std.
        
        For numeric fields: computes delta_abs and delta_pct (similar to metric_deltas).
        For status field: shows prev→curr transition.
        
        Args:
            prev_trend: Trend dict from previous run (from metadata.json['trend'])
            curr_trend: Trend dict from current run (from metadata.json['trend'])
            
        Returns:
            Dict mapping trend field names to delta dicts with prev, curr, delta_abs, delta_pct.
            Returns empty dict if either trend is missing or metric_name doesn't match.
        """
        import math
        
        if not prev_trend or not curr_trend:
            return {}
        
        # Only compare if metric_name matches (trends are metric-specific)
        prev_metric = prev_trend.get('metric_name')
        curr_metric = curr_trend.get('metric_name')
        if prev_metric != curr_metric:
            logger.debug(
                f"Skipping trend comparison: metric_name mismatch "
                f"(prev={prev_metric}, curr={curr_metric})"
            )
            return {}
        
        deltas = {}
        
        # Fields to compare (numeric fields)
        numeric_fields = [
            'slope_per_day',
            'current_estimate',
            'ewma_value',
            'residual_std'
        ]
        
        # Integer fields
        integer_fields = ['n_runs']
        
        # Status field (non-numeric)
        status_field = 'status'
        
        # Compare numeric fields
        for field in numeric_fields:
            prev_val = prev_trend.get(field)
            curr_val = curr_trend.get(field)
            
            # Skip if either value is None or NaN
            if prev_val is None or curr_val is None:
                continue
            
            try:
                prev_float = float(prev_val)
                curr_float = float(curr_val)
                
                # Skip NaN or Inf values
                if math.isnan(prev_float) or math.isinf(prev_float) or \
                   math.isnan(curr_float) or math.isinf(curr_float):
                    continue
                
                delta_abs = curr_float - prev_float
                # For percentage, handle zero and very small values
                if abs(prev_float) > 1e-10:
                    delta_pct = (delta_abs / abs(prev_float)) * 100
                else:
                    delta_pct = 0.0 if abs(delta_abs) < 1e-10 else float('inf')
                
                deltas[field] = {
                    'prev': round(prev_float, 6),
                    'curr': round(curr_float, 6),
                    'delta_abs': round(delta_abs, 6),
                    'delta_pct': round(delta_pct, 4)
                }
            except (ValueError, TypeError):
                continue
        
        # Compare integer fields
        for field in integer_fields:
            prev_val = prev_trend.get(field)
            curr_val = curr_trend.get(field)
            
            if prev_val is None or curr_val is None:
                continue
            
            try:
                prev_int = int(prev_val)
                curr_int = int(curr_val)
                
                delta_abs = curr_int - prev_int
                delta_pct = (delta_abs / prev_int * 100) if prev_int != 0 else 0.0
                
                deltas[field] = {
                    'prev': prev_int,
                    'curr': curr_int,
                    'delta_abs': delta_abs,
                    'delta_pct': round(delta_pct, 4)
                }
            except (ValueError, TypeError):
                continue
        
        # Compare status field (non-numeric)
        prev_status = prev_trend.get(status_field)
        curr_status = curr_trend.get(status_field)
        
        if prev_status is not None or curr_status is not None:
            deltas[status_field] = {
                'prev': prev_status,
                'curr': curr_status,
                'delta_abs': None,
                'delta_pct': None
            }
            # Add transition string if status changed
            if prev_status != curr_status:
                deltas[status_field]['transition'] = f"{prev_status}→{curr_status}"
        
        return deltas
    
    def _classify_metric_impact(
        self,
        metric_deltas: Dict[str, Dict[str, Any]],
        top_k: int = 5
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Classify overall impact from metric deltas and extract top regressions/improvements.
        
        All entries in metric_deltas are significant by design (zero-delta entries are filtered out).
        
        Uses per-metric polarity to correctly identify improvements vs regressions:
        - For metrics where higher is better (auc, composite_score, mean_importance): positive delta = improvement
        - For metrics where lower is better (std_score): positive delta = regression
        
        Returns:
            Tuple of (impact_label, top_regressions, top_improvements)
            - impact_label: none|noise|minor|major (worst impact across all metrics)
            - top_regressions: List of up to top_k metrics with worst regressions (using signed_delta)
            - top_improvements: List of up to top_k metrics with best improvements (using signed_delta)
        """
        if not metric_deltas:
            return 'none', [], []
        
        # Per-metric polarity: True = higher is better, False = lower is better
        higher_is_better = {
            'auc': True,
            'composite_score': True,
            'mean_importance': True,
            'std_score': False,  # Lower variability is better
            'n_models': True,  # More models is better (if applicable)
            'pos_rate': True,  # Higher positive rate is better (if applicable)
            'n_effective_cs': True,  # More effective samples is better
        }
        
        # Impact hierarchy: none < noise < minor < major
        impact_hierarchy = {'none': 0, 'noise': 1, 'minor': 2, 'major': 3}
        
        # Find worst impact across all metrics
        worst_impact = 'none'
        regressions = []
        improvements = []
        
        for metric_name, delta_info in metric_deltas.items():
            impact = delta_info.get('impact_label', 'none')
            if impact_hierarchy.get(impact, 0) > impact_hierarchy.get(worst_impact, 0):
                worst_impact = impact
            
            # Compute signed delta based on metric polarity
            delta_abs = delta_info.get('delta_abs', 0)
            is_higher_better = higher_is_better.get(metric_name, True)  # Default to higher is better
            signed_delta = delta_abs if is_higher_better else -delta_abs
            
            # Collect regressions (negative signed_delta) and improvements (positive signed_delta)
            if signed_delta < 0:
                regressions.append({
                    'metric': metric_name,
                    'delta_abs': delta_abs,
                    'signed_delta': signed_delta,  # For ranking
                    'delta_pct': delta_info.get('delta_pct', 0),
                    'z_score': delta_info.get('z_score'),
                    'impact': impact
                })
            elif signed_delta > 0:
                improvements.append({
                    'metric': metric_name,
                    'delta_abs': delta_abs,
                    'signed_delta': signed_delta,  # For ranking
                    'delta_pct': delta_info.get('delta_pct', 0),
                    'z_score': delta_info.get('z_score'),
                    'impact': impact
                })
        
        # Sort regressions by worst signed_delta (most negative first)
        regressions.sort(key=lambda x: x['signed_delta'])
        # Sort improvements by best signed_delta (most positive first)
        improvements.sort(key=lambda x: x['signed_delta'], reverse=True)
        
        # Return top K
        top_regressions = regressions[:top_k]
        top_improvements = improvements[:top_k]
        
        return worst_impact, top_regressions, top_improvements
    
    def _determine_severity(
        self,
        changed_keys: List[str],
        input_changes: Dict,
        process_changes: Dict,
        output_changes: Dict,
        metric_deltas: Dict[str, Dict[str, float]],
        excluded_factors_changed: Dict[str, Any],
        excluded_factors_summary: Optional[str],
        summary: Dict[str, Any]
    ) -> Tuple[ChangeSeverity, str]:
        """
        Determine severity of changes (SST-style: purely derived from report).
        
        Returns:
            Tuple of (severity, reason)
        """
        total_changes = summary.get('total_changes', len(changed_keys))
        output_changes_count = summary.get('output_changes', len(output_changes.get('keys', [])))
        metric_deltas_count = summary.get('metric_deltas_significant', summary.get('metric_deltas_count', len(metric_deltas)))  # Use significant count
        input_changes_count = summary.get('input_changes', len(input_changes.get('keys', [])))
        process_changes_count = summary.get('process_changes', len(process_changes.get('keys', [])))
        has_excluded_factors = summary.get('excluded_factors_changed', bool(excluded_factors_changed))
        output_digest_changes = summary.get('output_digest_changes', [])
        
        # CRITICAL: Separate "reproducibility variance" from "regression"
        # If output digests differ, that's a reproducibility variance issue
        # But performance impact should be assessed separately using impact_label
        impact_label = summary.get('impact_label', 'none')
        has_reproducibility_variance = bool(output_digest_changes)
        
        # CRITICAL FIX: If no changes at all, severity must be NONE
        if total_changes == 0 and metric_deltas_count == 0:
            if has_excluded_factors and excluded_factors_summary:
                # Only excluded factors changed (hyperparams, seeds, versions)
                return ChangeSeverity.MINOR, f"Only excluded factors changed: {excluded_factors_summary}"
            else:
                return ChangeSeverity.NONE, "No changes detected"
        
        # Critical: hard invariants (data, target, features, split, leakage)
        critical_paths = [
            'inputs.data', 'inputs.target', 'inputs.features.feature_fingerprint',
            'process.split', 'process.leakage'
        ]
        
        for key in changed_keys:
            for critical in critical_paths:
                if key.startswith(critical):
                    return ChangeSeverity.CRITICAL, f"Critical change detected in {key}"
        
        # Handle reproducibility variance (digest mismatch) with performance impact assessment
        if has_reproducibility_variance:
            # Reproducibility variance is always a concern
            # But severity depends on whether there's actual performance impact
            if impact_label in ['none', 'noise']:
                # Reproducibility variance detected but performance impact is noise-level
                return ChangeSeverity.MAJOR, (
                    f"Output digests differ (reproducibility variance detected): {', '.join(output_digest_changes)}. "
                    f"Performance impact: {impact_label} (z-score analysis indicates noise-level changes only)."
                )
            elif impact_label == 'minor':
                # Reproducibility variance with minor performance impact
                return ChangeSeverity.MAJOR, (
                    f"Output digests differ (reproducibility variance detected): {', '.join(output_digest_changes)}. "
                    f"Performance impact: {impact_label} (minor changes detected)."
                )
            else:  # impact_label == 'major'
                # Reproducibility variance with major performance impact
                return ChangeSeverity.CRITICAL, (
                    f"Output digests differ (reproducibility variance detected): {', '.join(output_digest_changes)}. "
                    f"Performance impact: {impact_label} (significant changes detected)."
                )
        
        # Major: important config OR output/metric changes
        major_paths = [
            'inputs.config', 'process.training', 'process.environment'
        ]
        
        # Check for output changes or metric deltas (these indicate model behavior changed)
        if output_changes_count > 0 or metric_deltas_count > 0:
            # Use impact_label to determine severity for metric changes
            if metric_deltas_count > 0:
                if impact_label == 'major':
                    return ChangeSeverity.MAJOR, (
                        f"Output/metric changes detected: {metric_deltas_count} metric deltas, "
                        f"{output_changes_count} output changes. Performance impact: {impact_label}."
                    )
                elif impact_label == 'minor':
                    return ChangeSeverity.MINOR, (
                        f"Output/metric changes detected: {metric_deltas_count} metric deltas, "
                        f"{output_changes_count} output changes. Performance impact: {impact_label}."
                    )
                else:  # none or noise
                    return ChangeSeverity.MINOR, (
                        f"Output/metric changes detected: {metric_deltas_count} metric deltas, "
                        f"{output_changes_count} output changes. Performance impact: {impact_label} (noise-level)."
                    )
            else:
                return ChangeSeverity.MAJOR, f"Output changes detected: {output_changes_count} output changes"
        
        # Check for major config paths
        for key in changed_keys:
            for major in major_paths:
                if key.startswith(major):
                    return ChangeSeverity.MAJOR, f"Major config change detected in {key}"
        
        # Minor: only input/process changes (not output/metrics), OR only excluded factors
        if has_excluded_factors and excluded_factors_summary and total_changes == 0:
            # Only excluded factors changed (already handled above, but double-check)
            return ChangeSeverity.MINOR, f"Only excluded factors changed: {excluded_factors_summary}"
        
        # Minor: only metrics in changed_keys (but no metric_deltas - this is edge case)
        if changed_keys and all(key.startswith('outputs.metrics') for key in changed_keys) and metric_deltas_count == 0:
            return ChangeSeverity.MINOR, f"Only metric metadata changed (no actual metric deltas): {len(changed_keys)} keys"
        
        # Minor: only input/process changes (no output/metric changes)
        if (input_changes_count > 0 or process_changes_count > 0) and output_changes_count == 0 and metric_deltas_count == 0:
            return ChangeSeverity.MINOR, f"Only input/process changes: {input_changes_count} input, {process_changes_count} process"
        
        # Default to major if we have changes but don't fit above categories
        if changed_keys:
            return ChangeSeverity.MAJOR, f"Mixed changes detected: {total_changes} total changes across inputs/process/outputs"
        
        # Fallback (shouldn't reach here if logic is correct)
        return ChangeSeverity.NONE, "No changes detected (fallback)"
    
    def find_previous_comparable(
        self,
        snapshot: NormalizedSnapshot
    ) -> Optional[NormalizedSnapshot]:
        """Find previous comparable snapshot.
        
        Searches across runs in the same sample size bin to find previous comparable runs.
        """
        if not snapshot.comparison_group:
            return None
        
        # CRITICAL: to_key() now requires stage and may return None for invalid groups
        group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
        if group_key is None:
            logger.warning(f"Snapshot {snapshot.run_id} has invalid comparison group, cannot find comparable")
            return None
        
        # First, search in current run's snapshots
        # NOTE: We only use snapshots from snapshot_index.json search below (which verifies file existence)
        # In-memory snapshots from self._snapshots might be stale if runs were deleted
        candidates = []
        # Skip in-memory cache search - rely on snapshot_index.json which verifies file existence
        
        # Search across ALL runs in RESULTS to find previous comparable runs
        # This ensures we find exactly the same runs (same comparison_group) regardless of bin
        # CRITICAL: Also search in comparison group directories (cg-*_n-*_fam-*)
        if hasattr(self, 'run_dir') and self.run_dir:
            # Find RESULTS directory
            results_dir = self.run_dir
            while results_dir.parent.exists() and results_dir.name != "RESULTS":
                results_dir = results_dir.parent
                if results_dir.name == "RESULTS":
                    break
            
            if results_dir.name == "RESULTS":
                # Search both sample_* bins and comparison group directories (cg-*)
                # Handle both old structure (RESULTS/sample_*/) and new structure (RESULTS/runs/cg-*/)
                search_dirs = []
                runs_dir = results_dir / "runs"
                if runs_dir.exists() and runs_dir.is_dir():
                    # New structure: RESULTS/runs/cg-*/
                    search_dirs = [runs_dir]
                else:
                    # Old structure: RESULTS/sample_*/ or RESULTS/cg-*/
                    search_dirs = [results_dir]
                
                for base_dir in search_dirs:
                    for bin_dir in base_dir.iterdir():
                        if not bin_dir.is_dir():
                            continue
                        
                        # Search all runs in this bin/directory
                        for run_subdir in bin_dir.iterdir():
                            if not run_subdir.is_dir() or run_subdir.name == "METRICS":
                                continue
                            
                            # Check both target-first (globals/) and legacy (REPRODUCIBILITY/METRICS/)
                            # Prioritize target-first structure
                            run_snapshot_index = None
                            globals_snapshot_index = run_subdir / "globals" / "snapshot_index.json"
                            legacy_snapshot_index = run_subdir / "REPRODUCIBILITY" / "METRICS" / "snapshot_index.json"
                            
                            if globals_snapshot_index.exists():
                                run_snapshot_index = globals_snapshot_index
                            elif legacy_snapshot_index.exists():
                                run_snapshot_index = legacy_snapshot_index
                            
                            if run_snapshot_index and run_snapshot_index.exists():
                                try:
                                    with open(run_snapshot_index) as f:
                                        data = json.load(f)
                                    
                                    for key, snap_data in data.items():
                                        # Handle both old format (run_id key) and new format (run_id:stage key)
                                        if ':' in key:
                                            run_id = key.split(':', 1)[0]
                                        else:
                                            run_id = key
                                        
                                        # CRITICAL: Never pick the same run_id (even if different stage)
                                        if run_id == snapshot.run_id:
                                            continue
                                        
                                        try:
                                            snap = self._deserialize_snapshot(snap_data)
                                            # Double-check run_id (defense in depth)
                                            if snap.run_id == snapshot.run_id:
                                                continue
                                            
                                            # CRITICAL: Verify snapshot matches stage and view for exact matching
                                            if snap.stage != snapshot.stage:
                                                continue
                                            if snap.view != snapshot.view:
                                                continue
                                            
                                            # Only add if same comparison_group (exactly the same runs)
                                            snap_key = snap.comparison_group.to_key(snap.stage, strict=False) if snap.comparison_group else None
                                            if (snap.comparison_group and 
                                                snap_key is not None and snap_key == group_key):
                                                comparable, reason = self._check_comparability(snapshot, snap)
                                                if comparable:
                                                    # CRITICAL: Verify snapshot file actually exists on disk
                                                    # The snapshot_index.json might reference a snapshot from a deleted run
                                                    # We need to verify the snapshot.json file exists before using it
                                                    snapshot_file_exists = False
                                                    cohort_subdir_found = None
                                                    if snap.stage and snap.view and snap.target:
                                                        target_clean = snap.target.replace('/', '_').replace('\\', '_')
                                                        
                                                        # First, try target-first structure: targets/<target>/reproducibility/<view>/cohort=<cohort_id>/
                                                        target_repro_dir = run_subdir / "targets" / target_clean / "reproducibility"
                                                        if target_repro_dir.exists():
                                                            view_dir = target_repro_dir / snap.view
                                                            if view_dir.exists():
                                                                # For SYMBOL_SPECIFIC, check symbol subdirectories
                                                                if snap.view == "SYMBOL_SPECIFIC" and snap.symbol:
                                                                    symbol_dir = view_dir / f"symbol={snap.symbol}"
                                                                    if symbol_dir.exists():
                                                                        for cohort_subdir in symbol_dir.iterdir():
                                                                            if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                                                snapshot_file = cohort_subdir / "snapshot.json"
                                                                                if snapshot_file.exists():
                                                                                    try:
                                                                                        with open(snapshot_file, 'r') as f:
                                                                                            snapshot_data = json.load(f)
                                                                                            if snapshot_data.get('run_id') == snap.run_id:
                                                                                                snapshot_file_exists = True
                                                                                                cohort_subdir_found = cohort_subdir
                                                                                                break
                                                                                    except Exception:
                                                                                        continue
                                                                else:
                                                                    # CROSS_SECTIONAL or other views without symbol
                                                                    for cohort_subdir in view_dir.iterdir():
                                                                        if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                                            snapshot_file = cohort_subdir / "snapshot.json"
                                                                            if snapshot_file.exists():
                                                                                try:
                                                                                    with open(snapshot_file, 'r') as f:
                                                                                        snapshot_data = json.load(f)
                                                                                        if snapshot_data.get('run_id') == snap.run_id:
                                                                                            snapshot_file_exists = True
                                                                                            cohort_subdir_found = cohort_subdir
                                                                                            break
                                                                                except Exception:
                                                                                    continue
                                                        
                                                        # Fallback to legacy structure: REPRODUCIBILITY/<stage>/<view>/<target>/cohort=<cohort_id>/
                                                        if not snapshot_file_exists:
                                                            stage_dir = run_subdir / "REPRODUCIBILITY" / snap.stage / snap.view / target_clean
                                                            if stage_dir.exists():
                                                                # Search for cohort directories
                                                                for cohort_subdir in stage_dir.iterdir():
                                                                    if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                                        snapshot_file = cohort_subdir / "snapshot.json"
                                                                        if snapshot_file.exists():
                                                                            # Verify this snapshot.json matches the run_id
                                                                            try:
                                                                                with open(snapshot_file, 'r') as f:
                                                                                    snapshot_data = json.load(f)
                                                                                    if snapshot_data.get('run_id') == snap.run_id:
                                                                                        snapshot_file_exists = True
                                                                                        cohort_subdir_found = cohort_subdir
                                                                                        break
                                                                            except Exception:
                                                                                continue
                                                            
                                                    # If we found a matching snapshot file, use it
                                                    if snapshot_file_exists and cohort_subdir_found:
                                                        # CRITICAL: Reload metrics from actual metrics.json file
                                                        # The snapshot might have been saved before we fixed _normalize_outputs
                                                        # Try target-first metrics location first, then legacy
                                                        metrics_file = None
                                                        metrics_json = None
                                                        try:
                                                            from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                                                            metrics_dir = get_metrics_path_from_cohort_dir(cohort_subdir_found, base_output_dir=run_subdir)
                                                            if metrics_dir:
                                                                metrics_file = metrics_dir / "metrics.json"
                                                                if not metrics_file.exists():
                                                                    metrics_parquet = metrics_dir / "metrics.parquet"
                                                                    if metrics_parquet.exists():
                                                                        # Handle parquet separately
                                                                        import pandas as pd
                                                                        df = pd.read_parquet(metrics_parquet)
                                                                        if len(df) > 0:
                                                                            metrics_json = df.iloc[0].to_dict()
                                                                        metrics_file = None  # Mark as handled
                                                                    else:
                                                                        metrics_file = None
                                                        except Exception as e:
                                                            logger.debug(f"Failed to get metrics path from cohort_dir: {e}")
                                                        
                                                        # Fallback to legacy location
                                                        if not metrics_file and not metrics_json:
                                                            metrics_file = cohort_subdir_found / "metrics.json"
                                                        
                                                        if metrics_file and metrics_file.exists():
                                                            try:
                                                                if metrics_file.suffix == '.parquet':
                                                                    # Handle parquet
                                                                    import pandas as pd
                                                                    df = pd.read_parquet(metrics_file)
                                                                    if len(df) > 0:
                                                                        metrics_json = df.iloc[0].to_dict()
                                                                else:
                                                                    with open(metrics_file, 'r') as f:
                                                                        metrics_json = json.load(f)
                                                            except Exception as e:
                                                                logger.debug(f"Failed to reload metrics from {metrics_file}: {e}")
                                                        
                                                        # Process metrics_json if we have it
                                                        if metrics_json:
                                                            # Extract all numeric metrics (same logic as _normalize_outputs)
                                                            metrics_data = {
                                                                k: v for k, v in metrics_json.items()
                                                                if k not in ['diff_telemetry', 'run_id', 'timestamp', 'reproducibility_mode',
                                                                           'stage', 'target', 'metric_name', 'task_type', 'composite_definition',
                                                                           'composite_version', 'leakage', 'leakage_flag']
                                                                and (isinstance(v, (int, float)) or (isinstance(v, (list, dict)) and v))
                                                            }
                                                            # Update snapshot's outputs.metrics
                                                            if metrics_data:
                                                                snap.outputs['metrics'] = metrics_data
                                                                logger.debug(f"Reloaded metrics for snapshot {snap.run_id} from {metrics_file or 'parquet'}")
                                                            
                                                            # Mark source for auditability
                                                            if bin_dir.name.startswith("cg-"):
                                                                snap._comparison_source = "comparison_group_directory"
                                                            else:
                                                                snap._comparison_source = "snapshot_index"
                                                            candidates.append((snap.timestamp, snap))
                                                        else:
                                                            logger.debug(f"Skipping snapshot {snap.run_id} from {run_snapshot_index} - snapshot.json file not found on disk (run may have been deleted)")
                                                    else:
                                                        # If we can't determine the path, skip it (safety first)
                                                        logger.debug(f"Skipping snapshot {snap.run_id} - missing stage/view/target for path reconstruction")
                                        except Exception as e:
                                            logger.debug(f"Failed to deserialize snapshot from {run_snapshot_index}: {e}")
                                            continue
                                except Exception as e:
                                    logger.debug(f"Failed to read snapshot index {run_snapshot_index}: {e}")
                                    continue
        
        if not candidates:
            return None
        
        # Return most recent by monotonic sequence number (snapshot_seq)
        # This is the correct ordering method - mtime can change for unrelated reasons
        # (file copies, post-processing, filesystem quirks, coarse timestamp resolution)
        candidates_with_seq = []
        for timestamp, snap in candidates:
            # Use snapshot_seq if available (assigned at save time), fallback to timestamp
            seq = snap.snapshot_seq if hasattr(snap, 'snapshot_seq') and snap.snapshot_seq is not None else 0
            # If no seq, use timestamp as fallback (but log warning)
            if seq == 0:
                logger.debug(f"Snapshot {snap.run_id} has no snapshot_seq, using timestamp fallback")
                # Convert timestamp to numeric for comparison (ISO format)
                try:
                    from datetime import datetime
                    ts_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    seq = ts_dt.timestamp()
                except Exception:
                    seq = 0
            
            candidates_with_seq.append((seq, snap))
        
        # Sort by sequence (highest = most recent)
        candidates_with_seq.sort(key=lambda x: x[0], reverse=True)
        prev_snapshot = candidates_with_seq[0][1]
        
        # CRITICAL: Reload metrics from actual metrics.json file for the previous snapshot
        # The snapshot might have been saved before we fixed _normalize_outputs, so metrics might be incomplete
        # We need to reload from the actual metrics.json file to ensure we have all metrics for comparison
        prev_snapshot = self._reload_snapshot_metrics(prev_snapshot)
        
        return prev_snapshot
    
    def get_or_establish_baseline(
        self,
        snapshot: NormalizedSnapshot,
        metrics: Dict[str, float],
        cohort_dir: Path
    ) -> Tuple[Optional[BaselineState], bool]:
        """
        Get or establish baseline for comparison group.
        
        Baselines are stored per-cohort to ensure only exactly the same runs share baselines.
        
        Args:
            snapshot: Normalized snapshot
            metrics: Metrics dict
            cohort_dir: Cohort directory where baseline will be stored
        
        Returns:
            (BaselineState or None, is_new_baseline)
        """
        if not snapshot.comparison_group:
            return None, False
        
        # CRITICAL: to_key() now requires stage and may return None for invalid groups
        group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
        if group_key is None:
            logger.warning(f"Snapshot {snapshot.run_id} has invalid comparison group, cannot get/establish baseline")
            return None, False
        
        # Check cache first
        if group_key in self._baselines:
            return self._baselines[group_key], False
        
        # Load from cohort directory
        baseline = self._load_baseline_from_cohort(cohort_dir, group_key)
        if baseline:
            self._baselines[group_key] = baseline  # Cache it
            return baseline, False
        
        # Count comparable runs (search across runs in same bin)
        # CRITICAL: Never include the same run_id (even if different stage)
        comparable_runs = [
            snap for snap in self._snapshots.values()
            if (snap.comparison_group and 
                snap.comparison_group.to_key(snap.stage, strict=False) == group_key and
                snap.run_id != snapshot.run_id)  # Exclude same run_id
        ]
        
        # Search across ALL runs in RESULTS to find comparable runs with same comparison_group
        # This ensures baselines are established from exactly the same runs
        if hasattr(self, 'run_dir') and self.run_dir:
            # Find RESULTS directory
            results_dir = self.run_dir
            while results_dir.parent.exists() and results_dir.name != "RESULTS":
                results_dir = results_dir.parent
                if results_dir.name == "RESULTS":
                    break
            
            if results_dir.name == "RESULTS":
                # Search all sample_* bins
                for bin_dir in results_dir.iterdir():
                    if bin_dir.is_dir() and bin_dir.name.startswith("sample_"):
                        # Search all runs in this bin
                        for run_subdir in bin_dir.iterdir():
                            if run_subdir.is_dir() and run_subdir.name != "METRICS":
                                # Check both target-first (globals/) and legacy (REPRODUCIBILITY/METRICS/)
                                # Prioritize target-first structure
                                run_snapshot_index = None
                                globals_snapshot_index = run_subdir / "globals" / "snapshot_index.json"
                                legacy_snapshot_index = run_subdir / "REPRODUCIBILITY" / "METRICS" / "snapshot_index.json"
                                
                                if globals_snapshot_index.exists():
                                    run_snapshot_index = globals_snapshot_index
                                elif legacy_snapshot_index.exists():
                                    run_snapshot_index = legacy_snapshot_index
                                
                                if run_snapshot_index and run_snapshot_index.exists():
                                    try:
                                        with open(run_snapshot_index) as f:
                                            data = json.load(f)
                                            for key, snap_data in data.items():
                                                # Handle both old format (run_id key) and new format (run_id:stage key)
                                                if ':' in key:
                                                    run_id = key.split(':', 1)[0]
                                                else:
                                                    run_id = key
                                                
                                                # CRITICAL: Never pick the same run_id (even if different stage)
                                                if run_id == snapshot.run_id:
                                                    continue
                                                
                                                try:
                                                    snap = self._deserialize_snapshot(snap_data)
                                                    # Double-check run_id (defense in depth)
                                                    if snap.run_id == snapshot.run_id:
                                                        continue
                                                    # Only add if same comparison_group (exactly the same runs)
                                                    snap_key = snap.comparison_group.to_key(snap.stage, strict=False) if snap.comparison_group else None
                                                    if (snap.comparison_group and 
                                                        snap_key is not None and snap_key == group_key):
                                                        comparable_runs.append(snap)
                                                except Exception:
                                                    continue
                                    except Exception:
                                        continue
        
        if len(comparable_runs) < self.min_runs_for_baseline:
            return None, False
        
        # Establish baseline (use best metric run)
        best_run = None
        best_score = None
        
        for snap in comparable_runs:
            if snap.outputs.get('metrics', {}).get('auc'):
                score = snap.outputs['metrics']['auc']
                if best_score is None or score > best_score:
                    best_score = score
                    best_run = snap
        
        if best_run:
            baseline = BaselineState(
                comparison_group_key=group_key,
                baseline_run_id=best_run.run_id,
                baseline_timestamp=best_run.timestamp,
                baseline_metrics=best_run.outputs.get('metrics', {}),
                established_at=datetime.now().isoformat()
            )
            self._baselines[group_key] = baseline  # Cache it
            # Save to cohort directory (not global index)
            self._save_baseline_to_cohort(cohort_dir, baseline)
            return baseline, True
        
        return None, False
    
    def save_diff(
        self,
        diff: DiffResult,
        baseline_diff: Optional[DiffResult],
        cohort_dir: Path
    ) -> None:
        """
        Save diff results to cohort directory.
        
        Args:
            diff: Diff against previous run
            baseline_diff: Diff against baseline (if available)
            cohort_dir: Cohort directory
        """
        cohort_dir = Path(cohort_dir)
        # NEVER create REPRODUCIBILITY directories - only use target-first structure
        if "REPRODUCIBILITY" in str(cohort_dir):
            logger.warning(f"⚠️ Skipping diff save to legacy REPRODUCIBILITY path: {cohort_dir}")
            return
        
        # ========================================================================
        # PATCH 2: Extract identifiers from cohort_dir path
        # Priority 1: Use DiffResult fields (explicit, preferred)
        # Priority 2: Parse target-first path: targets/<target>/reproducibility/<VIEW>/...
        # Priority 3: Parse legacy path: STAGE/VIEW/target/cohort=.../
        # ========================================================================
        stage = diff.prev_stage or "UNKNOWN"
        view = diff.prev_view or "UNKNOWN"
        target = "UNKNOWN"
        cohort_id = None
        base_output_dir = None
        target_cohort_dir = None
        
        VALID_VIEWS = ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']
        
        # Try to extract from cohort_dir path
        try:
            parts = Path(cohort_dir).parts
            parsed_from_path = False
            
            # Priority 1: If target is already in DiffResult, use it
            if hasattr(diff, 'target') and diff.target and diff.target != "UNKNOWN":
                target = diff.target
                parsed_from_path = True
            
            # Priority 2: Try target-first path structure: targets/<target>/reproducibility/<VIEW>/...
            if not parsed_from_path and "targets" in parts:
                for i, part in enumerate(parts):
                    if part == "targets" and i + 1 < len(parts):
                        target = parts[i + 1]  # Target name is right after "targets"
                        # Find view after "reproducibility"
                        for j in range(i + 2, len(parts)):
                            if parts[j] == "reproducibility" and j + 1 < len(parts):
                                if parts[j + 1] in VALID_VIEWS:
                                    view = parts[j + 1]
                                break
                        # Find cohort_id
                        for j in range(i, len(parts)):
                            if parts[j].startswith('cohort='):
                                cohort_id = parts[j].replace('cohort=', '')
                                break
                        parsed_from_path = True
                        break
            
            # Priority 3: Legacy path structure: STAGE/VIEW/target/cohort=.../
            if not parsed_from_path:
                for i, part in enumerate(parts):
                    if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING']:
                        stage = part
                        if i + 1 < len(parts) and parts[i+1] in VALID_VIEWS:
                            view = parts[i+1]
                            if i + 2 < len(parts) and not parts[i+2].startswith('cohort='):
                                target = parts[i+2]
                        # Find cohort_id
                        for j in range(i, len(parts)):
                            if parts[j].startswith('cohort='):
                                cohort_id = parts[j].replace('cohort=', '')
                                break
                        # Warn that we're using legacy path parsing
                        logger.debug(f"Using legacy path parsing for diff (stage={stage}, item={target})")
                        break
            
            # If still UNKNOWN, log a warning
            if target == "UNKNOWN":
                logger.warning(
                    f"PATCH 2: Could not extract target from cohort_dir path: {cohort_dir}. "
                    f"Path parsing failed. Consider passing target explicitly."
                )
            
            # Only create target-first structure for TARGET_RANKING and FEATURE_SELECTION
            if stage in ['TARGET_RANKING', 'FEATURE_SELECTION'] and target != "UNKNOWN" and cohort_id:
                # Find base output directory (run directory)
                temp_dir = cohort_dir
                for _ in range(10):  # Limit depth
                    if (temp_dir / "targets").exists() or (temp_dir.parent / "targets").exists():
                        # Found run directory
                        if (temp_dir / "targets").exists():
                            base_output_dir = temp_dir
                        else:
                            base_output_dir = temp_dir.parent
                        break
                    if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                        break
                    temp_dir = temp_dir.parent
                
                if base_output_dir:
                    try:
                        from TRAINING.orchestration.utils.target_first_paths import (
                            get_target_reproducibility_dir, ensure_target_structure
                        )
                        
                        # CRITICAL FIX: Extract symbol FIRST, then determine view
                        # If a symbol exists, this is SYMBOL_SPECIFIC regardless of what path parsing said
                        symbol_for_target = None
                        parts = Path(cohort_dir).parts
                        for part in parts:
                            if part.startswith('symbol='):
                                symbol_for_target = part.replace('symbol=', '')
                                break
                        
                        # Normalize view: if symbol exists, it's SYMBOL_SPECIFIC
                        # Otherwise, use path-parsed view with INDIVIDUAL->SYMBOL_SPECIFIC normalization
                        if symbol_for_target:
                            view_for_target = 'SYMBOL_SPECIFIC'
                        else:
                            view_for_target = view
                            if stage == 'FEATURE_SELECTION' and view == 'INDIVIDUAL':
                                view_for_target = 'SYMBOL_SPECIFIC'
                        
                        # Ensure target structure exists
                        ensure_target_structure(base_output_dir, target)
                        
                        # Build target-first reproducibility path
                        # For CROSS_SECTIONAL: targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<cohort_id>/
                        # For SYMBOL_SPECIFIC: targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<cohort_id>/
                        target_repro_dir = get_target_reproducibility_dir(base_output_dir, target)
                        
                        if view_for_target == "SYMBOL_SPECIFIC" and symbol_for_target:
                            # Include symbol in path to prevent overwriting
                            target_cohort_dir = target_repro_dir / view_for_target / f"symbol={symbol_for_target}" / f"cohort={cohort_id}"
                        else:
                            target_cohort_dir = target_repro_dir / view_for_target / f"cohort={cohort_id}"
                        target_cohort_dir.mkdir(parents=True, exist_ok=True)
                        logger.debug(f"Created target-first cohort directory for diffs: {target_cohort_dir}")
                    except Exception as e:
                        logger.debug(f"Failed to create target-first structure for diffs (non-critical): {e}")
                        target_cohort_dir = None
        except Exception as e:
            logger.debug(f"Failed to extract identifiers for target-first structure: {e}")
            target_cohort_dir = None
        
        # Use target-first directory for all writes - NEVER use legacy REPRODUCIBILITY paths
        if target_cohort_dir is None:
            # If we couldn't create target-first structure, check if cohort_dir is already target-first
            if "REPRODUCIBILITY" not in str(cohort_dir):
                # It's already target-first, use it
                target_cohort_dir = cohort_dir
            else:
                logger.warning(f"⚠️ Cannot save diff: failed to create target-first structure from {cohort_dir}")
                return
        
        # Use target-first directory instead of original cohort_dir
        cohort_dir = target_cohort_dir
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # Tier A: Summary in diff_prev.json (lightweight, always present)
        # This includes: metric_deltas_count, impact_label, top_regressions, top_improvements
        prev_diff_dict = diff.to_dict()
        
        # Tier B: Structured per-metric deltas in metric_deltas.json (detailed, only if deltas exist)
        # This is the "real diff" with full structured deltas for each metric
        metric_deltas_file_path = None
        if diff.metric_deltas:
            metric_deltas_file = cohort_dir / "metric_deltas.json"
            # Use relative path from cohort_dir for portability
            metric_deltas_file_path = "metric_deltas.json"
            
            # Structure: keyed by metric name with full delta info
            metric_deltas_data = {
                'run_id': diff.current_run_id,
                'prev_run_id': diff.prev_run_id,
                'timestamp': datetime.now().isoformat(),
                # Identifiers for downstream joining
                'stage': stage,
                'view': view,
                'target': target,
                'metric_deltas': diff.metric_deltas,
                'summary': {
                    'total_metrics': len(diff.metric_deltas),
                    'impact_label': diff.summary.get('impact_label', 'none'),
                    'top_regressions': diff.summary.get('top_regressions', []),
                    'top_improvements': diff.summary.get('top_improvements', [])
                }
            }
            # Write to target-first structure only
            if target_cohort_dir:
                try:
                    target_metric_deltas_file = target_cohort_dir / "metric_deltas.json"
                    _write_atomic_json(target_metric_deltas_file, metric_deltas_data)
                    logger.debug(f"✅ Saved metric_deltas.json to target-first structure: {target_metric_deltas_file}")
                except Exception as e:
                    logger.warning(f"Failed to save metric_deltas.json to target-first structure: {e}")
            else:
                logger.warning(f"Target cohort directory not available, cannot save metric_deltas.json")
        
        # Add reference to metric_deltas.json in diff_prev.json summary (before writing)
        if metric_deltas_file_path:
            prev_diff_dict['summary']['metric_deltas_file'] = metric_deltas_file_path
        
        # Ensure summary includes impact classification (already computed in compute_diff)
        # Write to target-first structure only
        if target_cohort_dir:
            try:
                target_prev_diff_file = target_cohort_dir / "diff_prev.json"
                _write_atomic_json(target_prev_diff_file, prev_diff_dict)
                logger.debug(f"✅ Saved diff_prev.json to target-first structure: {target_prev_diff_file}")
            except Exception as e:
                logger.warning(f"Failed to save diff_prev.json to target-first structure: {e}")
        else:
            logger.warning(f"Target cohort directory not available, cannot save diff_prev.json")
        
        # Tier C: Full raw metrics remain in metrics.json (already written by MetricsWriter)
        # We don't duplicate them here - just reference the path
        
        # Save baseline diff if available (atomically) to target-first structure only
        if baseline_diff and target_cohort_dir:
            try:
                baseline_diff_dict = baseline_diff.to_dict()
                target_baseline_diff_file = target_cohort_dir / "diff_baseline.json"
                _write_atomic_json(target_baseline_diff_file, baseline_diff_dict)
                logger.debug(f"✅ Saved diff_baseline.json to target-first structure: {target_baseline_diff_file}")
            except Exception as e:
                logger.warning(f"Failed to save diff_baseline.json to target-first structure: {e}")
        
        if target_cohort_dir:
            logger.debug(f"✅ Saved diffs to {target_cohort_dir}")
    
    def _emit_trend_time_series(
        self,
        snapshot: NormalizedSnapshot,
        metrics: Dict[str, Any],
        cohort_dir: Path
    ) -> None:
        """
        Emit time series data for trend/drift analysis.
        
        Emits one row per metric key with:
        - identifiers: run_id, timestamp, comparison_group, stage, view, target, metric_name
        - values: auc, std_score, composite_score, mean_importance, pos_rate, n_effective_cs, n_models
        - derived: (rolling baseline, drift_z, ema, cusum computed later by trend analyzer)
        
        Stores in metrics_timeseries.parquet at target level (one level up from cohort_dir)
        for aggregation across runs.
        """
        if not metrics:
            return
        
        # Get comparison group key
        comparison_group_key = None
        if snapshot.comparison_group:
            comparison_group_key = snapshot.comparison_group.to_key(snapshot.stage, strict=False)
        
        # Extract target from target (for TARGET_RANKING) or from other sources
        target = snapshot.target or "unknown"
        
        # Build time series rows - one per metric
        # Key metrics to track
        metric_fields = [
            'auc', 'std_score', 'composite_score', 'mean_importance',
            'pos_rate', 'n_effective_cs', 'n_models'
        ]
        
        rows = []
        for metric_name in metric_fields:
            if metric_name not in metrics:
                continue
            
            value = metrics.get(metric_name)
            # Skip non-numeric values
            if not isinstance(value, (int, float)) or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
                continue
            
            row = {
                'run_id': snapshot.run_id,
                'timestamp': snapshot.timestamp,
                'comparison_group': comparison_group_key,
                'stage': snapshot.stage,
                'view': snapshot.view or 'UNKNOWN',
                'target': target,
                'metric_name': metric_name,
                'value': float(value),
                # Include key context metrics for trend analysis (same values across all rows for this run)
                'auc': metrics.get('auc') if 'auc' in metrics else None,
                'std_score': metrics.get('std_score') if 'std_score' in metrics else None,
                'composite_score': metrics.get('composite_score') if 'composite_score' in metrics else None,
                'mean_importance': metrics.get('mean_importance') if 'mean_importance' in metrics else None,
                'pos_rate': metrics.get('pos_rate') if 'pos_rate' in metrics else None,
                'n_effective_cs': metrics.get('n_effective_cs') if 'n_effective_cs' in metrics else None,
                'n_models': metrics.get('n_models') if 'n_models' in metrics else None
            }
            rows.append(row)
        
        if not rows:
            return
        
        # Store in metrics/ folder (not reproducibility/) for consistency
        # Map cohort_dir to metrics/ folder using helper function
        try:
            from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
            metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir)
            if metrics_dir:
                timeseries_file = metrics_dir / "metrics_timeseries.parquet"
            else:
                # Fallback: try to construct path manually
                logger.warning(f"Could not map cohort_dir to metrics path, using fallback: {cohort_dir}")
                # Extract target and view from cohort_dir
                parts = Path(cohort_dir).parts
                target = None
                view = None
                symbol = None
                for i, part in enumerate(parts):
                    if part == "targets" and i + 1 < len(parts):
                        target = parts[i + 1]
                    if part == "reproducibility" and i + 1 < len(parts):
                        view = parts[i + 1]
                        if view == "SYMBOL_SPECIFIC" and i + 2 < len(parts):
                            symbol_part = parts[i + 2]
                            if symbol_part.startswith("symbol="):
                                symbol = symbol_part.replace("symbol=", "")
                
                if target and view:
                    # Find base_output_dir
                    base_output_dir = Path(cohort_dir)
                    for _ in range(10):
                        if base_output_dir.name == "targets":
                            base_output_dir = base_output_dir.parent
                            break
                        if not base_output_dir.parent.exists():
                            break
                        base_output_dir = base_output_dir.parent
                    
                    from TRAINING.orchestration.utils.target_first_paths import get_target_metrics_dir
                    metrics_dir = get_target_metrics_dir(base_output_dir, target) / f"view={view}"
                    if symbol:
                        metrics_dir = metrics_dir / f"symbol={symbol}"
                    timeseries_file = metrics_dir / "metrics_timeseries.parquet"
                else:
                    # Last resort: use old location
                    target_dir = cohort_dir.parent
                    timeseries_file = target_dir / "metrics_timeseries.parquet"
        except Exception as e:
            logger.warning(f"Failed to map cohort_dir to metrics path: {e}, using fallback")
            # Fallback to old location
            target_dir = cohort_dir.parent
            timeseries_file = target_dir / "metrics_timeseries.parquet"
        
        try:
            # Read existing data if file exists
            existing_df = None
            if timeseries_file.exists():
                try:
                    existing_df = pd.read_parquet(timeseries_file)
                except Exception as e:
                    logger.debug(f"Could not read existing timeseries file {timeseries_file}: {e}")
            
            # Create new DataFrame from current rows
            new_df = pd.DataFrame(rows)
            
            # Append to existing data
            if existing_df is not None and len(existing_df) > 0:
                # Combine and deduplicate by (run_id, metric_name) to avoid duplicates
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                # Remove duplicates, keeping the most recent (last) entry
                combined_df = combined_df.drop_duplicates(
                    subset=['run_id', 'metric_name'],
                    keep='last'
                )
                # Sort by timestamp for easier querying
                combined_df = combined_df.sort_values('timestamp')
            else:
                combined_df = new_df
            
            # Write back to parquet
            timeseries_file.parent.mkdir(parents=True, exist_ok=True)
            combined_df.to_parquet(
                timeseries_file,
                index=False,
                engine='pyarrow',
                compression='snappy'
            )
            
            logger.debug(f"✅ Emitted {len(rows)} time series rows to {timeseries_file}")
        except Exception as e:
            logger.warning(f"Failed to emit trend time series to {timeseries_file}: {e}")
            # Don't fail the run if trend emission fails
    
    def finalize_run(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_dir: Path,
        cohort_metadata: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        resolved_metadata: Optional[Dict[str, Any]] = None,
        run_identity: Optional[Any] = None,  # NEW: RunIdentity SST object
        prediction_fingerprint: Optional[Dict] = None,  # NEW: Prediction fingerprint dict
    ) -> Optional[Dict[str, Any]]:
        """
        Finalize run: create snapshot, compute diffs, update baseline.
        
        This is the main entry point - call this after each run completes.
        
        CRITICAL: For SST consistency, pass `resolved_metadata` (the in-memory metadata dict
        that will be written to metadata.json). This ensures the snapshot/diff computation uses
        the exact same data that gets persisted, preventing coherence drift.
        
        Args:
            stage: Pipeline stage
            run_data: Run data from reproducibility tracker
            cohort_dir: Cohort directory
            cohort_metadata: Cohort metadata (fallback if resolved_metadata not provided)
            additional_data: Additional data
            resolved_metadata: In-memory metadata dict (SST - use this if available)
            run_identity: RunIdentity SST object with authoritative signatures
            prediction_fingerprint: Prediction fingerprint dict for predictions_sha256
        
        Returns:
            Diff telemetry data dict
        """
        # CRITICAL: Validate resolved_metadata matches current stage to prevent cross-stage contamination
        # Note: We don't strictly validate run_id format - run_ids are identifiers, not reproducibility factors.
        # What matters for reproducibility are fingerprints (data, config, feature, target) and comparison groups.
        if resolved_metadata:
            resolved_stage = resolved_metadata.get("stage")
            
            if resolved_stage != stage:
                raise ValueError(
                    f"Stage mismatch: resolved_metadata stage={resolved_stage}, current={stage}. "
                    f"This indicates cross-stage metadata contamination. Ensure full_metadata is stage-scoped."
                )
            # Run ID format differences (e.g., underscores vs T) don't affect reproducibility
            # Only log a debug message if formats differ significantly, but don't fail
            resolved_run_id = resolved_metadata.get("run_id")
            current_run_id = run_data.get('run_id') or run_data.get('timestamp')
            if resolved_run_id and current_run_id and resolved_run_id != current_run_id:
                # Normalize both to check if they represent the same timestamp
                # If they're just format differences, it's fine
                logger.debug(f"Run ID format differs (non-critical): resolved={resolved_run_id}, current={current_run_id}")
            
            # CRITICAL: Validate required fields are present and non-null for this stage
            # This ensures we catch incomplete SST before snapshot computation
            # BUT: Allow fallback extraction from run_data/additional_data for fields that might be set later
            required_fields = self._get_required_fields_for_stage(stage)
            missing_fields = []
            null_fields = []
            
            # Try to fill in missing fields from run_data or additional_data as fallback
            # This handles cases where metadata is built incrementally
            for field in required_fields:
                if field not in resolved_metadata or resolved_metadata[field] is None:
                    # Try fallback sources
                    fallback_value = None
                    if field == "target":
                        # For TARGET_RANKING, try multiple fallback sources
                        # target is passed to log_comparison() but might not be in run_data
                        fallback_value = (
                            run_data.get('target') or
                            run_data.get('target') or
                            run_data.get('target') or
                            (additional_data.get('target') if additional_data else None) or
                            (additional_data.get('target') if additional_data else None) or
                            (additional_data.get('target') if additional_data else None)
                        )
                        # Last resort: try to extract from cohort_dir path (e.g., .../TARGET_RANKING/SYMBOL_SPECIFIC/{target}/...)
                        if not fallback_value and cohort_dir:
                            try:
                                parts = Path(cohort_dir).parts
                                # Look for target in path (usually after view name)
                                for i, part in enumerate(parts):
                                    if part in ['TARGET_RANKING', 'FEATURE_SELECTION', 'TRAINING'] and i + 2 < len(parts):
                                        # Next part might be view, then target
                                        if parts[i+1] in ['CROSS_SECTIONAL', 'SYMBOL_SPECIFIC', 'LOSO', 'INDIVIDUAL']:
                                            if i + 2 < len(parts) and not parts[i+2].startswith('symbol=') and not parts[i+2].startswith('cohort='):
                                                fallback_value = parts[i+2]
                                                break
                                        elif not parts[i+1].startswith('symbol=') and not parts[i+1].startswith('cohort='):
                                            # No view, target is next
                                            fallback_value = parts[i+1]
                                            break
                            except Exception:
                                pass  # Don't fail if path parsing fails
                    
                    if fallback_value:
                        resolved_metadata[field] = fallback_value
                        logger.debug(f"Filled missing {field} from fallback: {fallback_value}")
                    elif field not in resolved_metadata:
                        missing_fields.append(field)
                    elif resolved_metadata[field] is None:
                        null_fields.append(field)
            
            if missing_fields or null_fields:
                error_parts = []
                if missing_fields:
                    error_parts.append(f"missing: {', '.join(missing_fields)}")
                if null_fields:
                    error_parts.append(f"null: {', '.join(null_fields)}")
                raise ValueError(
                    f"Incomplete resolved_metadata for stage={stage}: {', '.join(error_parts)}. "
                    f"Required fields must be present and non-null before finalize_run(). "
                    f"Ensure full_metadata is built AFTER all required fields are finalized."
                )
        
        # Create normalized snapshot (prefer resolved_metadata for SST consistency)
        # NEW: Pass run_identity and prediction_fingerprint for authoritative signatures
        snapshot = self.normalize_snapshot(
            stage=stage,
            run_data=run_data,
            cohort_metadata=cohort_metadata,
            additional_data=additional_data,
            cohort_dir=cohort_dir,
            resolved_metadata=resolved_metadata,
            run_identity=run_identity,
            prediction_fingerprint=prediction_fingerprint
        )
        
        # Save snapshot
        self.save_snapshot(snapshot, cohort_dir)
        
        # Find previous comparable run
        prev_snapshot = self.find_previous_comparable(snapshot)
        
        # Try to find previous snapshot's cohort directory for trend comparison
        prev_cohort_dir = None
        if prev_snapshot and prev_snapshot.stage and prev_snapshot.view and prev_snapshot.target:
            # Reconstruct path similar to find_previous_comparable
            target_clean = prev_snapshot.target.replace('/', '_').replace('\\', '_')
            if hasattr(self, 'run_dir') and self.run_dir:
                results_dir = self.run_dir
                while results_dir.parent.exists() and results_dir.name != "RESULTS":
                    results_dir = results_dir.parent
                    if results_dir.name == "RESULTS":
                        break
                
                if results_dir.name == "RESULTS":
                    # Search for the previous run's cohort directory
                    runs_dir = results_dir / "runs"
                    if runs_dir.exists():
                        for cg_dir in runs_dir.iterdir():
                            if not cg_dir.is_dir() or not cg_dir.name.startswith("cg-"):
                                continue
                            for run_dir in cg_dir.iterdir():
                                if not run_dir.is_dir():
                                    continue
                                stage_dir = run_dir / "REPRODUCIBILITY" / prev_snapshot.stage / prev_snapshot.view / target_clean
                                if stage_dir.exists():
                                    for cohort_subdir in stage_dir.iterdir():
                                        if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                            snapshot_file = cohort_subdir / "snapshot.json"
                                            if snapshot_file.exists():
                                                try:
                                                    with open(snapshot_file, 'r') as f:
                                                        snapshot_data = json.load(f)
                                                        if snapshot_data.get('run_id') == prev_snapshot.run_id:
                                                            prev_cohort_dir = cohort_subdir
                                                            break
                                                except Exception:
                                                    continue
                                    if prev_cohort_dir:
                                        break
                            if prev_cohort_dir:
                                break
        
        # Compute diff against previous
        if prev_snapshot:
            diff = self.compute_diff(snapshot, prev_snapshot, prev_cohort_dir=prev_cohort_dir, curr_cohort_dir=cohort_dir)
        else:
            # First run / no previous run: return stable shape with empty excluded factors
            # CRITICAL: If not comparable, severity must be CRITICAL with reason
            diff = DiffResult(
                prev_run_id=None,  # Use None instead of "none" for clarity
                current_run_id=snapshot.run_id,
                comparable=False,
                comparability_reason="No previous comparable run found",
                prev_timestamp=None,
                prev_snapshot_seq=None,
                prev_stage=None,
                prev_view=None,
                comparison_source=None,
                severity=ChangeSeverity.CRITICAL,
                severity_reason="No previous comparable run found - cannot determine changes",
                excluded_factors_changed={},  # Empty but present
                summary={
                    'total_changes': 0,
                    'input_changes': 0,
                    'process_changes': 0,
                    'output_changes': 0,
                    'metric_deltas_count': 0,
                    'excluded_factors_changed': False,
                    'excluded_factors_summary': None
                }
            )
        
        # Get or establish baseline (stored per-cohort for exact matching)
        metrics = snapshot.outputs.get('metrics', {})
        baseline_state, is_new = self.get_or_establish_baseline(snapshot, metrics, cohort_dir)
        
        # Compute diff against baseline
        baseline_diff = None
        if baseline_state:
            # Load baseline snapshot - search by run_id since keys now include target/view
            baseline_snapshot = None
            for snap in self._snapshots.values():
                if snap.run_id == baseline_state.baseline_run_id:
                    baseline_snapshot = snap
                    break
            if baseline_snapshot:
                # Try to find baseline snapshot's cohort directory
                baseline_cohort_dir = None
                if baseline_snapshot.stage and baseline_snapshot.view and baseline_snapshot.target:
                    target_clean = baseline_snapshot.target.replace('/', '_').replace('\\', '_')
                    if hasattr(self, 'run_dir') and self.run_dir:
                        results_dir = self.run_dir
                        while results_dir.parent.exists() and results_dir.name != "RESULTS":
                            results_dir = results_dir.parent
                            if results_dir.name == "RESULTS":
                                break
                        
                        if results_dir.name == "RESULTS":
                            runs_dir = results_dir / "runs"
                            if runs_dir.exists():
                                for cg_dir in runs_dir.iterdir():
                                    if not cg_dir.is_dir() or not cg_dir.name.startswith("cg-"):
                                        continue
                                    for run_dir in cg_dir.iterdir():
                                        if not run_dir.is_dir():
                                            continue
                                        stage_dir = run_dir / "REPRODUCIBILITY" / baseline_snapshot.stage / baseline_snapshot.view / target_clean
                                        if stage_dir.exists():
                                            for cohort_subdir in stage_dir.iterdir():
                                                if cohort_subdir.is_dir() and cohort_subdir.name.startswith("cohort="):
                                                    snapshot_file = cohort_subdir / "snapshot.json"
                                                    if snapshot_file.exists():
                                                        try:
                                                            with open(snapshot_file, 'r') as f:
                                                                snapshot_data = json.load(f)
                                                                if snapshot_data.get('run_id') == baseline_snapshot.run_id:
                                                                    baseline_cohort_dir = cohort_subdir
                                                                    break
                                                        except Exception:
                                                            continue
                                            if baseline_cohort_dir:
                                                break
                                    if baseline_cohort_dir:
                                        break
                
                baseline_diff = self.compute_diff(snapshot, baseline_snapshot, prev_cohort_dir=baseline_cohort_dir, curr_cohort_dir=cohort_dir)
        
        # Save diffs
        self.save_diff(diff, baseline_diff, cohort_dir)
        
        # Emit trend time series data for trend/drift analysis
        metrics = snapshot.outputs.get('metrics', {})
        if metrics:
            self._emit_trend_time_series(snapshot, metrics, cohort_dir)
        
        # Return diff data for integration into metadata/metrics
        diff_telemetry_data = {
            'diff': diff.to_dict(),
            'baseline_diff': baseline_diff.to_dict() if baseline_diff else None,
            'snapshot': snapshot.to_dict()
        }
        
        logger.info(f"✅ Telemetry finalized for {stage}:{snapshot.target or 'unknown'}")
        if diff.comparable:
            logger.info(f"   Changes: {len(diff.changed_keys)} keys, severity={diff.severity.value}")
            if diff.metric_deltas:
                for metric, delta in diff.metric_deltas.items():
                    logger.info(f"   {metric}: {delta['delta_abs']:+.4f} ({delta['delta_pct']:+.2f}%)")
            # Surface excluded factors loudly
            if diff.excluded_factors_changed and diff.summary.get('excluded_factors_summary'):
                logger.warning(f"   ⚠️  Excluded factors changed: {diff.summary['excluded_factors_summary']}")
        
        return diff_telemetry_data

