# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Metrics System (Sidecar-based, View-Isolated)

Writes model performance metrics as sidecar files next to existing artifacts (metadata.json, metrics.json, audit_report.json).
All metrics are stored locally on your infrastructure - no data is transmitted externally.

IMPORTANT: This system tracks MODEL PERFORMANCE METRICS only (e.g., ROC-AUC, R², feature importance).
It does NOT collect user data, personal information, or any data that leaves your infrastructure.
All metrics files are written to local disk in your REPRODUCIBILITY/ directory structure.

Key principles:
- Local-only: All metrics stored on your infrastructure, never transmitted externally
- Model performance tracking: Tracks ML model metrics (scores, feature importance, drift detection)
- View isolation: CROSS_SECTIONAL drift only compares to CROSS_SECTIONAL baselines
- SYMBOL_SPECIFIC drift only compares to SYMBOL_SPECIFIC baselines
- Sidecar placement: metrics files live in same cohort folder as existing JSONs
- Hierarchical rollups: per-target/symbol → view-level → stage-level
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd
import numpy as np

# SST: Import View and Stage enums for consistent view/stage handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage


def _write_atomic_json(file_path: Path, data: Dict[str, Any]) -> None:
    """
    Write JSON file atomically using temp file + rename with full durability.
    
    This ensures crash consistency AND power-loss safety:
    1. Write to temp file
    2. fsync(tempfile) - ensure data is on disk
    3. os.replace() - atomic rename (POSIX: atomic, Windows: best-effort)
    4. fsync(directory) - ensure directory entry is on disk
    
    This pattern is required for "audit-ready" systems that must survive sudden power loss.
    
    Args:
        file_path: Target file path
        data: Data to write (must be JSON-serializable)
    
    Raises:
        IOError: If write fails
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
        # This is critical for power-loss safety
        try:
            dir_fd = os.open(file_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)  # Sync directory entry
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            # Fallback: sync parent directory if available
            # Some systems don't support directory fsync
            pass
    except Exception as e:
        # Cleanup temp file on failure
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass
        raise IOError(f"Failed to write atomic JSON to {file_path}: {e}") from e

logger = logging.getLogger(__name__)


class MetricsWriter:
    """
    Writes model performance metrics as sidecar files in cohort directories.
    
    LOCAL-ONLY: All metrics are written to local disk. No data is transmitted externally.
    This system tracks MODEL PERFORMANCE METRICS (ROC-AUC, R², feature importance, etc.)
    for reproducibility and drift detection. It does NOT collect user data.
    
    PHASE 2: Unified canonical schema - single source of truth for metrics.
    
    Structure:
    - Sidecar files in each cohort folder: 
      * metrics.json (canonical schema, human-readable - supports both flat and nested structures)
      * metrics.parquet (same schema, wide format, queryable)
      * metrics_drift.json (comparison to baseline)
      * metrics_trend.json (temporal trends)
    - View-level rollups: CROSS_SECTIONAL/metrics_rollup.json, SYMBOL_SPECIFIC/metrics_rollup.json
    - Stage-level container: TARGET_RANKING/metrics_rollup.json
    
    Canonical schema (metrics.json/parquet):
    Supports both old flat structure and new grouped structure:
    - Old flat: {"auc": 0.5, "std_score": 0.1, ...}
    - New grouped: {"primary_metric": {"mean": 0.5, "std": 0.1}, "coverage": {"n_cs_valid": 11}, ...}
    
    Nested structures (dicts, lists) are stored as-is for backward compatibility.
    
    Privacy: All metrics are stored locally in REPRODUCIBILITY/ directory.
    No network calls, no external transmission, no user data collection.
    """
    
    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        baselines: Optional[Dict[str, Any]] = None,
        drift: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize metrics writer.
        
        Args:
            output_dir: Base output directory (REPRODUCIBILITY/ is under this)
            enabled: Whether metrics is enabled
            baselines: Baseline configuration (previous_run, rolling_window_k, last_good_run)
            drift: Drift detection configuration
        """
        self.output_dir = Path(output_dir)
        self.enabled = enabled
        
        # Default baselines
        self.baselines = {
            "previous_run": True,
            "rolling_window_k": 10,
            "last_good_run": True
        }
        if baselines:
            self.baselines.update(baselines)
        
        # Default drift config
        self.drift = {
            "psi_threshold": 0.2,
            "ks_threshold": 0.1
        }
        if drift:
            self.drift.update(drift)
    
    def write_cohort_metrics(
        self,
        cohort_dir: Path,
        stage: Union[str, Stage],  # SST: Accept both string and Stage enum
        view: Union[str, View],  # SST: Accept both string and View enum
        target: Optional[str],
        symbol: Optional[str],
        run_id: str,
        metrics: Dict[str, Any],
        baseline_key: Optional[str] = None,
        diff_telemetry: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write metrics sidecar files for a cohort.
        
        Args:
            cohort_dir: Path to cohort directory (where metadata.json, metrics.json live)
            stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, etc.)
            view: View type (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
            target: Target name (optional)
            symbol: Symbol name (optional, only for SYMBOL_SPECIFIC)
            run_id: Current run identifier
            metrics: Metrics dictionary (from run_data)
            baseline_key: Optional baseline key for drift comparison
        """
        if not self.enabled:
            logger.debug("Metrics is disabled, skipping write")
            return
        
        cohort_dir = Path(cohort_dir)
        
        # SST: Normalize stage and view to strings for JSON serialization
        if isinstance(stage, Stage):
            stage = stage.value
        elif hasattr(stage, 'value'):
            stage = stage.value
        else:
            stage = str(stage)
        
        if isinstance(view, View):
            view = view.value
        elif hasattr(view, 'value'):
            view = view.value
        else:
            view = str(view)
        
        # Create cohort directory if it doesn't exist (metrics should still work even if
        # _save_to_cohort didn't create it yet, or if there's a race condition)
        if not cohort_dir.exists():
            try:
                cohort_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created cohort directory for metrics: {cohort_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to create cohort directory {cohort_dir}: {e}")
                # Fallback: try writing at target level instead
                if target and view:
                    fallback_dir = self._get_fallback_metrics_dir(stage, view, target, symbol)
                    if fallback_dir:
                        logger.info(f"📁 Writing metrics to fallback location: {fallback_dir}")
                        try:
                            self._write_metrics(fallback_dir, run_id, metrics, stage=stage, reproducibility_mode="COHORT_AWARE")
                            if baseline_key:
                                self._write_drift(
                                    fallback_dir, stage, view, target, symbol, run_id, metrics, baseline_key
                                )
                            logger.info(f"✅ Metrics written to fallback location: {fallback_dir}")
                        except Exception as e2:
                            logger.error(f"❌ Failed to write metrics to fallback location {fallback_dir}: {e2}")
                    else:
                        logger.warning(f"⚠️  Could not determine fallback metrics directory for stage={stage}, view={view}, target={target}")
                else:
                    logger.warning(f"⚠️  Cannot use fallback: missing target or view (target={target}, view={view})")
                return
        
        # SST Architecture (Option A): Write canonical metrics to cohort_dir (reproducibility location)
        # metrics.parquet is canonical, metrics.json is debug export
        # Then create reference pointer in targets/<target>/metrics/ for convenience
        
        # Write canonical metrics to cohort_dir (reproducibility location)
        self._write_metrics(
            cohort_dir, run_id, metrics,
            stage=stage,
            reproducibility_mode="COHORT_AWARE",
            diff_telemetry=diff_telemetry
        )
        logger.debug(f"✅ Wrote canonical metrics to cohort directory: {cohort_dir}")
        
        # Create reference pointer in targets/<target>/metrics/ for convenience
        if target:
            try:
                from TRAINING.orchestration.utils.target_first_paths import (
                    get_target_metrics_dir, ensure_target_structure
                )
                # Find base output directory (walk up from cohort_dir to find run root) using SST helper
                from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                base_output_dir = get_run_root(cohort_dir)
                
                # If we found a run directory, create reference pointer
                if (base_output_dir / "targets").exists() or (base_output_dir / "REPRODUCIBILITY").exists():
                    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                    target_clean = normalize_target_name(target)
                    ensure_target_structure(base_output_dir, target_clean)
                    target_metrics_dir = get_target_metrics_dir(base_output_dir, target_clean)
                    
                    # Organize by view if multiple views exist
                    # SST: Use View enum for comparison
                    view_enum = View.from_string(view) if isinstance(view, str) else view
                    if view_enum in (View.CROSS_SECTIONAL, View.SYMBOL_SPECIFIC):
                        view_metrics_dir = target_metrics_dir / f"view={view_enum.value}"
                        view_metrics_dir.mkdir(parents=True, exist_ok=True)
                        
                        # For SYMBOL_SPECIFIC, organize by symbol
                        if view_enum == View.SYMBOL_SPECIFIC and symbol:
                            symbol_metrics_dir = view_metrics_dir / f"symbol={symbol}"
                            symbol_metrics_dir.mkdir(parents=True, exist_ok=True)
                            ref_dir = symbol_metrics_dir
                        else:
                            ref_dir = view_metrics_dir
                    else:
                        ref_dir = target_metrics_dir
                    
                    # Write reference pointer to canonical location
                    self._write_metrics_reference(
                        ref_dir, cohort_dir, run_id, target, view, symbol, base_output_dir
                    )
            except Exception as e:
                logger.debug(f"Failed to write metrics reference pointer: {e}")
        
        # Write metrics_drift.json (comparison to baseline)
        if baseline_key:
            try:
                self._write_drift(
                    cohort_dir, stage, view, target, symbol, run_id, metrics, baseline_key
                )
                logger.debug(f"✅ Wrote metrics_drift to {cohort_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to write metrics_drift to {cohort_dir}: {e}")
    
    def _write_metrics(
        self,
        cohort_dir: Path,
        run_id: str,
        metrics: Dict[str, Any],
        stage: Optional[Union[str, Stage]] = None,  # SST: Accept both string and Stage enum
        reproducibility_mode: str = "COHORT_AWARE",
        diff_telemetry: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write metrics.json and metrics.parquet with unified canonical schema.
        
        PHASE 2: Unified schema - single source of truth for metrics.
        - metrics.json: Human-readable JSON (supports both flat and nested structures)
        - metrics.parquet: Queryable Parquet (same schema, wide format)
        
        Schema (canonical):
        Supports both old flat structure and new grouped structure:
        - Old flat: {"auc": 0.5, "std_score": 0.1, ...}
        - New grouped: {"primary_metric": {"mean": 0.5, "std": 0.1}, "coverage": {"n_cs_valid": 11}, ...}
        
        Nested structures (dicts, lists) are stored as-is for backward compatibility.
        """
        # Build canonical flat schema (same as reproducibility_tracker metrics.json)
        metrics_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "reproducibility_mode": reproducibility_mode,
        }
        
        # Add stage if provided
        # SST: Ensure stage is a string (handle enum inputs)
        if stage:
            if isinstance(stage, Stage):
                metrics_data["stage"] = stage.value
            elif hasattr(stage, 'value'):
                metrics_data["stage"] = stage.value
            else:
                metrics_data["stage"] = str(stage)
        
        # Add all metrics as flat keys (exclude metadata fields)
        for key, value in metrics.items():
            if key in ['timestamp', 'cohort_metadata', 'additional_data']:
                continue
            
            # Convert to appropriate type
            if isinstance(value, (int, float, np.integer, np.floating)):
                metrics_data[key] = float(value)
            elif isinstance(value, (bool, np.bool_)):
                metrics_data[key] = bool(value)
            elif isinstance(value, (str, type(None))):
                metrics_data[key] = value
            elif isinstance(value, (list, dict)):
                # Store complex values as-is (JSON serializable)
                metrics_data[key] = value
            else:
                # Convert other types to string
                metrics_data[key] = str(value)
        
        # NEW: Add lightweight diff telemetry fields to metrics (queryable/aggregatable)
        # 
        # CRITICAL DESIGN PRINCIPLE: metadata.json is the Single Source of Truth (SST).
        # metrics.json contains only DERIVED, lightweight fields for dashboards/alerts.
        # 
        # This split prevents:
        # - Bloated metrics store (high-cardinality blobs)
        # - Loss of forensic detail when something regresses
        # 
        # Rules:
        # - metadata.json = canonical, high-detail, high-cardinality, audit trail (SST)
        # - metrics.json = low-cardinality, queryable signals (derived, disposable)
        # - If metrics need regeneration, derive from metadata.json
        # - Never treat metrics.json as authoritative - always refer to metadata.json for truth
        if diff_telemetry:
            diff = diff_telemetry.get('diff', {})
            snapshot = diff_telemetry.get('snapshot', {})
            
            # Lightweight fields for querying/aggregation
            # DO NOT include: full excluded_factors.changes, full comparison_group, full fingerprints
            # These belong only in metadata.json (SST)
            excluded_factors = diff.get('excluded_factors_changed', {})
            
            # Count rule: Number of leaf keys whose value differs (one key = one change)
            # PRECISE DEFINITION: Matches _count_excluded_factors_changed() in diff_telemetry.py
            # This ensures consistency between count and summary formatter ("(+N more)" display)
            excluded_factors_count = (
                len(excluded_factors.get('hyperparameters', {})) +
                (1 if excluded_factors.get('train_seed') else 0) +
                len([k for k in excluded_factors.get('versions', {}).keys() 
                     if k in ['python_version', 'cuda_version', 'library_versions']] if excluded_factors.get('versions') else [])
            ) if excluded_factors else 0
            
            # diff_telemetry_digest: Copy from metadata.json (where it's computed)
            # This allows cross-checking that metrics correspond to metadata without recomputing
            # Algorithm: Full SHA256 hash of canonical JSON (sorted keys, strict JSON-primitive-only) → 64 hex chars
            # See metadata.json for the exact blob that was hashed
            digest = snapshot.get('diff_telemetry_digest')
            if digest is None:
                # Fallback: try to get from full diff_telemetry if available
                digest = diff_telemetry.get('snapshot', {}).get('diff_telemetry_digest')
            
            metrics_data['diff_telemetry'] = {
                'comparable': 1 if diff.get('comparable', False) else 0,
                'excluded_factors_changed': 1 if excluded_factors else 0,
                'excluded_factors_changed_count': excluded_factors_count,  # Number of leaf keys that changed
                'excluded_factors_summary': diff.get('summary', {}).get('excluded_factors_summary'),
                'diff_telemetry_digest': digest,  # Hash of full metadata diff_telemetry (for integrity verification)
                # Optional: comparison_group_key (only if metrics backend can tolerate high cardinality)
                # Gate behind config flag or conditional - for now, removed to keep metrics lightweight
                # 'comparison_group_key': (
                #     snapshot.get('comparison_group', {}).get('key') 
                #     if isinstance(snapshot.get('comparison_group'), dict) 
                #     else None
                # )
            }
        
        metrics_file_json = cohort_dir / "metrics.json"
        metrics_file_parquet = cohort_dir / "metrics.parquet"
        
        # Write JSON FIRST - it's more resilient and serves as fallback if Parquet fails
        # This ensures we always have metrics even if Parquet serialization has issues
        try:
            _write_atomic_json(metrics_file_json, metrics_data)
            logger.debug(f"✅ Wrote metrics.json to {cohort_dir}")
        except Exception as e:
            logger.warning(f"Failed to write metrics.json to {cohort_dir}: {e}")
        
        # Now try Parquet - may fail for complex nested types
        try:
            # SST: Use safe_dataframe_from_dict to handle enums and stringify keys
            from TRAINING.common.utils.file_utils import safe_dataframe_from_dict
            df_metrics = safe_dataframe_from_dict(metrics_data)
            df_metrics.to_parquet(metrics_file_parquet, index=False, engine='pyarrow', compression='snappy')
            logger.debug(f"✅ Wrote metrics.parquet to {cohort_dir}")
        except Exception as e:
            logger.warning(f"Failed to write metrics.parquet to {cohort_dir}: {e}")
    
    def _prepare_for_parquet(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for Parquet serialization by converting non-string dict keys.
        
        DEPRECATED: Use safe_dataframe_from_dict() instead, which handles both
        enum sanitization and key stringification.
        
        This method is kept for backward compatibility but delegates to
        the SST sanitization function.
        
        Args:
            data: Dictionary to prepare
            
        Returns:
            New dictionary with all nested dict keys converted to strings and enums to values
        """
        # SST: Use shared sanitization function
        from TRAINING.common.utils.file_utils import sanitize_for_serialization
        return sanitize_for_serialization(data)
    
    def _write_metrics_reference(
        self,
        ref_dir: Path,
        canonical_cohort_dir: Path,
        run_id: str,
        target: str,
        view: str,
        symbol: Optional[str],
        base_output_dir: Path
    ) -> None:
        """
        Write reference pointer to canonical metrics location.
        
        Creates latest_ref.json in targets/<target>/metrics/ that points to the canonical
        metrics.parquet in reproducibility/cohort directory.
        
        Args:
            ref_dir: Directory where reference should be written (targets/<target>/metrics/view=.../)
            canonical_cohort_dir: Canonical cohort directory (targets/<target>/reproducibility/<view>/cohort=<id>/)
            run_id: Run identifier
            target: Target name
            view: View type (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
            symbol: Symbol name (optional, for SYMBOL_SPECIFIC)
            base_output_dir: Base run output directory
        """
        try:
            import hashlib
            
            # Compute relative path from ref_dir to canonical location
            # ref_dir: targets/<target>/metrics/view=CROSS_SECTIONAL/
            # canonical: targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<id>/
            try:
                relative_path = os.path.relpath(canonical_cohort_dir / "metrics.parquet", ref_dir)
            except ValueError:
                # If paths are on different drives (Windows), use absolute path
                relative_path = str(canonical_cohort_dir / "metrics.parquet")
            
            # Compute SHA256 of canonical metrics.parquet if it exists
            canonical_parquet = canonical_cohort_dir / "metrics.parquet"
            sha256 = None
            if canonical_parquet.exists():
                try:
                    with open(canonical_parquet, 'rb') as f:
                        sha256 = hashlib.sha256(f.read()).hexdigest()
                except Exception as e:
                    logger.debug(f"Failed to compute SHA256 for metrics reference: {e}")
            
            # Extract cohort_id from canonical_cohort_dir path
            cohort_id = None
            for part in canonical_cohort_dir.parts:
                if part.startswith("cohort="):
                    cohort_id = part.replace("cohort=", "")
                    break
            
            # Create reference data
            ref_data = {
                "run_id": run_id,
                "cohort_id": cohort_id,
                "target": target,
                "view": view,
                "symbol": symbol,
                "canonical_path": str(canonical_cohort_dir / "metrics.parquet"),
                "relative_path": relative_path,
                "sha256": sha256,
                "timestamp": datetime.now().isoformat(),
                "note": "This is a pointer to canonical metrics.parquet in reproducibility/cohort directory"
            }
            
            # Write reference file
            ref_file = ref_dir / "latest_ref.json"
            _write_atomic_json(ref_file, ref_data)
            logger.debug(f"✅ Wrote metrics reference pointer to {ref_file}")
        except Exception as e:
            logger.debug(f"Failed to write metrics reference pointer: {e}")
    
    @staticmethod
    def export_metrics_json_from_parquet(parquet_path: Path, json_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Export metrics.json from canonical metrics.parquet (for debugging).
        
        This function reads the canonical metrics.parquet and exports it as JSON.
        Used when metrics.json is requested but only parquet exists.
        
        Args:
            parquet_path: Path to canonical metrics.parquet
            json_path: Optional path to write JSON (if None, returns dict only)
        
        Returns:
            Metrics dictionary
        """
        try:
            df = pd.read_parquet(parquet_path)
            if len(df) == 0:
                return {}
            
            # Convert first row to dict (parquet has one row per cohort)
            metrics_dict = df.iloc[0].to_dict()
            
            # Write JSON if path provided
            if json_path:
                _write_atomic_json(json_path, metrics_dict)
                logger.debug(f"✅ Exported metrics.json from parquet to {json_path}")
            
            return metrics_dict
        except Exception as e:
            logger.warning(f"Failed to export metrics.json from parquet {parquet_path}: {e}")
            return {}
    
    def _write_drift(
        self,
        cohort_dir: Path,
        stage: Union[str, Stage],  # SST: Accept both string and Stage enum
        view: Union[str, View],  # SST: Accept both string and View enum
        target: Optional[str],
        symbol: Optional[str],
        run_id: str,
        current_metrics: Dict[str, Any],
        baseline_key: str
    ) -> None:
        """
        Write metrics_drift.json comparing current run to baseline.

        Enhanced with fingerprints, drift tiers, critical metrics, and sanity checks.
        Baseline key format: (stage, view, target[, symbol])
        This ensures view isolation: CS only compares to CS, SS only compares to SS.

        IMPORTANT: Only compares runs with matching fingerprints (seed, config_hash, data_fingerprint)
        to ensure we're checking determinism (same inputs) rather than comparing different configurations.
        """
        # SST: Normalize stage and view to strings for JSON serialization
        if isinstance(stage, Stage):
            stage = stage.value
        elif hasattr(stage, 'value'):
            stage = stage.value
        else:
            stage = str(stage)
        
        if isinstance(view, View):
            view = view.value
        elif hasattr(view, 'value'):
            view = view.value
        else:
            view = str(view)
        # Load current metadata for fingerprints FIRST (needed for baseline matching)
        current_metadata_file = cohort_dir / "metadata.json"
        current_metadata = {}
        if current_metadata_file.exists():
            try:
                with open(current_metadata_file, 'r') as f:
                    current_metadata = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load current metadata: {e}")
        
        # Find baseline cohort directory (filtered by matching fingerprints)
        baseline_cohort_dir = self._find_baseline_cohort(
            cohort_dir.parent.parent,  # Go up to view level
            stage, view, target, symbol, baseline_key,
            current_metadata=current_metadata  # Pass current metadata for fingerprint matching
        )
        
        if baseline_cohort_dir is None or not baseline_cohort_dir.exists():
            logger.debug(f"No baseline found for drift comparison: {baseline_key} (no matching fingerprints)")
            return
        
        # SST Architecture: Load baseline metrics from canonical location (cohort_dir) first
        baseline_data = None
        
        # 1. Try canonical metrics.parquet in cohort directory (SST)
        baseline_metrics_parquet = baseline_cohort_dir / "metrics.parquet"
        if baseline_metrics_parquet.exists():
            try:
                df = pd.read_parquet(baseline_metrics_parquet)
                if len(df) > 0:
                    baseline_data = df.iloc[0].to_dict()
                    logger.debug(f"✅ Loaded baseline metrics from canonical parquet: {baseline_metrics_parquet}")
            except Exception as e:
                logger.debug(f"Failed to read baseline metrics.parquet: {e}")
        
        # 2. Fallback to metrics.json in cohort directory (debug export)
        if baseline_data is None:
            baseline_metrics_json = baseline_cohort_dir / "metrics.json"
            if baseline_metrics_json.exists():
                try:
                    with open(baseline_metrics_json, 'r') as f:
                        baseline_data = json.load(f)
                    logger.debug(f"✅ Loaded baseline metrics from JSON: {baseline_metrics_json}")
                except Exception as e:
                    logger.debug(f"Failed to read baseline metrics.json: {e}")
        
        # 3. Fallback to reference pointer in metrics/ directory
        if baseline_data is None and target:
            try:
                from TRAINING.orchestration.utils.target_first_paths import get_target_metrics_dir
                # Find base output directory using SST helper
                from TRAINING.orchestration.utils.target_first_paths import run_root as get_run_root
                base_output_dir = get_run_root(baseline_cohort_dir)
                
                if (base_output_dir / "targets").exists():
                    from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
                    target_clean = normalize_target_name(target)
                    target_metrics_dir = get_target_metrics_dir(base_output_dir, target_clean)
                    # SST: Use View enum for comparison
                    view_enum = View.from_string(view) if isinstance(view, str) else view
                    if view_enum in (View.CROSS_SECTIONAL, View.SYMBOL_SPECIFIC):
                        view_metrics_dir = target_metrics_dir / f"view={view_enum.value}"
                        if view_enum == View.SYMBOL_SPECIFIC and symbol:
                            ref_dir = view_metrics_dir / f"symbol={symbol}"
                        else:
                            ref_dir = view_metrics_dir
                        
                        # Try to follow reference pointer
                        ref_file = ref_dir / "latest_ref.json"
                        if ref_file.exists():
                            try:
                                with open(ref_file, 'r') as f:
                                    ref_data = json.load(f)
                                canonical_path = Path(ref_data.get("canonical_path", ""))
                                if canonical_path.exists():
                                    baseline_data = self.export_metrics_json_from_parquet(canonical_path)
                                    logger.debug(f"✅ Loaded baseline metrics via reference pointer: {canonical_path}")
                            except Exception as e:
                                logger.debug(f"Failed to follow reference pointer: {e}")
            except Exception as e:
                logger.debug(f"Failed to load baseline metrics via reference: {e}")
        
        # 4. Last resort: try legacy locations for backward compatibility
        if baseline_data is None:
            try:
                from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                baseline_metrics_dir = get_metrics_path_from_cohort_dir(baseline_cohort_dir)
                if baseline_metrics_dir:
                    legacy_parquet = baseline_metrics_dir / "metrics.parquet"
                    if legacy_parquet.exists():
                        df = pd.read_parquet(legacy_parquet)
                        if len(df) > 0:
                            baseline_data = df.iloc[0].to_dict()
                    elif (baseline_metrics_dir / "metrics.json").exists():
                        with open(baseline_metrics_dir / "metrics.json", 'r') as f:
                            baseline_data = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load baseline metrics from legacy location: {e}")
        
        if baseline_data is None:
            logger.debug(f"Baseline metrics not found for baseline_cohort_dir: {baseline_cohort_dir}")
            return
        
        # Load baseline metadata for fingerprints
        baseline_metadata_file = baseline_cohort_dir / "metadata.json"
        baseline_metadata = {}
        if baseline_metadata_file.exists():
            try:
                with open(baseline_metadata_file, 'r') as f:
                    baseline_metadata = json.load(f)
            except Exception as e:
                logger.debug(f"Failed to load baseline metadata: {e}")
        
        try:
            
            # Extract baseline metrics
            if "metrics" in baseline_data:
                baseline_metrics = baseline_data["metrics"]
            else:
                baseline_metrics = baseline_data
            
            # Extract fingerprints
            baseline_git_commit = baseline_metadata.get("git_commit") or baseline_data.get("git_commit")
            current_git_commit = current_metadata.get("git_commit") or self._get_git_commit()
            
            baseline_config_hash = baseline_metadata.get("config_hash")
            current_config_hash = current_metadata.get("config_hash")
            
            # Compute data fingerprint (date range + n_effective + n_symbols)
            baseline_data_fingerprint = self._compute_data_fingerprint(baseline_metadata)
            current_data_fingerprint = self._compute_data_fingerprint(current_metadata)
            
            # Compute drift
            drift_results = {
                "current_run_id": run_id,
                "baseline_run_id": baseline_data.get("run_id", "unknown"),
                "baseline_key": baseline_key,
                "timestamp": datetime.now().isoformat(),
                "view": view,
                "target": target,
                "symbol": symbol,
                # Fingerprints for proving baseline is actually different
                "fingerprints": {
                    "baseline": {
                        "git_commit": baseline_git_commit,
                        "config_hash": baseline_config_hash,
                        "data_fingerprint": baseline_data_fingerprint,
                        "timestamp": baseline_metadata.get("created_at") or baseline_data.get("timestamp")
                    },
                    "current": {
                        "git_commit": current_git_commit,
                        "config_hash": current_config_hash,
                        "data_fingerprint": current_data_fingerprint,
                        "timestamp": current_metadata.get("created_at") or datetime.now().isoformat()
                    }
                },
                "drift_metrics": {}
            }
            
            # Extract seed for fingerprint matching
            baseline_seed = baseline_metadata.get("seed")
            current_seed = current_metadata.get("seed")
            
            # Sanity check: verify fingerprints match (required for determinism check)
            fingerprints_match = (
                baseline_seed == current_seed and
                baseline_config_hash == current_config_hash and
                baseline_data_fingerprint == current_data_fingerprint
            )
            
            if not fingerprints_match:
                # Fingerprints don't match - this shouldn't happen if _find_baseline_cohort worked correctly
                drift_results["sanity_check"] = {
                    "status": "FINGERPRINT_MISMATCH",
                    "message": f"Fingerprints don't match - seed: {baseline_seed} vs {current_seed}, config: {baseline_config_hash} vs {current_config_hash}, data: {baseline_data_fingerprint} vs {current_data_fingerprint}. This comparison is invalid for determinism checking."
                }
                # Still write the drift file but mark it as invalid
            elif run_id == baseline_data.get("run_id"):
                drift_results["sanity_check"] = {
                    "status": "SELF_COMPARISON",
                    "message": "Comparing run to itself - this should not happen"
                }
            else:
                # Fingerprints match and run_ids differ - valid determinism check
                drift_results["sanity_check"] = {
                    "status": "OK",
                    "message": "Baseline and current runs have matching fingerprints (seed, config, data) - valid determinism check"
                }
            
            # Critical metrics to track (silent killers)
            critical_metrics = [
                "label_window", "horizon", "lookahead_guard_state",
                "cv_scheme_id", "fold_count", "purge_gap",
                "leakage_flag", "leakage_events_count",
                "missingness_rate", "winsorization_clip", "outlier_rate"
            ]
            
            # Compare numeric metrics
            for metric_name in set(current_metrics.keys()) & set(baseline_metrics.keys()):
                curr_val = current_metrics.get(metric_name)
                base_val = baseline_metrics.get(metric_name)
                
                if not isinstance(curr_val, (int, float)) or not isinstance(base_val, (int, float)):
                    continue
                
                delta = float(curr_val) - float(base_val)
                
                # Fix rel_delta for zeros: explicit handling
                if base_val == 0:
                    if curr_val == 0:
                        rel_delta = 0.0
                        rel_delta_status = "both_zero"
                    else:
                        rel_delta = None
                        rel_delta_status = "undefined_zero_baseline"
                else:
                    rel_delta = delta / float(base_val)
                    rel_delta_status = "defined"
                
                # Classify drift with tiers (OK/WARN/ALERT)
                drift_tier = self._classify_drift_tier(delta, rel_delta, metric_name in critical_metrics)
                
                drift_results["drift_metrics"][metric_name] = {
                    "current": float(curr_val),
                    "baseline": float(base_val),
                    "delta": delta,
                    "rel_delta": rel_delta,
                    "rel_delta_status": rel_delta_status,
                    "baseline_zero": base_val == 0,
                    "tier": drift_tier,
                    "is_critical": metric_name in critical_metrics,
                    # Backward compatibility: legacy status field
                    "status": "STABLE" if drift_tier == "OK" else ("DRIFTING" if drift_tier == "WARN" else "DIVERGED")
                }
            
            # Track critical metrics from metadata if not in metrics
            for metric_name in critical_metrics:
                if metric_name not in drift_results["drift_metrics"]:
                    curr_val = current_metadata.get(metric_name) or current_metrics.get(metric_name)
                    base_val = baseline_metadata.get(metric_name) or baseline_metrics.get(metric_name)
                    
                    if curr_val is not None and base_val is not None:
                        if isinstance(curr_val, (int, float)) and isinstance(base_val, (int, float)):
                            delta = float(curr_val) - float(base_val)
                            rel_delta = delta / float(base_val) if base_val != 0 else None
                            rel_delta_status = "undefined_zero_baseline" if base_val == 0 else "defined"
                            
                            drift_tier = self._classify_drift_tier(delta, rel_delta, True)
                            drift_results["drift_metrics"][metric_name] = {
                                "current": float(curr_val),
                                "baseline": float(base_val),
                                "delta": delta,
                                "rel_delta": rel_delta,
                                "rel_delta_status": rel_delta_status,
                                "baseline_zero": base_val == 0,
                                "tier": drift_tier,
                                "is_critical": True,
                                # Backward compatibility: legacy status field
                                "status": "STABLE" if drift_tier == "OK" else ("DRIFTING" if drift_tier == "WARN" else "DIVERGED")
                            }
            
            # Write drift files (JSON for human-readable, Parquet for queryable)
            drift_file_json = cohort_dir / "metrics_drift.json"
            drift_file_parquet = cohort_dir / "metrics_drift.parquet"
            try:
                # Write JSON (human-readable)
                # SST: Use safe_json_dump to handle enums
                from TRAINING.common.utils.file_utils import safe_json_dump
                with open(drift_file_json, 'w') as f:
                    safe_json_dump(drift_results, f, indent=2)
                
                # Write Parquet (queryable, long format)
                # Flatten drift_metrics dict into DataFrame
                parquet_rows = []
                fingerprints = drift_results.get("fingerprints", {})
                sanity_check = drift_results.get("sanity_check", {})
                
                for metric_name, drift_info in drift_results.get("drift_metrics", {}).items():
                    if isinstance(drift_info, dict):
                        row = {
                            "current_run_id": drift_results.get("current_run_id"),
                            "baseline_run_id": drift_results.get("baseline_run_id"),
                            "baseline_key": drift_results.get("baseline_key"),
                            "view": drift_results.get("view"),
                            "target": drift_results.get("target"),
                            "symbol": drift_results.get("symbol"),
                            "timestamp": drift_results.get("timestamp"),
                            "metric_name": metric_name,
                            "current_value": drift_info.get("current"),
                            "baseline_value": drift_info.get("baseline"),
                            "delta": drift_info.get("delta"),
                            "rel_delta": drift_info.get("rel_delta"),
                            "rel_delta_status": drift_info.get("rel_delta_status"),
                            "baseline_zero": drift_info.get("baseline_zero", False),
                            "tier": drift_info.get("tier"),
                            "is_critical": drift_info.get("is_critical", False),
                            "status": drift_info.get("status"),  # Legacy field for backward compatibility
                            # Fingerprints
                            "baseline_git_commit": fingerprints.get("baseline", {}).get("git_commit"),
                            "current_git_commit": fingerprints.get("current", {}).get("git_commit"),
                            "baseline_config_hash": fingerprints.get("baseline", {}).get("config_hash"),
                            "current_config_hash": fingerprints.get("current", {}).get("config_hash"),
                            "baseline_data_fingerprint": fingerprints.get("baseline", {}).get("data_fingerprint"),
                            "current_data_fingerprint": fingerprints.get("current", {}).get("data_fingerprint"),
                            # Sanity check
                            "sanity_check_status": sanity_check.get("status"),
                            "sanity_check_message": sanity_check.get("message")
                        }
                        parquet_rows.append(row)
                
                if parquet_rows:
                    # SST: Sanitize each row before creating DataFrame (handles enums)
                    from TRAINING.common.utils.file_utils import sanitize_for_serialization
                    sanitized_rows = [sanitize_for_serialization(row) for row in parquet_rows]
                    df_drift = pd.DataFrame(sanitized_rows)
                    df_drift.to_parquet(drift_file_parquet, index=False, engine='pyarrow', compression='snappy')
            except Exception as e:
                logger.warning(f"Failed to write metrics_drift files to {cohort_dir}: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to compute drift for {cohort_dir}: {e}")
    
    def _find_baseline_cohort(
        self,
        view_dir: Path,
        stage: str,
        view: str,
        target: Optional[str],
        symbol: Optional[str],
        baseline_key: str,
        current_metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Path]:
        """
        Find baseline cohort directory using baseline_key, filtered by matching fingerprints.
        
        baseline_key format: (stage, view, target[, symbol])
        This ensures we only compare within the same view.
        
        IMPORTANT: Only returns baselines with matching fingerprints (seed, config_hash, data_fingerprint)
        to ensure we're checking determinism (same inputs) rather than comparing different configurations.
        
        Args:
            current_metadata: Current run metadata for fingerprint matching
        """
        # Parse baseline_key to extract target/symbol
        # Format: "TARGET_RANKING:CROSS_SECTIONAL:y_will_swing_high_60m_0.05" or
        #         "TARGET_RANKING:SYMBOL_SPECIFIC:y_will_swing_high_60m_0.05:AAPL"
        parts = baseline_key.split(":")
        if len(parts) < 3:
            return None
        
        baseline_stage, baseline_view, baseline_target = parts[0], parts[1], parts[2]
        baseline_symbol = parts[3] if len(parts) > 3 else None
        
        # Ensure view isolation: must match view
        if baseline_view != view:
            logger.warning(f"View mismatch: baseline_view={baseline_view} != current_view={view}")
            return None
        
        # Find target directory
        if baseline_target:
            target_dir = view_dir / baseline_target
            if not target_dir.exists():
                return None
        else:
            target_dir = view_dir
        
        # Find symbol directory (for SYMBOL_SPECIFIC)
        # SST: Use View enum for comparison
        view_enum = View.from_string(view) if isinstance(view, str) else view
        if baseline_symbol and view_enum == View.SYMBOL_SPECIFIC:
            symbol_dir = target_dir / f"symbol={baseline_symbol}"
            if not symbol_dir.exists():
                return None
        else:
            symbol_dir = target_dir
        
        # Extract current fingerprints for matching
        current_seed = current_metadata.get("seed") if current_metadata else None
        current_config_hash = current_metadata.get("config_hash") if current_metadata else None
        current_data_fingerprint = self._compute_data_fingerprint(current_metadata) if current_metadata else None
        
        # Find all cohort directories
        cohort_dirs = sorted(
            [d for d in symbol_dir.iterdir() if d.is_dir() and d.name.startswith("cohort=")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Filter by matching fingerprints (seed, config_hash, data_fingerprint)
        # Only compare runs with identical inputs to check determinism
        matching_cohorts = []
        for cohort_dir in cohort_dirs:
            # Load baseline metadata
            baseline_metadata_file = cohort_dir / "metadata.json"
            if not baseline_metadata_file.exists():
                continue
            
            try:
                with open(baseline_metadata_file, 'r') as f:
                    baseline_metadata = json.load(f)
            except Exception:
                continue
            
            # Extract baseline fingerprints
            baseline_seed = baseline_metadata.get("seed")
            baseline_config_hash = baseline_metadata.get("config_hash")
            baseline_data_fingerprint = self._compute_data_fingerprint(baseline_metadata)
            
            # Check if fingerprints match (all must match for determinism check)
            seed_match = (current_seed is None and baseline_seed is None) or (current_seed == baseline_seed)
            config_match = (current_config_hash is None and baseline_config_hash is None) or (current_config_hash == baseline_config_hash)
            data_match = (current_data_fingerprint is None and baseline_data_fingerprint is None) or (current_data_fingerprint == baseline_data_fingerprint)
            
            if seed_match and config_match and data_match:
                matching_cohorts.append(cohort_dir)
        
        # Skip current run (most recent), return previous matching run
        if len(matching_cohorts) > 1:
            return matching_cohorts[1]  # Second most recent matching run
        elif len(matching_cohorts) == 1:
            # Only one matching run - check if it's the current run
            # If it's the same cohort_id, skip it (self-comparison)
            if current_metadata:
                current_cohort_id = current_metadata.get("cohort_id")
                baseline_metadata_file = matching_cohorts[0] / "metadata.json"
                if baseline_metadata_file.exists():
                    try:
                        with open(baseline_metadata_file, 'r') as f:
                            baseline_metadata = json.load(f)
                        if baseline_metadata.get("cohort_id") == current_cohort_id:
                            return None  # Same cohort, skip self-comparison
                    except Exception:
                        pass
            return matching_cohorts[0]  # Only one matching run
        
        return None  # No matching baseline found
    
    def _classify_drift(self, delta: float, rel_delta: Optional[float]) -> str:
        """Classify drift status based on thresholds (legacy method, use _classify_drift_tier)."""
        tier = self._classify_drift_tier(delta, rel_delta, False)
        if tier == "OK":
            return "STABLE"
        elif tier == "WARN":
            return "DRIFTING"
        else:
            return "DIVERGED"
    
    def _classify_drift_tier(self, delta: float, rel_delta: Optional[float], is_critical: bool = False) -> str:
        """
        Classify drift into tiers: OK / WARN / ALERT.
        
        Args:
            delta: Absolute difference
            rel_delta: Relative difference (None if baseline is zero)
            is_critical: Whether this is a critical metric (stricter thresholds)
        
        Returns:
            "OK", "WARN", or "ALERT"
        """
        # Stricter thresholds for critical metrics
        if is_critical:
            warn_threshold_rel = 0.01  # 1% for critical metrics
            alert_threshold_rel = 0.03  # 3% for critical metrics
            warn_threshold_abs = 0.001
            alert_threshold_abs = 0.01
        else:
            warn_threshold_rel = 0.05  # 5% for normal metrics
            alert_threshold_rel = 0.20  # 20% for normal metrics
            warn_threshold_abs = 0.01
            alert_threshold_abs = 0.10
        
        if rel_delta is not None:
            abs_rel_delta = abs(rel_delta)
            if abs_rel_delta < warn_threshold_rel:
                return "OK"
            elif abs_rel_delta < alert_threshold_rel:
                return "WARN"
            else:
                return "ALERT"
        else:
            # Use absolute delta when rel_delta is undefined
            abs_delta = abs(delta)
            if abs_delta < warn_threshold_abs:
                return "OK"
            elif abs_delta < alert_threshold_abs:
                return "WARN"
            else:
                return "ALERT"
    
    def _compute_data_fingerprint(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Compute data fingerprint from metadata.
        
        Fingerprint includes: date range, n_effective, n_symbols, universe_sig
        """
        try:
            import hashlib
            fingerprint_parts = []
            
            date_start = metadata.get("date_start") or metadata.get("start_date")
            date_end = metadata.get("date_end") or metadata.get("end_date")
            if date_start and date_end:
                fingerprint_parts.append(f"dates:{date_start}:{date_end}")
            
            n_effective = metadata.get("n_effective")
            if n_effective is not None:
                fingerprint_parts.append(f"n_eff:{n_effective}")
            
            n_symbols = metadata.get("n_symbols") or metadata.get("num_symbols")
            if n_symbols is not None:
                fingerprint_parts.append(f"n_sym:{n_symbols}")
            
            universe_sig = metadata.get("universe_sig")
            if universe_sig:
                fingerprint_parts.append(f"universe:{universe_sig}")
            
            if fingerprint_parts:
                fingerprint_str = "|".join(fingerprint_parts)
                return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
        except Exception as e:
            logger.debug(f"Failed to compute data fingerprint: {e}")
        return None
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash. Delegates to SST module."""
        from TRAINING.common.utils.git_utils import get_git_commit
        return get_git_commit(short=True)
    
    def _get_fallback_metrics_dir(
        self,
        stage: str,
        view: str,
        target: str,
        symbol: Optional[str]
    ) -> Optional[Path]:
        """
        Get fallback directory for metrics when cohort directory doesn't exist.
        
        Falls back to target-level directory: REPRODUCIBILITY/{stage}/{view}/{target}/metrics/
        """
        try:
            repro_dir = self.output_dir / "REPRODUCIBILITY"
            if not repro_dir.exists():
                return None
            
            stage_dir = repro_dir / stage.upper()
            if not stage_dir.exists():
                return None
            
            view_dir = stage_dir / view
            if not view_dir.exists():
                return None
            
            target_dir = view_dir / target
            if not target_dir.exists():
                return None
            
            # Create metrics subdirectory at target level
            metrics_dir = target_dir / "metrics"
            # SST: Use View enum for comparison
            view_enum = View.from_string(view) if isinstance(view, str) else view
            if symbol and view_enum == View.SYMBOL_SPECIFIC:
                metrics_dir = target_dir / f"symbol={symbol}" / "metrics"
            
            metrics_dir.mkdir(parents=True, exist_ok=True)
            return metrics_dir
        except Exception as e:
            logger.debug(f"Failed to get fallback metrics directory: {e}")
            return None
    
    def generate_view_rollup(
        self,
        view_dir: Path,
        stage: Union[str, Stage],  # SST: Accept both string and Stage enum
        view: Union[str, View],  # SST: Accept both string and View enum
        run_id: str
    ) -> None:
        """
        Generate view-level rollup (CROSS_SECTIONAL/metrics_rollup.json or SYMBOL_SPECIFIC/metrics_rollup.json).

        Args:
            view_dir: Path to view directory (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
            stage: Pipeline stage - string or Stage enum
            view: View type - string or View enum
            run_id: Current run identifier
        """
        if not self.enabled:
            return

        view_dir = Path(view_dir)
        if not view_dir.exists():
            return

        # SST: Normalize stage and view enums to strings for JSON serialization
        if isinstance(stage, Stage):
            stage = stage.value
        elif hasattr(stage, 'value'):
            stage = stage.value
        else:
            stage = str(stage)
        
        if isinstance(view, View):
            view = view.value
        elif hasattr(view, 'value'):
            view = view.value
        else:
            view = str(view)

        rollup_data = {
            "run_id": run_id,
            "stage": stage,
            "view": view,
            "timestamp": datetime.now().isoformat(),
            "targets": {},
            "symbols": {},
            "aggregated_metrics": {}
        }
        
        # Collect metrics from all cohort directories in this view
        all_metrics = []
        
        # For CROSS_SECTIONAL: iterate per-target directories
        # SST: Use View enum for comparison
        view_enum = View.from_string(view) if isinstance(view, str) else view
        if view_enum == View.CROSS_SECTIONAL:
            for target_dir in view_dir.iterdir():
                if not target_dir.is_dir() or target_dir.name.startswith("metrics"):
                    continue
                
                target = target_dir.name
                target_metrics = []
                
                # Find most recent cohort for this target
                cohort_dirs = sorted(
                    [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("cohort=")],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if cohort_dirs:
                    latest_cohort = cohort_dirs[0]
                    metrics_file = latest_cohort / "metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                target_data = json.load(f)
                                # PHASE 2: Unified schema - flat structure, no nested "metrics" key
                                target_metrics = {k: v for k, v in target_data.items() 
                                                 if k not in ['run_id', 'timestamp', 'stage', 'reproducibility_mode']}
                                rollup_data["targets"][target] = target_metrics
                                all_metrics.append(target_metrics)
                        except Exception as e:
                            logger.debug(f"Failed to load metrics for {target}: {e}")
        
        # For SYMBOL_SPECIFIC: iterate per-target, then per-symbol
        elif view_enum == View.SYMBOL_SPECIFIC:
            for target_dir in view_dir.iterdir():
                if not target_dir.is_dir() or target_dir.name.startswith("metrics"):
                    continue
                
                target = target_dir.name
                
                for symbol_dir in target_dir.iterdir():
                    if not symbol_dir.is_dir() or not symbol_dir.name.startswith("symbol="):
                        continue
                    
                    symbol_name = symbol_dir.name.replace("symbol=", "")
                    
                    # Find most recent cohort for this symbol+target
                    cohort_dirs = sorted(
                        [d for d in symbol_dir.iterdir() if d.is_dir() and d.name.startswith("cohort=")],
                        key=lambda x: x.stat().st_mtime,
                        reverse=True
                    )
                    
                    if cohort_dirs:
                        latest_cohort = cohort_dirs[0]
                        metrics_file = latest_cohort / "metrics.json"
                        if metrics_file.exists():
                            try:
                                with open(metrics_file, 'r') as f:
                                    symbol_data = json.load(f)
                                    # PHASE 2: Unified schema - flat structure, no nested "metrics" key
                                    symbol_metrics = {k: v for k, v in symbol_data.items() 
                                                     if k not in ['run_id', 'timestamp', 'stage', 'reproducibility_mode']}
                                    
                                    if target not in rollup_data["targets"]:
                                        rollup_data["targets"][target] = {}
                                    rollup_data["targets"][target][symbol_name] = symbol_metrics
                                    
                                    if symbol_name not in rollup_data["symbols"]:
                                        rollup_data["symbols"][symbol_name] = {}
                                    rollup_data["symbols"][symbol_name][target] = symbol_metrics
                                    
                                    all_metrics.append(symbol_metrics)
                            except Exception as e:
                                logger.debug(f"Failed to load metrics for {target}/{symbol_name}: {e}")
        
        # Compute aggregated metrics (mean across all targets/symbols)
        if all_metrics:
            numeric_keys = set()
            for m in all_metrics:
                numeric_keys.update(k for k, v in m.items() if isinstance(v, (int, float)))
            
            for key in numeric_keys:
                values = [m.get(key) for m in all_metrics if isinstance(m.get(key), (int, float))]
                if values:
                    rollup_data["aggregated_metrics"][key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)) if len(values) > 1 else 0.0,
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "count": len(values)
                    }
        
        # Write rollup files (JSON for human-readable, Parquet for queryable)
        rollup_file_json = view_dir / "metrics_rollup.json"
        rollup_file_parquet = view_dir / "metrics_rollup.parquet"
        try:
            # Write JSON rollup
            # SST: Use safe_json_dump to handle enums
            from TRAINING.common.utils.file_utils import safe_json_dump
            with open(rollup_file_json, 'w') as f:
                safe_json_dump(rollup_data, f, indent=2)
            
            # Write Parquet rollup (flattened, queryable format)
            # Convert rollup_data to DataFrame with one row per target/symbol
            parquet_rows = []
            for target, target_data in rollup_data.get("targets", {}).items():
                if isinstance(target_data, dict) and "symbols" not in target_data:
                    # CROSS_SECTIONAL: target-level metrics
                    row = {
                        "run_id": run_id,
                        "stage": stage,
                        "view": view,
                        "target": target,
                        "symbol": None,
                        "timestamp": rollup_data.get("timestamp"),
                        **{k: v for k, v in target_data.items() if isinstance(v, (int, float, str, bool)) or v is None}
                    }
                    parquet_rows.append(row)
                elif isinstance(target_data, dict):
                    # SYMBOL_SPECIFIC: iterate through symbols
                    for symbol_name, symbol_metrics in target_data.get("symbols", {}).items():
                        row = {
                            "run_id": run_id,
                            "stage": stage,
                            "view": view,
                            "target": target,
                            "symbol": symbol_name,
                            "timestamp": rollup_data.get("timestamp"),
                            **{k: v for k, v in symbol_metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
                        }
                        parquet_rows.append(row)
            
            # Also add aggregated metrics as a summary row
            if rollup_data.get("aggregated_metrics"):
                agg_row = {
                    "run_id": run_id,
                    "stage": stage,
                    "view": view,
                    "target": None,  # Aggregated across all targets
                    "symbol": None,  # Aggregated across all symbols
                    "timestamp": rollup_data.get("timestamp"),
                    **{k: v.get("mean") if isinstance(v, dict) and "mean" in v else v 
                       for k, v in rollup_data["aggregated_metrics"].items()}
                }
                parquet_rows.append(agg_row)
            
            if parquet_rows:
                # SST: Sanitize each row before creating DataFrame (handles enums)
                from TRAINING.common.utils.file_utils import sanitize_for_serialization
                sanitized_rows = [sanitize_for_serialization(row) for row in parquet_rows]
                df_rollup = pd.DataFrame(sanitized_rows)
                df_rollup.to_parquet(rollup_file_parquet, index=False, engine='pyarrow', compression='snappy')
                logger.debug(f"✅ Wrote view rollup: {rollup_file_json.name} and {rollup_file_parquet.name}")
        except Exception as e:
            logger.warning(f"Failed to write view rollup to {rollup_file_json}: {e}")
    
    def generate_stage_rollup(
        self,
        stage_dir: Path,
        stage: Union[str, Stage],  # SST: Accept both string and Stage enum
        run_id: str
    ) -> None:
        """
        Generate stage-level container rollup (TARGET_RANKING/metrics_rollup.json).

        This is a container that references view-level rollups, no drift mixing.
        """
        if not self.enabled:
            return

        stage_dir = Path(stage_dir)
        if not stage_dir.exists():
            return

        # SST: Normalize stage enum to string for JSON serialization
        if isinstance(stage, Stage):
            stage = stage.value
        elif hasattr(stage, 'value'):
            stage = stage.value
        else:
            stage = str(stage)

        rollup_data = {
            "run_id": run_id,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "views": {}
        }
        
        # Load view-level rollups
        # SST: Use View enum values
        for view in [View.CROSS_SECTIONAL.value, View.SYMBOL_SPECIFIC.value]:
            view_dir = stage_dir / view
            if view_dir.exists():
                view_rollup_file = view_dir / "metrics_rollup.json"
                if view_rollup_file.exists():
                    try:
                        with open(view_rollup_file, 'r') as f:
                            rollup_data["views"][view] = json.load(f)
                    except Exception as e:
                        logger.debug(f"Failed to load view rollup for {view}: {e}")
        
        # Write stage rollup files (JSON for human-readable, Parquet for queryable)
        rollup_file_json = stage_dir / "metrics_rollup.json"
        rollup_file_parquet = stage_dir / "metrics_rollup.parquet"
        try:
            # Write JSON rollup
            # SST: Use safe_json_dump to handle enums
            from TRAINING.common.utils.file_utils import safe_json_dump
            with open(rollup_file_json, 'w') as f:
                safe_json_dump(rollup_data, f, indent=2)
            
            # Write Parquet rollup (flattened from view-level rollups)
            parquet_rows = []
            for view_name, view_data in rollup_data.get("views", {}).items():
                # Extract targets from view rollup
                for target, target_data in view_data.get("targets", {}).items():
                    if isinstance(target_data, dict) and "symbols" not in target_data:
                        # CROSS_SECTIONAL: target-level
                        row = {
                            "run_id": run_id,
                            "stage": stage,
                            "view": view_name,
                            "target": target,
                            "symbol": None,
                            "timestamp": rollup_data.get("timestamp"),
                            **{k: v for k, v in target_data.items() if isinstance(v, (int, float, str, bool)) or v is None}
                        }
                        parquet_rows.append(row)
                    elif isinstance(target_data, dict):
                        # SYMBOL_SPECIFIC: iterate through symbols
                        for symbol_name, symbol_metrics in target_data.get("symbols", {}).items():
                            row = {
                                "run_id": run_id,
                                "stage": stage,
                                "view": view_name,
                                "target": target,
                                "symbol": symbol_name,
                                "timestamp": rollup_data.get("timestamp"),
                                **{k: v for k, v in symbol_metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
                            }
                            parquet_rows.append(row)
            
            if parquet_rows:
                # SST: Sanitize each row before creating DataFrame (handles enums)
                from TRAINING.common.utils.file_utils import sanitize_for_serialization
                sanitized_rows = [sanitize_for_serialization(row) for row in parquet_rows]
                df_rollup = pd.DataFrame(sanitized_rows)
                df_rollup.to_parquet(rollup_file_parquet, index=False, engine='pyarrow', compression='snappy')
                logger.debug(f"✅ Wrote stage rollup: {rollup_file_json.name} and {rollup_file_parquet.name}")
        except Exception as e:
            logger.warning(f"Failed to write stage rollup to {rollup_file_json}: {e}")


def load_metrics_config() -> Dict[str, Any]:
    """Load metrics configuration from safety.yaml.
    
    Backward compatible: checks safety.metrics.* first, falls back to safety.telemetry.*
    """
    try:
        from CONFIG.config_loader import get_cfg
        # Try new path first (safety.metrics.*)
        enabled = get_cfg("safety.metrics.enabled", default=None, config_name="safety_config")
        if enabled is None:
            # Fallback to old path (safety.telemetry.*) for backward compatibility
            enabled = get_cfg("safety.telemetry.enabled", default=True, config_name="safety_config")
        
        # Baselines
        prev_run = get_cfg("safety.metrics.baselines.previous_run", default=None, config_name="safety_config")
        if prev_run is None:
            prev_run = get_cfg("safety.telemetry.baselines.previous_run", default=True, config_name="safety_config")
        
        rolling_k = get_cfg("safety.metrics.baselines.rolling_window_k", default=None, config_name="safety_config")
        if rolling_k is None:
            rolling_k = get_cfg("safety.telemetry.baselines.rolling_window_k", default=10, config_name="safety_config")
        
        last_good = get_cfg("safety.metrics.baselines.last_good_run", default=None, config_name="safety_config")
        if last_good is None:
            last_good = get_cfg("safety.telemetry.baselines.last_good_run", default=True, config_name="safety_config")
        
        # Drift
        psi_thresh = get_cfg("safety.metrics.drift.psi_threshold", default=None, config_name="safety_config")
        if psi_thresh is None:
            psi_thresh = get_cfg("safety.telemetry.drift.psi_threshold", default=0.2, config_name="safety_config")
        
        ks_thresh = get_cfg("safety.metrics.drift.ks_threshold", default=None, config_name="safety_config")
        if ks_thresh is None:
            ks_thresh = get_cfg("safety.telemetry.drift.ks_threshold", default=0.1, config_name="safety_config")
        
        return {
            "enabled": enabled,
            "baselines": {
                "previous_run": prev_run,
                "rolling_window_k": rolling_k,
                "last_good_run": last_good
            },
            "drift": {
                "psi_threshold": psi_thresh,
                "ks_threshold": ks_thresh
            }
        }
    except Exception as e:
        logger.warning(f"Failed to load metrics config: {e}, using defaults")
        return {
            "enabled": True,
            "baselines": {"previous_run": True, "rolling_window_k": 10, "last_good_run": True},
            "drift": {"psi_threshold": 0.2, "ks_threshold": 0.1}
        }


def aggregate_metrics_facts(
    repro_dir: Path,
    facts_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Aggregate all metrics.json files into a Parquet facts table.
    
    This creates/updates a long-format Parquet table with explicit dimensions:
    - run_id, stage, view, target, symbol, universe_sig, cohort_id
    - All metric values as separate columns
    
    Checks both target-first structure (targets/<target>/metrics/) and legacy REPRODUCIBILITY structure.
    
    Args:
        repro_dir: REPRODUCIBILITY directory root (or run directory)
        facts_file: Optional path to facts table (defaults to repro_dir/metrics_facts.parquet)
    
    Returns:
        DataFrame with all metrics facts
    """
    if facts_file is None:
        facts_file = repro_dir / "metrics_facts.parquet"
    
    repro_dir = Path(repro_dir)
    if not repro_dir.exists():
        logger.warning(f"Directory does not exist: {repro_dir}")
        return pd.DataFrame()
    
    # Find run directory (may be passed REPRODUCIBILITY or run root)
    run_dir = repro_dir
    if repro_dir.name == "REPRODUCIBILITY":
        run_dir = repro_dir.parent
    elif (repro_dir / "REPRODUCIBILITY").exists():
        repro_dir = repro_dir / "REPRODUCIBILITY"
    
    # Collect all metrics.json files (unified canonical schema)
    rows = []
    
    # First, check target-first structure (targets/<target>/metrics/)
    targets_dir = run_dir / "targets"
    if targets_dir.exists():
        for target_dir in targets_dir.iterdir():
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            metrics_dir = target_dir / "metrics"
            if metrics_dir.exists():
                # Check for view-organized metrics
                for item in metrics_dir.iterdir():
                    if item.is_dir() and item.name.startswith("view="):
                        view = item.name.replace("view=", "")
                        metrics_file = item / "metrics.json"
                        if metrics_file.exists():
                            row = _extract_metrics_row(
                                metrics_file, "TRAINING", view, target, None, 
                                "default", None
                            )
                            if row:
                                rows.append(row)
                    elif item.name == "metrics.json":
                        # Direct metrics file
                        metrics_file = item
                        row = _extract_metrics_row(
                            metrics_file, "TRAINING", "CROSS_SECTIONAL", target, None,
                            "default", None
                        )
                        if row:
                            rows.append(row)
    
    # Also walk legacy REPRODUCIBILITY structure
    if repro_dir.exists():
        for stage_dir in repro_dir.iterdir():
            if not stage_dir.is_dir() or stage_dir.name in ["artifact_index.parquet", "metrics_facts.parquet", "stats.json"]:
                continue
            
            stage = stage_dir.name
        
        # Check if this stage has view subdirectories (TARGET_RANKING, FEATURE_SELECTION)
        view_dirs = []
        for item in stage_dir.iterdir():
            if item.is_dir() and item.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
                view_dirs.append(item)
            elif item.is_dir() and item.name.startswith("cohort="):
                # Legacy structure: stage directly contains cohorts (no view separation)
                # Treat as CROSS_SECTIONAL
                view_dirs = [stage_dir]  # Process stage_dir directly
                break
        
        if not view_dirs:
            # No view subdirectories, check for direct cohort directories
            for item in stage_dir.iterdir():
                if item.is_dir() and item.name.startswith("cohort="):
                    view_dirs = [stage_dir]
                    break
        
        # Process each view
        for view_dir in view_dirs:
            view = view_dir.name if view_dir.name in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"] else "CROSS_SECTIONAL"
            
            # Walk through target/symbol structure
            for target_path in view_dir.iterdir():
                if not target_path.is_dir():
                    continue
                
                target = target_path.name
                symbol = None
                universe_sig = None
                
                # Check if this is a symbol directory (SYMBOL_SPECIFIC view)
                # SST: Use View enum for comparison
                view_enum = View.from_string(view) if isinstance(view, str) else view
                if view_enum == View.SYMBOL_SPECIFIC:
                    for symbol_path in target_path.iterdir():
                        if not symbol_path.is_dir():
                            continue
                        
                        if symbol_path.name.startswith("symbol="):
                            symbol = symbol_path.name.replace("symbol=", "")
                            
                            # Look for cohorts in symbol directory
                            for cohort_path in symbol_path.iterdir():
                                if not cohort_path.is_dir() or not cohort_path.name.startswith("cohort="):
                                    continue
                                
                                cohort_id = cohort_path.name.replace("cohort=", "")
                                metrics_file = cohort_path / "metrics.json"
                                
                                if metrics_file.exists():
                                    row = _extract_metrics_row(
                                        metrics_file, stage, view, target, symbol, 
                                        cohort_id, universe_sig
                                    )
                                    if row:
                                        rows.append(row)
                else:
                    # CROSS_SECTIONAL: cohorts directly under target
                    for cohort_path in target_path.iterdir():
                        if not cohort_path.is_dir() or not cohort_path.name.startswith("cohort="):
                            continue
                        
                        cohort_id = cohort_path.name.replace("cohort=", "")
                        metrics_file = cohort_path / "metrics.json"
                        
                        if metrics_file.exists():
                            # Try to extract universe_sig from metadata
                            metadata_file = cohort_path / "metadata.json"
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    universe_sig = metadata.get('universe_sig')
                                except Exception:
                                    pass
                            
                            row = _extract_metrics_row(
                                metrics_file, stage, view, target, symbol,
                                cohort_id, universe_sig
                            )
                            if row:
                                rows.append(row)
    
    if not rows:
        logger.debug("No metrics found to aggregate")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Load existing facts table and append
    if facts_file.exists():
        try:
            existing_df = pd.read_parquet(facts_file)
            # Deduplicate by (run_id, stage, view, target, symbol, cohort_id)
            # Keep latest timestamp
            df = pd.concat([existing_df, df], ignore_index=True)
            df = df.sort_values('timestamp', ascending=False).drop_duplicates(
                subset=['run_id', 'stage', 'view', 'target', 'symbol', 'cohort_id'],
                keep='first'
            )
        except Exception as e:
            logger.warning(f"Failed to load existing facts table, creating new: {e}")
    
    # Save facts table
    try:
        facts_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(facts_file, index=False, engine='pyarrow', compression='snappy')
        logger.info(f"✅ Aggregated {len(df)} metrics facts to {facts_file}")
    except Exception as e:
        logger.warning(f"Failed to save metrics facts table: {e}")
    
    return df


def _extract_metrics_row(
    metrics_file: Path,
    stage: str,
    view: str,
    target: str,
    symbol: Optional[str],
    cohort_id: str,
    universe_sig: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Extract a single row from metrics.json (unified canonical schema) for facts table.
    
    PHASE 2: Updated to handle flat schema (no nested "metrics" key).
    
    Returns:
        Dict with dimensions and metrics, or None if extraction fails
    """
    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        
        # Extract dimensions
        row = {
            'run_id': data.get('run_id', ''),
            'timestamp': data.get('timestamp', ''),
            'stage': stage,
            'view': view,
            'target': target,
            'symbol': symbol if symbol else None,
            'universe_sig': universe_sig if universe_sig else None,
            'cohort_id': cohort_id
        }
        
        # PHASE 2: Extract metrics from flat schema (all keys except dimension keys)
        dimension_keys = {'run_id', 'timestamp', 'stage', 'reproducibility_mode'}
        for key, value in data.items():
            if key in dimension_keys:
                continue  # Skip dimension keys
            
            # Convert to appropriate type
            if isinstance(value, (int, float, np.integer, np.floating)):
                row[key] = float(value)
            elif isinstance(value, (bool, np.bool_)):
                row[key] = bool(value)
            elif isinstance(value, str):
                row[key] = str(value)
            elif value is None:
                row[key] = None
            else:
                # Complex types: store as JSON string
                row[key] = json.dumps(value) if value else None
        
        return row
    except Exception as e:
        logger.debug(f"Failed to extract metrics row from {metrics_file}: {e}")
        return None
