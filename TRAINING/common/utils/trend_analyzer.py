# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Trend Analyzer for Cohort Reproducibility

Analyzes trends over time within identical cohort series, detecting:
- Performance drift (improving/declining metrics)
- Pipeline regressions (same inputs, different outputs)
- Feature churn impact (PROGRESS series with breakpoints)

Supports two views:
- STRICT_SERIES: Only trend when all comparability keys match (reproducibility)
- PROGRESS_SERIES: Allow controlled changes, mark breakpoints (iteration progress)

By default, uses comparison_group_key for grouping (lenient, like metrics diffs).
This allows runs with the same comparison group to be grouped together even if
some hashes differ. Set use_comparison_group=False to use strict SeriesKey matching.

Usage:
    from TRAINING.common.utils.trend_analyzer import TrendAnalyzer
    
    analyzer = TrendAnalyzer(reproducibility_dir=Path("RESULTS/.../REPRODUCIBILITY"))
    trends = analyzer.analyze_all_series(view="STRICT", half_life_days=7.0)
    analyzer.write_trend_report(trends, output_path=Path("TREND_REPORT.json"))
"""

import json
import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

# Import SST for comparison group key construction
from TRAINING.common.utils.fingerprinting import construct_comparison_group_key_from_dict

# SST: Import View and Stage enums for consistent view/stage handling
from TRAINING.orchestration.utils.scope_resolution import View, Stage

logger = logging.getLogger(__name__)


class SeriesView(str, Enum):
    """Series grouping view."""
    STRICT = "STRICT"  # All comparability keys must match
    PROGRESS = "PROGRESS"  # Allow feature hash changes, mark breakpoints


@dataclass(frozen=True)
class SeriesKey:
    """
    Immutable key for grouping runs into comparable series.
    
    For STRICT view: all fields must match exactly.
    For PROGRESS view: cohort_id, stage, target must match; feature_registry_hash can change (breakpoint).
    """
    cohort_id: str
    stage: str
    target: str
    data_fingerprint: str
    feature_registry_hash: Optional[str] = None
    fold_boundaries_hash: Optional[str] = None
    label_definition_hash: Optional[str] = None
    
    def to_strict_key(self) -> str:
        """Generate strict grouping key (all fields must match)."""
        parts = [
            self.cohort_id,
            self.stage,
            self.target,
            self.data_fingerprint,
            self.feature_registry_hash or "none",
            self.fold_boundaries_hash or "none",
            self.label_definition_hash or "none"
        ]
        return "|".join(parts)
    
    def to_progress_key(self) -> str:
        """Generate progress grouping key (allows feature hash changes)."""
        parts = [
            self.cohort_id,
            self.stage,
            self.target,
            self.data_fingerprint,
            self.fold_boundaries_hash or "none",
            self.label_definition_hash or "none"
        ]
        return "|".join(parts)


@dataclass
class TrendResult:
    """Result of trend analysis for a single series."""
    series_key: SeriesKey
    view: SeriesView
    metric_name: str
    n_runs: int
    status: str  # "ok", "insufficient_runs", "no_variance"
    
    # Trend parameters
    half_life_days: float
    intercept: Optional[float] = None
    slope_per_day: Optional[float] = None
    current_estimate: Optional[float] = None
    residual_std: Optional[float] = None
    
    # EWMA (alternative to regression)
    ewma_value: Optional[float] = None
    ewma_alpha: Optional[float] = None
    
    # Alerts
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Breakpoints (for PROGRESS view)
    breakpoints: List[Dict[str, Any]] = field(default_factory=list)


class TrendAnalyzer:
    """
    Analyzes trends across runs within comparable series.
    
    Groups runs by strict comparability keys, applies exponential decay weighting,
    and computes weighted regression trends.
    """
    
    def __init__(
        self,
        reproducibility_dir: Path,
        half_life_days: float = 7.0,
        min_runs_for_trend: int = 2,  # Minimum 2 runs for trend (slope requires 2 points)
        suspicious_slope_threshold: float = -0.01  # Alert if slope < -0.01 per day
    ):
        """
        Initialize trend analyzer.
        
        Args:
            reproducibility_dir: Path to REPRODUCIBILITY directory
            half_life_days: Exponential decay half-life in days (default: 7)
            min_runs_for_trend: Minimum runs required for trend fitting (default: 2, needs at least 2 points for slope)
            suspicious_slope_threshold: Alert threshold for negative slope (default: -0.01/day)
        """
        self.reproducibility_dir = Path(reproducibility_dir)
        self.half_life_days = half_life_days
        self.min_runs_for_trend = min_runs_for_trend
        self.suspicious_slope_threshold = suspicious_slope_threshold
    
    def load_artifact_index(self) -> pd.DataFrame:
        """
        Load or build unified artifact index from REPRODUCIBILITY structure.
        
        Returns:
            DataFrame with one row per run, normalized columns
        """
        index_path = self.reproducibility_dir / "artifact_index.parquet"
        
        # Try to load existing index
        if index_path.exists():
            try:
                df = pd.read_parquet(index_path)
                logger.debug(f"Loaded existing artifact index: {len(df)} runs")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}, rebuilding...")
                # Remove corrupted file
                try:
                    index_path.unlink()
                    logger.info("Removed corrupted artifact_index.parquet, will rebuild")
                except Exception:
                    pass
        
        # Build index from REPRODUCIBILITY structure and target-first structure
        logger.info("Building artifact index from REPRODUCIBILITY and target-first structure...")
        rows = []
        
        # Find run directory (reproducibility_dir might be REPRODUCIBILITY or run root)
        run_dir = self.reproducibility_dir
        if self.reproducibility_dir.name == "REPRODUCIBILITY":
            run_dir = self.reproducibility_dir.parent
        elif (self.reproducibility_dir / "REPRODUCIBILITY").exists():
            run_dir = self.reproducibility_dir
        
        # Detect if we're in a comparison group directory structure (RESULTS/runs/cg-*/)
        # If so, search across all runs in the same comparison group
        comparison_group_dir = None
        runs_to_process = [run_dir]  # Default: just process the single run
        
        # Check if we're in RESULTS/runs/cg-*/run_name/ structure
        temp_dir = run_dir
        for _ in range(5):  # Limit depth
            if temp_dir.parent.name == "runs" and temp_dir.parent.parent.name == "RESULTS":
                # We're in RESULTS/runs/cg-*/run_name/
                comparison_group_dir = temp_dir.parent
                # Get all runs in this comparison group
                runs_to_process = [
                    d for d in comparison_group_dir.iterdir()
                    if d.is_dir() and ((d / "targets").exists() or (d / "globals").exists() or (d / "REPRODUCIBILITY").exists())
                ]
                logger.info(f"Found comparison group directory: {comparison_group_dir.name}, processing {len(runs_to_process)} runs")
                break
            if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                break
            temp_dir = temp_dir.parent
        
        # Process all runs in the comparison group (or just the single run if not in comparison group structure)
        for current_run_dir in runs_to_process:
            # First, check target-first structure (targets/<target>/reproducibility/<view>/cohort=<cohort_id>/)
            targets_dir = current_run_dir / "targets"
            if targets_dir.exists():
                for target_dir in targets_dir.iterdir():
                    if not target_dir.is_dir():
                        continue
                    target = target_dir.name
                    repro_dir = target_dir / "reproducibility"
                    if repro_dir.exists():
                        # Walk through view subdirectories (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)
                        for view_dir in repro_dir.iterdir():
                            if not view_dir.is_dir():
                                continue
                            view = view_dir.name
                            
                            # For SYMBOL_SPECIFIC, need to check symbol subdirectories
                            # SST: Use View enum for comparison
                            view_enum = View.from_string(view) if isinstance(view, str) else view
                            if view_enum == View.SYMBOL_SPECIFIC:
                                for symbol_dir in view_dir.iterdir():
                                    if not symbol_dir.is_dir() or not symbol_dir.name.startswith("symbol="):
                                        continue
                                    symbol = symbol_dir.name.replace("symbol=", "")
                                    
                                    # Walk through cohort directories under symbol
                                    for cohort_dir in symbol_dir.iterdir():
                                        if not cohort_dir.is_dir() or not cohort_dir.name.startswith("cohort="):
                                            continue
                                        
                                        metadata_file = cohort_dir / "metadata.json"
                                        # Try to read metrics from metrics/ folder first, fallback to cohort_dir
                                        metrics_file = None
                                        metrics_data = {}
                                        try:
                                            from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                                            metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir, base_output_dir=current_run_dir)
                                            if metrics_dir:
                                                metrics_file = metrics_dir / "metrics.json"
                                                if not metrics_file.exists():
                                                    # Try parquet
                                                    metrics_parquet = metrics_dir / "metrics.parquet"
                                                    if metrics_parquet.exists():
                                                        import pandas as pd
                                                        df = pd.read_parquet(metrics_parquet)
                                                        if len(df) > 0:
                                                            metrics_data = df.iloc[0].to_dict()
                                                            metrics_file = metrics_parquet  # For path reference
                                        except Exception as e:
                                            logger.debug(f"Failed to map cohort_dir to metrics path: {e}")
                                        
                                        # Fallback to legacy location in cohort_dir
                                        if not metrics_data and not metrics_file:
                                            metrics_file = cohort_dir / "metrics.json"
                                        
                                        if metadata_file.exists() or (metrics_file and metrics_file.exists()):
                                            try:
                                                metadata = {}
                                                if metadata_file.exists():
                                                    with open(metadata_file, 'r') as f:
                                                        metadata = json.load(f)
                                                
                                                if not metrics_data and metrics_file and metrics_file.exists():
                                                    if metrics_file.suffix == '.parquet':
                                                        import pandas as pd
                                                        df = pd.read_parquet(metrics_file)
                                                        if len(df) > 0:
                                                            metrics_data = df.iloc[0].to_dict()
                                                    else:
                                                        with open(metrics_file, 'r') as f:
                                                            metrics_data = json.load(f)
                                                
                                                # Extract identifiers
                                                run_id = metadata.get('run_id') or metrics_data.get('run_id') or current_run_dir.name
                                                stage = metadata.get('stage', 'UNKNOWN')
                                                cohort_id = cohort_dir.name.replace('cohort=', '')
                                                
                                                row = {
                                                    'run_id': run_id,
                                                    'stage': stage,
                                                    'target': target,
                                                    'view': view,
                                                    'symbol': symbol,
                                                    'cohort_id': cohort_id,
                                                    'metadata_path': str(metadata_file.relative_to(current_run_dir)) if metadata_file.exists() else None,
                                                    'metrics_path': str(metrics_file.relative_to(current_run_dir)) if metrics_file.exists() else None,
                                                    **metadata,
                                                    **{k: v for k, v in metrics_data.items() if k not in metadata}
                                                }
                                                rows.append(row)
                                            except Exception as e:
                                                logger.debug(f"Failed to process target-first metrics for {target}/{view}/{symbol}/{cohort_dir.name}: {e}")
                            else:
                                # CROSS_SECTIONAL: cohort directories directly under view
                                for cohort_dir in view_dir.iterdir():
                                    if not cohort_dir.is_dir() or not cohort_dir.name.startswith("cohort="):
                                        continue
                                    
                                    metadata_file = cohort_dir / "metadata.json"
                                    # Try to read metrics from metrics/ folder first, fallback to cohort_dir
                                    metrics_file = None
                                    metrics_data = {}
                                    try:
                                        from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                                        metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir, base_output_dir=current_run_dir)
                                        if metrics_dir:
                                            metrics_file = metrics_dir / "metrics.json"
                                            if not metrics_file.exists():
                                                # Try parquet
                                                metrics_parquet = metrics_dir / "metrics.parquet"
                                                if metrics_parquet.exists():
                                                    import pandas as pd
                                                    df = pd.read_parquet(metrics_parquet)
                                                    if len(df) > 0:
                                                        metrics_data = df.iloc[0].to_dict()
                                                        metrics_file = metrics_parquet  # For path reference
                                    except Exception as e:
                                        logger.debug(f"Failed to map cohort_dir to metrics path: {e}")
                                    
                                    # Fallback to legacy location in cohort_dir
                                    if not metrics_data and not metrics_file:
                                        metrics_file = cohort_dir / "metrics.json"
                                    
                                    if metadata_file.exists() or (metrics_file and metrics_file.exists()):
                                        try:
                                            metadata = {}
                                            if metadata_file.exists():
                                                with open(metadata_file, 'r') as f:
                                                    metadata = json.load(f)
                                            
                                            if not metrics_data and metrics_file and metrics_file.exists():
                                                if metrics_file.suffix == '.parquet':
                                                    import pandas as pd
                                                    df = pd.read_parquet(metrics_file)
                                                    if len(df) > 0:
                                                        metrics_data = df.iloc[0].to_dict()
                                                else:
                                                    with open(metrics_file, 'r') as f:
                                                        metrics_data = json.load(f)
                                            
                                            # Extract identifiers
                                            run_id = metadata.get('run_id') or metrics_data.get('run_id') or current_run_dir.name
                                            stage = metadata.get('stage', 'UNKNOWN')
                                            cohort_id = cohort_dir.name.replace('cohort=', '')
                                            
                                            row = {
                                                'run_id': run_id,
                                                'stage': stage,
                                                'target': target,
                                                'view': view,
                                                'cohort_id': cohort_id,
                                                'metadata_path': str(metadata_file.relative_to(current_run_dir)) if metadata_file.exists() else None,
                                                'metrics_path': str(metrics_file.relative_to(current_run_dir)) if metrics_file.exists() else None,
                                                **metadata,
                                                **{k: v for k, v in metrics_data.items() if k not in metadata}
                                            }
                                            rows.append(row)
                                        except Exception as e:
                                            logger.debug(f"Failed to process target-first metrics for {target}/{view}/{cohort_dir.name}: {e}")
        
        # Also walk legacy REPRODUCIBILITY directory (only for the original reproducibility_dir, not all runs)
        # This handles legacy structure that might not be in target-first format
        if self.reproducibility_dir.exists() and not comparison_group_dir:
            for stage_dir in self.reproducibility_dir.iterdir():
                if not stage_dir.is_dir() or stage_dir.name.startswith('.'):
                    continue
                
                stage = stage_dir.name
                
                # Handle nested structure: STAGE/MODE/target/cohort=.../
                for item_dir in self._walk_stage_directory(stage_dir):
                    for cohort_dir in item_dir.iterdir():
                        if not cohort_dir.is_dir() or not cohort_dir.name.startswith('cohort='):
                            continue
                        
                        # Load metadata and metrics
                        metadata_file = cohort_dir / "metadata.json"
                        # Try to read metrics from metrics/ folder first, fallback to cohort_dir
                        metrics_file = None
                        metrics = {}
                        try:
                            from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                            metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir, base_output_dir=self.reproducibility_dir.parent if self.reproducibility_dir.name == "REPRODUCIBILITY" else self.reproducibility_dir)
                            if metrics_dir:
                                metrics_file = metrics_dir / "metrics.json"
                                if not metrics_file.exists():
                                    # Try parquet
                                    metrics_parquet = metrics_dir / "metrics.parquet"
                                    if metrics_parquet.exists():
                                        import pandas as pd
                                        df = pd.read_parquet(metrics_parquet)
                                        if len(df) > 0:
                                            metrics = df.iloc[0].to_dict()
                                            metrics_file = metrics_parquet  # For path reference
                        except Exception as e:
                            logger.debug(f"Failed to map cohort_dir to metrics path: {e}")
                        
                        # Fallback to legacy location in cohort_dir
                        if not metrics and not metrics_file:
                            metrics_file = cohort_dir / "metrics.json"
                        
                        if not metadata_file.exists():
                            continue
                        
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            if not metrics and metrics_file and metrics_file.exists():
                                if metrics_file.suffix == '.parquet':
                                    import pandas as pd
                                    df = pd.read_parquet(metrics_file)
                                    if len(df) > 0:
                                        metrics = df.iloc[0].to_dict()
                                else:
                                    with open(metrics_file, 'r') as f:
                                        metrics = json.load(f)
                            
                            # Extract normalized row
                            row = self._extract_index_row(stage, item_dir, cohort_dir, metadata, metrics)
                            if row:
                                rows.append(row)
                        except Exception as e:
                            logger.debug(f"Failed to load {metadata_file}: {e}")
                            continue
        
        if not rows:
            logger.warning("No runs found in target-first structure (targets/, globals/) or legacy REPRODUCIBILITY directory")
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        
        # Save index for future use
        try:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(index_path)
            logger.info(f"Saved artifact index: {len(df)} runs")
        except Exception as e:
            logger.warning(f"Failed to save artifact index: {e}")
        
        return df
    
    def _walk_stage_directory(self, stage_dir: Path) -> List[Path]:
        """Walk stage directory to find all item directories (targets/features/models)."""
        items = []
        
        # Check if there's a MODE subdirectory (CROSS_SECTIONAL/INDIVIDUAL)
        for mode_dir in stage_dir.iterdir():
            if not mode_dir.is_dir():
                continue
            
            # Check if this is a MODE directory or direct item
            if mode_dir.name in ['CROSS_SECTIONAL', 'INDIVIDUAL']:
                # Walk MODE/target/... structure
                for item_dir in mode_dir.iterdir():
                    if item_dir.is_dir() and not item_dir.name.startswith('cohort='):
                        items.append(item_dir)
            else:
                # Direct item directory
                if not mode_dir.name.startswith('cohort='):
                    items.append(mode_dir)
        
        return items
    
    def _construct_comparison_group_key(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Construct comparison_group_key from metadata.
        
        Uses SST function from fingerprinting.py to avoid duplication.

        Args:
            metadata: Metadata dictionary from metadata.json

        Returns:
            Comparison group key string, or None if not available
        """
        # First, try to get it directly from diff_telemetry
        diff_telemetry = metadata.get('diff_telemetry', {})
        comparison_group_key = diff_telemetry.get('comparison_group_key')
        if comparison_group_key:
            return comparison_group_key

        # If not available, construct from comparison_group dict using SST
        comparison_group = diff_telemetry.get('comparison_group', {})
        if not comparison_group:
            return None

        # Use SST function (debug mode for backward compatible short-hash format)
        return construct_comparison_group_key_from_dict(comparison_group, mode="debug")
    
    def _extract_index_row(
        self,
        stage: str,
        item_dir: Path,
        cohort_dir: Path,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract normalized index row from metadata and metrics."""
        try:
            # Extract item name from path
            target = item_dir.name
            
            # Parse cohort_id from directory name
            cohort_id = metadata.get('cohort_id', '')
            if not cohort_id and cohort_dir.name.startswith('cohort='):
                cohort_id = cohort_dir.name.replace('cohort=', '')
            
            # Extract comparability keys
            cv_details = metadata.get('cv_details', {})
            
            # Extract comparison_group_key (for lenient grouping like metrics diffs)
            comparison_group_key = self._construct_comparison_group_key(metadata)
            
            row = {
                # Identity
                'run_id': metadata.get('run_id', ''),
                'created_at': pd.Timestamp(metadata.get('created_at', datetime.now().isoformat())),
                'git_commit': metadata.get('git_commit', ''),
                'schema_version': metadata.get('schema_version', 1),
                
                # Stage + item
                'stage': stage,
                'target': target,
                'target': metadata.get('target', target),
                'symbol': metadata.get('symbol'),
                'model_family': metadata.get('model_family'),
                
                # Strict comparability keys (for backward compatibility)
                'cohort_id': cohort_id,
                'data_fingerprint': metadata.get('data_fingerprint'),
                'feature_registry_hash': metadata.get('feature_registry_hash'),
                'fold_boundaries_hash': cv_details.get('fold_boundaries_hash'),
                'label_definition_hash': cv_details.get('label_definition_hash') or metadata.get('label_definition_hash'),
                
                # Comparison group key (for lenient grouping like metrics diffs)
                'comparison_group_key': comparison_group_key,
                
                # CV details
                'horizon_minutes': cv_details.get('horizon_minutes') or metadata.get('horizon_minutes'),
                'purge_minutes': cv_details.get('purge_minutes') or metadata.get('purge_minutes'),
                'embargo_minutes': cv_details.get('embargo_minutes') or metadata.get('embargo_minutes'),
                'folds': cv_details.get('folds'),
                
                # Metrics (stage-specific, store as JSON blob)
                'metrics': json.dumps(metrics)
            }
            
            # Paths - try to get metrics path from metrics/ folder (computed outside dict)
            metadata_file_path = cohort_dir / "metadata.json"
            metrics_file_path = None
            try:
                from TRAINING.orchestration.utils.target_first_paths import get_metrics_path_from_cohort_dir
                metrics_dir = get_metrics_path_from_cohort_dir(cohort_dir)
                if metrics_dir:
                    metrics_file_path = metrics_dir / "metrics.json"
                    if not metrics_file_path.exists():
                        metrics_file_path = metrics_dir / "metrics.parquet"
            except Exception:
                pass
            # Fallback to legacy location
            if not metrics_file_path or not metrics_file_path.exists():
                metrics_file_path = cohort_dir / "metrics.json"
            
            row['metadata_path'] = str(metadata_file_path) if metadata_file_path.exists() else None
            row['metrics_path'] = str(metrics_file_path) if metrics_file_path and metrics_file_path.exists() else None
            
            # Extract stage-specific metric fields
            # Use SST accessors to handle both old flat and new grouped structures
            from TRAINING.orchestration.utils.reproducibility.utils import extract_auc, extract_n_effective
            
            # SST: Use Stage enum for comparison
            stage_enum = Stage.from_string(stage) if isinstance(stage, str) else stage
            if stage_enum == Stage.TARGET_RANKING:
                row['auc_mean'] = extract_auc(metrics)  # Handles both old and new structures
                # Try new structure first, then fallback to old
                row['auc_std'] = (metrics.get('primary_metric', {}).get('std') or 
                                 metrics.get('primary_metric', {}).get('skill_se') or 
                                 metrics.get('std_score'))
                row['composite_score'] = (metrics.get('score', {}).get('composite') or 
                                         metrics.get('composite_score'))
                row['importance_mean'] = (metrics.get('score', {}).get('components', {}).get('mean_importance') or 
                                         metrics.get('mean_importance'))
                row['n_effective'] = extract_n_effective(metrics) or metadata.get('n_effective')
            elif stage_enum == Stage.FEATURE_SELECTION:
                # Feature selection metrics (to be defined based on actual structure)
                row['n_selected'] = (metrics.get('features', {}).get('selected') or 
                                   metrics.get('n_selected') or 
                                   metrics.get('n_features'))
            elif stage_enum == Stage.TRAINING:
                row['primary_metric'] = extract_auc(metrics)  # Handles both old and new structures
                row['train_time_sec'] = metrics.get('train_time_sec')
            
            return row
        except Exception as e:
            logger.debug(f"Failed to extract index row: {e}")
            return None
    
    def group_into_series(
        self,
        df: pd.DataFrame,
        view: SeriesView = SeriesView.STRICT,
        use_comparison_group: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group runs into comparable series based on view.
        
        Args:
            df: Artifact index DataFrame
            view: STRICT or PROGRESS
            use_comparison_group: If True, use comparison_group_key for grouping (lenient, like metrics diffs).
                                 If False, use strict SeriesKey matching (backward compatible).
        
        Returns:
            Dict mapping series_key -> list of run rows
        """
        series = {}
        
        # Check if comparison_group_key is available in the DataFrame
        has_comparison_group = use_comparison_group and 'comparison_group_key' in df.columns
        
        for _, row in df.iterrows():
            # Build SeriesKey (always needed for TrendResult.series_key)
            key = SeriesKey(
                cohort_id=str(row.get('cohort_id', '')),
                stage=str(row.get('stage', '')),
                target=str(row.get('target', row.get('target', ''))),
                data_fingerprint=str(row.get('data_fingerprint', '')) if pd.notna(row.get('data_fingerprint')) else '',
                feature_registry_hash=str(row.get('feature_registry_hash', '')) if pd.notna(row.get('feature_registry_hash')) else None,
                fold_boundaries_hash=str(row.get('fold_boundaries_hash', '')) if pd.notna(row.get('fold_boundaries_hash')) else None,
                label_definition_hash=str(row.get('label_definition_hash', '')) if pd.notna(row.get('label_definition_hash')) else None
            )
            
            # Choose grouping key: use comparison_group_key if available and enabled, else use SeriesKey
            if has_comparison_group:
                comparison_group_key = row.get('comparison_group_key')
                if pd.notna(comparison_group_key) and comparison_group_key:
                    # Use comparison_group_key for grouping (lenient, like metrics diffs)
                    series_key = str(comparison_group_key)
                else:
                    # Fall back to SeriesKey if comparison_group_key is missing
                    if view == SeriesView.STRICT:
                        series_key = key.to_strict_key()
                    else:
                        series_key = key.to_progress_key()
            else:
                # Use strict SeriesKey matching (backward compatible)
                if view == SeriesView.STRICT:
                    series_key = key.to_strict_key()
                else:
                    series_key = key.to_progress_key()
            
            # Convert row to dict
            run_dict = row.to_dict()
            run_dict['_series_key'] = key  # Store key object for later use
            
            if series_key not in series:
                series[series_key] = []
            series[series_key].append(run_dict)
        
        return series
    
    def exp_decay_weights(self, timestamps: List[datetime], half_life_days: float) -> np.ndarray:
        """
        Compute exponential decay weights for timestamps.
        
        Args:
            timestamps: List of datetime objects
            half_life_days: Half-life in days
        
        Returns:
            Array of weights (sum to 1.0)
        """
        now = datetime.now(timezone.utc)
        ages = []
        for ts in timestamps:
            # Ensure timezone-aware
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Compute timedelta and convert to days
            delta = now - ts
            age_days = delta.total_seconds() / 86400.0
            ages.append(age_days)
        
        ages = np.array(ages, dtype=float)
        w = 0.5 ** (ages / half_life_days)
        # Normalize to sum to 1.0
        w = w / (w.sum() + 1e-12)
        return w
    
    def weighted_linear_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        Weighted least squares for y = a + b*x.
        
        Args:
            x: Independent variable (days since first run)
            y: Dependent variable (metric values)
            w: Weights (normalized)
        
        Returns:
            (intercept, slope, y_hat, residual_std)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        w = np.asarray(w, dtype=float)
        
        # Normalize weights
        w = w / (w.sum() + 1e-12)
        
        # Weighted least squares: beta = (X' W X)^-1 X' W y
        X = np.column_stack([np.ones_like(x), x])  # [1, x]
        W = np.diag(w)
        
        try:
            beta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ y)
            a, b = float(beta[0]), float(beta[1])
            
            y_hat = X @ beta
            resid = y - y_hat
            # Weighted residual std
            resid_std = math.sqrt(float((w * resid * resid).sum()) + 1e-12)
            
            return a, b, y_hat, resid_std
        except np.linalg.LinAlgError:
            # Fallback: unweighted fit
            beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
            a, b = float(beta[0]), float(beta[1])
            y_hat = X @ beta
            resid = y - y_hat
            resid_std = math.sqrt(float((resid * resid).mean()) + 1e-12)
            return a, b, y_hat, resid_std
    
    def compute_ewma(
        self,
        values: np.ndarray,
        timestamps: List[datetime],
        half_life_days: float
    ) -> Tuple[float, float]:
        """
        Compute exponentially weighted moving average.
        
        Args:
            values: Metric values (sorted by time)
            timestamps: Corresponding timestamps
            half_life_days: Half-life for decay
        
        Returns:
            (ewma_value, alpha) where alpha is the smoothing factor
        """
        if len(values) == 0:
            return 0.0, 0.0
        
        # Compute alpha from half-life
        # For EWMA: alpha = 1 - exp(-ln(2) / half_life_days)
        alpha = 1.0 - math.exp(-math.log(2) / half_life_days)
        
        # Apply EWMA (most recent gets highest weight)
        ewma = values[0]
        for i in range(1, len(values)):
            ewma = alpha * values[i] + (1 - alpha) * ewma
        
        return float(ewma), alpha
    
    def analyze_series_trend(
        self,
        runs: List[Dict[str, Any]],
        metric_field: str,
        view: SeriesView,
        series_key: SeriesKey
    ) -> TrendResult:
        """
        Analyze trend for a single series and metric.
        
        Args:
            runs: List of run dictionaries (from artifact index)
            metric_field: Field name in metrics to analyze (e.g., 'auc_mean')
            view: STRICT or PROGRESS
            series_key: SeriesKey for this series
        
        Returns:
            TrendResult
        """
        # Filter runs with valid metric values
        valid_runs = [
            r for r in runs
            if metric_field in r and pd.notna(r[metric_field]) and r[metric_field] is not None
        ]
        
        if len(valid_runs) < self.min_runs_for_trend:
            # Use "first_run" status for single runs to distinguish from multiple insufficient runs
            status = "first_run" if len(valid_runs) == 1 else "insufficient_runs"
            return TrendResult(
                series_key=series_key,
                view=view,
                metric_name=metric_field,
                n_runs=len(valid_runs),
                status=status,
                half_life_days=self.half_life_days
            )
        
        # Sort by time
        valid_runs.sort(key=lambda r: r['created_at'])
        
        # Extract values and timestamps
        timestamps = [r['created_at'] for r in valid_runs]
        values = np.array([r[metric_field] for r in valid_runs], dtype=float)
        
        # Check for variance
        if np.std(values) < 1e-10:
            return TrendResult(
                series_key=series_key,
                view=view,
                metric_name=metric_field,
                n_runs=len(valid_runs),
                status="no_variance",
                half_life_days=self.half_life_days
            )
        
        # Compute exponential decay weights
        weights = self.exp_decay_weights(timestamps, self.half_life_days)
        
        # Compute EWMA
        ewma_value, ewma_alpha = self.compute_ewma(values, timestamps, self.half_life_days)
        
        # Compute weighted regression
        t0 = timestamps[0]
        x = np.array([(ts - t0).total_seconds() / 86400.0 for ts in timestamps], dtype=float)
        
        intercept, slope, y_hat, resid_std = self.weighted_linear_fit(x, values, weights)
        
        # Current estimate (at now)
        now = datetime.now(timezone.utc)
        x_now = (now - t0).total_seconds() / 86400.0
        current_estimate = intercept + slope * x_now
        
        # Detect breakpoints (for PROGRESS view)
        breakpoints = []
        if view == SeriesView.PROGRESS:
            # Detect feature hash changes
            feature_hashes = [r.get('feature_registry_hash') for r in valid_runs]
            for i in range(1, len(feature_hashes)):
                if feature_hashes[i] != feature_hashes[i-1] and feature_hashes[i] is not None:
                    breakpoints.append({
                        'run_index': i,
                        'run_id': valid_runs[i]['run_id'],
                        'created_at': valid_runs[i]['created_at'].isoformat(),
                        'reason': 'feature_registry_hash_changed',
                        'previous_hash': feature_hashes[i-1],
                        'new_hash': feature_hashes[i]
                    })
        
        # Generate alerts
        alerts = []
        if slope < self.suspicious_slope_threshold:
            alerts.append({
                'type': 'negative_slope',
                'severity': 'warning',
                'message': f"Slope = {slope:.6f} per day (declining trend)",
                'slope': slope,
                'threshold': self.suspicious_slope_threshold
            })
        
        # Check if current deviates significantly from prediction
        if resid_std > 0:
            deviation = abs(values[-1] - current_estimate)
            if deviation > 2 * resid_std:
                alerts.append({
                    'type': 'unexpected_deviation',
                    'severity': 'info',
                    'message': f"Current value ({values[-1]:.4f}) deviates from prediction ({current_estimate:.4f}) by {deviation:.4f} (> 2 * {resid_std:.4f})",
                    'deviation': deviation,
                    'residual_std': resid_std
                })
        
        return TrendResult(
            series_key=series_key,
            view=view,
            metric_name=metric_field,
            n_runs=len(valid_runs),
            status="ok",
            half_life_days=self.half_life_days,
            intercept=intercept,
            slope_per_day=slope,
            current_estimate=current_estimate,
            residual_std=resid_std,
            ewma_value=ewma_value,
            ewma_alpha=ewma_alpha,
            alerts=alerts,
            breakpoints=breakpoints
        )
    
    def analyze_all_series(
        self,
        view: SeriesView = SeriesView.STRICT,
        metric_fields: Optional[List[str]] = None,
        use_comparison_group: bool = True
    ) -> Dict[str, List[TrendResult]]:
        """
        Analyze trends for all series in the artifact index.
        
        Args:
            view: STRICT or PROGRESS
            metric_fields: List of metric fields to analyze (None = auto-detect per stage)
            use_comparison_group: If True, use comparison_group_key for grouping (lenient, like metrics diffs).
                                 If False, use strict SeriesKey matching (backward compatible).
        
        Returns:
            Dict mapping series_key -> list of TrendResult (one per metric)
        """
        df = self.load_artifact_index()
        if len(df) == 0:
            logger.warning("No runs found in artifact index")
            return {}
        
        # Group into series
        series = self.group_into_series(df, view, use_comparison_group=use_comparison_group)
        grouping_mode = "comparison_group" if use_comparison_group else "strict"
        logger.info(f"Found {len(series)} series for {view.value} view (grouping: {grouping_mode})")
        
        # Determine metric fields per stage
        if metric_fields is None:
            metric_fields_by_stage = {
                'TARGET_RANKING': ['auc_mean', 'composite_score', 'importance_mean'],
                'FEATURE_SELECTION': ['n_selected'],
                'TRAINING': ['primary_metric'],
                'MODEL_TRAINING': ['primary_metric']
            }
        else:
            metric_fields_by_stage = {stage: metric_fields for stage in df['stage'].unique()}
        
        # Analyze each series
        all_trends = {}
        skipped_series = []
        
        for series_key_str, runs in series.items():
            # Get series key object from first run
            series_key = runs[0].get('_series_key')
            if not series_key:
                logger.debug(f"SKIP(series={series_key_str[:80]}..., reason=missing_series_key)")
                skipped_series.append({
                    'series_key': series_key_str[:80],
                    'reason': 'missing_series_key',
                    'n_runs': len(runs)
                })
                continue
            
            # Check minimum runs requirement
            if len(runs) < self.min_runs_for_trend:
                if len(runs) == 1:
                    # Single run: log as first run (expected for new series)
                    logger.debug(
                        f"SKIP(series={series_key.stage}:{series_key.target}, "
                        f"reason=first_run, n=1, min={self.min_runs_for_trend}). "
                        f"Trends will be available after {self.min_runs_for_trend - 1} more run(s)."
                    )
                else:
                    logger.info(
                        f"SKIP(series={series_key.stage}:{series_key.target}, "
                        f"reason=insufficient_runs, n={len(runs)}, min={self.min_runs_for_trend})"
                    )
                skipped_series.append({
                    'series_key': f"{series_key.stage}:{series_key.target}",
                    'reason': 'first_run' if len(runs) == 1 else 'insufficient_runs',
                    'n_runs': len(runs),
                    'min_required': self.min_runs_for_trend
                })
                continue
            
            stage = series_key.stage
            metric_fields = metric_fields_by_stage.get(stage, ['primary_metric'])
            
            trends = []
            for metric_field in metric_fields:
                # Check if any run has this metric
                if not any(metric_field in r and pd.notna(r.get(metric_field)) for r in runs):
                    logger.debug(
                        f"SKIP(series={series_key.stage}:{series_key.target}, "
                        f"metric={metric_field}, reason=missing_metric)"
                    )
                    continue
                
                trend = self.analyze_series_trend(runs, metric_field, view, series_key)
                
                # Log skip reasons from analyze_series_trend
                if trend.status != "ok":
                    logger.info(
                        f"SKIP(series={series_key.stage}:{series_key.target}, "
                        f"metric={metric_field}, reason={trend.status}, n={trend.n_runs})"
                    )
                
                trends.append(trend)
            
            if trends:
                all_trends[series_key_str] = trends
            else:
                logger.debug(
                    f"SKIP(series={series_key.stage}:{series_key.target}, "
                    f"reason=no_valid_metrics, available_fields={list(set().union(*[list(r.keys()) for r in runs[:3]]))})"
                )
        
        # Log summary
        n_analyzed = len(all_trends)
        n_skipped = len(skipped_series)
        logger.info(f"Analyzed {n_analyzed} series, skipped {n_skipped} series")
        
        # Always log skip reasons at INFO level (not just DEBUG) for visibility
        if skipped_series:
            for skip in skipped_series[:5]:  # Show first 5
                logger.info(f"  SKIP: {skip['series_key']} - {skip['reason']} (n={skip.get('n_runs', '?')}, min={skip.get('min_required', '?')})")
        
        return all_trends
    
    def write_trend_report(
        self,
        trends: Dict[str, List[TrendResult]],
        output_path: Path
    ) -> None:
        """Write trend report to JSON file."""
        report = {
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'half_life_days': self.half_life_days,
            'min_runs_for_trend': self.min_runs_for_trend,
            'n_series': len(trends),
            'series': {}
        }
        
        for series_key, trend_list in trends.items():
            series_data = {
                'n_trends': len(trend_list),
                'trends': []
            }
            
            for trend in trend_list:
                trend_dict = {
                    'metric_name': trend.metric_name,
                    'status': trend.status,
                    'n_runs': trend.n_runs,
                    'slope_per_day': trend.slope_per_day,
                    'current_estimate': trend.current_estimate,
                    'ewma_value': trend.ewma_value,
                    'residual_std': trend.residual_std,
                    'alerts': trend.alerts,
                    'breakpoints': trend.breakpoints
                }
                series_data['trends'].append(trend_dict)
            
            report['series'][series_key] = series_data
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Wrote trend report to {output_path}")
    
    def write_cohort_trend(
        self,
        cohort_dir: Path,
        stage: str,
        target: str,
        trends: Optional[Dict[str, List[TrendResult]]] = None
    ) -> Optional[Path]:
        """
        Write trend.json to cohort directory (similar to metadata.json and metrics.json).
        
        Args:
            cohort_dir: Cohort directory path (where metadata.json and metrics.json are written)
            stage: Stage name (e.g., "TARGET_RANKING")
            target: Target name
            trends: Optional pre-computed trends dict. If None, will compute trends for this series.
        
        Returns:
            Path to written trend.json file, or None if writing failed
        """
        try:
            # If trends not provided, compute them
            if trends is None:
                df = self.load_artifact_index()
                if len(df) == 0:
                    logger.debug("No runs found in artifact index, skipping trend.json write")
                    return None
                
                # Group into series
                series = self.group_into_series(df, view=SeriesView.STRICT, use_comparison_group=True)
                
                # Find the series that matches this stage and target
                matching_series_key = None
                for series_key_str, runs in series.items():
                    # Check if this series matches
                    series_key = runs[0].get('_series_key') if runs else None
                    if series_key and series_key.stage == stage and series_key.target == target:
                        matching_series_key = series_key_str
                        break
                
                if not matching_series_key:
                    logger.debug(f"No matching series found for stage={stage}, target={target}, skipping trend.json write")
                    return None
                
                # Analyze trends for this series
                runs = series[matching_series_key]
                series_key = runs[0].get('_series_key')
                
                # Determine metric fields per stage
                metric_fields_by_stage = {
                    'TARGET_RANKING': ['auc_mean', 'composite_score', 'importance_mean'],
                    'FEATURE_SELECTION': ['n_selected'],
                    'TRAINING': ['primary_metric'],
                    'MODEL_TRAINING': ['primary_metric']
                }
                metric_fields = metric_fields_by_stage.get(stage, ['primary_metric'])
                
                trend_list = []
                for metric_field in metric_fields:
                    if any(metric_field in r and pd.notna(r.get(metric_field)) for r in runs):
                        trend = self.analyze_series_trend(runs, metric_field, SeriesView.STRICT, series_key)
                        if trend.status == "ok":
                            trend_list.append(trend)
                
                if not trend_list:
                    logger.debug(f"No valid trends found for stage={stage}, target={target}, skipping trend.json write")
                    return None
                
                trends = {matching_series_key: trend_list}
            
            # Build trend.json structure (similar to metadata.json/metrics.json format)
            trend_data = {
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'stage': stage,
                'target': target,
                'half_life_days': self.half_life_days,
                'min_runs_for_trend': self.min_runs_for_trend,
                'trends': []
            }
            
            # Extract trends for this series (should be only one series_key in trends dict)
            for series_key_str, trend_list in trends.items():
                for trend in trend_list:
                    trend_dict = {
                        'metric_name': trend.metric_name,
                        'status': trend.status,
                        'n_runs': trend.n_runs,
                        'slope_per_day': trend.slope_per_day,
                        'current_estimate': trend.current_estimate,
                        'ewma_value': trend.ewma_value,
                        'residual_std': trend.residual_std,
                        'alerts': trend.alerts,
                        'breakpoints': trend.breakpoints
                    }
                    trend_data['trends'].append(trend_dict)
            
            # Write to cohort directory using atomic write
            trend_file = cohort_dir / "trend.json"
            from TRAINING.common.utils.file_utils import write_atomic_json
            write_atomic_json(trend_file, trend_data)
            
            logger.info(f"✅ Wrote trend.json to {cohort_dir.name}/")
            return trend_file
            
        except Exception as e:
            logger.debug(f"Failed to write trend.json to {cohort_dir}: {e}")
            return None
    
    def write_across_runs_timeseries(
        self,
        results_dir: Path,
        target: str,
        stage: str = "TRAINING",
        view: str = "CROSS_SECTIONAL"
    ) -> Optional[Dict[str, Path]]:
        """
        Write across-runs time series to trend_reports/by_target/<target>/.
        
        This creates time series files indexed by run_id and timestamp for:
        - Performance metrics (performance_timeseries.parquet)
        - Routing scores (routing_score_timeseries.parquet)
        - Feature importance (feature_importance_timeseries.parquet)
        
        Args:
            results_dir: RESULTS directory (parent of runs/)
            target: Target name
            stage: Stage name (default: "TRAINING")
            view: View type (default: "CROSS_SECTIONAL")
        
        Returns:
            Dict mapping metric name to file path, or None if failed
        """
        try:
            # Find RESULTS directory (walk up from reproducibility_dir if needed)
            if results_dir.name != "RESULTS":
                # Try to find RESULTS directory
                current = Path(results_dir)
                for _ in range(10):
                    if current.name == "RESULTS":
                        results_dir = current
                        break
                    if not current.parent.exists():
                        break
                    current = current.parent
            
            # Create trend_reports structure outside run directories
            trend_reports_dir = results_dir / "trend_reports"
            by_target_dir = trend_reports_dir / "by_target"
            from TRAINING.orchestration.utils.target_first_paths import normalize_target_name
            target_trend_dir = by_target_dir / normalize_target_name(target)
            target_trend_dir.mkdir(parents=True, exist_ok=True)
            
            # Load artifact index
            df = self.load_artifact_index()
            if len(df) == 0:
                logger.debug(f"No runs found for across-runs timeseries for {target}")
                return None
            
            # Filter for this target and stage
            target_df = df[
                (df['target'] == target) & 
                (df.get('stage', '') == stage)
            ].copy()
            
            if len(target_df) == 0:
                logger.debug(f"No runs found for target={target}, stage={stage}")
                return None
            
            # Sort by timestamp
            if 'timestamp' in target_df.columns:
                target_df = target_df.sort_values('timestamp')
            
            written_files = {}
            
            # 1. Performance timeseries
            perf_columns = ['run_id', 'timestamp', 'auc_mean', 'composite_score', 'primary_metric', 
                          'auc', 'std_score', 'importance_mean']
            perf_cols = [c for c in perf_columns if c in target_df.columns]
            if perf_cols:
                perf_df = target_df[perf_cols].copy()
                if 'timestamp' in perf_df.columns:
                    perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'], errors='coerce')
                perf_path = target_trend_dir / "performance_timeseries.parquet"
                perf_df.to_parquet(perf_path, index=False, engine='pyarrow', compression='snappy')
                written_files['performance'] = perf_path
                logger.debug(f"Wrote performance timeseries: {perf_path}")
            
            # 2. Routing score timeseries (if available)
            routing_cols = ['run_id', 'timestamp', 'routing_score', 'confidence', 'score_tier']
            routing_cols = [c for c in routing_cols if c in target_df.columns]
            if routing_cols:
                routing_df = target_df[routing_cols].copy()
                if 'timestamp' in routing_df.columns:
                    routing_df['timestamp'] = pd.to_datetime(routing_df['timestamp'], errors='coerce')
                routing_path = target_trend_dir / "routing_score_timeseries.parquet"
                routing_df.to_parquet(routing_path, index=False, engine='pyarrow', compression='snappy')
                written_files['routing_score'] = routing_path
                logger.debug(f"Wrote routing score timeseries: {routing_path}")
            
            # 3. Feature importance timeseries (if available)
            # This would need to be aggregated from feature importance snapshots
            # For now, we'll create a placeholder structure
            feat_importance_cols = ['run_id', 'timestamp', 'n_selected', 'feature_registry_hash']
            feat_importance_cols = [c for c in feat_importance_cols if c in target_df.columns]
            if feat_importance_cols:
                feat_df = target_df[feat_importance_cols].copy()
                if 'timestamp' in feat_df.columns:
                    feat_df['timestamp'] = pd.to_datetime(feat_df['timestamp'], errors='coerce')
                feat_path = target_trend_dir / "feature_importance_timeseries.parquet"
                feat_df.to_parquet(feat_path, index=False, engine='pyarrow', compression='snappy')
                written_files['feature_importance'] = feat_path
                logger.debug(f"Wrote feature importance timeseries: {feat_path}")
            
            if written_files:
                logger.info(f"✅ Wrote across-runs timeseries for {target}: {len(written_files)} files")
                return written_files
            else:
                logger.debug(f"No timeseries data available for {target}")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to write across-runs timeseries for {target}: {e}")
            return None
    
    def write_run_snapshot(
        self,
        results_dir: Path,
        run_id: str,
        trends: Optional[Dict[str, List[TrendResult]]] = None
    ) -> Optional[Path]:
        """
        Write cached snapshot for a run to trend_reports/by_run/<run_id>/.
        
        Args:
            results_dir: RESULTS directory (parent of runs/)
            run_id: Run identifier
            trends: Optional pre-computed trends dict
        
        Returns:
            Path to written snapshot file, or None if failed
        """
        try:
            # Find RESULTS directory
            if results_dir.name != "RESULTS":
                current = Path(results_dir)
                for _ in range(10):
                    if current.name == "RESULTS":
                        results_dir = current
                        break
                    if not current.parent.exists():
                        break
                    current = current.parent
            
            # Create trend_reports structure
            trend_reports_dir = results_dir / "trend_reports"
            by_run_dir = trend_reports_dir / "by_run"
            run_snapshot_dir = by_run_dir / run_id
            run_snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # If trends not provided, try to compute from artifact index
            if trends is None:
                df = self.load_artifact_index()
                run_df = df[df.get('run_id', '') == run_id]
                if len(run_df) == 0:
                    logger.debug(f"No data found for run_id={run_id} in artifact index")
                    return None
                
                # Group into series and compute trends
                series = self.group_into_series(df, view=SeriesView.STRICT, use_comparison_group=True)
                trends = {}
                for series_key_str, runs in series.items():
                    if any(r.get('run_id') == run_id for r in runs):
                        # Compute trends for this series
                        series_key = runs[0].get('_series_key') if runs else None
                        if series_key:
                            metric_fields = ['primary_metric', 'auc_mean', 'composite_score']
                            trend_list = []
                            for metric_field in metric_fields:
                                if any(metric_field in r and pd.notna(r.get(metric_field)) for r in runs):
                                    trend = self.analyze_series_trend(runs, metric_field, SeriesView.STRICT, series_key)
                                    if trend.status == "ok":
                                        trend_list.append(trend)
                            if trend_list:
                                trends[series_key_str] = trend_list
            
            if not trends:
                logger.debug(f"No trends computed for run_id={run_id}")
                return None
            
            # Build snapshot structure
            snapshot = {
                'run_id': run_id,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'half_life_days': self.half_life_days,
                'min_runs_for_trend': self.min_runs_for_trend,
                'series': {}
            }
            
            for series_key_str, trend_list in trends.items():
                series_data = {
                    'n_trends': len(trend_list),
                    'trends': []
                }
                for trend in trend_list:
                    trend_dict = {
                        'metric_name': trend.metric_name,
                        'status': trend.status,
                        'n_runs': trend.n_runs,
                        'slope_per_day': trend.slope_per_day,
                        'current_estimate': trend.current_estimate,
                        'ewma_value': trend.ewma_value,
                        'residual_std': trend.residual_std,
                        'alerts': trend.alerts,
                        'breakpoints': trend.breakpoints
                    }
                    series_data['trends'].append(trend_dict)
                snapshot['series'][series_key_str] = series_data
            
            # Write snapshot
            snapshot_path = run_snapshot_dir / f"{run_id}_summary.json"
            with open(snapshot_path, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)
            
            logger.info(f"✅ Wrote run snapshot: {snapshot_path}")
            return snapshot_path
            
        except Exception as e:
            logger.warning(f"Failed to write run snapshot for {run_id}: {e}")
            return None