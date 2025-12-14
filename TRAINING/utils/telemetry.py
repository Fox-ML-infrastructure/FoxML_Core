"""
Telemetry System (Sidecar-based, View-Isolated)

Writes telemetry as sidecar files next to existing artifacts (metadata.json, metrics.json, audit_report.json).
Telemetry follows the exact same directory structure as existing artifacts.

Key principles:
- View isolation: CROSS_SECTIONAL drift only compares to CROSS_SECTIONAL baselines
- SYMBOL_SPECIFIC drift only compares to SYMBOL_SPECIFIC baselines
- Sidecar placement: telemetry files live in same cohort folder as existing JSONs
- Hierarchical rollups: per-target/symbol → view-level → stage-level
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TelemetryWriter:
    """
    Writes telemetry as sidecar files in cohort directories.
    
    Structure:
    - Sidecar files in each cohort folder: telemetry_metrics.json, telemetry_drift.json, telemetry_trend.json
    - View-level rollups: CROSS_SECTIONAL/telemetry_rollup.json, SYMBOL_SPECIFIC/telemetry_rollup.json
    - Stage-level container: TARGET_RANKING/telemetry_rollup.json
    """
    
    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        baselines: Optional[Dict[str, Any]] = None,
        drift: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize telemetry writer.
        
        Args:
            output_dir: Base output directory (REPRODUCIBILITY/ is under this)
            enabled: Whether telemetry is enabled
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
    
    def write_cohort_telemetry(
        self,
        cohort_dir: Path,
        stage: str,
        view: str,
        target: Optional[str],
        symbol: Optional[str],
        run_id: str,
        metrics: Dict[str, Any],
        baseline_key: Optional[str] = None
    ) -> None:
        """
        Write telemetry sidecar files for a cohort.
        
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
            return
        
        cohort_dir = Path(cohort_dir)
        if not cohort_dir.exists():
            logger.warning(f"Cohort directory does not exist: {cohort_dir}")
            return
        
        # Write telemetry_metrics.json (facts for this cohort)
        self._write_telemetry_metrics(cohort_dir, run_id, metrics)
        
        # Write telemetry_drift.json (comparison to baseline)
        if baseline_key:
            self._write_telemetry_drift(
                cohort_dir, stage, view, target, symbol, run_id, metrics, baseline_key
            )
    
    def _write_telemetry_metrics(
        self,
        cohort_dir: Path,
        run_id: str,
        metrics: Dict[str, Any]
    ) -> None:
        """Write telemetry_metrics.json with facts for this cohort."""
        telemetry_data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Extract telemetry-relevant metrics
        for key, value in metrics.items():
            if key in ['timestamp', 'cohort_metadata', 'additional_data']:
                continue
            
            # Convert to float if numeric
            if isinstance(value, (int, float)):
                telemetry_data["metrics"][key] = float(value)
            elif isinstance(value, (list, dict)):
                # Store complex values as-is (for now)
                telemetry_data["metrics"][key] = value
            else:
                # Store other types as string representation
                telemetry_data["metrics"][key] = str(value)
        
        metrics_file_json = cohort_dir / "telemetry_metrics.json"
        metrics_file_parquet = cohort_dir / "telemetry_metrics.parquet"
        try:
            # Write JSON (human-readable)
            with open(metrics_file_json, 'w') as f:
                json.dump(telemetry_data, f, indent=2)
            
            # Write Parquet (queryable, long format)
            # Flatten metrics dict into DataFrame
            parquet_rows = [{
                "run_id": run_id,
                "timestamp": telemetry_data.get("timestamp"),
                "metric_name": k,
                "metric_value": v if isinstance(v, (int, float)) else None,
                "metric_value_str": str(v) if not isinstance(v, (int, float)) else None
            } for k, v in telemetry_data.get("metrics", {}).items()]
            
            if parquet_rows:
                df_metrics = pd.DataFrame(parquet_rows)
                df_metrics.to_parquet(metrics_file_parquet, index=False, engine='pyarrow', compression='snappy')
        except Exception as e:
            logger.warning(f"Failed to write telemetry_metrics to {cohort_dir}: {e}")
    
    def _write_telemetry_drift(
        self,
        cohort_dir: Path,
        stage: str,
        view: str,
        target: Optional[str],
        symbol: Optional[str],
        run_id: str,
        current_metrics: Dict[str, Any],
        baseline_key: str
    ) -> None:
        """
        Write telemetry_drift.json comparing current run to baseline.
        
        Baseline key format: (stage, view, target[, symbol])
        This ensures view isolation: CS only compares to CS, SS only compares to SS.
        """
        # Find baseline cohort directory
        baseline_cohort_dir = self._find_baseline_cohort(
            cohort_dir.parent.parent,  # Go up to view level
            stage, view, target, symbol, baseline_key
        )
        
        if baseline_cohort_dir is None or not baseline_cohort_dir.exists():
            logger.debug(f"No baseline found for drift comparison: {baseline_key}")
            return
        
        # Load baseline metrics
        baseline_metrics_file = baseline_cohort_dir / "telemetry_metrics.json"
        if not baseline_metrics_file.exists():
            # Fallback to metrics.json
            baseline_metrics_file = baseline_cohort_dir / "metrics.json"
            if not baseline_metrics_file.exists():
                logger.debug(f"Baseline metrics not found: {baseline_metrics_file}")
                return
        
        try:
            with open(baseline_metrics_file, 'r') as f:
                baseline_data = json.load(f)
            
            # Extract baseline metrics
            if "metrics" in baseline_data:
                baseline_metrics = baseline_data["metrics"]
            else:
                baseline_metrics = baseline_data
            
            # Compute drift
            drift_results = {
                "current_run_id": run_id,
                "baseline_run_id": baseline_data.get("run_id", "unknown"),
                "baseline_key": baseline_key,
                "timestamp": datetime.now().isoformat(),
                "view": view,
                "target": target,
                "symbol": symbol,
                "drift_metrics": {}
            }
            
            # Compare numeric metrics
            for metric_name in set(current_metrics.keys()) & set(baseline_metrics.keys()):
                curr_val = current_metrics.get(metric_name)
                base_val = baseline_metrics.get(metric_name)
                
                if not isinstance(curr_val, (int, float)) or not isinstance(base_val, (int, float)):
                    continue
                
                delta = float(curr_val) - float(base_val)
                rel_delta = delta / float(base_val) if base_val != 0 else None
                
                drift_results["drift_metrics"][metric_name] = {
                    "current": float(curr_val),
                    "baseline": float(base_val),
                    "delta": delta,
                    "rel_delta": rel_delta,
                    "status": self._classify_drift(delta, rel_delta)
                }
            
            # Write drift files (JSON for human-readable, Parquet for queryable)
            drift_file_json = cohort_dir / "telemetry_drift.json"
            drift_file_parquet = cohort_dir / "telemetry_drift.parquet"
            try:
                # Write JSON (human-readable)
                with open(drift_file_json, 'w') as f:
                    json.dump(drift_results, f, indent=2)
                
                # Write Parquet (queryable, long format)
                # Flatten drift_metrics dict into DataFrame
                parquet_rows = []
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
                            "status": drift_info.get("status")
                        }
                        parquet_rows.append(row)
                
                if parquet_rows:
                    df_drift = pd.DataFrame(parquet_rows)
                    df_drift.to_parquet(drift_file_parquet, index=False, engine='pyarrow', compression='snappy')
            except Exception as e:
                logger.warning(f"Failed to write telemetry_drift files to {cohort_dir}: {e}")
            
        except Exception as e:
            logger.warning(f"Failed to compute drift for {cohort_dir}: {e}")
    
    def _find_baseline_cohort(
        self,
        view_dir: Path,
        stage: str,
        view: str,
        target: Optional[str],
        symbol: Optional[str],
        baseline_key: str
    ) -> Optional[Path]:
        """
        Find baseline cohort directory using baseline_key.
        
        baseline_key format: (stage, view, target[, symbol])
        This ensures we only compare within the same view.
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
        if baseline_symbol and view == "SYMBOL_SPECIFIC":
            symbol_dir = target_dir / f"symbol={baseline_symbol}"
            if not symbol_dir.exists():
                return None
        else:
            symbol_dir = target_dir
        
        # Find most recent cohort (previous run)
        cohort_dirs = sorted(
            [d for d in symbol_dir.iterdir() if d.is_dir() and d.name.startswith("cohort=")],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Skip current run, return previous
        if len(cohort_dirs) > 1:
            return cohort_dirs[1]  # Second most recent (previous run)
        elif len(cohort_dirs) == 1:
            return cohort_dirs[0]  # Only one run, use it as baseline
        
        return None
    
    def _classify_drift(self, delta: float, rel_delta: Optional[float]) -> str:
        """Classify drift status based on thresholds."""
        if rel_delta is not None:
            abs_rel_delta = abs(rel_delta)
            if abs_rel_delta < 0.05:  # 5% change
                return "STABLE"
            elif abs_rel_delta < 0.20:  # 20% change
                return "DRIFTING"
            else:
                return "DIVERGED"
        else:
            # Use absolute delta
            if abs(delta) < 0.01:
                return "STABLE"
            elif abs(delta) < 0.10:
                return "DRIFTING"
            else:
                return "DIVERGED"
    
    def generate_view_rollup(
        self,
        view_dir: Path,
        stage: str,
        view: str,
        run_id: str
    ) -> None:
        """
        Generate view-level rollup (CROSS_SECTIONAL/telemetry_rollup.json or SYMBOL_SPECIFIC/telemetry_rollup.json).
        
        Args:
            view_dir: Path to view directory (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
            stage: Pipeline stage
            view: View type
            run_id: Current run identifier
        """
        if not self.enabled:
            return
        
        view_dir = Path(view_dir)
        if not view_dir.exists():
            return
        
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
        if view == "CROSS_SECTIONAL":
            for target_dir in view_dir.iterdir():
                if not target_dir.is_dir() or target_dir.name.startswith("telemetry"):
                    continue
                
                target_name = target_dir.name
                target_metrics = []
                
                # Find most recent cohort for this target
                cohort_dirs = sorted(
                    [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("cohort=")],
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )
                
                if cohort_dirs:
                    latest_cohort = cohort_dirs[0]
                    metrics_file = latest_cohort / "telemetry_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                target_data = json.load(f)
                                target_metrics = target_data.get("metrics", {})
                                rollup_data["targets"][target_name] = target_metrics
                                all_metrics.append(target_metrics)
                        except Exception as e:
                            logger.debug(f"Failed to load metrics for {target_name}: {e}")
        
        # For SYMBOL_SPECIFIC: iterate per-target, then per-symbol
        elif view == "SYMBOL_SPECIFIC":
            for target_dir in view_dir.iterdir():
                if not target_dir.is_dir() or target_dir.name.startswith("telemetry"):
                    continue
                
                target_name = target_dir.name
                
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
                        metrics_file = latest_cohort / "telemetry_metrics.json"
                        if metrics_file.exists():
                            try:
                                with open(metrics_file, 'r') as f:
                                    symbol_data = json.load(f)
                                    symbol_metrics = symbol_data.get("metrics", {})
                                    
                                    if target_name not in rollup_data["targets"]:
                                        rollup_data["targets"][target_name] = {}
                                    rollup_data["targets"][target_name][symbol_name] = symbol_metrics
                                    
                                    if symbol_name not in rollup_data["symbols"]:
                                        rollup_data["symbols"][symbol_name] = {}
                                    rollup_data["symbols"][symbol_name][target_name] = symbol_metrics
                                    
                                    all_metrics.append(symbol_metrics)
                            except Exception as e:
                                logger.debug(f"Failed to load metrics for {target_name}/{symbol_name}: {e}")
        
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
        rollup_file_json = view_dir / "telemetry_rollup.json"
        rollup_file_parquet = view_dir / "telemetry_rollup.parquet"
        try:
            # Write JSON rollup
            with open(rollup_file_json, 'w') as f:
                json.dump(rollup_data, f, indent=2)
            
            # Write Parquet rollup (flattened, queryable format)
            # Convert rollup_data to DataFrame with one row per target/symbol
            parquet_rows = []
            for target_name, target_data in rollup_data.get("targets", {}).items():
                if isinstance(target_data, dict) and "symbols" not in target_data:
                    # CROSS_SECTIONAL: target-level metrics
                    row = {
                        "run_id": run_id,
                        "stage": stage,
                        "view": view,
                        "target": target_name,
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
                            "target": target_name,
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
                df_rollup = pd.DataFrame(parquet_rows)
                df_rollup.to_parquet(rollup_file_parquet, index=False, engine='pyarrow', compression='snappy')
                logger.debug(f"✅ Wrote view rollup: {rollup_file_json.name} and {rollup_file_parquet.name}")
        except Exception as e:
            logger.warning(f"Failed to write view rollup to {rollup_file_json}: {e}")
    
    def generate_stage_rollup(
        self,
        stage_dir: Path,
        stage: str,
        run_id: str
    ) -> None:
        """
        Generate stage-level container rollup (TARGET_RANKING/telemetry_rollup.json).
        
        This is a container that references view-level rollups, no drift mixing.
        """
        if not self.enabled:
            return
        
        stage_dir = Path(stage_dir)
        if not stage_dir.exists():
            return
        
        rollup_data = {
            "run_id": run_id,
            "stage": stage,
            "timestamp": datetime.now().isoformat(),
            "views": {}
        }
        
        # Load view-level rollups
        for view in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
            view_dir = stage_dir / view
            if view_dir.exists():
                view_rollup_file = view_dir / "telemetry_rollup.json"
                if view_rollup_file.exists():
                    try:
                        with open(view_rollup_file, 'r') as f:
                            rollup_data["views"][view] = json.load(f)
                    except Exception as e:
                        logger.debug(f"Failed to load view rollup for {view}: {e}")
        
        # Write stage rollup files (JSON for human-readable, Parquet for queryable)
        rollup_file_json = stage_dir / "telemetry_rollup.json"
        rollup_file_parquet = stage_dir / "telemetry_rollup.parquet"
        try:
            # Write JSON rollup
            with open(rollup_file_json, 'w') as f:
                json.dump(rollup_data, f, indent=2)
            
            # Write Parquet rollup (flattened from view-level rollups)
            parquet_rows = []
            for view_name, view_data in rollup_data.get("views", {}).items():
                # Extract targets from view rollup
                for target_name, target_data in view_data.get("targets", {}).items():
                    if isinstance(target_data, dict) and "symbols" not in target_data:
                        # CROSS_SECTIONAL: target-level
                        row = {
                            "run_id": run_id,
                            "stage": stage,
                            "view": view_name,
                            "target": target_name,
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
                                "target": target_name,
                                "symbol": symbol_name,
                                "timestamp": rollup_data.get("timestamp"),
                                **{k: v for k, v in symbol_metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
                            }
                            parquet_rows.append(row)
            
            if parquet_rows:
                df_rollup = pd.DataFrame(parquet_rows)
                df_rollup.to_parquet(rollup_file_parquet, index=False, engine='pyarrow', compression='snappy')
                logger.debug(f"✅ Wrote stage rollup: {rollup_file_json.name} and {rollup_file_parquet.name}")
        except Exception as e:
            logger.warning(f"Failed to write stage rollup to {rollup_file_json}: {e}")


def load_telemetry_config() -> Dict[str, Any]:
    """Load telemetry configuration from safety.yaml."""
    try:
        from CONFIG.config_loader import get_cfg
        return {
            "enabled": get_cfg("safety.telemetry.enabled", default=True, config_name="safety_config"),
            "baselines": {
                "previous_run": get_cfg("safety.telemetry.baselines.previous_run", default=True, config_name="safety_config"),
                "rolling_window_k": get_cfg("safety.telemetry.baselines.rolling_window_k", default=10, config_name="safety_config"),
                "last_good_run": get_cfg("safety.telemetry.baselines.last_good_run", default=True, config_name="safety_config")
            },
            "drift": {
                "psi_threshold": get_cfg("safety.telemetry.drift.psi_threshold", default=0.2, config_name="safety_config"),
                "ks_threshold": get_cfg("safety.telemetry.drift.ks_threshold", default=0.1, config_name="safety_config")
            }
        }
    except Exception as e:
        logger.warning(f"Failed to load telemetry config: {e}, using defaults")
        return {
            "enabled": True,
            "baselines": {"previous_run": True, "rolling_window_k": 10, "last_good_run": True},
            "drift": {"psi_threshold": 0.2, "ks_threshold": 0.1}
        }


def aggregate_telemetry_facts(
    repro_dir: Path,
    facts_file: Optional[Path] = None
) -> pd.DataFrame:
    """
    Aggregate all telemetry_metrics.json files into a Parquet facts table.
    
    This creates/updates a long-format Parquet table with explicit dimensions:
    - run_id, stage, view, target, symbol, universe_id, cohort_id
    - All metric values as separate columns
    
    Structure matches current telemetry tracking exactly.
    
    Args:
        repro_dir: REPRODUCIBILITY directory root
        facts_file: Optional path to facts table (defaults to repro_dir/telemetry_facts.parquet)
    
    Returns:
        DataFrame with all telemetry facts
    """
    if facts_file is None:
        facts_file = repro_dir / "telemetry_facts.parquet"
    
    repro_dir = Path(repro_dir)
    if not repro_dir.exists():
        logger.warning(f"REPRODUCIBILITY directory does not exist: {repro_dir}")
        return pd.DataFrame()
    
    # Collect all telemetry_metrics.json files
    rows = []
    
    # Walk REPRODUCIBILITY structure
    for stage_dir in repro_dir.iterdir():
        if not stage_dir.is_dir() or stage_dir.name in ["artifact_index.parquet", "telemetry_facts.parquet", "stats.json"]:
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
                universe_id = None
                
                # Check if this is a symbol directory (SYMBOL_SPECIFIC view)
                if view == "SYMBOL_SPECIFIC":
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
                                telemetry_file = cohort_path / "telemetry_metrics.json"
                                
                                if telemetry_file.exists():
                                    row = _extract_telemetry_row(
                                        telemetry_file, stage, view, target, symbol, 
                                        cohort_id, universe_id
                                    )
                                    if row:
                                        rows.append(row)
                else:
                    # CROSS_SECTIONAL: cohorts directly under target
                    for cohort_path in target_path.iterdir():
                        if not cohort_path.is_dir() or not cohort_path.name.startswith("cohort="):
                            continue
                        
                        cohort_id = cohort_path.name.replace("cohort=", "")
                        telemetry_file = cohort_path / "telemetry_metrics.json"
                        
                        if telemetry_file.exists():
                            # Try to extract universe_id from metadata
                            metadata_file = cohort_path / "metadata.json"
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    universe_id = metadata.get('universe_id')
                                except Exception:
                                    pass
                            
                            row = _extract_telemetry_row(
                                telemetry_file, stage, view, target, symbol,
                                cohort_id, universe_id
                            )
                            if row:
                                rows.append(row)
    
    if not rows:
        logger.debug("No telemetry metrics found to aggregate")
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
        logger.info(f"✅ Aggregated {len(df)} telemetry facts to {facts_file}")
    except Exception as e:
        logger.warning(f"Failed to save telemetry facts table: {e}")
    
    return df


def _extract_telemetry_row(
    telemetry_file: Path,
    stage: str,
    view: str,
    target: str,
    symbol: Optional[str],
    cohort_id: str,
    universe_id: Optional[str]
) -> Optional[Dict[str, Any]]:
    """
    Extract a single row from telemetry_metrics.json for facts table.
    
    Returns:
        Dict with dimensions and metrics, or None if extraction fails
    """
    try:
        with open(telemetry_file, 'r') as f:
            data = json.load(f)
        
        # Extract dimensions
        row = {
            'run_id': data.get('run_id', ''),
            'timestamp': data.get('timestamp', ''),
            'stage': stage,
            'view': view,
            'target': target,
            'symbol': symbol if symbol else None,
            'universe_id': universe_id if universe_id else None,
            'cohort_id': cohort_id
        }
        
        # Extract metrics (flatten nested metrics dict)
        metrics = data.get('metrics', {})
        for key, value in metrics.items():
            # Convert to appropriate type
            if isinstance(value, (int, float)):
                row[key] = float(value)
            elif isinstance(value, bool):
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
        logger.debug(f"Failed to extract telemetry row from {telemetry_file}: {e}")
        return None
