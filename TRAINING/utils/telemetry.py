"""
Telemetry System

Implements a cube-based telemetry system with explicit dimensions and rollups.
Generates atomic metrics at finest granularity, then rolls them up.

Dimensions:
- run_id
- view: CROSS_SECTIONAL | SYMBOL_SPECIFIC | BOTH
- target
- symbol (nullable for cross-sectional aggregates)
- universe_id (optional, for symbol set identification)

Granularity levels:
1. Per symbol (individual)
2. Per cross-sectional (group)
3. Per target (across symbols)
4. Per run (global rollup)
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
    Writes telemetry facts to Parquet and generates rollups.
    
    Facts table schema:
    - run_id: str
    - baseline_run_id: Optional[str]
    - level: str (run|view|target|symbol|target_symbol)
    - view: Optional[str] (CROSS_SECTIONAL|SYMBOL_SPECIFIC|BOTH)
    - target: Optional[str]
    - symbol: Optional[str]
    - universe_id: Optional[str]
    - metric_name: str
    - metric_value: float
    - metric_unit: Optional[str]
    - status: Optional[str]
    - severity: Optional[str]
    - extras_json: Optional[str] (JSON string for small blobs)
    """
    
    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        levels: Optional[Dict[str, bool]] = None,
        baselines: Optional[Dict[str, Any]] = None,
        drift: Optional[Dict[str, Any]] = None,
        rollups: Optional[Dict[str, bool]] = None
    ):
        """
        Initialize telemetry writer.
        
        Args:
            output_dir: Base output directory (telemetry/ subdirectory created here)
            enabled: Whether telemetry is enabled
            levels: Dict of level flags (run, view, target, symbol, target_symbol)
            baselines: Baseline configuration (previous_run, rolling_window_k, last_good_run)
            drift: Drift detection configuration
            rollups: Rollup flags (per_symbol, per_cross_sectional, per_target, per_run)
        """
        self.output_dir = Path(output_dir)
        self.telemetry_dir = self.output_dir / "REPRODUCIBILITY" / "TELEMETRY"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        self.enabled = enabled
        
        # Default levels
        self.levels = {
            "run": True,
            "view": True,
            "target": True,
            "symbol": True,
            "target_symbol": False  # Expensive, opt-in
        }
        if levels:
            self.levels.update(levels)
        
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
            "feature_set_mode": "topk_importance",  # or "fixed_list"
            "topk": 50,
            "psi_threshold": 0.2,
            "ks_threshold": 0.1
        }
        if drift:
            self.drift.update(drift)
        
        # Default rollups
        self.rollups = {
            "per_symbol": True,
            "per_cross_sectional": True,
            "per_target": True,
            "per_run": True
        }
        if rollups:
            self.rollups.update(rollups)
        
        # Facts accumulator (list of dicts, converted to DataFrame on flush)
        self.facts: List[Dict[str, Any]] = []
        
        # Track current run_id
        self.current_run_id: Optional[str] = None
    
    def record_fact(
        self,
        run_id: str,
        level: str,
        metric_name: str,
        metric_value: float,
        view: Optional[str] = None,
        target: Optional[str] = None,
        symbol: Optional[str] = None,
        universe_id: Optional[str] = None,
        baseline_run_id: Optional[str] = None,
        metric_unit: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        extras: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record a single telemetry fact.
        
        Args:
            run_id: Current run identifier
            level: Granularity level (run|view|target|symbol|target_symbol)
            metric_name: Name of the metric
            metric_value: Metric value (float)
            view: View type (CROSS_SECTIONAL|SYMBOL_SPECIFIC|BOTH)
            target: Target name (optional)
            symbol: Symbol name (optional)
            universe_id: Universe identifier (optional)
            baseline_run_id: Baseline run for comparison (optional)
            metric_unit: Unit of measurement (optional)
            status: Status indicator (optional)
            severity: Severity level (optional)
            extras: Additional metadata as dict (will be JSON-serialized)
        """
        if not self.enabled:
            return
        
        # Validate level
        valid_levels = ["run", "view", "target", "symbol", "target_symbol"]
        if level not in valid_levels:
            logger.warning(f"Invalid level '{level}', must be one of {valid_levels}")
            return
        
        # Check if level is enabled
        if not self.levels.get(level, False):
            return
        
        fact = {
            "run_id": run_id,
            "baseline_run_id": baseline_run_id,
            "level": level,
            "view": view,
            "target": target,
            "symbol": symbol,
            "universe_id": universe_id,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "metric_unit": metric_unit,
            "status": status,
            "severity": severity,
            "extras_json": json.dumps(extras) if extras else None
        }
        
        self.facts.append(fact)
        self.current_run_id = run_id
    
    def flush(self) -> None:
        """Write facts to Parquet and generate rollups."""
        if not self.enabled or not self.facts:
            return
        
        if not self.current_run_id:
            logger.warning("No run_id set, cannot flush telemetry")
            return
        
        # Convert facts to DataFrame
        df_facts = pd.DataFrame(self.facts)
        
        # Write facts table
        facts_file = self.telemetry_dir / "facts.parquet"
        df_facts.to_parquet(facts_file, index=False, engine='pyarrow')
        logger.info(f"✅ Telemetry: Saved {len(self.facts)} facts to {facts_file}")
        
        # Generate rollups
        self._generate_rollups(df_facts)
        
        # Generate summary JSON
        self._generate_summary(df_facts)
        
        # Clear facts accumulator
        self.facts = []
    
    def _generate_rollups(self, df_facts: pd.DataFrame) -> None:
        """Generate rollup aggregations."""
        rollups = []
        
        # Per symbol aggregate
        if self.rollups.get("per_symbol", False):
            symbol_rollups = self._rollup_symbol(df_facts)
            rollups.extend(symbol_rollups)
        
        # Per cross-sectional aggregate
        if self.rollups.get("per_cross_sectional", False):
            cs_rollups = self._rollup_cross_sectional(df_facts)
            rollups.extend(cs_rollups)
        
        # Per target aggregate
        if self.rollups.get("per_target", False):
            target_rollups = self._rollup_target(df_facts)
            rollups.extend(target_rollups)
        
        # Per run aggregate
        if self.rollups.get("per_run", False):
            run_rollups = self._rollup_run(df_facts)
            rollups.extend(run_rollups)
        
        if rollups:
            df_rollups = pd.DataFrame(rollups)
            rollups_file = self.telemetry_dir / "rollups.parquet"
            df_rollups.to_parquet(rollups_file, index=False, engine='pyarrow')
            logger.info(f"✅ Telemetry: Saved {len(rollups)} rollups to {rollups_file}")
    
    def _rollup_symbol(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rollup by (run_id, view, symbol) and optionally (run_id, target, view, symbol)."""
        rollups = []
        
        # Group by (run_id, view, symbol)
        symbol_groups = df[df['symbol'].notna()].groupby(['run_id', 'view', 'symbol'])
        for (run_id, view, symbol), group in symbol_groups:
            rollup = {
                "rollup_type": "per_symbol",
                "run_id": run_id,
                "view": view,
                "symbol": symbol,
                "target": None,
                "universe_id": group['universe_id'].iloc[0] if group['universe_id'].notna().any() else None,
                "n_metrics": len(group),
                "metrics": self._aggregate_metrics(group)
            }
            rollups.append(rollup)
        
        # Group by (run_id, target, view, symbol) if target_symbol level enabled
        if self.levels.get("target_symbol", False):
            target_symbol_groups = df[(df['target'].notna()) & (df['symbol'].notna())].groupby(['run_id', 'target', 'view', 'symbol'])
            for (run_id, target, view, symbol), group in target_symbol_groups:
                rollup = {
                    "rollup_type": "per_target_symbol",
                    "run_id": run_id,
                    "view": view,
                    "symbol": symbol,
                    "target": target,
                    "universe_id": group['universe_id'].iloc[0] if group['universe_id'].notna().any() else None,
                    "n_metrics": len(group),
                    "metrics": self._aggregate_metrics(group)
                }
                rollups.append(rollup)
        
        return rollups
    
    def _rollup_cross_sectional(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rollup by (run_id, view, universe_id) and optionally (run_id, target, view, universe_id)."""
        rollups = []
        
        # Group by (run_id, view, universe_id)
        cs_groups = df[df['view'].notna()].groupby(['run_id', 'view', 'universe_id'])
        for (run_id, view, universe_id), group in cs_groups:
            rollup = {
                "rollup_type": "per_cross_sectional",
                "run_id": run_id,
                "view": view,
                "symbol": None,
                "target": None,
                "universe_id": universe_id,
                "n_metrics": len(group),
                "n_symbols": group['symbol'].nunique() if group['symbol'].notna().any() else 0,
                "metrics": self._aggregate_metrics(group)
            }
            rollups.append(rollup)
        
        # Group by (run_id, target, view, universe_id)
        target_cs_groups = df[(df['target'].notna()) & (df['view'].notna())].groupby(['run_id', 'target', 'view', 'universe_id'])
        for (run_id, target, view, universe_id), group in target_cs_groups:
            rollup = {
                "rollup_type": "per_target_cross_sectional",
                "run_id": run_id,
                "view": view,
                "symbol": None,
                "target": target,
                "universe_id": universe_id,
                "n_metrics": len(group),
                "n_symbols": group['symbol'].nunique() if group['symbol'].notna().any() else 0,
                "metrics": self._aggregate_metrics(group)
            }
            rollups.append(rollup)
        
        return rollups
    
    def _rollup_target(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rollup by (run_id, target, view) and optionally (run_id, target, view, universe_id)."""
        rollups = []
        
        # Group by (run_id, target, view)
        target_groups = df[df['target'].notna()].groupby(['run_id', 'target', 'view'])
        for (run_id, target, view), group in target_groups:
            rollup = {
                "rollup_type": "per_target",
                "run_id": run_id,
                "view": view,
                "symbol": None,
                "target": target,
                "universe_id": group['universe_id'].iloc[0] if group['universe_id'].notna().any() else None,
                "n_metrics": len(group),
                "n_symbols": group['symbol'].nunique() if group['symbol'].notna().any() else 0,
                "metrics": self._aggregate_metrics(group)
            }
            rollups.append(rollup)
        
        return rollups
    
    def _rollup_run(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Rollup by (run_id) and (run_id, view)."""
        rollups = []
        
        # Group by (run_id)
        run_groups = df.groupby('run_id')
        for run_id, group in run_groups:
            rollup = {
                "rollup_type": "per_run",
                "run_id": run_id,
                "view": None,
                "symbol": None,
                "target": None,
                "universe_id": None,
                "n_metrics": len(group),
                "n_targets": group['target'].nunique() if group['target'].notna().any() else 0,
                "n_symbols": group['symbol'].nunique() if group['symbol'].notna().any() else 0,
                "n_views": group['view'].nunique() if group['view'].notna().any() else 0,
                "metrics": self._aggregate_metrics(group)
            }
            rollups.append(rollup)
        
        # Group by (run_id, view)
        run_view_groups = df[df['view'].notna()].groupby(['run_id', 'view'])
        for (run_id, view), group in run_view_groups:
            rollup = {
                "rollup_type": "per_run_view",
                "run_id": run_id,
                "view": view,
                "symbol": None,
                "target": None,
                "universe_id": None,
                "n_metrics": len(group),
                "n_targets": group['target'].nunique() if group['target'].notna().any() else 0,
                "n_symbols": group['symbol'].nunique() if group['symbol'].notna().any() else 0,
                "metrics": self._aggregate_metrics(group)
            }
            rollups.append(rollup)
        
        return rollups
    
    def _aggregate_metrics(self, group: pd.DataFrame) -> Dict[str, Any]:
        """Aggregate metrics from a group."""
        metrics = {}
        
        for metric_name in group['metric_name'].unique():
            metric_group = group[group['metric_name'] == metric_name]
            values = metric_group['metric_value'].values
            
            metrics[metric_name] = {
                "count": len(values),
                "mean": float(np.mean(values)) if len(values) > 0 else None,
                "std": float(np.std(values)) if len(values) > 1 else None,
                "min": float(np.min(values)) if len(values) > 0 else None,
                "max": float(np.max(values)) if len(values) > 0 else None,
                "median": float(np.median(values)) if len(values) > 0 else None,
                "unit": metric_group['metric_unit'].iloc[0] if metric_group['metric_unit'].notna().any() else None
            }
        
        return metrics
    
    def _generate_summary(self, df_facts: pd.DataFrame) -> None:
        """Generate human-readable summary JSON."""
        summary = {
            "run_id": self.current_run_id,
            "timestamp": datetime.now().isoformat(),
            "n_facts": len(df_facts),
            "levels": {
                level: len(df_facts[df_facts['level'] == level])
                for level in df_facts['level'].unique()
            },
            "views": list(df_facts['view'].dropna().unique()),
            "targets": list(df_facts['target'].dropna().unique()),
            "symbols": list(df_facts['symbol'].dropna().unique()),
            "n_metrics": df_facts['metric_name'].nunique(),
            "metric_names": sorted(df_facts['metric_name'].unique().tolist())
        }
        
        summary_file = self.telemetry_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Telemetry: Saved summary to {summary_file}")
    
    def compute_drift(
        self,
        current_run_id: str,
        baseline_run_id: Optional[str] = None,
        baseline_mode: str = "previous_run"  # "previous_run" | "rolling_window" | "last_good_run"
    ) -> Dict[str, Any]:
        """
        Compute drift metrics by comparing current run to baseline.
        
        Args:
            current_run_id: Current run identifier
            baseline_run_id: Specific baseline run (optional, overrides baseline_mode)
            baseline_mode: How to select baseline ("previous_run" | "rolling_window" | "last_good_run")
        
        Returns:
            Dict with drift metrics per level
        """
        if not self.enabled:
            return {}
        
        # Load facts table
        facts_file = self.telemetry_dir / "facts.parquet"
        if not facts_file.exists():
            logger.warning(f"Facts table not found: {facts_file}")
            return {}
        
        df_facts = pd.read_parquet(facts_file)
        
        # Get baseline run_id
        if baseline_run_id is None:
            baseline_run_id = self._get_baseline_run_id(df_facts, current_run_id, baseline_mode)
        
        if baseline_run_id is None:
            logger.warning(f"No baseline found for run {current_run_id}")
            return {}
        
        # Filter to current and baseline runs
        df_current = df_facts[df_facts['run_id'] == current_run_id].copy()
        df_baseline = df_facts[df_facts['run_id'] == baseline_run_id].copy()
        
        if len(df_current) == 0 or len(df_baseline) == 0:
            logger.warning(f"Insufficient data for drift comparison: current={len(df_current)}, baseline={len(df_baseline)}")
            return {}
        
        # Compute drift at each level
        drift_results = {
            "current_run_id": current_run_id,
            "baseline_run_id": baseline_run_id,
            "baseline_mode": baseline_mode,
            "timestamp": datetime.now().isoformat(),
            "levels": {}
        }
        
        # Per symbol drift
        if self.rollups.get("per_symbol", False):
            drift_results["levels"]["symbol"] = self._compute_level_drift(
                df_current, df_baseline, groupby=['view', 'symbol']
            )
        
        # Per cross-sectional drift
        if self.rollups.get("per_cross_sectional", False):
            drift_results["levels"]["cross_sectional"] = self._compute_level_drift(
                df_current, df_baseline, groupby=['view', 'universe_id']
            )
        
        # Per target drift
        if self.rollups.get("per_target", False):
            drift_results["levels"]["target"] = self._compute_level_drift(
                df_current, df_baseline, groupby=['target', 'view']
            )
        
        # Per run drift
        if self.rollups.get("per_run", False):
            drift_results["levels"]["run"] = self._compute_level_drift(
                df_current, df_baseline, groupby=[]
            )
        
        # Save drift results
        drift_file = self.telemetry_dir / f"drift_{current_run_id}.json"
        with open(drift_file, 'w') as f:
            json.dump(drift_results, f, indent=2)
        
        logger.info(f"✅ Telemetry: Computed drift for {current_run_id} vs {baseline_run_id}")
        
        return drift_results
    
    def _get_baseline_run_id(
        self,
        df_facts: pd.DataFrame,
        current_run_id: str,
        baseline_mode: str
    ) -> Optional[str]:
        """Get baseline run_id based on mode."""
        available_runs = sorted(df_facts['run_id'].unique(), reverse=True)
        
        if baseline_mode == "previous_run":
            # Get run immediately before current
            try:
                idx = available_runs.index(current_run_id)
                if idx + 1 < len(available_runs):
                    return available_runs[idx + 1]
            except ValueError:
                pass
        
        elif baseline_mode == "rolling_window":
            # Get k-th previous run
            k = self.baselines.get("rolling_window_k", 10)
            try:
                idx = available_runs.index(current_run_id)
                if idx + k < len(available_runs):
                    return available_runs[idx + k]
            except ValueError:
                pass
        
        elif baseline_mode == "last_good_run":
            # Find last run marked as "good" (would need status field)
            # For now, fall back to previous_run
            return self._get_baseline_run_id(df_facts, current_run_id, "previous_run")
        
        return None
    
    def _compute_level_drift(
        self,
        df_current: pd.DataFrame,
        df_baseline: pd.DataFrame,
        groupby: List[str]
    ) -> Dict[str, Any]:
        """Compute drift metrics for a specific level."""
        drift_metrics = []
        
        # Group current and baseline
        if groupby:
            current_groups = df_current.groupby(groupby)
            baseline_groups = df_baseline.groupby(groupby)
        else:
            # Single group for run-level
            current_groups = {tuple(): df_current}
            baseline_groups = {tuple(): df_baseline}
        
        for group_key in set(list(current_groups.keys()) + list(baseline_groups.keys())):
            df_curr_group = current_groups.get(group_key, pd.DataFrame())
            df_base_group = baseline_groups.get(group_key, pd.DataFrame())
            
            if len(df_curr_group) == 0 or len(df_base_group) == 0:
                continue
            
            # Compute drift for each metric
            for metric_name in set(df_curr_group['metric_name'].unique()) & set(df_base_group['metric_name'].unique()):
                curr_values = df_curr_group[df_curr_group['metric_name'] == metric_name]['metric_value'].values
                base_values = df_base_group[df_base_group['metric_name'] == metric_name]['metric_value'].values
                
                if len(curr_values) == 0 or len(base_values) == 0:
                    continue
                
                # Simple drift metrics (can be extended with PSI/KS)
                curr_mean = float(np.mean(curr_values))
                base_mean = float(np.mean(base_values))
                delta = curr_mean - base_mean
                rel_delta = delta / base_mean if base_mean != 0 else None
                
                drift_metrics.append({
                    "group_key": dict(zip(groupby, group_key)) if groupby else "run",
                    "metric_name": metric_name,
                    "current_mean": curr_mean,
                    "baseline_mean": base_mean,
                    "delta": delta,
                    "rel_delta": rel_delta,
                    "status": self._classify_drift(delta, rel_delta, metric_name)
                })
        
        return {
            "n_comparisons": len(drift_metrics),
            "metrics": drift_metrics
        }
    
    def _classify_drift(
        self,
        delta: float,
        rel_delta: Optional[float],
        metric_name: str
    ) -> str:
        """Classify drift status based on thresholds."""
        # Get thresholds from config
        psi_threshold = self.drift.get("psi_threshold", 0.2)
        ks_threshold = self.drift.get("ks_threshold", 0.1)
        
        # Simple classification (can be extended)
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


def load_telemetry_config() -> Dict[str, Any]:
    """Load telemetry configuration from safety.yaml."""
    try:
        from CONFIG.config_loader import get_cfg
        return {
            "enabled": get_cfg("safety.telemetry.enabled", default=True, config_name="safety_config"),
            "levels": {
                "run": get_cfg("safety.telemetry.levels.run", default=True, config_name="safety_config"),
                "view": get_cfg("safety.telemetry.levels.view", default=True, config_name="safety_config"),
                "target": get_cfg("safety.telemetry.levels.target", default=True, config_name="safety_config"),
                "symbol": get_cfg("safety.telemetry.levels.symbol", default=True, config_name="safety_config"),
                "target_symbol": get_cfg("safety.telemetry.levels.target_symbol", default=False, config_name="safety_config")
            },
            "baselines": {
                "previous_run": get_cfg("safety.telemetry.baselines.previous_run", default=True, config_name="safety_config"),
                "rolling_window_k": get_cfg("safety.telemetry.baselines.rolling_window_k", default=10, config_name="safety_config"),
                "last_good_run": get_cfg("safety.telemetry.baselines.last_good_run", default=True, config_name="safety_config")
            },
            "drift": {
                "feature_set_mode": get_cfg("safety.telemetry.drift.feature_set_mode", default="topk_importance", config_name="safety_config"),
                "topk": get_cfg("safety.telemetry.drift.topk", default=50, config_name="safety_config"),
                "psi_threshold": get_cfg("safety.telemetry.drift.psi_threshold", default=0.2, config_name="safety_config"),
                "ks_threshold": get_cfg("safety.telemetry.drift.ks_threshold", default=0.1, config_name="safety_config")
            },
            "rollups": {
                "per_symbol": get_cfg("safety.telemetry.rollups.per_symbol", default=True, config_name="safety_config"),
                "per_cross_sectional": get_cfg("safety.telemetry.rollups.per_cross_sectional", default=True, config_name="safety_config"),
                "per_target": get_cfg("safety.telemetry.rollups.per_target", default=True, config_name="safety_config"),
                "per_run": get_cfg("safety.telemetry.rollups.per_run", default=True, config_name="safety_config")
            }
        }
    except Exception as e:
        logger.warning(f"Failed to load telemetry config: {e}, using defaults")
        return {
            "enabled": True,
            "levels": {"run": True, "view": True, "target": True, "symbol": True, "target_symbol": False},
            "baselines": {"previous_run": True, "rolling_window_k": 10, "last_good_run": True},
            "drift": {"feature_set_mode": "topk_importance", "topk": 50, "psi_threshold": 0.2, "ks_threshold": 0.1},
            "rollups": {"per_symbol": True, "per_cross_sectional": True, "per_target": True, "per_run": True}
        }
