"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

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
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChangeSeverity(str, Enum):
    """Severity levels for changes."""
    CRITICAL = "critical"  # Hard invariants (splits, targets, leakage)
    MAJOR = "major"  # Important but not breaking (hyperparams, versions)
    MINOR = "minor"  # Soft changes (metrics, minor config)
    NONE = "none"  # No meaningful change


class ComparabilityStatus(str, Enum):
    """Comparability status for runs."""
    COMPARABLE = "comparable"  # Same comparison group, can diff
    INCOMPARABLE = "incomparable"  # Different groups, don't diff
    PARTIAL = "partial"  # Some overlap, diff with warnings


@dataclass
class ComparisonGroup:
    """Defines what makes runs comparable."""
    experiment_id: Optional[str] = None
    dataset_signature: Optional[str] = None  # Hash of universe + time rules
    task_signature: Optional[str] = None  # Hash of target + horizon + objective
    routing_signature: Optional[str] = None  # Hash of routing config
    
    def to_key(self) -> str:
        """Generate comparison group key."""
        parts = []
        if self.experiment_id:
            parts.append(f"exp={self.experiment_id}")
        if self.dataset_signature:
            parts.append(f"data={self.dataset_signature[:8]}")
        if self.task_signature:
            parts.append(f"task={self.task_signature[:8]}")
        if self.routing_signature:
            parts.append(f"route={self.routing_signature[:8]}")
        return "|".join(parts) if parts else "default"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedSnapshot:
    """Normalized snapshot for diffing (SST-compliant)."""
    # Core identifiers
    run_id: str
    timestamp: str
    stage: str  # TARGET_RANKING, FEATURE_SELECTION, TRAINING
    view: Optional[str] = None  # CROSS_SECTIONAL, SYMBOL_SPECIFIC
    target: Optional[str] = None
    symbol: Optional[str] = None
    
    # Fingerprints (for change detection)
    config_fingerprint: Optional[str] = None
    data_fingerprint: Optional[str] = None
    feature_fingerprint: Optional[str] = None
    target_fingerprint: Optional[str] = None
    
    # Inputs (what was fed to the run)
    inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Process (what happened during execution)
    process: Dict[str, Any] = field(default_factory=dict)
    
    # Outputs (what was produced)
    outputs: Dict[str, Any] = field(default_factory=dict)
    
    # Comparability
    comparison_group: Optional[ComparisonGroup] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        d = asdict(self)
        if self.comparison_group:
            d['comparison_group'] = self.comparison_group.to_dict()
        return d
    
    def to_hash(self) -> str:
        """Generate hash of normalized snapshot (for deduplication)."""
        # Hash only the diffable parts (exclude run_id, timestamp)
        hashable = {
            'stage': self.stage,
            'view': self.view,
            'target': self.target,
            'symbol': self.symbol,
            'config_fingerprint': self.config_fingerprint,
            'data_fingerprint': self.data_fingerprint,
            'feature_fingerprint': self.feature_fingerprint,
            'target_fingerprint': self.target_fingerprint,
            'inputs': self._normalize_for_hash(self.inputs),
            'process': self._normalize_for_hash(self.process),
            'outputs': self._normalize_for_hash(self.outputs)
        }
        json_str = json.dumps(hashable, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def _normalize_for_hash(obj: Any) -> Any:
        """Normalize object for hashing (sort, round floats, etc.)."""
        if isinstance(obj, dict):
            return {k: NormalizedSnapshot._normalize_for_hash(v) for k, v in sorted(obj.items())}
        elif isinstance(obj, (list, tuple)):
            return [NormalizedSnapshot._normalize_for_hash(v) for v in sorted(obj) if v is not None]
        elif isinstance(obj, float):
            # Round to 6 decimal places for stability
            return round(obj, 6) if not np.isnan(obj) and not np.isinf(obj) else None
        elif isinstance(obj, (int, str, bool, type(None))):
            return obj
        else:
            return str(obj)


@dataclass
class DiffResult:
    """Result of diffing two snapshots."""
    prev_run_id: str
    current_run_id: str
    comparable: bool
    comparability_reason: Optional[str] = None
    
    # Change detection
    changed_keys: List[str] = field(default_factory=list)  # Canonical paths
    severity: ChangeSeverity = ChangeSeverity.NONE
    
    # Summary
    summary: Dict[str, Any] = field(default_factory=dict)
    
    # Patch operations (JSON-Patch style)
    patch: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metric deltas
    metric_deltas: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['severity'] = self.severity.value
        return d


@dataclass
class BaselineState:
    """Baseline state for a comparison group."""
    comparison_group_key: str
    baseline_run_id: str
    baseline_timestamp: str
    baseline_metrics: Dict[str, float]
    established_at: str
    update_count: int = 0
    regression_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        
        # Find RESULTS directory (walk up if we're in a subdirectory)
        results_dir = self.output_dir
        if results_dir.name != "RESULTS":
            # Walk up to find RESULTS directory
            temp_dir = results_dir
            for _ in range(10):  # Limit depth
                if temp_dir.name == "RESULTS":
                    results_dir = temp_dir
                    break
                if not temp_dir.parent.exists():
                    break
                temp_dir = temp_dir.parent
        
        # Global telemetry directory for index files (shared across all runs)
        self.telemetry_dir = results_dir / "REPRODUCIBILITY" / "TELEMETRY"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        
        # Index files (global, not per-run)
        self.snapshot_index = self.telemetry_dir / "snapshot_index.json"
        self.baseline_index = self.telemetry_dir / "baseline_index.json"
        
        # Load existing indices
        self._snapshots: Dict[str, NormalizedSnapshot] = {}
        self._baselines: Dict[str, BaselineState] = {}
        self._load_indices()
    
    def _load_indices(self):
        """Load snapshot and baseline indices."""
        if self.snapshot_index.exists():
            try:
                with open(self.snapshot_index) as f:
                    data = json.load(f)
                    for run_id, snap_data in data.items():
                        self._snapshots[run_id] = self._deserialize_snapshot(snap_data)
            except Exception as e:
                logger.warning(f"Failed to load snapshot index: {e}")
        
        if self.baseline_index.exists():
            try:
                with open(self.baseline_index) as f:
                    data = json.load(f)
                    for key, baseline_data in data.items():
                        self._baselines[key] = BaselineState(**baseline_data)
            except Exception as e:
                logger.warning(f"Failed to load baseline index: {e}")
    
    def _save_indices(self):
        """Save snapshot and baseline indices."""
        try:
            snapshot_data = {
                run_id: snap.to_dict() for run_id, snap in self._snapshots.items()
            }
            with open(self.snapshot_index, 'w') as f:
                json.dump(snapshot_data, f, indent=2, default=str)
            
            baseline_data = {
                key: baseline.to_dict() for key, baseline in self._baselines.items()
            }
            with open(self.baseline_index, 'w') as f:
                json.dump(baseline_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save indices: {e}")
    
    def _deserialize_snapshot(self, data: Dict[str, Any]) -> NormalizedSnapshot:
        """Deserialize snapshot from dict."""
        comp_group = None
        if 'comparison_group' in data:
            comp_group = ComparisonGroup(**data['comparison_group'])
        
        return NormalizedSnapshot(
            run_id=data['run_id'],
            timestamp=data['timestamp'],
            stage=data['stage'],
            view=data.get('view'),
            target=data.get('target'),
            symbol=data.get('symbol'),
            config_fingerprint=data.get('config_fingerprint'),
            data_fingerprint=data.get('data_fingerprint'),
            feature_fingerprint=data.get('feature_fingerprint'),
            target_fingerprint=data.get('target_fingerprint'),
            inputs=data.get('inputs', {}),
            process=data.get('process', {}),
            outputs=data.get('outputs', {}),
            comparison_group=comp_group
        )
    
    def normalize_snapshot(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_metadata: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> NormalizedSnapshot:
        """
        Create normalized snapshot from run data.
        
        Args:
            stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, TRAINING)
            run_data: Run data dict (from reproducibility tracker)
            cohort_metadata: Cohort metadata
            additional_data: Additional data dict
        
        Returns:
            NormalizedSnapshot
        """
        # Extract core identifiers
        run_id = run_data.get('run_id') or run_data.get('timestamp', datetime.now().isoformat())
        timestamp = run_data.get('timestamp', datetime.now().isoformat())
        view = additional_data.get('view') if additional_data else None
        target = additional_data.get('target') if additional_data else None
        symbol = additional_data.get('symbol') if additional_data else None
        
        # Build fingerprints
        config_fp = self._compute_config_fingerprint(run_data, additional_data)
        data_fp = self._compute_data_fingerprint(cohort_metadata, additional_data)
        feature_fp = self._compute_feature_fingerprint(run_data, additional_data)
        target_fp = self._compute_target_fingerprint(run_data, additional_data)
        
        # Build comparison group
        comparison_group = self._build_comparison_group(
            stage, run_data, cohort_metadata, additional_data,
            config_fp, data_fp, task_fp=target_fp
        )
        
        # Normalize inputs
        inputs = self._normalize_inputs(run_data, cohort_metadata, additional_data)
        
        # Normalize process
        process = self._normalize_process(run_data, additional_data)
        
        # Normalize outputs
        outputs = self._normalize_outputs(run_data, additional_data)
        
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
            inputs=inputs,
            process=process,
            outputs=outputs,
            comparison_group=comparison_group
        )
    
    def _compute_config_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute config fingerprint."""
        config_parts = []
        
        # Extract config-relevant fields
        if additional_data:
            for key in ['strategy', 'model_family', 'n_features', 'min_cs', 'max_cs_samples']:
                if key in additional_data:
                    config_parts.append(f"{key}={additional_data[key]}")
        
        if run_data.get('additional_data'):
            for key in ['strategy', 'model_family']:
                if key in run_data['additional_data']:
                    config_parts.append(f"{key}={run_data['additional_data'][key]}")
        
        if config_parts:
            config_str = "|".join(sorted(config_parts))
            return hashlib.sha256(config_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_data_fingerprint(
        self,
        cohort_metadata: Optional[Dict[str, Any]],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute data fingerprint."""
        data_parts = []
        
        if cohort_metadata:
            # Extract data-relevant fields
            for key in ['n_symbols', 'date_range_start', 'date_range_end', 'min_cs', 'max_cs_samples']:
                if key in cohort_metadata:
                    val = cohort_metadata[key]
                    if val is not None:
                        data_parts.append(f"{key}={val}")
        
        if additional_data:
            for key in ['n_symbols', 'date_range']:
                if key in additional_data:
                    val = additional_data[key]
                    if val is not None:
                        data_parts.append(f"{key}={val}")
        
        if data_parts:
            data_str = "|".join(sorted(data_parts))
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return None
    
    def _compute_feature_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute feature fingerprint."""
        features = None
        
        if additional_data and 'n_features' in additional_data:
            # For now, use count (could hash actual feature list if available)
            features = f"count={additional_data['n_features']}"
        
        if run_data.get('additional_data') and 'n_features' in run_data['additional_data']:
            features = f"count={run_data['additional_data']['n_features']}"
        
        if features:
            return hashlib.sha256(features.encode()).hexdigest()[:16]
        return None
    
    def _compute_target_fingerprint(
        self,
        run_data: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        """Compute target fingerprint."""
        target = None
        
        if additional_data and 'target' in additional_data:
            target = additional_data['target']
        elif run_data.get('item_name'):
            # Extract target from item_name (format: "target:family" or "target:symbol:family")
            parts = run_data['item_name'].split(':')
            target = parts[0] if parts else None
        
        if target:
            return hashlib.sha256(target.encode()).hexdigest()[:16]
        return None
    
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
        """Build comparison group."""
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
        
        return ComparisonGroup(
            experiment_id=experiment_id,
            dataset_signature=data_fp,
            task_signature=task_fp,
            routing_signature=routing_signature
        )
    
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
                'date_range_start': cohort_metadata.get('date_range_start'),
                'date_range_end': cohort_metadata.get('date_range_end'),
                'n_samples': cohort_metadata.get('n_samples')
            }
        
        # Target provenance
        target = None
        if additional_data and 'target' in additional_data:
            target = additional_data['target']
        elif run_data.get('item_name'):
            parts = run_data['item_name'].split(':')
            target = parts[0] if parts else None
        
        if target:
            inputs['target'] = {
                'target_name': target,
                'view': additional_data.get('view') if additional_data else None
            }
        
        # Feature set provenance
        if additional_data and 'n_features' in additional_data:
            inputs['features'] = {
                'n_features': additional_data['n_features'],
                'feature_fingerprint': self._compute_feature_fingerprint(run_data, additional_data)
            }
        
        return inputs
    
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
        additional_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Normalize outputs section."""
        outputs = {}
        
        # Performance metrics
        if run_data.get('metrics'):
            metrics = run_data['metrics']
            outputs['metrics'] = {
                'mean_score': metrics.get('mean_score'),
                'std_score': metrics.get('std_score'),
                'composite_score': metrics.get('composite_score')
            }
        
        # Stability metrics (if available)
        if run_data.get('additional_data'):
            if 'stability' in run_data['additional_data']:
                outputs['stability'] = run_data['additional_data']['stability']
        
        return outputs
    
    def save_snapshot(
        self,
        snapshot: NormalizedSnapshot,
        cohort_dir: Path
    ) -> None:
        """
        Save normalized snapshot to cohort directory.
        
        Args:
            snapshot: Normalized snapshot
            cohort_dir: Cohort directory (where metadata.json lives)
        """
        cohort_dir = Path(cohort_dir)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # Save full snapshot
        snapshot_file = cohort_dir / "snapshot.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)
        
        # Update index
        self._snapshots[snapshot.run_id] = snapshot
        self._save_indices()
        
        logger.debug(f"✅ Saved snapshot to {snapshot_file}")
    
    def compute_diff(
        self,
        current_snapshot: NormalizedSnapshot,
        prev_snapshot: NormalizedSnapshot
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
            return DiffResult(
                prev_run_id=prev_snapshot.run_id,
                current_run_id=current_snapshot.run_id,
                comparable=False,
                comparability_reason=reason
            )
        
        # Compute changes
        changed_keys = []
        patch = []
        severity = ChangeSeverity.NONE
        
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
        metric_deltas = self._compute_metric_deltas(
            prev_snapshot.outputs,
            current_snapshot.outputs
        )
        
        # Determine severity
        severity = self._determine_severity(changed_keys, input_changes, process_changes)
        
        # Build summary
        summary = {
            'total_changes': len(changed_keys),
            'input_changes': len(input_changes['keys']),
            'process_changes': len(process_changes['keys']),
            'output_changes': len(output_changes['keys']),
            'metric_deltas_count': len(metric_deltas)
        }
        
        return DiffResult(
            prev_run_id=prev_snapshot.run_id,
            current_run_id=current_snapshot.run_id,
            comparable=True,
            changed_keys=changed_keys,
            severity=severity,
            summary=summary,
            patch=patch,
            metric_deltas=metric_deltas
        )
    
    def _check_comparability(
        self,
        current: NormalizedSnapshot,
        prev: NormalizedSnapshot
    ) -> Tuple[bool, Optional[str]]:
        """Check if two snapshots are comparable."""
        # Must be same stage
        if current.stage != prev.stage:
            return False, f"Different stages: {current.stage} vs {prev.stage}"
        
        # Must be same view
        if current.view != prev.view:
            return False, f"Different views: {current.view} vs {prev.view}"
        
        # Must be same target (if specified)
        if current.target and prev.target and current.target != prev.target:
            return False, f"Different targets: {current.target} vs {prev.target}"
        
        # Check comparison groups
        if current.comparison_group and prev.comparison_group:
            cg_curr = current.comparison_group.to_key()
            cg_prev = prev.comparison_group.to_key()
            if cg_curr != cg_prev:
                return False, f"Different comparison groups: {cg_curr} vs {cg_prev}"
        
        return True, None
    
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
    
    def _compute_metric_deltas(
        self,
        prev_outputs: Dict[str, Any],
        current_outputs: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metric deltas."""
        deltas = {}
        
        prev_metrics = prev_outputs.get('metrics', {})
        current_metrics = current_outputs.get('metrics', {})
        
        for key in set(prev_metrics.keys()) | set(current_metrics.keys()):
            prev_val = prev_metrics.get(key)
            curr_val = current_metrics.get(key)
            
            if prev_val is not None and curr_val is not None:
                try:
                    prev_float = float(prev_val)
                    curr_float = float(curr_val)
                    
                    delta_abs = curr_float - prev_float
                    delta_pct = (delta_abs / abs(prev_float) * 100) if prev_float != 0 else 0.0
                    
                    deltas[key] = {
                        'absolute': round(delta_abs, 6),
                        'percent': round(delta_pct, 2),
                        'previous': prev_float,
                        'current': curr_float
                    }
                except (ValueError, TypeError):
                    pass
        
        return deltas
    
    def _determine_severity(
        self,
        changed_keys: List[str],
        input_changes: Dict,
        process_changes: Dict
    ) -> ChangeSeverity:
        """Determine severity of changes."""
        # Critical: hard invariants
        critical_paths = [
            'inputs.data', 'inputs.target', 'inputs.features.feature_fingerprint',
            'process.split', 'process.leakage'
        ]
        
        for key in changed_keys:
            for critical in critical_paths:
                if key.startswith(critical):
                    return ChangeSeverity.CRITICAL
        
        # Major: important config
        major_paths = [
            'inputs.config', 'process.training', 'process.environment'
        ]
        
        for key in changed_keys:
            for major in major_paths:
                if key.startswith(major):
                    return ChangeSeverity.MAJOR
        
        # Minor: metrics only
        if all(key.startswith('outputs.metrics') for key in changed_keys):
            return ChangeSeverity.MINOR
        
        # Default to major if mixed
        if changed_keys:
            return ChangeSeverity.MAJOR
        
        return ChangeSeverity.NONE
    
    def find_previous_comparable(
        self,
        snapshot: NormalizedSnapshot
    ) -> Optional[NormalizedSnapshot]:
        """Find previous comparable snapshot."""
        if not snapshot.comparison_group:
            return None
        
        group_key = snapshot.comparison_group.to_key()
        
        # Find all snapshots in same comparison group
        candidates = []
        for run_id, snap in self._snapshots.items():
            if run_id == snapshot.run_id:
                continue
            
            comparable, _ = self._check_comparability(snapshot, snap)
            if comparable:
                candidates.append((snap.timestamp, snap))
        
        if not candidates:
            return None
        
        # Return most recent
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    
    def get_or_establish_baseline(
        self,
        snapshot: NormalizedSnapshot,
        metrics: Dict[str, float]
    ) -> Tuple[Optional[BaselineState], bool]:
        """
        Get or establish baseline for comparison group.
        
        Returns:
            (BaselineState or None, is_new_baseline)
        """
        if not snapshot.comparison_group:
            return None, False
        
        group_key = snapshot.comparison_group.to_key()
        
        # Check if baseline exists
        if group_key in self._baselines:
            return self._baselines[group_key], False
        
        # Count comparable runs
        comparable_runs = [
            snap for snap in self._snapshots.values()
            if snap.comparison_group and snap.comparison_group.to_key() == group_key
        ]
        
        if len(comparable_runs) < self.min_runs_for_baseline:
            return None, False
        
        # Establish baseline (use best metric run)
        best_run = None
        best_score = None
        
        for snap in comparable_runs:
            if snap.outputs.get('metrics', {}).get('mean_score'):
                score = snap.outputs['metrics']['mean_score']
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
            self._baselines[group_key] = baseline
            self._save_indices()
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
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # Save prev diff
        prev_diff_file = cohort_dir / "diff_prev.json"
        with open(prev_diff_file, 'w') as f:
            json.dump(diff.to_dict(), f, indent=2, default=str)
        
        # Save baseline diff if available
        if baseline_diff:
            baseline_diff_file = cohort_dir / "diff_baseline.json"
            with open(baseline_diff_file, 'w') as f:
                json.dump(baseline_diff.to_dict(), f, indent=2, default=str)
        
        logger.debug(f"✅ Saved diffs to {cohort_dir}")
    
    def finalize_run(
        self,
        stage: str,
        run_data: Dict[str, Any],
        cohort_dir: Path,
        cohort_metadata: Optional[Dict[str, Any]] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Finalize run: create snapshot, compute diffs, update baseline.
        
        This is the main entry point - call this after each run completes.
        
        Args:
            stage: Pipeline stage
            run_data: Run data from reproducibility tracker
            cohort_dir: Cohort directory
            cohort_metadata: Cohort metadata
            additional_data: Additional data
        """
        # Create normalized snapshot
        snapshot = self.normalize_snapshot(
            stage=stage,
            run_data=run_data,
            cohort_metadata=cohort_metadata,
            additional_data=additional_data
        )
        
        # Save snapshot
        self.save_snapshot(snapshot, cohort_dir)
        
        # Find previous comparable run
        prev_snapshot = self.find_previous_comparable(snapshot)
        
        # Compute diff against previous
        if prev_snapshot:
            diff = self.compute_diff(snapshot, prev_snapshot)
        else:
            diff = DiffResult(
                prev_run_id="none",
                current_run_id=snapshot.run_id,
                comparable=False,
                comparability_reason="No previous comparable run found"
            )
        
        # Get or establish baseline
        metrics = snapshot.outputs.get('metrics', {})
        baseline_state, is_new = self.get_or_establish_baseline(snapshot, metrics)
        
        # Compute diff against baseline
        baseline_diff = None
        if baseline_state:
            # Load baseline snapshot
            baseline_snapshot = self._snapshots.get(baseline_state.baseline_run_id)
            if baseline_snapshot:
                baseline_diff = self.compute_diff(snapshot, baseline_snapshot)
        
        # Save diffs
        self.save_diff(diff, baseline_diff, cohort_dir)
        
        logger.info(f"✅ Telemetry finalized for {stage}:{snapshot.target or 'unknown'}")
        if diff.comparable:
            logger.info(f"   Changes: {len(diff.changed_keys)} keys, severity={diff.severity.value}")
            if diff.metric_deltas:
                for metric, delta in diff.metric_deltas.items():
                    logger.info(f"   {metric}: {delta['absolute']:+.4f} ({delta['percent']:+.2f}%)")

