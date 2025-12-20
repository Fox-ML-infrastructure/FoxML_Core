"""
Reproducibility Tracking Module

Tracks and compares run results across pipeline stages to verify reproducible behavior.
Supports target ranking, feature selection, and other pipeline stages.

Usage:
    from TRAINING.orchestration.utils.reproducibility_tracker import ReproducibilityTracker
    
    tracker = ReproducibilityTracker(output_dir=Path("results"))
    tracker.log_comparison(
        stage="target_ranking",
        item_name="y_will_swing_low_15m_0.05",
        metrics={
            "mean_score": 0.751,
            "std_score": 0.029,
            "mean_importance": 0.23,
            "composite_score": 0.764,
            "metric_name": "ROC-AUC"
        }
    )
"""

import json
import logging
import hashlib
import traceback
import os
import sys
import platform
import socket
import fcntl
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from enum import Enum
import math
import pandas as pd

# Import RunContext and AuditEnforcer for automated audit-grade tracking
try:
    from TRAINING.orchestration.utils.run_context import RunContext
    from TRAINING.common.utils.audit_enforcer import AuditEnforcer, AuditMode
    _AUDIT_AVAILABLE = True
except ImportError:
    _AUDIT_AVAILABLE = False
    RunContext = None
    AuditEnforcer = None
    AuditMode = None

# Use root logger to ensure messages are visible regardless of calling script's logger setup
logger = logging.getLogger(__name__)
# Ensure this logger propagates to root so messages are visible
logger.propagate = True

# Schema version for reproducibility files
REPRODUCIBILITY_SCHEMA_VERSION = 2  # v2: Tagged unions for ambiguous nulls

# Import from modular components
from TRAINING.orchestration.utils.reproducibility.utils import (
    collect_environment_info,
    compute_comparable_key,
    _get_main_logger,
    make_tagged_scalar,
    make_tagged_not_applicable,
    make_tagged_per_target_feature,
    make_tagged_auto,
    make_tagged_not_computed,
    make_tagged_omitted,
    extract_scalar_from_tagged,
    extract_embargo_minutes,
    extract_folds,
    Stage,
    RouteType,
    TargetRankingView
)
from TRAINING.common.utils.file_utils import write_atomic_json as _write_atomic_json


def _construct_comparison_group_key_from_dict(comparison_group: Dict[str, Any]) -> str:
    """
    Construct comparison_group_key from comparison_group dict, matching ComparisonGroup.to_key() logic.
    
    This ensures consistency between how keys are constructed in reproducibility_tracker and trend_analyzer.
    
    Args:
        comparison_group: Dictionary with comparison group fields (from diff_telemetry snapshot)
        
    Returns:
        Comparison group key string (returns "default" if no parts, matching ComparisonGroup.to_key())
    """
    if not comparison_group:
        return "default"
    
    parts = []
    if comparison_group.get('experiment_id'):
        parts.append(f"exp={comparison_group['experiment_id']}")
    if comparison_group.get('dataset_signature'):
        parts.append(f"data={comparison_group['dataset_signature'][:8]}")
    if comparison_group.get('task_signature'):
        parts.append(f"task={comparison_group['task_signature'][:8]}")
    if comparison_group.get('routing_signature'):
        parts.append(f"route={comparison_group['routing_signature'][:8]}")
    # CRITICAL: Include exact N_effective to ensure only identical sample sizes compare
    if comparison_group.get('n_effective') is not None:
        parts.append(f"n={comparison_group['n_effective']}")
    # CRITICAL: Include model_family (different families = different outcomes)
    if comparison_group.get('model_family'):
        parts.append(f"family={comparison_group['model_family']}")
    # CRITICAL: Include feature signature (different features = different outcomes)
    if comparison_group.get('feature_signature'):
        parts.append(f"features={comparison_group['feature_signature'][:8]}")
    # CRITICAL: Include hyperparameters signature (different HPs = different outcomes)
    if comparison_group.get('hyperparameters_signature'):
        parts.append(f"hps={comparison_group['hyperparameters_signature'][:8]}")
    # CRITICAL: Include train_seed (different seeds = different outcomes)
    if comparison_group.get('train_seed') is not None:
        parts.append(f"seed={comparison_group['train_seed']}")
    # CRITICAL: Include library versions signature (different versions = different outcomes)
    if comparison_group.get('library_versions_signature'):
        parts.append(f"libs={comparison_group['library_versions_signature'][:8]}")
    
    return "|".join(parts) if parts else "default"


# All utility functions are now imported from reproducibility.utils (see imports above)


# extract_folds is already imported from reproducibility.utils above


class ReproducibilityTracker:
    """
    Tracks run results and compares them to previous runs for reproducibility verification.
    
    Uses tolerance bands with STABLE/DRIFTING/DIVERGED classification instead of binary
    SAME/DIFFERENT. Only escalates to warnings for meaningful differences.
    """
    
    def __init__(
        self,
        output_dir: Path,
        log_file_name: str = "reproducibility_log.json",
        max_runs_per_item: int = 10,
        score_tolerance: float = 0.001,  # Legacy: kept for backward compat, but thresholds loaded from config
        importance_tolerance: float = 0.01,  # Legacy: kept for backward compat
        search_previous_runs: bool = False,  # If True, search parent directories for previous runs
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,  # Override config thresholds
        use_z_score: Optional[bool] = None,  # Override config use_z_score
        audit_mode: str = "warn"  # Audit enforcement mode: "off" | "warn" | "strict"
    ):
        """
        Initialize reproducibility tracker.
        
        Args:
            output_dir: Directory where reproducibility logs are stored (module-specific)
            log_file_name: Name of the JSON log file
            max_runs_per_item: Maximum number of runs to keep per item (prevents log bloat)
            score_tolerance: Legacy tolerance (kept for backward compat, but config thresholds used)
            importance_tolerance: Legacy tolerance (kept for backward compat, but config thresholds used)
            search_previous_runs: If True, search parent directories for previous runs from same module
            thresholds: Optional override for config thresholds (dict with 'roc_auc', 'composite', 'importance' keys)
            use_z_score: Optional override for config use_z_score setting
        """
        self.output_dir = Path(output_dir)
        # Store log file in module-specific directory: output_dir/reproducibility_log.json
        # This ensures each module (target_rankings, feature_selections, training_results) has its own log
        self.log_file = self.output_dir / log_file_name
        self.max_runs_per_item = max_runs_per_item
        self.search_previous_runs = search_previous_runs
        
        # Helper: Get base directory for REPRODUCIBILITY (should be at run level, not module level)
        # If output_dir is a module subdirectory, go up one level; otherwise use output_dir itself
        self._repro_base_dir = self._get_repro_base_dir()
        
        # Load thresholds from config using centralized utilities
        from TRAINING.orchestration.utils.reproducibility.config_loader import (
            load_thresholds,
            load_use_z_score,
            load_cohort_aware,
            load_n_ratio_threshold,
            load_cohort_config_keys
        )
        self.thresholds = load_thresholds(thresholds)
        self.use_z_score = load_use_z_score(use_z_score)
        
        # Load cohort-aware settings
        self.cohort_aware = load_cohort_aware()
        self.n_ratio_threshold = load_n_ratio_threshold()
        self.cohort_config_keys = load_cohort_config_keys()
        
        # Initialize audit enforcer
        audit_mode = self._load_audit_mode()
        if _AUDIT_AVAILABLE:
            self.audit_enforcer = AuditEnforcer(mode=audit_mode)
        else:
            self.audit_enforcer = None
            if audit_mode != "off":
                logger.warning("Audit enforcement not available (RunContext/AuditEnforcer not imported), disabling audit")
        
        # Initialize stats tracking
        self.stats_file = self._repro_base_dir / "REPRODUCIBILITY" / "stats.json"
        
        # Initialize metrics writer (if enabled)
        try:
            from TRAINING.common.utils.metrics import MetricsWriter, load_metrics_config
            metrics_config = load_metrics_config()
            if metrics_config.get("enabled", False):
                self.metrics = MetricsWriter(
                    output_dir=self._repro_base_dir,  # Base output dir (run level, not module-specific)
                    enabled=metrics_config.get("enabled", True),
                    baselines=metrics_config.get("baselines"),
                    drift=metrics_config.get("drift")
                )
                logger.info(f"✅ Metrics initialized and enabled (output_dir={self._repro_base_dir})")
            else:
                self.metrics = None
                logger.debug("Metrics is disabled in config")
        except Exception as e:
            logger.warning(f"⚠️  Metrics not available: {e}")
            import traceback
            logger.debug(f"Metrics initialization traceback: {traceback.format_exc()}")
            self.metrics = None
    
    def _get_repro_base_dir(self) -> Path:
        """
        Get the base directory for REPRODUCIBILITY structure.
        
        REPRODUCIBILITY should be at the run level, not the module level.
        If output_dir is a module subdirectory (target_rankings/, feature_selections/, training_results/),
        or inside REPRODUCIBILITY/{STAGE}/... structure, walk up to the run directory.
        
        Returns:
            Path to the run-level directory where REPRODUCIBILITY should be created
        """
        # Module subdirectories that indicate we need to go up one level
        module_subdirs = {"target_rankings", "feature_selections", "training_results"}
        
        # Walk up from output_dir to find the run-level directory
        current_dir = self.output_dir
        
        # If we're inside REPRODUCIBILITY/{STAGE}/... structure, walk up to run level
        # Check if we're in a REPRODUCIBILITY subdirectory
        if "REPRODUCIBILITY" in current_dir.parts:
            # Find the index of REPRODUCIBILITY in the path
            repro_idx = None
            for i, part in enumerate(current_dir.parts):
                if part == "REPRODUCIBILITY":
                    repro_idx = i
                    break
            
            if repro_idx is not None and repro_idx > 0:
                # Go up to the directory before REPRODUCIBILITY (run level)
                return Path(*current_dir.parts[:repro_idx])
        
        # If output_dir is a module subdirectory, go up to run level
        if current_dir.name in module_subdirs:
            return current_dir.parent
        
        # Otherwise, output_dir is already at run level
        return current_dir
    
    # Config loading methods now use centralized utilities
    def _load_thresholds(self, override: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
        """Load reproducibility thresholds from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_thresholds
        return load_thresholds(override)
    
    def _load_use_z_score(self, override: Optional[bool] = None) -> bool:
        """Load use_z_score setting from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_use_z_score
        return load_use_z_score(override)
    
    def _load_audit_mode(self) -> str:
        """Load audit mode from config. Defaults to 'off'."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_audit_mode
        return load_audit_mode()
    
    def _load_cohort_aware(self) -> bool:
        """Load cohort_aware setting from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_cohort_aware
        return load_cohort_aware()
    
    def _load_n_ratio_threshold(self) -> float:
        """Load n_ratio_threshold from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_n_ratio_threshold
        return load_n_ratio_threshold()
    
    def _load_cohort_config_keys(self) -> List[str]:
        """Load cohort_config_keys from config."""
        from TRAINING.orchestration.utils.reproducibility.config_loader import load_cohort_config_keys
        return load_cohort_config_keys()
    
    @staticmethod
    def _compute_sample_size_bin(n_effective: int) -> Dict[str, Any]:
        """
        Compute sample size bin info (same logic as IntelligentTrainer._get_sample_size_bin).
        
        **Boundary Rules (CRITICAL - DO NOT CHANGE WITHOUT VERSIONING):**
        - Boundaries are EXCLUSIVE upper bounds: `bin_min <= N_effective < bin_max`
        - Example: `sample_25k-50k` means `25000 <= N_effective < 50000`
        
        **Binning Scheme Version:** `sample_bin_v1`
        
        Returns:
            Dict with keys: bin_name, bin_min, bin_max, binning_scheme_version
        """
        BINNING_SCHEME_VERSION = "sample_bin_v1"
        
        # Define bins with EXCLUSIVE upper bounds (bin_min <= N < bin_max)
        bins = [
            (0, 5000, "sample_0-5k"),
            (5000, 10000, "sample_5k-10k"),
            (10000, 25000, "sample_10k-25k"),
            (25000, 50000, "sample_25k-50k"),
            (50000, 100000, "sample_50k-100k"),
            (100000, 250000, "sample_100k-250k"),
            (250000, 500000, "sample_250k-500k"),
            (500000, 1000000, "sample_500k-1M"),
            (1000000, float('inf'), "sample_1M+")
        ]
        
        for bin_min, bin_max, bin_name in bins:
            if bin_min <= n_effective < bin_max:
                return {
                    "bin_name": bin_name,
                    "bin_min": bin_min,
                    "bin_max": bin_max if bin_max != float('inf') else None,
                    "binning_scheme_version": BINNING_SCHEME_VERSION
                }
        
        # Fallback (should never reach here)
        return {
            "bin_name": "sample_unknown",
            "bin_min": None,
            "bin_max": None,
            "binning_scheme_version": BINNING_SCHEME_VERSION
        }
    
    def _find_previous_log_files(self) -> List[Path]:
        """Find all previous reproducibility log files in parent directories (for same module)."""
        if not self.search_previous_runs:
            return []
        
        previous_logs = []
        try:
            current_dir = self.output_dir
            module_name = self.output_dir.name
            
            # Search up to 3 levels up for previous runs
            for _ in range(3):
                parent = current_dir.parent
                if not parent or parent == current_dir:
                    break
                
                # Look for timestamped directories (format: *_YYYYMMDD_HHMMSS or similar)
                if parent.exists():
                    try:
                        # Check if parent contains module subdirectories (target_rankings, feature_selections, etc.)
                        for sibling_dir in parent.iterdir():
                            try:
                                if sibling_dir.is_dir() and sibling_dir != self.output_dir:
                                    # Check if this sibling has the same module subdirectory
                                    module_log = sibling_dir / module_name / self.log_file.name
                                    if module_log.exists():
                                        previous_logs.append(module_log)
                            except (PermissionError, OSError) as e:
                                logger.debug(f"Could not access sibling directory {sibling_dir}: {e}")
                                continue
                    except (PermissionError, OSError) as e:
                        logger.debug(f"Could not iterate parent directory {parent} for previous logs: {e}")
                        continue
                
                current_dir = parent
        except Exception as e:
            logger.warning(f"Error searching for previous log files: {e}")
            # Don't fail completely, just return empty list
        
        return previous_logs
    
    def load_previous_run(
        self,
        stage: str,
        item_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load the previous run's summary for a stage/item combination.
        
        Searches current log file first, then previous runs if search_previous_runs=True.
        
        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            item_name: Name of the item (e.g., target name, symbol name)
        
        Returns:
            Dictionary with previous run results, or None if no previous run exists
        """
        key = f"{stage}:{item_name}"
        all_item_runs = []
        
        # First, try current log file
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    all_runs = json.load(f)
                item_runs = all_runs.get(key, [])
                if item_runs:
                    all_item_runs.extend(item_runs)
                    logger.debug(f"Found {len(item_runs)} run(s) in current log: {self.log_file}")
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Could not read current log file {self.log_file}: {e}")
        
        # Then, search previous runs if enabled
        if self.search_previous_runs:
            previous_logs = self._find_previous_log_files()
            for prev_log in previous_logs:
                try:
                    with open(prev_log, 'r') as f:
                        all_runs = json.load(f)
                    item_runs = all_runs.get(key, [])
                    if item_runs:
                        all_item_runs.extend(item_runs)
                        logger.debug(f"Found {len(item_runs)} run(s) in previous log: {prev_log}")
                except (json.JSONDecodeError, IOError) as e:
                    logger.debug(f"Could not read previous log file {prev_log}: {e}")
        
        if not all_item_runs:
            logger.debug(f"No previous runs found for {key}")
            if self.log_file.exists():
                try:
                    with open(self.log_file, 'r') as f:
                        all_runs = json.load(f)
                        logger.debug(f"Available keys in current log: {list(all_runs.keys())[:10]}")
                except:
                    pass
            return None
        
        # Sort by timestamp and return most recent
        all_item_runs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        logger.debug(f"Found {len(all_item_runs)} total previous run(s) for {key}, using most recent")
        return all_item_runs[0]
    
    def save_run(
        self,
        stage: str,
        item_name: str,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save the current run's summary to the reproducibility log.
        
        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            item_name: Name of the item (e.g., target name, symbol name)
            metrics: Dictionary of metrics to track (must include at least mean_score, std_score)
            additional_data: Optional additional data to store with the run
        """
        # Load existing runs
        all_runs = {}
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    all_runs = json.load(f)
            except (json.JSONDecodeError, IOError):
                all_runs = {}
        
        # Create key for this stage/item combination
        key = f"{stage}:{item_name}"
        
        # Initialize entry if needed
        if key not in all_runs:
            all_runs[key] = []
        
        # Create summary entry
        summary = {
            "timestamp": datetime.now().isoformat(),
            "stage": stage,
            "item_name": item_name,
            "reproducibility_mode": "LEGACY",  # Track which mode was used
            **{k: float(v) if isinstance(v, (int, float)) else v 
               for k, v in metrics.items()}
        }
        
        if additional_data:
            summary["additional_data"] = additional_data
        
        # Append to item's run history (keep last N runs)
        all_runs[key].append(summary)
        if len(all_runs[key]) > self.max_runs_per_item:
            all_runs[key] = all_runs[key][-self.max_runs_per_item:]
        
        # Save back to file
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, 'w') as f:
                json.dump(all_runs, f, indent=2)
                f.flush()  # Ensure immediate write
                os.fsync(f.fileno())  # Force write to disk
        except IOError as e:
            logger.warning(f"Could not save reproducibility log: {e}")
    
    def _classify_diff(
        self,
        prev_value: float,
        curr_value: float,
        prev_std: Optional[float],
        metric_type: str  # 'roc_auc', 'composite', or 'importance'
    ) -> Tuple[str, float, float, Optional[float]]:
        """
        Classify difference into STABLE/DRIFTING/DIVERGED tiers.
        
        Args:
            prev_value: Previous run value
            curr_value: Current run value
            prev_std: Previous run standard deviation (for z-score calculation)
            metric_type: Type of metric ('roc_auc', 'composite', 'importance')
        
        Returns:
            Tuple of (classification, abs_diff, rel_diff, z_score)
            classification: 'STABLE', 'DRIFTING', or 'DIVERGED'
        """
        diff = curr_value - prev_value
        abs_diff = abs(diff)
        
        # Calculate relative difference
        rel_diff = (abs_diff / max(abs(prev_value), 1e-8)) * 100 if prev_value != 0 else 0.0
        
        # Calculate z-score if std available and use_z_score enabled
        z_score = None
        if self.use_z_score and prev_std is not None and prev_std > 0:
            # Pooled std: use average of previous and current if available
            # For now, use previous std
            z_score = abs_diff / prev_std
        
        # Get thresholds for this metric type
        thresholds = self.thresholds.get(metric_type, self.thresholds.get('roc_auc'))
        abs_thr = thresholds.get('abs', 0.005)
        rel_thr = thresholds.get('rel', 0.02)
        z_thr = thresholds.get('z_score', 1.0)
        
        # Classification logic: require BOTH effect size AND statistical significance for DIVERGED
        # This prevents flagging tiny, statistically insignificant changes as DIVERGED
        # 
        # STABLE: small change AND not statistically significant
        # DRIFTING: moderate change OR borderline statistical significance
        # DIVERGED: large change AND statistically significant
        
        if z_score is not None:
            # Use z-score for statistical significance, abs/rel for effect size
            # Require BOTH big effect AND statistical significance for DIVERGED
            big_effect = abs_diff >= abs_thr or rel_diff >= rel_thr
            statistically_significant = z_score >= z_thr
            
            # For DIVERGED: need BOTH big effect AND statistical significance
            # Use stricter z_thr (2.0) for DIVERGED to require ~95% confidence
            div_thr = max(z_thr * 2.0, 2.0)  # At least 2.0 for DIVERGED
            is_diverged = big_effect and z_score >= div_thr
            
            if not big_effect and not statistically_significant:
                classification = 'STABLE'
            elif is_diverged:
                classification = 'DIVERGED'
            else:
                classification = 'DRIFTING'
        else:
            # Fallback to abs/rel thresholds (no z-score available)
            # Still require both abs AND rel for DIVERGED
            big_abs = abs_diff >= abs_thr
            big_rel = rel_diff >= rel_thr
            is_diverged = big_abs and big_rel
            
            if not big_abs and not big_rel:
                classification = 'STABLE'
            elif is_diverged:
                classification = 'DIVERGED'
            else:
                classification = 'DRIFTING'
        
        return classification, abs_diff, rel_diff, z_score
    
    def _extract_route_type(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Extract route type from additional_data (for feature_selection stage).
        
        **CRITICAL**: For FEATURE_SELECTION, map view to route_type:
        - view="CROSS_SECTIONAL" → route_type="CROSS_SECTIONAL"
        - view="SYMBOL_SPECIFIC" → route_type="INDIVIDUAL"
        
        This ensures metrics is scoped correctly (features compared per-target, per-view, per-symbol).
        
        Returns:
            "CROSS_SECTIONAL", "INDIVIDUAL", or None
        """
        if not additional_data:
            return None
        
        # Check explicit route_type
        route_type = additional_data.get('route_type')
        if route_type:
            return route_type.upper()
        
        # FIX: For FEATURE_SELECTION, map view to route_type (same as TARGET_RANKING)
        # This ensures proper scoping: features compared per-target, per-view, per-symbol
        view = additional_data.get('view')
        if view:
            if view.upper() == "CROSS_SECTIONAL":
                return "CROSS_SECTIONAL"
            elif view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
                return "INDIVIDUAL"  # SYMBOL_SPECIFIC maps to INDIVIDUAL for FEATURE_SELECTION
        
        # Infer from other fields (fallback)
        if additional_data.get('cross_sectional') or additional_data.get('is_cross_sectional'):
            return "CROSS_SECTIONAL"
        elif additional_data.get('symbol_specific') or additional_data.get('is_symbol_specific'):
            return "INDIVIDUAL"
        
        # Default: assume CROSS_SECTIONAL if not specified
        return "CROSS_SECTIONAL"
    
    def _extract_symbol(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract symbol name from additional_data."""
        if not additional_data:
            return None
        return additional_data.get('symbol')
    
    def _extract_model_family(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Extract model family from additional_data."""
        if not additional_data:
            return None
        return additional_data.get('model_family') or additional_data.get('family')
    
    def _extract_cohort_metadata(
        self,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Extract cohort metadata from metrics and additional_data.
        
        Returns:
            Dict with cohort metadata or None if insufficient data
        """
        if not self.cohort_aware:
            return None
        
        cohort = {}
        
        # Extract N_effective_cs (sample size)
        n_effective = metrics.get('N_effective_cs') or metrics.get('n_samples') or metrics.get('sample_size')
        if n_effective is None and additional_data:
            n_effective = additional_data.get('N_effective_cs') or additional_data.get('n_samples') or additional_data.get('sample_size')
        
        if n_effective is None:
            return None  # Can't form cohort without sample size
        
        cohort['N_effective_cs'] = int(n_effective)
        
        # Extract n_symbols
        n_symbols = metrics.get('n_symbols')
        if n_symbols is None and additional_data:
            n_symbols = additional_data.get('n_symbols')
        cohort['n_symbols'] = int(n_symbols) if n_symbols is not None else 0
        
        # Extract date_range
        date_range = {}
        if additional_data:
            if 'date_range' in additional_data:
                date_range = additional_data['date_range']
            elif 'start_ts' in additional_data or 'end_ts' in additional_data:
                date_range = {
                    'start_ts': additional_data.get('start_ts'),
                    'end_ts': additional_data.get('end_ts')
                }
        cohort['date_range'] = date_range
        
        # Extract config hash components
        cs_config = {}
        if additional_data:
            config_data = additional_data.get('cs_config', {})
            for key in self.cohort_config_keys:
                if key in config_data:
                    cs_config[key] = config_data[key]
                elif key in additional_data:
                    cs_config[key] = additional_data[key]
        cohort['cs_config'] = cs_config
        
        return cohort
    
    def _compute_cohort_id(self, cohort: Dict[str, Any], mode: Optional[str] = None) -> str:
        """
        Compute readable cohort ID from metadata.
        
        Format: {mode}_{date_range}_{universe}_{config}_{version}
        Example: cs_2023Q1_universeA_min_cs3_v1
        """
        # For TARGET_RANKING: mode is view (CROSS_SECTIONAL→"cr", SYMBOL_SPECIFIC→"sy", LOSO→"lo")
        # For FEATURE_SELECTION/TRAINING: mode is route_type (CROSS_SECTIONAL→"cr", INDIVIDUAL→"in")
        mode_prefix = (mode or "cs").lower()[:2]
        
        # Extract date range
        date_start = cohort.get('date_range', {}).get('start_ts', '')
        date_end = cohort.get('date_range', {}).get('end_ts', '')
        
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
        cs_config = cohort.get('cs_config', {})
        universe = cs_config.get('universe_id', 'default')
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
        
        # Add short hash for uniqueness if needed
        # Create deterministic hash for final uniqueness check
        hash_str = "|".join([
            str(cohort.get('N_effective_cs', '')),
            str(cohort.get('n_symbols', '')),
            date_start,
            date_end,
            json.dumps(cs_config, sort_keys=True)
        ])
        short_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:8]
        
        return f"{cohort_id}_{short_hash}"
    
    def _get_cohort_dir(
        self,
        stage: str,
        item_name: str,
        cohort_id: str,
        route_type: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None
    ) -> Path:
        """
        Get directory for a specific cohort following the structured layout.
        
        Structure:
        REPRODUCIBILITY/
          {STAGE}/
            {MODE}/  (for FEATURE_SELECTION, TRAINING)
              {item_name}/
                {symbol}/  (for INDIVIDUAL mode)
                  {model_family}/  (for TRAINING)
                    cohort={cohort_id}/
        
        Args:
            stage: Pipeline stage (e.g., "target_ranking", "feature_selection", "model_training")
            item_name: Item name (e.g., target name)
            cohort_id: Cohort identifier
            route_type: Optional route type ("CROSS_SECTIONAL" or "INDIVIDUAL")
            symbol: Optional symbol name (for INDIVIDUAL mode)
            model_family: Optional model family (for TRAINING stage)
        
        Returns:
            Path to cohort directory
        """
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
        
        # Normalize stage name to uppercase
        stage_upper = stage.upper().replace("MODEL_TRAINING", "TRAINING")
        
        # Build path components
        path_parts = [stage_upper]
        
        # For TARGET_RANKING, add view subdirectory (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO)
        if stage_upper == "TARGET_RANKING":
            # Check if view is provided in additional_data or route_type
            view = None
            if route_type and route_type.upper() in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "LOSO"]:
                view = route_type.upper()
            # If view not in route_type, check if we can infer from symbol presence
            if view is None and symbol:
                view = "SYMBOL_SPECIFIC"  # Default for symbol-specific
            if view is None:
                view = "CROSS_SECTIONAL"  # Default
            path_parts.append(view)
        
        # Add mode subdirectory for FEATURE_SELECTION and TRAINING
        elif stage_upper in ["FEATURE_SELECTION", "TRAINING"]:
            if route_type:
                mode = route_type.upper()
                # FIX: Accept SYMBOL_SPECIFIC as valid mode for FEATURE_SELECTION (maps from view)
                if mode not in ["CROSS_SECTIONAL", "INDIVIDUAL", "SYMBOL_SPECIFIC"]:
                    # If route_type is INDIVIDUAL and we have symbol, treat as SYMBOL_SPECIFIC
                    if mode == "INDIVIDUAL" and symbol:
                        mode = "SYMBOL_SPECIFIC"
                    else:
                        mode = "INDIVIDUAL"
            else:
                mode = "CROSS_SECTIONAL"  # Default
            path_parts.append(mode)
        
        # Add target/item_name
        path_parts.append(item_name)
        
        # Add symbol for SYMBOL_SPECIFIC/LOSO views (TARGET_RANKING) or SYMBOL_SPECIFIC/INDIVIDUAL mode (FEATURE_SELECTION/TRAINING)
        if stage_upper == "TARGET_RANKING":
            if view in ["SYMBOL_SPECIFIC", "LOSO"] and symbol:
                path_parts.append(f"symbol={symbol}")
        elif stage_upper in ["FEATURE_SELECTION", "TRAINING"]:
            # For FEATURE_SELECTION/TRAINING, add symbol if mode is SYMBOL_SPECIFIC or INDIVIDUAL
            if symbol and (route_type and route_type.upper() in ["INDIVIDUAL", "SYMBOL_SPECIFIC"]):
                path_parts.append(f"symbol={symbol}")
        
        # Add model_family for TRAINING
        if stage_upper == "TRAINING" and model_family:
            path_parts.append(f"model_family={model_family}")
        
        # Add cohort directory
        path_parts.append(f"cohort={cohort_id}")
        
        return repro_dir / Path(*path_parts)
    
    def _save_to_cohort(
        self,
        stage: str,
        item_name: str,
        cohort_id: str,
        cohort_metadata: Dict[str, Any],
        run_data: Dict[str, Any],
        route_type: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None,
        additional_data: Optional[Dict[str, Any]] = None,
        trend_metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save run to cohort-specific directory with structured layout.
        
        Creates:
        - metadata.json: Full cohort metadata
        - metrics.json: Run metrics
        - drift.json: Comparison to previous run (if applicable)
        """
        cohort_dir = self._get_cohort_dir(stage, item_name, cohort_id, route_type, symbol, model_family)
        cohort_dir.mkdir(parents=True, exist_ok=True)
        
        # Log where files are being written (INFO level so it's visible)
        main_logger = _get_main_logger()
        try:
            # Try to get a relative path for readability
            repro_base = cohort_dir.parent.parent.parent.parent
            rel_path = cohort_dir.relative_to(repro_base) if repro_base.exists() else cohort_dir
            log_msg = f"📁 Reproducibility: Writing cohort data to {rel_path}"
        except (ValueError, AttributeError):
            log_msg = f"📁 Reproducibility: Writing cohort data to {cohort_dir}"
        
        if main_logger != logger:
            main_logger.info(log_msg)
        else:
            logger.info(log_msg)
        
        # Generate run_id
        run_id = run_data.get('run_id') or run_data.get('timestamp', datetime.now().isoformat())
        run_id_clean = run_id.replace(':', '-').replace('.', '-').replace('T', '_')
        
        # Normalize stage (accept both string and Stage enum)
        if isinstance(stage, Stage):
            stage_normalized = stage.value
        else:
            stage_normalized = stage.upper().replace("MODEL_TRAINING", "TRAINING")
        
        # Normalize route_type (accept both string and RouteType enum)
        # For TARGET_RANKING, use view from additional_data if available
        if stage_normalized == "TARGET_RANKING" and not route_type:
            if additional_data and 'view' in additional_data:
                route_type = additional_data['view']  # Use view as route_type for TARGET_RANKING
        elif route_type and isinstance(route_type, RouteType):
            route_type = route_type.value
        
        # Extract symbols list from cohort_metadata, additional_data
        # Try multiple sources to get the actual symbol list
        symbols_list = None
        if additional_data and 'symbols' in additional_data:
            symbols_list = additional_data['symbols']
        elif cohort_metadata and 'symbols' in cohort_metadata:
            symbols_list = cohort_metadata['symbols']
        elif additional_data and 'symbol_list' in additional_data:
            symbols_list = additional_data['symbol_list']
        
        # For TARGET_RANKING with SYMBOL_SPECIFIC/LOSO view, use symbol from additional_data if available
        if stage_normalized == "TARGET_RANKING" and not symbol:
            if additional_data and 'symbol' in additional_data:
                symbol = additional_data['symbol']  # Override symbol from additional_data
        
        # Normalize symbols: convert to list, remove duplicates, sort for stable diffs
        if symbols_list is not None:
            if isinstance(symbols_list, str):
                # Handle comma-separated string
                symbols_list = [s.strip() for s in symbols_list.split(',')]
            elif not isinstance(symbols_list, (list, tuple)):
                # Try to convert other iterables
                try:
                    symbols_list = list(symbols_list)
                except (TypeError, ValueError):
                    symbols_list = None
        
        # Clean and sort symbols (remove duplicates, sort for stable git diffs)
        # The extractor already provides sorted, deduplicated list, but be defensive
        if symbols_list:
            symbols_list = sorted(set(str(s).strip() for s in symbols_list if s))
            if not symbols_list:  # Empty after cleaning
                symbols_list = None
        
        # Build full metadata with schema version and explicit IDs
        # For TARGET_RANKING, include view metadata
        # Schema v2: Use tagged unions for ambiguous nulls (omit non-applicable fields)
        full_metadata = {
            "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
            "cohort_id": cohort_id,
            "run_id": run_id_clean,
            "stage": stage_normalized,  # Already normalized to uppercase
            "route_type": route_type.upper() if (route_type and isinstance(route_type, str)) else (route_type.value if hasattr(route_type, 'value') else str(route_type).upper() if route_type else None),
            "view": (additional_data.get('view') if additional_data else None) if stage_normalized == "TARGET_RANKING" else None,  # Add view for TARGET_RANKING
            "target_name": item_name,  # Changed from "target" to match finalize_run() expectations
            "n_effective": cohort_metadata.get('N_effective_cs', 0),  # Changed from "N_effective" to match finalize_run() expectations
            "n_symbols": cohort_metadata.get('n_symbols', 0),
            "symbols": symbols_list,  # Sorted, deduplicated list of symbols
            "date_range_start": cohort_metadata.get('date_range', {}).get('start_ts'),  # Changed from "date_start" to match finalize_run() expectations
            "date_range_end": cohort_metadata.get('date_range', {}).get('end_ts'),  # Changed from "date_end" to match finalize_run() expectations
            "universe_id": cohort_metadata.get('cs_config', {}).get('universe_id'),
            "min_cs": cohort_metadata.get('cs_config', {}).get('min_cs'),
            "max_cs_samples": cohort_metadata.get('cs_config', {}).get('max_cs_samples'),
            "leakage_filter_version": cohort_metadata.get('cs_config', {}).get('leakage_filter_version', 'v1'),
            "cs_config_hash": hashlib.sha256(
                json.dumps(cohort_metadata.get('cs_config', {}), sort_keys=True).encode()
            ).hexdigest()[:8],
            "seed": run_data.get('seed') or (additional_data.get('seed') if additional_data else None),
            "git_commit": self._get_git_commit(),
            "created_at": datetime.now().isoformat()
        }
        
        # Schema v2: Omit non-applicable fields instead of null
        # Only include symbol if route_type is INDIVIDUAL or SYMBOL_SPECIFIC
        route_normalized = route_type.upper() if route_type else None
        if symbol and (route_normalized == "INDIVIDUAL" or 
                      (stage_normalized == "TARGET_RANKING" and additional_data and 
                       additional_data.get('view') in ['SYMBOL_SPECIFIC', 'LOSO'])):
            full_metadata["symbol"] = symbol
        # Otherwise omit (cross-sectional doesn't have a single symbol)
        
        # Only include model_family if specified
        if model_family:
            full_metadata["model_family"] = model_family
        # Otherwise omit (not applicable for multi-model or unspecified)
        
        # Add audit-grade fields: data fingerprint and per-symbol stats
        if cohort_metadata.get('data_fingerprint'):
            full_metadata['data_fingerprint'] = cohort_metadata['data_fingerprint']
        
        if cohort_metadata.get('per_symbol_stats'):
            full_metadata['per_symbol_stats'] = cohort_metadata['per_symbol_stats']
        
        # Add CV details from additional_data
        if additional_data:
            cv_details = {}
            
            # CV method and parameters
            cv_enabled = True
            if 'cv_method' in additional_data:
                cv_details['cv_method'] = additional_data['cv_method']
                if additional_data['cv_method'] in ('none', None):
                    cv_enabled = False
            elif 'cv_scheme' in additional_data:
                cv_details['cv_method'] = additional_data['cv_scheme']
                if additional_data['cv_scheme'] in ('none', None):
                    cv_enabled = False
            elif 'cv_skipped' in additional_data and additional_data['cv_skipped']:
                cv_enabled = False
                cv_details['cv_method'] = 'none'
            else:
                cv_details['cv_method'] = 'purged_kfold'  # Default assumption
            
            # Add explicit enabled flag
            cv_details['enabled'] = cv_enabled
            
            # Horizon, purge, embargo
            if 'horizon_minutes' in additional_data:
                cv_details['horizon_minutes'] = additional_data['horizon_minutes']
            if 'purge_minutes' in additional_data:
                cv_details['purge_minutes'] = additional_data['purge_minutes']
            elif 'purge_time' in additional_data:
                # Extract minutes from Timedelta string
                try:
                    purge_str = str(additional_data['purge_time'])
                    if 'days' in purge_str:
                        # Parse Timedelta string
                        import re
                        match = re.search(r'(\d+)\s*days?\s*(\d+):(\d+):(\d+)', purge_str)
                        if match:
                            days, hours, minutes, seconds = map(int, match.groups())
                            cv_details['purge_minutes'] = days * 24 * 60 + hours * 60 + minutes + seconds / 60
                    else:
                        # Try to extract minutes directly
                        match = re.search(r'(\d+)\s*min', purge_str, re.I)
                        if match:
                            cv_details['purge_minutes'] = int(match.group(1))
                except Exception:
                    pass
            
            # Schema v2: embargo_minutes as tagged union
            # FIX: If CV is enabled (cv_method is set), embargo should be explicitly set to 0 if None,
            # not marked as "not_applicable". Only mark as "not_applicable" if CV is actually disabled.
            if 'embargo_minutes' in additional_data:
                embargo_val = additional_data['embargo_minutes']
                if embargo_val is None:
                    # Check if embargo is per-target-feature (has feature_time_meta_map)
                    if 'feature_time_meta_map' in additional_data and additional_data['feature_time_meta_map']:
                        # Per-target-feature: store reference to artifact
                        embargo_map_path = None
                        embargo_map_sha256 = None
                        # Try to find embargo map artifact
                        if 'embargo_map_path' in additional_data:
                            embargo_map_path = additional_data['embargo_map_path']
                        if 'embargo_map_sha256' in additional_data:
                            embargo_map_sha256 = additional_data['embargo_map_sha256']
                        
                        # Compute rollup stats if available
                        rollup = None
                        if embargo_map_path or embargo_map_sha256:
                            # Try to compute rollup from feature_time_meta_map
                            embargo_values = []
                            for feat_meta in additional_data['feature_time_meta_map'].values():
                                if hasattr(feat_meta, 'embargo_minutes'):
                                    embargo_values.append(feat_meta.embargo_minutes)
                            if embargo_values:
                                import numpy as np
                                rollup = {
                                    "min": float(np.min(embargo_values)),
                                    "p50": float(np.median(embargo_values)),
                                    "max": float(np.max(embargo_values)),
                                    "unique_count": len(set(embargo_values))
                                }
                        
                        cv_details['embargo_minutes'] = make_tagged_per_target_feature(
                            ref_path=embargo_map_path,
                            ref_sha256=embargo_map_sha256,
                            rollup=rollup
                        )
                    else:
                        # CRITICAL FIX: If CV method is set (CV is enabled), embargo should be explicitly 0,
                        # not "not_applicable". Only mark as "not_applicable" if CV is actually disabled.
                        cv_method = cv_details.get('cv_method', '')
                        cv_skipped = additional_data.get('cv_skipped', False)
                        if cv_method and cv_method != 'none' and not cv_skipped:
                            # CV is enabled but embargo is None -> explicitly set to 0 (disabled)
                            cv_details['embargo_minutes'] = make_tagged_scalar(0.0)
                            cv_details['embargo_enabled'] = False
                        else:
                            # CV is disabled or not applicable
                            cv_details['embargo_minutes'] = make_tagged_not_applicable(reason="cv_disabled_or_not_applicable")
                else:
                    # Scalar value
                    cv_details['embargo_minutes'] = make_tagged_scalar(embargo_val)
                    # If embargo is explicitly set (non-zero), mark as enabled
                    if embargo_val != 0:
                        cv_details['embargo_enabled'] = True
                    else:
                        cv_details['embargo_enabled'] = False
            
            # Schema v2: folds as tagged union
            if 'cv_folds' in additional_data:
                folds_val = additional_data['cv_folds']
            elif 'n_splits' in additional_data:
                folds_val = additional_data['n_splits']
            else:
                folds_val = None
            
            if folds_val is not None:
                # Check if it was auto-computed
                if 'cv_folds_auto' in additional_data and additional_data.get('cv_folds_auto', False):
                    cv_details['folds'] = make_tagged_auto(value=folds_val)
                else:
                    cv_details['folds'] = make_tagged_scalar(folds_val)
            elif 'cv_skipped' in additional_data and additional_data['cv_skipped']:
                cv_details['folds'] = make_tagged_not_applicable(reason="cv_disabled")
            # Otherwise omit (not computed yet)
            
            # Fold boundaries hash
            if 'fold_boundaries' in additional_data:
                fold_boundaries = additional_data['fold_boundaries']
                try:
                    fold_boundaries_str = json.dumps(fold_boundaries, sort_keys=True)
                    cv_details['fold_boundaries_hash'] = hashlib.sha256(fold_boundaries_str.encode()).hexdigest()[:16]
                    # Also store the actual boundaries (for debugging)
                    cv_details['fold_boundaries'] = fold_boundaries
                except Exception:
                    pass
            elif 'fold_timestamps' in additional_data:
                # Use fold_timestamps to compute hash
                fold_timestamps = additional_data['fold_timestamps']
                try:
                    fold_timestamps_str = json.dumps(fold_timestamps, sort_keys=True)
                    cv_details['fold_boundaries_hash'] = hashlib.sha256(fold_timestamps_str.encode()).hexdigest()[:16]
                    # Also store the timestamps (for debugging)
                    cv_details['fold_timestamps'] = fold_timestamps
                except Exception:
                    pass
            
            # Feature lookback max minutes
            if 'feature_lookback_max_minutes' in additional_data:
                cv_details['feature_lookback_max_minutes'] = additional_data['feature_lookback_max_minutes']
            elif 'max_feature_lookback_minutes' in additional_data:
                cv_details['feature_lookback_max_minutes'] = additional_data['max_feature_lookback_minutes']
            
            # Label definition hash
            if 'label_definition_hash' in additional_data:
                cv_details['label_definition_hash'] = additional_data['label_definition_hash']
            elif 'target_config_hash' in additional_data:
                cv_details['label_definition_hash'] = additional_data['target_config_hash']
            
            # Splitter implementation (if available)
            if 'splitter_impl' in additional_data:
                cv_details['splitter_impl'] = additional_data['splitter_impl']
            elif 'cv_splitter_class' in additional_data:
                cv_details['splitter_impl'] = additional_data['cv_splitter_class']
            
            if cv_details:
                full_metadata['cv_details'] = cv_details
        
        # Add trend metadata (if computed)
        if trend_metadata:
            full_metadata['trend'] = trend_metadata
        
        # Add feature registry hash
        if additional_data and 'feature_registry_hash' in additional_data:
            full_metadata['feature_registry_hash'] = additional_data['feature_registry_hash']
        elif additional_data and 'feature_names' in additional_data:
            # Compute hash from feature names (sorted for stability)
            try:
                feature_names = additional_data['feature_names']
                if isinstance(feature_names, (list, tuple)):
                    feature_names_sorted = sorted([str(f) for f in feature_names])
                    feature_registry_str = "|".join(feature_names_sorted)
                    full_metadata['feature_registry_hash'] = hashlib.sha256(feature_registry_str.encode()).hexdigest()[:16]
            except Exception:
                pass
        
        # Add sample size bin metadata (for directory organization, NOT series identity)
        # Compute from n_effective if not provided, ensuring consistency
        # This allows backward compatibility and binning scheme versioning
        # CRITICAL: Use 'n_effective' (new field name), not 'N_effective' (old field name)
        n_effective = full_metadata.get('n_effective') or full_metadata.get('N_effective')  # Support both for backwards compat
        if n_effective and n_effective > 0:
            # Use provided bin info if available, otherwise compute from N_effective
            if additional_data and 'sample_size_bin' in additional_data:
                full_metadata['sample_size_bin'] = additional_data['sample_size_bin']
            else:
                # Compute bin info using same logic as IntelligentTrainer
                # This ensures consistency even if bin info wasn't passed through
                bin_info = ReproducibilityTracker._compute_sample_size_bin(n_effective)
                if bin_info:
                    full_metadata['sample_size_bin'] = bin_info
        
        # NEW: Add dropped features metadata (if provided)
        if additional_data and 'dropped_features' in additional_data:
            full_metadata['dropped_features'] = additional_data['dropped_features']
        
        # NEW: Add environment information (audit-grade metadata)
        try:
            env_info = collect_environment_info()
            if env_info:
                full_metadata['environment'] = env_info
        except Exception as e:
            logger.debug(f"Failed to collect environment info: {e}")
        
        # NEW: Add data source details (if available)
        data_source_info = {}
        if additional_data:
            if 'data_source' in additional_data:
                data_source_info['source'] = additional_data['data_source']
            if 'dataset_id' in additional_data:
                data_source_info['dataset_id'] = additional_data['dataset_id']
            elif 'dataset_manifest_hash' in additional_data:
                data_source_info['dataset_manifest_hash'] = additional_data['dataset_manifest_hash']
            if 'bar_size' in additional_data:
                data_source_info['bar_size'] = additional_data['bar_size']
            elif 'data_interval_minutes' in additional_data:
                data_source_info['bar_size'] = f"{additional_data['data_interval_minutes']}m"
            elif 'timeframe' in additional_data:
                data_source_info['bar_size'] = additional_data['timeframe']
            if 'timezone' in additional_data:
                data_source_info['timezone'] = additional_data['timezone']
            if 'market_calendar' in additional_data:
                data_source_info['market_calendar'] = additional_data['market_calendar']
            elif 'session_filters' in additional_data:
                data_source_info['session_filters'] = additional_data['session_filters']
        
        if data_source_info:
            full_metadata['data_source'] = data_source_info
        
        # NEW: Add evaluation details
        evaluation_info = {}
        if additional_data:
            if 'target_definition' in additional_data:
                evaluation_info['target_definition'] = additional_data['target_definition']
            elif 'target_config' in additional_data:
                # Store a hash or summary of target config
                try:
                    target_config = additional_data['target_config']
                    if isinstance(target_config, dict):
                        target_config_str = json.dumps(target_config, sort_keys=True)
                        evaluation_info['target_config_hash'] = hashlib.sha256(target_config_str.encode()).hexdigest()[:16]
                except Exception:
                    pass
            
            # Feature counts
            if 'feature_names' in additional_data:
                feature_names = additional_data['feature_names']
                if isinstance(feature_names, (list, tuple)):
                    evaluation_info['n_features'] = len(feature_names)
                    # Count features by family (if feature names follow a pattern)
                    family_counts = {}
                    for feat_name in feature_names:
                        # Common pattern: family_feature or family__feature
                        parts = str(feat_name).split('_', 1)
                        if len(parts) >= 1:
                            family = parts[0]
                            family_counts[family] = family_counts.get(family, 0) + 1
                    if family_counts:
                        evaluation_info['feature_family_counts'] = family_counts
            
            if 'feature_registry_version' in additional_data:
                evaluation_info['feature_registry_version'] = additional_data['feature_registry_version']
        
        if evaluation_info:
            full_metadata['evaluation'] = evaluation_info
        
        # NEW: Add training information (hyperparameters, train_seed) for TRAINING and FEATURE_SELECTION stages
        # CRITICAL: This is needed for comparability - different HPs/seeds = different outcomes
        # FEATURE_SELECTION also uses models (LightGBM, etc.) with hyperparameters that affect feature selection
        if stage_normalized in ["TRAINING", "FEATURE_SELECTION"] and additional_data:
            training_info = {}
            
            # Extract train_seed (distinct from split_seed)
            train_seed = (
                additional_data.get('train_seed') or
                additional_data.get('seed') or
                run_data.get('train_seed') or
                run_data.get('seed')
            )
            if train_seed is not None:
                try:
                    training_info['train_seed'] = int(train_seed)
                except (ValueError, TypeError):
                    pass
            
            # Extract hyperparameters from training config
            hyperparameters = {}
            if 'training' in additional_data and isinstance(additional_data['training'], dict):
                training_config = additional_data['training']
                # Extract all hyperparameters (exclude model_family, strategy, seeds - those are handled separately)
                excluded_keys = {'model_family', 'strategy', 'split_seed', 'train_seed', 'seed'}
                for key, value in training_config.items():
                    if key not in excluded_keys and value is not None:
                        hyperparameters[key] = value
            elif 'hyperparameters' in additional_data:
                # Direct hyperparameters dict
                hp_dict = additional_data['hyperparameters']
                if isinstance(hp_dict, dict):
                    hyperparameters = hp_dict
            
            if hyperparameters:
                training_info['hyperparameters'] = hyperparameters
            
            if training_info:
                full_metadata['training'] = training_info
        
        # NEW: Compute comparable_key for run comparison
        try:
            comparable_key = compute_comparable_key(
                stage=stage_normalized,
                target_name=item_name,
                route_type=full_metadata.get('route_type'),
                view=full_metadata.get('view'),
                symbol=full_metadata.get('symbol'),
                date_range_start=full_metadata.get('date_range_start'),
                date_range_end=full_metadata.get('date_range_end'),
                cv_details=full_metadata.get('cv_details'),
                feature_registry_hash=full_metadata.get('feature_registry_hash'),
                label_definition_hash=full_metadata.get('cv_details', {}).get('label_definition_hash') if full_metadata.get('cv_details') else None,
                min_cs=full_metadata.get('min_cs'),
                max_cs_samples=full_metadata.get('max_cs_samples'),
                universe_id=full_metadata.get('universe_id')
            )
            if comparable_key:
                full_metadata['comparable_key'] = comparable_key
        except Exception as e:
            logger.debug(f"Failed to compute comparable_key: {e}")
        
        # CRITICAL: Initialize telemetry if not already initialized
        # Telemetry is needed for diff tracking and should be available for all runs
        # Check if telemetry exists as instance variable or needs to be created
        telemetry = getattr(self, '_telemetry', None)
        if telemetry is None:
            try:
                from TRAINING.orchestration.utils.diff_telemetry import DiffTelemetry
                
                # Determine base output directory for telemetry
                # CRITICAL: Find the run directory (contains REPRODUCIBILITY), not just RESULTS
                # The run directory is needed so DiffTelemetry can create snapshot_index.json in the right place
                base_output_dir = Path(cohort_dir)
                run_dir = None
                
                # Walk up from cohort_dir to find the run directory (has REPRODUCIBILITY subdirectory)
                temp_dir = base_output_dir
                for _ in range(10):  # Limit depth
                    if (temp_dir / "REPRODUCIBILITY").exists():
                        run_dir = temp_dir
                        break
                    if not temp_dir.parent.exists():
                        break
                    temp_dir = temp_dir.parent
                
                # If we found a run directory, use it; otherwise fall back to finding RESULTS
                if run_dir:
                    base_output_dir = run_dir
                else:
                    # Fallback: walk up to find RESULTS directory
                    temp_dir = Path(cohort_dir)
                    for _ in range(10):
                        if temp_dir.name == "RESULTS":
                            base_output_dir = temp_dir
                            break
                        if not temp_dir.parent.exists():
                            break
                        temp_dir = temp_dir.parent
                    
                    # If we couldn't find RESULTS, use the output_dir from tracker
                    if base_output_dir.name != "RESULTS":
                        base_output_dir = self._repro_base_dir
                
                # Initialize telemetry (creates TELEMETRY directory if needed)
                telemetry = DiffTelemetry(output_dir=base_output_dir)
                # Store as instance variable for reuse
                self._telemetry = telemetry
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize diff telemetry: {e}")
                import traceback
                logger.debug(f"Telemetry initialization traceback: {traceback.format_exc()}")
                telemetry = None
                self._telemetry = None
        
        # CRITICAL: Call finalize_run() BEFORE adding diff_telemetry to full_metadata
        # This ensures snapshot/diff computation uses the exact same resolved_metadata that will be written
        # Pass full_metadata (without diff_telemetry) as resolved_metadata for SST consistency
        diff_telemetry_data = None
        if telemetry:
            try:
                # Extract experiment_id if available
                experiment_id = None
                if additional_data and 'experiment_id' in additional_data:
                    experiment_id = additional_data['experiment_id']
                elif run_data.get('additional_data') and 'experiment_id' in run_data.get('additional_data', {}):
                    experiment_id = run_data['additional_data']['experiment_id']
                
                # Add experiment_id to additional_data if not present
                if experiment_id and additional_data and 'experiment_id' not in additional_data:
                    additional_data = additional_data.copy()
                    additional_data['experiment_id'] = experiment_id
                
                # CRITICAL: Pass full_metadata (without diff_telemetry) as resolved_metadata for SST consistency
                # This ensures snapshot/diff computation uses the exact same data that will be written to metadata.json
                # full_metadata is already built above (lines 1077-1292), we just haven't added diff_telemetry yet
                diff_telemetry_data = telemetry.finalize_run(
                    stage=stage_normalized,
                    run_data=run_data,
                    cohort_dir=cohort_dir,
                    cohort_metadata=cohort_metadata,
                    additional_data=additional_data,
                    resolved_metadata=full_metadata  # CRITICAL: Pass in-memory metadata for SST consistency
                )
                
                # Store diff telemetry data for integration into metadata/metrics
                if diff_telemetry_data:
                    if additional_data is None:
                        additional_data = {}
                    additional_data['diff_telemetry'] = diff_telemetry_data
            except Exception as e:
                logger.warning(f"⚠️  Diff telemetry failed (non-critical): {e}")
                import traceback
                logger.debug(f"Diff telemetry traceback: {traceback.format_exc()}")
        
        # NEW: Add diff telemetry to metadata (full audit trail)
        # CRITICAL: diff_telemetry is optional - older runs may not have it
        # CRITICAL: Only process if diff_telemetry_data was successfully computed (not None)
        if additional_data and 'diff_telemetry' in additional_data and diff_telemetry_data is not None:
            diff_telemetry = additional_data['diff_telemetry']
            snapshot = diff_telemetry.get('snapshot', {})
            diff = diff_telemetry.get('diff', {})
            
            # Extract comparison group key
            comparison_group = snapshot.get('comparison_group')
            comparison_group_key = None
            if isinstance(comparison_group, dict):
                # Construct key from dict fields (matching ComparisonGroup.to_key() logic)
                comparison_group_key = _construct_comparison_group_key_from_dict(comparison_group)
            elif hasattr(comparison_group, 'to_key'):
                comparison_group_key = comparison_group.to_key()
            
            # Extract diff telemetry metadata (full detail for audit trail)
            # CRITICAL: Always write valid diff_telemetry object, even if not comparable
            # Schema mismatches should still produce valid structure with comparability.reason set
            diff_telemetry_blob = {
                'fingerprint_schema_version': snapshot.get('fingerprint_schema_version'),
                'comparison_group_key': comparison_group_key,
                'comparison_group': comparison_group if isinstance(comparison_group, dict) else (
                    comparison_group.to_dict() if hasattr(comparison_group, 'to_dict') else None
                ),
                'fingerprints': {
                    'config_fingerprint': snapshot.get('config_fingerprint'),
                    'data_fingerprint': snapshot.get('data_fingerprint'),
                    'feature_fingerprint': snapshot.get('feature_fingerprint'),
                    'target_fingerprint': snapshot.get('target_fingerprint')
                },
                'fingerprint_sources': snapshot.get('fingerprint_sources', {}),
                'comparability': {
                    'comparable': diff.get('comparable', False),
                    'comparability_reason': diff.get('comparability_reason') or None,  # Explicit None for clarity
                    'prev_run_id': diff.get('prev_run_id')  # Can be None for first run
                },
                'excluded_factors': {
                    'changed': bool(diff.get('excluded_factors_changed', {})),
                    'summary': diff.get('summary', {}).get('excluded_factors_summary'),
                    'changes': diff.get('excluded_factors_changed', {})  # Full payload - empty dict if no changes
                }
            }
            
            # Compute digest of diff_telemetry blob for integrity verification
            # Algorithm: SHA256 hash of canonical JSON (sorted keys, strict JSON-primitive-only)
            # Digest is full SHA256 hash (64 hex characters, 256 bits of entropy)
            # 
            # CRITICAL: diff_telemetry must contain only JSON-primitive types (str/int/float/bool/null/lists/dicts).
            # If non-primitive types are present, this indicates a normalization bug upstream.
            # We fail fast (raise) rather than silently coerce to strings, to catch bugs early.
            try:
                # Strict serialization - will raise TypeError if non-primitive types present
                # This ensures we catch normalization bugs immediately rather than hiding them
                canonical_json = json.dumps(diff_telemetry_blob, sort_keys=True)
                
                # Use full SHA256 hash (64 hex characters, 256 bits) for maximum collision resistance
                digest = hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()
                diff_telemetry_blob['diff_telemetry_digest'] = digest
            except (TypeError, ValueError) as e:
                # Non-JSON-primitive types detected - this is a normalization bug
                # CRITICAL: Don't raise - still write metadata.json without diff_telemetry rather than failing completely
                logger.error(f"diff_telemetry contains non-JSON-primitive types (normalization bug): {e}")
                logger.error("Cannot compute diff_telemetry_digest - diff_telemetry must contain only JSON-primitive types")
                logger.error("Writing metadata.json without diff_telemetry to prevent complete failure")
                # Clear diff_telemetry_blob so it's not added to metadata
                diff_telemetry_blob = None
            except Exception as e:
                # CRITICAL: Don't raise - still write metadata.json without diff_telemetry rather than failing completely
                logger.error(f"Unexpected error computing diff_telemetry_digest: {e}")
                logger.error("Writing metadata.json without diff_telemetry to prevent complete failure")
                # Clear diff_telemetry_blob so it's not added to metadata
                diff_telemetry_blob = None
            
            # CRITICAL: Only add diff_telemetry if blob was successfully created (not None)
            # If digest computation failed, diff_telemetry_blob is set to None and we skip it
            # This ensures metadata.json is still written even if diff telemetry has issues
            if diff_telemetry_blob is not None and 'diff_telemetry_digest' in diff_telemetry_blob:
                full_metadata['diff_telemetry'] = diff_telemetry_blob
            else:
                logger.warning("Skipping diff_telemetry in metadata.json due to computation failure")
        
        # Determine target-first directory (for TARGET_RANKING and FEATURE_SELECTION stages)
        target_cohort_dir = None
        if stage_normalized in ["TARGET_RANKING", "FEATURE_SELECTION"]:
            try:
                from TRAINING.orchestration.utils.target_first_paths import (
                    get_target_reproducibility_dir, ensure_target_structure
                )
                
                # Determine view from route_type or additional_data
                view_for_target = None
                if stage_normalized == "TARGET_RANKING":
                    # For TARGET_RANKING, view comes from route_type or additional_data
                    if route_type and route_type.upper() in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "LOSO"]:
                        view_for_target = route_type.upper()
                    elif additional_data and 'view' in additional_data:
                        view_for_target = additional_data['view'].upper()
                    if not view_for_target:
                        view_for_target = "CROSS_SECTIONAL"  # Default
                elif stage_normalized == "FEATURE_SELECTION":
                    # For FEATURE_SELECTION, map route_type to view
                    if route_type:
                        if route_type.upper() == "CROSS_SECTIONAL":
                            view_for_target = "CROSS_SECTIONAL"
                        elif route_type.upper() in ["INDIVIDUAL", "SYMBOL_SPECIFIC"]:
                            view_for_target = "SYMBOL_SPECIFIC"  # Map INDIVIDUAL to SYMBOL_SPECIFIC for consistency
                    elif additional_data and 'view' in additional_data:
                        view_for_target = additional_data['view'].upper()
                    if not view_for_target:
                        view_for_target = "CROSS_SECTIONAL"  # Default
                
                # Get base output directory (run directory, not REPRODUCIBILITY subdirectory)
                base_output_dir = self._repro_base_dir
                
                # Ensure target structure exists
                ensure_target_structure(base_output_dir, item_name)
                
                # Build target-first reproducibility path: 
                # For CROSS_SECTIONAL: targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<cohort_id>/
                # For SYMBOL_SPECIFIC: targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<cohort_id>/
                target_repro_dir = get_target_reproducibility_dir(base_output_dir, item_name)
                if view_for_target == "SYMBOL_SPECIFIC" and symbol:
                    # Include symbol in path to prevent overwriting
                    target_cohort_dir = target_repro_dir / view_for_target / f"symbol={symbol}" / f"cohort={cohort_id}"
                else:
                    target_cohort_dir = target_repro_dir / view_for_target / f"cohort={cohort_id}"
                target_cohort_dir.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created target-first cohort directory: {target_cohort_dir}")
            except Exception as e:
                # Don't fail if target-first structure creation fails - old structure is primary
                # But log at INFO level so we can see if there are issues
                logger.info(f"⚠️ Failed to create target-first structure for {item_name}/{view_for_target}/cohort={cohort_id} (non-critical): {e}")
                import traceback
                logger.debug(f"Target-first structure creation traceback: {traceback.format_exc()}")
                target_cohort_dir = None
        
        # Save metadata.json atomically to both old and new structures
        metadata_file = cohort_dir / "metadata.json"
        try:
            _write_atomic_json(metadata_file, full_metadata)
            # Log at INFO level so it's visible
            main_logger = _get_main_logger()
            if main_logger != logger:
                main_logger.info(f"✅ Reproducibility: Saved metadata.json to {metadata_file.name} in {metadata_file.parent.name}/")
            else:
                logger.info(f"✅ Reproducibility: Saved metadata.json to {metadata_file.name} in {metadata_file.parent.name}/")
        except (IOError, OSError) as e:
            logger.warning(f"Failed to save metadata.json to {metadata_file}: {e}, error_type=IO_ERROR")
            self._increment_error_counter("write_failures", "IO_ERROR")
            raise  # Re-raise to prevent silent failure
        
        # Also write to target-first structure
        if target_cohort_dir:
            try:
                target_metadata_file = target_cohort_dir / "metadata.json"
                _write_atomic_json(target_metadata_file, full_metadata)
                logger.debug(f"✅ Also saved metadata.json to target-first structure: {target_metadata_file}")
            except Exception as e:
                logger.debug(f"Failed to save metadata.json to target-first structure (non-critical): {e}")
        
        # Write metrics sidecar files (if enabled)
        if self.metrics:
            # Determine view from route_type
            view = None
            if route_type:
                if route_type in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
                    view = route_type
                elif route_type == "INDIVIDUAL":
                    view = "SYMBOL_SPECIFIC"
            
            # Determine target (for TARGET_RANKING and FEATURE_SELECTION stages, item_name is the target)
            target = item_name if stage_normalized in ["TARGET_RANKING", "FEATURE_SELECTION"] else None
            
            # Generate baseline key for drift comparison: (stage, view, target[, symbol])
            # For FEATURE_SELECTION, use route_type as view (CROSS_SECTIONAL or INDIVIDUAL)
            baseline_key = None
            if target:
                # For TARGET_RANKING, view comes from route_type (CROSS_SECTIONAL, SYMBOL_SPECIFIC)
                # For FEATURE_SELECTION, route_type is CROSS_SECTIONAL or INDIVIDUAL (maps to view)
                if stage_normalized == "TARGET_RANKING" and view:
                    baseline_key = f"{stage_normalized}:{view}:{target}"
                    if symbol and view == "SYMBOL_SPECIFIC":
                        baseline_key += f":{symbol}"
                elif stage_normalized == "FEATURE_SELECTION" and route_type:
                    # Map route_type to view for FEATURE_SELECTION
                    fs_view = route_type if route_type in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"] else "CROSS_SECTIONAL"
                    baseline_key = f"{stage_normalized}:{fs_view}:{target}"
                    if symbol and route_type == "INDIVIDUAL":
                        baseline_key += f":{symbol}"
            
            logger.debug(f"📊 Writing metrics for stage={stage_normalized}, target={target}, view={view}, cohort_dir={cohort_dir}")
            
            # Write metrics sidecar files in cohort directory
            # Note: metrics will create cohort_dir if it doesn't exist, or fall back to target level
            metrics_written = False
            try:
                # Extract diff telemetry data if available
                diff_telemetry = None
                if additional_data and 'diff_telemetry' in additional_data:
                    diff_telemetry = additional_data['diff_telemetry']
                
                self.metrics.write_cohort_metrics(
                    cohort_dir=cohort_dir,
                    stage=stage_normalized,
                    view=view or "UNKNOWN",
                    target=target,
                    symbol=symbol,
                    run_id=run_id_clean,
                    metrics=run_data,
                    baseline_key=baseline_key,
                    diff_telemetry=diff_telemetry
                )
                metrics_written = True
            except Exception as e:
                logger.warning(f"⚠️  Failed to write metrics metadata to cohort directory: {e}")
                import traceback
                logger.debug(f"Metrics write traceback: {traceback.format_exc()}")
            
            # Safety fallback: If cohort-level write failed and we have target/view info, try target-level write
            if not metrics_written and target and view:
                try:
                    fallback_dir = self.metrics._get_fallback_metrics_dir(stage_normalized, view, target, symbol)
                    if fallback_dir:
                        logger.info(f"📁 Attempting metrics fallback write to: {fallback_dir}")
                        self.metrics._write_metrics(fallback_dir, run_id_clean, run_data, stage=stage_normalized, reproducibility_mode="COHORT_AWARE")
                        if baseline_key:
                            self.metrics._write_drift(
                                fallback_dir, stage_normalized, view, target, symbol, run_id_clean, run_data, baseline_key
                            )
                        logger.info(f"✅ Metrics written to fallback location: {fallback_dir}")
                except Exception as e2:
                    logger.warning(f"⚠️  Metrics fallback write also failed: {e2}")
            
            # Aggregate metrics facts table (append-only, after all cohorts saved)
            # This is called per-cohort, but we'll aggregate at the end of the run
            # For now, we'll aggregate on-demand or at end of stage
            
            # PHASE 3: Diff telemetry (first-class change tracking)
            try:
                from TRAINING.orchestration.utils.diff_telemetry import DiffTelemetry
                
                # Determine base output directory for telemetry
                # Walk up from cohort_dir to find run-level directory
                base_output_dir = Path(cohort_dir)
                while base_output_dir.name not in ["RESULTS"] and base_output_dir.parent.exists():
                    base_output_dir = base_output_dir.parent
                    if base_output_dir.name == "RESULTS":
                        break
                
                # If we're still in REPRODUCIBILITY or a subdirectory, walk up more
                if base_output_dir.name not in ["RESULTS"]:
                    # Try to find RESULTS directory
                    temp_dir = base_output_dir
                    for _ in range(5):  # Limit depth
                        if temp_dir.name == "RESULTS":
                            base_output_dir = temp_dir
                            break
                        if not temp_dir.parent.exists():
                            break
                        temp_dir = temp_dir.parent
                
                # If we couldn't find RESULTS, use the output_dir from tracker
                if base_output_dir.name != "RESULTS":
                    base_output_dir = self.output_dir
                
                # Initialize telemetry (creates TELEMETRY directory if needed)
                telemetry = DiffTelemetry(output_dir=base_output_dir)
                
                # Extract experiment_id if available
                experiment_id = None
                if additional_data and 'experiment_id' in additional_data:
                    experiment_id = additional_data['experiment_id']
                elif run_data.get('additional_data') and 'experiment_id' in run_data.get('additional_data', {}):
                    experiment_id = run_data['additional_data']['experiment_id']
                
                # Add experiment_id to additional_data if not present
                if experiment_id and additional_data and 'experiment_id' not in additional_data:
                    additional_data = additional_data.copy()
                    additional_data['experiment_id'] = experiment_id
                
                # CRITICAL: Use existing full_metadata (built at line 1077) for SST consistency
                # This ensures snapshot/diff computation uses the exact same data that gets written to metadata.json
                # full_metadata is already built above with all outcome-influencing fields
                # We just haven't added diff_telemetry to it yet (that happens after finalize_run())
                
                # Finalize run with diff telemetry (pass resolved_metadata for SST consistency)
                diff_telemetry_data = telemetry.finalize_run(
                    stage=stage_normalized,
                    run_data=run_data,
                    cohort_dir=cohort_dir,
                    cohort_metadata=cohort_metadata,
                    additional_data=additional_data,
                    resolved_metadata=full_metadata  # CRITICAL: Use same full_metadata as will be written to metadata.json
                )
                
                # Store diff telemetry data for integration into metadata/metrics
                if diff_telemetry_data:
                    if additional_data is None:
                        additional_data = {}
                    additional_data['diff_telemetry'] = diff_telemetry_data
            except Exception as e:
                # CRITICAL: Log at WARNING level so it's visible - this indicates a real problem
                logger.warning(f"⚠️  Diff telemetry failed for {stage_normalized}:{item_name}: {e}")
                import traceback
                logger.debug(f"Diff telemetry traceback: {traceback.format_exc()}")
                # Don't re-raise - metadata.json and metrics.json should still be written even if diff telemetry fails
        
        # PHASE 2: Unified schema - build metrics_data for _update_index (always needed)
        # Metrics writer writes metrics.json/parquet, but we still need metrics_data for index
        metrics_data = {
            "run_id": run_id_clean,
            "timestamp": datetime.now().isoformat(),
            "reproducibility_mode": "COHORT_AWARE",  # Track which mode was used
            "stage": stage_normalized,  # Ensure consistent uppercase naming
            **{k: v for k, v in run_data.items() 
               if k not in ['timestamp', 'cohort_metadata', 'additional_data']}
        }
        
        # PHASE 2: Only write as fallback if metrics failed and we don't have metrics.json yet
        metrics_file = cohort_dir / "metrics.json"
        if not metrics_file.exists() and not metrics_written:
            # Fallback: write metrics.json using unified canonical schema (atomically)
            try:
                _write_atomic_json(metrics_file, metrics_data)
                # Also write metrics.parquet for consistency
                try:
                    import pandas as pd
                    df_metrics = pd.DataFrame([metrics_data])
                    metrics_parquet = cohort_dir / "metrics.parquet"
                    df_metrics.to_parquet(metrics_parquet, index=False, engine='pyarrow', compression='snappy')
                    
                    # Also write to target-first structure
                    if target_cohort_dir:
                        try:
                            target_metrics_file = target_cohort_dir / "metrics.json"
                            _write_atomic_json(target_metrics_file, metrics_data)
                            target_metrics_parquet = target_cohort_dir / "metrics.parquet"
                            df_metrics.to_parquet(target_metrics_parquet, index=False, engine='pyarrow', compression='snappy')
                            logger.debug(f"✅ Also saved metrics.json/parquet to target-first structure")
                        except Exception as e:
                            logger.debug(f"Failed to save metrics to target-first structure (non-critical): {e}")
                except Exception as e_parquet:
                    logger.debug(f"Failed to write metrics.parquet fallback: {e_parquet}")
                # Log at INFO level so it's visible
                main_logger = _get_main_logger()
                if main_logger != logger:
                    main_logger.info(f"✅ Reproducibility: Saved metrics.json (fallback) to {metrics_file.name} in {metrics_file.parent.name}/")
                else:
                    logger.info(f"✅ Reproducibility: Saved metrics.json (fallback) to {metrics_file.name} in {metrics_file.parent.name}/")
            except (IOError, OSError) as e:
                logger.warning(f"Failed to save metrics.json (fallback) to {metrics_file}: {e}, error_type=IO_ERROR")
                self._increment_error_counter("write_failures", "IO_ERROR")
                # Don't raise - metrics might have written it, or we'll try again
        elif metrics_written and target_cohort_dir:
            # Metrics were written by MetricsWriter, also write to target-first structure
            try:
                # Read the metrics.json that was written
                if metrics_file.exists():
                    import json
                    with open(metrics_file, 'r') as f:
                        metrics_data_written = json.load(f)
                    target_metrics_file = target_cohort_dir / "metrics.json"
                    _write_atomic_json(target_metrics_file, metrics_data_written)
                    logger.debug(f"✅ Also saved metrics.json to target-first structure: {target_metrics_file}")
                
                # Also copy metrics.parquet if it exists
                metrics_parquet = cohort_dir / "metrics.parquet"
                if metrics_parquet.exists():
                    import shutil
                    target_metrics_parquet = target_cohort_dir / "metrics.parquet"
                    shutil.copy2(metrics_parquet, target_metrics_parquet)
                    logger.debug(f"✅ Also saved metrics.parquet to target-first structure: {target_metrics_parquet}")
            except Exception as e:
                logger.debug(f"Failed to save metrics to target-first structure (non-critical): {e}")
        
        # Update index.parquet
        try:
            self._update_index(
                stage, item_name, route_type, symbol, model_family,
                cohort_id, run_id_clean, full_metadata, metrics_data, cohort_dir
            )
        except Exception as e:
            error_type = "IO_ERROR" if isinstance(e, (IOError, OSError)) else "SERIALIZATION_ERROR" if isinstance(e, (json.JSONDecodeError, TypeError)) else "UNKNOWN_ERROR"
            logger.warning(f"Failed to update index.parquet: {e}, error_type={error_type}")
            self._increment_error_counter("index_update_failures", error_type)
            # Don't re-raise - index update failure shouldn't break the run
        
        # Post-run decision hook: Evaluate and persist decisions
        try:
            from TRAINING.decisioning.decision_engine import DecisionEngine
            from TRAINING.ranking.utils.resolved_config import get_cfg
            
            repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
            index_file = repro_dir / "index.parquet"
            if index_file.exists():
                # Check if Bayesian policy is enabled
                use_bayesian = get_cfg("training.decisions.use_bayesian", default=False, config_name="training_config")
                
                engine = DecisionEngine(
                    index_file,
                    apply_mode=False,  # Assist mode by default
                    use_bayesian=use_bayesian,
                    base_dir=self.output_dir.parent
                )
                # Get segment_id from index if available
                segment_id_for_decision = None
                try:
                    if index_file.exists():
                        df_temp = pd.read_parquet(index_file)
                        mask = df_temp['cohort_id'] == cohort_id
                        if mask.any():
                            segment_id_for_decision = df_temp[mask]['segment_id'].iloc[-1] if 'segment_id' in df_temp.columns else None
                except Exception:
                    pass
                decision_result = engine.evaluate(cohort_id, run_id_clean, segment_id=segment_id_for_decision)
                engine.persist(decision_result, self.output_dir.parent)
                
                # Update Bayesian state if enabled
                if use_bayesian and engine.bayesian_policy:
                    try:
                        # Get applied patch template from decision result
                        applied_patch_template = None
                        if decision_result.policy_results and 'bayesian_patch' in decision_result.policy_results:
                            bayesian_result = decision_result.policy_results['bayesian_patch']
                            applied_patch_template = bayesian_result.get('recommended_patch')
                        
                        # Update Bayesian state with observed reward
                        engine.update_bayesian_state(
                            decision_result=decision_result,
                            current_run_metrics=metrics,
                            applied_patch_template=applied_patch_template
                        )
                    except Exception as e:
                        logger.debug(f"Bayesian state update failed (non-critical): {e}")
                
                # Store decision fields in metrics for index update
                metrics['decision_level'] = decision_result.decision_level
                metrics['decision_action_mask'] = decision_result.decision_action_mask
                metrics['decision_reason_codes'] = decision_result.decision_reason_codes
                if decision_result.decision_level > 0:
                    logger.info(f"📊 Decision: level={decision_result.decision_level}, actions={decision_result.decision_action_mask}, reasons={decision_result.decision_reason_codes}")
                    # Log Bayesian metadata if available
                    if decision_result.policy_results and 'bayesian_metadata' in decision_result.policy_results:
                        bayes_meta = decision_result.policy_results['bayesian_metadata']
                        logger.info(f"🎲 Bayesian: confidence={bayes_meta.get('confidence', 0):.3f}, expected_gain={bayes_meta.get('expected_gain', 0):.4f}")
        except Exception as e:
            logger.debug(f"Decision evaluation failed (non-critical): {e}")
            # Don't re-raise - decision evaluation is optional
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            from TRAINING.common.subprocess_utils import safe_subprocess_run
            result = safe_subprocess_run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"Failed to get git commit: {e}")
        return None
    
    def _parse_run_started_at(self, run_id: str, created_at: Optional[str] = None) -> str:
        """
        Parse run_started_at from run_id or use created_at.
        
        run_id formats:
        - YYYYMMDD_HHMMSS_* (preferred)
        - YYYY-MM-DDTHH:MM:SS* (ISO format)
        - Other formats fall back to created_at
        """
        import re
        from datetime import datetime
        
        # Try to parse from run_id first
        # Format: YYYYMMDD_HHMMSS_*
        match = re.match(r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', run_id)
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                return dt.isoformat() + 'Z'
            except ValueError:
                pass
        
        # Try ISO format: YYYY-MM-DDTHH:MM:SS*
        match = re.match(r'(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})', run_id)
        if match:
            year, month, day, hour, minute, second = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
                return dt.isoformat() + 'Z'
            except ValueError:
                pass
        
        # Fall back to created_at
        if created_at:
            try:
                # Ensure it's in ISO format with timezone
                dt = pd.to_datetime(created_at, utc=True)
                return dt.isoformat()
            except Exception:
                pass
        
        # Last resort: use current time
        return datetime.now().isoformat() + 'Z'
    
    def _increment_error_counter(self, counter_name: str, error_type: str = "UNKNOWN") -> None:
        """
        Increment error counter in stats.json.
        
        Args:
            counter_name: Name of the counter (e.g., "write_failures", "index_update_failures")
            error_type: Type of error (e.g., "IO_ERROR", "SERIALIZATION_ERROR")
        """
        try:
            # Load existing stats
            if self.stats_file.exists():
                try:
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                except (json.JSONDecodeError, IOError):
                    stats = {}
            else:
                stats = {}
            
            # Initialize counters if needed
            if "errors" not in stats:
                stats["errors"] = {}
            if counter_name not in stats["errors"]:
                stats["errors"][counter_name] = {}
            if error_type not in stats["errors"][counter_name]:
                stats["errors"][counter_name][error_type] = 0
            
            # Increment
            stats["errors"][counter_name][error_type] = stats["errors"][counter_name].get(error_type, 0) + 1
            stats["last_updated"] = datetime.now().isoformat()
            
            # Save
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
                f.flush()  # Ensure immediate write
                os.fsync(f.fileno())  # Force write to disk
        except Exception as e:
            # Don't log here to avoid recursion - stats are best-effort
            pass
    
    def _increment_mode_counter(self, mode: str) -> None:
        """Increment mode usage counter (COHORT_AWARE vs LEGACY)."""
        try:
            if self.stats_file.exists():
                try:
                    with open(self.stats_file, 'r') as f:
                        stats = json.load(f)
                except (json.JSONDecodeError, IOError):
                    stats = {}
            else:
                stats = {}
            
            if "modes" not in stats:
                stats["modes"] = {}
            if mode not in stats["modes"]:
                stats["modes"][mode] = 0
            
            stats["modes"][mode] = stats["modes"].get(mode, 0) + 1
            stats["last_updated"] = datetime.now().isoformat()
            
            self.stats_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump(stats, f, indent=2)
        except Exception:
            pass  # Best-effort stats
    
    def _update_index(
        self,
        stage: str,
        item_name: str,
        route_type: Optional[str],
        symbol: Optional[str],
        model_family: Optional[str],
        cohort_id: str,
        run_id: str,
        metadata: Dict[str, Any],
        metrics: Dict[str, Any],
        cohort_dir: Path
    ) -> None:
        """Update the global index.parquet file."""
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
        index_file = repro_dir / "index.parquet"
        
        # Normalize stage
        if isinstance(stage, Stage):
            phase = stage.value
        else:
            phase = stage.upper().replace("MODEL_TRAINING", "TRAINING")
        
        # Normalize route_type
        # For TARGET_RANKING, route_type is actually the view (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO)
        if route_type and isinstance(route_type, RouteType):
            mode = route_type.value
        else:
            mode = route_type.upper() if route_type else None
        
        # For TARGET_RANKING, mode is the view (already handled in _get_cohort_dir)
        
        # Compute segment_id for decision-making (segments reset on identity breaks)
        segment_id = None
        try:
            from TRAINING.common.utils.regression_analysis import prepare_segments
            # Load existing index to compute segment
            if index_file.exists():
                try:
                    df_existing = pd.read_parquet(index_file)
                    # Filter to same cohort
                    cohort_mask = df_existing['cohort_id'] == cohort_id
                    if cohort_mask.any():
                        df_cohort = df_existing[cohort_mask].copy()
                        # Prepare segments (adds segment_id column)
                        df_cohort = prepare_segments(df_cohort, time_col='run_started_at')
                        # Get segment_id for this run (will be computed based on identity fields)
                        # We need to check if this run's identity fields match previous segment
                        if len(df_cohort) > 0:
                            last_segment = df_cohort['segment_id'].iloc[-1]
                            last_row = df_cohort.iloc[-1]
                            # Check if identity fields changed
                            identity_cols = ["data_fingerprint", "featureset_fingerprint", "config_hash", "git_commit"]
                            identity_changed = False
                            for col in identity_cols:
                                new_val = metadata.get(col) or (metrics.get(col) if col in metrics else None)
                                old_val = last_row.get(col)
                                if new_val != old_val and (new_val is not None or old_val is not None):
                                    identity_changed = True
                                    break
                            if identity_changed:
                                segment_id = int(last_segment) + 1
                            else:
                                segment_id = int(last_segment)
                        else:
                            segment_id = 0
                    else:
                        segment_id = 0  # First run in cohort
                except Exception:
                    segment_id = 0  # Fallback: first segment
            else:
                segment_id = 0  # First run ever
        except ImportError:
            segment_id = None  # Regression analysis not available
        
        # Extract CV details from metadata
        cv_details = metadata.get("cv_details", {})
        
        # Extract regression features for cohort-based tracking
        # Target ranking metrics
        cs_auc = metrics.get("mean_score") or metrics.get("auc") or metrics.get("cs_auc")
        cs_logloss = metrics.get("logloss") or metrics.get("cs_logloss")
        cs_pr_auc = metrics.get("pr_auc") or metrics.get("cs_pr_auc")
        
        # Symbol-specific metrics (from per_symbol_stats or metrics)
        per_symbol_stats = metadata.get("per_symbol_stats", {})
        sym_aucs = None
        if per_symbol_stats and isinstance(per_symbol_stats, dict):
            # Extract AUCs from per_symbol_stats
            sym_aucs = [stats.get("auc") for stats in per_symbol_stats.values() if isinstance(stats, dict) and "auc" in stats]
        elif "sym_aucs" in metrics:
            sym_aucs = metrics.get("sym_aucs")
        
        sym_auc_mean = None
        sym_auc_median = None
        sym_auc_iqr = None
        sym_auc_min = None
        sym_auc_max = None
        if sym_aucs and len(sym_aucs) > 0:
            import numpy as np
            sym_aucs_clean = [a for a in sym_aucs if a is not None and not np.isnan(a)]
            if len(sym_aucs_clean) > 0:
                sym_auc_mean = float(np.mean(sym_aucs_clean))
                sym_auc_median = float(np.median(sym_aucs_clean))
                q75, q25 = np.percentile(sym_aucs_clean, [75, 25])
                sym_auc_iqr = float(q75 - q25)
                sym_auc_min = float(np.min(sym_aucs_clean))
                sym_auc_max = float(np.max(sym_aucs_clean))
        
        # Fraction of symbols with good AUC (threshold from config or default 0.65)
        frac_symbols_good = None
        if sym_aucs and len(sym_aucs) > 0:
            try:
                from CONFIG.config_loader import get_cfg
                threshold = float(get_cfg("training.target_routing.cs_auc_threshold", default=0.65, config_name="training_config"))
            except Exception:
                threshold = 0.65
            good_count = sum(1 for a in sym_aucs if a is not None and a >= threshold)
            frac_symbols_good = good_count / len(sym_aucs) if len(sym_aucs) > 0 else None
        
        # Route information (for stability tracking)
        route = metadata.get("route_type") or metadata.get("view") or mode
        route_changed = None  # Will be computed when comparing runs
        route_entropy = None  # Will be computed from route history
        
        # Class balance (pos_rate)
        pos_rate = metrics.get("pos_rate") or metrics.get("positive_rate") or metrics.get("class_balance")
        
        # Feature counts
        n_features_pre = metrics.get("n_features_pre") or metrics.get("features_safe") or metadata.get("features_safe")
        n_features_post_prune = metrics.get("n_features_post_prune") or metrics.get("features_final") or metadata.get("features_final")
        n_features_selected = metrics.get("n_features_selected") or metrics.get("n_selected") or n_features_post_prune
        
        # Temporal safety (purge/embargo)
        # Schema v2: Extract scalar from tagged unions (backward compatible with v1)
        purge_minutes_used = cv_details.get("purge_minutes") or metadata.get("purge_minutes")
        embargo_minutes_used = extract_embargo_minutes(metadata, cv_details)
        
        # Feature stability metrics (if available)
        jaccard_topK = metrics.get("jaccard_top_k") or metrics.get("jaccard_topK")
        rank_corr_spearman = metrics.get("rank_corr") or metrics.get("rank_correlation") or metrics.get("spearman_corr")
        importance_concentration = metrics.get("importance_concentration") or metrics.get("top10_importance_share")
        
        # Operational metrics
        runtime_sec = metrics.get("runtime_sec") or metrics.get("train_time_sec") or metrics.get("wall_clock_time")
        peak_ram_mb = metrics.get("peak_ram_mb") or metrics.get("peak_memory_mb")
        # Schema v2: Extract scalar from tagged unions (backward compatible with v1)
        cv_folds_executed = extract_folds(metadata, cv_details)
        
        # Identity fields (categorical, not regressed)
        data_fingerprint = metadata.get("data_fingerprint")
        featureset_fingerprint = metadata.get("featureset_fingerprint") or metrics.get("featureset_fingerprint")
        config_hash = metadata.get("cs_config_hash") or metadata.get("config_hash")
        git_commit = metadata.get("git_commit")
        
        # Create new row with all regression features
        new_row = {
            # Identity (categorical)
            "phase": phase,
            "mode": mode,
            "target": item_name,
            "symbol": symbol,
            "model_family": model_family,
            "cohort_id": cohort_id,
            "run_id": run_id,
            "segment_id": segment_id,  # For decision-making (segments reset on identity breaks)
            "data_fingerprint": data_fingerprint,
            "featureset_fingerprint": featureset_fingerprint,
            "config_hash": config_hash,
            "git_commit": git_commit,
            
            # Sample size
            "N_effective": metadata.get("N_effective", 0),
            "n_symbols": metadata.get("n_symbols", 0),
            
            # Target ranking metrics (Y variables for regression)
            "auc": cs_auc,
            "cs_auc": cs_auc,
            "cs_logloss": cs_logloss,
            "cs_pr_auc": cs_pr_auc,
            "sym_auc_mean": sym_auc_mean,
            "sym_auc_median": sym_auc_median,
            "sym_auc_iqr": sym_auc_iqr,
            "sym_auc_min": sym_auc_min,
            "sym_auc_max": sym_auc_max,
            "frac_symbols_good": frac_symbols_good,
            "composite_score": metrics.get("composite_score"),
            "mean_importance": metrics.get("mean_importance"),
            
            # Route stability
            "route": route,
            "route_changed": route_changed,  # Will be computed when comparing
            "route_entropy": route_entropy,  # Will be computed from history
            
            # Class balance
            "pos_rate": pos_rate,
            
            # Feature counts (X variables for regression)
            "n_features_pre": n_features_pre,
            "n_features_post_prune": n_features_post_prune,
            "n_features_selected": n_features_selected,
            
            # Temporal safety (X variables)
            "purge_minutes_used": purge_minutes_used,
            "embargo_minutes_used": embargo_minutes_used,
            "horizon_minutes": cv_details.get("horizon_minutes") or metadata.get("horizon_minutes"),
            
            # Feature stability (X variables)
            "jaccard_topK": jaccard_topK,
            "rank_corr_spearman": rank_corr_spearman,
            "importance_concentration": importance_concentration,
            
            # Operational metrics (Y variables)
            "runtime_sec": runtime_sec,
            "peak_ram_mb": peak_ram_mb,
            "cv_folds_executed": cv_folds_executed,
            
            # Timestamps
            "date": metadata.get("date_start"),
            "created_at": metadata.get("created_at", datetime.now().isoformat()),
            
            # Time for regression (explicit, monotonic)
            "run_started_at": self._parse_run_started_at(run_id, metadata.get("created_at")),
            
            # Decision fields (from decision engine, if available)
            "decision_level": metrics.get("decision_level") or 0,
            "decision_action_mask": json.dumps(metrics.get("decision_action_mask") or []) if metrics.get("decision_action_mask") else None,
            "decision_reason_codes": json.dumps(metrics.get("decision_reason_codes") or []) if metrics.get("decision_reason_codes") else None,
            
            # Path
            "path": str(cohort_dir.relative_to(repro_dir))
        }
        
        # Load existing index or create new
        if index_file.exists():
            try:
                df = pd.read_parquet(index_file)
            except Exception:
                df = pd.DataFrame()
        else:
            df = pd.DataFrame()
        
        # Append new row
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df], ignore_index=True)
        
        # Remove duplicates (keep latest) - idempotency: same run_id+phase will be deduped
        df = df.drop_duplicates(
            subset=["phase", "mode", "target", "symbol", "model_family", "cohort_id", "run_id"],
            keep="last"
        )
        
        # Save with file locking for concurrency safety
        try:
            index_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use file locking to prevent race conditions
            # Create lock file (same directory, different extension)
            lock_file = index_file.with_suffix('.lock')
            
            with open(lock_file, 'w') as lock_f:
                try:
                    # Acquire exclusive lock (blocks until available)
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                    
                    # Re-read index in case another process updated it while we were waiting
                    if index_file.exists():
                        try:
                            df_existing = pd.read_parquet(index_file)
                            # Merge with existing data (dedupe by run_id + phase for idempotency)
                            # This ensures reruns with same run_id don't create duplicate entries
                            df = pd.concat([df_existing, df], ignore_index=True)
                            # Dedupe by (run_id, phase) - keep last (most recent) entry
                            df = df.drop_duplicates(subset=['run_id', 'phase'], keep='last')
                            
                            # Additional safety: if same run_id+phase exists, log debug
                            existing_mask = (
                                (df_existing['run_id'] == new_row['run_id']) &
                                (df_existing['phase'] == new_row['phase'])
                            )
                            if existing_mask.any():
                                logger.debug(
                                    f"Idempotency: Updating existing index entry for "
                                    f"run_id={new_row['run_id']}, phase={new_row['phase']}"
                                )
                        except Exception:
                            # If read fails, use our new data
                            pass
                    
                    # Write updated index
                    df.to_parquet(index_file, index=False)
                    
                    # Parquet files are automatically flushed, but ensure directory is synced
                    if hasattr(os, 'sync'):
                        try:
                            os.sync()  # Sync filesystem (if available)
                        except AttributeError:
                            pass  # os.sync not available on all systems
                    
                    # Lock is automatically released when file is closed
                except Exception as e:
                    # Release lock on error
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                    raise
        except Exception as e:
            error_type = "IO_ERROR" if isinstance(e, (IOError, OSError)) else "SERIALIZATION_ERROR" if isinstance(e, (json.JSONDecodeError, TypeError)) else "UNKNOWN_ERROR"
            logger.warning(f"Failed to save index.parquet to {index_file}: {e}, error_type={error_type}")
            # Don't re-raise - index update failure shouldn't break the run
    
    
    def _find_matching_cohort(
        self,
        stage: str,
        item_name: str,
        cohort_metadata: Dict[str, Any],
        route_type: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None
    ) -> Optional[str]:
        """Find matching cohort ID from index.parquet."""
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
        index_file = repro_dir / "index.parquet"
        
        if not index_file.exists():
            return None
        
        try:
            df = pd.read_parquet(index_file)
            
            phase = stage.upper().replace("MODEL_TRAINING", "TRAINING")
            
            # Filter for same phase, target, mode, symbol, model_family
            mask = (df['phase'] == phase) & (df['target'] == item_name)
            
            if route_type:
                mask &= (df['mode'] == route_type.upper())
            if symbol:
                mask &= (df['symbol'] == symbol)
            if model_family:
                mask &= (df['model_family'] == model_family)
            
            candidates = df[mask]
            
            if len(candidates) == 0:
                return None
            
            # Try exact match first (same cohort_id)
            target_id = self._compute_cohort_id(cohort_metadata, route_type)
            exact_match = candidates[candidates['cohort_id'] == target_id]
            if len(exact_match) > 0:
                return target_id
            
            # Try close match (similar N, same config)
            n_target = cohort_metadata.get('N_effective_cs', 0)
            n_ratio_threshold = self.n_ratio_threshold
            
            for _, row in candidates.iterrows():
                n_existing = row.get('N_effective', 0)
                if n_existing == 0:
                    continue
                
                n_ratio = min(n_target, n_existing) / max(n_target, n_existing)
                if n_ratio >= n_ratio_threshold:
                    # Load metadata to check config match
                    try:
                        prev_path = repro_dir / row['path']
                        metadata_file = prev_path / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                prev_meta = json.load(f)
                            
                            # Check config match
                            config_match = (
                                prev_meta.get('universe_id') == cohort_metadata.get('cs_config', {}).get('universe_id') and
                                prev_meta.get('min_cs') == cohort_metadata.get('cs_config', {}).get('min_cs') and
                                prev_meta.get('leakage_filter_version') == cohort_metadata.get('cs_config', {}).get('leakage_filter_version', 'v1')
                            )
                            if config_match:
                                return row['cohort_id']
                    except Exception as e:
                        logger.debug(f"Failed to check config match for cohort {row['cohort_id']}: {e}")
                        continue
            
            return None
        except Exception as e:
            logger.debug(f"Failed to find matching cohort: {e}")
            return None
    
    def _compare_within_cohort(
        self,
        prev_run: Dict[str, Any],
        curr_run: Dict[str, Any],
        metric_type: str = 'roc_auc'
    ) -> Tuple[str, float, float, Optional[float], Dict[str, Any]]:
        """
        Compare runs within the same cohort using sample-adjusted statistics.
        
        Returns:
            (classification, abs_diff, rel_diff, z_score, stats_dict)
        """
        prev_value = float(prev_run.get('mean_score', 0.0))
        curr_value = float(curr_run.get('mean_score', 0.0))
        
        prev_std = float(prev_run.get('std_score', 0.0)) if prev_run.get('std_score') else None
        curr_std = float(curr_run.get('std_score', 0.0)) if curr_run.get('std_score') else None
        
        # Get sample sizes
        prev_n = prev_run.get('N_effective_cs') or prev_run.get('n_samples') or prev_run.get('sample_size')
        curr_n = curr_run.get('N_effective_cs') or curr_run.get('n_samples') or curr_run.get('sample_size')
        
        if prev_n is None or curr_n is None:
            # Fallback to non-sample-adjusted comparison
            class_result = self._classify_diff(prev_value, curr_value, prev_std, metric_type)
            return class_result + ({'sample_adjusted': False},)
        
        prev_n = int(prev_n)
        curr_n = int(curr_n)
        
        # Sample-adjusted variance estimation
        # For AUC: var ≈ AUC * (1 - AUC) / N
        if prev_value > 0 and prev_value < 1:
            var_prev = prev_value * (1 - prev_value) / prev_n
        else:
            var_prev = (prev_std ** 2) / prev_n if prev_std and prev_std > 0 else None
        
        if curr_value > 0 and curr_value < 1:
            var_curr = curr_value * (1 - curr_value) / curr_n
        else:
            var_curr = (curr_std ** 2) / curr_n if curr_std and curr_std > 0 else None
        
        # Compute z-score
        delta = curr_value - prev_value
        abs_diff = abs(delta)
        rel_diff = (abs_diff / max(abs(prev_value), 1e-8)) * 100 if prev_value != 0 else 0.0
        
        z_score = None
        if var_prev is not None and var_curr is not None:
            sigma = math.sqrt(var_prev + var_curr)
            if sigma > 0:
                z_score = abs_diff / sigma
        
        # Classification using z-score
        thresholds = self.thresholds.get(metric_type, self.thresholds.get('roc_auc'))
        z_thr = thresholds.get('z_score', 1.0)
        
        if z_score is not None:
            if z_score < 1.0:
                classification = 'STABLE'
            elif z_score < 2.0:
                classification = 'DRIFTING'
            else:
                classification = 'DIVERGED'
        else:
            # Fallback to non-sample-adjusted
            classification, _, _, _ = self._classify_diff(prev_value, curr_value, prev_std, metric_type)
        
        stats = {
            'prev_n': prev_n,
            'curr_n': curr_n,
            'var_prev': var_prev,
            'var_curr': var_curr,
            'z_score': z_score,
            'sample_adjusted': True
        }
        
        return classification, abs_diff, rel_diff, z_score, stats
    
    def get_last_comparable_run(
        self,
        stage: str,
        item_name: str,
        route_type: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None,
        cohort_id: Optional[str] = None,
        current_N: Optional[int] = None,
        n_ratio_threshold: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find the last comparable run from index.parquet.
        
        Args:
            stage: Pipeline stage
            item_name: Target/item name
            route_type: Route type (CROSS_SECTIONAL/INDIVIDUAL)
            symbol: Symbol name (for INDIVIDUAL mode)
            model_family: Model family (for TRAINING)
            cohort_id: Cohort ID (if already computed)
            current_N: Current N_effective (for N ratio check)
            n_ratio_threshold: Override default N ratio threshold
        
        Returns:
            Previous run metrics dict or None if no comparable run found
        """
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
        index_file = repro_dir / "index.parquet"
        
        if not index_file.exists():
            return None
        
        try:
            df = pd.read_parquet(index_file)
            
            phase = stage.upper().replace("MODEL_TRAINING", "TRAINING")
            
            # Filter for matching stage, target, mode, symbol, model_family
            mask = (df['phase'] == phase) & (df['target'] == item_name)
            
            # FIX: Handle null mode/symbol for backward compatibility
            # For FEATURE_SELECTION, require mode non-null (new runs must have mode)
            # For other stages, allow nulls (backward compatibility)
            if route_type:
                route_upper = route_type.upper()
                if stage.upper() == "FEATURE_SELECTION":
                    # For FEATURE_SELECTION, require mode non-null (new runs must have mode)
                    mask &= (df['mode'].notna()) & (df['mode'] == route_upper)
                else:
                    # For other stages, allow nulls (backward compatibility)
                    mask &= ((df['mode'].isna()) | (df['mode'] == route_upper))
            
            # FIX: Handle null symbol for backward compatibility
            # For INDIVIDUAL mode, require symbol non-null
            # For CROSS_SECTIONAL, allow nulls (backward compatibility)
            if symbol:
                if route_type and route_type.upper() == "INDIVIDUAL":
                    # For INDIVIDUAL mode, require symbol non-null
                    mask &= (df['symbol'].notna()) & (df['symbol'] == symbol)
                else:
                    # For CROSS_SECTIONAL, allow nulls (backward compatibility)
                    mask &= ((df['symbol'].isna()) | (df['symbol'] == symbol))
            elif route_type and route_type.upper() == "CROSS_SECTIONAL":
                # For CROSS_SECTIONAL, require symbol is null (prevent history forking)
                mask &= (df['symbol'].isna())
            
            if model_family:
                mask &= (df['model_family'] == model_family)
            
            # FIX: Always filter by cohort_id or data_fingerprint (don't compare across cohorts)
            # This prevents noisy comparisons when underlying dataset changes
            if cohort_id:
                mask &= (df['cohort_id'] == cohort_id)
            else:
                # Try to compute cohort_id from current run metadata if available
                # Or filter by data_fingerprint if available (stronger than cohort_id)
                # For now, log warning and allow comparison (may be noisy)
                logger.debug("No cohort_id provided to get_last_comparable_run, comparisons may be noisy")
            
            candidates = df[mask].sort_values('date', ascending=False)
            
            if len(candidates) == 0:
                return None
            
            # Apply N ratio filter if current_N provided
            threshold = n_ratio_threshold or self.n_ratio_threshold
            if current_N is not None:
                for _, row in candidates.iterrows():
                    prev_n = row.get('N_effective', 0)
                    if prev_n == 0:
                        continue
                    
                    n_ratio = min(current_N, prev_n) / max(current_N, prev_n)
                    if n_ratio >= threshold:
                        # Load metrics from path
                        try:
                            prev_path = repro_dir / row['path']
                            metrics_file = prev_path / "metrics.json"
                            if metrics_file.exists():
                                with open(metrics_file, 'r') as f:
                                    metrics = json.load(f)
                                # Also load metadata for cohort_id
                                metadata_file = prev_path / "metadata.json"
                                if metadata_file.exists():
                                    with open(metadata_file, 'r') as f:
                                        metadata = json.load(f)
                                    metrics['cohort_id'] = metadata.get('cohort_id')
                                    metrics['N_effective'] = metadata.get('N_effective')
                                return metrics
                        except Exception as e:
                            logger.debug(f"Failed to load previous run from {row['path']}: {e}")
                            continue
                
                # No run passed N ratio filter
                return None
            else:
                # No N filter - just return latest
                latest = candidates.iloc[0]
                try:
                    prev_path = repro_dir / latest['path']
                    metrics_file = prev_path / "metrics.json"
                    if metrics_file.exists():
                        with open(metrics_file, 'r') as f:
                            metrics = json.load(f)
                        metadata_file = prev_path / "metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            metrics['cohort_id'] = metadata.get('cohort_id')
                            metrics['N_effective'] = metadata.get('N_effective')
                        return metrics
                except Exception as e:
                    logger.debug(f"Failed to load previous run from {latest['path']}: {e}")
                    return None
        except Exception as e:
            logger.debug(f"Failed to query index for previous run: {e}")
            return None
    
    def _compute_drift(
        self,
        prev_run: Dict[str, Any],
        curr_run: Dict[str, Any],
        cohort_metadata: Dict[str, Any],
        stage: str,
        item_name: str,
        route_type: Optional[str],
        symbol: Optional[str],
        model_family: Optional[str],
        cohort_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        Compute drift comparison and return drift.json data.
        
        Explicitly links both runs (current + previous) for self-contained drift.json.
        """
        prev_n = prev_run.get('N_effective_cs') or prev_run.get('N_effective') or prev_run.get('n_samples', 0)
        curr_n = cohort_metadata.get('N_effective_cs', 0)
        
        n_ratio = min(prev_n, curr_n) / max(prev_n, curr_n) if max(prev_n, curr_n) > 0 else 0.0
        
        # Extract previous run metadata
        prev_run_id = prev_run.get('run_id') or prev_run.get('timestamp', 'unknown')
        prev_cohort_id = prev_run.get('cohort_id') or prev_run.get('cohort_metadata', {}).get('cohort_id', 'unknown')
        prev_auc = float(prev_run.get('mean_score', 0.0))
        
        # Current run metadata
        curr_auc = float(curr_run.get('mean_score', 0.0))
        
        if n_ratio < self.n_ratio_threshold:
            # Defensive: handle both string and Enum for route_type/stage
            route_type_str = None
            if route_type:
                if isinstance(route_type, str):
                    route_type_str = route_type.upper()
                elif hasattr(route_type, 'value'):
                    route_type_str = route_type.value.upper() if isinstance(route_type.value, str) else str(route_type.value).upper()
                else:
                    route_type_str = str(route_type).upper()
            
            stage_str = stage
            if isinstance(stage, str):
                stage_str = stage.upper().replace("MODEL_TRAINING", "TRAINING")
            elif hasattr(stage, 'value'):
                stage_str = stage.value.upper().replace("MODEL_TRAINING", "TRAINING") if isinstance(stage.value, str) else str(stage.value).upper()
            else:
                stage_str = str(stage).upper().replace("MODEL_TRAINING", "TRAINING")
            
            return {
                "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
                "stage": stage_str,
                "route_type": route_type_str,
                "target": item_name,
                "symbol": symbol,
                "model_family": model_family,
                "current": {
                    "run_id": run_id,
                    "cohort_id": cohort_id,
                    "N_effective": curr_n,
                    "auc": curr_auc
                },
                "previous": {
                    "run_id": prev_run_id,
                    "cohort_id": prev_cohort_id,
                    "N_effective": prev_n,
                    "auc": prev_auc
                },
                "status": "INCOMPARABLE",
                "reason": f"N_effective ratio={n_ratio:.3f} ({prev_n} vs {curr_n}) < {self.n_ratio_threshold}",
                "n_ratio": n_ratio,
                "threshold": self.n_ratio_threshold,
                "created_at": datetime.now().isoformat()
            }
        
        # Sample-adjusted comparison
        prev_std = float(prev_run.get('std_score', 0.0)) if prev_run.get('std_score') else None
        
        classification, abs_diff, rel_diff, z_score, stats = self._compare_within_cohort(
            prev_run, curr_run, 'roc_auc'
        )
        
        # Build status label
        if stats.get('sample_adjusted', False):
            status_label = f"{classification}_SAMPLE_ADJUSTED"
        else:
            status_label = classification
        
        # Build reason string
        if z_score is not None:
            reason = f"n_ratio={n_ratio:.3f}, |z|={abs(z_score):.2f}"
            if z_score < 1.0:
                reason += " → stable"
            elif z_score < 2.0:
                reason += " → drifting"
            else:
                reason += " → diverged"
        else:
            reason = f"n_ratio={n_ratio:.3f}, abs_diff={abs_diff:.4f}"
        
        # Defensive: handle both string and Enum for route_type
        route_type_str = None
        if route_type:
            if isinstance(route_type, str):
                route_type_str = route_type.upper()
            elif hasattr(route_type, 'value'):
                route_type_str = route_type.value.upper() if isinstance(route_type.value, str) else str(route_type.value).upper()
            else:
                route_type_str = str(route_type).upper()
        
        # Defensive: handle both string and Enum for stage
        stage_str = stage
        if isinstance(stage, str):
            stage_str = stage.upper().replace("MODEL_TRAINING", "TRAINING")
        elif hasattr(stage, 'value'):
            stage_str = stage.value.upper().replace("MODEL_TRAINING", "TRAINING") if isinstance(stage.value, str) else str(stage.value).upper()
        else:
            stage_str = str(stage).upper().replace("MODEL_TRAINING", "TRAINING")
        
        return {
            "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
            "stage": stage_str,
            "route_type": route_type_str,
            "target": item_name,
            "symbol": symbol,
            "model_family": model_family,
            "current": {
                "run_id": run_id,
                "cohort_id": cohort_id,
                "N_effective": curr_n,
                "auc": curr_auc
            },
            "previous": {
                "run_id": prev_run_id,
                "cohort_id": prev_cohort_id,
                "N_effective": prev_n,
                "auc": prev_auc
            },
            "delta_auc": curr_auc - prev_auc,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "z_score": z_score,
            "status": status_label,
            "reason": reason,
            "n_ratio": n_ratio,
            "sample_adjusted": stats.get('sample_adjusted', False),
            "created_at": datetime.now().isoformat()
        }
    
    def log_comparison(
        self,
        stage: str,
        item_name: str,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None,
        route_type: Optional[str] = None,
        symbol: Optional[str] = None,
        model_family: Optional[str] = None,
        cohort_metadata: Optional[Dict[str, Any]] = None  # NEW: Allow passing pre-extracted cohort_metadata
    ) -> None:
        """
        Compare current run to previous run and log the comparison for reproducibility verification.
        
        Uses tolerance bands with STABLE/DRIFTING/DIVERGED classification. Only escalates to
        warnings for meaningful differences (DIVERGED).
        
        If cohort_aware=True and cohort metadata is available, organizes runs by cohort and
        only compares within the same cohort using sample-adjusted statistics.
        
        This method should never raise exceptions - all errors are logged and handled gracefully.
        
        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            item_name: Name of the item (e.g., target name, symbol name)
            metrics: Dictionary of metrics to track and compare
            additional_data: Optional additional data to store with the run
            route_type: Optional route type (e.g., "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", "INDIVIDUAL")
            symbol: Optional symbol name (for symbol-specific views)
            model_family: Optional model family (for training stage)
        """
        try:
            # Extract cohort metadata if available
            # CRITICAL: If cohort_metadata is already provided (e.g., from log_run()), use it directly
            # This avoids redundant extraction and ensures we use the same metadata that was extracted from RunContext
            if cohort_metadata is None:
                try:
                    cohort_metadata = self._extract_cohort_metadata(metrics, additional_data)
                except Exception as e:
                    logger.warning(f"Failed to extract cohort metadata for {stage}:{item_name}: {e}. Falling back to legacy mode.")
                    logger.debug(f"Cohort metadata extraction traceback: {traceback.format_exc()}")
                    cohort_metadata = None
            
            # Cohort-aware mode is the default - use it if enabled and metadata is available
            use_cohort_aware = self.cohort_aware and cohort_metadata is not None
            if self.cohort_aware and not use_cohort_aware:
                # Use INFO level so it's visible - this is important for debugging
                main_logger = _get_main_logger()
                msg = (f"⚠️  Reproducibility: Cohort-aware mode enabled (default) but insufficient metadata for {stage}:{item_name}. "
                       f"Falling back to legacy mode. "
                       f"Metrics keys: {list(metrics.keys())}, "
                       f"Additional data keys: {list(additional_data.keys()) if additional_data else 'None'}. "
                       f"To enable cohort-aware mode, pass N_effective_cs, n_symbols, date_range, and cs_config in metrics/additional_data.")
                if main_logger != logger:
                    main_logger.info(msg)
                else:
                    logger.info(msg)
            elif use_cohort_aware:
                # Log when cohort-aware mode is successfully used
                main_logger = _get_main_logger()
                n_info = f"N={cohort_metadata.get('N_effective_cs', '?')}, symbols={cohort_metadata.get('n_symbols', '?')}"
                msg = f"✅ Reproducibility: Using cohort-aware mode for {stage}:{item_name} ({n_info})"
                if main_logger != logger:
                    main_logger.debug(msg)
                else:
                    logger.debug(msg)
            
            # Extract route_type, symbol, model_family
            # Use provided parameters if available, otherwise extract from additional_data
            # For TARGET_RANKING, route_type comes from "view" field in additional_data
            # For FEATURE_SELECTION, map view to route_type (CROSS_SECTIONAL → CROSS_SECTIONAL, SYMBOL_SPECIFIC → INDIVIDUAL)
            if route_type is None:
                if stage.upper() == "TARGET_RANKING":
                    route_type = additional_data.get("view") if additional_data else None
                    if route_type:
                        route_type = route_type.upper()  # Normalize to uppercase
                elif stage.upper() == "FEATURE_SELECTION":
                    # FIX: Map view to route_type for FEATURE_SELECTION (ensures proper metrics scoping)
                    view = additional_data.get("view") if additional_data else None
                    if view:
                        if view.upper() == "CROSS_SECTIONAL":
                            route_type = "CROSS_SECTIONAL"
                        elif view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
                            route_type = "INDIVIDUAL"  # SYMBOL_SPECIFIC maps to INDIVIDUAL for FEATURE_SELECTION
                    if not route_type:
                        # Fallback to extraction method
                        route_type = self._extract_route_type(additional_data)
                else:
                    route_type = self._extract_route_type(additional_data) if stage.lower() in ["feature_selection", "model_training", "training"] else None
            
            if symbol is None:
                symbol = self._extract_symbol(additional_data)
            
            if model_family is None:
                model_family = self._extract_model_family(additional_data)
            
            if use_cohort_aware:
                # Cohort-aware path: find matching cohort
                main_logger = _get_main_logger()
                n_info = f"N={cohort_metadata.get('N_effective_cs', '?')}, symbols={cohort_metadata.get('n_symbols', '?')}"
                if main_logger != logger:
                    main_logger.debug(f"🔍 Reproducibility: Searching for matching cohort for {stage}:{item_name} ({n_info})")
                else:
                    logger.debug(f"🔍 Reproducibility: Searching for matching cohort for {stage}:{item_name} ({n_info})")
                
                cohort_id = self._find_matching_cohort(stage, item_name, cohort_metadata, route_type, symbol, model_family)
                
                if cohort_id is None:
                    # New cohort - save as baseline
                    cohort_id = self._compute_cohort_id(cohort_metadata, route_type)
                    run_data = {
                        "timestamp": datetime.now().isoformat(),
                        "stage": stage,
                        "item_name": item_name,
                        **{k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in metrics.items()},
                        "cohort_metadata": cohort_metadata
                    }
                    if additional_data:
                        run_data["additional_data"] = additional_data
                    
                    # FIX: Pass symbol and model_family to _save_to_cohort so symbol subdirectory is created
                    self._save_to_cohort(stage, item_name, cohort_id, cohort_metadata, run_data, route_type, symbol, model_family, additional_data)
                    self._increment_mode_counter("COHORT_AWARE")
                    
                    main_logger = _get_main_logger()
                    n_info = f"N={cohort_metadata['N_effective_cs']}, symbols={cohort_metadata['n_symbols']}"
                    if cohort_metadata.get('date_range', {}).get('start_ts'):
                        date_info = f", date_range={cohort_metadata['date_range']['start_ts']}→{cohort_metadata['date_range'].get('end_ts', '')}"
                    else:
                        date_info = ""
                    
                    msg = f"📊 Reproducibility: First run for {stage}:{item_name} (new cohort: {n_info}{date_info})"
                    if main_logger != logger:
                        main_logger.info(msg)
                    else:
                        logger.info(msg)
                    return
                
                # Load previous run from index (only same cohort)
                previous = self.get_last_comparable_run(
                    stage=stage,
                    item_name=item_name,
                    route_type=route_type,
                    symbol=symbol,
                    model_family=model_family,
                    cohort_id=cohort_id,  # Key: only same cohort
                    current_N=cohort_metadata.get('N_effective_cs', 0),
                    n_ratio_threshold=self.n_ratio_threshold
                )
                
                if previous is None:
                    # First run in this cohort
                    run_data = {
                        "timestamp": datetime.now().isoformat(),
                        "stage": stage,
                        "item_name": item_name,
                        **{k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in metrics.items()},
                        "cohort_metadata": cohort_metadata
                    }
                    if additional_data:
                        run_data["additional_data"] = additional_data
                    
                    self._save_to_cohort(stage, item_name, cohort_id, cohort_metadata, run_data, route_type, symbol, model_family, additional_data)
                    self._increment_mode_counter("COHORT_AWARE")
                    
                    main_logger = _get_main_logger()
                    n_info = f"N={cohort_metadata['N_effective_cs']}, symbols={cohort_metadata['n_symbols']}"
                    route_info = f" [{route_type}]" if route_type else ""
                    symbol_info = f" symbol={symbol}" if symbol else ""
                    model_info = f" model={model_family}" if model_family else ""
                    msg = f"📊 Reproducibility: First run in cohort for {stage}:{item_name}{route_info}{symbol_info}{model_info} ({n_info})"
                    if main_logger != logger:
                        main_logger.info(msg)
                    else:
                        logger.info(msg)
                    return
                
                # Extract metrics for comparison (only reached if previous exists)
                metric_name = metrics.get("metric_name", "Score")
                current_mean = float(metrics.get("mean_score", 0.0))
                previous_mean = float(previous.get("mean_score", 0.0))
                
                current_std = float(metrics.get("std_score", 0.0))
                previous_std = float(previous.get("std_score", 0.0))
                
                # Compare importance if present
                current_importance = float(metrics.get("mean_importance", 0.0))
                previous_importance = float(previous.get("mean_importance", 0.0))
                
                # Compare composite score if present
                current_composite = float(metrics.get("composite_score", current_mean))
                previous_composite = float(previous.get("composite_score", previous_mean))
                
                # Compute route_changed and route_entropy for regression tracking
                prev_route = previous.get('route') or previous.get('route_type') or previous.get('view') or previous.get('mode')
                curr_route = additional_data.get('route') if additional_data else None
                if curr_route is None:
                    curr_route = route_type or additional_data.get('view') if additional_data else None
                route_changed = 1 if (prev_route and curr_route and prev_route != curr_route) else 0
                
                # Compute route_entropy from route history (if we have access to index)
                route_entropy = None
                try:
                    repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
                    index_file = repro_dir / "index.parquet"
                    if index_file.exists():
                        df = pd.read_parquet(index_file)
                        # Get route history for this cohort/target
                        cohort_id = self._compute_cohort_id(cohort_metadata, route_type)
                        mask = (df['cohort_id'] == cohort_id) & (df['target'] == item_name)
                        if route_type:
                            mask &= (df['mode'] == route_type.upper())
                        route_history = df[mask]['route'].dropna().tolist()
                        if len(route_history) >= 3:
                            # Compute entropy: -sum(p * log2(p))
                            from collections import Counter
                            route_counts = Counter(route_history)
                            total = len(route_history)
                            entropy = -sum((count / total) * math.log2(count / total) 
                                         for count in route_counts.values() if count > 0)
                            route_entropy = float(entropy)
                except Exception:
                    pass  # Non-critical, continue without entropy
                
                # Use sample-adjusted comparison if cohort-aware and within same cohort
                # Prepare current run data for comparison
                # Store route_changed and route_entropy in metrics for index update
                if route_changed is not None:
                    metrics['route_changed'] = route_changed
                if route_entropy is not None:
                    metrics['route_entropy'] = route_entropy
                if curr_route:
                    metrics['route'] = curr_route
                
                curr_run_data = {
                    **metrics,
                    'N_effective_cs': cohort_metadata.get('N_effective_cs'),
                    'n_samples': cohort_metadata.get('N_effective_cs'),
                    'sample_size': cohort_metadata.get('N_effective_cs'),
                    'route': curr_route,
                    'route_changed': route_changed,
                    'route_entropy': route_entropy
                }
                prev_run_data = {
                    **previous,
                    'N_effective_cs': previous.get('cohort_metadata', {}).get('N_effective_cs') or previous.get('N_effective_cs'),
                    'n_samples': previous.get('cohort_metadata', {}).get('N_effective_cs') or previous.get('n_samples'),
                    'sample_size': previous.get('cohort_metadata', {}).get('N_effective_cs') or previous.get('sample_size'),
                    'route': prev_route
                }
                
                mean_class, mean_abs, mean_rel, mean_z, mean_stats = self._compare_within_cohort(
                    prev_run_data, curr_run_data, 'roc_auc'
                )
                composite_class, composite_abs, composite_rel, composite_z, _ = self._compare_within_cohort(
                    prev_run_data, curr_run_data, 'composite'
                )
                importance_class, importance_abs, importance_rel, importance_z, _ = self._compare_within_cohort(
                    prev_run_data, curr_run_data, 'importance'
                )
            else:
                # Legacy path: use flat structure
                main_logger = _get_main_logger()
                if main_logger != logger:
                    main_logger.info(f"📋 Reproducibility: Using legacy mode for {stage}:{item_name} (files in {self.log_file.parent.name}/)")
                else:
                    logger.info(f"📋 Reproducibility: Using legacy mode for {stage}:{item_name} (files in {self.log_file.parent.name}/)")
                
                previous = self.load_previous_run(stage, item_name)
                
                if previous is None:
                    # Use main logger if available for better visibility
                    main_logger = _get_main_logger()
                    # Only log once - use main logger if available, otherwise use module logger
                    if main_logger != logger:
                        main_logger.info(f"📊 Reproducibility: First run for {stage}:{item_name} (no previous run to compare)")
                    else:
                        logger.info(f"📊 Reproducibility: First run for {stage}:{item_name} (no previous run to compare)")
                    # Save current run for next time
                    self.save_run(stage, item_name, metrics, additional_data)
                    self._increment_mode_counter("LEGACY")
                    return
                
                # Extract metrics for comparison (only reached if previous exists)
                metric_name = metrics.get("metric_name", "Score")
                current_mean = float(metrics.get("mean_score", 0.0))
                previous_mean = float(previous.get("mean_score", 0.0))
                
                current_std = float(metrics.get("std_score", 0.0))
                previous_std = float(previous.get("std_score", 0.0))
                
                # Compare importance if present
                current_importance = float(metrics.get("mean_importance", 0.0))
                previous_importance = float(previous.get("mean_importance", 0.0))
                
                # Compare composite score if present
                current_composite = float(metrics.get("composite_score", current_mean))
                previous_composite = float(previous.get("composite_score", previous_mean))
                
                # Legacy comparison
                mean_class, mean_abs, mean_rel, mean_z = self._classify_diff(
                    previous_mean, current_mean, previous_std, 'roc_auc'
                )
                composite_class, composite_abs, composite_rel, composite_z = self._classify_diff(
                    previous_composite, current_composite, None, 'composite'
                )
                importance_class, importance_abs, importance_rel, importance_z = self._classify_diff(
                    previous_importance, current_importance, None, 'importance'
                )
                mean_stats = {}
            
            # Overall classification: use worst case
            if 'DIVERGED' in [mean_class, composite_class, importance_class]:
                overall_class = 'DIVERGED'
            elif 'DRIFTING' in [mean_class, composite_class, importance_class]:
                overall_class = 'DRIFTING'
            else:
                overall_class = 'STABLE'
            
            # Determine log level and emoji based on classification
            if overall_class == 'STABLE':
                log_level = logger.info
                emoji = "ℹ️"
            elif overall_class == 'DRIFTING':
                log_level = logger.info
                emoji = "ℹ️"
            else:  # DIVERGED
                log_level = logger.warning
                emoji = "⚠️"
            
            # Use main logger if available for better visibility
            main_logger = _get_main_logger()
            
            # Build comparison log message
            mean_diff = current_mean - previous_mean
            composite_diff = current_composite - previous_composite
            importance_diff = current_importance - previous_importance
            
            # Format z-score if available
            z_info = ""
            if mean_z is not None:
                z_info = f", z={mean_z:.2f}"
            
            # Main status line
            cohort_info = ""
            if use_cohort_aware and cohort_metadata:
                n_info = f"N={cohort_metadata['N_effective_cs']}, symbols={cohort_metadata['n_symbols']}"
                if mean_stats.get('sample_adjusted'):
                    cohort_info = f" [cohort: {n_info}, sample-adjusted]"
                else:
                    cohort_info = f" [cohort: {n_info}]"
            
            status_msg = f"{emoji} Reproducibility: {overall_class}{cohort_info}"
            if overall_class == 'STABLE':
                status_msg += f" (Δ {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); within tolerance)"
            elif overall_class == 'DRIFTING':
                status_msg += f" (Δ {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); small drift detected)"
            else:  # DIVERGED
                status_msg += f" (Δ {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); exceeds tolerance)"
            
            # Only log once - use main logger if available, otherwise use module logger
            if main_logger != logger:
                if overall_class == 'DIVERGED':
                    main_logger.warning(status_msg)
                else:
                    main_logger.info(status_msg)
            else:
                log_level(status_msg)
            
            # Detailed comparison (always log for traceability)
            prev_n_info = ""
            curr_n_info = ""
            if use_cohort_aware and cohort_metadata:
                prev_n = previous.get('cohort_metadata', {}).get('N_effective_cs') or previous.get('N_effective_cs') or previous.get('n_samples')
                curr_n = cohort_metadata.get('N_effective_cs')
                if prev_n:
                    prev_n_info = f", N={int(prev_n)}"
                if curr_n:
                    curr_n_info = f", N={int(curr_n)}"
            
            prev_msg = f"   Previous: {metric_name}={previous_mean:.3f}±{previous_std:.3f}{prev_n_info}, " \
                       f"importance={previous_importance:.2f}, composite={previous_composite:.3f}"
            if main_logger != logger:
                main_logger.info(prev_msg)
            else:
                logger.info(prev_msg)
            
            curr_msg = f"   Current:  {metric_name}={current_mean:.3f}±{current_std:.3f}{curr_n_info}, " \
                       f"importance={current_importance:.2f}, composite={current_composite:.3f}"
            if main_logger != logger:
                main_logger.info(curr_msg)
            else:
                logger.info(curr_msg)
            
            # Diff line with classifications
            diff_parts = [f"{metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{', z=' + f'{mean_z:.2f}' if mean_z else ''}) [{mean_class}]"]
            diff_parts.append(f"composite={composite_diff:+.4f} ({composite_rel:+.2f}%) [{composite_class}]")
            diff_parts.append(f"importance={importance_diff:+.2f} ({importance_rel:+.2f}%) [{importance_class}]")
            diff_msg = f"   Diff:     {', '.join(diff_parts)}"
            if main_logger != logger:
                main_logger.info(diff_msg)
            else:
                logger.info(diff_msg)
            
            # Warning only for DIVERGED
            if overall_class == 'DIVERGED':
                warn_msg = f"   ⚠️  Results differ significantly from previous run - check for non-deterministic behavior, config changes, or data differences"
                if main_logger != logger:
                    main_logger.warning(warn_msg)
                else:
                    logger.warning(warn_msg)
            
            # Save current run for next time
            if use_cohort_aware:
                run_data = {
                    "timestamp": datetime.now().isoformat(),
                    "stage": stage,
                    "item_name": item_name,
                    **{k: float(v) if isinstance(v, (int, float)) else v 
                       for k, v in metrics.items()},
                    "cohort_metadata": cohort_metadata
                }
                if additional_data:
                    run_data["additional_data"] = additional_data
                
                cohort_id = self._compute_cohort_id(cohort_metadata, route_type)
                self._save_to_cohort(stage, item_name, cohort_id, cohort_metadata, run_data, route_type, symbol, model_family, additional_data)
                self._increment_mode_counter("COHORT_AWARE")
                
                # Compute trend analysis for this series (if enough runs exist)
                trend_metadata = None
                try:
                    if _AUDIT_AVAILABLE:
                        from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView
                        
                        # Get reproducibility base directory
                        repro_base = cohort_dir.parent.parent.parent
                        if repro_base.exists():
                            trend_analyzer = TrendAnalyzer(
                                reproducibility_dir=repro_base,
                                half_life_days=7.0,
                                min_runs_for_trend=2  # Minimum 2 runs for trend (slope requires 2 points)
                            )
                            
                            # Analyze STRICT series
                            all_trends = trend_analyzer.analyze_all_series(view=SeriesView.STRICT)
                            
                            # Find trend for this series
                            for series_key_str, trend_list in all_trends.items():
                                # Check if this series matches
                                if any(t.series_key.target == item_name and 
                                       t.series_key.stage == stage_normalized for t in trend_list):
                                    # Write trend.json to cohort directory (similar to metadata.json and metrics.json)
                                    if cohort_dir and cohort_dir.exists():
                                        try:
                                            trend_analyzer.write_cohort_trend(
                                                cohort_dir=cohort_dir,
                                                stage=stage_normalized,
                                                target=item_name,
                                                trends={series_key_str: trend_list}  # Pass pre-computed trends
                                            )
                                        except Exception as e:
                                            logger.debug(f"Failed to write trend.json: {e}")
                                    
                                    # Find trend for primary metric
                                    primary_metric = metrics.get("metric_name", "mean_score")
                                    for trend in trend_list:
                                        if trend.metric_name in ["auc_mean", "mean_score", primary_metric.lower()] if primary_metric else True:
                                            if trend.status == "ok":
                                                slope_str = f"{trend.slope_per_day:+.6f}" if trend.slope_per_day else "N/A"
                                                main_logger = _get_main_logger()
                                                trend_msg = (
                                                    f"📈 Trend ({trend.metric_name}): "
                                                    f"slope={slope_str}/day, "
                                                    f"current={trend.current_estimate:.4f}, "
                                                    f"ewma={trend.ewma_value:.4f}, "
                                                    f"n={trend.n_runs} runs"
                                                )
                                                if main_logger != logger:
                                                    main_logger.info(trend_msg)
                                                else:
                                                    logger.info(trend_msg)
                                                
                                                # Log trend alerts
                                                if trend.alerts:
                                                    for alert in trend.alerts:
                                                        alert_msg = f"  {'⚠️' if alert.get('severity') == 'warning' else 'ℹ️'}  {alert['message']}"
                                                        if main_logger != logger:
                                                            (main_logger.warning if alert.get('severity') == 'warning' else main_logger.info)(alert_msg)
                                                        else:
                                                            (logger.warning if alert.get('severity') == 'warning' else logger.info)(alert_msg)
                                                
                                                # Store trend metadata for inclusion in metadata.json
                                                trend_metadata = {
                                                    "enabled": True,
                                                    "view": "STRICT",
                                                    "series_key": series_key_str[:100],  # Truncate for readability
                                                    "metric_name": trend.metric_name,
                                                    "n_runs": trend.n_runs,
                                                    "status": trend.status,
                                                    "slope_per_day": trend.slope_per_day,
                                                    "current_estimate": trend.current_estimate,
                                                    "ewma_value": trend.ewma_value,
                                                    "residual_std": trend.residual_std,
                                                    "half_life_days": trend.half_life_days,
                                                    "n_alerts": len(trend.alerts),
                                                    "applied": False  # Currently only logged, not used for decisions
                                                }
                                            break
                                    break
                except Exception as e:
                    logger.debug(f"Could not compute trend analysis: {e}")
                
                # Compute and save drift.json if previous run exists
                if previous:
                    run_id_clean = run_data.get('run_id') or run_data.get('timestamp', datetime.now().isoformat()).replace(':', '-').replace('.', '-').replace('T', '_')
                    try:
                        drift_data = self._compute_drift(
                            previous, run_data, cohort_metadata,
                            stage, item_name, route_type, symbol, model_family,
                            cohort_id, run_id_clean
                        )
                        cohort_dir = self._get_cohort_dir(stage, item_name, cohort_id, route_type, symbol, model_family)
                        drift_file = cohort_dir / "drift.json"
                        try:
                            with open(drift_file, 'w') as f:
                                json.dump(drift_data, f, indent=2)
                                f.flush()  # Ensure immediate write
                                os.fsync(f.fileno())  # Force write to disk
                        except (IOError, OSError) as e:
                            logger.warning(f"Failed to save drift.json to {drift_file}: {e}, error_type=IO_ERROR")
                            self._increment_error_counter("write_failures", "IO_ERROR")
                            # Don't re-raise - drift file failure shouldn't break the run
                    except Exception as e:
                        logger.warning(f"Failed to compute drift for {stage}:{item_name}: {e}")
                        logger.debug(f"Drift computation traceback: {traceback.format_exc()}")
            else:
                # CRITICAL: Even in legacy mode, try to write metadata.json and metrics.json to cohort directory
                # This ensures files are written for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
                # Build minimal cohort metadata from available data
                minimal_cohort_metadata = {}
                if metrics:
                    # Try to extract N_effective from metrics
                    n_effective = metrics.get('N_effective_cs') or metrics.get('n_samples') or metrics.get('sample_size')
                    if n_effective:
                        minimal_cohort_metadata['N_effective_cs'] = int(n_effective)
                
                if additional_data:
                    n_symbols = additional_data.get('n_symbols')
                    if n_symbols:
                        minimal_cohort_metadata['n_symbols'] = int(n_symbols)
                    
                    # Extract date range if available
                    date_range = {}
                    if 'date_range' in additional_data:
                        date_range = additional_data['date_range']
                    elif 'start_ts' in additional_data or 'end_ts' in additional_data:
                        date_range = {
                            'start_ts': additional_data.get('start_ts'),
                            'end_ts': additional_data.get('end_ts')
                        }
                    if date_range:
                        minimal_cohort_metadata['date_range'] = date_range
                    
                    # Extract cs_config if available
                    if 'cs_config' in additional_data:
                        minimal_cohort_metadata['cs_config'] = additional_data['cs_config']
                    elif 'min_cs' in additional_data or 'max_cs_samples' in additional_data:
                        minimal_cohort_metadata['cs_config'] = {
                            'min_cs': additional_data.get('min_cs'),
                            'max_cs_samples': additional_data.get('max_cs_samples'),
                            'leakage_filter_version': additional_data.get('leakage_filter_version', 'v1')
                        }
                
                # If we have minimal cohort metadata, try to write to cohort directory
                if minimal_cohort_metadata.get('N_effective_cs'):
                    try:
                        # Compute cohort_id from minimal metadata
                        minimal_cohort_id = self._compute_cohort_id(minimal_cohort_metadata, route_type)
                        run_data = {
                            "timestamp": datetime.now().isoformat(),
                            "stage": stage,
                            "item_name": item_name,
                            **{k: float(v) if isinstance(v, (int, float)) else v 
                               for k, v in metrics.items()},
                            "cohort_metadata": minimal_cohort_metadata
                        }
                        if additional_data:
                            run_data["additional_data"] = additional_data
                        
                        # Try to write to cohort directory (even with minimal metadata)
                        self._save_to_cohort(stage, item_name, minimal_cohort_id, minimal_cohort_metadata, run_data, route_type, symbol, model_family, additional_data)
                        self._increment_mode_counter("LEGACY_WITH_COHORT_WRITE")
                        logger.info(f"📊 Reproducibility: Wrote metadata.json/metrics.json to cohort directory (legacy mode with minimal metadata)")
                    except Exception as e:
                        # If cohort write fails, fall back to legacy save_run
                        logger.warning(f"Failed to write to cohort directory in legacy mode: {e}. Falling back to legacy save_run.")
                        logger.debug(f"Cohort write traceback: {traceback.format_exc()}")
                        self.save_run(stage, item_name, metrics, additional_data)
                        self._increment_mode_counter("LEGACY")
                else:
                    # No cohort metadata available - use legacy save_run
                    self.save_run(stage, item_name, metrics, additional_data)
                    self._increment_mode_counter("LEGACY")
        except Exception as e:
            # Final safety net - ensure log_comparison never raises
            error_type = "IO_ERROR" if isinstance(e, (IOError, OSError)) else "SERIALIZATION_ERROR" if isinstance(e, (json.JSONDecodeError, TypeError)) else "UNKNOWN_ERROR"
            
            logger.error(
                f"Reproducibility tracking failed completely for {stage}:{item_name}. "
                f"error_type={error_type}, reason={str(e)}"
            )
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Update stats counter
            self._increment_error_counter("total_failures", error_type)
            
            # Don't re-raise - reproducibility tracking should never break the main pipeline
    
    def log_run(
        self,
        ctx: Any,  # RunContext (using Any to avoid circular import issues)
        metrics: Dict[str, Any],
        additional_data_override: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Automated audit-grade reproducibility tracking using RunContext.
        
        This is the recommended API for new code. It:
        1. Extracts all metadata from RunContext automatically
        2. Validates with AuditEnforcer
        3. Saves metadata/metrics
        4. Compares to previous run and writes audit report
        
        Args:
            ctx: RunContext containing all run data and configuration
            metrics: Dictionary of metrics (mean_score, std_score, etc.)
        
        Returns:
            Dict with audit_report and saved metadata paths
        
        Raises:
            ValueError: If required fields are missing (in COHORT_AWARE mode) or audit validation fails (in strict mode)
        """
        if not _AUDIT_AVAILABLE:
            # Fallback to legacy API
            logger.warning("RunContext not available, falling back to legacy log_comparison API")
            self.log_comparison(
                stage=ctx.stage,
                item_name=ctx.target_name or ctx.target_column or "unknown",
                metrics=metrics,
                additional_data=ctx.to_dict()
            )
            return {"mode": "legacy_fallback"}
        
        # 1. Validate required fields
        # FIX: If required fields are missing in COHORT_AWARE mode, downgrade to NON_COHORT and log warning
        if self.cohort_aware:
            missing = ctx.validate_required_fields("COHORT_AWARE")
            if missing:
                logger.warning(
                    f"⚠️  Missing required fields for COHORT_AWARE mode: {missing}. "
                    f"Downgrading to NON_COHORT mode for this run. "
                    f"RunContext should contain: {ctx.get_required_fields('COHORT_AWARE')}"
                )
                # Downgrade to NON_COHORT mode for this run (don't fail)
                use_cohort_aware = False
            else:
                use_cohort_aware = True
        else:
            use_cohort_aware = False
        
        # 2. Auto-derive purge/embargo if not set
        if ctx.purge_minutes is None and ctx.horizon_minutes is not None:
            purge_min, embargo_min = ctx.derive_purge_embargo()
            ctx.purge_minutes = purge_min
            if ctx.embargo_minutes is None:
                ctx.embargo_minutes = embargo_min
            logger.info(f"Auto-derived purge={purge_min:.1f}m, embargo={embargo_min:.1f}m from horizon={ctx.horizon_minutes}m")
        
        # 3. Extract metadata from RunContext (only if use_cohort_aware is True)
        from TRAINING.orchestration.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
        
        if use_cohort_aware:
            cohort_metadata = extract_cohort_metadata(
                X=ctx.X,
                y=ctx.y,
                symbols=ctx.symbols,
                time_vals=ctx.time_vals,
                mtf_data=ctx.mtf_data,
                min_cs=ctx.min_cs,
                max_cs_samples=ctx.max_cs_samples,
                leakage_filter_version=ctx.leakage_filter_version,
                universe_id=ctx.universe_id,
                compute_data_fingerprint=True,
                compute_per_symbol_stats=True
            )
            
            # Format for tracker
            cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
        else:
            # NON_COHORT mode: use minimal metadata
            cohort_metrics = {}
            cohort_additional_data = {}
        
        # Build additional_data with CV details
        additional_data = {
            **cohort_additional_data,
            "cv_method": ctx.cv_method,
            "cv_folds": ctx.cv_folds,
            "horizon_minutes": ctx.horizon_minutes,
            "purge_minutes": ctx.purge_minutes,
            "embargo_minutes": ctx.embargo_minutes,
            "feature_lookback_max_minutes": ctx.feature_lookback_max_minutes,
            "data_interval_minutes": ctx.data_interval_minutes,
            "feature_names": ctx.feature_names,
            "seed": ctx.seed,
            "train_seed": ctx.seed  # Also pass as train_seed for FEATURE_SELECTION/TRAINING
        }
        
        # Merge additional_data_override if provided (e.g., hyperparameters for FEATURE_SELECTION)
        if additional_data_override:
            additional_data.update(additional_data_override)
        
        # Add fold timestamps if available
        if ctx.fold_timestamps:
            additional_data["fold_timestamps"] = ctx.fold_timestamps
        
        # Add label definition hash
        if ctx.target_column:
            label_def_str = f"{ctx.target_column}|{ctx.target_name or ctx.target_column}"
            additional_data["label_definition_hash"] = hashlib.sha256(label_def_str.encode()).hexdigest()[:16]
        
        # Add view metadata for TARGET_RANKING
        # FIX: Add view to additional_data for both TARGET_RANKING and FEATURE_SELECTION
        # This ensures proper metrics scoping (features compared per-target, per-view, per-symbol)
        if hasattr(ctx, 'view') and ctx.view:
            additional_data["view"] = ctx.view
        # Also add symbol for SYMBOL_SPECIFIC/INDIVIDUAL views
        if hasattr(ctx, 'symbol') and ctx.symbol:
            additional_data["symbol"] = ctx.symbol
        
        # Merge metrics
        metrics_with_cohort = {**metrics, **cohort_metrics}
        
        # 4. Load previous run metadata for comparison
        # FIX: For FEATURE_SELECTION, map view to route_type (ensures proper metrics scoping)
        route_type_for_cohort = ctx.route_type if hasattr(ctx, 'route_type') else None
        if ctx.stage == "target_ranking" and hasattr(ctx, 'view') and ctx.view:
            route_type_for_cohort = ctx.view
        elif ctx.stage == "feature_selection" and hasattr(ctx, 'view') and ctx.view:
            # Map view to route_type for FEATURE_SELECTION
            # FIX: Use SYMBOL_SPECIFIC directly (not INDIVIDUAL) to match directory structure
            if ctx.view.upper() == "CROSS_SECTIONAL":
                route_type_for_cohort = "CROSS_SECTIONAL"
            elif ctx.view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
                route_type_for_cohort = "SYMBOL_SPECIFIC"  # Use SYMBOL_SPECIFIC to match directory structure
        
        cohort_id = self._compute_cohort_id(cohort_metadata, route_type_for_cohort)
        previous_metadata = None
        try:
            cohort_dir = self._get_cohort_dir(
                ctx.stage,
                ctx.target_name or ctx.target_column or "unknown",
                cohort_id,
                route_type_for_cohort,  # Use view for TARGET_RANKING
                ctx.symbol,
                ctx.model_family
            )
            if cohort_dir.exists():
                metadata_file = cohort_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        previous_metadata = json.load(f)
        except Exception as e:
            logger.debug(f"Could not load previous metadata: {e}")
        
        # 5. Validate with AuditEnforcer (before saving)
        if self.audit_enforcer:
            # Build temporary metadata for validation
            temp_metadata = {
                "cohort_id": cohort_id,
                "data_fingerprint": cohort_metadata.get("data_fingerprint"),
                "feature_registry_hash": None,  # Will be computed in _save_to_cohort
                "cv_details": {
                    "cv_method": ctx.cv_method,
                    "horizon_minutes": ctx.horizon_minutes,
                    "purge_minutes": ctx.purge_minutes,
                    "embargo_minutes": ctx.embargo_minutes,
                    "folds": ctx.cv_folds,
                    "feature_lookback_max_minutes": ctx.feature_lookback_max_minutes
                }
            }
            if ctx.fold_timestamps:
                fold_str = json.dumps(ctx.fold_timestamps, sort_keys=True, default=str)
                temp_metadata["cv_details"]["fold_boundaries_hash"] = hashlib.sha256(fold_str.encode()).hexdigest()[:16]
            
            is_valid, audit_report = self.audit_enforcer.validate(temp_metadata, metrics_with_cohort, previous_metadata)
            
            if not is_valid and self.audit_enforcer.mode == AuditMode.STRICT:
                # Already raised by enforcer, but be explicit
                raise ValueError(f"Audit validation failed: {audit_report}")
        else:
            audit_report = {"mode": "off", "violations": [], "warnings": []}
        
        # 6. Save using existing log_comparison (which handles cohort-aware saving)
        # For TARGET_RANKING, pass view as route_type
        route_type_for_log = ctx.route_type
        if ctx.stage == "target_ranking" and hasattr(ctx, 'view') and ctx.view:
            route_type_for_log = ctx.view
        
        # CRITICAL: Wrap log_comparison in try/except to ensure we can still write audit report
        # even if log_comparison fails. log_comparison itself has exception handling, but
        # we want to be defensive here.
        try:
            # CRITICAL: Pass the already-extracted cohort_metadata directly to log_comparison()
            # This ensures we use the same metadata that was extracted from RunContext, avoiding redundant extraction
            self.log_comparison(
                stage=ctx.stage,
                item_name=ctx.target_name or ctx.target_column or "unknown",
                metrics=metrics_with_cohort,
                additional_data=additional_data,
                route_type=route_type_for_log,  # Use view for TARGET_RANKING
                symbol=ctx.symbol,
                cohort_metadata=cohort_metadata  # Pass pre-extracted cohort_metadata from RunContext
            )
        except Exception as e:
            # log_comparison should never raise (it has its own exception handling),
            # but if it does, log it and continue so we can still write audit report
            logger.error(f"log_comparison raised unexpected exception (this should not happen): {e}")
            logger.debug(f"log_comparison exception traceback: {traceback.format_exc()}")
            # Continue - we'll still write audit report below
        
        # 7. Write audit report
        audit_report_path = None
        cohort_dir = None
        try:
            # FIX: Use view as route_type for TARGET_RANKING and FEATURE_SELECTION when getting cohort directory
            route_type_for_cohort_dir = route_type_for_log  # Use same as log_comparison
            if ctx.stage == "target_ranking" and hasattr(ctx, 'view') and ctx.view:
                route_type_for_cohort_dir = ctx.view
            elif ctx.stage == "feature_selection" and hasattr(ctx, 'view') and ctx.view:
                # Map view to route_type for FEATURE_SELECTION
                # FIX: Use SYMBOL_SPECIFIC directly (not INDIVIDUAL) to match directory structure
                if ctx.view.upper() == "CROSS_SECTIONAL":
                    route_type_for_cohort_dir = "CROSS_SECTIONAL"
                elif ctx.view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
                    route_type_for_cohort_dir = "SYMBOL_SPECIFIC"  # Use SYMBOL_SPECIFIC to match directory structure
            
            cohort_dir = self._get_cohort_dir(
                ctx.stage,
                ctx.target_name or ctx.target_column or "unknown",
                cohort_id,
                route_type_for_cohort_dir,  # Use view for TARGET_RANKING
                ctx.symbol,
                ctx.model_family
            )
            # CRITICAL: Ensure cohort_dir exists (it should have been created by _save_to_cohort)
            if not cohort_dir.exists():
                cohort_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"⚠️  Cohort directory {cohort_dir.name}/ did not exist - created it. This may indicate _save_to_cohort() was not called.")
            
            # Verify metadata files exist (should have been written by _save_to_cohort)
            metadata_file = cohort_dir / "metadata.json"
            metrics_file = cohort_dir / "metrics.json"
            if not metadata_file.exists() or not metrics_file.exists():
                logger.warning(
                    f"⚠️  Metadata files missing in {cohort_dir.name}/: "
                    f"metadata.json={'missing' if not metadata_file.exists() else 'exists'}, "
                    f"metrics.json={'missing' if not metrics_file.exists() else 'exists'}. "
                    f"Attempting to write them now as fallback."
                )
                
                # CRITICAL FALLBACK: If _save_to_cohort() didn't write the files, write them here
                # This ensures metadata.json and metrics.json are always written
                try:
                    # Build minimal metadata from available data
                    # CRITICAL: Use NEW field names to match finalize_run() expectations
                    minimal_metadata = {
                        "schema_version": REPRODUCIBILITY_SCHEMA_VERSION,
                        "cohort_id": cohort_id,
                        "run_id": ctx.run_id if hasattr(ctx, 'run_id') else None,
                        "stage": ctx.stage,
                        "route_type": route_type_for_cohort_dir,
                        "view": ctx.view if hasattr(ctx, 'view') else None,
                        "target_name": ctx.target_name or ctx.target_column or "unknown",  # Changed from "target" to "target_name"
                        "n_effective": cohort_metadata.get('N_effective_cs', 0) if cohort_metadata else 0,  # Changed from "N_effective" to "n_effective"
                        "n_symbols": cohort_metadata.get('n_symbols', 0) if cohort_metadata else 0,
                        "date_range_start": cohort_metadata.get('date_range', {}).get('start_ts') if cohort_metadata else None,  # Changed from "date_start" to "date_range_start"
                        "date_range_end": cohort_metadata.get('date_range', {}).get('end_ts') if cohort_metadata else None,  # Changed from "date_end" to "date_range_end"
                        "created_at": datetime.now().isoformat()
                    }
                    
                    # Write metadata.json if missing
                    if not metadata_file.exists():
                        _write_atomic_json(metadata_file, minimal_metadata)
                        logger.info(f"✅ Wrote metadata.json (fallback) to {cohort_dir.name}/")
                    
                    # Write metrics.json if missing
                    if not metrics_file.exists() and self.metrics:
                        minimal_metrics = {
                            "run_id": minimal_metadata.get("run_id"),
                            "timestamp": minimal_metadata.get("created_at"),
                            "stage": ctx.stage,
                            **{k: v for k, v in metrics_with_cohort.items() if k not in ['timestamp', 'cohort_metadata', 'additional_data']}
                        }
                        self.metrics.write_cohort_metrics(
                            cohort_dir=cohort_dir,
                            stage=ctx.stage,
                            view=ctx.view if hasattr(ctx, 'view') else "UNKNOWN",
                            target=ctx.target_name or ctx.target_column or "unknown",
                            symbol=ctx.symbol if hasattr(ctx, 'symbol') else None,
                            run_id=minimal_metadata.get("run_id") or datetime.now().isoformat(),
                            metrics=minimal_metrics
                        )
                        logger.info(f"✅ Wrote metrics.json (fallback) to {cohort_dir.name}/")
                except Exception as e:
                    logger.error(f"❌ Failed to write fallback metadata/metrics: {e}")
                    logger.debug(f"Fallback write traceback: {traceback.format_exc()}")
                
                audit_report_path = cohort_dir / "audit_report.json"
                with open(audit_report_path, 'w') as f:
                    json.dump(audit_report, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as e:
            logger.debug(f"Could not write audit report: {e}")
        
        # 8. Compute trend analysis for this series (if enough runs exist)
        trend_summary = None
        try:
            if _AUDIT_AVAILABLE:
                from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView
                
                # Get reproducibility base directory
                repro_base = cohort_dir.parent.parent.parent if cohort_dir and cohort_dir.exists() else self._repro_base_dir / "REPRODUCIBILITY"
                
                if repro_base.exists():
                    trend_analyzer = TrendAnalyzer(
                        reproducibility_dir=repro_base,
                        half_life_days=7.0,
                        min_runs_for_trend=2  # Minimum 2 runs for trend (slope requires 2 points)
                    )
                    
                    # Analyze STRICT series for this specific target
                    all_trends = trend_analyzer.analyze_all_series(view=SeriesView.STRICT)
                    
                    # Find trend for this series
                    series_key_str = None
                    for sk, trend_list in all_trends.items():
                        # Check if this series matches
                        if any(t.series_key.target == (ctx.target_name or ctx.target_column) and 
                               t.series_key.stage == ctx.stage.upper() for t in trend_list):
                            series_key_str = sk
                            break
                    
                    if series_key_str and series_key_str in all_trends:
                        trends = all_trends[series_key_str]
                        
                        # Write trend.json to cohort directory (similar to metadata.json and metrics.json)
                        if cohort_dir and cohort_dir.exists():
                            try:
                                target_name = ctx.target_name or ctx.target_column or "unknown"
                                trend_analyzer.write_cohort_trend(
                                    cohort_dir=cohort_dir,
                                    stage=ctx.stage.upper(),
                                    target=target_name,
                                    trends={series_key_str: trends}  # Pass pre-computed trends
                                )
                                
                                # Also write across-runs timeseries to trend_reports/
                                try:
                                    # Find RESULTS directory (walk up from reproducibility_dir)
                                    results_dir = self._repro_base_dir.parent if hasattr(self, '_repro_base_dir') else None
                                    if results_dir is None:
                                        # Try to find RESULTS by walking up from cohort_dir
                                        current = Path(cohort_dir)
                                        for _ in range(10):
                                            if current.name == "RESULTS":
                                                results_dir = current
                                                break
                                            if not current.parent.exists():
                                                break
                                            current = current.parent
                                    
                                    if results_dir and results_dir.name == "RESULTS":
                                        trend_analyzer.write_across_runs_timeseries(
                                            results_dir=results_dir,
                                            target=target_name,
                                            stage=ctx.stage.upper(),
                                            view=ctx.view if hasattr(ctx, 'view') else "CROSS_SECTIONAL"
                                        )
                                        
                                        # Write run snapshot
                                        if hasattr(ctx, 'run_id') and ctx.run_id:
                                            trend_analyzer.write_run_snapshot(
                                                results_dir=results_dir,
                                                run_id=ctx.run_id,
                                                trends={series_key_str: trends}
                                            )
                                except Exception as e2:
                                    logger.debug(f"Failed to write across-runs timeseries: {e2}")
                            except Exception as e:
                                logger.debug(f"Failed to write trend.json: {e}")
                        
                        # Find trend for the primary metric
                        primary_metric = metrics.get("metric_name", "mean_score")
                        if primary_metric:
                            # Try to find matching metric trend
                            for trend in trends:
                                if trend.metric_name in ["auc_mean", "mean_score", primary_metric.lower()]:
                                    if trend.status == "ok":
                                        trend_summary = {
                                            "slope_per_day": trend.slope_per_day,
                                            "current_estimate": trend.current_estimate,
                                            "ewma_value": trend.ewma_value,
                                            "n_runs": trend.n_runs,
                                            "residual_std": trend.residual_std,
                                            "alerts": trend.alerts
                                        }
                                        
                                        # Log trend summary
                                        slope_str = f"{trend.slope_per_day:+.6f}" if trend.slope_per_day else "N/A"
                                        logger.info(
                                            f"📈 Trend ({trend.metric_name}): slope={slope_str}/day, "
                                            f"current={trend.current_estimate:.4f}, "
                                            f"ewma={trend.ewma_value:.4f}, "
                                            f"n={trend.n_runs} runs"
                                        )
                                        
                                        # Log alerts if any
                                        if trend.alerts:
                                            for alert in trend.alerts:
                                                if alert.get('severity') == 'warning':
                                                    logger.warning(f"  ⚠️  {alert['message']}")
                                                else:
                                                    logger.info(f"  ℹ️  {alert['message']}")
                                    break
        except Exception as e:
            logger.debug(f"Could not compute trend analysis: {e}")
            # Don't fail if trend analysis fails
        
        return {
            "audit_report": audit_report,
            "audit_report_path": str(audit_report_path) if audit_report_path else None,
            "cohort_id": cohort_id,
            "metadata_path": str(cohort_dir / "metadata.json") if cohort_dir and cohort_dir.exists() else None,
            "trend_summary": trend_summary
        }
    
    def generate_trend_summary(
        self,
        view: str = "STRICT",
        min_runs_for_trend: int = 2  # Minimum 2 runs for trend (slope requires 2 points)
    ) -> Dict[str, Any]:
        """
        Generate trend summary for all series in the reproducibility directory.
        
        This can be called at the end of a run to show overall trend status.
        
        Args:
            view: "STRICT" or "PROGRESS"
            min_runs_for_trend: Minimum runs required for trend fitting
        
        Returns:
            Dict with trend summary statistics
        """
        if not _AUDIT_AVAILABLE:
            return {"status": "trend_analyzer_not_available"}
        
        try:
            from TRAINING.common.utils.trend_analyzer import TrendAnalyzer, SeriesView
            
            # Get reproducibility base directory
            repro_base = self._repro_base_dir / "REPRODUCIBILITY"
            if not repro_base.exists():
                # Try alternative location
                repro_base = self.output_dir / "REPRODUCIBILITY"
            
            if not repro_base.exists():
                return {"status": "reproducibility_directory_not_found"}
            
            trend_analyzer = TrendAnalyzer(
                reproducibility_dir=repro_base,
                half_life_days=7.0,
                min_runs_for_trend=min_runs_for_trend
            )
            
            # Analyze trends
            series_view = SeriesView(view.upper() if view.upper() in ["STRICT", "PROGRESS"] else "STRICT")
            all_trends = trend_analyzer.analyze_all_series(view=series_view)
            
            # Generate summary
            summary = {
                "status": "ok",
                "view": series_view.value,
                "n_series": len(all_trends),
                "n_trends": sum(len(t) for t in all_trends.values()),
                "series_with_trends": [],
                "alerts": [],
                "declining_trends": []
            }
            
            for series_key_str, trend_list in all_trends.items():
                for trend in trend_list:
                    if trend.status == "ok":
                        series_info = {
                            "series_key": series_key_str[:100],  # Truncate for readability
                            "metric": trend.metric_name,
                            "slope_per_day": trend.slope_per_day,
                            "current_estimate": trend.current_estimate,
                            "n_runs": trend.n_runs
                        }
                        summary["series_with_trends"].append(series_info)
                        
                        # Collect alerts
                        if trend.alerts:
                            summary["alerts"].extend(trend.alerts)
                        
                        # Flag declining trends
                        if trend.slope_per_day and trend.slope_per_day < -0.001:
                            summary["declining_trends"].append({
                                "metric": trend.metric_name,
                                "slope": trend.slope_per_day,
                                "series": series_key_str[:100]
                            })
            
            # Log summary
            logger.info(f"📊 Trend Summary ({series_view.value}): {summary['n_series']} series, {summary['n_trends']} trends")
            if summary["declining_trends"]:
                logger.warning(f"  ⚠️  {len(summary['declining_trends'])} declining trends detected")
                for decl in summary["declining_trends"][:5]:  # Show first 5
                    logger.warning(f"    - {decl['metric']}: slope={decl['slope']:.6f}/day")
            if summary["alerts"]:
                logger.info(f"  ℹ️  {len(summary['alerts'])} trend alerts")
            
            return summary
        except Exception as e:
            logger.debug(f"Could not generate trend summary: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_metrics_rollups(
        self,
        stage: str,
        run_id: str
    ) -> None:
        """
        Generate view-level and stage-level metrics rollups.
        
        Should be called after all cohorts for a stage are saved.
        
        Args:
            stage: Pipeline stage (TARGET_RANKING, FEATURE_SELECTION, etc.)
            run_id: Current run identifier
        """
        if not self.metrics:
            return
        
        repro_dir = self._repro_base_dir / "REPRODUCIBILITY"
        if not repro_dir.exists():
            repro_dir = self.output_dir / "REPRODUCIBILITY"
        
        if not repro_dir.exists():
            return
        
        stage_dir = repro_dir / stage.upper()
        if not stage_dir.exists():
            return
        
        # Generate view-level rollups
        for view in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
            view_dir = stage_dir / view
            if view_dir.exists():
                self.metrics.generate_view_rollup(view_dir, stage.upper(), view, run_id)
        
        # Generate stage-level rollup
        self.metrics.generate_stage_rollup(stage_dir, stage.upper(), run_id)
        
        # Aggregate metrics facts table (append to Parquet)
        try:
            from TRAINING.common.utils.metrics import aggregate_metrics_facts
            aggregate_metrics_facts(repro_dir)
        except Exception as e:
            logger.debug(f"Failed to aggregate metrics facts table: {e}")
