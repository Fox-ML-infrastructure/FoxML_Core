"""
Reproducibility Tracking Module

Tracks and compares run results across pipeline stages to verify deterministic behavior.
Supports target ranking, feature selection, and other pipeline stages.

Usage:
    from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
    
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
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# Use root logger to ensure messages are visible regardless of calling script's logger setup
logger = logging.getLogger(__name__)
# Ensure this logger propagates to root so messages are visible
logger.propagate = True

# Also try to get the calling script's logger if available (for better integration)
def _get_main_logger():
    """Try to get the main script's logger for better log integration"""
    # Check common logger names used in scripts (in order of preference)
    for logger_name in ['rank_target_predictability', 'multi_model_feature_selection', '__main__']:
        main_logger = logging.getLogger(logger_name)
        if main_logger.handlers:
            return main_logger
    # Fallback to root logger (always has handlers if logging is configured)
    root_logger = logging.getLogger()
    return root_logger


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
        use_z_score: Optional[bool] = None  # Override config use_z_score
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
        
        # Load thresholds from config
        self.thresholds = self._load_thresholds(thresholds)
        self.use_z_score = self._load_use_z_score(use_z_score)
    
    def _load_thresholds(self, override: Optional[Dict[str, Dict[str, float]]] = None) -> Dict[str, Dict[str, float]]:
        """Load reproducibility thresholds from config."""
        if override:
            return override
        
        try:
            from CONFIG.config_loader import get_safety_config
            safety_cfg = get_safety_config()
            safety_section = safety_cfg.get('safety', {})
            repro_cfg = safety_section.get('reproducibility', {})
            thresholds_cfg = repro_cfg.get('thresholds', {})
            
            # Default thresholds if config missing
            defaults = {
                'roc_auc': {'abs': 0.005, 'rel': 0.02, 'z_score': 1.0},
                'composite': {'abs': 0.02, 'rel': 0.05, 'z_score': 1.5},
                'importance': {'abs': 0.05, 'rel': 0.20, 'z_score': 2.0}
            }
            
            # Merge config with defaults
            thresholds = {}
            for metric in ['roc_auc', 'composite', 'importance']:
                thresholds[metric] = defaults[metric].copy()
                if metric in thresholds_cfg:
                    thresholds[metric].update(thresholds_cfg[metric])
            
            return thresholds
        except Exception as e:
            logger.debug(f"Could not load reproducibility thresholds from config: {e}, using defaults")
            # Return defaults
            return {
                'roc_auc': {'abs': 0.005, 'rel': 0.02, 'z_score': 1.0},
                'composite': {'abs': 0.02, 'rel': 0.05, 'z_score': 1.5},
                'importance': {'abs': 0.05, 'rel': 0.20, 'z_score': 2.0}
            }
    
    def _load_use_z_score(self, override: Optional[bool] = None) -> bool:
        """Load use_z_score setting from config."""
        if override is not None:
            return override
        
        try:
            from CONFIG.config_loader import get_safety_config
            safety_cfg = get_safety_config()
            safety_section = safety_cfg.get('safety', {})
            repro_cfg = safety_section.get('reproducibility', {})
            return repro_cfg.get('use_z_score', True)
        except Exception:
            return True  # Default: use z-score
    
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
        
        # Classification logic: must pass BOTH abs AND rel thresholds for STABLE
        # Use z-score if available, otherwise use abs/rel
        if z_score is not None:
            # Use z-score as primary criterion
            if z_score < z_thr and abs_diff < abs_thr and rel_diff < rel_thr:
                classification = 'STABLE'
            elif z_score < 2 * z_thr and abs_diff < 2 * abs_thr and rel_diff < 2 * rel_thr:
                classification = 'DRIFTING'
            else:
                classification = 'DIVERGED'
        else:
            # Fallback to abs/rel thresholds
            if abs_diff < abs_thr and rel_diff < rel_thr:
                classification = 'STABLE'
            elif abs_diff < 2 * abs_thr and rel_diff < 2 * rel_thr:
                classification = 'DRIFTING'
            else:
                classification = 'DIVERGED'
        
        return classification, abs_diff, rel_diff, z_score
    
    def log_comparison(
        self,
        stage: str,
        item_name: str,
        metrics: Dict[str, Any],
        additional_data: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Compare current run to previous run and log the comparison for reproducibility verification.
        
        Uses tolerance bands with STABLE/DRIFTING/DIVERGED classification. Only escalates to
        warnings for meaningful differences (DIVERGED).
        
        Args:
            stage: Pipeline stage name (e.g., "target_ranking", "feature_selection")
            item_name: Name of the item (e.g., target name, symbol name)
            metrics: Dictionary of metrics to track and compare
            additional_data: Optional additional data to store with the run
        """
        previous = self.load_previous_run(stage, item_name)
        
        if previous is None:
            # Use main logger if available for better visibility
            main_logger = _get_main_logger()
            main_logger.info(f"📊 Reproducibility: First run for {stage}:{item_name} (no previous run to compare)")
            logger.info(f"📊 Reproducibility: First run for {stage}:{item_name} (no previous run to compare)")
            # Save current run for next time
            self.save_run(stage, item_name, metrics, additional_data)
            return
        
        # Extract metrics for comparison
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
        
        # Classify differences
        mean_class, mean_abs, mean_rel, mean_z = self._classify_diff(
            previous_mean, current_mean, previous_std, 'roc_auc'
        )
        composite_class, composite_abs, composite_rel, composite_z = self._classify_diff(
            previous_composite, current_composite, None, 'composite'
        )
        importance_class, importance_abs, importance_rel, importance_z = self._classify_diff(
            previous_importance, current_importance, None, 'importance'
        )
        
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
        status_msg = f"{emoji} Reproducibility: {overall_class}"
        if overall_class == 'STABLE':
            status_msg += f" (Δ {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); within tolerance)"
        elif overall_class == 'DRIFTING':
            status_msg += f" (Δ {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); small drift detected)"
        else:  # DIVERGED
            status_msg += f" (Δ {metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{z_info}); exceeds tolerance)"
        
        log_level(status_msg)
        if main_logger != logger:
            if overall_class == 'DIVERGED':
                main_logger.warning(status_msg)
            else:
                main_logger.info(status_msg)
        
        # Detailed comparison (always log for traceability)
        prev_msg = f"   Previous: {metric_name}={previous_mean:.3f}±{previous_std:.3f}, " \
                   f"importance={previous_importance:.2f}, composite={previous_composite:.3f}"
        main_logger.info(prev_msg)
        logger.info(prev_msg)
        
        curr_msg = f"   Current:  {metric_name}={current_mean:.3f}±{current_std:.3f}, " \
                   f"importance={current_importance:.2f}, composite={current_composite:.3f}"
        main_logger.info(curr_msg)
        logger.info(curr_msg)
        
        # Diff line with classifications
        diff_parts = [f"{metric_name}={mean_diff:+.4f} ({mean_rel:+.2f}%{', z=' + f'{mean_z:.2f}' if mean_z else ''}) [{mean_class}]"]
        diff_parts.append(f"composite={composite_diff:+.4f} ({composite_rel:+.2f}%) [{composite_class}]")
        diff_parts.append(f"importance={importance_diff:+.2f} ({importance_rel:+.2f}%) [{importance_class}]")
        diff_msg = f"   Diff:     {', '.join(diff_parts)}"
        main_logger.info(diff_msg)
        logger.info(diff_msg)
        
        # Warning only for DIVERGED
        if overall_class == 'DIVERGED':
            warn_msg = f"   ⚠️  Results differ significantly from previous run - check for non-deterministic behavior, config changes, or data differences"
            main_logger.warning(warn_msg)
            logger.warning(warn_msg)
        
        # Save current run for next time
        self.save_run(stage, item_name, metrics, additional_data)
