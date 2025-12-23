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
Training Plan Consumer

Filters targets and symbols based on training plan to ensure only
approved training jobs are executed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


def load_training_plan(training_plan_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load master training plan from disk.
    
    The training phase should only read master_training_plan.json (single source of truth).
    Falls back to training_plan.json for backward compatibility.
    
    Args:
        training_plan_dir: Directory containing master_training_plan.json
    
    Returns:
        Training plan dict or None if not found
    
    Raises:
        ValueError: If training_plan_dir is invalid
    """
    # Validate input
    if training_plan_dir is None:
        logger.warning("training_plan_dir is None, cannot load plan")
        return None
    
    try:
        training_plan_dir = Path(training_plan_dir)
    except Exception as e:
        logger.warning(f"Invalid training_plan_dir: {e}")
        return None
    
    if not training_plan_dir.exists():
        logger.debug(f"Training plan directory does not exist: {training_plan_dir}")
        return None
    
    if not training_plan_dir.is_dir():
        logger.warning(f"Training plan path is not a directory: {training_plan_dir}")
        return None
    
    # Try master plan first (canonical source)
    master_path = training_plan_dir / "master_training_plan.json"
    if master_path.exists() and master_path.is_file():
        try:
            with open(master_path, 'r') as f:
                plan = json.load(f)
                # Validate plan structure
                if not isinstance(plan, dict):
                    logger.warning(f"Training plan is not a dict, got {type(plan)}")
                    return None
                logger.debug(f"Loaded master training plan from {master_path}")
                return plan
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in master training plan {master_path}: {e}")
        except PermissionError as e:
            logger.warning(f"Permission denied reading {master_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load master training plan from {master_path}: {e}")
    
    # Fallback to convenience mirror for backward compatibility
    json_path = training_plan_dir / "training_plan.json"
    if json_path.exists() and json_path.is_file():
        try:
            with open(json_path, 'r') as f:
                plan = json.load(f)
                # Validate plan structure
                if not isinstance(plan, dict):
                    logger.warning(f"Training plan is not a dict, got {type(plan)}")
                    return None
                logger.debug(f"Loaded training plan from convenience mirror (consider migrating to master_training_plan.json)")
                return plan
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in training plan {json_path}: {e}")
        except PermissionError as e:
            logger.warning(f"Permission denied reading {json_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load training plan from {json_path}: {e}")
    
    return None


def filter_targets_by_training_plan(
    targets: List[str],
    training_plan: Dict[str, Any],
    training_type: str = "cross_sectional"
) -> List[str]:
    """
    Filter targets based on training plan.
    
    Args:
        targets: List of target names
        training_plan: Training plan dict
        training_type: "cross_sectional" or "symbol_specific"
    
    Returns:
        Filtered list of targets that have jobs in the plan
    """
    if training_plan is None:
        return targets
    
    if not targets:
        logger.warning("Empty targets list provided to filter_targets_by_training_plan")
        return []
    
    jobs = training_plan.get("jobs", [])
    
    if not jobs:
        # CRITICAL: 0 jobs indicates plan is disabled or routing produced no valid jobs
        # Check if dev_mode is enabled (allows fallback)
        dev_mode = False
        try:
            from CONFIG.config_loader import get_cfg
            routing_config = get_cfg("training_config.routing", default={}, config_name="training_config")
            dev_mode = routing_config.get("dev_mode", False)
        except Exception:
            pass
        
        if dev_mode:
            # Dev mode: allow fallback but mark as non-production
            logger.warning(
                f"⚠️ Training plan has 0 jobs - plan is disabled (dev_mode=true). "
                f"Possible reasons: 1) Plan generation failed, 2) Routing produced no valid jobs, "
                f"3) All targets were filtered out, or 4) Plan generation was skipped. "
                f"Using fallback: returning all {len(targets)} targets without plan filtering (non_production=true)."
            )
            return targets
        else:
            # Production mode: hard fail
            raise ValueError(
                f"Training plan has 0 jobs. This indicates routing produced no valid jobs. "
                f"Possible reasons: 1) All targets failed routing thresholds, 2) Stability requirements not met, "
                f"3) Sample sizes too small, 4) Scores too low. "
                f"Check routing_config.yaml thresholds or enable dev_mode for testing. "
                f"Targets attempted: {targets}"
            )
    
    # Get targets that have jobs of the specified type
    allowed_targets = set()
    for job in jobs:
        if not isinstance(job, dict):
            logger.warning(f"Skipping invalid job (not a dict) in training plan")
            continue
        if job.get("training_type") == training_type:
            target = job.get("target")
            if target:
                allowed_targets.add(target)
    
    if not allowed_targets:
        logger.warning(f"No {training_type} jobs found in training plan, returning all targets")
        return targets
    
    filtered = [t for t in targets if t in allowed_targets]
    logger.info(f"Filtered {len(targets)} targets → {len(filtered)} targets (based on {training_type} jobs in plan)")
    
    return filtered


def filter_symbols_by_training_plan(
    target: str,
    symbols: List[str],
    training_plan: Dict[str, Any]
) -> List[str]:
    """
    Filter symbols for a target based on training plan.
    
    Args:
        target: Target name
        symbols: List of symbol names
        training_plan: Training plan dict
    
    Returns:
        Filtered list of symbols that have symbol-specific jobs for this target
    """
    if training_plan is None:
        return symbols
    
    if not target or not isinstance(target, str):
        logger.warning(f"Invalid target provided to filter_symbols_by_training_plan: {target}")
        return symbols
    
    if not symbols:
        logger.warning(f"Empty symbols list provided for target {target}")
        return []
    
    if not isinstance(symbols, list):
        logger.warning(f"symbols is not a list, got {type(symbols)}")
        return symbols
    
    jobs = training_plan.get("jobs", [])
    if not isinstance(jobs, list):
        logger.warning(f"Training plan jobs is not a list, got {type(jobs)}")
        return symbols
    
    if not jobs:
        logger.debug(f"No jobs in training plan for target {target}, returning all symbols")
        return symbols
    
    # Get symbols that have symbol-specific jobs for this target
    allowed_symbols = set()
    for job in jobs:
        if not isinstance(job, dict):
            logger.warning(f"Skipping invalid job (not a dict) in training plan")
            continue
        
        try:
            job_target = job.get("target")
            job_type = job.get("training_type")
            job_symbol = job.get("symbol")
            
            if (job_target == target and
                job_type == "symbol_specific" and
                job_symbol is not None):
                if isinstance(job_symbol, str) and job_symbol:
                    allowed_symbols.add(job_symbol)
        except Exception as e:
            logger.warning(f"Error processing job in filter_symbols_by_training_plan: {e}")
            continue
    
    if not allowed_symbols:
        logger.debug(f"No symbol-specific jobs found for {target}, returning all symbols")
        return symbols
    
    filtered = [s for s in symbols if s in allowed_symbols]
    logger.info(f"Filtered {len(symbols)} symbols → {len(filtered)} symbols for {target} (based on training plan)")
    
    return filtered


def get_training_jobs_for_target_symbol(
    training_plan: Dict[str, Any],
    target: str,
    symbol: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get training jobs for a (target, symbol) pair.
    
    Args:
        training_plan: Training plan dict
        target: Target name
        symbol: Optional symbol name (None for CS jobs)
    
    Returns:
        List of job dicts matching the criteria
    """
    if training_plan is None:
        return []
    
    if not isinstance(training_plan, dict):
        logger.warning(f"training_plan is not a dict, got {type(training_plan)}")
        return []
    
    if not target or not isinstance(target, str):
        logger.warning(f"Invalid target: {target}")
        return []
    
    jobs = training_plan.get("jobs", [])
    if not isinstance(jobs, list):
        logger.warning(f"jobs is not a list, got {type(jobs)}")
        return []
    
    matching_jobs = []
    for job in jobs:
        if not isinstance(job, dict):
            logger.warning(f"Skipping invalid job (not a dict)")
            continue
        
        try:
            job_target = job.get("target")
            if job_target != target:
                continue
            
            if symbol is None:
                # CS jobs have symbol=None
                job_symbol = job.get("symbol")
                if job_symbol is None:
                    matching_jobs.append(job)
            else:
                # Symbol-specific jobs
                job_symbol = job.get("symbol")
                if job_symbol == symbol:
                    matching_jobs.append(job)
        except Exception as e:
            logger.warning(f"Error processing job: {e}")
            continue
    
    return matching_jobs


def get_cs_jobs(training_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all cross-sectional jobs from training plan.
    
    Args:
        training_plan: Training plan dict (None = return empty list)
    
    Returns:
        List of CS job dicts
    """
    if training_plan is None:
        return []
    
    if not isinstance(training_plan, dict):
        logger.warning(f"training_plan is not a dict, got {type(training_plan)}")
        return []
    
    jobs = training_plan.get("jobs", [])
    if not isinstance(jobs, list):
        logger.warning(f"jobs is not a list, got {type(jobs)}")
        return []
    
    cs_jobs = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        try:
            if job.get("training_type") == "cross_sectional":
                cs_jobs.append(job)
        except Exception as e:
            logger.warning(f"Error processing job in get_cs_jobs: {e}")
            continue
    
    return cs_jobs


def get_symbol_jobs(training_plan: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Get all symbol-specific jobs from training plan.
    
    Args:
        training_plan: Training plan dict (None = return empty list)
    
    Returns:
        List of symbol-specific job dicts
    """
    if training_plan is None:
        return []
    
    if not isinstance(training_plan, dict):
        logger.warning(f"training_plan is not a dict, got {type(training_plan)}")
        return []
    
    jobs = training_plan.get("jobs", [])
    if not isinstance(jobs, list):
        logger.warning(f"jobs is not a list, got {type(jobs)}")
        return []
    
    symbol_jobs = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        try:
            if job.get("training_type") == "symbol_specific":
                symbol_jobs.append(job)
        except Exception as e:
            logger.warning(f"Error processing job in get_symbol_jobs: {e}")
            continue
    
    return symbol_jobs


def get_jobs_for_target(training_plan: Optional[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    """
    Get all jobs for a target (both CS and symbol-specific).
    
    Args:
        training_plan: Training plan dict (None = return empty list)
        target: Target name
    
    Returns:
        List of job dicts for this target
    """
    if training_plan is None:
        return []
    
    if not isinstance(training_plan, dict):
        logger.warning(f"training_plan is not a dict, got {type(training_plan)}")
        return []
    
    if not target or not isinstance(target, str):
        logger.warning(f"Invalid target: {target}")
        return []
    
    jobs = training_plan.get("jobs", [])
    if not isinstance(jobs, list):
        logger.warning(f"jobs is not a list, got {type(jobs)}")
        return []
    
    target_jobs = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        try:
            if job.get("target") == target:
                target_jobs.append(job)
        except Exception as e:
            logger.warning(f"Error processing job in get_jobs_for_target: {e}")
            continue
    
    return target_jobs


def get_jobs_for_symbol(training_plan: Optional[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
    """
    Get all jobs for a symbol (across all targets).
    
    Args:
        training_plan: Training plan dict (None = return empty list)
        symbol: Symbol name
    
    Returns:
        List of job dicts for this symbol
    """
    if training_plan is None:
        return []
    
    jobs = training_plan.get("jobs", [])
    return [job for job in jobs if job.get("symbol") == symbol]


def get_model_families_for_job(
    training_plan: Optional[Dict[str, Any]],
    target: str,
    symbol: Optional[str] = None,
    training_type: Optional[str] = None
) -> Optional[List[str]]:
    """
    Get model families for a specific job from training plan.
    
    Args:
        training_plan: Training plan dict (None = return None)
        target: Target name
        symbol: Optional symbol name (None for CS)
        training_type: Optional training type filter
    
    Returns:
        List of model families or None if job not found
    """
    if not target:
        logger.warning("Empty target provided to get_model_families_for_job")
        return None
    
    jobs = get_training_jobs_for_target_symbol(training_plan, target, symbol)
    
    if not jobs:
        return None
    
    # If training_type specified, filter by it
    if training_type:
        jobs = [j for j in jobs if j.get("training_type") == training_type]
    
    if not jobs:
        return None
    
    # Return model families from first matching job
    # (In practice, there should be only one job per (target, symbol, type))
    families = jobs[0].get("model_families")
    
    # Validate families is a list
    if families is not None and not isinstance(families, list):
        logger.warning(f"model_families for job {jobs[0].get('job_id')} is not a list: {type(families)}")
        return None
    
    return families


def should_train_target_symbol(
    training_plan: Optional[Dict[str, Any]],
    target: str,
    symbol: Optional[str] = None,
    training_type: Optional[str] = None
) -> bool:
    """
    Check if a (target, symbol) pair should be trained based on training plan.
    
    Args:
        training_plan: Training plan dict (None = allow all)
        target: Target name
        symbol: Optional symbol name (None for CS)
        training_type: Optional training type filter
    
    Returns:
        True if training should proceed
    """
    if training_plan is None:
        return True  # No plan = allow all
    
    jobs = get_training_jobs_for_target_symbol(training_plan, target, symbol)
    
    if not jobs:
        return False
    
    # If training_type specified, filter by it
    if training_type:
        jobs = [j for j in jobs if j.get("training_type") == training_type]
    
    return len(jobs) > 0


def apply_training_plan_filter(
    targets: List[str],
    symbols: List[str],
    training_plan_dir: Optional[Path],
    use_cs_plan: bool = True,
    use_symbol_plan: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Apply training plan filter to targets and symbols.
    
    Args:
        targets: List of target names
        symbols: List of symbol names
        training_plan_dir: Directory containing master_training_plan.json (None = no filtering)
        use_cs_plan: If True, filter targets for CS training
        use_symbol_plan: If True, filter symbols per target for symbol-specific training
    
    Returns:
        Tuple of (filtered_targets, filtered_symbols_by_target)
    """
    # Validate inputs
    if not isinstance(targets, list):
        logger.warning(f"targets is not a list, got {type(targets)}, returning empty")
        return [], {}
    
    if not isinstance(symbols, list):
        logger.warning(f"symbols is not a list, got {type(symbols)}, using empty list")
        symbols = []
    
    # Default return value
    default_symbols_by_target = {t: symbols for t in targets if isinstance(t, str) and t}
    
    if training_plan_dir is None:
        return targets, default_symbols_by_target
    
    # Validate training_plan_dir
    try:
        training_plan_dir = Path(training_plan_dir)
    except Exception as e:
        logger.warning(f"Invalid training_plan_dir: {e}, proceeding without filtering")
        return targets, default_symbols_by_target
    
    if not training_plan_dir.exists():
        logger.debug(f"Training plan directory does not exist: {training_plan_dir}, proceeding without filtering")
        return targets, default_symbols_by_target
    
    # Load training plan with error handling
    try:
        training_plan = load_training_plan(training_plan_dir)
    except Exception as e:
        logger.warning(f"Failed to load training plan from {training_plan_dir}: {e}, proceeding without filtering")
        return targets, default_symbols_by_target
    
    if training_plan is None:
        logger.warning(f"Training plan not found at {training_plan_dir}, proceeding without filtering")
        return targets, default_symbols_by_target
    
    # Filter targets for CS training - with error handling
    try:
        if use_cs_plan:
            filtered_targets = filter_targets_by_training_plan(targets, training_plan, "cross_sectional")
        else:
            filtered_targets = targets
    except Exception as e:
        logger.warning(f"Failed to filter targets: {e}, using all targets")
        filtered_targets = targets
    
    # Validate filtered_targets
    if not isinstance(filtered_targets, list):
        logger.warning(f"filter_targets_by_training_plan returned non-list: {type(filtered_targets)}, using all targets")
        filtered_targets = targets
    
    # Filter symbols per target for symbol-specific training - with error handling
    filtered_symbols_by_target = {}
    for target in filtered_targets:
        if not isinstance(target, str) or not target:
            logger.warning(f"Skipping invalid target: {target}")
            filtered_symbols_by_target[target] = symbols
            continue
        
        try:
            if use_symbol_plan:
                filtered_symbols = filter_symbols_by_training_plan(target, symbols, training_plan)
            else:
                filtered_symbols = symbols
            
            # Validate filtered_symbols
            if not isinstance(filtered_symbols, list):
                logger.warning(f"filter_symbols_by_training_plan returned non-list for {target}, using all symbols")
                filtered_symbols = symbols
            
            filtered_symbols_by_target[target] = filtered_symbols
        except Exception as e:
            logger.warning(f"Failed to filter symbols for target {target}: {e}, using all symbols")
            filtered_symbols_by_target[target] = symbols
    
    logger.info(f"Applied training plan filter: {len(targets)} targets → {len(filtered_targets)} targets")
    
    return filtered_targets, filtered_symbols_by_target


def validate_training_plan(training_plan: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate training plan structure and content.
    
    Args:
        training_plan: Training plan dict
    
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    if training_plan is None:
        return False, ["Training plan is None"]
    
    if not isinstance(training_plan, dict):
        return False, [f"Training plan is not a dict, got {type(training_plan)}"]
    
    # Check required top-level keys
    required_keys = ["metadata", "jobs", "summary"]
    for key in required_keys:
        if key not in training_plan:
            warnings.append(f"Missing required key: {key}")
    
    # Validate metadata
    metadata = training_plan.get("metadata", {})
    if not isinstance(metadata, dict):
        warnings.append("metadata is not a dict")
    else:
        if "generated_at" not in metadata:
            warnings.append("metadata missing 'generated_at'")
        if "total_jobs" not in metadata:
            warnings.append("metadata missing 'total_jobs'")
    
    # Validate jobs
    jobs = training_plan.get("jobs", [])
    if not isinstance(jobs, list):
        warnings.append(f"jobs is not a list, got {type(jobs)}")
    else:
        # Check job structure
        required_job_keys = ["job_id", "target", "route", "training_type"]
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                warnings.append(f"Job {i} is not a dict, got {type(job)}")
                continue
            
            # Validate required keys
            for key in required_job_keys:
                if key not in job:
                    warnings.append(f"Job {i} missing required key: {key}")
            
            # Validate key types
            if "target" in job and not isinstance(job["target"], str):
                warnings.append(f"Job {i}: target is not a string, got {type(job['target'])}")
            
            if "model_families" in job and not isinstance(job["model_families"], list):
                warnings.append(f"Job {i}: model_families is not a list, got {type(job['model_families'])}")
    
    # Validate summary
    summary = training_plan.get("summary", {})
    if not isinstance(summary, dict):
        warnings.append("summary is not a dict")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def get_training_plan_summary(training_plan: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get a summary of the training plan for logging/reporting.
    
    Args:
        training_plan: Training plan dict (None = return empty summary)
    
    Returns:
        Summary dict with counts and statistics
    """
    if training_plan is None:
        return {
            "total_jobs": 0,
            "cs_jobs": 0,
            "symbol_jobs": 0,
            "by_route": {},
            "by_type": {}
        }
    
    summary = training_plan.get("summary", {})
    metadata = training_plan.get("metadata", {})
    
    return {
        "total_jobs": metadata.get("total_jobs", 0),
        "cs_jobs": summary.get("total_cs_jobs", 0),
        "symbol_jobs": summary.get("total_symbol_jobs", 0),
        "by_route": summary.get("by_route", {}),
        "by_type": summary.get("by_type", {}),
        "generated_at": metadata.get("generated_at", "unknown"),
        "run_id": metadata.get("run_id", "unknown")
    }
