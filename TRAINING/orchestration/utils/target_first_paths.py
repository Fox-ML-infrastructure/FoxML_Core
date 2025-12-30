"""
Target-First Path Resolution Utilities

Helper functions for organizing run outputs using target-first structure.
Target is the stable join key - all per-target artifacts live together under targets/<target>/.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Set
import logging

logger = logging.getLogger(__name__)

# Module-level warn-once set for view/resolved_mode mismatches
_view_mismatch_warned: Set[tuple] = set()


def get_target_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get the target directory for a given target.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name (e.g., "fwd_ret_5d")
    
    Returns:
        Path to targets/<target>/ directory
    """
    return base_output_dir / "targets" / target


def get_globals_dir(base_output_dir: Path) -> Path:
    """
    Get the globals directory for run-level summaries.
    
    Args:
        base_output_dir: Base run output directory
    
    Returns:
        Path to globals/ directory
    """
    return base_output_dir / "globals"


def get_target_decision_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get decision directory for a target.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/decision/
    """
    return get_target_dir(base_output_dir, target) / "decision"


def get_target_models_dir(base_output_dir: Path, target: str, family: Optional[str] = None) -> Path:
    """
    Get models directory for a target (optionally for a specific family).
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
        family: Optional model family name (e.g., "lightgbm")
    
    Returns:
        Path to targets/<target>/models/ or targets/<target>/models/<family>/
    """
    models_dir = get_target_dir(base_output_dir, target) / "models"
    if family:
        return models_dir / family
    return models_dir


def get_target_metrics_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get metrics directory for a target.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/metrics/
    """
    return get_target_dir(base_output_dir, target) / "metrics"


def get_target_trends_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get trends directory for a target (within-run trends).
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/trends/
    """
    return get_target_dir(base_output_dir, target) / "trends"


def get_target_reproducibility_dir(base_output_dir: Path, target: str) -> Path:
    """
    Get reproducibility directory for a target.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    
    Returns:
        Path to targets/<target>/reproducibility/
    """
    return get_target_dir(base_output_dir, target) / "reproducibility"


def get_global_trends_dir(base_output_dir: Path) -> Path:
    """
    Get global trends directory (cross-target within-run analyses).
    
    Args:
        base_output_dir: Base run output directory
    
    Returns:
        Path to globals/trends/
    """
    return get_globals_dir(base_output_dir) / "trends"


def ensure_target_structure(base_output_dir: Path, target: str) -> None:
    """
    Ensure all target directories exist.
    
    Args:
        base_output_dir: Base run output directory
        target: Target name
    """
    target_dir = get_target_dir(base_output_dir, target)
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "decision").mkdir(exist_ok=True)
    (target_dir / "models").mkdir(exist_ok=True)
    (target_dir / "metrics").mkdir(exist_ok=True)
    (target_dir / "trends").mkdir(exist_ok=True)
    (target_dir / "reproducibility").mkdir(exist_ok=True)


def ensure_globals_structure(base_output_dir: Path) -> None:
    """
    Ensure globals directory structure exists.
    
    Args:
        base_output_dir: Base run output directory
    """
    globals_dir = get_globals_dir(base_output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    (globals_dir / "trends").mkdir(exist_ok=True)


def initialize_run_structure(base_output_dir: Path) -> None:
    """
    Initialize the target-first run structure.
    
    Creates:
    - targets/ directory
    - globals/ directory with trends/ subdirectory
    
    Args:
        base_output_dir: Base run output directory
    """
    base_output_dir.mkdir(parents=True, exist_ok=True)
    (base_output_dir / "targets").mkdir(exist_ok=True)
    ensure_globals_structure(base_output_dir)
    
    # Only create essential directories - no legacy REPRODUCIBILITY structure
    (base_output_dir / "cache").mkdir(exist_ok=True)
    (base_output_dir / "logs").mkdir(exist_ok=True)


def get_metrics_path_from_cohort_dir(cohort_dir: Path, base_output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Map a cohort directory to the corresponding metrics directory.
    
    Maps from:
    - targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<id>/ 
      → targets/<target>/metrics/view=CROSS_SECTIONAL/
    - targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<id>/
      → targets/<target>/metrics/view=SYMBOL_SPECIFIC/symbol=<symbol>/
    
    Args:
        cohort_dir: Cohort directory path (from reproducibility/)
        base_output_dir: Optional base output directory (will be inferred if not provided)
    
    Returns:
        Path to metrics directory, or None if path cannot be resolved
    """
    cohort_dir = Path(cohort_dir)
    
    # Find base_output_dir if not provided
    if base_output_dir is None:
        temp_dir = cohort_dir
        for _ in range(10):
            if temp_dir.name == "targets" and (temp_dir.parent / "targets").exists():
                base_output_dir = temp_dir.parent
                break
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        if base_output_dir is None:
            logger.warning(f"Could not find base_output_dir from cohort_dir: {cohort_dir}")
            return None
    
    # Extract target, view, and symbol from cohort_dir path
    parts = cohort_dir.parts
    target = None
    view = None
    symbol = None
    
    # Find target (should be after "targets")
    for i, part in enumerate(parts):
        if part == "targets" and i + 1 < len(parts):
            target = parts[i + 1]
            break
    
    if not target:
        logger.warning(f"Could not extract target from cohort_dir: {cohort_dir}")
        return None
    
    # Find view (CROSS_SECTIONAL or SYMBOL_SPECIFIC) and symbol
    for i, part in enumerate(parts):
        if part == "reproducibility" and i + 1 < len(parts):
            view = parts[i + 1]
            if view == "SYMBOL_SPECIFIC" and i + 2 < len(parts):
                symbol_part = parts[i + 2]
                if symbol_part.startswith("symbol="):
                    symbol = symbol_part.replace("symbol=", "")
            break
    
    if not view:
        logger.warning(f"Could not extract view from cohort_dir: {cohort_dir}")
        return None
    
    # Build metrics path
    metrics_dir = get_target_metrics_dir(base_output_dir, target) / f"view={view}"
    if symbol:
        metrics_dir = metrics_dir / f"symbol={symbol}"
    
    return metrics_dir


def get_cohort_dir_from_metrics_path(metrics_path: Path, base_output_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Map a metrics path to the corresponding cohort directory (for finding diffs).
    
    Maps from:
    - targets/<target>/metrics/view=CROSS_SECTIONAL/metrics.json
      → targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<id>/
    - targets/<target>/metrics/view=SYMBOL_SPECIFIC/symbol=<symbol>/metrics.json
      → targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<id>/
    
    Note: This function returns the view-level directory. The specific cohort directory
    must be determined by finding the matching cohort_id.
    
    Args:
        metrics_path: Metrics file or directory path (from metrics/)
        base_output_dir: Optional base output directory (will be inferred if not provided)
    
    Returns:
        Path to reproducibility view directory, or None if path cannot be resolved
    """
    metrics_path = Path(metrics_path)
    
    # If it's a file, get the parent directory
    if metrics_path.is_file():
        metrics_dir = metrics_path.parent
    else:
        metrics_dir = metrics_path
    
    # Find base_output_dir if not provided
    if base_output_dir is None:
        temp_dir = metrics_dir
        for _ in range(10):
            if temp_dir.name == "targets" and (temp_dir.parent / "targets").exists():
                base_output_dir = temp_dir.parent
                break
            if not temp_dir.parent.exists():
                break
            temp_dir = temp_dir.parent
        
        if base_output_dir is None:
            logger.warning(f"Could not find base_output_dir from metrics_path: {metrics_path}")
            return None
    
    # Extract target, view, and symbol from metrics path
    parts = metrics_dir.parts
    target = None
    view = None
    symbol = None
    
    # Find target (should be after "targets")
    for i, part in enumerate(parts):
        if part == "targets" and i + 1 < len(parts):
            target = parts[i + 1]
            break
    
    if not target:
        logger.warning(f"Could not extract target from metrics_path: {metrics_path}")
        return None
    
    # Find view and symbol
    for i, part in enumerate(parts):
        if part == "metrics" and i + 1 < len(parts):
            view_part = parts[i + 1]
            if view_part.startswith("view="):
                view = view_part.replace("view=", "")
                if i + 2 < len(parts):
                    symbol_part = parts[i + 2]
                    if symbol_part.startswith("symbol="):
                        symbol = symbol_part.replace("symbol=", "")
            break
    
    if not view:
        logger.warning(f"Could not extract view from metrics_path: {metrics_path}")
        return None
    
    # Build reproducibility path
    repro_dir = get_target_reproducibility_dir(base_output_dir, target) / view
    if symbol:
        repro_dir = repro_dir / f"symbol={symbol}"
    
    return repro_dir


def run_root(output_dir: Path) -> Path:
    """
    Get run root directory (has targets/, globals/, cache/).
    
    Walks up from output_dir to find the run directory that contains
    targets/, globals/, or cache/ directories.
    
    Args:
        output_dir: Any path within the run directory
    
    Returns:
        Path to run root directory
    """
    current = Path(output_dir).resolve()
    for _ in range(20):  # Limit search depth
        if (current / "targets").exists() or (current / "globals").exists() or (current / "cache").exists():
            return current
        if not current.parent.exists() or current.parent == current:
            break
        current = current.parent
    
    # Fallback: return original if we can't find run root
    logger.warning(f"Could not find run root from {output_dir}, using as-is")
    return Path(output_dir).resolve()


def training_results_root(run_root: Path) -> Path:
    """
    DEPRECATED: Get training_results/ directory.
    
    This function is deprecated. All models now go to targets/<target>/models/.
    Use ArtifactPaths.model_dir() instead.
    
    Args:
        run_root: Run root directory
    
    Returns:
        Path to training_results/ directory (deprecated - for backward compatibility only)
    """
    import warnings
    warnings.warn(
        "training_results_root() is deprecated. Use ArtifactPaths.model_dir() instead. "
        "All models now go to targets/<target>/models/.",
        DeprecationWarning,
        stacklevel=2
    )
    return Path(run_root) / "training_results"


def globals_dir(run_root: Path, kind: Optional[str] = None) -> Path:
    """
    Get globals directory with optional subfolder.
    
    Args:
        run_root: Run root directory
        kind: Optional subfolder name ("routing", "training", "summaries", "rankings", or None for root)
    
    Returns:
        Path to globals/ or globals/{kind}/ directory
    """
    base = get_globals_dir(Path(run_root))
    if kind:
        return base / kind
    return base


def target_repro_dir(
    run_root: Path, 
    target: str, 
    view: str, 
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None  # NEW: For cross-run reproducibility
) -> Path:
    """
    Get reproducibility directory for target, scoped by view/universe/symbol.
    
    Args:
        run_root: Run root directory
        target: Target name
        view: REQUIRED view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
        universe_sig: Optional universe signature (hash of sorted symbols list).
                      When provided, adds universe={universe_sig}/ to path for
                      cross-run reproducibility and collision prevention.
    
    Returns:
        - If view="CROSS_SECTIONAL" (no universe_sig): targets/{target}/reproducibility/CROSS_SECTIONAL/
        - If view="CROSS_SECTIONAL" (with universe_sig): targets/{target}/reproducibility/CROSS_SECTIONAL/universe={universe_sig}/
        - If view="SYMBOL_SPECIFIC" (no universe_sig): targets/{target}/reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/
        - If view="SYMBOL_SPECIFIC" (with universe_sig): targets/{target}/reproducibility/SYMBOL_SPECIFIC/universe={universe_sig}/symbol={symbol}/
    
    Raises:
        ValueError: If view is None or invalid, or if SYMBOL_SPECIFIC without symbol
    
    Note:
        This function uses the passed view parameter. Callers should pass resolved_mode from run context (SST)
        to ensure consistency. If view differs from resolved_mode, a warning is logged.
        
        universe_sig should be computed from compute_universe_signature(symbols) and passed in for
        cross-run reproducibility. Different universes get different directories.
    """
    if view is None:
        raise ValueError("view parameter is required for feature selection artifacts")
    if view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
        raise ValueError(f"Invalid view: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
    if view == "SYMBOL_SPECIFIC" and symbol is None:
        raise ValueError("symbol parameter is required when view='SYMBOL_SPECIFIC'")
    
    # Derive path_mode from SST resolved_mode (don't mutate view - callers may rely on it)
    path_mode = view  # Default to caller's view
    try:
        from TRAINING.orchestration.utils.run_context import get_resolved_mode
        resolved_mode = get_resolved_mode(run_root)
        if resolved_mode and resolved_mode != view:
            # Warn-once: don't spam logs for every file/target
            warn_key = (view, resolved_mode)
            if warn_key not in _view_mismatch_warned:
                _view_mismatch_warned.add(warn_key)
                logger.warning(
                    f"View mismatch (once): passed view={view} but resolved_mode={resolved_mode}. "
                    f"Using resolved_mode={resolved_mode} for path construction."
                )
            path_mode = resolved_mode  # Use SST for paths, but don't mutate view
    except Exception:
        pass  # Silently ignore if run context not available
    
    base_repro_dir = get_target_reproducibility_dir(Path(run_root), target)
    
    # Build path with optional universe_sig using path_mode (SST-derived)
    if path_mode == "CROSS_SECTIONAL":
        path = base_repro_dir / "CROSS_SECTIONAL"
        if universe_sig:
            path = path / f"universe={universe_sig}"
        return path
    else:  # SYMBOL_SPECIFIC
        path = base_repro_dir / "SYMBOL_SPECIFIC"
        if universe_sig:
            path = path / f"universe={universe_sig}"
        return path / f"symbol={symbol}"


def target_repro_file_path(
    run_root: Path, 
    target: str, 
    filename: str, 
    view: str, 
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None
) -> Path:
    """
    Get file path in view-scoped reproducibility directory.
    
    This function constructs the path to a file in the reproducibility directory,
    scoped by view/universe/symbol. For feature selection artifacts, view is REQUIRED.
    
    Args:
        run_root: Run root directory
        target: Target name
        filename: Filename (e.g., "selected_features.txt")
        view: REQUIRED view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
        universe_sig: Optional universe signature for cross-run reproducibility
    
    Returns:
        Path to file in view-scoped reproducibility directory
    
    Raises:
        ValueError: If view is None or invalid, or if SYMBOL_SPECIFIC without symbol
    """
    repro_dir = target_repro_dir(run_root, target, view, symbol, universe_sig)
    return repro_dir / filename


def model_output_dir(
    training_results_root: Path, 
    family: str, 
    view: str, 
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None
) -> Path:
    """
    Get model output directory.
    
    Args:
        training_results_root: Training results root directory
        family: Model family name (e.g., "lightgbm")
        view: View name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
        universe_sig: Optional universe signature for cross-run reproducibility
    
    Returns:
        Path to training_results/{family}/view={view}/[universe={universe_sig}/][symbol={symbol}/]
    """
    family_dir = Path(training_results_root) / family
    view_dir = family_dir / f"view={view}"
    
    # Add universe scoping if provided
    if universe_sig:
        view_dir = view_dir / f"universe={universe_sig}"
    
    if view == "SYMBOL_SPECIFIC" and symbol:
        return view_dir / f"symbol={symbol}"
    elif view == "CROSS_SECTIONAL":
        return view_dir
    else:
        # Invalid combination - log warning and return view_dir
        if view == "SYMBOL_SPECIFIC" and not symbol:
            logger.warning(f"SYMBOL_SPECIFIC view requires symbol parameter, returning view directory without symbol")
        return view_dir

