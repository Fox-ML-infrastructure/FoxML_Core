"""
Target-First Path Resolution Utilities

Helper functions for organizing run outputs using target-first structure.
Target is the stable join key - all per-target artifacts live together under targets/<target>/.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


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
    Get training_results/ directory.
    
    Args:
        run_root: Run root directory
    
    Returns:
        Path to training_results/ directory
    """
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


def target_repro_dir(run_root: Path, target: str, view: Optional[str] = None, symbol: Optional[str] = None) -> Path:
    """
    Get reproducibility directory for target, optionally scoped by view/symbol.
    
    Args:
        run_root: Run root directory
        target: Target name
        view: Optional view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
    
    Returns:
        - If view=None: targets/{target}/reproducibility/
        - If view="CROSS_SECTIONAL": targets/{target}/reproducibility/CROSS_SECTIONAL/
        - If view="SYMBOL_SPECIFIC" and symbol: targets/{target}/reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/
    """
    base_repro_dir = get_target_reproducibility_dir(Path(run_root), target)
    
    if view == "CROSS_SECTIONAL":
        return base_repro_dir / "CROSS_SECTIONAL"
    elif view == "SYMBOL_SPECIFIC" and symbol:
        return base_repro_dir / "SYMBOL_SPECIFIC" / f"symbol={symbol}"
    elif view is None:
        return base_repro_dir
    else:
        # Invalid combination - log warning and return base
        if view == "SYMBOL_SPECIFIC" and not symbol:
            logger.warning(f"SYMBOL_SPECIFIC view requires symbol parameter, returning base reproducibility directory")
        return base_repro_dir


def target_repro_file_path(run_root: Path, target: str, filename: str, view: Optional[str] = None, symbol: Optional[str] = None) -> Path:
    """
    Get file path in reproducibility directory (for reading with fallback or writing).
    
    This function constructs the path to a file in the reproducibility directory,
    scoped by view/symbol if provided. For reading, callers should check both
    new (view/symbol-scoped) and old (root) locations.
    
    Args:
        run_root: Run root directory
        target: Target name
        filename: Filename (e.g., "selected_features.txt")
        view: Optional view name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
    
    Returns:
        Path to file in reproducibility directory (view/symbol-scoped if provided)
    """
    repro_dir = target_repro_dir(run_root, target, view, symbol)
    return repro_dir / filename


def model_output_dir(training_results_root: Path, family: str, view: str, symbol: Optional[str] = None) -> Path:
    """
    Get model output directory.
    
    Args:
        training_results_root: Training results root directory
        family: Model family name (e.g., "lightgbm")
        view: View name ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC")
        symbol: Optional symbol name (required if view is "SYMBOL_SPECIFIC")
    
    Returns:
        Path to training_results/{family}/view={view}/[symbol={symbol}/]
    """
    family_dir = Path(training_results_root) / family
    view_dir = family_dir / f"view={view}"
    
    if view == "SYMBOL_SPECIFIC" and symbol:
        return view_dir / f"symbol={symbol}"
    elif view == "CROSS_SECTIONAL":
        return view_dir
    else:
        # Invalid combination - log warning and return view_dir
        if view == "SYMBOL_SPECIFIC" and not symbol:
            logger.warning(f"SYMBOL_SPECIFIC view requires symbol parameter, returning view directory without symbol")
        return view_dir

