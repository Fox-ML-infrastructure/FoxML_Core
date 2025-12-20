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

