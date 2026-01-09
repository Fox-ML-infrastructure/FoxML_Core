# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Path Setup Utilities

Utilities for setting up Python path for imports.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple


def ensure_project_path(project_root: Optional[Path] = None) -> Path:
    """
    Ensure project root is in sys.path.
    
    Args:
        project_root: Optional project root path. If None, auto-detects from current file.
    
    Returns:
        Project root path
    """
    if project_root is None:
        # Auto-detect: assume this file is in TRAINING/common/utils/
        # Go up 3 levels: utils -> common -> TRAINING -> repo root
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]
    
    project_root = Path(project_root).resolve()
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def ensure_training_path(training_root: Optional[Path] = None) -> Path:
    """
    Ensure TRAINING directory is in sys.path.
    
    Args:
        training_root: Optional TRAINING root path. If None, auto-detects.
    
    Returns:
        TRAINING root path
    """
    if training_root is None:
        # Auto-detect: assume this file is in TRAINING/common/utils/
        current_file = Path(__file__).resolve()
        training_root = current_file.parents[2]  # utils -> common -> TRAINING
    
    training_root = Path(training_root).resolve()
    
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    
    return training_root


def ensure_config_path(config_dir: Optional[Path] = None, project_root: Optional[Path] = None) -> Path:
    """
    Ensure CONFIG directory is in sys.path.
    
    Args:
        config_dir: Optional CONFIG directory path. If None, auto-detects from project_root.
        project_root: Optional project root path. If None, auto-detects.
    
    Returns:
        CONFIG directory path
    """
    if config_dir is None:
        if project_root is None:
            project_root = ensure_project_path()
        else:
            project_root = Path(project_root).resolve()
        config_dir = project_root / "CONFIG"
    
    config_dir = Path(config_dir).resolve()
    
    if str(config_dir) not in sys.path:
        sys.path.insert(0, str(config_dir))
    
    return config_dir


def ensure_current_directory() -> None:
    """
    Ensure current directory ('.') is in sys.path for relative imports.
    """
    if '.' not in sys.path:
        sys.path.insert(0, '.')


def setup_all_paths(project_root: Optional[Path] = None) -> Tuple[Path, Path, Path]:
    """
    Set up all paths (project, TRAINING, CONFIG, current directory).
    
    Args:
        project_root: Optional project root path. If None, auto-detects.
    
    Returns:
        Tuple of (project_root, training_root, config_dir)
    """
    project_root = ensure_project_path(project_root)
    training_root = ensure_training_path()
    config_dir = ensure_config_path(project_root=project_root)
    ensure_current_directory()
    
    return project_root, training_root, config_dir

