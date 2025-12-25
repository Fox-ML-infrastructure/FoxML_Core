# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Process Cleanup Utilities

Utilities for cleaning up process resources, especially for multiprocessing.
"""

import atexit
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def setup_loky_cleanup(joblib_temp: Optional[Path] = None) -> None:
    """
    Set up loky worker cleanup to prevent resource tracker warnings.
    
    Registers an atexit handler to cleanly shutdown loky workers.
    Also sets up joblib temp folder if provided.
    
    Args:
        joblib_temp: Optional path to joblib temp folder
    """
    # Set up joblib temp folder if provided
    if joblib_temp is not None:
        joblib_temp.mkdir(parents=True, exist_ok=True)
        import os
        os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_temp))
    
    # Register loky shutdown handler
    try:
        from joblib.externals.loky import get_reusable_executor
        
        @atexit.register
        def _loky_shutdown():
            """Force clean loky worker shutdown at exit to prevent semlock/file leaks."""
            try:
                get_reusable_executor().shutdown(wait=True, kill_workers=True)
            except Exception:
                pass  # Ignore errors during shutdown
        
        logger.debug("Loky cleanup handler registered")
    except Exception:
        # Loky not available, skip
        pass


def setup_loky_cleanup_from_config() -> None:
    """
    Set up loky cleanup using config loader.
    
    Loads joblib_temp from config if available.
    """
    try:
        from CONFIG.config_loader import get_cfg
        joblib_temp = get_cfg("system.paths.joblib_temp", config_name="system_config")
        if joblib_temp:
            joblib_temp_path = Path(joblib_temp)
        else:
            joblib_temp_path = Path.home() / "trainer_tmp" / "joblib"
    except Exception:
        joblib_temp_path = Path.home() / "trainer_tmp" / "joblib"
    
    setup_loky_cleanup(joblib_temp_path)

