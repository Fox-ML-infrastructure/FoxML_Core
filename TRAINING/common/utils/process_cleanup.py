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

