# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Safe Subprocess Utilities

Helper functions for subprocess calls that avoid readline library conflicts.
These utilities set safe environment variables to prevent deadlocks from
Conda/system readline library mismatches.
"""

import os
import subprocess
import logging
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_safe_subprocess_env() -> Dict[str, str]:
    """
    Get a safe environment dictionary for subprocess calls.
    
    This prevents readline library conflicts that can cause:
    - "sh: symbol lookup error: sh: undefined symbol: rl_print_keybinding"
    - Process deadlocks/hangs when subprocess calls fail and retry indefinitely
    
    Also filters LD_LIBRARY_PATH to remove AppImage mount paths that can cause
    library conflicts (e.g., Cursor AppImage's readline shadowing system libs).
    
    Returns:
        Environment dictionary with safe settings for subprocess calls
    """
    env = os.environ.copy()
    
    # Disable readline features to avoid library conflicts
    env.setdefault('TERM', 'dumb')  # Disable readline features
    env.setdefault('SHELL', '/usr/bin/bash')  # Use bash instead of sh if available
    env.setdefault('INPUTRC', '/dev/null')  # Disable readline config
    
    # CRITICAL: Filter LD_LIBRARY_PATH to remove AppImage mount paths
    # AppImage mount paths (e.g., /tmp/.mount_Cursor*) can shadow system/conda libs
    # and cause readline ABI mismatches when subprocesses load libraries
    ld_path = env.get('LD_LIBRARY_PATH', '')
    if ld_path:
        # Split by colon, filter out AppImage mount paths, rejoin
        paths = ld_path.split(':')
        filtered_paths = [
            p for p in paths
            if p and not p.startswith('/tmp/.mount_')  # Remove AppImage mounts
        ]
        if filtered_paths:
            env['LD_LIBRARY_PATH'] = ':'.join(filtered_paths)
        else:
            # If all paths were AppImage mounts, remove LD_LIBRARY_PATH entirely
            # This lets the system use default library search paths
            env.pop('LD_LIBRARY_PATH', None)
    
    return env


def safe_subprocess_run(
    cmd: List[str],
    *,
    timeout: Optional[float] = None,
    cwd: Optional[str] = None,
    check: bool = False,
    capture_output: bool = True,
    text: bool = True,
    **kwargs
) -> subprocess.CompletedProcess:
    """
    Run a subprocess command with safe environment variables to avoid readline conflicts.
    
    This wrapper around subprocess.run() automatically sets TERM=dumb, SHELL=/usr/bin/bash,
    and INPUTRC=/dev/null to prevent readline library conflicts that can cause process deadlocks.
    
    Args:
        cmd: Command to run (list of strings)
        timeout: Timeout in seconds (default: None)
        cwd: Working directory (default: None)
        check: If True, raise CalledProcessError on non-zero exit (default: False)
        capture_output: If True, capture stdout and stderr (default: True)
        text: If True, decode output as text (default: True)
        **kwargs: Additional arguments passed to subprocess.run()
    
    Returns:
        CompletedProcess object with returncode, stdout, stderr
    
    Raises:
        FileNotFoundError: If command not found
        subprocess.TimeoutExpired: If timeout exceeded
        subprocess.CalledProcessError: If check=True and returncode != 0
        OSError: Can occur with readline library conflicts (symbol lookup errors)
    
    Example:
        >>> result = safe_subprocess_run(['git', 'rev-parse', '--short', 'HEAD'], timeout=2)
        >>> if result.returncode == 0:
        ...     print(result.stdout.strip())
    """
    env = get_safe_subprocess_env()
    
    try:
        return subprocess.run(
            cmd,
            env=env,
            timeout=timeout,
            cwd=cwd,
            check=check,
            capture_output=capture_output,
            text=text,
            **kwargs
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        # Re-raise these as-is (expected errors)
        raise
    except OSError as e:
        # OSError can occur with readline library conflicts (symbol lookup errors)
        # Log and re-raise with helpful message
        error_msg = str(e)
        if 'symbol lookup error' in error_msg or 'rl_print_keybinding' in error_msg:
            logger.warning(
                f"Subprocess failed due to readline library conflict: {e}\n"
                f"Command: {' '.join(cmd)}\n"
                f"Fix: Run 'conda install -c conda-forge readline=8.2' or 'conda update readline'"
            )
        raise
    except Exception as e:
        # Catch-all for other unexpected errors
        logger.debug(f"Subprocess call failed: {e} (command: {' '.join(cmd)})")
        raise
