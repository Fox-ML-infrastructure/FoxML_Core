# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Git utilities - Single Source of Truth for git operations.

This module provides canonical functions for git operations.
Do not duplicate these functions elsewhere.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_git_commit(short: bool = True) -> Optional[str]:
    """
    Get current git commit hash.
    
    This is the SST (Single Source of Truth) for getting git commit info.
    All other modules should import from here.
    
    Args:
        short: If True, return short hash (12 chars). If False, return full hash.
    
    Returns:
        Git commit hash string, or None if not in a git repo or git unavailable.
    """
    try:
        from TRAINING.common.subprocess_utils import safe_subprocess_run
        
        cmd = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
        result = safe_subprocess_run(cmd, timeout=5)
        
        if result.returncode == 0:
            commit = result.stdout.strip()
            # Ensure consistent length for short hash
            return commit[:12] if short else commit
    except ImportError:
        # Fallback if subprocess_utils not available
        try:
            import subprocess
            cmd = ['git', 'rev-parse', '--short', 'HEAD'] if short else ['git', 'rev-parse', 'HEAD']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                commit = result.stdout.strip()
                return commit[:12] if short else commit
        except Exception as e:
            logger.debug(f"Failed to get git commit (fallback): {e}")
    except Exception as e:
        logger.debug(f"Failed to get git commit: {e}")
    
    return None


def get_git_sha() -> Optional[str]:
    """
    Alias for get_git_commit() for backward compatibility.
    
    DEPRECATED: Use get_git_commit() instead.
    """
    return get_git_commit(short=True)


# Aliases for backward compatibility
_get_git_commit = get_git_commit
_get_git_commit_hash = get_git_commit
