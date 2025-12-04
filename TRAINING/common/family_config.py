"""
Copyright (c) 2025 Fox ML Infrastructure

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
Family Configuration Loader

Loads family-specific thread policies and runtime settings from YAML config.
"""


import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Cache the loaded config
_FAMILY_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_PATH = Path(__file__).parent.parent / "config" / "family_config.yaml"

def load_family_config() -> Dict[str, Any]:
    """Load family configuration from YAML file (cached)."""
    global _FAMILY_CONFIG
    
    if _FAMILY_CONFIG is not None:
        return _FAMILY_CONFIG
    
    if not _CONFIG_PATH.exists():
        # Return minimal default config if file doesn't exist
        return {
            "families": {},
            "thread_policies": {}
        }
    
    with open(_CONFIG_PATH, 'r') as f:
        _FAMILY_CONFIG = yaml.safe_load(f)
    
    return _FAMILY_CONFIG

def get_family_info(family_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific family.
    
    Args:
        family_name: Family name (e.g., "LightGBM", "MLP")
    
    Returns:
        Dictionary with family configuration
    """
    config = load_family_config()
    families = config.get("families", {})
    
    # Remove "Trainer" suffix if present
    base_name = family_name.replace("Trainer", "")
    
    return families.get(base_name, {
        "thread_policy": "omp_heavy",  # Safe default
        "needs_tf": False,
        "needs_torch": False,
        "ridge_solver": "auto"
    })

def get_thread_policy(family_name: str) -> str:
    """Get the thread policy for a family."""
    family_info = get_family_info(family_name)
    return family_info.get("thread_policy", "omp_heavy")

def get_policy_env_vars(policy: str) -> Dict[str, str]:
    """
    Get environment variables for a thread policy.
    
    Args:
        policy: Policy name (e.g., "omp_heavy", "cpu_blas_only", "tf_cpu")
    
    Returns:
        Dictionary of environment variables to set
    """
    config = load_family_config()
    policies = config.get("thread_policies", {})
    
    if policy not in policies:
        return {}
    
    policy_config = policies[policy]
    
    # Extract env vars (skip description)
    env_vars = {
        k: str(v) for k, v in policy_config.items()
        if k != "description" and v is not None
    }
    
    return env_vars

def apply_thread_policy(family_name: str, base_env: Dict[str, str]) -> Dict[str, str]:
    """
    Apply thread policy for a family to an environment dict.
    
    Args:
        family_name: Family name
        base_env: Base environment dictionary
    
    Returns:
        Updated environment dictionary
    """
    policy = get_thread_policy(family_name)
    policy_env = get_policy_env_vars(policy)
    
    # Merge policy env into base env
    env = base_env.copy()
    
    # Special handling for CUDA_VISIBLE_DEVICES
    if "CUDA_VISIBLE_DEVICES" in policy_env:
        if policy_env["CUDA_VISIBLE_DEVICES"] == "-1":
            # Force CPU mode
            env["CUDA_VISIBLE_DEVICES"] = "-1"
        else:
            # Use TRAINER_GPU_IDS if available
            env["CUDA_VISIBLE_DEVICES"] = os.getenv("TRAINER_GPU_IDS", policy_env["CUDA_VISIBLE_DEVICES"])
    
    # Apply all other policy env vars
    for key, value in policy_env.items():
        if key != "CUDA_VISIBLE_DEVICES":
            env[key] = value
    
    return env

def get_ridge_solver(family_name: str) -> str:
    """Get the Ridge solver to use for a family."""
    family_info = get_family_info(family_name)
    return family_info.get("ridge_solver", "auto")

def should_expect_gpu(family_name: str) -> bool:
    """Check if a family should expect GPU availability."""
    policy = get_thread_policy(family_name)
    return policy in {"tf_gpu", "torch_gpu"}

