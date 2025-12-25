# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

"""
Reproducibility Utility Functions

Utility functions for reproducibility tracking (environment info, tagged unions, etc.).
"""

import json
import logging
import hashlib
import sys
import platform
import socket
from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


def collect_environment_info() -> Dict[str, Any]:
    """
    Collect environment information for audit-grade metadata.
    
    Returns:
        Dict with python_version, platform, hostname, cuda_version, dependencies_hash
    """
    env_info = {
        "python_version": sys.version.split()[0],  # e.g., "3.10.12"
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
    }
    
    # Hostname (optional, may fail in some environments)
    try:
        env_info["hostname"] = socket.gethostname()
    except Exception:
        pass
    
    # CUDA version (if available)
    try:
        import subprocess
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Extract version from nvcc output
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    import re
                    match = re.search(r'release\s+(\d+\.\d+)', line, re.I)
                    if match:
                        env_info["cuda_version"] = match.group(1)
                        break
    except Exception:
        pass
    
    # GPU name (if available)
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            env_info["gpu_name"] = result.stdout.strip().split('\n')[0]
    except Exception:
        pass
    
    # Dependencies lock hash (hash of requirements.txt or environment.yml)
    deps_hash = None
    repo_root = Path(__file__).resolve().parents[4]  # utils -> reproducibility -> utils -> orchestration -> TRAINING -> repo root
    for lock_file in ["requirements.txt", "environment.yml", "poetry.lock", "uv.lock"]:
        lock_path = repo_root / lock_file
        if lock_path.exists():
            try:
                with open(lock_path, 'rb') as f:
                    deps_hash = hashlib.sha256(f.read()).hexdigest()[:16]
                    env_info["dependencies_lock_file"] = lock_file
                    break
            except Exception:
                pass
    
    if deps_hash:
        env_info["dependencies_lock_hash"] = deps_hash
    
    # Collect library versions (CRITICAL: for comparability)
    library_versions = {}
    critical_libs = [
        'pandas', 'numpy', 'scipy', 'scikit-learn', 'sklearn',
        'lightgbm', 'xgboost', 'catboost',
        'torch', 'tensorflow', 'keras',
        'joblib', 'polars'
    ]
    
    for lib_name in critical_libs:
        try:
            # Handle scikit-learn vs sklearn naming
            import_name = 'sklearn' if lib_name == 'scikit-learn' else lib_name
            mod = __import__(import_name)
            if hasattr(mod, '__version__'):
                library_versions[lib_name] = mod.__version__
        except (ImportError, AttributeError):
            # Library not installed or no version attribute
            pass
    
    if library_versions:
        env_info["library_versions"] = library_versions
    
    return env_info


def compute_comparable_key(
    stage: str,
    target_name: str,
    route_type: Optional[str],
    view: Optional[str],
    symbol: Optional[str],
    date_range_start: Optional[str],
    date_range_end: Optional[str],
    cv_details: Optional[Dict[str, Any]],
    feature_registry_hash: Optional[str],
    label_definition_hash: Optional[str],
    min_cs: Optional[int],
    max_cs_samples: Optional[int],
    universe_id: Optional[str]
) -> str:
    """
    Compute a comparable key for run comparison.
    
    Runs with the same comparable_key should produce similar results
    (allowing for acceptable variance from randomness).
    
    Args:
        stage: Pipeline stage
        target_name: Target name
        route_type: Route type (CROSS_SECTIONAL, INDIVIDUAL, etc.)
        view: View type (for TARGET_RANKING)
        symbol: Symbol (for SYMBOL_SPECIFIC)
        date_range_start: Start timestamp
        date_range_end: End timestamp
        cv_details: CV configuration details
        feature_registry_hash: Feature registry hash
        label_definition_hash: Label definition hash
        min_cs: Minimum cross-sectional samples
        max_cs_samples: Maximum cross-sectional samples
        universe_id: Universe identifier
    
    Returns:
        Hex hash of comparable key (16 chars)
    """
    parts = []
    
    # Core identity
    parts.append(f"stage={stage}")
    parts.append(f"target={target_name}")
    
    # Route/view
    if route_type:
        parts.append(f"route={route_type}")
    if view:
        parts.append(f"view={view}")
    if symbol:
        parts.append(f"symbol={symbol}")
    
    # Data range
    if date_range_start:
        parts.append(f"start={date_range_start}")
    if date_range_end:
        parts.append(f"end={date_range_end}")
    
    # Universe/split config
    if universe_id:
        parts.append(f"universe={universe_id}")
    if min_cs is not None:
        parts.append(f"min_cs={min_cs}")
    if max_cs_samples is not None:
        parts.append(f"max_cs={max_cs_samples}")
    
    # CV config (critical for comparability)
    if cv_details:
        cv_parts = []
        if 'cv_method' in cv_details:
            cv_parts.append(f"method={cv_details['cv_method']}")
        if 'horizon_minutes' in cv_details:
            cv_parts.append(f"horizon={cv_details['horizon_minutes']}")
        if 'purge_minutes' in cv_details:
            cv_parts.append(f"purge={cv_details['purge_minutes']}")
        # Embargo: extract scalar value if tagged
        embargo_val = cv_details.get('embargo_minutes')
        if embargo_val:
            if isinstance(embargo_val, dict) and embargo_val.get('kind') == 'scalar':
                cv_parts.append(f"embargo={embargo_val['value']}")
            elif isinstance(embargo_val, (int, float)):
                cv_parts.append(f"embargo={embargo_val}")
        # Folds: extract scalar value if tagged
        folds_val = cv_details.get('folds')
        if folds_val:
            if isinstance(folds_val, dict) and folds_val.get('kind') == 'scalar':
                cv_parts.append(f"folds={folds_val['value']}")
            elif isinstance(folds_val, (int, float)):
                cv_parts.append(f"folds={folds_val}")
        if cv_parts:
            parts.append(f"cv:{'|'.join(cv_parts)}")
    
    # Feature and label definitions
    if feature_registry_hash:
        parts.append(f"features={feature_registry_hash}")
    if label_definition_hash:
        parts.append(f"label={label_definition_hash}")
    
    # Compute hash
    key_str = "|".join(parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


class Stage(str, Enum):
    """Pipeline stage constants."""
    TARGET_RANKING = "TARGET_RANKING"
    FEATURE_SELECTION = "FEATURE_SELECTION"
    TRAINING = "TRAINING"
    MODEL_TRAINING = "MODEL_TRAINING"  # Alias for TRAINING
    PLANNING = "PLANNING"


class RouteType(str, Enum):
    """Route type constants for feature selection and training."""
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    INDIVIDUAL = "INDIVIDUAL"


class TargetRankingView(str, Enum):
    """View constants for target ranking evaluation."""
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"
    LOSO = "LOSO"  # Leave-One-Symbol-Out (optional)


def get_main_logger() -> logging.Logger:
    """Try to get the main script's logger for better log integration"""
    # Check common logger names used in scripts (in order of preference)
    for logger_name in ['rank_target_predictability', 'multi_model_feature_selection', '__main__']:
        main_logger = logging.getLogger(logger_name)
        if main_logger.handlers:
            return main_logger
    # Fallback to root logger (always has handlers if logging is configured)
    root_logger = logging.getLogger()
    return root_logger


# Alias for backward compatibility
_get_main_logger = get_main_logger


# Tagged union helpers for handling nullable/optional values
def make_tagged_scalar(value: Any) -> Dict[str, Any]:
    """Create a tagged scalar value."""
    return {"kind": "scalar", "value": value}


def make_tagged_not_applicable(reason: str) -> Dict[str, Any]:
    """Create a tagged N/A value."""
    return {"kind": "not_applicable", "reason": reason}


def make_tagged_per_target_feature(
    ref_path: Optional[str] = None,
    ref_sha256: Optional[str] = None,
    rollup: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a tagged 'per-target-feature' value with optional reference and rollup."""
    result = {"kind": "per_target_feature"}
    if ref_path:
        result["ref"] = {"path": ref_path}
        if ref_sha256:
            result["ref"]["sha256"] = ref_sha256
    if rollup:
        result["rollup"] = rollup
    return result


def make_tagged_auto(value: Optional[Any] = None) -> Dict[str, Any]:
    """Create a tagged auto value."""
    result = {"kind": "auto"}
    if value is not None:
        result["value"] = value
    return result


def make_tagged_not_computed(reason: Optional[str] = None) -> Dict[str, Any]:
    """Create a tagged not_computed value."""
    result = {"kind": "not_computed"}
    if reason:
        result["reason"] = reason
    return result


def make_tagged_omitted() -> None:
    """Create a tagged omitted value (None)."""
    return None


def extract_scalar_from_tagged(value: Any, default: Any = None) -> Any:
    """
    Extract scalar value from tagged union or return value as-is if already scalar.
    
    Handles both schema v1 (scalar/null) and v2 (tagged union) formats.
    
    Args:
        value: Tagged union dict or scalar value
        default: Default value if not applicable or not computed
    
    Returns:
        Scalar value or default
    """
    if value is None:
        return default
    
    # If it's a dict with "kind" key, it's a tagged union (schema v2)
    if isinstance(value, dict) and "kind" in value:
        kind = value.get("kind")
        if kind == "scalar":
            return value.get("value", default)
        elif kind == "auto":
            return value.get("value", default)
        elif kind == "per_target_feature":
            # For per-target-feature, return rollup median if available, else default
            rollup = value.get("rollup", {})
            if rollup and "p50" in rollup:
                return rollup["p50"]
            elif rollup and "min" in rollup:
                return rollup["min"]  # Conservative: use min
            return default
        elif kind in ["not_applicable", "not_computed"]:
            return default
        else:
            # Unknown kind, return default
            return default
    
    # Already a scalar (schema v1 or direct value)
    return value


def extract_embargo_minutes(
    metadata: Dict[str, Any],
    cv_details: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Extract embargo_minutes from metadata, handling both v1 and v2 schemas.
    
    For v2 per-target-feature, returns rollup median if available.
    """
    if cv_details is None:
        cv_details = metadata.get("cv_details", {})
    
    embargo_raw = cv_details.get("embargo_minutes") or metadata.get("embargo_minutes")
    result = extract_scalar_from_tagged(embargo_raw)
    
    # Convert to float if numeric
    if result is not None:
        try:
            return float(result)
        except (ValueError, TypeError):
            return None
    return None


def extract_folds(
    metadata: Dict[str, Any],
    cv_details: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """Extract folds from metadata, handling both v1 and v2 schemas."""
    if cv_details is None:
        cv_details = metadata.get("cv_details", {})
    
    folds_raw = cv_details.get("folds") or cv_details.get("cv_folds") or metadata.get("cv_folds")
    result = extract_scalar_from_tagged(folds_raw)
    
    # Convert to int if numeric
    if result is not None:
        try:
            return int(result)
        except (ValueError, TypeError):
            return None
    return None

