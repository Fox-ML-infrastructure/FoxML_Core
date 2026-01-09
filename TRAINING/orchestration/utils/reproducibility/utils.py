# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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
    target: str,
    view: Optional[str],
    symbol: Optional[str],
    date_start: Optional[str],
    date_end: Optional[str],
    cv_details: Optional[Dict[str, Any]],
    feature_registry_hash: Optional[str],
    label_definition_hash: Optional[str],
    min_cs: Optional[int],
    max_cs_samples: Optional[int],
    universe_sig: Optional[str]
) -> str:
    """
    Compute a comparable key for run comparison.
    
    Runs with the same comparable_key should produce similar results
    (allowing for acceptable variance from randomness).
    
    Args:
        stage: Pipeline stage
        target: Target name
        view: Route type (CROSS_SECTIONAL, INDIVIDUAL, etc.)
        view: View type (for TARGET_RANKING)
        symbol: Symbol (for SYMBOL_SPECIFIC)
        date_start: Start timestamp
        date_end: End timestamp
        cv_details: CV configuration details
        feature_registry_hash: Feature registry hash
        label_definition_hash: Label definition hash
        min_cs: Minimum cross-sectional samples
        max_cs_samples: Maximum cross-sectional samples
        universe_sig: Universe identifier
    
    Returns:
        Hex hash of comparable key (16 chars)
    """
    parts = []
    
    # Core identity
    parts.append(f"stage={stage}")
    parts.append(f"target={target}")
    
    # Route/view
    if view:
        parts.append(f"route={view}")
    if view:
        parts.append(f"view={view}")
    if symbol:
        parts.append(f"symbol={symbol}")
    
    # Data range
    if date_start:
        parts.append(f"start={date_start}")
    if date_end:
        parts.append(f"end={date_end}")
    
    # Universe/split config
    if universe_sig:
        parts.append(f"universe={universe_sig}")
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


# Import canonical enums from scope_resolution (single source of truth)
try:
    from TRAINING.orchestration.utils.scope_resolution import View, Stage as ScopeStage
    
    # Re-export Stage with additional values for backward compat
    class Stage(str, Enum):
        """Pipeline stage constants."""
        TARGET_RANKING = "TARGET_RANKING"
        FEATURE_SELECTION = "FEATURE_SELECTION"
        TRAINING = "TRAINING"
        MODEL_TRAINING = "MODEL_TRAINING"  # Alias for TRAINING (deprecated)
        PLANNING = "PLANNING"
        
        @classmethod
        def from_string(cls, s: str) -> "Stage":
            """Normalize string to Stage enum."""
            return ScopeStage.from_string(s) if s in ("TARGET_RANKING", "FEATURE_SELECTION", "TRAINING", "MODEL_TRAINING") else cls(s.upper())
    
    # Alias TargetRankingView to View for backward compat
    class TargetRankingView(str, Enum):
        """View constants for target ranking evaluation."""
        CROSS_SECTIONAL = View.CROSS_SECTIONAL.value
        SYMBOL_SPECIFIC = View.SYMBOL_SPECIFIC.value
        LOSO = "LOSO"  # Leave-One-Symbol-Out (optional, not in scope_resolution)
        
except ImportError:
    # Fallback if scope_resolution not available
    class Stage(str, Enum):
        """Pipeline stage constants."""
        TARGET_RANKING = "TARGET_RANKING"
        FEATURE_SELECTION = "FEATURE_SELECTION"
        TRAINING = "TRAINING"
        MODEL_TRAINING = "MODEL_TRAINING"
        PLANNING = "PLANNING"
    
    class TargetRankingView(str, Enum):
        """View constants for target ranking evaluation."""
        CROSS_SECTIONAL = "CROSS_SECTIONAL"
        SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"
        LOSO = "LOSO"


class RouteType(str, Enum):
    """Route type constants for feature selection and training."""
    CROSS_SECTIONAL = "CROSS_SECTIONAL"
    SYMBOL_SPECIFIC = "SYMBOL_SPECIFIC"
    # DEPRECATED: INDIVIDUAL is an alias for SYMBOL_SPECIFIC - do not use in new code
    INDIVIDUAL = "SYMBOL_SPECIFIC"


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
    
    folds_raw = cv_details.get("folds") or metadata.get("folds")
    result = extract_scalar_from_tagged(folds_raw)
    
    # Convert to int if numeric
    if result is not None:
        try:
            return int(result)
        except (ValueError, TypeError):
            return None
    return None


# =============================================================================
# SST Accessor Functions
# =============================================================================
# These functions extract values using the canonical SST field names,
# with fallbacks for legacy field names during migration.


def extract_n_effective(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[int]:
    """
    Extract sample size using SST field name 'n_effective'.
    
    Accepts legacy names for backward compatibility:
    - n_effective (SST canonical)
    - n_effective_cs
    - n_effective
    - n_samples
    - sample_size
    
    Args:
        data: Primary dict to extract from (metrics, metadata, etc.)
        additional_data: Optional secondary dict to check
    
    Returns:
        Sample size as int, or None if not found
    """
    # SST canonical first, then legacy names
    keys = ['n_effective', 'n_effective_cs', 'n_effective', 'n_samples', 'sample_size']
    
    for key in keys:
        val = data.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                continue
    
    # Check additional_data if provided
    if additional_data:
        for key in keys:
            val = additional_data.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
    
    return None


def extract_universe_sig(
    data: Dict[str, Any],
    cs_config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract universe signature using SST field name 'universe_sig'.
    
    Accepts legacy names for backward compatibility:
    - universe_sig (SST canonical)
    - universe_sig
    - universe_signature
    
    Args:
        data: Primary dict to extract from
        cs_config: Optional nested cs_config dict to check
    
    Returns:
        Universe signature string, or None if not found
    """
    # Check top-level first
    for key in ['universe_sig', 'universe_sig', 'universe_signature']:
        val = data.get(key)
        if val:
            return str(val)
    
    # Check cs_config if provided or nested in data
    config = cs_config or data.get('cs_config', {})
    if isinstance(config, dict):
        for key in ['universe_sig', 'universe_sig', 'universe_signature']:
            val = config.get(key)
            if val:
                return str(val)
    
    return None


def extract_date_range(
    data: Dict[str, Any],
    cohort_metadata: Optional[Dict[str, Any]] = None
) -> tuple[Optional[str], Optional[str]]:
    """
    Extract date range using SST field names 'date_start' and 'date_end'.
    
    Accepts legacy names for backward compatibility:
    - date_start / date_end (SST canonical)
    - date_start / date_end
    - start_ts / end_ts (from date_range dict)
    
    Args:
        data: Primary dict to extract from
        cohort_metadata: Optional cohort metadata dict to check
    
    Returns:
        (date_start, date_end) tuple of ISO format strings, or (None, None)
    """
    date_start = None
    date_end = None
    
    # Check primary data
    date_start = (
        data.get('date_start') or
        data.get('date_start')
    )
    date_end = (
        data.get('date_end') or
        data.get('date_end')
    )
    
    # Check date_range dict
    if date_start is None or date_end is None:
        date_range = data.get('date_range', {})
        if isinstance(date_range, dict):
            date_start = date_start or date_range.get('start_ts')
            date_end = date_end or date_range.get('end_ts')
    
    # Check cohort_metadata if provided
    if cohort_metadata and (date_start is None or date_end is None):
        date_start = date_start or cohort_metadata.get('date_start')
        date_end = date_end or cohort_metadata.get('date_end')
        
        # Check nested date_range in cohort_metadata
        if date_start is None or date_end is None:
            date_range = cohort_metadata.get('date_range', {})
            if isinstance(date_range, dict):
                date_start = date_start or date_range.get('start_ts')
                date_end = date_end or date_range.get('end_ts')
    
    return date_start, date_end


def extract_pos_rate(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Extract positive class rate using SST field name 'pos_rate'.
    
    Accepts legacy names for backward compatibility:
    - pos_rate (SST canonical)
    - positive_rate
    - class_balance
    
    Args:
        data: Primary dict to extract from (metrics, metadata, etc.)
        additional_data: Optional secondary dict to check
    
    Returns:
        Positive rate as float, or None if not found
    """
    keys = ['pos_rate', 'positive_rate', 'class_balance']
    
    for key in keys:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    
    if additional_data:
        for key in keys:
            val = additional_data.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
    
    return None


def extract_feature_counts(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract feature counts using SST field names.
    
    SST fields:
    - n_features_pre: features before pruning
    - n_features_post: features after pruning  
    - n_features_selected: features selected for training
    
    Accepts legacy names for backward compatibility:
    - n_features_pre / features_safe
    - n_features_post / n_features_post_prune / features_final
    - n_features_selected / n_selected
    
    Args:
        data: Primary dict to extract from
        additional_data: Optional secondary dict to check
    
    Returns:
        (n_features_pre, n_features_post, n_features_selected) tuple
    """
    def _extract_int(keys: List[str]) -> Optional[int]:
        for key in keys:
            val = data.get(key)
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    continue
            if additional_data:
                val = additional_data.get(key)
                if val is not None:
                    try:
                        return int(val)
                    except (ValueError, TypeError):
                        continue
        return None
    
    n_pre = _extract_int(['n_features_pre', 'features_safe'])
    n_post = _extract_int(['n_features_post', 'n_features_post_prune', 'features_final'])
    n_selected = _extract_int(['n_features_selected', 'n_selected']) or n_post
    
    return n_pre, n_post, n_selected


def extract_target(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract target identifier using SST field name 'target'.
    
    Accepts legacy names for backward compatibility:
    - target (SST canonical)
    - target
    - target_column
    - target
    
    Args:
        data: Primary dict to extract from
        additional_data: Optional secondary dict to check
    
    Returns:
        Target identifier string, or None if not found
    """
    keys = ['target', 'target', 'target_column', 'target']
    
    for key in keys:
        val = data.get(key)
        if val:
            return str(val)
    
    if additional_data:
        for key in keys:
            val = additional_data.get(key)
            if val:
                return str(val)
    
    return None


def extract_model_family(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract model family using SST field name 'model_family'.
    
    Accepts legacy names for backward compatibility:
    - model_family (SST canonical)
    - family
    
    Args:
        data: Primary dict to extract from
        additional_data: Optional secondary dict to check
    
    Returns:
        Model family string, or None if not found
    """
    keys = ['model_family', 'family']
    
    for key in keys:
        val = data.get(key)
        if val:
            return str(val)
    
    if additional_data:
        for key in keys:
            val = additional_data.get(key)
            if val:
                return str(val)
    
    return None


def extract_run_id(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract run ID using SST field name 'run_id'.
    
    Accepts legacy names for backward compatibility:
    - run_id (SST canonical)
    - timestamp
    
    Args:
        data: Primary dict to extract from
        additional_data: Optional secondary dict to check
    
    Returns:
        Run ID string, or None if not found
    """
    keys = ['run_id', 'timestamp']
    
    for key in keys:
        val = data.get(key)
        if val:
            return str(val)
    
    if additional_data:
        for key in keys:
            val = additional_data.get(key)
            if val:
                return str(val)
    
    return None


def extract_cv_details(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Extract CV details using SST field name 'cv_details'.
    
    Normalizes nested CV configuration into a flat structure.
    
    Args:
        data: Primary dict to extract from
        additional_data: Optional secondary dict to check
    
    Returns:
        Dict with cv_details (may be empty if not found)
    """
    # Check for cv_details dict
    cv_details = data.get('cv_details', {})
    if cv_details:
        return cv_details
    
    if additional_data:
        cv_details = additional_data.get('cv_details', {})
        if cv_details:
            return cv_details
    
    # Build from individual fields
    result = {}
    cv_fields = ['cv_method', 'folds', 'folds', 'purge_minutes', 'embargo_minutes', 
                 'horizon_minutes', 'fold_boundaries_hash', 'label_definition_hash']
    
    for key in cv_fields:
        val = data.get(key)
        if val is not None:
            result[key] = val
    
    if additional_data:
        for key in cv_fields:
            if key not in result:
                val = additional_data.get(key)
                if val is not None:
                    result[key] = val
    
    return result


def extract_purge_minutes(
    data: Dict[str, Any],
    cv_details: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Extract purge minutes using SST field name 'purge_minutes'.
    
    Args:
        data: Primary dict to extract from (metadata)
        cv_details: Optional cv_details dict
    
    Returns:
        Purge minutes as float, or None if not found
    """
    # Check cv_details first
    if cv_details is None:
        cv_details = data.get('cv_details', {})
    
    val = cv_details.get('purge_minutes') or data.get('purge_minutes')
    
    if val is not None:
        # Handle tagged union format
        result = extract_scalar_from_tagged(val)
        if result is not None:
            try:
                return float(result)
            except (ValueError, TypeError):
                pass
    return None


def extract_horizon_minutes(
    data: Dict[str, Any],
    cv_details: Optional[Dict[str, Any]] = None
) -> Optional[float]:
    """
    Extract horizon minutes using SST field name 'horizon_minutes'.
    
    Args:
        data: Primary dict to extract from (metadata)
        cv_details: Optional cv_details dict
    
    Returns:
        Horizon minutes as float, or None if not found
    """
    # Check cv_details first
    if cv_details is None:
        cv_details = data.get('cv_details', {})
    
    val = cv_details.get('horizon_minutes') or data.get('horizon_minutes')
    
    if val is not None:
        # Handle tagged union format
        result = extract_scalar_from_tagged(val)
        if result is not None:
            try:
                return float(result)
            except (ValueError, TypeError):
                pass
    return None


def extract_view(
    data: Dict[str, Any],
    additional_data: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Extract view using SST field name 'view'.
    
    Accepts legacy names for backward compatibility:
    - view (SST canonical)
    - view
    - view
    
    Normalizes INDIVIDUAL -> SYMBOL_SPECIFIC for consistency.
    
    Args:
        data: Primary dict to extract from
        additional_data: Optional secondary dict to check
    
    Returns:
        View string (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.) or None if not found
    """
    keys = ['view', 'view', 'view']
    
    for key in keys:
        val = data.get(key)
        if val:
            # Normalize INDIVIDUAL -> SYMBOL_SPECIFIC
            if val == "INDIVIDUAL":
                return "SYMBOL_SPECIFIC"
            return str(val)
    
    if additional_data:
        for key in keys:
            val = additional_data.get(key)
            if val:
                # Normalize INDIVIDUAL -> SYMBOL_SPECIFIC
                if val == "INDIVIDUAL":
                    return "SYMBOL_SPECIFIC"
                return str(val)
    
    return None


# =============================================================================
# Metrics SST Accessor Functions
# =============================================================================


def extract_auc(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract AUC/primary metric value using SST field names.
    
    Accepts both old flat structure and new grouped structure:
    - Old: 'auc' (deprecated but still supported)
    - New: 'primary_metric.mean' (nested path)
    - New: 'primary_metric.skill_mean' (nested path)
    - Also checks nested structure if data has 'primary_metric' key
    
    Args:
        data: Dict to extract from (metrics, etc.)
    
    Returns:
        Primary metric value as float, or None if not found
    """
    # Try old flat key first (backward compatibility)
    if 'auc' in data:
        val = data.get('auc')
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass
    
    # Try new nested structure
    if 'primary_metric' in data and isinstance(data['primary_metric'], dict):
        pm = data['primary_metric']
        # Try mean first, then skill_mean
        for key in ['mean', 'skill_mean']:
            if key in pm:
                val = pm[key]
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        continue
    
    # Try flattened dot-notation keys (if already flattened)
    for key in ['primary_metric.mean', 'primary_metric.skill_mean']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    
    return None


def extract_logloss(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract logloss using SST field name 'logloss'.
    
    Accepts legacy names: logloss, logloss
    """
    for key in ['logloss', 'logloss']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_pr_auc(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract PR-AUC using SST field name 'pr_auc'.
    
    Accepts legacy names: pr_auc, pr_auc
    """
    for key in ['pr_auc', 'pr_auc']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_runtime_sec(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract runtime using SST field name 'runtime_sec'.
    
    Accepts legacy names: runtime_sec, train_time_sec, wall_clock_time
    """
    for key in ['runtime_sec', 'train_time_sec', 'wall_clock_time']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_peak_memory_mb(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract peak memory using SST field name 'peak_memory_mb'.
    
    Accepts legacy names: peak_memory_mb, peak_ram_mb
    """
    for key in ['peak_memory_mb', 'peak_ram_mb']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_jaccard_top_k(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract Jaccard top-k using SST field name 'jaccard_top_k'.
    
    Accepts legacy names: jaccard_top_k, jaccard_topK
    """
    for key in ['jaccard_top_k', 'jaccard_topK']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_rank_corr(data: Dict[str, Any]) -> Optional[float]:
    """
    Extract rank correlation using SST field name 'rank_corr'.
    
    Accepts legacy names: rank_corr, rank_correlation, spearman_corr
    """
    for key in ['rank_corr', 'rank_correlation', 'spearman_corr']:
        val = data.get(key)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                continue
    return None


def extract_config_hash(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract config hash using SST field name 'config_hash'.
    
    Accepts legacy names: config_hash, config_hash
    """
    for key in ['config_hash', 'config_hash']:
        val = data.get(key)
        if val:
            return str(val)
    return None


def extract_featureset_hash(data: Dict[str, Any]) -> Optional[str]:
    """
    Extract featureset hash using SST field name 'featureset_hash'.
    
    Accepts legacy names: featureset_hash, featureset_hash
    """
    for key in ['featureset_hash', 'featureset_hash']:
        val = data.get(key)
        if val:
            return str(val)
    return None

