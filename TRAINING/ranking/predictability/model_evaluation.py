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
Target Predictability Ranking

Uses multiple model families to evaluate which of your 63 targets are most predictable.
This helps prioritize compute: train models on high-predictability targets first.

Methodology:
1. For each target, train multiple model families on sample data
2. Calculate predictability scores:
   - Model R² scores (cross-validated)
   - Feature importance magnitude (mean absolute SHAP/importance)
   - Consistency across models (low std = high confidence)
3. Rank targets by composite predictability score
4. Output ranked list with recommendations

Usage:
  # Rank all enabled targets
  python SCRIPTS/rank_target_predictability.py
  
  # Test on specific symbols first
  python SCRIPTS/rank_target_predictability.py --symbols AAPL,MSFT,GOOGL
  
  # Rank specific targets
  python SCRIPTS/rank_target_predictability.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import yaml
import json
from collections import defaultdict
import warnings

# Add project root FIRST (before any scripts.* imports)
# TRAINING/ranking/rank_target_predictability.py -> parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg, get_safety_config
    _CONFIG_AVAILABLE = True
except ImportError:
    pass  # Logger not yet initialized, will be set up below

def _get_importance_top_fraction() -> float:
    """Get the top fraction for importance analysis from config."""
    if _CONFIG_AVAILABLE:
        try:
            # Load from feature_selection/multi_model.yaml
            fraction = float(get_cfg("aggregation.importance_top_fraction", default=0.10, config_name="multi_model"))
            return fraction
        except Exception:
            return 0.10  # FALLBACK_DEFAULT_OK
    return 0.10  # FALLBACK_DEFAULT_OK

# Import logging config utilities
try:
    from CONFIG.logging_config_utils import get_module_logging_config, get_backend_logging_config
    _LOGGING_CONFIG_AVAILABLE = True
except ImportError:
    _LOGGING_CONFIG_AVAILABLE = False
    # Fallback: create a simple config-like object
    class _DummyLoggingConfig:
        def __init__(self):
            self.gpu_detail = False
            self.cv_detail = False
            self.edu_hints = False
            self.detail = False

# Import checkpoint utility (after path is set)
from TRAINING.utils.checkpoint import CheckpointManager

# Import unified task type system
from TRAINING.utils.task_types import (
    TaskType, TargetConfig, ModelConfig, 
    is_compatible, create_model_configs_from_yaml
)
from TRAINING.utils.task_metrics import evaluate_by_task, compute_composite_score
from TRAINING.utils.target_validation import validate_target, check_cv_compatibility

# Suppress expected warnings (harmless)
warnings.filterwarnings('ignore', message='X does not have valid feature names')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')
warnings.filterwarnings('ignore', message='invalid value encountered in divide')
warnings.filterwarnings('ignore', message='invalid value encountered in true_divide')

# Setup logging with journald support
from TRAINING.utils.logging_setup import setup_logging
logger = setup_logging(
    script_name="rank_target_predictability",
    level=logging.INFO,
    use_journald=True
)



# Import dependencies
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.predictability.composite_score import calculate_composite_score
from TRAINING.ranking.predictability.data_loading import load_sample_data, prepare_features_and_target, get_model_config
from TRAINING.ranking.predictability.leakage_detection import detect_leakage, _save_feature_importances, _log_suspicious_features, find_near_copy_features, _detect_leaking_features


def _compute_suspicion_score(
    train_score: float,
    cv_score: Optional[float],
    feature_importances: Dict[str, float],
    task_type: str = 'classification'
) -> float:
    """
    Compute suspicion score for perfect train accuracy.
    
    Higher score = more suspicious (likely real leakage, not just overfitting).
    
    Signals that increase suspicion:
    - CV too good to be true (cv_mean >= 0.85)
    - Generalization gap too small with perfect train (gap < 0.05)
    - Single feature domination (top1_importance / sum >= 0.40)
    
    Signals that decrease suspicion:
    - CV is normal-ish (0.55-0.75)
    - Large gap (classic overfit)
    - Feature dominance not extreme
    """
    suspicion = 0.0
    
    # Signal 1: CV too good to be true
    if cv_score is not None:
        if cv_score >= 0.85:
            suspicion += 0.4  # High suspicion
        elif cv_score >= 0.75:
            suspicion += 0.2  # Medium suspicion
        elif cv_score < 0.55:
            suspicion -= 0.2  # Low suspicion (normal performance)
    
    # Signal 2: Generalization gap (small gap with perfect train = suspicious)
    if cv_score is not None:
        gap = train_score - cv_score
        if gap < 0.05 and train_score >= 0.99:
            suspicion += 0.3  # Very suspicious: perfect train but CV also high
        elif gap > 0.20:
            suspicion -= 0.2  # Large gap = classic overfit (less suspicious)
    
    # Signal 3: Feature dominance
    if feature_importances:
        importances = list(feature_importances.values())
        if importances:
            total_importance = sum(importances)
            if total_importance > 0:
                top1_importance = max(importances)
                dominance_ratio = top1_importance / total_importance
                if dominance_ratio >= 0.50:
                    suspicion += 0.3  # Single feature dominates
                elif dominance_ratio >= 0.40:
                    suspicion += 0.2  # High dominance
                elif dominance_ratio < 0.20:
                    suspicion -= 0.1  # Low dominance (less suspicious)
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, suspicion))


def _log_canonical_summary(
    target_name: str,
    target_column: str,
    symbols: List[str],
    time_vals: Optional[np.ndarray],
    interval: Optional[Union[int, str]],
    horizon: Optional[int],
    rows: int,
    features_safe: int,
    features_pruned: int,
    leak_scan_verdict: str,
    auto_fix_verdict: str,
    auto_fix_reason: Optional[str],
    cv_metric: str,
    composite: float,
    leakage_flag: str,
    cohort_path: Optional[str],
    splitter_name: Optional[str] = None,
    purge_minutes: Optional[float] = None,
    embargo_minutes: Optional[float] = None,
    max_feature_lookback_minutes: Optional[float] = None,
    n_splits: Optional[int] = None,
    lookback_budget_minutes: Optional[Union[float, str]] = None,
    purge_include_feature_lookback: Optional[bool] = None,
    gatekeeper_threshold_source: Optional[str] = None
):
    """
    Log canonical run summary block (one block that can be screenshot for PR comments).
    
    This provides a stable anchor for reviewers to quickly understand:
    - What was evaluated
    - Data characteristics
    - Feature pipeline
    - Leakage status
    - Performance metrics
    - Reproducibility path
    """
    # Extract date range from time_vals if available
    date_range = "N/A"
    if time_vals is not None and len(time_vals) > 0:
        try:
            import pandas as pd
            if isinstance(time_vals[0], (int, float)):
                time_series = pd.to_datetime(time_vals, unit='ns')
            else:
                time_series = pd.Series(time_vals)
            if len(time_series) > 0:
                date_range = f"{time_series.min().strftime('%Y-%m-%d')} → {time_series.max().strftime('%Y-%m-%d')}"
        except Exception:
            pass
    
    # Format symbols (show first 5, then count)
    if len(symbols) <= 5:
        symbols_str = ', '.join(symbols)
    else:
        symbols_str = f"{', '.join(symbols[:5])}, ... ({len(symbols)} total)"
    
    # Format interval/horizon
    interval_str = f"{interval}" if interval else "auto"
    horizon_str = f"{horizon}m" if horizon else "N/A"
    
    # Format auto-fix info
    auto_fix_str = auto_fix_verdict
    if auto_fix_reason:
        auto_fix_str += f" (reason={auto_fix_reason})"
    
    logger.info("=" * 60)
    logger.info("TARGET_RANKING SUMMARY")
    logger.info("=" * 60)
    logger.info(f"target: {target_column:<40} horizon: {horizon_str:<8} interval: {interval_str}")
    logger.info(f"symbols: {len(symbols)} ({symbols_str})")
    logger.info(f"date: {date_range}")
    logger.info(f"rows: {rows:<10} features: safe={features_safe} → pruned={features_pruned}")
    logger.info(f"leak_scan: {leak_scan_verdict:<6} auto_fix: {auto_fix_str}")
    logger.info(f"cv: {cv_metric:<25} composite: {composite:.3f}")
    
    # CV splitter and leakage budget details (CRITICAL for audit)
    if splitter_name:
        logger.info(f"splitter: {splitter_name}")
    if n_splits is not None:
        logger.info(f"n_splits: {n_splits}")
    if purge_minutes is not None:
        logger.info(f"purge_minutes: {purge_minutes:.1f}m")
    if embargo_minutes is not None:
        logger.info(f"embargo_minutes: {embargo_minutes:.1f}m")
    if max_feature_lookback_minutes is not None:
        logger.info(f"max_feature_lookback_minutes: {max_feature_lookback_minutes:.1f}m")
    
    # Config trace for leakage detection settings (CRITICAL for auditability)
    logger.info("")
    logger.info("📋 CONFIG TRACE: Leakage Detection Settings")
    logger.info("-" * 60)
    if lookback_budget_minutes is not None:
        if isinstance(lookback_budget_minutes, str):
            logger.info(f"  lookback_budget_minutes: {lookback_budget_minutes} (source: config)")
        else:
            logger.info(f"  lookback_budget_minutes: {lookback_budget_minutes:.1f}m (source: config)")
    else:
        logger.info(f"  lookback_budget_minutes: auto (not set, using actual max)")
    if purge_include_feature_lookback is not None:
        logger.info(f"  purge_include_feature_lookback: {purge_include_feature_lookback} (source: config)")
    else:
        logger.info(f"  purge_include_feature_lookback: N/A (not available)")
    if gatekeeper_threshold_source is not None:
        logger.info(f"  gatekeeper_threshold_source: {gatekeeper_threshold_source}")
    else:
        logger.info(f"  gatekeeper_threshold_source: N/A (not available)")
    logger.info("-" * 60)
    logger.info("")
    
    if cohort_path:
        logger.info(f"repro: {cohort_path}")
    logger.info("=" * 60)

def _enforce_final_safety_gate(
    X: np.ndarray,
    feature_names: List[str],
    resolved_config: Any,
    interval_minutes: float,
    logger: logging.Logger,
    dropped_tracker: Optional[Any] = None  # NEW: Optional DroppedFeaturesTracker
) -> Tuple[np.ndarray, List[str]]:
    """
    Final Gatekeeper: Enforce safety at the last possible moment.
    
    This runs AFTER all loading/merging/sanitization is done.
    It physically drops features that violate the purge limit from the dataframe.
    This is the "worry-free" auto-corrector that handles race conditions.
    
    Why this is needed:
    - Schema loader might add features after sanitization
    - Registry might allow features that violate purge
    - Ghost features might slip through multiple layers
    - This is the absolute last check before data touches the model
    
    Args:
        X: Feature matrix (numpy array)
        feature_names: List of feature names
        resolved_config: ResolvedConfig object with purge_minutes
        interval_minutes: Data interval in minutes
        logger: Logger instance
    
    Returns:
        (filtered_X, filtered_feature_names) tuple
    """
    if X is None or len(feature_names) == 0:
        return X, feature_names
    
    purge_limit = resolved_config.purge_minutes if resolved_config else None
    if purge_limit is None or purge_limit == 0:
        # No purge, no rules - allow all features
        return X, feature_names
    
    # Load over_budget_action from config
    over_budget_action = "drop"  # Default: drop (for backward compatibility)
    try:
        from CONFIG.config_loader import get_cfg
        over_budget_action = get_cfg("safety.leakage_detection.over_budget_action", default="drop", config_name="safety_config")
    except Exception:
        pass
    
    # Load lookback_budget_minutes cap from config (if set)
    # This is the explicit cap that should be enforced
    lookback_budget_cap = None
    budget_cap_provenance = None
    try:
        from CONFIG.config_loader import get_cfg, get_config_path
        budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
        config_path = get_config_path("safety_config")
        budget_cap_provenance = f"safety_config.yaml:{config_path} → safety.leakage_detection.lookback_budget_minutes = {budget_cap_raw} (default='auto')"
        if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
            lookback_budget_cap = float(budget_cap_raw)
    except Exception as e:
        budget_cap_provenance = f"config lookup failed: {e}"
    
    # Define maximum allowed lookback
    # Priority: 1) config cap (lookback_budget_minutes), 2) purge-derived (purge_limit * 0.99)
    safe_lookback_max_source = None
    if lookback_budget_cap is not None:
        # Use explicit cap from config
        safe_lookback_max = lookback_budget_cap
        safe_lookback_max_source = "budget_cap"
        logger.info(f"🛡️ Gatekeeper threshold: {safe_lookback_max:.1f}m (source: lookback_budget_minutes cap)")
        logger.info(f"   📋 CONFIG TRACE: {budget_cap_provenance}")
    else:
        # Fallback to purge-derived limit (with 1% safety buffer)
        safe_lookback_max = purge_limit * 0.99
        safe_lookback_max_source = "purge_derived"
        logger.warning(
            f"⚠️ Gatekeeper threshold: {safe_lookback_max:.1f}m (source: purge_derived, purge={purge_limit:.1f}m). "
            f"Consider setting lookback_budget_minutes cap to avoid circular dependency."
        )
        logger.info(f"   📋 CONFIG TRACE: {budget_cap_provenance}")
    
    # Get feature registry for lookback calculation
    registry = None
    try:
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
    except Exception:
        pass
    
    # CRITICAL: Use apply_lookback_cap() to follow the same structure as all other phases
    # This ensures consistency: same canonical map, same quarantine logic, same invariants
    # Gatekeeper has extra logic (X matrix manipulation, daily pattern heuristics, dropped_tracker),
    # so we use apply_lookback_cap() for the core structure and preserve the extra logic
    from TRAINING.utils.lookback_cap_enforcement import apply_lookback_cap
    from CONFIG.config_loader import get_cfg
    
    # Load policy and log_mode from config
    policy = "drop"  # Gatekeeper uses "drop" by default (over_budget_action controls behavior)
    try:
        policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
    except Exception:
        pass
    
    log_mode = "summary"
    try:
        log_mode = get_cfg("safety.leakage_detection.log_mode", default="summary", config_name="safety_config")
    except Exception:
        pass
    
    # Get feature_time_meta_map and base_interval from resolved_config if available
    feature_time_meta_map = None
    base_interval_minutes = None
    if resolved_config:
        feature_time_meta_map = resolved_config.feature_time_meta_map if hasattr(resolved_config, 'feature_time_meta_map') else None
        base_interval_minutes = resolved_config.base_interval_minutes if hasattr(resolved_config, 'base_interval_minutes') else None
    
    # Use apply_lookback_cap() - follows standard 6-step structure
    # This ensures gatekeeper uses the same canonical map and quarantine logic as all other phases
    cap_result = apply_lookback_cap(
        features=feature_names,
        interval_minutes=interval_minutes,
        cap_minutes=safe_lookback_max,
        policy=policy,
        stage="GATEKEEPER",
        registry=registry,
        feature_time_meta_map=feature_time_meta_map,
        base_interval_minutes=base_interval_minutes,
        log_mode=log_mode
    )
    
    # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
    # This is the authoritative feature set - downstream code must use this, not raw lists
    enforced = cap_result.to_enforced_set(stage="GATEKEEPER", cap_minutes=safe_lookback_max)
    
    # Extract results (for backward compatibility with existing gatekeeper logic)
    safe_features = enforced.features  # Use enforced.features (the truth)
    quarantined_features = list(enforced.quarantined.keys()) + enforced.unknown  # All quarantined
    
    # DIAGNOSTIC: Count features with _d suffix (day-based patterns)
    import re
    day_suffix_features = [f for f in feature_names if re.search(r'_\d+d$', f, re.I)]
    logger.info(f"🔍 GATEKEEPER DIAGNOSTIC: Found {len(day_suffix_features)} features with _Xd suffix pattern")
    if day_suffix_features:
        logger.info(f"   Sample _Xd features: {day_suffix_features[:5]}")
    
    # Build lookup dict from canonical map for per-feature iteration (needed for X matrix manipulation)
    # Use the canonical map from enforced result (SST)
    from TRAINING.utils.leakage_budget import _feat_key
    feature_lookback_dict = {}
    for feat_name in feature_names:
        feat_key = _feat_key(feat_name)
        lookback = enforced.canonical_map.get(feat_key)
        if lookback is None:
            lookback = float("inf")  # Unknown = unsafe
        feature_lookback_dict[feat_name] = lookback
    
    # GATEKEEPER-SPECIFIC: Additional logic for "daily/24h naming pattern" heuristic
    # This is gatekeeper-specific and not part of apply_lookback_cap()
    # We need to check this for features that passed apply_lookback_cap but might still violate purge
    # due to the "daily/24h naming pattern" heuristic
    
    # Build dropped_features and dropped_indices from quarantined_features
    # Also check for "daily/24h naming pattern" heuristic (gatekeeper-specific)
    dropped_features = []
    dropped_indices = []
    violating_features = []  # Track violations for hard_stop/warn modes
    
    # First, add all quarantined features from apply_lookback_cap()
    for idx, feature_name in enumerate(feature_names):
        if feature_name in quarantined_features:
            lookback_minutes = feature_lookback_dict.get(feature_name, float("inf"))
            if lookback_minutes == float("inf"):
                reason = "unknown lookback (cannot infer - treated as unsafe)"
            else:
                reason = f"lookback ({lookback_minutes:.1f}m) > safe_limit ({safe_lookback_max:.1f}m)"
            dropped_features.append((feature_name, reason))
            dropped_indices.append(idx)
            violating_features.append((feature_name, reason))
            continue
        
        # CRITICAL: Use the canonical map lookback directly (single source of truth)
        # The canonical map already includes all inference logic (patterns, heuristics, etc.)
        # So if a feature has lookback > cap in the canonical map, it should be dropped
        # The "daily/24h naming pattern" heuristic is redundant - canonical map already handles it
        lookback_minutes = feature_lookback_dict.get(feature_name)
        if lookback_minutes is None:
            lookback_minutes = float("inf")  # Unknown = unsafe
        
        if lookback_minutes == float("inf"):
            # Unknown lookback - already handled by apply_lookback_cap, but check again for safety
            continue
        
        # Calendar features have 0m lookback and should NEVER be dropped
        is_calendar_feature = (lookback_minutes == 0.0)
        
        # CRITICAL: If canonical map says lookback > cap, drop it (canonical map is the truth)
        # This ensures gatekeeper sees the same lookbacks as POST_PRUNE
        # The canonical map already includes all inference (patterns, heuristics, etc.)
        if lookback_minutes > safe_lookback_max and not is_calendar_feature:
            # This feature should have been quarantined by apply_lookback_cap
            # But if it wasn't (e.g., due to a bug), drop it here as a safety net
            reason = f"lookback ({lookback_minutes:.1f}m) > safe_limit ({safe_lookback_max:.1f}m) [canonical map]"
            dropped_features.append((feature_name, reason))
            dropped_indices.append(idx)
            violating_features.append((feature_name, reason))
    
    # Handle violations based on over_budget_action
    if violating_features:
        if over_budget_action == "hard_stop":
            # Hard-stop: fail the run if any violating feature exists
            violation_list = ", ".join([f"{name} ({reason})" for name, reason in violating_features[:10]])
            if len(violating_features) > 10:
                violation_list += f" ... and {len(violating_features) - 10} more"
            raise RuntimeError(
                f"🚨 OVER_BUDGET VIOLATION (policy: hard_stop - training blocked): "
                f"{len(violating_features)} features exceed purge limit ({purge_limit:.1f}m, safe_lookback_max={safe_lookback_max:.1f}m). "
                f"Violating features: {violation_list}"
            )
        elif over_budget_action == "warn":
            # Warn: allow but log violations (NOT recommended for production)
            logger.warning(
                f"⚠️ OVER_BUDGET VIOLATION (policy: warn - allowing violating features): "
                f"{len(violating_features)} features exceed purge limit ({purge_limit:.1f}m, safe_lookback_max={safe_lookback_max:.1f}m)"
            )
            logger.info(f"   Violating features ({len(violating_features)}):")
            for feat_name, feat_reason in violating_features[:10]:
                logger.warning(f"   ⚠️ {feat_name}: {feat_reason}")
            if len(violating_features) > 10:
                logger.warning(f"   ... and {len(violating_features) - 10} more")
            # Don't drop - just warn
        # else: over_budget_action == "drop" - handled below
    
    # Mutate the Dataframe (drop columns) - only if action is "drop"
    if dropped_features:
        # Log policy context with explicit source
        source_str = safe_lookback_max_source if safe_lookback_max_source else "unknown"
        logger.warning(
            f"🛡️ FINAL GATEKEEPER: Dropping {len(dropped_features)} features that violate lookback threshold "
            f"(threshold={safe_lookback_max:.1f}m, source={source_str}, purge_limit={purge_limit:.1f}m)"
        )
        logger.info(f"   Policy: drop_features (auto-drop violating features)")
        logger.info(f"   Drop list ({len(dropped_features)} features):")
        for feat_name, feat_reason in dropped_features[:10]:  # Show first 10
            logger.warning(f"   🗑️ {feat_name}: {feat_reason}")
        if len(dropped_features) > 10:
            logger.warning(f"   ... and {len(dropped_features) - 10} more")
        
        # CRITICAL: Slice X immediately using enforced.features (no rediscovery)
        # The enforced.features list IS the authoritative order - X columns must match it
        # Build indices for safe features (enforced.features)
        feature_indices = [i for i, name in enumerate(feature_names) if name in enforced.features]
        if len(feature_indices) == len(enforced.features):
            X = X[:, feature_indices]
            feature_names = enforced.features.copy()  # Use enforced.features (the truth)
        else:
            # Fallback: use dropped_indices (shouldn't happen, but safety net)
            logger.warning(
                f"   ⚠️ Gatekeeper: Index mismatch. Expected {len(enforced.features)} features, "
                f"got {len(feature_indices)} indices. Using dropped_indices fallback."
            )
            keep_indices = [i for i in range(X.shape[1]) if i not in dropped_indices]
            X = X[:, keep_indices]
            feature_names = [name for idx, name in enumerate(feature_names) if idx not in dropped_indices]
        
        logger.info(f"   ✅ After final gatekeeper: {X.shape[1]} features remaining")
        
        # NEW: Track dropped features for telemetry with structured reasons
        if dropped_tracker is not None:
            from TRAINING.utils.dropped_features_tracker import DropReason
            
            # Capture input/output for stage record
            input_features_before_gatekeeper = feature_names.copy() if 'feature_names' in locals() else []
            
            # Create structured reasons
            structured_reasons = {}
            for feat_name, reason_str in dropped_features:
                # Parse reason string to extract structured info
                reason_code = "LOOKBACK_CAP"
                measured_value = None
                threshold_value = safe_lookback_max
                
                # Try to extract lookback value from reason string
                import re
                lookback_match = re.search(r'lookback \(([\d.]+)m\)', reason_str)
                if lookback_match:
                    measured_value = float(lookback_match.group(1))
                
                # Get config provenance
                config_provenance = f"lookback_budget_minutes={safe_lookback_max:.1f}m (source={safe_lookback_max_source})"
                
                structured_reasons[feat_name] = DropReason(
                    reason_code=reason_code,
                    stage="gatekeeper",
                    human_reason=reason_str,
                    measured_value=measured_value,
                    threshold_value=threshold_value,
                    config_provenance=config_provenance
                )
            
            # Get config provenance dict
            config_provenance_dict = {
                "safe_lookback_max": safe_lookback_max,
                "safe_lookback_max_source": safe_lookback_max_source,
                "purge_limit": purge_limit,
                "over_budget_action": over_budget_action
            }
            
            dropped_tracker.add_gatekeeper_drops(
                [name for name, _ in dropped_features],
                structured_reasons,
                input_features=input_features_before_gatekeeper,
                output_features=feature_names,  # After drop
                config_provenance=config_provenance_dict
            )
    
    return X, feature_names


def train_and_evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    task_type: TaskType,
    model_families: List[str] = None,
    multi_model_config: Dict[str, Any] = None,
    target_column: str = None,  # For leak reporting and horizon extraction
    data_interval_minutes: int = 5,  # Data bar interval (default: 5-minute bars)
    time_vals: Optional[np.ndarray] = None,  # Timestamps for each sample (for fold timestamp tracking)
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (for consistency)
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    output_dir: Optional[Path] = None,  # Optional output directory for stability snapshots
    resolved_config: Optional[Any] = None,  # NEW: ResolvedConfig with correct purge/embargo (post-pruning)
    dropped_tracker: Optional[Any] = None,  # NEW: Optional DroppedFeaturesTracker for telemetry
    view: str = "CROSS_SECTIONAL",  # View type for REPRODUCIBILITY structure
    symbol: Optional[str] = None  # Symbol name for SYMBOL_SPECIFIC view
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], float, Dict[str, List[Tuple[str, float]]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    """
    Train multiple models and return task-aware metrics + importance magnitude
    
    Args:
        X: Feature matrix
        y: Target array
        feature_names: List of feature names
        task_type: TaskType enum (REGRESSION, BINARY_CLASSIFICATION, MULTICLASS_CLASSIFICATION)
        model_families: List of model family names to use
        multi_model_config: Multi-model config dict
    
    Returns:
        model_metrics: Dict of {model_name: {metric_name: value}} per model (full metrics)
        model_scores: Dict of {model_name: primary_score} per model (backward compat)
        mean_importance: Mean absolute feature importance
        all_suspicious_features: Dict of {model_name: [(feature, importance), ...]}
        all_feature_importances: Dict of {model_name: {feature: importance}}
        fold_timestamps: List of {fold_idx, train_start, train_end, test_start, test_end} per fold
        perfect_correlation_models: Set of model names that triggered perfect correlation warnings
    
    Always returns 7 values, even on error (returns empty dicts, 0.0, empty list, and empty set)
    """
    # Get logging config for this module (at function start)
    if _LOGGING_CONFIG_AVAILABLE:
        log_cfg = get_module_logging_config('rank_target_predictability')
        lgbm_backend_cfg = get_backend_logging_config('lightgbm')
        catboost_backend_cfg = get_backend_logging_config('catboost')
    else:
        log_cfg = _DummyLoggingConfig()
        lgbm_backend_cfg = type('obj', (object,), {'native_verbosity': -1, 'show_sparse_warnings': True})()
        catboost_backend_cfg = type('obj', (object,), {'native_verbosity': 1, 'show_sparse_warnings': True})()
    
    # Initialize return values (ensures we always return 6 values)
    model_metrics = {}
    model_scores = {}
    importance_magnitudes = []
    all_suspicious_features = {}  # {model_name: [(feature, importance), ...]}
    all_feature_importances = {}  # {model_name: {feature: importance}} for detailed export
    fold_timestamps = []  # List of fold timestamp info
    
    try:
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        import lightgbm as lgb
        from TRAINING.utils.purged_time_series_split import PurgedTimeSeriesSplit
        from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
        from TRAINING.utils.feature_pruning import quick_importance_prune
    except Exception as e:
        logger.warning(f"Failed to import required libraries: {e}")
        return {}, {}, 0.0, {}, {}, []
    
    # Helper function for CV with early stopping (for gradient boosting models)
    def cross_val_score_with_early_stopping(model, X, y, cv, scoring, early_stopping_rounds=None, n_jobs=1):
        # Load default early stopping rounds from config
        if early_stopping_rounds is None:
            if _CONFIG_AVAILABLE:
                try:
                    early_stopping_rounds = int(get_cfg("preprocessing.validation.early_stopping_rounds", default=50, config_name="preprocessing_config"))
                except Exception:
                    early_stopping_rounds = 50
            else:
                early_stopping_rounds = 50
        """
        Cross-validation with early stopping support for gradient boosting models.
        
        cross_val_score doesn't support early stopping callbacks, so we need a manual loop.
        This prevents overfitting by stopping when validation performance plateaus.
        """
        scores = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Clone model for this fold
                from sklearn.base import clone
                fold_model = clone(model)
                
                # Train with early stopping
                # Check if model supports early stopping (LightGBM/XGBoost)
                supports_eval_set = hasattr(fold_model, 'fit') and 'eval_set' in fold_model.fit.__code__.co_varnames
                supports_early_stopping = hasattr(fold_model, 'fit') and 'early_stopping_rounds' in fold_model.fit.__code__.co_varnames
                
                if supports_eval_set:
                    # LightGBM style: uses callbacks
                    # Check by module name for reliability (str(type()) can be fragile)
                    model_module = type(fold_model).__module__
                    if 'lightgbm' in model_module.lower():
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
                        )
                    # XGBoost style: early_stopping_rounds is set in constructor (XGBoost 2.0+)
                    # Don't pass it to fit() - it's already in the model
                    elif 'xgboost' in model_module.lower():
                        import xgboost as xgb
                        # XGBoost 2.0+ has early_stopping_rounds in constructor, not fit()
                        # Check if model already has it set, otherwise use eval_set only
                        fold_model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                    else:
                        # Fallback: try eval_set without callbacks
                        fold_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    # Standard fit for models without early stopping
                    fold_model.fit(X_train, y_train)
                
                # Evaluate on validation set
                if scoring == 'r2':
                    from sklearn.metrics import r2_score
                    y_pred = fold_model.predict(X_val)
                    score = r2_score(y_val, y_pred)
                elif scoring == 'roc_auc':
                    from sklearn.metrics import roc_auc_score
                    y_proba = fold_model.predict_proba(X_val)[:, 1] if hasattr(fold_model, 'predict_proba') else fold_model.predict(X_val)
                    if len(np.unique(y_val)) == 2:
                        score = roc_auc_score(y_val, y_proba)
                    else:
                        score = np.nan
                elif scoring == 'accuracy':
                    from sklearn.metrics import accuracy_score
                    y_pred = fold_model.predict(X_val)
                    score = accuracy_score(y_val, y_pred)
                else:
                    # Fallback to default scorer
                    from sklearn.metrics import get_scorer
                    scorer = get_scorer(scoring)
                    score = scorer(fold_model, X_val, y_val)
                
                scores.append(score)
            except Exception as e:
                logger.debug(f"  Fold {fold_idx + 1} failed: {e}")
                scores.append(np.nan)
        
        return np.array(scores)
    
    # NEW: Initialize dropped features tracker for telemetry
    if dropped_tracker is None:
        from TRAINING.utils.dropped_features_tracker import DroppedFeaturesTracker
        dropped_tracker = DroppedFeaturesTracker()
    
    # ARCHITECTURAL IMPROVEMENT: Pre-prune low-importance features before expensive training
    # This reduces noise and prevents "Curse of Dimensionality" issues
    # Drop features with < 0.01% cumulative importance using a fast LightGBM model
    original_feature_count = len(feature_names)
    # Load feature count threshold from config
    try:
        from CONFIG.config_loader import get_cfg
        feature_count_threshold = int(get_cfg("safety.leakage_detection.model_evaluation.feature_count_pruning_threshold", default=100, config_name="safety_config"))
    except Exception:
        feature_count_threshold = 100
    if original_feature_count > feature_count_threshold:  # Only prune if we have many features
        logger.info(f"  Pre-pruning features: {original_feature_count} features")
        
        # Determine task type string for pruning
        if task_type == TaskType.REGRESSION:
            task_str = 'regression'
        elif task_type == TaskType.BINARY_CLASSIFICATION:
            task_str = 'classification'
        else:
            task_str = 'classification'
        
        try:
            # Generate deterministic seed for feature pruning based on target
            from TRAINING.common.determinism import stable_seed_from
            # Use target_column if available, otherwise use default
            target_name_for_seed = target_column if target_column else 'pruning'
            prune_seed = stable_seed_from([target_name_for_seed, 'feature_pruning'])
            
            # Load feature pruning config
            if _CONFIG_AVAILABLE:
                try:
                    cumulative_threshold = get_cfg("preprocessing.feature_pruning.cumulative_threshold", default=0.0001, config_name="preprocessing_config")
                    min_features = get_cfg("preprocessing.feature_pruning.min_features", default=50, config_name="preprocessing_config")
                    n_estimators = get_cfg("preprocessing.feature_pruning.n_estimators", default=50, config_name="preprocessing_config")
                except Exception:
                    cumulative_threshold = 0.0001
                    min_features = 50
                    n_estimators = 50
            else:
                cumulative_threshold = 0.0001
                min_features = 50
                n_estimators = 50
            
            X_pruned, feature_names_pruned, pruning_stats = quick_importance_prune(
                X, y, feature_names,
                cumulative_threshold=cumulative_threshold,
                min_features=min_features,
                task_type=task_str,
                n_estimators=n_estimators,
                random_state=prune_seed
            )
            
            # NEW: Track pruning drops for telemetry with stage record
            if dropped_tracker is not None and 'dropped_features' in pruning_stats:
                # Get config provenance
                config_provenance_dict = {
                    "cumulative_threshold": cumulative_threshold,
                    "min_features": min_features,
                    "n_estimators": n_estimators,
                    "task_type": task_str
                }
                
                dropped_tracker.add_pruning_drops(
                    pruning_stats['dropped_features'],
                    pruning_stats,
                    input_features=feature_names,
                    output_features=feature_names_pruned,
                    config_provenance=config_provenance_dict
                )
            
            if pruning_stats.get('dropped_count', 0) > 0:
                logger.info(f"  ✅ Pruned: {original_feature_count} → {len(feature_names_pruned)} features "
                          f"(dropped {pruning_stats['dropped_count']} low-importance features)")
                
                # Check for duplicates before assignment
                if len(feature_names_pruned) != len(set(feature_names_pruned)):
                    duplicates = [name for name in set(feature_names_pruned) if feature_names_pruned.count(name) > 1]
                    logger.error(f"  🚨 DUPLICATE COLUMN NAMES in pruned features: {duplicates}")
                    raise ValueError(f"Duplicate feature names after pruning: {duplicates}")
                
                feature_names_before_prune = feature_names.copy()
                X = X_pruned
                feature_names = feature_names_pruned
                
                # Log feature set transition
                from TRAINING.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("PRUNER_SELECTED", feature_names, previous_names=feature_names_before_prune, logger_instance=logger)
                
                # CRITICAL: Re-run gatekeeper after pruning (pruning can surface long-lookback features)
                # Pruning drops low-importance features, which might have been masking long-lookback features
                # We must re-enforce the lookback cap on the pruned set
                if resolved_config is not None:
                    logger.info(f"  🔄 Re-running gatekeeper on pruned feature set (pruning may have surfaced long-lookback features)")
                    X, feature_names = _enforce_final_safety_gate(
                        X, feature_names, resolved_config, data_interval_minutes, logger, dropped_tracker=dropped_tracker
                    )
                    logger.info(f"  ✅ After post-prune gatekeeper: {len(feature_names)} features remaining")
                    
                    # CRITICAL: Get EnforcedFeatureSet from gatekeeper (if available)
                    # This is the authoritative feature set after post-prune gatekeeper
                    post_prune_gatekeeper_enforced = None
                    if hasattr(resolved_config, '_gatekeeper_enforced'):
                        post_prune_gatekeeper_enforced = resolved_config._gatekeeper_enforced
                        # Validate that feature_names matches enforced.features
                        if feature_names != post_prune_gatekeeper_enforced.features:
                            logger.warning(
                                f"  ⚠️ Post-prune gatekeeper: feature_names != enforced.features. "
                                f"This indicates a bug - X was sliced but feature_names wasn't updated correctly."
                            )
                            # Fix it: use enforced.features (the truth)
                            feature_names = post_prune_gatekeeper_enforced.features.copy()
            else:
                logger.info(f"  No features pruned (all above threshold)")
                from TRAINING.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("PRUNER_SELECTED", feature_names, previous_names=None, logger_instance=logger)
            
            # CRITICAL: Recompute resolved_config with feature_lookback_max from PRUNED features
            # This prevents paying 1440m purge for features we don't even use
            from TRAINING.utils.leakage_budget import compute_budget
            from TRAINING.utils.resolved_config import compute_feature_lookback_max, create_resolved_config
            
            # Get registry for lookback calculation
            registry = None
            try:
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry()
            except Exception:
                pass
            
            # Compute budget from PRUNED feature set
            if resolved_config and resolved_config.horizon_minutes:
                budget, _, _ = compute_budget(
                    feature_names,
                    data_interval_minutes,
                    resolved_config.horizon_minutes,
                    registry=registry,
                    stage="pre_gatekeeper_prune_check"
                )
                resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
                
                # Enforce leakage policy after pruning (final feature set)
                # Design: purge covers feature lookback, embargo covers target horizon
                if resolved_config.purge_minutes is not None:
                    purge_minutes = resolved_config.purge_minutes
                    embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else purge_minutes
                    
                    # Load policy and buffer from config
                    policy = "strict"
                    buffer_minutes = 5.0  # Default
                    try:
                        from CONFIG.config_loader import get_cfg
                        policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                        buffer_minutes = float(get_cfg("safety.leakage_detection.lookback_buffer_minutes", default=5.0, config_name="safety_config"))
                    except Exception:
                        pass
                    
                    # Constraint 1: purge must cover feature lookback
                    purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                    purge_violation = purge_minutes < purge_required
                    
                    # Constraint 2: embargo must cover target horizon
                    # Guard: horizon_minutes may be None (e.g., for some target types)
                    if budget.horizon_minutes is not None:
                        embargo_required = budget.horizon_minutes + buffer_minutes
                        embargo_violation = embargo_minutes < embargo_required
                    else:
                        # If horizon is None, skip embargo validation (not applicable)
                        embargo_violation = False
                        embargo_required = None
                    
                    if purge_violation or embargo_violation:
                        violations = []
                        if purge_violation:
                            violations.append(
                                f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m) "
                                f"[max_lookback={budget.max_feature_lookback_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        if embargo_violation:
                            violations.append(
                                f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m) "
                                f"[horizon={budget.horizon_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        
                        msg = f"🚨 LEAKAGE VIOLATION (post-pruning): {'; '.join(violations)}"
                        
                        if policy == "strict":
                            raise RuntimeError(msg + " (policy: strict - training blocked)")
                        elif policy == "warn":
                            logger.error(msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
                        # Note: drop_features policy already handled in gatekeeper, so we just warn here
                    elif embargo_required is None:
                        # Log that embargo validation was skipped due to missing horizon
                        logger.debug(f"   ℹ️  Embargo validation skipped: horizon_minutes is None (not applicable for this target type)")
            
            # Get n_symbols_available from mtf_data
            n_symbols_available = len(mtf_data) if 'mtf_data' in locals() else 1
            
            # Load lookback cap from config
            # Priority: 1) lookback_budget_minutes (new explicit cap), 2) ranking_mode_max_lookback_minutes (legacy)
            max_lookback_cap = None
            try:
                from CONFIG.config_loader import get_cfg
                # Try new explicit cap first
                budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                    max_lookback_cap = float(budget_cap_raw)
                    logger.debug(f"Using lookback_budget_minutes cap: {max_lookback_cap:.1f}m")
                else:
                    # Fallback to legacy ranking_mode_max_lookback_minutes
                    max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
                    if max_lookback_cap is not None:
                        max_lookback_cap = float(max_lookback_cap)
                        logger.debug(f"Using ranking_mode_max_lookback_minutes cap: {max_lookback_cap:.1f}m (legacy)")
            except Exception:
                pass
            
            # Compute feature lookback from PRUNED features
            # Get fingerprint for validation
            from TRAINING.utils.cross_sectional_data import _log_feature_set, _compute_feature_fingerprint
            _log_feature_set("POST_PRUNE", feature_names, previous_names=None, logger_instance=logger)
            post_prune_fp, post_prune_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
            # CRITICAL: Store feature_names for invariant check later
            post_prune_feature_names = feature_names.copy()  # Store for later comparison
            
            # CRITICAL: Use lookback_budget_minutes cap (if set) for POST_PRUNE recompute
            # This ensures consistency with gatekeeper threshold
            lookback_budget_cap_for_recompute = None
            try:
                from CONFIG.config_loader import get_cfg
                budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                    lookback_budget_cap_for_recompute = float(budget_cap_raw)
            except Exception:
                pass
            
            # Use lookback_budget_minutes cap if set, else use ranking_mode_max_lookback_minutes
            effective_cap = lookback_budget_cap_for_recompute if lookback_budget_cap_for_recompute is not None else max_lookback_cap
            
            # CRITICAL: Use apply_lookback_cap() to enforce (quarantine unknowns), not just compute
            # This ensures unknowns are dropped at POST_PRUNE, not just logged
            from TRAINING.utils.lookback_cap_enforcement import apply_lookback_cap
            from CONFIG.config_loader import get_cfg
            
            # Load policy
            policy = "drop"  # Default: drop (over_budget_action)
            try:
                policy = get_cfg("safety.leakage_detection.policy", default="drop", config_name="safety_config")
                over_budget_action = get_cfg("safety.leakage_detection.over_budget_action", default="drop", config_name="safety_config")
                # Use over_budget_action for behavior, policy for logging
                if over_budget_action == "hard_stop":
                    policy = "strict"
                elif over_budget_action == "drop":
                    policy = "drop"
            except Exception:
                pass
            
            # Enforce cap (this will quarantine unknowns in strict mode, drop them in drop mode)
            cap_result = apply_lookback_cap(
                features=feature_names,
                interval_minutes=data_interval_minutes,
                cap_minutes=effective_cap,
                policy=policy,
                stage="POST_PRUNE",
                registry=registry,
                log_mode="summary"
            )
            
            # CRITICAL: Convert to EnforcedFeatureSet (SST contract)
            enforced_post_prune = cap_result.to_enforced_set(stage="POST_PRUNE", cap_minutes=effective_cap)
            
            # PHASE 2: Create and store FeatureSet artifact for reuse (eliminates recomputation)
            post_prune_artifact = None
            if output_dir is not None:
                try:
                    from TRAINING.utils.feature_set_artifact import create_artifact_from_enforced
                    post_prune_artifact = create_artifact_from_enforced(
                        enforced_post_prune,
                        stage="POST_PRUNE",
                        removal_reasons={f: "pruned" for f in set(feature_names) - set(enforced_post_prune.features)}
                    )
                    artifact_dir = output_dir / "REPRODUCIBILITY" / "FEATURESET_ARTIFACTS"
                    post_prune_artifact.save(artifact_dir)
                except Exception as e:
                    logger.debug(f"  ⚠️  Failed to persist POST_PRUNE artifact: {e}")
            
            # Extract results (for backward compatibility)
            safe_features_post_prune = enforced_post_prune.features  # Use enforced.features (the truth)
            quarantined_post_prune = list(enforced_post_prune.quarantined.keys()) + enforced_post_prune.unknown
            canonical_map_from_post_prune = enforced_post_prune.canonical_map
            
            # CRITICAL: Slice X immediately using enforced.features (no rediscovery)
            # The enforced.features list IS the authoritative order - X columns must match it
            if len(safe_features_post_prune) < len(feature_names):
                logger.info(
                    f"  🔄 POST_PRUNE enforcement: {len(feature_names)} → {len(safe_features_post_prune)} "
                    f"(quarantined={len(quarantined_post_prune)})"
                )
                # Slice X to match enforced.features
                if X is not None and len(X.shape) == 2:
                    # Build indices for safe features (enforced.features)
                    feature_indices = [i for i, f in enumerate(feature_names) if f in enforced_post_prune.features]
                    if feature_indices and len(feature_indices) == len(enforced_post_prune.features):
                        X = X[:, feature_indices]
                    else:
                        logger.warning(
                            f"  ⚠️ POST_PRUNE: Could not slice X (indices mismatch). "
                            f"Expected {len(enforced_post_prune.features)} features, got {len(feature_indices)} indices."
                        )
                feature_names = enforced_post_prune.features.copy()  # Use enforced.features (the truth)
                # Update fingerprint after enforcement
                post_prune_fp, post_prune_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
                post_prune_feature_names = feature_names.copy()  # Update stored list
                # Store EnforcedFeatureSet for downstream use
                post_prune_enforced = enforced_post_prune
                
                # CRITICAL: Hard-fail check: POST_PRUNE must have ZERO unknowns in strict mode
                # This is the contract: post-enforcement stages should never see unknowns
                if len(enforced_post_prune.unknown) > 0:
                    policy = "strict"
                    try:
                        from CONFIG.config_loader import get_cfg
                        policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                    except Exception:
                        pass
                    
                    if policy == "strict":
                        error_msg = (
                            f"🚨 POST_PRUNE CONTRACT VIOLATION: {len(enforced_post_prune.unknown)} features have unknown lookback (inf). "
                            f"In strict mode, post-enforcement stages must have ZERO unknowns. "
                            f"These should have been quarantined at gatekeeper. "
                            f"Sample: {enforced_post_prune.unknown[:10]}"
                        )
                        logger.error(error_msg)
                        raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
                    else:
                        logger.warning(
                            f"⚠️ POST_PRUNE: {len(enforced_post_prune.unknown)} features have unknown lookback (inf). "
                            f"Policy={policy} allows this, but this is unexpected after enforcement."
                        )
                
                # CRITICAL: Boundary assertion - validate feature_names matches POST_PRUNE EnforcedFeatureSet
                from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
                try:
                    assert_featureset_fingerprint(
                        label="POST_PRUNE",
                        expected=post_prune_enforced,
                        actual_features=feature_names,
                        logger_instance=logger,
                        allow_reorder=False  # Strict order check
                    )
                except RuntimeError as e:
                    # This should never happen if we used enforced.features.copy() above
                    logger.error(f"POST_PRUNE assertion failed (unexpected): {e}")
                    # Fix it: use enforced.features (the truth)
                    feature_names = post_prune_enforced.features.copy()
                    logger.info(f"Fixed: Updated feature_names to match post_prune_enforced.features")
            
            # Now compute lookback from SAFE features only (no unknowns)
            # Use the canonical map from enforcement (already computed)
            lookback_result = compute_feature_lookback_max(
                safe_features_post_prune, data_interval_minutes, max_lookback_cap_minutes=effective_cap,
                expected_fingerprint=post_prune_fp,
                stage="POST_PRUNE",
                canonical_lookback_map=canonical_map_from_post_prune  # Use same map from enforcement
            )
            # Handle dataclass return
            if hasattr(lookback_result, 'max_minutes'):
                computed_lookback = lookback_result.max_minutes
                top_offenders = lookback_result.top_offenders
                lookback_fingerprint = lookback_result.fingerprint
                canonical_map_from_post_prune = lookback_result.canonical_lookback_map if hasattr(lookback_result, 'canonical_lookback_map') else None
            else:
                # Tuple return (backward compatibility)
                computed_lookback, top_offenders = lookback_result
                lookback_fingerprint = None
                canonical_map_from_post_prune = None
            
            # Validate fingerprint
            if lookback_fingerprint and lookback_fingerprint != post_prune_fp:
                logger.error(
                    f"🚨 FINGERPRINT MISMATCH (POST_PRUNE): computed={lookback_fingerprint} != expected={post_prune_fp}"
                )
            
            if computed_lookback is not None:
                feature_lookback_max_minutes = computed_lookback
                
                # CRITICAL INVARIANT CHECK: max(lookback_map[features]) == actual_max_from_features
                # This prevents regression and ensures canonical map consistency
                if canonical_map_from_post_prune is not None:
                    from TRAINING.utils.leakage_budget import _feat_key
                    
                    # Extract lookbacks for current features from canonical map
                    feature_lookbacks_from_map = []
                    for feat_name in feature_names:
                        feat_key = _feat_key(feat_name)
                        lookback = canonical_map_from_post_prune.get(feat_key)
                        if lookback is not None and lookback != float("inf"):
                            feature_lookbacks_from_map.append(lookback)
                    
                    if feature_lookbacks_from_map:
                        max_from_map = max(feature_lookbacks_from_map)
                        # Allow small floating-point differences (1.0 minute tolerance)
                        if abs(max_from_map - computed_lookback) > 1.0:
                            error_msg = (
                                f"🚨 INVARIANT VIOLATION (POST_PRUNE): "
                                f"max(canonical_map[features])={max_from_map:.1f}m != "
                                f"computed_lookback={computed_lookback:.1f}m. "
                                f"This indicates canonical map inconsistency. "
                                f"Feature set: {len(feature_names)} features, "
                                f"canonical map entries: {len([k for k in canonical_map_from_post_prune.keys() if k in [_feat_key(f) for f in feature_names]])}"
                            )
                            logger.error(error_msg)
                            
                            # Hard-fail in strict mode
                            policy = "strict"
                            try:
                                from CONFIG.config_loader import get_cfg
                                policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                            except Exception:
                                pass
                            
                            if policy == "strict":
                                raise RuntimeError(error_msg)
                        else:
                            logger.debug(
                                f"✅ INVARIANT CHECK (POST_PRUNE): "
                                f"max(canonical_map[features])={max_from_map:.1f}m == "
                                f"computed_lookback={computed_lookback:.1f}m ✓"
                            )
                
                # SANITY CHECK: Verify top_offenders matches reported max and is from current feature set
                if top_offenders:
                    actual_max_in_list = top_offenders[0][1]
                    current_feature_set = set(feature_names)
                    
                    # Verify all top features are in current feature set (should always be true now)
                    top_feature_names = {f for f, _ in top_offenders[:5]}
                    missing = top_feature_names - current_feature_set
                    if missing:
                        logger.error(
                            f"🚨 CRITICAL: Top lookback features not in current feature set: {missing}. "
                            f"This indicates top_offenders was built from wrong feature set."
                        )
                    
                    # Only warn about max mismatch if fingerprint validation passed (invariant-checked stage)
                    # For POST_PRUNE stage, mismatch is a real error if fingerprint matches
                    if lookback_fingerprint and lookback_fingerprint == post_prune_fp:
                        # This is an invariant-checked stage, so mismatch is a real error
                        if abs(actual_max_in_list - computed_lookback) > 1.0:
                            logger.error(
                                f"🚨 Lookback max mismatch (POST_PRUNE): reported={computed_lookback:.1f}m "
                                f"but top feature={actual_max_in_list:.1f}m. "
                                f"This indicates lookback computation bug."
                            )
                    
                    # CRITICAL: Update resolved_config with the recomputed lookback (from pruned features)
                    # This ensures the budget object reflects the actual feature set
                    if resolved_config is not None:
                        resolved_config.feature_lookback_max_minutes = computed_lookback
                    
                    # Log top features (only if > 4 hours for debugging)
                    if computed_lookback > 240:
                        fingerprint_str = lookback_fingerprint if lookback_fingerprint else (lookback_result.fingerprint if hasattr(lookback_result, 'fingerprint') else 'N/A')
                        logger.info(f"  📊 Feature lookback (POST_PRUNE): max={computed_lookback:.1f}m, fingerprint={fingerprint_str}, n_features={len(feature_names)}")
                        logger.info(f"    Top lookback features: {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
                        
                        # Check if lookback_budget_minutes cap is set
                        try:
                            from CONFIG.config_loader import get_cfg
                            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                                budget_cap = float(budget_cap_raw)
                                if computed_lookback > budget_cap:
                                    exceeding_features = [(f, m) for f, m in top_offenders if m > budget_cap + 1.0]
                                    exceeding_count = len(exceeding_features)
                                    
                                    # CRITICAL: In strict mode, this is a hard-stop
                                    policy = "strict"
                                    try:
                                        from CONFIG.config_loader import get_cfg
                                        policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                                    except Exception:
                                        pass
                                    
                                    error_msg = (
                                        f"🚨 POST_PRUNE CAP VIOLATION: actual_max={computed_lookback:.1f}m > cap={budget_cap:.1f}m. "
                                        f"Feature set contains {exceeding_count} features exceeding cap. "
                                        f"Gatekeeper should have dropped these features. "
                                        f"Top offenders: {', '.join([f'{f}({m:.0f}m)' for f, m in exceeding_features[:10]])}"
                                    )
                                    
                                    if policy == "strict":
                                        raise RuntimeError(error_msg + " (policy: strict - training blocked)")
                                    else:
                                        logger.error(error_msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
                        except Exception:
                            pass
            else:
                feature_lookback_max_minutes = None
            
            # Recompute resolved_config with actual pruned feature lookback
            # This overrides the baseline config created earlier
            # CRITICAL: Use computed_lookback (from POST_PRUNE recompute) not feature_lookback_max_minutes variable
            # The variable might be stale if computed_lookback was None
            if resolved_config is not None:
                # Use the value we just computed and stored in resolved_config (line 867)
                # OR use feature_lookback_max_minutes if computed_lookback was None
                final_lookback = resolved_config.feature_lookback_max_minutes if resolved_config.feature_lookback_max_minutes is not None else feature_lookback_max_minutes
                
                # Override with post-prune config
                resolved_config = create_resolved_config(
                    requested_min_cs=resolved_config.requested_min_cs,
                    n_symbols_available=n_symbols_available,
                    max_cs_samples=resolved_config.max_cs_samples,
                    interval_minutes=resolved_config.interval_minutes,
                    horizon_minutes=resolved_config.horizon_minutes,
                    feature_lookback_max_minutes=final_lookback,  # Use final computed value
                    purge_buffer_bars=resolved_config.purge_buffer_bars,
                    default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
                    features_safe=resolved_config.features_safe,
                    features_dropped_nan=resolved_config.features_dropped_nan,
                    features_final=len(feature_names),  # Updated count
                    view=resolved_config.view,
                    symbol=resolved_config.symbol,
                    feature_names=feature_names,  # Pruned features
                    recompute_lookback=False,  # Already computed above
                    experiment_config=experiment_config  # NEW: Pass experiment_config for base_interval_minutes
                )
                if log_cfg.cv_detail:
                    logger.info(f"  ✅ Resolved config (post-prune): purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
                
                # CRITICAL: Enforce leakage policy after pruning (final feature set)
                if resolved_config.purge_minutes is not None and resolved_config.feature_lookback_max_minutes is not None:
                    from TRAINING.utils.leakage_budget import compute_budget
                    
                    # Get registry
                    registry = None
                    try:
                        from TRAINING.common.feature_registry import get_registry
                        registry = get_registry()
                    except Exception:
                        pass
                    
                    # Compute budget from final pruned features
                    # CRITICAL: This is the ACTUAL budget for the final feature set
                    # Use lookback_budget_minutes cap (if set) for consistency
                    lookback_budget_cap_for_budget = None
                    budget_cap_provenance_budget = None
                    try:
                        from CONFIG.config_loader import get_cfg, get_config_path
                        budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
                        config_path = get_config_path("safety_config")
                        budget_cap_provenance_budget = f"safety_config.yaml:{config_path} → safety.leakage_detection.lookback_budget_minutes = {budget_cap_raw} (default='auto')"
                        if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                            lookback_budget_cap_for_budget = float(budget_cap_raw)
                    except Exception as e:
                        budget_cap_provenance_budget = f"config lookup failed: {e}"
                    
                    # PHASE 2: Reuse POST_PRUNE artifact to eliminate recomputation
                    canonical_map_from_post_prune = None
                    if 'post_prune_artifact' in locals() and post_prune_artifact is not None:
                        # Use canonical map from artifact (single source of truth)
                        canonical_map_from_post_prune = post_prune_artifact.canonical_lookback_map
                        logger.debug(f"  ✅ POST_PRUNE_policy_check: Reusing canonical map from POST_PRUNE artifact (n_features={len(post_prune_artifact.features)})")
                    elif 'lookback_result' in locals() and hasattr(lookback_result, 'canonical_lookback_map'):
                        # Fallback: use from lookback_result (backward compatibility)
                        canonical_map_from_post_prune = lookback_result.canonical_lookback_map
                    elif 'lookback_result' in locals() and hasattr(lookback_result, 'lookback_map'):
                        # Backward compatibility
                        canonical_map_from_post_prune = lookback_result.lookback_map
                    
                    # If we don't have the canonical map, we MUST recompute using compute_feature_lookback_max
                    # to ensure we get the same result as POST_PRUNE
                    if canonical_map_from_post_prune is None:
                        logger.warning(
                            f"⚠️ POST_PRUNE_policy_check: No canonical map available from POST_PRUNE artifact. "
                            f"Recomputing using compute_feature_lookback_max to ensure consistency."
                        )
                        # Recompute using the same function as POST_PRUNE
                        lookback_result_for_policy = compute_feature_lookback_max(
                            feature_names,
                            data_interval_minutes,
                            max_lookback_cap_minutes=lookback_budget_cap_for_budget,
                            expected_fingerprint=post_prune_fp if 'post_prune_fp' in locals() else None,
                            stage="POST_PRUNE_policy_check",
                            registry=registry
                        )
                        if hasattr(lookback_result_for_policy, 'canonical_lookback_map'):
                            canonical_map_from_post_prune = lookback_result_for_policy.canonical_lookback_map
                        elif hasattr(lookback_result_for_policy, 'lookback_map'):
                            canonical_map_from_post_prune = lookback_result_for_policy.lookback_map
                    
                    # Log config trace for budget compute (only if not using artifact)
                    if 'post_prune_artifact' not in locals() or post_prune_artifact is None:
                        logger.info(f"📋 CONFIG TRACE (POST_PRUNE_policy_check budget): {budget_cap_provenance_budget}")
                        logger.info(f"   → max_lookback_cap_minutes passed to compute_budget: {lookback_budget_cap_for_budget}")
                        logger.info(f"   → Computing budget from {len(feature_names)} features, expected fingerprint: {post_prune_fp if 'post_prune_fp' in locals() else 'None'}")
                        logger.info(f"   → Using canonical map from POST_PRUNE: {'YES' if canonical_map_from_post_prune is not None else 'NO (will recompute)'}")
                    
                    budget, budget_fp, budget_order_fp = compute_budget(
                        feature_names,
                        data_interval_minutes,
                        resolved_config.horizon_minutes,
                        registry=registry,
                        max_lookback_cap_minutes=lookback_budget_cap_for_budget,  # Pass cap to compute_budget
                        expected_fingerprint=post_prune_fp if 'post_prune_fp' in locals() else None,
                        stage="POST_PRUNE_policy_check",
                        canonical_lookback_map=canonical_map_from_post_prune,  # CRITICAL: Use same map as POST_PRUNE (from artifact if available)
                        feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config and hasattr(resolved_config, 'feature_time_meta_map') else None,
                        base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None
                    )
                    
                    # Log the computed budget for debugging
                    logger.info(f"   → Budget computed: actual_max={budget.max_feature_lookback_minutes:.1f}m, cap={budget.cap_max_lookback_minutes}, fingerprint={budget_fp}")
                    
                    # CRITICAL: Update resolved_config with the NEW budget (from pruned features)
                    # This ensures budget.actual_max reflects the actual feature set
                    resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
                    
                    # Validate fingerprint
                    if 'post_prune_fp' in locals() and budget_fp != post_prune_fp:
                        logger.error(
                            f"🚨 FINGERPRINT MISMATCH (POST_PRUNE_policy_check): budget={budget_fp} != expected={post_prune_fp}"
                        )
                    purge_minutes = resolved_config.purge_minutes
                    embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else purge_minutes
                    
                    # Load policy and buffer from config
                    policy = "strict"
                    buffer_minutes = 5.0  # Default
                    try:
                        from CONFIG.config_loader import get_cfg
                        policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                        buffer_minutes = float(get_cfg("safety.leakage_detection.lookback_buffer_minutes", default=5.0, config_name="safety_config"))
                    except Exception:
                        pass
                    
                    # Constraint 1: purge must cover feature lookback
                    purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                    purge_violation = purge_minutes < purge_required
                    
                    # Constraint 2: embargo must cover target horizon
                    # Guard: horizon_minutes may be None (e.g., for some target types)
                    if budget.horizon_minutes is not None:
                        embargo_required = budget.horizon_minutes + buffer_minutes
                        embargo_violation = embargo_minutes < embargo_required
                    else:
                        # If horizon is None, skip embargo validation (not applicable)
                        embargo_violation = False
                        embargo_required = None
                    
                    if purge_violation or embargo_violation:
                        violations = []
                        if purge_violation:
                            violations.append(
                                f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m) "
                                f"[max_lookback={budget.max_feature_lookback_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        if embargo_violation:
                            violations.append(
                                f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m) "
                                f"[horizon={budget.horizon_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                            )
                        
                        msg = f"🚨 LEAKAGE VIOLATION (post-pruning): {'; '.join(violations)}"
                        
                        if policy == "strict":
                            raise RuntimeError(msg + " (policy: strict - training blocked)")
                        elif policy == "warn":
                            logger.error(msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
                        # Note: drop_features policy already handled in gatekeeper, so we just warn here
                    elif embargo_required is None:
                        # Log that embargo validation was skipped due to missing horizon
                        logger.debug(f"   ℹ️  Embargo validation skipped: horizon_minutes is None (not applicable for this target type)")
            
            # Save stability snapshot for quick pruning (non-invasive hook)
            # Only save if output_dir is available (optional feature)
            if 'full_importance_dict' in pruning_stats and output_dir is not None:
                try:
                    from TRAINING.stability.feature_importance import save_snapshot_hook
                    # Build REPRODUCIBILITY path for snapshots (same structure as feature importances)
                    target_name_clean = (target_column if target_column else 'unknown').replace('/', '_').replace('\\', '_')
                    
                    # Detect context: FEATURE_SELECTION or TARGET_RANKING
                    # If output_dir is already at target level in FEATURE_SELECTION, use it directly
                    if output_dir and "FEATURE_SELECTION" in output_dir.parts:
                        # We're in FEATURE_SELECTION context - use output_dir directly (already at target level)
                        # output_dir is: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
                        snapshot_base_dir = output_dir
                    else:
                        # TARGET_RANKING context - construct path
                        # Determine base output directory (RESULTS/{run}/)
                        if output_dir and output_dir.name == "target_rankings":
                            base_output_dir = output_dir.parent
                        elif output_dir:
                            base_output_dir = output_dir
                            # Walk up to run level if we're inside REPRODUCIBILITY
                            while "REPRODUCIBILITY" in base_output_dir.parts and base_output_dir.name != "RESULTS":
                                base_output_dir = base_output_dir.parent
                                if not base_output_dir.parent.exists():
                                    break
                        else:
                            # No output_dir provided - skip snapshot
                            snapshot_base_dir = None
                        
                        if snapshot_base_dir:
                            repro_base = base_output_dir / "REPRODUCIBILITY" / "TARGET_RANKING"
                            if view == "SYMBOL_SPECIFIC" and symbol:
                                snapshot_base_dir = repro_base / view / target_name_clean / f"symbol={symbol}"
                            else:
                                snapshot_base_dir = repro_base / view / target_name_clean
                    
                    save_snapshot_hook(
                        target_name=target_column if target_column else 'unknown',
                        method="quick_pruner",
                        importance_dict=pruning_stats['full_importance_dict'],
                        universe_id=view,  # Use view parameter
                        output_dir=snapshot_base_dir,  # Save in REPRODUCIBILITY structure
                        auto_analyze=None,  # Load from config
                    )
                except Exception as e:
                    logger.debug(f"Stability snapshot save failed for quick_pruner (non-critical): {e}")
        except RuntimeError as e:
            # CRITICAL: Re-raise RuntimeError (strict mode violations, etc.)
            # These are safety-critical and should not be swallowed
            if "policy: strict" in str(e) or "training blocked" in str(e):
                logger.error(f"  🚨 Feature pruning failed with strict policy violation: {e}")
                raise  # Re-raise - strict mode violations must abort
            else:
                # Other RuntimeErrors - log and continue (might be recoverable)
                logger.warning(f"  Feature pruning failed: {e}, using all features")
                logger.exception("  Pruning exception details (non-critical):")
        except Exception as e:
            logger.warning(f"  Feature pruning failed: {e}, using all features")
            logger.exception("  Pruning exception details (non-critical):")  # Better error logging
            # Continue with original features (baseline resolved_config already assigned)
    
    # CRITICAL: Create resolved_config AFTER pruning (or if pruning skipped)
    # This ensures feature_lookback_max is computed from actual features used in training
    if resolved_config is None:
        from TRAINING.utils.resolved_config import compute_feature_lookback_max, create_resolved_config
        
        # Get n_symbols_available from cohort_context
        n_symbols_available = len(mtf_data) if 'mtf_data' in locals() else 1
        
        # Load ranking mode cap from config
        max_lookback_cap = None
        try:
            from CONFIG.config_loader import get_cfg
            max_lookback_cap = get_cfg("safety.leakage_detection.ranking_mode_max_lookback_minutes", default=None, config_name="safety_config")
            if max_lookback_cap is not None:
                max_lookback_cap = float(max_lookback_cap)
        except Exception:
            pass
        
        # Compute feature lookback from actual features (pruned or unpruned)
        from TRAINING.utils.cross_sectional_data import _compute_feature_fingerprint
        current_fp, current_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
        
        # CRITICAL INVARIANT CHECK: Verify featureset matches POST_PRUNE (if it exists)
        # This detects featureset mis-wire: if feature_names changed between POST_PRUNE and strict check
        if 'post_prune_fp' in locals() and post_prune_fp is not None:
            # Use reusable invariant check helper (if EnforcedFeatureSet available)
            if 'post_prune_enforced' in locals():
                from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
                assert_featureset_fingerprint(
                    label="MODEL_TRAIN_INPUT",
                    expected=post_prune_enforced,
                    actual_features=feature_names,
                    logger_instance=logger,
                    allow_reorder=False  # Strict order check (default)
                )
            else:
                # Fallback: manual check (for backward compatibility)
                # Check exact list equality first (not just hash)
                if 'post_prune_feature_names' in locals() and feature_names == post_prune_feature_names:
                    logger.debug(
                        f"✅ INVARIANT CHECK PASSED: exact list match, n_features={len(feature_names)}"
                    )
                elif current_fp != post_prune_fp:
                    logger.error(
                        f"🚨 FEATURESET MIS-WIRE DETECTED: current fingerprint={current_fp[:16]} != POST_PRUNE={post_prune_fp[:16]}. "
                        f"Feature list passed to strict check differs from POST_PRUNE. "
                        f"Current n_features={len(feature_names)}, POST_PRUNE fingerprint={post_prune_fp[:16]}. "
                        f"This indicates feature_names was modified or wrong variable passed."
                    )
                    # Log sample differences for debugging
                    if 'post_prune_feature_names' in locals():
                        current_set = set(feature_names)
                        post_prune_set = set(post_prune_feature_names)
                        added = current_set - post_prune_set
                        removed = post_prune_set - current_set
                        if added:
                            logger.error(f"   Added features: {list(added)[:10]}")
                        if removed:
                            logger.error(f"   Removed features: {list(removed)[:10]}")
                        # Check order divergence
                        if not added and not removed and len(feature_names) == len(post_prune_feature_names):
                            for i, (exp, act) in enumerate(zip(post_prune_feature_names, feature_names)):
                                if exp != act:
                                    logger.error(
                                        f"   Order divergence at index {i}: expected={exp}, actual={act}"
                                    )
                                    break
                    raise RuntimeError(
                        f"FEATURESET MIS-WIRE: feature_names passed to strict check (fingerprint={current_fp[:16]}) "
                        f"does not match POST_PRUNE (fingerprint={post_prune_fp[:16]}). "
                        f"This indicates a bug: feature list was modified or wrong variable passed."
                    )
                else:
                    logger.debug(
                        f"✅ INVARIANT CHECK PASSED: current fingerprint={current_fp[:16]} == POST_PRUNE={post_prune_fp[:16]}, "
                        f"n_features={len(feature_names)}"
                    )
        
        lookback_result = compute_feature_lookback_max(
            feature_names, data_interval_minutes, max_lookback_cap_minutes=max_lookback_cap,
            expected_fingerprint=current_fp,
            stage="fallback_lookback_compute"
        )
        # Handle dataclass return
        if hasattr(lookback_result, 'max_minutes'):
            computed_lookback = lookback_result.max_minutes
            top_offenders = lookback_result.top_offenders
            lookback_fingerprint = lookback_result.fingerprint
        else:
            # Tuple return (backward compatibility)
            computed_lookback, top_offenders = lookback_result
            lookback_fingerprint = None
        
        # Validate fingerprint (only if we have it)
        if lookback_fingerprint and lookback_fingerprint != current_fp:
            logger.error(
                f"🚨 FINGERPRINT MISMATCH (fallback): computed={lookback_fingerprint} != expected={current_fp}"
            )
        
        if computed_lookback is not None:
            feature_lookback_max_minutes = computed_lookback
            # SANITY CHECK: Verify top_offenders matches reported max and is from current feature set
            if top_offenders:
                actual_max_in_list = top_offenders[0][1]
                current_feature_set = set(feature_names)
                
                # Verify all top features are in current feature set (should always be true now)
                top_feature_names = {f for f, _ in top_offenders[:5]}
                missing = top_feature_names - current_feature_set
                if missing:
                    logger.error(
                        f"🚨 CRITICAL: Top lookback features not in current feature set: {missing}. "
                        f"This indicates top_offenders was built from wrong feature set."
                    )
                
                # Only warn about max mismatch if fingerprint validation passed (invariant-checked stage)
                # For fallback stage, mismatch might be expected
                if lookback_fingerprint and lookback_fingerprint == current_fp:
                    # This is an invariant-checked stage, so mismatch is a real error
                    if abs(actual_max_in_list - computed_lookback) > 1.0:
                        logger.error(
                            f"🚨 Lookback max mismatch (fallback): reported={computed_lookback:.1f}m "
                            f"but top feature={actual_max_in_list:.1f}m. "
                            f"This indicates lookback computation bug."
                        )
                
                # Log top features (only if > 4 hours for debugging)
                if computed_lookback > 240:
                    fingerprint_str = lookback_fingerprint if lookback_fingerprint else (lookback_result.fingerprint if hasattr(lookback_result, 'fingerprint') else 'N/A')
                    logger.info(f"  📊 Feature lookback analysis: max={computed_lookback:.1f}m, fingerprint={fingerprint_str}")
                    logger.info(f"    Top lookback features (from {len(feature_names)} features): {', '.join([f'{f}({m:.0f}m)' for f, m in top_offenders[:5]])}")
        else:
            # Fallback: use conservative estimate if cannot compute
            if data_interval_minutes is not None and data_interval_minutes > 0:
                max_lookback_bars = 288  # 1 day of 5m bars
                feature_lookback_max_minutes = max_lookback_bars * data_interval_minutes
            else:
                feature_lookback_max_minutes = None
        
        # Extract horizon from target_column if available
        target_horizon_minutes = None
        if target_column:
            try:
                from TRAINING.utils.leakage_filtering import _extract_horizon, _load_leakage_config
                leakage_config = _load_leakage_config()
                target_horizon_minutes = _extract_horizon(target_column, leakage_config)
            except Exception:
                pass
        
        # Create resolved config with actual feature lookback
        resolved_config = create_resolved_config(
            requested_min_cs=1,  # Not used in train_and_evaluate_models context
            n_symbols_available=n_symbols_available,
            max_cs_samples=None,
            interval_minutes=data_interval_minutes,
            horizon_minutes=target_horizon_minutes,
            feature_lookback_max_minutes=feature_lookback_max_minutes,
            purge_buffer_bars=5,
            default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
            features_safe=original_feature_count,
            features_dropped_nan=0,
            features_final=len(feature_names),
            view="CROSS_SECTIONAL",  # Default for train_and_evaluate_models
            symbol=None,
            feature_names=feature_names,
            recompute_lookback=False,  # Already computed above
            experiment_config=experiment_config  # NEW: Pass experiment_config for base_interval_minutes
        )
        
        if log_cfg.cv_detail:
            logger.info(f"  ✅ Resolved config created: purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
    
    # Get CV config (with fallback if multi_model_config is None or cross_validation is None)
    if multi_model_config is None:
        cv_config = {}
        # Try to load from config if multi_model_config not provided
        try:
            from CONFIG.config_loader import get_cfg
            cv_folds = int(get_cfg("training.cv_folds", default=3, config_name="intelligent_training_config"))
            cv_n_jobs = int(get_cfg("training.cv_n_jobs", default=1, config_name="intelligent_training_config"))
        except Exception:
            cv_folds = 3
            cv_n_jobs = 1
    else:
        cv_config = multi_model_config.get('cross_validation', {})
        # Ensure cv_config is never None (handle case where key exists but value is None)
        if cv_config is None:
            cv_config = {}
        # SST: Try to get from config first, then fallback to cv_config or defaults
        try:
            from CONFIG.config_loader import get_cfg
            cv_folds = int(get_cfg("training.cv_folds", default=cv_config.get('cv_folds', 3), config_name="intelligent_training_config"))
            cv_n_jobs = int(get_cfg("training.cv_n_jobs", default=cv_config.get('n_jobs', 1), config_name="intelligent_training_config"))
        except Exception:
            # Fallback to cv_config or defaults if config loader fails
            cv_folds = cv_config.get('cv_folds', 3)
            cv_n_jobs = cv_config.get('n_jobs', 1)
    
    # CRITICAL: Use PurgedTimeSeriesSplit to prevent temporal leakage
    # Standard K-Fold shuffles data randomly, which destroys time patterns
    # TimeSeriesSplit respects time order but doesn't prevent overlap leakage
    # PurgedTimeSeriesSplit enforces a gap between train/test = target horizon
    
    # Calculate purge_overlap based on target horizon
    # Extract target horizon (in minutes) from target column name
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    
    # Auto-detect data interval from timestamps if available, otherwise use parameter
    # CRITICAL: Using wrong interval causes data leakage (e.g., 1m data with 5m assumption leaks 4 minutes)
    if time_vals is not None and len(time_vals) > 1:
        try:
            # Convert to pandas Timestamp if needed
            # Handle both numeric (nanoseconds) and datetime timestamps
            if isinstance(time_vals[0], (int, float, np.integer, np.floating)):
                # Handle numeric timestamps (nanoseconds or Unix timestamp)
                time_series = pd.to_datetime(time_vals, unit='ns')
            elif isinstance(time_vals, np.ndarray) and time_vals.dtype.kind == 'M':
                # Already datetime64 array
                time_series = pd.Series(time_vals)
            else:
                time_series = pd.Series(time_vals)
            
            # Ensure time_series is datetime type for proper diff calculation
            if not pd.api.types.is_datetime64_any_dtype(time_series):
                time_series = pd.to_datetime(time_series)
            
            # CRITICAL: For panel data, multiple rows share the same timestamp
            # Calculate diff on UNIQUE timestamps, not all rows (otherwise median will be 0)
            unique_times = time_series.unique()
            unique_times_sorted = pd.Series(unique_times).sort_values()
            
            # Calculate median time difference between unique timestamps
            time_diffs = unique_times_sorted.diff().dropna()
            # time_diffs should be TimedeltaIndex when time_series is datetime
            if isinstance(time_diffs, pd.TimedeltaIndex) and len(time_diffs) > 0:
                median_diff_minutes = abs(time_diffs.median().total_seconds()) / 60.0
            elif len(time_diffs) > 0:
                # Fallback: if diff didn't produce Timedeltas, calculate manually
                median_diff = time_diffs.median()
                if isinstance(median_diff, pd.Timedelta):
                    median_diff_minutes = abs(median_diff.total_seconds()) / 60.0
                elif isinstance(median_diff, (int, float, np.integer, np.floating)):
                    # Assume nanoseconds if numeric (use abs to handle unsorted timestamps)
                    median_diff_minutes = abs(float(median_diff)) / 1e9 / 60.0
                else:
                    raise ValueError(f"Unexpected median_diff type: {type(median_diff)}")
            else:
                # No differences (all timestamps identical) - use default
                median_diff_minutes = data_interval_minutes
                logger.warning(f"  All timestamps identical, cannot detect interval, using parameter: {data_interval_minutes}m")
            
            # Round to common intervals (1m, 5m, 15m, 30m, 60m)
            common_intervals = [1, 5, 15, 30, 60]
            detected_interval = min(common_intervals, key=lambda x: abs(x - median_diff_minutes))
            
            # Only use auto-detection if it's close to a common interval (load tolerance from config)
            try:
                from CONFIG.config_loader import get_cfg
                tolerance = float(get_cfg("safety.leakage_detection.model_evaluation.interval_detection_tolerance", default=0.2, config_name="safety_config"))
            except Exception:
                tolerance = 0.2
            if abs(median_diff_minutes - detected_interval) / detected_interval < tolerance:
                data_interval_minutes = detected_interval
                logger.info(f"  Auto-detected data interval: {median_diff_minutes:.1f}m → {data_interval_minutes}m (from timestamps)")
            else:
                # Fall back to parameter if detection is unclear
                logger.warning(f"  Auto-detection unclear ({median_diff_minutes:.1f}m), using parameter: {data_interval_minutes}m")
        except Exception as e:
            logger.warning(f"  Failed to auto-detect interval from timestamps: {e}, using parameter: {data_interval_minutes}m")
    else:
        # Use parameter value (default: 5)
        logger.info(f"  Using data interval from parameter: {data_interval_minutes}m")
    
    # CRITICAL FIX: Recompute purge_minutes from FINAL featureset (post-gatekeeper + post-prune)
    # The resolved_config.purge_minutes may have been computed from pre-prune featureset
    # We need to ensure purge is computed from the ACTUAL features used in training
    from TRAINING.utils.leakage_budget import compute_budget
    from TRAINING.utils.resolved_config import derive_purge_embargo
    from TRAINING.utils.cross_sectional_data import _compute_feature_fingerprint
    
    # Compute fingerprint of final featureset for validation
    final_featureset_fp, _ = _compute_feature_fingerprint(feature_names, set_invariant=True)
    
    # Get registry and feature_time_meta_map for budget computation
    registry = None
    feature_time_meta_map = None
    try:
        from TRAINING.common.feature_registry import get_registry
        registry = get_registry()
    except Exception:
        pass
    
    # Get feature_time_meta_map from resolved_config if available
    if resolved_config is not None and hasattr(resolved_config, 'feature_time_meta_map'):
        feature_time_meta_map = resolved_config.feature_time_meta_map
    
    # Compute budget from FINAL featureset (the one actually used in training)
    # This ensures purge is computed from the correct featureset
    budget_final, budget_fp, _ = compute_budget(
        feature_names,
        data_interval_minutes,
        target_horizon_minutes if target_horizon_minutes is not None else 60.0,
        registry=registry,
        max_lookback_cap_minutes=None,  # Don't cap - we want actual max for purge computation
        stage="CV_SPLITTER_CREATION",
        feature_time_meta_map=feature_time_meta_map,
        base_interval_minutes=resolved_config.base_interval_minutes if resolved_config is not None else None
    )
    
    # Validate fingerprint matches
    if budget_fp != final_featureset_fp:
        logger.error(
            f"🚨 FINGERPRINT MISMATCH (CV_SPLITTER): budget={budget_fp} != final_featureset={final_featureset_fp}. "
            f"This indicates a bug in feature set tracking."
        )
    else:
        logger.debug(f"✅ CV_SPLITTER: purge computed from fingerprint={budget_fp[:8]} (matches MODEL_TRAIN_INPUT)")
    
    # Load purge settings from config
    if _CONFIG_AVAILABLE:
        try:
            purge_buffer_bars = int(get_cfg("pipeline.leakage.purge_buffer_bars", default=5, config_name="pipeline_config"))
            purge_include_feature_lookback = get_cfg("safety.leakage_detection.purge_include_feature_lookback", default=True, config_name="safety_config")
        except Exception:
            purge_buffer_bars = 5
            purge_include_feature_lookback = True
    else:
        purge_buffer_bars = 5
        purge_include_feature_lookback = True
    
    # Compute purge from FINAL featureset lookback
    # If purge_include_feature_lookback is True, purge must cover feature lookback
    feature_lookback_max_minutes = budget_final.max_feature_lookback_minutes
    
    # Use centralized derivation function (base purge from horizon)
    purge_minutes_val, embargo_minutes_val = derive_purge_embargo(
        horizon_minutes=target_horizon_minutes,
        interval_minutes=data_interval_minutes,
        feature_lookback_max_minutes=None,  # derive_purge_embargo doesn't use this - we apply it separately below
        purge_buffer_bars=purge_buffer_bars,
        default_purge_minutes=85.0
    )
    
    # CRITICAL: Apply purge_include_feature_lookback policy (same logic as create_resolved_config)
    # If purge_include_feature_lookback=True, purge must be >= feature_lookback_max + interval
    if purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        from TRAINING.utils.duration_parser import enforce_purge_audit_rule, format_duration
        
        purge_in = purge_minutes_val
        lookback_in = feature_lookback_max_minutes
        interval_for_rule = data_interval_minutes
        
        # Enforce audit rule: purge >= lookback_max (with interval-aware rounding)
        purge_out, min_purge, changed = enforce_purge_audit_rule(
            purge_in * 60.0,  # Convert minutes to seconds
            lookback_in * 60.0,  # Convert minutes to seconds
            interval=interval_for_rule * 60.0 if interval_for_rule is not None else None,
            buffer_frac=0.01,  # 1% safety buffer
            strict_greater=True
        )
        
        if changed:
            purge_minutes_val = purge_out.to_minutes()
            logger.info(
                f"⚠️  CV_SPLITTER: Increased purge from {purge_in:.1f}m to {purge_minutes_val:.1f}m "
                f"(min required: {format_duration(min_purge)}) to satisfy purge_include_feature_lookback=True. "
                f"Feature lookback: {lookback_in:.1f}m"
            )
    
    # CRITICAL ASSERT: Verify purge_include_feature_lookback policy is correctly applied
    if purge_include_feature_lookback and feature_lookback_max_minutes is not None:
        min_required_purge = feature_lookback_max_minutes + (data_interval_minutes if data_interval_minutes is not None else 5.0)
        assert purge_minutes_val >= min_required_purge, (
            f"🚨 BUG: purge_include_feature_lookback=True but purge ({purge_minutes_val:.1f}m) < "
            f"required ({min_required_purge:.1f}m = lookback {feature_lookback_max_minutes:.1f}m + interval {data_interval_minutes:.1f}m). "
            f"This indicates the purge_include_feature_lookback logic is not being applied correctly."
        )
    
    # Log purge computation with fingerprint for validation
    logger.info(
        f"📊 CV_SPLITTER: purge_minutes={purge_minutes_val:.1f}m computed from final_featureset "
        f"(fingerprint={final_featureset_fp[:8]}, actual_max_lookback={feature_lookback_max_minutes:.1f}m, "
        f"purge_include_feature_lookback={purge_include_feature_lookback}, "
        f"min_required={feature_lookback_max_minutes + (data_interval_minutes if data_interval_minutes is not None else 5.0):.1f}m if include_lookback=True)"
    )
    
    # CRITICAL: Validate purge doesn't exceed data span (hard-stop if invalid)
    # Check for explicit override config
    allow_invalid_cv = False
    try:
        from CONFIG.config_loader import get_cfg
        allow_invalid_cv = get_cfg("safety.leakage_detection.cv.allow_invalid_cv", default=False, config_name="safety_config")
    except Exception:
        pass
    
    if time_vals is not None and len(time_vals) > 0:
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if hasattr(time_series, 'min') and hasattr(time_series, 'max'):
            try:
                time_min = time_series.min()
                time_max = time_series.max()
                # Handle both datetime and numeric (nanoseconds) timestamps
                if isinstance(time_min, (pd.Timestamp, pd.DatetimeTZDtype)):
                    # Already datetime - use total_seconds()
                    data_span_minutes = (time_max - time_min).total_seconds() / 60.0
                elif isinstance(time_min, (int, float, np.integer, np.floating)):
                    # Numeric (likely nanoseconds) - convert to timedelta
                    time_min_dt = pd.to_datetime(time_min, unit='ns')
                    time_max_dt = pd.to_datetime(time_max, unit='ns')
                    data_span_minutes = (time_max_dt - time_min_dt).total_seconds() / 60.0
                else:
                    # Try to convert to datetime
                    time_min_dt = pd.to_datetime(time_min)
                    time_max_dt = pd.to_datetime(time_max)
                    data_span_minutes = (time_max_dt - time_min_dt).total_seconds() / 60.0
                
                if purge_minutes_val >= data_span_minutes:
                    error_msg = (
                        f"🚨 INVALID CV CONFIGURATION: purge_minutes ({purge_minutes_val:.1f}m) >= data_span ({data_span_minutes:.1f}m). "
                        f"This will produce empty/invalid CV folds. "
                        f"Either: 1) Set lookback_budget_minutes cap to drop long-lookback features, "
                        f"2) Load more data (≥ {purge_minutes_val/1440:.1f} trading days), or "
                        f"3) Disable purge_include_feature_lookback in config."
                    )
                    if allow_invalid_cv:
                        logger.error(f"{error_msg} (override: allow_invalid_cv=true - proceeding anyway)")
                    else:
                        raise RuntimeError(error_msg)
            except RuntimeError:
                raise  # Re-raise RuntimeError (our hard-stop)
            except Exception as e:
                # Other exceptions (type conversion, etc.) - log but don't hard-stop
                logger.warning(f"  Failed to validate purge vs data span: {e}, skipping validation")
    
    purge_time = pd.Timedelta(minutes=purge_minutes_val)
    
    # Check for duplicate column names before training
    if len(feature_names) != len(set(feature_names)):
        duplicates = [name for name in set(feature_names) if feature_names.count(name) > 1]
        logger.error(f"  🚨 DUPLICATE COLUMN NAMES before training: {duplicates}")
        raise ValueError(f"Duplicate feature names before training: {duplicates}")
    
    # Log feature set before training and compute fingerprint
    # CRITICAL: This fingerprint represents the ACTUAL features used in training (POST_PRUNE, not just post-gatekeeper)
    # Pruning happens earlier in this function (line ~635), so feature_names here is the final pruned set
    # All subsequent lookback computations must use this same fingerprint for validation
    from TRAINING.utils.cross_sectional_data import _log_feature_set, _compute_feature_fingerprint
    _log_feature_set("MODEL_TRAIN_INPUT", feature_names, previous_names=None, logger_instance=logger)
    model_train_input_fingerprint, model_train_input_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
    logger.info(f"📊 MODEL_TRAIN_INPUT fingerprint={model_train_input_fingerprint} (n_features={len(feature_names)}, POST_PRUNE)")
    
    # CRITICAL: Validate that purge was computed from the same featureset
    if 'final_featureset_fp' in locals() and final_featureset_fp != model_train_input_fingerprint:
        logger.error(
            f"🚨 FINGERPRINT MISMATCH: purge computed from {final_featureset_fp[:8]} but MODEL_TRAIN_INPUT={model_train_input_fingerprint[:8]}. "
            f"This indicates purge was computed from wrong featureset!"
        )
    elif 'final_featureset_fp' in locals():
        logger.debug(f"✅ Purge fingerprint validation: purge={final_featureset_fp[:8]} == MODEL_TRAIN_INPUT={model_train_input_fingerprint[:8]}")
    
    # Create purged time series split with time-based purging
    # CRITICAL: Validate time_vals alignment and sorting before using time-based purging
    if time_vals is not None and len(time_vals) == len(X):
        # Ensure time_vals is sorted (required for binary search in PurgedTimeSeriesSplit)
        time_series = pd.Series(time_vals) if not isinstance(time_vals, pd.Series) else time_vals
        if not time_series.is_monotonic_increasing:
            logger.warning("⚠️  time_vals is not sorted! Sorting X, y, and time_vals together")
            sort_idx = np.argsort(time_vals)
            X = X[sort_idx]
            y = y[sort_idx]
            time_vals = time_series.iloc[sort_idx].values if isinstance(time_series, pd.Series) else time_series[sort_idx]
            logger.info(f"  Sorted data by timestamp (preserving alignment)")
        
        # PHASE 1: Pre-CV compatibility check for degenerate folds (first-class handling)
        # Check if target is compatible with CV before creating splitter
        from TRAINING.utils.target_validation import check_cv_compatibility
        is_cv_compatible, cv_compatibility_reason = check_cv_compatibility(y, task_type, cv_folds)
        
        # Get degenerate fold fallback policy from config
        cv_degenerate_fallback = "reduce_folds"  # Default
        cv_min_folds = 2  # Default minimum folds
        try:
            from CONFIG.config_loader import get_cfg
            cv_degenerate_fallback = get_cfg("training.cv_degenerate_fallback", default="reduce_folds", config_name="intelligent_training_config")
            cv_min_folds = int(get_cfg("training.cv_min_folds", default=2, config_name="intelligent_training_config"))
        except Exception:
            pass
        
        # Apply fallback policy if target is not CV-compatible
        original_cv_folds = cv_folds
        if not is_cv_compatible:
            logger.info(
                f"  ℹ️  CV compatibility check: {cv_compatibility_reason}. "
                f"Using fallback policy: {cv_degenerate_fallback}"
            )
            
            if cv_degenerate_fallback == "reduce_folds":
                # Reduce folds until compatible or reach minimum
                while cv_folds > cv_min_folds:
                    cv_folds -= 1
                    is_compatible, reason = check_cv_compatibility(y, task_type, cv_folds)
                    if is_compatible:
                        logger.info(
                            f"  ℹ️  Reduced CV folds from {original_cv_folds} to {cv_folds} to handle degenerate target. "
                            f"Reason: {cv_compatibility_reason}"
                        )
                        break
                    cv_compatibility_reason = reason
                
                # If still not compatible at minimum folds, skip CV
                if cv_folds == cv_min_folds and not check_cv_compatibility(y, task_type, cv_folds)[0]:
                    logger.info(
                        f"  ℹ️  Target still degenerate at minimum folds ({cv_min_folds}). "
                        f"Will skip CV and train on full dataset for importance only."
                    )
                    cv_folds = 0  # Signal to skip CV
            elif cv_degenerate_fallback == "skip_cv":
                logger.info(
                    f"  ℹ️  Skipping CV due to degenerate target. "
                    f"Will train on full dataset for importance only. Reason: {cv_compatibility_reason}"
                )
                cv_folds = 0  # Signal to skip CV
            elif cv_degenerate_fallback == "different_splitter":
                # For classification, use StratifiedKFold if available
                logger.info(
                    f"  ℹ️  Using alternative splitter for degenerate target. "
                    f"Reason: {cv_compatibility_reason}"
                )
                # Note: This would require implementing alternative splitter logic
                # For now, fall back to reduce_folds
                cv_folds = max(cv_min_folds, cv_folds - 1)
                logger.info(f"  ℹ️  Falling back to reduce_folds: {cv_folds} folds")
        
        # Create splitter only if we have valid folds
        skip_cv = False
        if cv_folds > 0:
            tscv = PurgedTimeSeriesSplit(
                n_splits=cv_folds, 
                purge_overlap_time=purge_time,
                time_column_values=time_vals
            )
            if log_cfg.cv_detail:
                logger.info(f"  Using PurgedTimeSeriesSplit (TIME-BASED): {cv_folds} folds, purge_time={purge_time}")
        else:
            # Skip CV - will train on full dataset
            tscv = None
            skip_cv = True
            logger.info(f"  ℹ️  Skipping CV (degenerate target). Will train on full dataset for importance only.")
        
        # CRITICAL: Validate CV folds before training to prevent IndexError
        # Convert splitter generator to list to inspect all folds
        if skip_cv:
            # Skip fold validation if CV is skipped
            all_folds = []
            n_folds_generated = 0
            valid_folds = []
            n_valid_folds = 0
        else:
            all_folds = list(tscv.split(X, y))
            n_folds_generated = len(all_folds)
        
        if n_folds_generated == 0 and not skip_cv:
            raise RuntimeError(
                f"🚨 No CV folds generated. This usually means purge/embargo ({purge_time:.1f}m) is too large "
                f"relative to data span. Either: 1) Reduce lookback_budget_minutes cap to drop long-lookback features, "
                f"2) Load more data (≥ {purge_time/1440:.1f} trading days), or "
                f"3) Disable purge_include_feature_lookback in config."
            )
        
        # Determine if this is a classification task
        is_binary = task_type == TaskType.BINARY_CLASSIFICATION
        is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
        is_classification = is_binary or is_multiclass
        
        # Validate each fold (skip if CV is skipped)
        if not skip_cv:
            valid_folds = []
            for fold_idx, (train_idx, test_idx) in enumerate(all_folds):
                # Check that indices are non-empty
                if len(train_idx) == 0:
                    logger.warning(f"  ⚠️  Fold {fold_idx + 1}: Empty training set (skipping)")
                    continue
                if len(test_idx) == 0:
                    logger.warning(f"  ⚠️  Fold {fold_idx + 1}: Empty test set (skipping)")
                    continue
                
                # For classification, check that both classes are present in training set
                if is_classification:
                    train_y = y[train_idx]
                    unique_classes = np.unique(train_y[~np.isnan(train_y)])
                    if len(unique_classes) < 2:
                        logger.warning(
                            f"  ⚠️  Fold {fold_idx + 1}: Training set has only {len(unique_classes)} class(es) "
                            f"(classes: {unique_classes.tolist()}), skipping"
                        )
                        continue
                
                valid_folds.append((train_idx, test_idx))
            
            n_valid_folds = len(valid_folds)
        else:
            n_valid_folds = 0
        
        if n_valid_folds == 0 and not skip_cv:
            raise RuntimeError(
                f"🚨 No valid CV folds after validation. Generated {n_folds_generated} folds, but all were invalid. "
                f"This usually means: 1) purge/embargo ({purge_time:.1f}m) is too large relative to data span, "
                f"2) Target is degenerate (single class or extreme imbalance), or "
                f"3) Data span is insufficient. "
                f"Either: 1) Reduce lookback_budget_minutes cap, 2) Load more data, or "
                f"3) Check target distribution."
            )
        
        if not skip_cv:
            if n_valid_folds < n_folds_generated:
                logger.warning(
                    f"  ⚠️  Only {n_valid_folds}/{n_folds_generated} folds are valid. "
                    f"Proceeding with {n_valid_folds} folds."
                )
            
            # Create a wrapper splitter that only yields valid folds
            class ValidatedSplitter:
                def __init__(self, valid_folds):
                    self.valid_folds = valid_folds
                    self.n_splits = len(valid_folds)
                
                def split(self, X, y=None, groups=None):
                    for train_idx, test_idx in self.valid_folds:
                        yield train_idx, test_idx
                
                def get_n_splits(self, X=None, y=None, groups=None):
                    return self.n_splits
            
            tscv = ValidatedSplitter(valid_folds)
            if log_cfg.cv_detail:
                logger.info(f"  ✅ CV fold validation: {n_valid_folds} valid folds (from {n_folds_generated} generated)")
    else:
        # CRITICAL: Row-count based purging is INVALID for panel data (multiple symbols per timestamp)
        # With 50 symbols, 1 bar = 50 rows. Using row counts causes catastrophic leakage.
        # We MUST fail loudly rather than silently producing invalid results.
        raise ValueError(
            f"CRITICAL: time_vals is required for panel data (cross-sectional). "
            f"Row-count based purging is INVALID when multiple symbols share the same timestamp. "
            f"With {len(X)} samples, row-count purging would cause 100% data leakage. "
            f"Please ensure cross_sectional_data.py returns time_vals."
        )
    
    # Capture fold timestamps if time_vals is provided
    if time_vals is not None and len(time_vals) == len(X):
        try:
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X, y)):
                train_times = time_vals[train_idx]
                test_times = time_vals[test_idx]
                fold_timestamps.append({
                    'fold_idx': fold_idx + 1,
                    'train_start': pd.Timestamp(train_times.min()) if len(train_times) > 0 else None,
                    'train_end': pd.Timestamp(train_times.max()) if len(train_times) > 0 else None,
                    'test_start': pd.Timestamp(test_times.min()) if len(test_times) > 0 else None,
                    'test_end': pd.Timestamp(test_times.max()) if len(test_times) > 0 else None,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                })
            if log_cfg.cv_detail:
                logger.info(f"  Captured timestamps for {len(fold_timestamps)} folds")
        except Exception as e:
            logger.warning(f"  Failed to capture fold timestamps: {e}")
            fold_timestamps = []
    
    if model_families is None:
        # Load from multi-model config if available
        if multi_model_config:
            model_families_dict = multi_model_config.get('model_families', {})
            if model_families_dict is None or not isinstance(model_families_dict, dict):
                logger.warning("model_families in config is None or not a dict. Using defaults.")
                model_families = ['lightgbm', 'random_forest', 'neural_network']
            else:
                model_families = [
                    name for name, config in model_families_dict.items()
                    if config is not None and isinstance(config, dict) and config.get('enabled', False)
                ]
                # Sort for deterministic order (ensures reproducible aggregations)
                model_families = sorted(model_families)
            logger.debug(f"Using {len(model_families)} models from config: {', '.join(model_families)}")
        else:
            model_families = ['lightgbm', 'random_forest', 'neural_network']
    
    # Create ModelConfig objects for this task type
    model_configs = create_model_configs_from_yaml(multi_model_config, task_type) if multi_model_config else []
    # Filter to only enabled model families
    model_configs = [mc for mc in model_configs if mc.name in model_families]
    
    # Note: model_metrics, model_scores, importance_magnitudes already initialized at function start
    
    # Determine task characteristics
    unique_vals = np.unique(y[~np.isnan(y)])
    is_binary = task_type == TaskType.BINARY_CLASSIFICATION
    is_multiclass = task_type == TaskType.MULTICLASS_CLASSIFICATION
    is_classification = is_binary or is_multiclass
    
    # Select scoring metric based on task type
    if task_type == TaskType.REGRESSION:
        scoring = 'r2'
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        scoring = 'roc_auc'
    else:  # MULTICLASS_CLASSIFICATION
        scoring = 'accuracy'
    
    # Helper function to detect perfect correlation (data leakage)
    # Track which models had perfect correlation warnings (for auto-fixer)
    _perfect_correlation_models = set()
    
    # Load thresholds from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            _correlation_threshold = float(leakage_cfg.get('auto_fix_thresholds', {}).get('perfect_correlation', 0.999))
            _suspicious_score_threshold = float(leakage_cfg.get('model_alerts', {}).get('suspicious_score', 0.99))
        except Exception:
            _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
    else:
        # Load from safety config
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                _correlation_threshold = float(leakage_cfg.get('auto_fix_thresholds', {}).get('perfect_correlation', 0.999))
                _suspicious_score_threshold = float(leakage_cfg.get('model_alerts', {}).get('suspicious_score', 0.99))
            except Exception:
                _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
                _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
        else:
            _correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            _suspicious_score_threshold = 0.99  # FALLBACK_DEFAULT_OK
    
    # NOTE: Removed _critical_leakage_detected flag - training accuracy alone is not
    # a reliable leakage signal for tree-based models. Real defense: schema filters + pre-scan.
    
    def _check_for_perfect_correlation(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> bool:
        """
        Check if predictions are perfectly correlated with targets.
        
        NOTE: High training accuracy alone is NOT a reliable signal for leakage, especially
        for tree-based models (Random Forest, LightGBM) which can overfit to 100% training
        accuracy through memorization even without leakage.
        
        This function now only logs a warning for debugging purposes. Real leakage defense
        comes from:
        - Explicit feature filters (schema, pattern-based exclusions)
        - Pre-training near-copy scan
        - Time-purged cross-validation
        
        Returns True if perfect correlation detected (for tracking), but does NOT trigger
        early exit or mark target as LEAKAGE_DETECTED.
        """
        try:
            # Tree-based models can easily overfit to 100% training accuracy
            tree_models = {'random_forest', 'lightgbm', 'xgboost', 'catboost'}
            is_tree_model = model_name.lower() in tree_models
            
            # For classification, check if predictions match exactly
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if len(y_true) == len(y_pred):
                    accuracy = np.mean(y_true == y_pred)
                    # Use > with epsilon to prevent false triggers from rounding
                    # Default epsilon: 1e-6 (prevents 0.9990 == 0.9990 false positives)
                    epsilon = 1e-6
                    if accuracy > (_correlation_threshold + epsilon):  # Configurable threshold (default: 99.9%)
                        metric_name = "training accuracy"
                        
                        if is_tree_model:
                            # Tree models: This is likely just overfitting, not leakage
                            logger.warning(
                                f"  ⚠️  {model_name} reached {accuracy:.1%} {metric_name} "
                                f"(threshold: {_correlation_threshold:.1%}). "
                                f"This may just be overfitting - tree ensembles can memorize training data. "
                                f"Check CV metrics instead. Real leakage defense: schema filters + pre-scan."
                            )
                        else:
                            # Non-tree models: Still suspicious but less likely to be false positive
                            logger.warning(
                                f"  ⚠️  {model_name} reached {accuracy:.1%} {metric_name} "
                                f"(threshold: {_correlation_threshold:.1%}). "
                                f"High training accuracy detected - investigate if CV metrics are also suspiciously high."
                            )
                        
                        _perfect_correlation_models.add(model_name)  # Track for debugging/auto-fixer
                        return True  # Return True for tracking, but don't trigger early exit
            
            # For regression, check correlation
            elif task_type == TaskType.REGRESSION:
                if len(y_true) == len(y_pred):
                    corr = np.corrcoef(y_true, y_pred)[0, 1]
                    # Use > with epsilon to prevent false triggers from rounding
                    # Default epsilon: 1e-6 (prevents 0.9990 == 0.9990 false positives)
                    epsilon = 1e-6
                    if not np.isnan(corr) and abs(corr) > (_correlation_threshold + epsilon):
                        if is_tree_model:
                            logger.warning(
                                f"  ⚠️  {model_name} has correlation {corr:.4f} "
                                f"(threshold: {_correlation_threshold:.4f}). "
                                f"This may just be overfitting - check CV metrics instead."
                            )
                        else:
                            logger.warning(
                                f"  ⚠️  {model_name} has correlation {corr:.4f} "
                                f"(threshold: {_correlation_threshold:.4f}). "
                                f"High correlation detected - investigate if CV metrics are also suspiciously high."
                            )
                        
                        _perfect_correlation_models.add(model_name)  # Track for debugging
                        return True  # Return True for tracking, but don't trigger early exit
        except Exception:
            pass
        return False
    
    # Helper function to compute and store full task-aware metrics
    def _compute_and_store_metrics(model_name: str, model, X: np.ndarray, y: np.ndarray,
                                   primary_score: float, task_type: TaskType):
        """
        Compute full task-aware metrics and store in both model_metrics and model_scores.
        
        Args:
            model_name: Name of the model
            model: Fitted model
            X: Feature matrix (for predictions)
            y: True target values
            primary_score: Primary score from CV (R², AUC, or accuracy)
            task_type: TaskType enum
        """
        # Defensive check: ensure model_scores and model_metrics are dicts
        nonlocal model_scores, model_metrics
        if model_scores is None or not isinstance(model_scores, dict):
            logger.warning(f"model_scores is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_scores = {}
        if model_metrics is None or not isinstance(model_metrics, dict):
            logger.warning(f"model_metrics is None or not a dict in _compute_and_store_metrics, reinitializing")
            model_metrics = {}
        
        # Store primary score for backward compatibility
        model_scores[model_name] = primary_score
        
            # Compute full task-aware metrics
        try:
            # Calculate training accuracy/correlation BEFORE checking for perfect correlation
            # This is needed for auto-fixer to detect high training scores
            training_accuracy = None
            if task_type in {TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION}:
                if hasattr(model, 'predict_proba'):
                    if task_type == TaskType.BINARY_CLASSIFICATION:
                        y_proba = model.predict_proba(X)[:, 1]
                        try:
                            from CONFIG.config_loader import get_cfg
                            binary_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.binary_classification_threshold", default=0.5, config_name="safety_config"))
                        except Exception:
                            binary_threshold = 0.5
                        y_pred_train = (y_proba >= binary_threshold).astype(int)
                    else:
                        y_proba = model.predict_proba(X)
                        y_pred_train = y_proba.argmax(axis=1)
                else:
                    y_pred_train = model.predict(X)
                if len(y) == len(y_pred_train):
                    training_accuracy = np.mean(y == y_pred_train)
            elif task_type == TaskType.REGRESSION:
                y_pred_train = model.predict(X)
                if len(y) == len(y_pred_train):
                    corr = np.corrcoef(y, y_pred_train)[0, 1]
                    if not np.isnan(corr):
                        training_accuracy = abs(corr)  # Store absolute correlation for regression
            
            if task_type == TaskType.REGRESSION:
                y_pred = model.predict(X)
                # Check for perfect correlation (leakage) - this sets _critical_leakage_detected flag
                if _check_for_perfect_correlation(y, y_pred, model_name):
                    logger.error(f"  CRITICAL: {model_name} shows signs of data leakage! Check feature filtering.")
                    # Early exit: don't compute more metrics, return immediately
                    return
                full_metrics = evaluate_by_task(task_type, y, y_pred, return_ic=True)
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)[:, 1]  # Probability of class 1
                    # Load binary classification threshold from config
                    try:
                        from CONFIG.config_loader import get_cfg
                        binary_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.binary_classification_threshold", default=0.5, config_name="safety_config"))
                    except Exception:
                        binary_threshold = 0.5
                    y_pred = (y_proba >= binary_threshold).astype(int)
                else:
                    # Fallback for models without predict_proba
                    y_pred = model.predict(X)
                    y_proba = np.clip(y_pred, 0, 1)  # Assume predictions are probabilities
                # Check for perfect correlation (for debugging/tracking only - not a leakage signal)
                _check_for_perfect_correlation(y, y_pred, model_name)
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            else:  # MULTICLASS_CLASSIFICATION
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X)
                    y_pred = y_proba.argmax(axis=1)
                else:
                    # Fallback: one-hot encode predictions
                    y_pred = model.predict(X)
                    n_classes = len(np.unique(y[~np.isnan(y)]))
                    y_proba = np.eye(n_classes)[y_pred.astype(int)]
                # Check for perfect correlation (for debugging/tracking only - not a leakage signal)
                _check_for_perfect_correlation(y, y_pred, model_name)
                full_metrics = evaluate_by_task(task_type, y, y_proba)
            
            # Store full metrics (training metrics from evaluate_by_task)
            model_metrics[model_name] = full_metrics
            
            # CRITICAL: Overwrite training metrics with CV scores (primary_score is from CV)
            # This ensures model_metrics contains CV scores, not training scores
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name]['r2'] = primary_score  # CV R²
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name]['roc_auc'] = primary_score  # CV AUC
            else:  # MULTICLASS_CLASSIFICATION
                model_metrics[model_name]['accuracy'] = primary_score  # CV accuracy
            
            # Also store training accuracy/correlation for auto-fixer detection
            # This is the in-sample training score (not CV), which is what triggers leakage warnings
            if training_accuracy is not None:
                if task_type == TaskType.REGRESSION:
                    model_metrics[model_name]['training_r2'] = training_accuracy
                else:
                    model_metrics[model_name]['training_accuracy'] = training_accuracy
        except Exception as e:
            logger.warning(f"Failed to compute full metrics for {model_name}: {e}")
            # Fallback to primary score only
            if task_type == TaskType.REGRESSION:
                model_metrics[model_name] = {'r2': primary_score}
            elif task_type == TaskType.BINARY_CLASSIFICATION:
                model_metrics[model_name] = {'roc_auc': primary_score}
            else:
                model_metrics[model_name] = {'accuracy': primary_score}
    
    # Helper function to update both model_scores and model_metrics
    # NOTE: This is now mainly for backward compat - full metrics computed after training
    def _update_model_score(model_name: str, score: float):
        """Update model_scores (backward compat) - full metrics computed separately"""
        model_scores[model_name] = score
    
    # Check for degenerate target BEFORE training models
    # A target is degenerate if it has < 2 unique values or one class has < 2 samples
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.debug(f"    Skipping: Target has only {len(unique_vals)} unique value(s)")
        return {}, {}, 0.0, {}, {}, [], set()  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps, perfect_correlation_models
    
    # For classification, check class balance
    if is_binary or is_multiclass:
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.debug(f"    Skipping: Smallest class has only {min_class_count} sample(s)")
            return {}, {}, 0.0, {}, {}, [], set()  # model_metrics, model_scores, mean_importance, suspicious_features, feature_importances, fold_timestamps, perfect_correlation_models
    
    # LightGBM
    if 'lightgbm' in model_families:
        try:
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                from CONFIG.config_loader import get_cfg
                # SST: All values from config, no hardcoded defaults
                test_enabled = get_cfg('gpu.lightgbm.test_enabled', default=True, config_name='gpu_config')
                test_n_estimators = get_cfg('gpu.lightgbm.test_n_estimators', default=1, config_name='gpu_config')
                test_samples = get_cfg('gpu.lightgbm.test_samples', default=10, config_name='gpu_config')
                test_features = get_cfg('gpu.lightgbm.test_features', default=5, config_name='gpu_config')
                gpu_device_id = get_cfg('gpu.lightgbm.gpu_device_id', default=0, config_name='gpu_config')
                gpu_platform_id = get_cfg('gpu.lightgbm.gpu_platform_id', default=0, config_name='gpu_config')
                try_cuda_first = get_cfg('gpu.lightgbm.try_cuda_first', default=True, config_name='gpu_config')
                preferred_device = get_cfg('gpu.lightgbm.device', default='cuda', config_name='gpu_config')
                
                if test_enabled and try_cuda_first:
                    # Try CUDA first (fastest)
                    try:
                        test_model = lgb.LGBMRegressor(device='cuda', n_estimators=test_n_estimators, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                        test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                        gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                        logger.info(f"  ✅ Using GPU (CUDA) for LightGBM (device_id={gpu_device_id})")
                    except Exception as cuda_error:
                        # Try OpenCL
                        try:
                            test_model = lgb.LGBMRegressor(device='gpu', n_estimators=test_n_estimators, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                            logger.info(f"  ✅ Using GPU (OpenCL) for LightGBM (platform_id={gpu_platform_id}, device_id={gpu_device_id})")
                        except Exception as opencl_error:
                            logger.warning(f"  ⚠️  LightGBM GPU not available (CUDA: {cuda_error}, OpenCL: {opencl_error}), using CPU")
                elif test_enabled and preferred_device in ['cuda', 'gpu']:
                    # Use preferred device directly
                    try:
                        if preferred_device == 'cuda':
                            test_model = lgb.LGBMRegressor(device='cuda', n_estimators=test_n_estimators, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                            gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                        else:
                            test_model = lgb.LGBMRegressor(device='gpu', n_estimators=test_n_estimators, gpu_platform_id=gpu_platform_id, gpu_device_id=gpu_device_id, verbose=lgbm_backend_cfg.native_verbosity)
                            gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                        test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                        logger.info(f"  ✅ Using GPU ({preferred_device.upper()}) for LightGBM")
                    except Exception as gpu_error:
                        logger.warning(f"  ⚠️  LightGBM GPU ({preferred_device}) not available: {gpu_error}, using CPU")
                else:
                    # Skip test, use preferred device from config
                    if preferred_device in ['cuda', 'gpu']:
                        if preferred_device == 'cuda':
                            gpu_params = {'device': 'cuda', 'gpu_device_id': gpu_device_id}
                        else:
                            gpu_params = {'device': 'gpu', 'gpu_platform_id': gpu_platform_id, 'gpu_device_id': gpu_device_id}
                        logger.info(f"  Using GPU ({preferred_device.upper()}) for LightGBM (test disabled)")
            except Exception as e:
                logger.warning(f"  ⚠️  LightGBM GPU config error: {e}, using CPU")
            
            # Get config values
            lgb_config = get_model_config('lightgbm', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(lgb_config, dict):
                lgb_config = {}
            # Remove objective, device, and verbose from config (we set these explicitly)
            # CRITICAL: Remove verbose to prevent double argument error
            lgb_config_clean = {k: v for k, v in lgb_config.items() if k not in ['device', 'objective', 'metric', 'verbose']}
            
            # Set verbose level from backend config
            # Note: verbose is a model constructor parameter, not fit() parameter
            verbose_level = lgbm_backend_cfg.native_verbosity
            
            if is_binary:
                model = lgb.LGBMClassifier(
                    objective='binary',
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=n_classes,
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            else:
                model = lgb.LGBMRegressor(
                    objective='regression',
                    verbose=verbose_level,  # Enable verbose for GPU verification
                    **lgb_config_clean,
                    **gpu_params
                )
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            early_stopping_rounds = lgb_config.get('early_stopping_rounds', 50) if isinstance(lgb_config, dict) else 50
            
            if log_cfg.cv_detail:
                logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for LightGBM")
            scores = cross_val_score_with_early_stopping(
                model, X, y, cv=tscv, scoring=scoring, 
                early_stopping_rounds=early_stopping_rounds, n_jobs=1  # n_jobs=1 for early stopping compatibility
            )
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once on full data (with early stopping on a validation split) to get importance
            # CRITICAL: Use time-aware split (load ratio from config) - don't shuffle time series data
            # Guard against empty arrays
            try:
                from CONFIG.config_loader import get_cfg
                time_split_ratio = float(get_cfg("preprocessing.validation.time_aware_split_ratio", default=0.8, config_name="preprocessing_config"))
                min_samples_for_split = int(get_cfg("preprocessing.validation.min_samples_for_split", default=10, config_name="preprocessing_config"))
            except Exception:
                time_split_ratio = 0.8
                min_samples_for_split = 10
            
            if len(X) < min_samples_for_split:
                logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                split_idx = len(X)
            else:
                split_idx = int(len(X) * time_split_ratio)
                split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
            
            if split_idx < len(X):
                X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                y_train_final, y_val_final = y[:split_idx], y[split_idx:]
            else:
                # Fallback: use all data if too small
                X_train_final, X_val_final = X, X
                y_train_final, y_val_final = y, y
            # Log GPU usage if available (controlled by config)
            if 'device' in gpu_params and log_cfg.gpu_detail:
                logger.info(f"  🚀 Training LightGBM on {gpu_params['device'].upper()} (device_id={gpu_params.get('gpu_device_id', 0)})")
                logger.info(f"  📊 Dataset size: {len(X_train_final)} samples, {X_train_final.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            model.fit(
                X_train_final, y_train_final,
                eval_set=[(X_val_final, y_val_final)],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)]
            )
            
            # Verify GPU was actually used (only if gpu_detail enabled)
            if 'device' in gpu_params and log_cfg.gpu_detail:
                # Check model parameters to see what device was actually used
                try:
                    model_params = model.get_params()
                    actual_device = model_params.get('device', 'unknown')
                    if actual_device != 'cpu':
                        logger.info(f"  ✅ LightGBM confirmed using {actual_device.upper()}")
                    else:
                        logger.warning(f"  ⚠️  LightGBM fell back to CPU despite GPU params")
                        logger.warning(f"     This can happen if dataset is too small or GPU not properly configured")
                except:
                    logger.debug("  Could not verify device from model params")
            
            # CRITICAL: Check for suspiciously high scores (likely leakage)
            has_leak = False
            if not np.isnan(primary_score) and primary_score >= _suspicious_score_threshold:
                # Use task-appropriate metric name
                if task_type == TaskType.REGRESSION:
                    metric_name = "R²"
                elif task_type == TaskType.BINARY_CLASSIFICATION:
                    metric_name = "ROC-AUC"
                else:
                    metric_name = "Accuracy"
                logger.error(f"  🚨 LEAKAGE ALERT: lightgbm {metric_name}={primary_score:.4f} >= 0.99 - likely data leakage!")
                logger.error(f"    Features: {len(feature_names)} features")
                logger.error(f"    Analyzing feature importances to identify leaks...")
                has_leak = True
            
            # LEAK DETECTION: Analyze feature importance for suspicious patterns
            importances = model.feature_importances_
            # Load importance threshold from config
            if _CONFIG_AVAILABLE:
                try:
                    safety_cfg = get_safety_config()
                    # safety_config.yaml has a top-level 'safety' key
                    safety_section = safety_cfg.get('safety', {})
                    leakage_cfg = safety_section.get('leakage_detection', {})
                    importance_threshold = float(leakage_cfg.get('importance', {}).get('single_feature_threshold', 0.50))
                except Exception:
                    importance_threshold = 0.50
            else:
                importance_threshold = 0.50
            
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='lightgbm',
                threshold=importance_threshold,
                force_report=has_leak  # Always report top features if score indicates leak
            )
            if suspicious_features:
                all_suspicious_features['lightgbm'] = suspicious_features
            
            # Store all feature importances for detailed export
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
            # Reindex to match exact feature_names order (fills missing with 0.0)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['lightgbm'] = importance_dict
            
            # Log importance keys vs train input (now guaranteed to match order)
            importance_keys = list(importance_dict.keys())  # Use list to preserve order
            train_input_keys = feature_names  # Already a list
            if len(importance_keys) != len(train_input_keys):
                missing = set(train_input_keys) - set(importance_keys)
                logger.warning(f"  ⚠️  IMPORTANCE_KEYS mismatch: {len(importance_keys)} keys vs {len(train_input_keys)} train features")
                logger.warning(f"    Missing from importance: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
            elif importance_keys == train_input_keys:
                # Keys match AND order matches - safe to log fingerprint
                from TRAINING.utils.cross_sectional_data import _log_feature_set
                _log_feature_set("IMPORTANCE_KEYS", importance_keys, previous_names=feature_names, logger_instance=logger)
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('lightgbm', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top fraction features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importances) * top_fraction))
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
        except Exception as e:
            logger.warning(f"LightGBM failed: {e}")
    
    # Random Forest
    if 'random_forest' in model_families:
        try:
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            
            # Get config values
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                model = RandomForestClassifier(**rf_config)
            else:
                model = RandomForestRegressor(**rf_config)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # Deep trees/GBMs can memorize noise, making feature importance biased.
            # TODO: Future enhancement - use permutation importance calculated on CV test folds
            # For now, this is acceptable but be aware that importance may be inflated
            model.fit(X, y)
            
            # Check for suspicious scores
            has_leak = not np.isnan(primary_score) and primary_score >= _suspicious_score_threshold
            
            # LEAK DETECTION: Analyze feature importance
            importances = model.feature_importances_
            suspicious_features = _detect_leaking_features(
                feature_names, importances, model_name='random_forest', 
                threshold=0.50, force_report=has_leak
            )
            if suspicious_features:
                all_suspicious_features['random_forest'] = suspicious_features
            
            # Store all feature importances for detailed export
            # CRITICAL: Align importance to feature_names order to ensure fingerprint match
            importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
            # Reindex to match exact feature_names order (fills missing with 0.0)
            importance_series = importance_series.reindex(feature_names, fill_value=0.0)
            importance_dict = importance_series.to_dict()
            all_feature_importances['random_forest'] = importance_dict
            
            # Log importance keys vs train input (only once per model, use random_forest as representative)
            # Now guaranteed to match order
            if 'random_forest' not in all_feature_importances or len(all_feature_importances) == 1:
                importance_keys = list(importance_dict.keys())  # Use list to preserve order
                train_input_keys = feature_names  # Already a list
                if len(importance_keys) != len(train_input_keys):
                    missing = set(train_input_keys) - set(importance_keys)
                    logger.warning(f"  ⚠️  IMPORTANCE_KEYS mismatch (random_forest): {len(importance_keys)} keys vs {len(train_input_keys)} train features")
                    logger.warning(f"    Missing from importance: {sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")
                elif importance_keys == train_input_keys:
                    # Keys match AND order matches - safe to log fingerprint
                    from TRAINING.utils.cross_sectional_data import _log_feature_set
                    _log_feature_set("IMPORTANCE_KEYS", importance_keys, previous_names=feature_names, logger_instance=logger)
            
            # Compute and store full task-aware metrics
            _compute_and_store_metrics('random_forest', model, X, y, primary_score, task_type)
            
            # Use percentage of total importance in top fraction features (0-1 scale, interpretable)
            total_importance = np.sum(importances)
            if total_importance > 0:
                top_fraction = _get_importance_top_fraction()
                top_k = max(1, int(len(importances) * top_fraction))
                top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                # Normalize to 0-1: what % of total importance is in top 10%?
                importance_ratio = top_importance_sum / total_importance
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
            
        except Exception as e:
            logger.warning(f"RandomForest failed: {e}")
    
    # Neural Network
    if 'neural_network' in model_families:
        try:
            from sklearn.neural_network import MLPRegressor, MLPClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.compose import TransformedTargetRegressor
            from sklearn.pipeline import Pipeline
            
            # Get config values
            nn_config = get_model_config('neural_network', multi_model_config)
            
            if is_binary or is_multiclass:
                # For classification: Pipeline handles imputation and scaling within CV folds
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPClassifier(**nn_config))
                ]
                pipeline = Pipeline(steps)
                model = pipeline
                y_for_training = y
            else:
                # For regression: Pipeline for features + TransformedTargetRegressor for target
                # This ensures no data leakage - all scaling/imputation happens within CV folds
                feature_steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('model', MLPRegressor(**nn_config))
                ]
                feature_pipeline = Pipeline(feature_steps)
                model = TransformedTargetRegressor(
                    regressor=feature_pipeline,
                    transformer=StandardScaler()
                )
                y_for_training = y
            
            # Neural networks need special handling for degenerate targets
            # Suppress convergence warnings (they're noisy and we handle failures gracefully)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                try:
                    # Pipeline ensures imputation/scaling happens within each CV fold (no leakage)
                    scores = cross_val_score(model, X, y_for_training, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                except ValueError as e:
                    if "least populated class" in str(e) or "too few" in str(e):
                        logger.debug(f"    Neural Network: Target too imbalanced for CV")
                        primary_score = np.nan
                        model_metrics['neural_network'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['neural_network'] = np.nan
                    else:
                        raise
            
            # Fit on raw data (Pipeline handles preprocessing internally)
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            if not np.isnan(primary_score):
                model.fit(X, y_for_training)
                
                # Compute and store full task-aware metrics (Pipeline handles preprocessing)
                _compute_and_store_metrics('neural_network', model, X, y_for_training, primary_score, task_type)
            
            baseline_score = model.score(X, y_for_training)
            
            perm_scores = []
            for i in range(min(10, X.shape[1])):  # Sample 10 features
                X_perm = X.copy()
                # Use deterministic seed for permutation
                from TRAINING.common.determinism import stable_seed_from
                perm_seed = stable_seed_from(['permutation', target_column if 'target_column' in locals() else 'default', f'feature_{i}'])
                np.random.seed(perm_seed)
                np.random.shuffle(X_perm[:, i])
                perm_score = model.score(X_perm, y_for_training)
                perm_scores.append(abs(baseline_score - perm_score))
            
            importance_magnitudes.append(np.mean(perm_scores))
            
        except Exception as e:
            logger.warning(f"NeuralNetwork failed: {e}")
    
    # XGBoost
    if 'xgboost' in model_families:
        try:
            import xgboost as xgb
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                from CONFIG.config_loader import get_cfg
                # SST: All values from config, no hardcoded defaults
                xgb_device = get_cfg('gpu.xgboost.device', default='cpu', config_name='gpu_config')
                xgb_tree_method = get_cfg('gpu.xgboost.tree_method', default='hist', config_name='gpu_config')
                # Note: gpu_id removed in XGBoost 3.1+, use device='cuda:0' format if needed
                # For now, just use 'cuda' for default GPU
                test_enabled = get_cfg('gpu.xgboost.test_enabled', default=True, config_name='gpu_config')
                test_n_estimators = get_cfg('gpu.xgboost.test_n_estimators', default=1, config_name='gpu_config')
                test_samples = get_cfg('gpu.xgboost.test_samples', default=10, config_name='gpu_config')
                test_features = get_cfg('gpu.xgboost.test_features', default=5, config_name='gpu_config')
                
                if xgb_device == 'cuda':
                    if test_enabled:
                        # XGBoost 3.1+ uses device='cuda' with tree_method='hist' (gpu_id removed)
                        try:
                            test_model = xgb.XGBRegressor(tree_method='hist', device='cuda', n_estimators=test_n_estimators, verbosity=0)
                            test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                            gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda'}
                            logger.info("  ✅ Using GPU (CUDA) for XGBoost")
                        except Exception as gpu_test_error:
                            # Try legacy API: tree_method='gpu_hist' (for XGBoost < 2.0)
                            try:
                                test_model = xgb.XGBRegressor(tree_method='gpu_hist', n_estimators=test_n_estimators, verbosity=0)
                                test_model.fit(np.random.rand(test_samples, test_features), np.random.rand(test_samples))
                                gpu_params = {'tree_method': 'gpu_hist'}  # Legacy API doesn't use device parameter
                                logger.info("  ✅ Using GPU (CUDA) for XGBoost (legacy API: gpu_hist)")
                            except Exception as legacy_error:
                                logger.warning(f"  ⚠️  XGBoost GPU test failed (new API: {gpu_test_error}, legacy API: {legacy_error}), falling back to CPU")
                    else:
                        # Skip test, use config values directly
                        gpu_params = {'tree_method': xgb_tree_method, 'device': 'cuda'}
                        logger.info("  Using GPU (CUDA) for XGBoost (test disabled)")
                else:
                    logger.info("  Using CPU for XGBoost (device='cpu' in config)")
            except Exception as e:
                logger.warning(f"  ⚠️  XGBoost GPU config error, using CPU: {e}")
            
            # Get config values
            xgb_config = get_model_config('xgboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(xgb_config, dict):
                xgb_config = {}
            # Remove task-specific parameters (we set these explicitly based on task type)
            # CRITICAL: Extract early_stopping_rounds from config - it goes in constructor for XGBoost 2.0+
            # Also remove tree_method and device if present (we set these from GPU config)
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', None)
            xgb_config_clean = {k: v for k, v in xgb_config.items() 
                              if k not in ['objective', 'eval_metric', 'early_stopping_rounds', 'tree_method', 'device', 'gpu_id']}
            
            # XGBoost 2.0+ requires early_stopping_rounds in constructor, not fit()
            if early_stopping_rounds is not None:
                xgb_config_clean['early_stopping_rounds'] = early_stopping_rounds
            
            # Add GPU params if available (will override any tree_method/device in config)
            xgb_config_clean.update(gpu_params)
            
            if is_binary:
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    **xgb_config_clean
                )
            elif is_multiclass:
                n_classes = len(unique_vals)
                model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    **xgb_config_clean
                )
            else:
                model = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    **xgb_config_clean
                )
            
            # Log GPU usage if available (controlled by config)
            if 'device' in gpu_params and gpu_params.get('device') == 'cuda' and log_cfg.gpu_detail:
                logger.info("  🚀 Training XGBoost on CUDA")
                logger.info(f"  📊 Dataset size: {len(X)} samples, {X.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            # CRITICAL FIX: Use manual CV loop with early stopping for gradient boosting
            # Get early stopping rounds from config (default: 50)
            # NOTE: For XGBoost 2.0+, early_stopping_rounds is set in constructor above, not passed to fit()
            early_stopping_rounds = xgb_config.get('early_stopping_rounds', 50) if isinstance(xgb_config, dict) else 50
            
            logger.info(f"  Using CV with early stopping (rounds={early_stopping_rounds}) for XGBoost")
            try:
                # XGBoost uses same early stopping interface as LightGBM
                scores = cross_val_score_with_early_stopping(
                    model, X, y, cv=tscv, scoring=scoring,
                    early_stopping_rounds=early_stopping_rounds, n_jobs=1
                )
                valid_scores = scores[~np.isnan(scores)]
                primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            except ValueError as e:
                if "Invalid classes" in str(e) or "Expected" in str(e):
                    logger.debug(f"    XGBoost: Target degenerate in some CV folds")
                    primary_score = np.nan
                    model_metrics['xgboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                    model_scores['xgboost'] = np.nan
                else:
                    raise
            
            # Train once on full data (with early stopping) to get importance and full metrics
            # CRITICAL: Use time-aware split (last 20% as validation) - don't shuffle time series data
            if not np.isnan(primary_score):
                # Guard against empty arrays
                if len(X) < 10:
                    logger.warning(f"  ⚠️  Too few samples ({len(X)}) for train/val split, fitting on all data")
                    split_idx = len(X)
                else:
                    # Load time-aware split ratio from config
                    try:
                        from CONFIG.config_loader import get_cfg
                        time_split_ratio = float(get_cfg("preprocessing.validation.time_aware_split_ratio", default=0.8, config_name="preprocessing_config"))
                    except Exception:
                        time_split_ratio = 0.8
                    split_idx = int(len(X) * time_split_ratio)
                    split_idx = max(1, split_idx)  # Ensure at least 1 sample in validation
                
                if split_idx < len(X):
                    X_train_final, X_val_final = X[:split_idx], X[split_idx:]
                    y_train_final, y_val_final = y[:split_idx], y[split_idx:]
                else:
                    # Fallback: use all data if too small
                    X_train_final, X_val_final = X, X
                    y_train_final, y_val_final = y, y
                # XGBoost 2.0+: early_stopping_rounds is set in constructor, not passed to fit()
                # The model already has it configured from the constructor above
                model.fit(
                    X_train_final, y_train_final,
                    eval_set=[(X_val_final, y_val_final)],
                    verbose=False
                )
                
                # Check for suspicious scores
                has_leak = primary_score >= _suspicious_score_threshold
                
                # Compute and store full task-aware metrics
                _compute_and_store_metrics('xgboost', model, X, y, primary_score, task_type)
                
                # LEAK DETECTION: Analyze feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    suspicious_features = _detect_leaking_features(
                        feature_names, importances, model_name='xgboost', 
                        threshold=0.50, force_report=has_leak
                    )
                    if suspicious_features:
                        all_suspicious_features['xgboost'] = suspicious_features
                    
                    # Store all feature importances for detailed export
                    # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                    importance_series = pd.Series(importances, index=feature_names[:len(importances)] if len(importances) <= len(feature_names) else feature_names)
                    # Reindex to match exact feature_names order (fills missing with 0.0)
                    importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                    importance_dict = importance_series.to_dict()
                    all_feature_importances['xgboost'] = importance_dict
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importances) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top 10%?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"XGBoost failed: {e}")
    
    # CatBoost
    if 'catboost' in model_families:
        try:
            import catboost as cb
            from catboost import Pool
            from TRAINING.utils.target_utils import is_classification_target, is_binary_classification_target
            
            # Determine task characteristics (use task_type, not y inspection for consistency)
            is_binary = task_type == TaskType.BINARY_CLASSIFICATION
            is_classification = task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]
            
            # GPU settings (will fallback to CPU if GPU not available)
            gpu_params = {}
            try:
                from CONFIG.config_loader import get_cfg
                # SST: All values from config, no hardcoded defaults
                # FIX: Rename to catboost_task_type to avoid overwriting task_type (TaskType enum)
                catboost_task_type = get_cfg('gpu.catboost.task_type', default='CPU', config_name='gpu_config')
                devices = get_cfg('gpu.catboost.devices', default='0', config_name='gpu_config')
                thread_count = get_cfg('gpu.catboost.thread_count', default=8, config_name='gpu_config')
                test_enabled = get_cfg('gpu.catboost.test_enabled', default=True, config_name='gpu_config')
                test_iterations = get_cfg('gpu.catboost.test_iterations', default=1, config_name='gpu_config')
                test_samples = get_cfg('gpu.catboost.test_samples', default=10, config_name='gpu_config')
                test_features = get_cfg('gpu.catboost.test_features', default=5, config_name='gpu_config')

                if catboost_task_type == 'GPU':
                    if test_enabled:
                        # Try GPU (CatBoost uses task_type='GPU' or devices parameter)
                        # Test if GPU is available
                        try:
                            test_model = cb.CatBoostRegressor(task_type='GPU', devices=devices, iterations=test_iterations, verbose=False)
                            # FIX: GPU mode requires Pool objects, not numpy arrays
                            test_X = np.random.rand(test_samples, test_features).astype('float32')
                            test_y = np.random.rand(test_samples).astype('float32')
                            test_pool = Pool(data=test_X, label=test_y)
                            test_model.fit(test_pool)
                            gpu_params = {'task_type': 'GPU', 'devices': devices}
                            logger.info(f"  ✅ Using GPU (CUDA) for CatBoost (devices={devices})")
                        except Exception as gpu_test_error:
                            logger.warning(f"  ⚠️  CatBoost GPU test failed, falling back to CPU: {gpu_test_error}")
                            gpu_params = {}  # Fallback to CPU
                    else:
                        # Skip test, use config values directly
                        gpu_params = {'task_type': 'GPU', 'devices': devices}
                        logger.info(f"  Using GPU (CUDA) for CatBoost (devices={devices}, test disabled)")
                else:
                    gpu_params = {}  # Use CPU (no GPU params)
                    logger.info("  Using CPU for CatBoost (task_type='CPU' in config)")
            except Exception as e:
                logger.warning(f"  ⚠️  CatBoost GPU config error, using CPU: {e}")
            
            # Get config values
            cb_config = get_model_config('catboost', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(cb_config, dict):
                cb_config = {}
            
            # Build params dict (copy to avoid mutating original)
            params = dict(cb_config)
            
            # Remove task_type, devices, and thread_count if present (we set these from GPU config)
            params.pop('task_type', None)
            params.pop('devices', None)
            params.pop('thread_count', None)  # Remove if present, we'll set from GPU config when using GPU
            
            # Add GPU params if available (will override any task_type/devices in config)
            params.update(gpu_params)
            
            # Add thread_count from GPU config when using GPU
            if gpu_params and gpu_params.get('task_type') == 'GPU' and 'thread_count' not in params:
                params['thread_count'] = thread_count
            
            # CatBoost Performance Diagnostics and Optimizations
            # Check for common issues that cause slow training (>20min for 50k samples)
            warnings_issued = []
            
            # 1. Check for excessive depth (exponential complexity: 2^d)
            depth = params.get('depth', 6)  # Default is 6
            if depth > 8:
                warnings_issued.append(f"⚠️  CatBoost depth={depth} is high (exponential complexity 2^{depth}). Consider depth ≤ 8 for faster training.")
            
            # 2. Check for text-like features (object/string dtype columns)
            # Convert X to DataFrame temporarily to check dtypes if feature_names available
            text_features_detected = []
            high_cardinality_features = []
            if feature_names and len(feature_names) == X.shape[1]:
                try:
                    # Create temporary DataFrame to check dtypes (pandas already imported at top)
                    X_df = pd.DataFrame(X, columns=feature_names)
                    
                    # Check for object/string dtype columns (potential text features)
                    object_cols = X_df.select_dtypes(include=['object', 'string']).columns.tolist()
                    if object_cols:
                        text_features_detected = object_cols
                        if 'text_features' not in params or not params.get('text_features'):
                            warnings_issued.append(
                                f"⚠️  CatBoost: Detected {len(object_cols)} text/object columns: {object_cols[:5]}{'...' if len(object_cols) > 5 else ''}. "
                                f"Add text_features=['col_name'] to params to avoid treating them as high-cardinality categoricals."
                            )
                    
                    # Check for high cardinality categoricals (potential ID columns)
                    # Only flag for DROP when multiple ID signals agree (categorical + high unique ratio + ID-like name)
                    # Numeric columns with high cardinality are normal (continuous features) - just warn, don't suggest dropping
                    cat_features_list = params.get('cat_features', [])
                    if isinstance(cat_features_list, (list, tuple)):
                        cat_features_set = set(cat_features_list)
                    else:
                        cat_features_set = set()
                    
                    for col in feature_names:
                        if col in X_df.columns:
                            try:
                                unique_count = X_df[col].nunique()
                                unique_ratio = unique_count / len(X_df) if len(X_df) > 0 else 0
                                
                                # Check if column is treated as categorical
                                is_categorical = (
                                    col in cat_features_set or
                                    X_df[col].dtype.name in ['object', 'category', 'string'] or
                                    str(X_df[col].dtype).startswith('category')
                                )
                                
                                # Check if it's numeric (float/int) - high cardinality is normal for continuous features
                                is_numeric = pd.api.types.is_numeric_dtype(X_df[col])
                                
                                # ID-like name patterns
                                id_patterns = ['_id', '_ID', 'id_', 'ID_', 'user_', 'User_', 'ip_', 'IP_', 'row_', 'Row_', 
                                              'uuid', 'UUID', 'tx_', 'order_', 'session_', 'hash_', '_key', '_Key']
                                has_id_name = any(pattern in col for pattern in id_patterns)
                                
                                # Check if values mostly occur once (median count per value <= 2)
                                value_counts = X_df[col].value_counts()
                                median_count = value_counts.median() if len(value_counts) > 0 else float('inf')
                                
                                # Only suggest DROP when multiple ID signals agree:
                                # 1. Treated as categorical (not numeric)
                                # 2. High unique ratio (>0.2 or >0.5 for strict)
                                # 3. Values mostly unique (median count <= 2) OR unique_ratio > 0.8
                                # 4. ID-like name OR near-perfect uniqueness
                                should_drop = (
                                    is_categorical and  # Must be categorical (not numeric)
                                    unique_ratio > 0.2 and  # High unique ratio
                                    (median_count <= 2 or unique_ratio > 0.8) and  # Mostly unique values
                                    (has_id_name or unique_ratio > 0.95)  # ID-like name OR near-perfect uniqueness
                                )
                                
                                if should_drop:
                                    high_cardinality_features.append((col, unique_count, unique_ratio, is_categorical))
                                elif is_numeric and unique_ratio > 0.8 and unique_count > 1000:
                                    # Numeric column with high cardinality - this is normal for continuous features
                                    # Just log a debug message, don't warn (this is expected behavior)
                                    logger.debug(f"  CatBoost: Column '{col}' is numeric with high cardinality ({unique_count} unique, {unique_ratio:.1%} unique ratio) - this is normal for continuous features")
                            except Exception:
                                pass  # Skip if can't compute unique count
                    
                    if high_cardinality_features:
                        id_cols = [col for col, _, _, _ in high_cardinality_features[:5]]
                        warnings_issued.append(
                            f"⚠️  CatBoost: Detected {len(high_cardinality_features)} high-cardinality ID-like CATEGORICAL columns: {id_cols}{'...' if len(high_cardinality_features) > 5 else ''}. "
                            f"These are treated as categorical with high unique ratios and ID-like names. Consider dropping or encoding differently (they don't generalize and slow training)."
                        )
                except Exception as e:
                    # If DataFrame conversion fails, skip diagnostics (non-critical)
                    logger.debug(f"  CatBoost diagnostics skipped (non-critical): {e}")
            
            # 3. Automatic metric_period injection for eval_set (reduces evaluation overhead)
            # Note: We use cross_val_score which doesn't use eval_set directly, but if params has eval_set,
            # we should add metric_period to reduce overhead
            # SST: Check config first, then use default
            if 'metric_period' not in params:
                # SST: Try to get from intelligent_training_config first, then model config, then default
                try:
                    from CONFIG.config_loader import get_cfg
                    # Check intelligent_training_config first (SST)
                    metric_period_from_config = get_cfg('training.catboost.metric_period', default=None, config_name='intelligent_training_config')
                    if metric_period_from_config is not None:
                        params['metric_period'] = metric_period_from_config
                    else:
                        # Fallback to model config (from multi_model.yaml) or default
                        params['metric_period'] = cb_config.get('metric_period', 50)  # Default: 50 if not in any config
                except Exception:
                    # If config loader fails, use model config or default
                    params['metric_period'] = cb_config.get('metric_period', 50)
                if log_cfg.edu_hints:
                    logger.debug(f"  CatBoost: Added metric_period={params['metric_period']} to reduce evaluation overhead (SST: from config or default)")
            
            # Log warnings if any
            if warnings_issued:
                logger.warning(f"  CatBoost Performance Warnings:")
                for warning in warnings_issued:
                    logger.warning(f"    {warning}")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 See KNOWN_ISSUES.md for CatBoost slow training troubleshooting")
            
            # CRITICAL: Enforce consistent loss/metric pairs for CatBoost
            # Binary classification: Logloss + AUC (not RMSE + roc_auc)
            # Regression: RMSE + RMSE
            # This prevents NaN from loss/metric mismatch
            
            # Auto-detect target type and set loss_function if not specified
            if "loss_function" not in params:
                if is_classification_target(y):
                    if is_binary_classification_target(y):
                        params["loss_function"] = "Logloss"  # Binary classification loss
                    else:
                        params["loss_function"] = "MultiClass"
                else:
                    params["loss_function"] = "RMSE"  # Regression loss
            
            # CRITICAL: Override config if it has inconsistent loss/metric pair
            # If task is binary classification but loss is RMSE, fix it
            if is_binary and params.get("loss_function") == "RMSE":
                logger.warning(
                    f"  ⚠️  CatBoost: Config has RMSE loss for binary classification. "
                    f"Overriding to Logloss to prevent NaN."
                )
                params["loss_function"] = "Logloss"
            
            # Set eval_metric to match loss_function (CatBoost uses 'AUC' not 'roc_auc')
            if "eval_metric" not in params:
                if is_binary:
                    params["eval_metric"] = "AUC"  # CatBoost's internal metric name
                elif is_classification_target(y):
                    params["eval_metric"] = "Accuracy"
                else:
                    params["eval_metric"] = "RMSE"
            
            # Ensure eval_metric is consistent with loss_function
            if is_binary and params.get("eval_metric") not in ["AUC", "Logloss"]:
                logger.warning(
                    f"  ⚠️  CatBoost: Config has eval_metric={params.get('eval_metric')} for binary classification. "
                    f"Overriding to AUC to prevent NaN."
                )
                params["eval_metric"] = "AUC"
            
            # If loss_function is specified in config, respect it (YAML in charge)
            # But we've already validated consistency above
            
            # CRITICAL: Verify GPU params are in params dict before instantiation
            # CatBoost REQUIRES task_type='GPU' to actually use GPU (devices alone is ignored)
            if gpu_params and 'task_type' in gpu_params:
                # Ensure GPU params are definitely in params (defensive check)
                params.update(gpu_params)
                # Explicit verification that task_type is set
                if params.get('task_type') != 'GPU':
                    logger.warning(f"  ⚠️  CatBoost GPU params updated but task_type is '{params.get('task_type')}', expected 'GPU'")
                else:
                    logger.debug(f"  ✅ CatBoost GPU verified: task_type={params.get('task_type')}, devices={params.get('devices')}")
            elif gpu_params:
                # GPU was requested but task_type missing - this is a bug
                logger.error(f"  ❌ CatBoost GPU requested but task_type missing from gpu_params: {gpu_params}")
            
            # Log final params for debugging (only if GPU was requested)
            if gpu_params and gpu_params.get('task_type') == 'GPU' and log_cfg.gpu_detail:
                logger.debug(f"  CatBoost final params (sample): task_type={params.get('task_type')}, devices={params.get('devices')}, iterations={params.get('iterations', 'default')}")
            
            # Set verbose level from backend config (similar to LightGBM)
            # CatBoost verbose: 0=silent, 1=info, 2=debug, >2=more verbose
            if 'verbose' not in params:
                params['verbose'] = catboost_backend_cfg.native_verbosity
            
            # CRITICAL: Choose model class based on task_type (not y inspection)
            # This ensures consistency: BINARY_CLASSIFICATION → CatBoostClassifier, REGRESSION → CatBoostRegressor
            # Using is_classification_target(y) can be inconsistent with task_type
            if is_binary:
                # Binary classification: must use CatBoostClassifier
                if params.get("loss_function") == "RMSE":
                    # This should have been fixed above, but double-check
                    logger.error(
                        f"  ❌ CatBoost: Binary classification but loss_function=RMSE. "
                        f"This should have been overridden. Fixing now."
                    )
                    params["loss_function"] = "Logloss"
                base_model = cb.CatBoostClassifier(**params)
                # Hard-stop: verify we got the right class
                if not isinstance(base_model, cb.CatBoostClassifier):
                    raise ValueError(
                        f"BINARY_CLASSIFICATION requires CatBoostClassifier, but got {type(base_model)}. "
                        f"This is a programming error."
                    )
            elif is_classification:
                # Multiclass classification: must use CatBoostClassifier
                base_model = cb.CatBoostClassifier(**params)
            else:
                # Regression: must use CatBoostRegressor
                base_model = cb.CatBoostRegressor(**params)
                # Hard-stop: verify we got the right class
                if not isinstance(base_model, cb.CatBoostRegressor):
                    raise ValueError(
                        f"REGRESSION requires CatBoostRegressor, but got {type(base_model)}. "
                        f"This is a programming error."
                    )
            
            # FIX: When GPU mode is enabled, CatBoost requires Pool objects instead of numpy arrays
            # Create a wrapper class that converts numpy arrays to Pool objects in fit() method
            use_gpu = 'task_type' in params and params.get('task_type') == 'GPU'
            
            if use_gpu:
                # Create a wrapper class that handles Pool conversion for GPU mode
                # FIX: Make sklearn-compatible by implementing get_params/set_params
                class CatBoostGPUWrapper:
                    """Wrapper for CatBoost models that converts numpy arrays to Pool objects when GPU is enabled."""
                    def __init__(self, base_model=None, cat_features=None, use_gpu=True, _model_class=None, **kwargs):
                        # If base_model is provided, use it; otherwise create from kwargs (for sklearn cloning)
                        if base_model is not None:
                            self.base_model = base_model
                            # Store the model class for sklearn cloning
                            self._model_class = type(base_model)
                        else:
                            # Recreate base model from kwargs (for sklearn clone)
                            # Determine model class from loss_function or use stored class
                            if _model_class is not None:
                                model_class = _model_class
                            else:
                                # Infer from loss_function in kwargs
                                loss_fn = kwargs.get('loss_function', 'RMSE')
                                if loss_fn in ['Logloss', 'MultiClass']:
                                    model_class = cb.CatBoostClassifier
                                else:
                                    model_class = cb.CatBoostRegressor
                            
                            # CRITICAL: Verify model class matches loss_function
                            # If loss_function is Logloss but model_class is Regressor, fix it
                            loss_fn = kwargs.get('loss_function', 'RMSE')
                            if loss_fn in ['Logloss', 'MultiClass'] and model_class == cb.CatBoostRegressor:
                                logger.warning(
                                    f"  ⚠️  CatBoost GPU wrapper: loss_function={loss_fn} but model_class=Regressor. "
                                    f"Fixing to Classifier."
                                )
                                model_class = cb.CatBoostClassifier
                            elif loss_fn == 'RMSE' and model_class == cb.CatBoostClassifier:
                                logger.warning(
                                    f"  ⚠️  CatBoost GPU wrapper: loss_function={loss_fn} but model_class=Classifier. "
                                    f"Fixing to Regressor."
                                )
                                model_class = cb.CatBoostRegressor
                            
                            self.base_model = model_class(**kwargs)
                            self._model_class = model_class
                        # FIX: For sklearn clone validation, ensure cat_features is set exactly as passed
                        # If None, use empty list; if already a list, use it directly; otherwise convert
                        if cat_features is None:
                            self.cat_features = []
                        elif isinstance(cat_features, list):
                            # Already a list - use it directly (sklearn expects this for clone validation)
                            self.cat_features = cat_features
                        else:
                            # Convert to list if it's not already
                            self.cat_features = list(cat_features)
                        self.use_gpu = use_gpu
                    
                    def get_params(self, deep=True):
                        """Get parameters for sklearn compatibility."""
                        # Get base model params and add wrapper-specific params
                        params = self.base_model.get_params(deep=deep)
                        # FIX: Return cat_features as-is (it's already a list from __init__)
                        # Sklearn's clone validation requires exact round-trip: get_params() -> __init__(**params) -> get_params()
                        params['cat_features'] = self.cat_features
                        params['use_gpu'] = self.use_gpu
                        params['_model_class'] = self._model_class
                        # Remove base_model from params (it's not a constructor arg)
                        params.pop('base_model', None)
                        return params
                    
                    def set_params(self, **params):
                        """Set parameters for sklearn compatibility."""
                        # Extract wrapper-specific params
                        cat_features = params.pop('cat_features', None)
                        use_gpu = params.pop('use_gpu', None)
                        model_class = params.pop('_model_class', None)
                        if cat_features is not None:
                            # FIX: Set exactly as passed (sklearn clone validation requires this)
                            if isinstance(cat_features, list):
                                self.cat_features = cat_features
                            else:
                                self.cat_features = list(cat_features) if cat_features else []
                        if use_gpu is not None:
                            self.use_gpu = use_gpu
                        if model_class is not None:
                            self._model_class = model_class
                        # Update base model params
                        self.base_model.set_params(**params)
                        return self
                    
                    def fit(self, X, y=None, **kwargs):
                        """Convert numpy arrays to Pool objects when GPU is enabled."""
                        # Convert X and y to Pool objects for GPU mode
                        if isinstance(X, np.ndarray):
                            train_pool = Pool(data=X, label=y, cat_features=self.cat_features)
                            return self.base_model.fit(train_pool, **kwargs)
                        elif isinstance(X, Pool):
                            # Already a Pool object
                            return self.base_model.fit(X, y, **kwargs)
                        else:
                            # Fallback: try direct fit (for other data types)
                            return self.base_model.fit(X, y, **kwargs)
                    
                    def predict(self, X, **kwargs):
                        """Delegate predict to base model."""
                        if isinstance(X, np.ndarray) and self.use_gpu:
                            # Convert to Pool for consistency, though predict may work with arrays
                            test_pool = Pool(data=X, cat_features=self.cat_features)
                            return self.base_model.predict(test_pool, **kwargs)
                        return self.base_model.predict(X, **kwargs)
                    
                    def score(self, X, y, **kwargs):
                        """Delegate score to base model."""
                        if isinstance(X, np.ndarray) and self.use_gpu:
                            test_pool = Pool(data=X, label=y, cat_features=self.cat_features)
                            return self.base_model.score(test_pool, **kwargs)
                        return self.base_model.score(X, y, **kwargs)
                    
                    def __getattr__(self, name):
                        """Delegate all other attributes to base model."""
                        return getattr(self.base_model, name)
                
                # Get categorical features from params if specified
                cat_features = params.get('cat_features', [])
                if isinstance(cat_features, list) and len(cat_features) > 0:
                    # If cat_features are column names, convert to indices
                    if feature_names and isinstance(cat_features[0], str):
                        cat_feature_indices = [feature_names.index(f) for f in cat_features if f in feature_names]
                    else:
                        cat_feature_indices = cat_features
                else:
                    cat_feature_indices = []
                
                model = CatBoostGPUWrapper(base_model=base_model, cat_features=cat_feature_indices, use_gpu=use_gpu)
            else:
                # CPU mode: use model directly (no Pool conversion needed)
                model = base_model

            # Log GPU usage if available (always log, not just when gpu_detail enabled)
            if 'task_type' in params and params.get('task_type') == 'GPU':
                logger.info(f"  🚀 Training CatBoost on GPU (devices={params.get('devices', '0')})")
                logger.info(f"  📊 Dataset size: {len(X)} samples, {X.shape[1]} features")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: CatBoost does quantization on CPU first, then trains on GPU")
                    logger.info(f"  💡 Watch GPU memory allocation (not just utilization %) to verify GPU usage")
            elif gpu_params and gpu_params.get('task_type') == 'GPU':
                # Fallback: log if GPU was requested but not in final params
                logger.warning(f"  ⚠️  CatBoost GPU requested but task_type not in final params (check config cleaning)")
                logger.warning(f"  ⚠️  Final params task_type: {params.get('task_type', 'MISSING')}")
                if log_cfg.edu_hints:
                    logger.info(f"  💡 Note: GPU is most efficient for large datasets (>100k samples)")
            
            # Check for outer parallelism that might cause CPU bottleneck
            # If CV is parallelized (cv_n_jobs > 1), this can cause CPU to peg even with thread_count limited
            if cv_n_jobs and cv_n_jobs > 1 and gpu_params and gpu_params.get('task_type') == 'GPU':
                logger.warning(
                    f"  ⚠️  CatBoost GPU training with parallel CV (n_jobs={cv_n_jobs}). "
                    f"Outer parallelism can cause CPU bottleneck even with thread_count limited. "
                    f"Consider setting cv_n_jobs=1 for GPU training."
                )
            
            # PHASE 1: Handle skipped CV (degenerate folds policy)
            if tscv is None:
                # CV was skipped due to degenerate target - skip CV and fit on full dataset
                primary_score = np.nan
                logger.info(f"  ℹ️  CatBoost: Skipping CV (degenerate target detected pre-CV). Fitting on full dataset for importance only.")
            else:
                # CRITICAL: Fold health check before CV to diagnose NaN issues
                # Log fold health for each fold and hard-fail on invalid folds
                logger.info(f"  🔍 CatBoost CV fold health check:")
                logger.info(f"     Objective: {params.get('loss_function', 'auto')}, Metric: {scoring}, Task: {task_type.name}")
                
                # Extract purge/embargo from resolved_config if available
                purge_minutes_val = None
                embargo_minutes_val = None
                if resolved_config:
                    purge_minutes_val = getattr(resolved_config, 'purge_minutes', None)
                    embargo_minutes_val = getattr(resolved_config, 'embargo_minutes', None)
                
                # Also try to get from purge_time if available (for logging)
                # purge_time is defined earlier in the function as pd.Timedelta
                try:
                    if purge_time is not None:
                        # purge_time is a Timedelta, convert to minutes
                        if hasattr(purge_time, 'total_seconds'):
                            purge_minutes_val = purge_time.total_seconds() / 60.0
                        elif isinstance(purge_time, (int, float)):
                            purge_minutes_val = purge_time
                except NameError:
                    # purge_time not in scope, use resolved_config values only
                    pass
                
                if purge_minutes_val and embargo_minutes_val:
                    logger.info(f"     Purge: {purge_minutes_val:.1f}m, Embargo: {embargo_minutes_val:.1f}m")
                elif purge_minutes_val:
                    logger.info(f"     Purge: {purge_minutes_val:.1f}m, Embargo: unknown")
                else:
                    logger.info(f"     Purge/Embargo: from resolved_config or defaults")
                
                # Check each fold before CV
                fold_violations = []
                all_folds_list = list(tscv.split(X, y))
                
                for fold_idx, (train_idx, val_idx) in enumerate(all_folds_list):
                    train_n = len(train_idx)
                    val_n = len(val_idx)
                    
                    # Basic checks
                    if val_n == 0:
                        fold_violations.append(f"Fold {fold_idx + 1}: val_n=0 (empty validation set)")
                        continue
                    
                    # Binary classification: check class balance in BOTH train and validation sets
                    if is_binary:
                        # Check validation set
                        val_y = y[val_idx]
                        val_y_clean = val_y[~np.isnan(val_y)]
                        val_unique = np.unique(val_y_clean) if len(val_y_clean) > 0 else np.array([])
                        val_pos_count = np.sum(val_y_clean == 1) if len(val_y_clean) > 0 else 0
                        val_neg_count = np.sum(val_y_clean == 0) if len(val_y_clean) > 0 else 0
                        
                        # Check training set (CRITICAL: single-class training causes NaN)
                        train_y = y[train_idx]
                        train_y_clean = train_y[~np.isnan(train_y)]
                        train_unique = np.unique(train_y_clean) if len(train_y_clean) > 0 else np.array([])
                        train_pos_count = np.sum(train_y_clean == 1) if len(train_y_clean) > 0 else 0
                        train_neg_count = np.sum(train_y_clean == 0) if len(train_y_clean) > 0 else 0
                        
                        # Check for violations
                        val_degenerate = val_pos_count == 0 or val_neg_count == 0
                        train_degenerate = train_pos_count == 0 or train_neg_count == 0
                        
                        if train_degenerate:
                            fold_violations.append(
                                f"Fold {fold_idx + 1}: Binary classification with degenerate TRAINING set "
                                f"(train_pos={train_pos_count}, train_neg={train_neg_count}, train_unique={train_unique.tolist()})"
                            )
                            logger.warning(
                                f"     ⚠️  Fold {fold_idx + 1}: train_n={train_n}, train_pos={train_pos_count}, train_neg={train_neg_count}, "
                                f"train_unique={train_unique.tolist()}, val_n={val_n}, val_pos={val_pos_count}, val_neg={val_neg_count}"
                            )
                        elif val_degenerate:
                            fold_violations.append(
                                f"Fold {fold_idx + 1}: Binary classification with degenerate validation set "
                                f"(val_pos={val_pos_count}, val_neg={val_neg_count}, val_unique={val_unique.tolist()})"
                            )
                            logger.warning(
                                f"     ⚠️  Fold {fold_idx + 1}: train_n={train_n}, train_pos={train_pos_count}, train_neg={train_neg_count}, "
                                f"val_n={val_n}, val_pos={val_pos_count}, val_neg={val_neg_count}, val_unique={val_unique.tolist()}"
                            )
                        else:
                            logger.info(
                                f"     ✅ Fold {fold_idx + 1}: train_n={train_n}, train_pos={train_pos_count}, train_neg={train_neg_count}, "
                                f"val_n={val_n}, val_pos={val_pos_count}, val_neg={val_neg_count}"
                            )
                    else:
                        # Regression or multiclass: just log sizes
                        logger.info(f"     ✅ Fold {fold_idx + 1}: train_n={train_n}, val_n={val_n}")
                    
                    # Ranking/group structure check (if groups are used)
                    # Note: For cross-sectional ranking, groups are typically timestamps
                    # If CatBoost is using ranking objective, we'd need group IDs here
                    # For now, just log if we detect ranking mode
                    if 'objective' in params and 'ranking' in str(params.get('objective', '')).lower():
                        # Ranking mode: would need group IDs to check group sizes
                        logger.warning(f"     ⚠️  Fold {fold_idx + 1}: Ranking mode detected but group structure not validated")
                
                # Log violations but don't hard-fail (let CV proceed and return NaN if needed)
                # This allows us to diagnose the issue while maintaining current behavior
                if fold_violations:
                    error_msg = (
                        f"🚨 CatBoost CV fold health check FAILED. Invalid folds detected:\n"
                        f"   " + "\n   ".join(fold_violations) + "\n"
                        f"   This will likely cause NaN scores. Fix by:\n"
                    )
                    if purge_minutes_val and embargo_minutes_val:
                        error_msg += f"   1) Reducing purge/embargo ({purge_minutes_val:.1f}m/{embargo_minutes_val:.1f}m) if too large\n"
                    else:
                        error_msg += f"   1) Reducing purge/embargo if too large\n"
                    error_msg += (
                        f"   2) Loading more data to ensure sufficient validation set size\n"
                        f"   3) For binary classification: ensure validation sets have both classes"
                    )
                    logger.error(error_msg)
                    logger.warning(f"     ⚠️  Proceeding with CV anyway (will likely return NaN)")
                else:
                    logger.info(f"     ✅ All {len(all_folds_list)} folds passed health check")
                
                try:
                    scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                    valid_scores = scores[~np.isnan(scores)]
                    primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
                except (ValueError, TypeError) as e:
                    error_str = str(e)
                    if "Invalid classes" in error_str or "Expected" in error_str:
                        logger.debug(f"    CatBoost: Target degenerate in some CV folds")
                        primary_score = np.nan
                        model_metrics['catboost'] = {'roc_auc': np.nan} if task_type == TaskType.BINARY_CLASSIFICATION else {'r2': np.nan} if task_type == TaskType.REGRESSION else {'accuracy': np.nan}
                        model_scores['catboost'] = np.nan
                    elif "Invalid data type" in error_str and "catboost.Pool" in error_str:
                        # FIX: If Pool conversion failed, log and re-raise with context
                        logger.error(f"  ❌ CatBoost GPU Pool conversion error: {e}")
                        logger.error(f"  💡 This may indicate a CatBoost version compatibility issue with GPU mode")
                        raise
                    else:
                        raise
            
            # Fit model and compute importance even if CV failed (NaN score)
            # Classification targets often fail CV due to degenerate folds, but we can still compute importance
            # from a model fit on the full dataset
            model_fitted = False
            if not np.isnan(primary_score):
                model.fit(X, y)
                model_fitted = True
                
                # Verify GPU is actually being used (post-fit check)
                if gpu_params and gpu_params.get('task_type') == 'GPU':
                    try:
                        actual_params = model.get_all_params()
                        actual_task_type = actual_params.get('task_type', 'UNKNOWN')
                        if actual_task_type != 'GPU':
                            logger.warning(
                                f"  ⚠️  CatBoost GPU requested but model reports task_type='{actual_task_type}'. "
                                f"GPU may not be active. Check model.get_all_params() for actual configuration."
                            )
                        elif log_cfg.gpu_detail:
                            logger.debug(f"  ✅ CatBoost GPU verified post-fit: task_type={actual_task_type}, devices={actual_params.get('devices', 'UNKNOWN')}")
                    except Exception as e:
                        logger.debug(f"  CatBoost post-fit GPU verification skipped (non-critical): {e}")

                # Compute and store full task-aware metrics
                _compute_and_store_metrics('catboost', model, X, y, primary_score, task_type)
            else:
                # CV failed (NaN score) - still try to fit and compute importance
                # This is especially important for classification targets that may fail CV due to degenerate folds
                # but can still provide useful feature importance from full-dataset fit
                logger.info(f"  ℹ️  CatBoost CV returned NaN (likely degenerate folds), but fitting on full dataset to compute importance")
                try:
                    model.fit(X, y)
                    model_fitted = True
                except Exception as e:
                    logger.warning(f"  ⚠️  CatBoost failed to fit on full dataset: {e}")
                    model_fitted = False
            
            # CatBoost requires training dataset to compute feature importance
            # FIX: For GPU wrapper, need to access base_model and handle Pool conversion
            # CRITICAL: Always compute and store importance if model trained successfully (even if CV failed)
            importance = None
            if model_fitted:
                try:
                    if hasattr(model, 'base_model'):
                        # Wrapper model - use base model
                        # For GPU mode, convert X to Pool if needed
                        if use_gpu and isinstance(X, np.ndarray):
                            importance_data = Pool(data=X, cat_features=model.cat_features)
                        else:
                            importance_data = X
                        importance = model.base_model.get_feature_importance(data=importance_data, type='PredictionValuesChange')
                    else:
                        # Direct model (CPU mode)
                        importance = model.get_feature_importance(data=X, type='PredictionValuesChange')
                except Exception as e:
                    logger.warning(f"  ⚠️  CatBoost feature importance computation failed: {e}")
                    logger.debug(f"  CatBoost importance error details:", exc_info=True)
                    # Try fallback: use numpy array directly (might work even in GPU mode)
                    try:
                        if hasattr(model, 'base_model'):
                            importance = model.base_model.get_feature_importance(data=X, type='PredictionValuesChange')
                        else:
                            importance = model.get_feature_importance(data=X, type='PredictionValuesChange')
                        logger.info(f"  ✅ CatBoost importance computed using fallback method")
                    except Exception as e2:
                        logger.warning(f"  ⚠️  CatBoost importance fallback also failed: {e2}")
                        importance = None
                
                # Store all feature importances for detailed export (same pattern as other models)
                # CRITICAL: Align importance to feature_names order to ensure fingerprint match
                if importance is not None and len(importance) > 0:
                    importance_series = pd.Series(importance, index=feature_names[:len(importance)] if len(importance) <= len(feature_names) else feature_names)
                    # Reindex to match exact feature_names order (fills missing with 0.0)
                    importance_series = importance_series.reindex(feature_names, fill_value=0.0)
                    importance_dict = importance_series.to_dict()
                    all_feature_importances['catboost'] = importance_dict
                    logger.debug(f"  ✅ CatBoost feature importance stored: {len(importance_dict)} features")
                else:
                    if importance is None:
                        logger.warning(f"  ⚠️  CatBoost feature importance is None (computation failed)")
                    else:
                        logger.warning(f"  ⚠️  CatBoost feature importance is empty (len={len(importance)})")
                    # Store empty dict to ensure CatBoost appears in output (even if empty)
                    # This ensures consistency - all models that train should have entries
                    all_feature_importances['catboost'] = {}
            else:
                # Model didn't fit - can't compute importance
                importance = np.array([])
                all_feature_importances['catboost'] = {}
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError:
            logger.warning("CatBoost not available (pip install catboost)")
        except Exception as e:
            logger.warning(f"CatBoost failed: {e}")
    
    # Lasso
    if 'lasso' in model_families:
        try:
            from sklearn.linear_model import Lasso
            from sklearn.pipeline import Pipeline
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Get config values
            lasso_config = get_model_config('lasso', multi_model_config)
            
            # Use sklearn-safe conversion (handles NaNs, dtypes, infs)
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # CRITICAL FIX: Pipeline ensures scaling happens within each CV fold (no leakage)
            # Lasso requires scaling for proper convergence (features must be on similar scales)
            # Note: X_dense is already imputed by make_sklearn_dense_X, so we only need scaler
            steps = [
                ('scaler', StandardScaler()),  # Required for Lasso convergence
                ('model', Lasso(**lasso_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # ⚠️ IMPORTANCE BIAS WARNING: This fits on the full dataset (in-sample)
            # See comment above for details
            pipeline.fit(X_dense, y)
            
            # Compute and store full task-aware metrics (Lasso is regression-only)
            if not np.isnan(primary_score) and task_type == TaskType.REGRESSION:
                _compute_and_store_metrics('lasso', pipeline, X_dense, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            importance = np.abs(model.coef_)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Lasso failed: {e}")
    
    # Ridge
    if 'ridge' in model_families:
        try:
            from sklearn.linear_model import Ridge, RidgeClassifier
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Ridge doesn't handle NaNs - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            ridge_config = get_model_config('ridge', multi_model_config)
            
            # CRITICAL: Use correct estimator based on task type
            # For classification: RidgeClassifier (not Ridge regression)
            # For regression: Ridge
            if is_binary or is_multiclass:
                est_cls = RidgeClassifier
            else:
                est_cls = Ridge
            
            # CRITICAL: Ridge requires scaling for proper convergence
            # Pipeline ensures scaling happens within each CV fold (no leakage)
            steps = [
                ('scaler', StandardScaler()),  # Required for Ridge convergence
                ('model', est_cls(**ridge_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Fit on full data for importance extraction (CV is done elsewhere)
            pipeline.fit(X_dense, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('ridge', pipeline, X_dense, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            # FIX: Handle both 1D (binary) and 2D (multiclass) coef_ shapes
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multiclass: use max absolute coefficient across classes
                importance = np.abs(coef).max(axis=0)
            else:
                # Binary or regression: use absolute coefficients
                importance = np.abs(coef)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Validate importance is not all zeros
            if np.all(importance == 0) or np.sum(importance) == 0:
                logger.warning(f"Ridge: All coefficients are zero (over-regularized or no signal)")
                importance_ratio = 0.0
            else:
                if len(importance) > 0:
                    total_importance = np.sum(importance)
                    if total_importance > 0:
                        top_fraction = _get_importance_top_fraction()
                        top_k = max(1, int(len(importance) * top_fraction))
                        top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                        importance_ratio = top_importance_sum / total_importance
                    else:
                        importance_ratio = 0.0
                else:
                    importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Ridge failed: {e}")
    
    # Elastic Net
    if 'elastic_net' in model_families:
        try:
            from sklearn.linear_model import ElasticNet, LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Elastic Net doesn't handle NaNs - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            elastic_net_config = get_model_config('elastic_net', multi_model_config)
            
            # CRITICAL: Use correct estimator based on task type
            # For classification: LogisticRegression with penalty='elasticnet' and solver='saga'
            # For regression: ElasticNet
            if is_binary or is_multiclass:
                # LogisticRegression with elasticnet penalty
                est_cls = LogisticRegression
                # ElasticNet requires solver='saga' for penalty='elasticnet'
                elastic_net_config = elastic_net_config.copy()
                elastic_net_config['penalty'] = 'elasticnet'
                elastic_net_config['solver'] = 'saga'  # Required for elasticnet penalty
                # l1_ratio maps to ElasticNet's l1_ratio (0 = pure L2, 1 = pure L1)
                if 'l1_ratio' not in elastic_net_config:
                    elastic_net_config['l1_ratio'] = elastic_net_config.get('l1_ratio', 0.5)
                # alpha maps to C (inverse regularization strength)
                if 'alpha' in elastic_net_config:
                    # Convert alpha to C (C = 1/alpha for consistency with sklearn)
                    alpha = elastic_net_config.pop('alpha')
                    elastic_net_config['C'] = 1.0 / alpha if alpha > 0 else 1.0
                elif 'C' not in elastic_net_config:
                    elastic_net_config['C'] = 1.0  # Default C=1.0
            else:
                # ElasticNet regression
                est_cls = ElasticNet
            
            # CRITICAL: Elastic Net requires scaling for proper convergence
            # Pipeline ensures scaling happens within each CV fold (no leakage)
            steps = [
                ('scaler', StandardScaler()),  # Required for ElasticNet convergence
                ('model', est_cls(**elastic_net_config))
            ]
            pipeline = Pipeline(steps)
            
            scores = cross_val_score(pipeline, X_dense, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Fit on full data for importance extraction (CV is done elsewhere)
            pipeline.fit(X_dense, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('elastic_net', pipeline, X_dense, y, primary_score, task_type)
            
            # Extract coefficients from the fitted model
            model = pipeline.named_steps['model']
            # FIX: Handle both 1D (binary) and 2D (multiclass) coef_ shapes
            coef = model.coef_
            if len(coef.shape) > 1:
                # Multiclass: use max absolute coefficient across classes
                importance = np.abs(coef).max(axis=0)
            else:
                # Binary or regression: use absolute coefficients
                importance = np.abs(coef)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Validate importance is not all zeros
            if np.all(importance == 0) or np.sum(importance) == 0:
                logger.warning(f"Elastic Net: All coefficients are zero (over-regularized or no signal)")
                importance_ratio = 0.0
            else:
                if len(importance) > 0:
                    total_importance = np.sum(importance)
                    if total_importance > 0:
                        top_fraction = _get_importance_top_fraction()
                        top_k = max(1, int(len(importance) * top_fraction))
                        top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                        importance_ratio = top_importance_sum / total_importance
                    else:
                        importance_ratio = 0.0
                else:
                    importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Elastic Net failed: {e}")
    
    # Mutual Information
    if 'mutual_information' in model_families:
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Mutual information doesn't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            mi_config = get_model_config('mutual_information', multi_model_config)
            
            # Get random_state from SST (determinism system) - no hardcoded defaults
            mi_random_state = mi_config.get('random_state')
            if mi_random_state is None:
                from TRAINING.common.determinism import stable_seed_from
                mi_random_state = stable_seed_from(['mutual_information', target_column if target_column else 'default'])
            
            # Suppress warnings for zero-variance features
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    importance = mutual_info_classif(X_dense, y, 
                                                    random_state=mi_random_state,
                                                    discrete_features=mi_config.get('discrete_features', 'auto'))
                else:
                    importance = mutual_info_regression(X_dense, y, 
                                                       random_state=mi_random_state,
                                                       discrete_features=mi_config.get('discrete_features', 'auto'))
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Handle NaN/inf
            importance = np.nan_to_num(importance, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Mutual information doesn't have R², so we use a proxy based on max MI
            # Normalize to 0-1 scale for importance
            if len(importance) > 0 and np.max(importance) > 0:
                importance_normalized = importance / np.max(importance)
                total_importance = np.sum(importance_normalized)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance_normalized) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance_normalized)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # For mutual information, we can't compute R² directly
            # Use a proxy: higher MI concentration = better predictability
            # Scale to approximate R² range (0-0.3 for good targets)
            model_scores['mutual_information'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Mutual Information failed: {e}")
    
    # Univariate Selection
    if 'univariate_selection' in model_families:
        try:
            from sklearn.feature_selection import f_regression, f_classif
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # F-tests don't handle NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Suppress division by zero warnings (expected for zero-variance features)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if is_binary or is_multiclass:
                    scores, pvalues = f_classif(X_dense, y)
                else:
                    scores, pvalues = f_regression(X_dense, y)
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Handle NaN/inf in scores (from zero-variance features)
            scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize F-statistics
            if len(scores) > 0 and np.max(scores) > 0:
                importance = scores / np.max(scores)
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            
            # F-statistics don't have R², use proxy
            model_scores['univariate_selection'] = min(0.3, importance_ratio * 0.3)
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Univariate Selection failed: {e}")
    
    # RFE
    if 'rfe' in model_families:
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            
            # RFE uses RandomForest which handles NaN, but let's impute for consistency
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Get config values with defaults
            rfe_config = get_model_config('rfe', multi_model_config)
            # FIX: Use .get() with default to prevent KeyError
            # Default to 20% of features or top_k if available, but at least 1
            default_n_features = max(1, int(0.2 * X_imputed.shape[1]))
            n_features_to_select = min(rfe_config.get('n_features_to_select', default_n_features), X_imputed.shape[1])
            step = rfe_config.get('step', 5)
            
            # Use random_forest config for RFE estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            if is_binary or is_multiclass:
                estimator = RandomForestClassifier(**rf_config)
            else:
                estimator = RandomForestRegressor(**rf_config)
            
            selector = RFE(estimator, n_features_to_select=n_features_to_select, step=step)
            selector.fit(X_imputed, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = selector.support_
            if np.any(selected_features):
                X_selected = X_imputed[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['rfe'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['rfe'] = np.nan
            
            # Convert ranking to importance
            ranking = selector.ranking_
            importance = 1.0 / (ranking + 1e-6)
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"RFE failed: {e}")
    
    # Boruta
    if 'boruta' in model_families:
        try:
            from boruta import BorutaPy
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Boruta doesn't support NaN - use sklearn-safe conversion
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            boruta_config = get_model_config('boruta', multi_model_config)
            
            # Use random_forest config for Boruta estimator
            rf_config = get_model_config('random_forest', multi_model_config)
            
            # Get random_state from SST (determinism system) - no hardcoded defaults
            boruta_random_state = boruta_config.get('random_state')
            if boruta_random_state is None:
                from TRAINING.common.determinism import stable_seed_from
                boruta_random_state = stable_seed_from(['boruta', target_column if target_column else 'default'])
            
            # Remove random_state from rf_config to prevent double argument error
            rf_config_clean = rf_config.copy()
            rf_config_clean.pop('random_state', None)
            
            if is_binary or is_multiclass:
                rf = RandomForestClassifier(**rf_config_clean, random_state=boruta_random_state)
            else:
                rf = RandomForestRegressor(**rf_config_clean, random_state=boruta_random_state)
            
            boruta = BorutaPy(rf, n_estimators='auto', verbose=0, 
                            random_state=boruta_random_state,
                            max_iter=boruta_config.get('max_iter', 100))
            boruta.fit(X_dense, y)
            
            # Get R² using cross-validation on selected features (proper validation)
            selected_features = boruta.support_
            if np.any(selected_features):
                X_selected = X_dense[:, selected_features]
                # Quick RF for scoring (use smaller config)
                quick_rf_config = get_model_config('random_forest', multi_model_config).copy()
                # Use smaller model for quick scoring
                quick_rf_config['n_estimators'] = 50
                quick_rf_config['max_depth'] = 8
                
                if is_binary or is_multiclass:
                    quick_rf = RandomForestClassifier(**quick_rf_config)
                else:
                    quick_rf = RandomForestRegressor(**quick_rf_config)
                
                # Use cross-validation for proper validation (not training score)
                scores = cross_val_score(quick_rf, X_selected, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
                valid_scores = scores[~np.isnan(scores)]
                model_scores['boruta'] = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            else:
                model_scores['boruta'] = np.nan
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Convert to importance
            ranking = boruta.ranking_
            selected = boruta.support_
            importance = np.where(selected, 1.0, np.where(ranking == 2, 0.5, 0.1))
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except ImportError:
            logger.warning("Boruta not available (pip install Boruta)")
        except Exception as e:
            logger.warning(f"Boruta failed: {e}")
    
    # Stability Selection
    if 'stability_selection' in model_families:
        try:
            from sklearn.linear_model import LassoCV, LogisticRegressionCV
            from TRAINING.utils.sklearn_safe import make_sklearn_dense_X
            
            # Stability selection uses Lasso/LogisticRegression which don't handle NaN
            X_dense, feature_names_dense = make_sklearn_dense_X(X, feature_names)
            
            # Get config values
            stability_config = get_model_config('stability_selection', multi_model_config)
            n_bootstrap = stability_config.get('n_bootstrap', 50)
            # Get random_state from SST (determinism system) - no hardcoded defaults
            random_state = stability_config.get('random_state')
            if random_state is None:
                from TRAINING.common.determinism import stable_seed_from
                random_state = stable_seed_from(['stability_selection', target_column if target_column else 'default'])
            stability_cv = stability_config.get('cv', 3)
            stability_n_jobs = stability_config.get('n_jobs', 1)
            stability_cs = stability_config.get('Cs', 10)
            stability_scores = np.zeros(X_dense.shape[1])
            bootstrap_r2_scores = []
            
            # Use lasso config for stability selection models
            lasso_config = get_model_config('lasso', multi_model_config)
            
            for _ in range(n_bootstrap):
                # Use deterministic seed for bootstrap sampling
                from TRAINING.common.determinism import stable_seed_from
                bootstrap_seed = stable_seed_from(['bootstrap', target_column if 'target_column' in locals() else 'default', f'iter_{i}'])
                np.random.seed(bootstrap_seed)
                indices = np.random.choice(len(X_dense), size=len(X_dense), replace=True)
                X_boot, y_boot = X_dense[indices], y[indices]
                
                try:
                    # Use TimeSeriesSplit for internal CV (even though bootstrap breaks temporal order,
                    # this maintains consistency with the rest of the codebase)
                    # Clean config to prevent double random_state argument
                    from TRAINING.utils.config_cleaner import clean_config_for_estimator
                    if is_binary or is_multiclass:
                        lr_config = {'Cs': stability_cs, 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lr_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_config, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        model = LogisticRegressionCV(**lr_config_clean, random_state=random_state)
                    else:
                        lasso_config_clean_dict = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': stability_n_jobs}
                        lasso_config_clean = clean_config_for_estimator(LassoCV, lasso_config_clean_dict, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        model = LassoCV(**lasso_config_clean, random_state=random_state)
                    
                    model.fit(X_boot, y_boot)
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    stability_scores += (np.abs(coef) > 1e-6).astype(int)
                    
                    # Get R² using cross-validation (proper validation, not training score)
                    # Note: Bootstrap samples break temporal order, but we still use TimeSeriesSplit
                    # for consistency (it won't help here, but maintains the pattern)
                    # Use a quick model for CV scoring
                    if is_binary or is_multiclass:
                        lr_cv_config = {'Cs': [1.0], 'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lr_cv_config_clean = clean_config_for_estimator(LogisticRegressionCV, lr_cv_config, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        cv_model = LogisticRegressionCV(**lr_cv_config_clean, random_state=random_state)
                    else:
                        lasso_cv_config = {'cv': tscv, 'max_iter': lasso_config.get('max_iter', 1000), 'n_jobs': 1}
                        lasso_cv_config_clean = clean_config_for_estimator(LassoCV, lasso_cv_config, extra_kwargs={'random_state': random_state}, family_name='stability_selection')
                        cv_model = LassoCV(**lasso_cv_config_clean, random_state=random_state)
                    cv_scores = cross_val_score(cv_model, X_boot, y_boot, cv=tscv, scoring=scoring, n_jobs=1, error_score=np.nan)
                    valid_cv_scores = cv_scores[~np.isnan(cv_scores)]
                    if len(valid_cv_scores) > 0:
                        bootstrap_r2_scores.append(valid_cv_scores.mean())
                except:
                    continue
            
            # Update feature_names to match dense array
            feature_names = feature_names_dense
            
            # Average R² across bootstraps
            if bootstrap_r2_scores:
                model_scores['stability_selection'] = np.mean(bootstrap_r2_scores)
            else:
                model_scores['stability_selection'] = np.nan
            
            # Normalize stability scores to importance
            importance = stability_scores / n_bootstrap
            if len(importance) > 0:
                total_importance = np.sum(importance)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importance) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importance)[-top_k:])
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
            else:
                importance_ratio = 0.0
            importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Stability Selection failed: {e}")
    
    # Histogram Gradient Boosting
    if 'histogram_gradient_boosting' in model_families:
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
            
            # Get config values
            hgb_config = get_model_config('histogram_gradient_boosting', multi_model_config)
            # Defensive check: ensure config is a dict
            if not isinstance(hgb_config, dict):
                hgb_config = {}
            # Remove task-specific parameters (loss is set automatically by classifier/regressor)
            hgb_config_clean = {k: v for k, v in hgb_config.items() if k != 'loss'}
            
            if is_binary or is_multiclass:
                model = HistGradientBoostingClassifier(**hgb_config_clean)
            else:
                model = HistGradientBoostingRegressor(**hgb_config_clean)
            
            scores = cross_val_score(model, X, y, cv=tscv, scoring=scoring, n_jobs=cv_n_jobs, error_score=np.nan)
            valid_scores = scores[~np.isnan(scores)]
            primary_score = valid_scores.mean() if len(valid_scores) > 0 else np.nan
            
            # Train once to get importance and full metrics
            model.fit(X, y)
            
            # Compute and store full task-aware metrics
            if not np.isnan(primary_score):
                _compute_and_store_metrics('histogram_gradient_boosting', model, X, y, primary_score, task_type)
            if hasattr(model, 'feature_importances_'):
                # Use percentage of total importance in top 10% features (0-1 scale, interpretable)
                importances = model.feature_importances_
                total_importance = np.sum(importances)
                if total_importance > 0:
                    top_fraction = _get_importance_top_fraction()
                    top_k = max(1, int(len(importances) * top_fraction))
                    top_importance_sum = np.sum(np.sort(importances)[-top_k:])
                    # Normalize to 0-1: what % of total importance is in top 10%?
                    importance_ratio = top_importance_sum / total_importance
                else:
                    importance_ratio = 0.0
                importance_magnitudes.append(importance_ratio)
        except Exception as e:
            logger.warning(f"Histogram Gradient Boosting failed: {e}")
    
    mean_importance = np.mean(importance_magnitudes) if importance_magnitudes else 0.0
    
    # model_scores already contains primary scores (backward compatible)
    # model_metrics contains full metrics dict
    # all_suspicious_features contains leak detection results (aggregated across all models)
    # all_feature_importances contains detailed per-feature importances for export
    return model_metrics, model_scores, mean_importance, all_suspicious_features, all_feature_importances, fold_timestamps, _perfect_correlation_models


def _save_feature_importances(
    target_column: str,
    symbol: str,
    feature_importances: Dict[str, Dict[str, float]],
    output_dir: Path = None,
    view: str = "CROSS_SECTIONAL"
) -> None:
    """
    Save detailed per-model, per-feature importance scores to CSV files.
    
    Creates structure:
    {output_dir}/feature_importances/
      {target_name}/
        {symbol}/
          lightgbm_importances.csv
          xgboost_importances.csv
          random_forest_importances.csv
          ...
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        feature_importances: Dict of {model_name: {feature: importance}}
        output_dir: Base output directory (defaults to results/)
    """
    if output_dir is None:
        output_dir = _REPO_ROOT / "results"
    
    # Create directory structure in REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/{symbol}/feature_importances/
    # This aligns with the reproducibility structure and keeps all target ranking outputs together
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    # Determine base directory for REPRODUCIBILITY (should be at run level)
    if output_dir.name == "target_rankings":
        # output_dir is target_rankings/, go up to run level
        repro_base = output_dir.parent / "REPRODUCIBILITY" / "TARGET_RANKING"
    else:
        # output_dir is already at run level
        repro_base = output_dir / "REPRODUCIBILITY" / "TARGET_RANKING"
    
    if view == "SYMBOL_SPECIFIC" and symbol:
        importances_dir = repro_base / view / target_name_clean / f"symbol={symbol}" / "feature_importances"
    else:
        importances_dir = repro_base / view / target_name_clean / "feature_importances"
    importances_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-model CSV files
    # Sort model names for deterministic order (ensures reproducible file output)
    for model_name in sorted(feature_importances.keys()):
        importances = feature_importances[model_name]
        if not importances:
            continue
        
        # Create DataFrame sorted by importance
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in sorted(importances.items())  # Sort features for deterministic order
        ])
        df = df.sort_values('importance', ascending=False)
        
        # Normalize to percentages
        total = df['importance'].sum()
        if total > 0:
            df['importance_pct'] = (df['importance'] / total * 100).round(2)
            df['cumulative_pct'] = df['importance_pct'].cumsum().round(2)
        else:
            df['importance_pct'] = 0.0
            df['cumulative_pct'] = 0.0
        
        # Reorder columns
        df = df[['feature', 'importance', 'importance_pct', 'cumulative_pct']]
        
        # Save to CSV
        csv_file = importances_dir / f"{model_name}_importances.csv"
        df.to_csv(csv_file, index=False)
        
        # Save stability snapshot (non-invasive hook)
        # Pass the same repro_base directory so snapshots are saved alongside feature importances
        try:
            from TRAINING.stability.feature_importance import save_snapshot_hook
            # Use the same base directory structure as feature importances
            snapshot_output_dir = importances_dir.parent  # Same level as feature_importances/
            save_snapshot_hook(
                target_name=target_column,
                method=model_name,
                importance_dict=importances,
                universe_id=view,  # Use view parameter (CROSS_SECTIONAL or SYMBOL_SPECIFIC)
                output_dir=snapshot_output_dir,  # Save snapshots in same directory structure
                auto_analyze=None,  # Load from config
            )
        except Exception as e:
            logger.debug(f"Stability snapshot save failed (non-critical): {e}")
    
    logger.info(f"  💾 Saved feature importances to: {importances_dir}")


def _log_suspicious_features(
    target_column: str,
    symbol: str,
    suspicious_features: Dict[str, List[Tuple[str, float]]]
) -> None:
    """
    Log suspicious features to a file for later analysis.
    
    Args:
        target_column: Name of the target being evaluated
        symbol: Symbol being evaluated
        suspicious_features: Dict of {model_name: [(feature, importance), ...]}
    """
    leak_report_file = _REPO_ROOT / "results" / "leak_detection_report.txt"
    leak_report_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(leak_report_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Target: {target_column} | Symbol: {symbol}\n")
        f.write(f"{'='*80}\n")
        
        for model_name, features in suspicious_features.items():
            if features:
                f.write(f"\n{model_name.upper()} - Suspicious Features:\n")
                f.write(f"{'-'*80}\n")
                for feat, imp in sorted(features, key=lambda x: x[1], reverse=True):
                    f.write(f"  {feat:50s} | Importance: {imp:.1%}\n")
                f.write("\n")
    
    logger.info(f"  Leak detection report saved to: {leak_report_file}")


def detect_leakage(
    mean_score: float,
    composite_score: float,
    mean_importance: float,
    target_name: str = "",
    model_scores: Dict[str, float] = None,
    task_type: TaskType = TaskType.REGRESSION
) -> str:
    """
    Detect potential data leakage based on suspicious patterns.
    
    Returns:
        "OK" - No signs of leakage
        "HIGH_R2" - R² > threshold (suspiciously high)
        "INCONSISTENT" - Composite score too high for R² (possible leakage)
        "SUSPICIOUS" - Multiple warning signs
    """
    flags = []
    
    # Load thresholds from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            warning_cfg = leakage_cfg.get('warning_thresholds', {})
        except Exception:
            warning_cfg = {}
    else:
        warning_cfg = {}
    
    # Determine threshold based on task type and target name
    if task_type == TaskType.REGRESSION:
        is_forward_return = target_name.startswith('fwd_ret_')
        if is_forward_return:
            # For forward returns: R² > 0.50 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('forward_return', {})
            high_threshold = float(reg_cfg.get('high', 0.50))
            very_high_threshold = float(reg_cfg.get('very_high', 0.60))
            metric_name = "R²"
        else:
            # For barrier targets: R² > 0.70 is suspicious
            reg_cfg = warning_cfg.get('regression', {}).get('barrier', {})
            high_threshold = float(reg_cfg.get('high', 0.70))
            very_high_threshold = float(reg_cfg.get('very_high', 0.80))
            metric_name = "R²"
    elif task_type == TaskType.BINARY_CLASSIFICATION:
        # ROC-AUC > 0.95 is suspicious (near-perfect classification)
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        # Accuracy > 0.95 is suspicious
        class_cfg = warning_cfg.get('classification', {})
        high_threshold = float(class_cfg.get('high', 0.90))
        very_high_threshold = float(class_cfg.get('very_high', 0.95))
        metric_name = "Accuracy"
    
    # Check 1: Suspiciously high mean score
    if mean_score > very_high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={mean_score:.3f} > {very_high_threshold:.2f} "
            f"(extremely high - likely leakage)"
        )
    elif mean_score > high_threshold:
        flags.append("HIGH_SCORE")
        logger.warning(
            f"LEAKAGE WARNING: {metric_name}={mean_score:.3f} > {high_threshold:.2f} "
            f"(suspiciously high - investigate)"
        )
    
    # Check 2: Individual model scores too high (even if mean is lower)
    if model_scores:
        high_model_count = sum(1 for score in model_scores.values() 
                              if not np.isnan(score) and score > high_threshold)
        if high_model_count >= 3:  # 3+ models with high scores
            flags.append("HIGH_SCORE")
            logger.warning(
                f"LEAKAGE WARNING: {high_model_count} models have {metric_name} > {high_threshold:.2f} "
                f"(models: {[k for k, v in model_scores.items() if not np.isnan(v) and v > high_threshold]})"
            )
    
    # Check 3: Composite score inconsistent with mean score
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        composite_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.composite_score_high_threshold", default=0.5, config_name="safety_config"))
        regression_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_low_threshold", default=0.2, config_name="safety_config"))
        classification_score_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_low_threshold", default=0.6, config_name="safety_config"))
    except Exception:
        composite_high_threshold = 0.5
        regression_score_low = 0.2
        classification_score_low = 0.6
    
    score_low_threshold = regression_score_low if task_type == TaskType.REGRESSION else classification_score_low
    if composite_score > composite_high_threshold and mean_score < score_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Composite={composite_score:.3f} but {metric_name}={mean_score:.3f} "
            f"(inconsistent - possible leakage)"
        )
    
    # Check 4: Very high importance with low score (might indicate leaked features)
    # Load thresholds from config
    try:
        from CONFIG.config_loader import get_cfg
        importance_high_threshold = float(get_cfg("safety.leakage_detection.model_evaluation.importance_high_threshold", default=0.7, config_name="safety_config"))
        regression_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.regression_score_very_low_threshold", default=0.1, config_name="safety_config"))
        classification_score_very_low = float(get_cfg("safety.leakage_detection.model_evaluation.classification_score_very_low_threshold", default=0.5, config_name="safety_config"))
    except Exception:
        importance_high_threshold = 0.7
        regression_score_very_low = 0.1
        classification_score_very_low = 0.5
    
    score_very_low_threshold = regression_score_very_low if task_type == TaskType.REGRESSION else classification_score_very_low
    if mean_importance > importance_high_threshold and mean_score < score_very_low_threshold:
        flags.append("INCONSISTENT")
        logger.warning(
            f"LEAKAGE WARNING: Importance={mean_importance:.2f} but {metric_name}={mean_score:.3f} "
            f"(high importance with low {metric_name} - check for leaked features)"
        )
    
    if len(flags) > 1:
        return "SUSPICIOUS"
    elif len(flags) == 1:
        return flags[0]
    else:
        return "OK"


def calculate_composite_score(
    mean_score: float,
    std_score: float,
    mean_importance: float,
    n_models: int,
    task_type: TaskType = TaskType.REGRESSION
) -> float:
    """
    Calculate composite predictability score
    
    Components:
    - Mean score: Higher is better (R² for regression, ROC-AUC/Accuracy for classification)
    - Consistency: Lower std is better
    - Importance magnitude: Higher is better
    - Model agreement: More models = more confidence
    """
    
    # Normalize components based on task type
    if task_type == TaskType.REGRESSION:
        # R² can be negative, so normalize to 0-1 range
        score_component = max(0, mean_score)  # Clamp negative R² to 0
        consistency_component = 1.0 / (1.0 + std_score)
        
        # R²-weighted importance
        if mean_score > 0:
            importance_component = mean_importance * (1.0 + mean_score)
        else:
            penalty = abs(mean_score) * 0.67
            importance_component = mean_importance * max(0.5, 1.0 - penalty)
    else:
        # Classification: ROC-AUC and Accuracy are already 0-1
        score_component = mean_score  # Already 0-1
        consistency_component = 1.0 / (1.0 + std_score)
        
        # Score-weighted importance (similar logic but for 0-1 scores)
        importance_component = mean_importance * (1.0 + mean_score)
    
    # Weighted average
    composite = (
        0.50 * score_component +        # 50% weight on score
        0.25 * consistency_component + # 25% on consistency
        0.25 * importance_component    # 25% on score-weighted importance
    )
    
    # Bonus for more models (up to 10% boost)
    model_bonus = min(0.1, n_models * 0.02)
    composite = composite * (1.0 + model_bonus)
    
    return composite



def evaluate_target_predictability(
    target_name: str,
    target_config: Dict[str, Any] | TargetConfig,
    symbols: List[str],
    data_dir: Path,
    model_families: List[str],
    multi_model_config: Dict[str, Any] = None,
    output_dir: Path = None,
    min_cs: int = 10,
    max_cs_samples: Optional[int] = None,
    max_rows_per_symbol: int = None,
    explicit_interval: Optional[Union[int, str]] = None,  # Explicit interval from config (e.g., "5m")
    experiment_config: Optional[Any] = None,  # Optional ExperimentConfig (for data.bar_interval)
    view: str = "CROSS_SECTIONAL",  # "CROSS_SECTIONAL", "SYMBOL_SPECIFIC", or "LOSO"
    symbol: Optional[str] = None  # Required for SYMBOL_SPECIFIC and LOSO views
) -> TargetPredictabilityScore:
    """Evaluate predictability of a single target across symbols"""
    
    # Ensure numpy is available (imported at module level, but ensure it's accessible)
    import numpy as np  # Use global import from top of file
    
    # Get logging config for this module (at function start)
    if _LOGGING_CONFIG_AVAILABLE:
        log_cfg = get_module_logging_config('rank_target_predictability')
    else:
        log_cfg = _DummyLoggingConfig()
    
    # ============================================================================
    # CONFIG TRACE: Data loading limits (with provenance)
    # ============================================================================
    import os
    config_provenance = {}
    
    # Load default max_rows_per_symbol from config if not provided
    if max_rows_per_symbol is None:
        # First check experiment config if available
        if experiment_config and hasattr(experiment_config, 'max_samples_per_symbol'):
            max_rows_per_symbol = experiment_config.max_samples_per_symbol
            config_provenance['max_rows_per_symbol'] = f"experiment_config.max_samples_per_symbol = {max_rows_per_symbol}"
            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config")
        else:
            # Try reading from experiment config YAML directly
            if experiment_config:
                try:
                    import yaml
                    exp_name = experiment_config.name
                    exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                    if exp_file.exists():
                        with open(exp_file, 'r') as f:
                            exp_yaml = yaml.safe_load(f) or {}
                        exp_data = exp_yaml.get('data', {})
                        if 'max_samples_per_symbol' in exp_data:
                            max_rows_per_symbol = exp_data['max_samples_per_symbol']
                            config_provenance['max_rows_per_symbol'] = f"experiment YAML data.max_samples_per_symbol = {max_rows_per_symbol}"
                            logger.debug(f"Using max_rows_per_symbol={max_rows_per_symbol} from experiment config YAML")
                except Exception:
                    pass
            
            # Fallback to pipeline config
            if max_rows_per_symbol is None:
                if _CONFIG_AVAILABLE:
                    try:
                        max_rows_per_symbol = int(get_cfg("pipeline.data_limits.default_max_rows_per_symbol_ranking", default=50000, config_name="pipeline_config"))
                        config_provenance['max_rows_per_symbol'] = f"pipeline_config.pipeline.data_limits.default_max_rows_per_symbol_ranking = {max_rows_per_symbol} (default=50000)"
                    except Exception:
                        max_rows_per_symbol = 50000
                        config_provenance['max_rows_per_symbol'] = f"hardcoded default = 50000"
                else:
                    max_rows_per_symbol = 50000
                    config_provenance['max_rows_per_symbol'] = f"hardcoded default = 50000 (config unavailable)"
    else:
        config_provenance['max_rows_per_symbol'] = f"passed as parameter = {max_rows_per_symbol}"
    
    # Trace max_cs_samples
    if max_cs_samples is None:
        # First check experiment config YAML
        if experiment_config:
            try:
                import yaml
                exp_name = experiment_config.name
                exp_file = Path("CONFIG/experiments") / f"{exp_name}.yaml"
                if exp_file.exists():
                    with open(exp_file, 'r') as f:
                        exp_yaml = yaml.safe_load(f) or {}
                    exp_data = exp_yaml.get('data', {})
                    if 'max_cs_samples' in exp_data:
                        max_cs_samples = exp_data['max_cs_samples']
                        config_provenance['max_cs_samples'] = f"experiment YAML data.max_cs_samples = {max_cs_samples}"
                        logger.debug(f"Using max_cs_samples={max_cs_samples} from experiment config YAML")
            except Exception:
                pass
        
        # Fallback to pipeline config
        if max_cs_samples is None:
            if _CONFIG_AVAILABLE:
                try:
                    max_cs_samples = int(get_cfg("pipeline.data_limits.max_cs_samples", default=1000, config_name="pipeline_config"))
                    config_provenance['max_cs_samples'] = f"pipeline_config.pipeline.data_limits.max_cs_samples = {max_cs_samples} (default=1000)"
                except Exception:
                    max_cs_samples = 1000
                    config_provenance['max_cs_samples'] = f"hardcoded default = 1000"
            else:
                max_cs_samples = 1000
                config_provenance['max_cs_samples'] = f"hardcoded default = 1000 (config unavailable)"
    else:
        config_provenance['max_cs_samples'] = f"passed as parameter = {max_cs_samples}"
    
    # Log config trace
    logger.info("=" * 80)
    logger.info("📋 CONFIG TRACE: Data Loading Limits (with provenance)")
    logger.info("=" * 80)
    logger.info(f"   Working directory: {os.getcwd()}")
    logger.info(f"   Experiment config: {experiment_config.name if experiment_config else 'None'}")
    logger.info("")
    logger.info("   🔍 Resolved values:")
    logger.info(f"      max_rows_per_symbol: {max_rows_per_symbol}")
    logger.info(f"         Source: {config_provenance.get('max_rows_per_symbol', 'unknown')}")
    logger.info(f"      max_cs_samples: {max_cs_samples}")
    logger.info(f"         Source: {config_provenance.get('max_cs_samples', 'unknown')}")
    logger.info(f"      min_cs: {min_cs}")
    logger.info("=" * 80)
    logger.info("")
    
    # Convert dict config to TargetConfig if needed
    if isinstance(target_config, dict):
        target_column = target_config['target_column']
        display_name = target_config.get('display_name', target_name)
        # Infer task type from column name (will be refined with actual data)
        task_type = TaskType.from_target_column(target_column)
        target_config_obj = TargetConfig(
            name=target_name,
            target_column=target_column,
            task_type=task_type,
            display_name=display_name,
            **{k: v for k, v in target_config.items() 
               if k not in ['target_column', 'display_name']}
        )
    else:
        target_config_obj = target_config
        target_column = target_config_obj.target_column
        display_name = target_config_obj.display_name or target_name
    # Validate view and symbol parameters
    if view == "SYMBOL_SPECIFIC" and symbol is None:
        raise ValueError(f"symbol parameter required for SYMBOL_SPECIFIC view")
    if view == "LOSO" and symbol is None:
        raise ValueError(f"symbol parameter required for LOSO view")
    if view == "CROSS_SECTIONAL" and symbol is not None:
        logger.warning(f"symbol={symbol} provided but view=CROSS_SECTIONAL, ignoring symbol")
        symbol = None
    
    view_display = f"{view}" + (f" (symbol={symbol})" if symbol else "")
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating: {display_name} ({target_column}) - {view_display}")
    logger.info(f"{'='*60}")
    
    # Load data based on view
    from TRAINING.utils.cross_sectional_data import load_mtf_data_for_ranking, prepare_cross_sectional_data_for_ranking
    from TRAINING.utils.leakage_filtering import filter_features_for_target
    from TRAINING.utils.target_conditional_exclusions import (
        generate_target_exclusion_list,
        load_target_exclusion_list
    )
    
    # For SYMBOL_SPECIFIC and LOSO, filter symbols
    symbols_to_load = symbols
    if view == "SYMBOL_SPECIFIC":
        symbols_to_load = [symbol]
    elif view == "LOSO":
        # LOSO: train on all symbols except symbol, validate on symbol
        symbols_to_load = [s for s in symbols if s != symbol]
        validation_symbol = symbol
    else:
        validation_symbol = None
    
    logger.info(f"Loading data for {len(symbols_to_load)} symbol(s) (max {max_rows_per_symbol} rows per symbol)...")
    if view == "LOSO":
        logger.info(f"  LOSO: Training on {len(symbols_to_load)} symbols, validating on {validation_symbol}")
    mtf_data = load_mtf_data_for_ranking(data_dir, symbols_to_load, max_rows_per_symbol=max_rows_per_symbol)
    
    if not mtf_data:
        logger.error(f"No data loaded for any symbols")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Apply leakage filtering to feature list BEFORE preparing data (with registry validation)
    # Get all columns from first symbol to determine available features
    sample_df = next(iter(mtf_data.values()))
    all_columns = sample_df.columns.tolist()

    # TARGET-CONDITIONAL EXCLUSIONS: Generate per-target exclusion list
    # This implements "Target-Conditional Feature Selection" - tailoring features to target physics
    target_conditional_exclusions = []
    exclusion_metadata = {}
    target_exclusion_dir = None
    
    if output_dir:
        # Determine base output directory (RESULTS/{run}/)
        # output_dir might be: RESULTS/{run}/target_rankings/ or RESULTS/{run}/
        if output_dir.name == "target_rankings":
            base_output_dir = output_dir.parent
        else:
            base_output_dir = output_dir
        
        # Save feature exclusions to REPRODUCIBILITY/TARGET_RANKING/{view}/{target}/feature_exclusions/
        # This keeps all target ranking outputs together in the reproducibility structure
        # Note: Exclusions are shared at target level (not per-symbol, not per-cohort)
        target_name_clean = target_name.replace('/', '_').replace('\\', '_')
        repro_base = base_output_dir / "REPRODUCIBILITY" / "TARGET_RANKING"
        target_exclusion_dir = repro_base / view / target_name_clean / "feature_exclusions"
        target_exclusion_dir.mkdir(parents=True, exist_ok=True)
        
        # Try to load existing exclusion list first (from RESULTS/{cohort}/{run}/feature_exclusions/)
        # This allows reusing exclusion lists across runs for consistency
        existing_exclusions = load_target_exclusion_list(target_name, target_exclusion_dir)
        if existing_exclusions is not None:
            target_conditional_exclusions = existing_exclusions
            logger.info(
                f"📋 Loaded existing target-conditional exclusions for {target_name}: "
                f"{len(target_conditional_exclusions)} features "
                f"(from {target_exclusion_dir})"
            )
        else:
            # Generate new exclusion list
            try:
                from TRAINING.common.feature_registry import get_registry
                registry = get_registry()
            except Exception:
                registry = None
            
            # Detect interval for lookback calculation
            from TRAINING.utils.data_interval import detect_interval_from_dataframe
            temp_interval = detect_interval_from_dataframe(sample_df, explicit_interval=explicit_interval)
            
            target_conditional_exclusions, exclusion_metadata = generate_target_exclusion_list(
                target_name=target_name,
                all_features=all_columns,
                interval_minutes=temp_interval,
                output_dir=target_exclusion_dir,
                registry=registry
            )
            
            if target_conditional_exclusions:
                logger.info(
                    f"📋 Generated target-conditional exclusions for {target_name}: "
                    f"{len(target_conditional_exclusions)} features excluded "
                    f"(horizon={exclusion_metadata.get('target_horizon_minutes', 'unknown')}m, "
                    f"semantics={exclusion_metadata.get('target_semantics', {})})"
                )
    else:
        # No output_dir - skip target-conditional exclusions (backward compatibility)
        logger.debug("No output_dir provided - skipping target-conditional exclusions")

    # Detect data interval for horizon conversion (use explicit_interval if provided)
    from TRAINING.utils.data_interval import detect_interval_from_dataframe
    detected_interval = detect_interval_from_dataframe(
        sample_df,
        timestamp_column='ts', 
        default=5,
        explicit_interval=explicit_interval,
        experiment_config=experiment_config
    )
    
    # Extract target horizon for error messages
    from TRAINING.utils.leakage_filtering import _load_leakage_config, _extract_horizon
    leakage_config = _load_leakage_config()
    target_horizon_minutes = _extract_horizon(target_column, leakage_config) if target_column else None
    target_horizon_bars = None
    if target_horizon_minutes is not None and detected_interval > 0:
        target_horizon_bars = int(target_horizon_minutes // detected_interval)
    
    # Use target-aware filtering with registry validation
    # Apply target-conditional exclusions BEFORE global filtering
    # This ensures target-specific rules are applied first
    columns_after_target_exclusions = [c for c in all_columns if c not in target_conditional_exclusions]
    
    if target_conditional_exclusions:
        logger.info(
            f"  🎯 Target-conditional exclusions: Removed {len(target_conditional_exclusions)} features "
            f"({len(columns_after_target_exclusions)} remaining before global filtering)"
        )
    
    # Apply global filtering (registry, patterns, etc.)
    safe_columns = filter_features_for_target(
        columns_after_target_exclusions,  # Use pre-filtered columns
        target_column,
        verbose=True,
        use_registry=True,  # Enable registry validation
        data_interval_minutes=detected_interval,
        for_ranking=True,  # Use permissive rules for ranking (allow basic OHLCV/TA)
        dropped_tracker=dropped_tracker if 'dropped_tracker' in locals() else None  # Pass tracker for sanitizer tracking
    )
    
    excluded_count = len(all_columns) - len(safe_columns) - 1  # -1 for target itself
    features_safe = len(safe_columns)
    logger.debug(f"Filtered out {excluded_count} potentially leaking features (kept {features_safe} safe features)")
    
    # NEW: Track early filter drops (schema/pattern/registry filtering) - set-based comparison
    if 'dropped_tracker' in locals() and dropped_tracker is not None and 'all_columns_before_filter' in locals():
        early_filtered = sorted(list(set(all_columns_before_filter) - set(safe_columns)))
        if early_filtered:
            dropped_tracker.add_early_filter_summary(
                filter_name="schema_pattern_registry",
                dropped_count=len(early_filtered),
                top_samples=early_filtered[:10],
                rule_hits=None  # Could be enhanced to track which rules hit
            )
    
    # CRITICAL: Check if we have enough features to train
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_REQUIRED = int(ranking_cfg.get('min_features_required', 2))
        except Exception:
            MIN_FEATURES_REQUIRED = 2
    else:
        MIN_FEATURES_REQUIRED = 2
    
    if len(safe_columns) < MIN_FEATURES_REQUIRED:
        # Always log both minutes and bars for clarity
        if target_horizon_minutes is not None and target_horizon_bars is not None:
            horizon_info = f"horizon_minutes={target_horizon_minutes:.1f}m, horizon_bars={target_horizon_bars} bars @ interval={detected_interval:.1f}m"
        elif target_horizon_bars is not None:
            horizon_info = f"horizon_bars={target_horizon_bars} bars @ interval={detected_interval:.1f}m"
        else:
            horizon_info = "this horizon"
        logger.error(
            f"❌ INSUFFICIENT FEATURES: Only {len(safe_columns)} features remain after filtering "
            f"(minimum required: {MIN_FEATURES_REQUIRED}). "
            f"This target may not be predictable with current feature set. "
            f"Consider:\n"
            f"  1. Adding more features to CONFIG/feature_registry.yaml with allowed_horizons including {horizon_info}\n"
            f"  2. Relaxing feature registry rules for short-horizon targets\n"
            f"  3. Checking if excluded_features.yaml is too restrictive\n"
            f"  4. Skipping this target and focusing on targets with longer horizons"
        )
        # Return -999.0 to indicate this target should be skipped (same as degenerate targets)
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=target_config_obj.task_type,
            mean_score=-999.0,  # Flag for filtering (same as degenerate targets)
            std_score=0.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            composite_score=0.0,
            leakage_flag="INSUFFICIENT_FEATURES"
        )
    
    # Track feature counts (will be updated after data preparation)
    features_dropped_nan = 0
    features_final = features_safe
    
    # NEW: Track NaN drops - capture BEFORE data prep (set-based comparison)
    feature_names_before_data_prep = safe_columns.copy() if 'safe_columns' in locals() else []
    
    # CRITICAL: Initialize resolved_config early to avoid "referenced before assignment" errors
    # We need it for SYMBOL_SPECIFIC view data preparation
    resolved_config = None
    from TRAINING.utils.resolved_config import create_resolved_config
    
    # Get n_symbols_available from mtf_data (needed for resolved_config creation)
    n_symbols_available = len(mtf_data) if mtf_data is not None else 0
    
    # Create baseline resolved_config early (WITH feature lookback computation)
    # This is needed for SYMBOL_SPECIFIC view data preparation
    selected_features = safe_columns.copy() if safe_columns else []
    resolved_config = create_resolved_config(
        requested_min_cs=min_cs if view != "SYMBOL_SPECIFIC" else 1,
        n_symbols_available=n_symbols_available,
        max_cs_samples=max_cs_samples,
        interval_minutes=detected_interval,
        horizon_minutes=target_horizon_minutes,
        feature_names=selected_features,
        experiment_config=experiment_config
    )
    
    # Prepare data based on view
    if view == "SYMBOL_SPECIFIC":
        # For symbol-specific, prepare single-symbol time series data
        # Use same function but with single symbol (min_cs=1 effectively)
        X, y, feature_names, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=1, max_cs_samples=max_cs_samples, feature_names=safe_columns,
            feature_time_meta_map=resolved_config.feature_time_meta_map if resolved_config else None,
            base_interval_minutes=resolved_config.base_interval_minutes if resolved_config else None
        )
        # Verify we only have one symbol
        unique_symbols = set(symbols_array) if symbols_array is not None else set()
        if len(unique_symbols) > 1:
            logger.warning(f"SYMBOL_SPECIFIC view expected 1 symbol, got {len(unique_symbols)}: {unique_symbols}")
    elif view == "LOSO":
        # LOSO: prepare training data (all symbols except validation symbol)
        X_train, y_train, feature_names_train, symbols_array_train, time_vals_train, resolved_data_config_train = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns
        )
        # Load validation symbol data separately
        validation_mtf_data = load_mtf_data_for_ranking(data_dir, [validation_symbol], max_rows_per_symbol=max_rows_per_symbol)
        X_val, y_val, feature_names_val, symbols_array_val, time_vals_val, resolved_data_config_val = prepare_cross_sectional_data_for_ranking(
            validation_mtf_data, target_column, min_cs=1, max_cs_samples=None, feature_names=safe_columns
        )
        # For LOSO, we'll use a special CV that trains on X_train and validates on X_val
        # For now, combine them and use a custom splitter (will be implemented in train_and_evaluate_models)
        # TODO: Implement LOSO-specific CV splitter
        logger.warning("LOSO view: Using combined data for now (LOSO-specific CV splitter not yet implemented)")
        X = X_train  # Will be handled by LOSO-specific logic
        y = y_train
        feature_names = feature_names_train
        symbols_array = symbols_array_train
        time_vals = time_vals_train
    else:
        # CROSS_SECTIONAL: standard pooled data
        X, y, feature_names, symbols_array, time_vals, resolved_data_config = prepare_cross_sectional_data_for_ranking(
            mtf_data, target_column, min_cs=min_cs, max_cs_samples=max_cs_samples, feature_names=safe_columns
        )
    
    # Update feature counts after data preparation
    if feature_names is not None:
        features_final = len(feature_names)
        features_dropped_nan = features_safe - features_final
    
    # Store cohort metadata context for later use in reproducibility tracking
    # These will be used to extract cohort metadata at the end of the function
    cohort_context = {
        'X': X,
        'y': y,  # Label vector for data fingerprint
        'time_vals': time_vals,
        'symbols_array': symbols_array,
        'mtf_data': mtf_data,
        'symbols': symbols,
        'min_cs': min_cs,
        'max_cs_samples': max_cs_samples
    }
    
    if X is None or y is None:
        logger.error(f"Failed to prepare cross-sectional data for {target_column}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # NOTE: resolved_config was already initialized earlier (before data preparation)
    # This section updates it with post-pruning feature information
    # Get n_symbols_available from mtf_data (if not already set)
    if 'n_symbols_available' not in locals():
        n_symbols_available = len(mtf_data) if mtf_data is not None else 0
    
    # Update resolved_config with post-pruning feature information
    # CRITICAL FIX: Compute feature lookback early to ensure purge is large enough
    # This prevents "ROLLING WINDOW LEAKAGE RISK" violations
    selected_features = feature_names.copy() if feature_names else []
    
    # Update config (WITH feature lookback computation for auto-adjustment)
    # The auto-fix logic in create_resolved_config will increase purge if feature_lookback > purge
    resolved_config = create_resolved_config(
        requested_min_cs=min_cs if view != "SYMBOL_SPECIFIC" else 1,
        n_symbols_available=n_symbols_available,
        max_cs_samples=max_cs_samples,
        interval_minutes=detected_interval,
        horizon_minutes=target_horizon_minutes,
        feature_lookback_max_minutes=None,  # Will be computed from feature_names
        purge_buffer_bars=5,  # Default from config
        default_purge_minutes=None,  # Loads from safety_config.yaml (SST)
        features_safe=features_safe,
        features_dropped_nan=features_dropped_nan,
        features_final=len(selected_features),
        view=view,
        symbol=symbol,
        feature_names=selected_features,  # Pass feature names for lookback computation
        recompute_lookback=True,  # CRITICAL: Compute feature lookback to auto-adjust purge
        experiment_config=experiment_config  # NEW: Pass experiment_config for base_interval_minutes
    )
    
    if log_cfg.cv_detail:
        logger.info(f"  ✅ Baseline resolved config (pre-prune): purge={resolved_config.purge_minutes:.1f}m, embargo={resolved_config.embargo_minutes:.1f}m")
    
    logger.info(f"Cross-sectional data: {len(X)} samples, {X.shape[1]} features")
    logger.info(f"Symbols: {len(set(symbols_array))} unique symbols")
    
    # Infer task type from data (needed for leak scan)
    y_sample = pd.Series(y).dropna()
    task_type = TaskType.from_target_column(target_column, y_sample.to_numpy())
    
    # PRE-TRAINING LEAK SCAN: Detect and remove near-copy features before model training
    logger.info("🔍 Pre-training leak scan: Checking for near-copy features...")
    feature_names_before_leak_scan = feature_names.copy()
    
    # Check for duplicate column names before leak scan
    if len(feature_names) != len(set(feature_names)):
        duplicates = [name for name in set(feature_names) if feature_names.count(name) > 1]
        logger.error(f"  🚨 DUPLICATE COLUMN NAMES DETECTED before leak scan: {duplicates}")
        raise ValueError(f"Duplicate feature names detected: {duplicates}")
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    leaky_features = find_near_copy_features(X_df, y_series, task_type)
    
    if leaky_features:
        logger.error(
            f"  ❌ CRITICAL: Found {len(leaky_features)} leaky features that are near-copies of target: {leaky_features}"
        )
        logger.error(
            f"  Removing leaky features and continuing with {X.shape[1] - len(leaky_features)} features..."
        )
        
        # Remove leaky features
        leaky_indices = [i for i, name in enumerate(feature_names) if name in leaky_features]
        X = np.delete(X, leaky_indices, axis=1)
        feature_names = [name for name in feature_names if name not in leaky_features]
        
        # CRITICAL: Reindex X columns to match feature_names order (prevent order drift)
        # After leak removal, ensure X columns match feature_names order exactly
        # This prevents "(order changed)" warnings and ensures deterministic column alignment
        if X.shape[1] != len(feature_names):
            logger.warning(
                f"  ⚠️ Column count mismatch after leak removal: X.shape[1]={X.shape[1]}, "
                f"len(feature_names)={len(feature_names)}. This should not happen."
            )
        # Note: For numpy arrays, column order is implicit via feature_names list
        # The feature_names list IS the authoritative order - X columns must match it
        
        logger.info(f"  After leak removal: {X.shape[1]} features remaining")
        from TRAINING.utils.cross_sectional_data import _log_feature_set
        _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
        
        # If we removed too many features, mark as insufficient
        # Load from config
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                ranking_cfg = leakage_cfg.get('ranking', {})
                MIN_FEATURES_AFTER_LEAK_REMOVAL = int(ranking_cfg.get('min_features_after_leak_removal', 2))
            except Exception:
                MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
        else:
            MIN_FEATURES_AFTER_LEAK_REMOVAL = 2
        
        if X.shape[1] < MIN_FEATURES_AFTER_LEAK_REMOVAL:
            logger.error(
                f"  ❌ Too few features remaining after leak removal ({X.shape[1]}). "
                f"Marking target as LEAKAGE_DETECTED."
            )
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=0.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={},
                composite_score=0.0,
                leakage_flag="LEAKAGE_DETECTED"
            )
    else:
        logger.info("  ✅ No obvious leaky features detected in pre-training scan")
        from TRAINING.utils.cross_sectional_data import _log_feature_set
        _log_feature_set("AFTER_LEAK_REMOVAL", feature_names, previous_names=feature_names_before_leak_scan, logger_instance=logger)
    
    # CRITICAL: Early exit if too few features (before wasting time training models)
    # Load from config
    if _CONFIG_AVAILABLE:
        try:
            safety_cfg = get_safety_config()
            # safety_config.yaml has a top-level 'safety' key
            safety_section = safety_cfg.get('safety', {})
            leakage_cfg = safety_section.get('leakage_detection', {})
            ranking_cfg = leakage_cfg.get('ranking', {})
            MIN_FEATURES_FOR_MODEL = int(ranking_cfg.get('min_features_for_model', 3))
        except Exception:
            MIN_FEATURES_FOR_MODEL = 3
    else:
        MIN_FEATURES_FOR_MODEL = 3
    
    if X.shape[1] < MIN_FEATURES_FOR_MODEL:
        logger.warning(
            f"Too few features ({X.shape[1]}) after filtering (minimum: {MIN_FEATURES_FOR_MODEL}); "
            f"marking target as degenerate and skipping model training."
        )
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if we get further
            mean_score=-999.0,  # Flag for filtering
            std_score=0.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            composite_score=0.0,
            leakage_flag="INSUFFICIENT_FEATURES"
        )
    
    # Task type already inferred above for leak scan
    
    # Validate target
    is_valid, error_msg = validate_target(y, task_type=task_type)
    if not is_valid:
        logger.warning(f"Skipping: {error_msg}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Check if target is degenerate
    unique_vals = np.unique(y[~np.isnan(y)])
    if len(unique_vals) < 2:
        logger.warning(f"Skipping: Target has only {len(unique_vals)} unique value(s)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # For classification, check if classes are too imbalanced for CV
    if len(unique_vals) <= 10:  # Likely classification
        class_counts = np.bincount(y[~np.isnan(y)].astype(int))
        min_class_count = class_counts[class_counts > 0].min()
        if min_class_count < 2:
            logger.warning(f"Skipping: Smallest class has only {min_class_count} sample(s) (too few for CV)")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
    
    # CRITICAL: Recompute resolved_config AFTER pruning (if pruning happened)
    # This ensures feature_lookback_max is computed from actual pruned features
    # If pruning didn't happen or failed, we keep the baseline config (already assigned above)
    # Note: Pruning happens inside train_and_evaluate_models, so we need to handle it there
    # For now, we'll recompute here if feature_names changed (indicating pruning happened externally)
    # The actual post-prune recomputation happens in train_and_evaluate_models
    
    # Log baseline config summary
    if log_cfg.cv_detail:
        resolved_config.log_summary(logger)

    # Log active leakage policy (CRITICAL for audit)
    policy = "strict"  # Default
    try:
        from CONFIG.config_loader import get_cfg
        policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
    except Exception:
        pass
    logger.info(f"🔒 Leakage policy: {policy} (strict=hard-stop, drop_features=auto-drop, warn=log-only)")
    
    # FINAL GATEKEEPER: Enforce safety at the last possible moment
    # This runs AFTER all loading/merging/sanitization is done
    # It physically drops features that violate the purge limit from the dataframe
    # This is the "worry-free" auto-corrector that handles race conditions
    from TRAINING.utils.cross_sectional_data import _log_feature_set, _compute_feature_fingerprint
    pre_gatekeeper_fp, _ = _compute_feature_fingerprint(feature_names, set_invariant=True)
    _log_feature_set("PRE_GATEKEEPER", feature_names, previous_names=None, logger_instance=logger)
    
    X, feature_names = _enforce_final_safety_gate(
        X=X,
        feature_names=feature_names,
        resolved_config=resolved_config,
        interval_minutes=detected_interval,
        logger=logger,
        dropped_tracker=dropped_tracker if 'dropped_tracker' in locals() else None
    )
    
    # CRITICAL: Log POST_GATEKEEPER stage explicitly
    post_gatekeeper_fp, post_gatekeeper_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
    _log_feature_set("POST_GATEKEEPER", feature_names, previous_names=None, logger_instance=logger)
    
    # CRITICAL: Hard-fail check: POST_GATEKEEPER must have ZERO unknowns in strict mode
    # This is the contract: post-enforcement stages should never see unknowns
    if resolved_config and hasattr(resolved_config, '_gatekeeper_enforced'):
        enforced_gatekeeper = resolved_config._gatekeeper_enforced
        
        # PHASE 1: Persist FeatureSet artifact for debugging
        if output_dir is not None:
            try:
                from TRAINING.utils.feature_set_artifact import create_artifact_from_enforced
                artifact = create_artifact_from_enforced(
                    enforced_gatekeeper,
                    stage="POST_GATEKEEPER",
                    removal_reasons={}
                )
                artifact_dir = output_dir / "REPRODUCIBILITY" / "FEATURESET_ARTIFACTS"
                artifact.save(artifact_dir)
            except Exception as e:
                logger.debug(f"  ⚠️  Failed to persist POST_GATEKEEPER artifact: {e}")
        if len(enforced_gatekeeper.unknown) > 0:
            policy = "strict"
            try:
                from CONFIG.config_loader import get_cfg
                policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
            except Exception:
                pass
            
            if policy == "strict":
                error_msg = (
                    f"🚨 POST_GATEKEEPER CONTRACT VIOLATION: {len(enforced_gatekeeper.unknown)} features have unknown lookback (inf). "
                    f"In strict mode, post-enforcement stages must have ZERO unknowns. "
                    f"Gatekeeper should have quarantined these. "
                    f"Sample: {enforced_gatekeeper.unknown[:10]}"
                )
                logger.error(error_msg)
                raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
            else:
                logger.warning(
                    f"⚠️ POST_GATEKEEPER: {len(enforced_gatekeeper.unknown)} features have unknown lookback (inf). "
                    f"Policy={policy} allows this, but this is unexpected after enforcement."
                )
        
        # CRITICAL: Boundary assertion - validate feature_names matches gatekeeper EnforcedFeatureSet
        from TRAINING.utils.lookback_policy import assert_featureset_fingerprint
        try:
            assert_featureset_fingerprint(
                label="POST_GATEKEEPER",
                expected=enforced_gatekeeper,
                actual_features=feature_names,
                logger_instance=logger,
                allow_reorder=False  # Strict order check
            )
        except RuntimeError as e:
            # Log but don't fail - this is a validation check
            logger.error(f"POST_GATEKEEPER assertion failed: {e}")
            # Fix it: use enforced.features (the truth)
            feature_names = enforced_gatekeeper.features.copy()
            logger.info(f"Fixed: Updated feature_names to match gatekeeper_enforced.features")
    
    # NOTE: MODEL_TRAIN_INPUT fingerprint will be computed in train_and_evaluate_models AFTER pruning
    # Pruning happens inside train_and_evaluate_models, so we can't set it here
    
    # CRITICAL: Recompute resolved_config.feature_lookback_max AFTER Final Gatekeeper
    # The audit system uses this value, so it must reflect the ACTUAL features that will be trained
    # (not the original features before the gatekeeper dropped problematic ones)
    if feature_names and len(feature_names) > 0:
        from TRAINING.utils.leakage_budget import compute_budget
        from TRAINING.utils.resolved_config import compute_feature_lookback_max
        
        # Get registry for lookback calculation
        registry = None
        try:
            from TRAINING.common.feature_registry import get_registry
            registry = get_registry()
        except Exception:
            pass
        
        # Load lookback_budget_minutes cap for consistency with gatekeeper
        lookback_budget_cap_for_budget = None
        budget_cap_provenance_post_gatekeeper = None
        try:
            from CONFIG.config_loader import get_cfg, get_config_path
            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
            config_path = get_config_path("safety_config")
            budget_cap_provenance_post_gatekeeper = f"safety_config.yaml:{config_path} → safety.leakage_detection.lookback_budget_minutes = {budget_cap_raw} (default='auto')"
            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                lookback_budget_cap_for_budget = float(budget_cap_raw)
        except Exception as e:
            budget_cap_provenance_post_gatekeeper = f"config lookup failed: {e}"
        
        # Log config trace for budget compute
        logger.info(f"📋 CONFIG TRACE (POST_GATEKEEPER budget): {budget_cap_provenance_post_gatekeeper}")
        logger.info(f"   → max_lookback_cap_minutes passed to compute_budget: {lookback_budget_cap_for_budget}")
        
        # Compute budget from FINAL feature set (post gatekeeper)
        # NOTE: MODEL_TRAIN_INPUT fingerprint will be computed later in train_and_evaluate_models AFTER pruning
        # For now, validate against post_gatekeeper fingerprint
        budget, computed_fp, computed_order_fp = compute_budget(
            feature_names,
            detected_interval,
            resolved_config.horizon_minutes if resolved_config else 60.0,
            registry=registry,
            max_lookback_cap_minutes=lookback_budget_cap_for_budget,  # Pass cap for consistency
            expected_fingerprint=post_gatekeeper_fp,
            stage="POST_GATEKEEPER"
        )
        
        # SANITY CHECK: Verify POST_GATEKEEPER max_lookback_minutes respects the cap
        # CRITICAL: Use the canonical map that was already computed (don't recompute)
        # The budget was computed using canonical_lookback_map, so we can use that same map
        if lookback_budget_cap_for_budget is not None:
            # Get canonical map from the budget computation (it was passed in)
            # We need to recompute it here since we don't have a reference, but we'll use the same logic
            # Actually, better: use compute_feature_lookback_max which builds the canonical map correctly
            from TRAINING.utils.leakage_budget import compute_feature_lookback_max
            lookback_result = compute_feature_lookback_max(
                feature_names,
                detected_interval,
                max_lookback_cap_minutes=lookback_budget_cap_for_budget,
                registry=registry,
                expected_fingerprint=post_gatekeeper_fp,
                stage="POST_GATEKEEPER_sanity_check"
            )
            # CRITICAL: Use the EXACT SAME oracle as final enforcement
            # This is the single source of truth - if it disagrees, we have split-brain
            actual_max_from_features = lookback_result.max_minutes if lookback_result.max_minutes is not None else 0.0
            budget_max = budget.max_feature_lookback_minutes
            
            # CRITICAL: Hard-fail on mismatch (split-brain detection)
            # Both should use the same canonical map, so they MUST agree
            if abs(actual_max_from_features - budget_max) > 1.0:
                # This is a real bug - different code paths are computing different lookbacks
                logger.error(
                    f"🚨 SPLIT-BRAIN DETECTED (POST_GATEKEEPER): "
                    f"budget.max={budget_max:.1f}m vs actual_max_from_features={actual_max_from_features:.1f}m. "
                    f"This indicates different code paths are computing different lookbacks. "
                    f"Both should use the same canonical map from compute_feature_lookback_max()."
                )
                # In strict mode, this is a hard-stop
                policy = "strict"
                try:
                    from CONFIG.config_loader import get_cfg
                    policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                except Exception:
                    pass
                
                if policy == "strict":
                    raise RuntimeError(
                        f"🚨 SPLIT-BRAIN DETECTED (POST_GATEKEEPER): "
                        f"budget.max={budget_max:.1f}m vs actual_max_from_features={actual_max_from_features:.1f}m. "
                        f"This indicates different code paths are computing different lookbacks. "
                        f"Training blocked until this is fixed."
                    )
            
            # Use actual max from features for the sanity check (the truth)
            if actual_max_from_features > lookback_budget_cap_for_budget:
                # CRITICAL: In strict mode, this is a hard-stop (gatekeeper should have caught this)
                policy = "strict"
                try:
                    from CONFIG.config_loader import get_cfg
                    policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                except Exception:
                    pass
                
                error_msg = (
                    f"🚨 POST_GATEKEEPER sanity check FAILED: actual_max_from_features={actual_max_from_features:.1f}m > cap={lookback_budget_cap_for_budget:.1f}m. "
                    f"Gatekeeper should have dropped features exceeding cap."
                )
                
                if policy == "strict":
                    raise RuntimeError(error_msg + " (policy: strict - training blocked)")
                else:
                    logger.error(error_msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
            else:
                logger.info(
                    f"✅ POST_GATEKEEPER sanity check PASSED: actual_max_from_features={actual_max_from_features:.1f}m <= cap={lookback_budget_cap_for_budget:.1f}m"
                )
        else:
            logger.debug(f"📊 POST_GATEKEEPER max_lookback: {budget.max_feature_lookback_minutes:.1f}m (no cap set)")
        
        # Validate fingerprint consistency (invariant check)
        if computed_fp != post_gatekeeper_fp:
            logger.error(
                f"🚨 FINGERPRINT MISMATCH (POST_GATEKEEPER): compute_budget={computed_fp} != "
                f"post_gatekeeper={post_gatekeeper_fp}. "
                f"This indicates lookback computed on different feature set than enforcement."
            )
        
        # Store for validation in train_and_evaluate_models
        gatekeeper_output_fingerprint = post_gatekeeper_fp
        
        # Update resolved_config with the new lookback (from features that actually remain)
        resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
        if log_cfg.cv_detail:
            logger.info(
                f"📊 Updated feature_lookback_max after Final Gatekeeper: {budget.max_feature_lookback_minutes:.1f}m "
                f"(from {len(feature_names)} remaining features, fingerprint={computed_fp}, stage=POST_GATEKEEPER)"
            )
        
            # CRITICAL: Enforce leakage policy (strict/drop_features/warn)
            # Design: purge covers feature lookback, embargo covers target horizon
            # Validate TWO separate constraints (not a single combined requirement)
            if resolved_config.purge_minutes is not None:
                purge_minutes = resolved_config.purge_minutes
                embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else purge_minutes
                
                # Load policy and over_budget_action from config
                policy = "strict"  # Default: strict
                over_budget_action = "drop"  # Default: drop (for gatekeeper behavior)
                buffer_minutes = 5.0  # Default buffer
                try:
                    from CONFIG.config_loader import get_cfg
                    policy = get_cfg("safety.leakage_detection.policy", default="strict", config_name="safety_config")
                    over_budget_action = get_cfg("safety.leakage_detection.over_budget_action", default="drop", config_name="safety_config")
                    buffer_minutes = float(get_cfg("safety.leakage_detection.lookback_buffer_minutes", default=5.0, config_name="safety_config"))
                except Exception:
                    pass
                
                # Constraint 1: purge must cover feature lookback
                purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                purge_violation = purge_minutes < purge_required
                
                # Constraint 2: embargo must cover target horizon
                # Guard: horizon_minutes may be None (e.g., for some target types)
                if budget.horizon_minutes is not None:
                    embargo_required = budget.horizon_minutes + buffer_minutes
                    embargo_violation = embargo_minutes < embargo_required
                else:
                    # If horizon is None, skip embargo validation (not applicable)
                    embargo_violation = False
                    embargo_required = None
                
                if purge_violation or embargo_violation:
                    # Build detailed violation message
                    violations = []
                    if purge_violation:
                        violations.append(
                            f"purge ({purge_minutes:.1f}m) < lookback_requirement ({purge_required:.1f}m) "
                            f"[max_lookback={budget.max_feature_lookback_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                        )
                    if embargo_violation:
                        violations.append(
                            f"embargo ({embargo_minutes:.1f}m) < horizon_requirement ({embargo_required:.1f}m) "
                            f"[horizon={budget.horizon_minutes:.1f}m + buffer={buffer_minutes:.1f}m]"
                        )
                    
                    msg = f"🚨 LEAKAGE VIOLATION: {'; '.join(violations)}"
                
                if policy == "strict":
                    # Hard-stop: raise exception
                    raise RuntimeError(msg + " (policy: strict - training blocked)")
                elif policy == "drop_features":
                    # Drop features that cause violation, recompute budget
                    logger.warning(msg + " (policy: drop_features - dropping violating features)")
                    # Find features with lookback > (purge - buffer)
                    # Note: purge covers lookback, not lookback+horizon
                    max_allowed_lookback = purge_minutes - buffer_minutes
                    violating_features = []
                    for feat_name in feature_names:
                        spec_lookback = None
                        if registry is not None:
                            try:
                                metadata = registry.get_feature_metadata(feat_name)
                                lag_bars = metadata.get('lag_bars')
                                if lag_bars is not None and lag_bars >= 0:
                                    spec_lookback = float(lag_bars * detected_interval)
                            except Exception:
                                pass
                        
                        from TRAINING.utils.leakage_budget import infer_lookback_minutes
                        lookback = infer_lookback_minutes(
                            feat_name,
                            detected_interval,
                            spec_lookback_minutes=spec_lookback,
                            registry=registry
                        )
                        
                        if lookback > max_allowed_lookback:
                            violating_features.append(feat_name)
                    
                    # Drop violating features
                    if violating_features:
                        logger.warning(f"   Dropping {len(violating_features)} features with lookback > {max_allowed_lookback:.1f}m")
                        logger.info(f"   Policy: drop_features (auto-drop violating features)")
                        logger.info(f"   Drop list ({len(violating_features)} features): {', '.join(violating_features[:10])}")
                        if len(violating_features) > 10:
                            logger.info(f"   ... and {len(violating_features) - 10} more")
                        keep_indices = [i for i, name in enumerate(feature_names) if name not in violating_features]
                        X = X[:, keep_indices]
                        feature_names = [name for i, name in enumerate(feature_names) if i in keep_indices]
                        
                        # Recompute budget on remaining features
                        from TRAINING.utils.cross_sectional_data import _compute_feature_fingerprint
                        after_drop_fp, after_drop_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
                        budget, budget_fp, budget_order_fp = compute_budget(
                            feature_names,
                            detected_interval,
                            resolved_config.horizon_minutes if resolved_config else 60.0,
                            registry=registry,
                            expected_fingerprint=after_drop_fp,
                            stage="after_policy_drop"
                        )
                        
                        # Validate fingerprint
                        if budget_fp != after_drop_fp:
                            logger.error(
                                f"🚨 FINGERPRINT MISMATCH (after_drop): budget={budget_fp} != expected={after_drop_fp}"
                            )
                        resolved_config.feature_lookback_max_minutes = budget.max_feature_lookback_minutes
                        
                        # Verify violation is resolved (check both constraints)
                        buffer_minutes = 5.0
                        purge_required = budget.max_feature_lookback_minutes + buffer_minutes
                        embargo_minutes = resolved_config.embargo_minutes if resolved_config.embargo_minutes is not None else resolved_config.purge_minutes
                        
                        # Guard: horizon_minutes may be None (e.g., for some target types)
                        if budget.horizon_minutes is not None:
                            embargo_required = budget.horizon_minutes + buffer_minutes
                            embargo_violation = embargo_minutes < embargo_required
                        else:
                            # If horizon is None, skip embargo validation (not applicable)
                            embargo_violation = False
                            embargo_required = None
                        
                        if resolved_config.purge_minutes < purge_required or embargo_violation:
                            violations = []
                            if resolved_config.purge_minutes < purge_required:
                                violations.append(f"purge ({resolved_config.purge_minutes:.1f}m) < {purge_required:.1f}m")
                            if embargo_violation:
                                violations.append(f"embargo ({embargo_minutes:.1f}m) < {embargo_required:.1f}m")
                            raise RuntimeError(
                                f"🚨 LEAKAGE VIOLATION PERSISTS after dropping features: {'; '.join(violations)}"
                            )
                        logger.info(
                            f"   ✅ Violation resolved: "
                            f"purge ({resolved_config.purge_minutes:.1f}m) >= {purge_required:.1f}m, "
                            f"embargo ({embargo_minutes:.1f}m) >= {embargo_required:.1f}m"
                        )
                else:  # policy == "warn"
                    # Log warning but continue (NOT recommended)
                    logger.error(msg + " (policy: warn - continuing with violation - NOT RECOMMENDED)")
    
    if X.shape[1] == 0:
        logger.error("❌ FINAL GATEKEEPER: All features were dropped! Cannot train models.")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )

    # Train and evaluate on cross-sectional data (single evaluation, not per-symbol)
    all_model_scores = []
    all_importances = []
    all_suspicious_features = {}
    fold_timestamps = None  # Initialize fold_timestamps for later use
    
    try:
        # Use detected_interval from outer scope (already computed above)
        # No need to recompute here
        
        # CRITICAL: Validate fingerprint consistency before training
        # The feature_names passed to train_and_evaluate_models should match post_gatekeeper fingerprint
        # NOTE: Pruning happens INSIDE train_and_evaluate_models, so MODEL_TRAIN_INPUT will be POST_PRUNE
        if 'gatekeeper_output_fingerprint' in locals():
            from TRAINING.utils.cross_sectional_data import _compute_feature_fingerprint
            current_fp, current_order_fp = _compute_feature_fingerprint(feature_names, set_invariant=True)
            if current_fp != gatekeeper_output_fingerprint:
                logger.warning(
                    f"⚠️  FINGERPRINT CHANGE (pre_training): POST_GATEKEEPER={gatekeeper_output_fingerprint} -> "
                    f"pre_training={current_fp}. "
                    f"This is expected if features were modified between gatekeeper and train_and_evaluate_models."
                )
        
        # NEW: Initialize dropped features tracker for telemetry (EARLY - before any filtering)
        # This must happen BEFORE filter_features_for_target so sanitizer can track drops
        from TRAINING.utils.dropped_features_tracker import DroppedFeaturesTracker
        dropped_tracker = DroppedFeaturesTracker()
        
        # NEW: Track early filter drops (schema/pattern/registry) - capture before filter_features_for_target
        all_columns_before_filter = columns_after_target_exclusions.copy() if 'columns_after_target_exclusions' in locals() else []
        
        result = train_and_evaluate_models(
            X, y, feature_names, task_type, model_families, multi_model_config,
            target_column=target_column,
            data_interval_minutes=detected_interval,  # Auto-detected or default
            time_vals=time_vals,  # Pass timestamps for fold tracking
            explicit_interval=explicit_interval,  # Pass explicit interval for consistency
            experiment_config=experiment_config,  # Pass experiment config
            output_dir=output_dir,  # Pass output directory for stability snapshots
            resolved_config=resolved_config,  # Pass resolved config with correct purge/embargo (post-pruning)
            dropped_tracker=dropped_tracker,  # Pass tracker for telemetry
            view=view,  # Pass view for REPRODUCIBILITY structure
            symbol=symbol  # Pass symbol for SYMBOL_SPECIFIC view
        )
        
        if result is None or len(result) != 7:
            logger.warning(f"train_and_evaluate_models returned unexpected value: {result}")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        model_metrics, primary_scores, importance, suspicious_features, feature_importances, fold_timestamps, _perfect_correlation_models = result
        
        # CRITICAL: Extract actual pruned feature count from feature_importances
        # feature_importances contains the features that were actually used (after pruning)
        actual_pruned_feature_count = 0
        if feature_importances:
            # Get feature count from first model's importances (all models use same features after pruning)
            first_model_importances = next(iter(feature_importances.values()))
            if isinstance(first_model_importances, dict):
                actual_pruned_feature_count = len(first_model_importances)
            elif isinstance(first_model_importances, (list, np.ndarray)):
                actual_pruned_feature_count = len(first_model_importances)
        # Fallback to len(feature_names) if we can't extract from importances
        if actual_pruned_feature_count == 0:
            actual_pruned_feature_count = len(feature_names) if feature_names else 0
        
        # NOTE: _perfect_correlation_models is now only for tracking/debugging.
        # High training accuracy alone is NOT a reliable leakage signal (especially for tree models),
        # so we no longer mark targets as LEAKAGE_DETECTED based on this.
        # Real leakage defense: schema filters + pre-training scan + time-purged CV.
        if _perfect_correlation_models:
            logger.debug(
                f"  Models with high training accuracy (may be overfitting): {_perfect_correlation_models}. "
                f"Check CV metrics to assess real predictive power."
            )
        
        # Save aggregated feature importances (respect view: CROSS_SECTIONAL vs SYMBOL_SPECIFIC)
        if feature_importances and output_dir:
            # Use view parameter if available, otherwise default to CROSS_SECTIONAL
            view_for_importances = view if 'view' in locals() else "CROSS_SECTIONAL"
            symbol_for_importances = symbol if ('symbol' in locals() and symbol) else view_for_importances
            _save_feature_importances(target_column, symbol_for_importances, feature_importances, output_dir, view=view_for_importances)
        
        # Store suspicious features
        if suspicious_features:
            all_suspicious_features = suspicious_features
            symbol_for_log = symbol if ('symbol' in locals() and symbol) else (view if 'view' in locals() else "CROSS_SECTIONAL")
            _log_suspicious_features(target_column, symbol_for_log, suspicious_features)
        
        # AUTO-FIX LEAKAGE: If leakage detected, automatically fix and re-run
        # Initialize autofix_info to None (will be set if auto-fixer runs)
        autofix_info = None
        
        # Load thresholds from config (with sensible defaults)
        if _CONFIG_AVAILABLE:
            try:
                safety_cfg = get_safety_config()
                # safety_config.yaml has a top-level 'safety' key
                safety_section = safety_cfg.get('safety', {})
                leakage_cfg = safety_section.get('leakage_detection', {})
                auto_fix_cfg = leakage_cfg.get('auto_fix_thresholds', {})
                cv_threshold = float(auto_fix_cfg.get('cv_score', 0.99))
                accuracy_threshold = float(auto_fix_cfg.get('training_accuracy', 0.999))
                r2_threshold = float(auto_fix_cfg.get('training_r2', 0.999))
                correlation_threshold = float(auto_fix_cfg.get('perfect_correlation', 0.999))
                auto_fix_enabled = leakage_cfg.get('auto_fix_enabled', True)
                auto_fix_min_confidence = float(leakage_cfg.get('auto_fix_min_confidence', 0.8))
                auto_fix_max_features = int(leakage_cfg.get('auto_fix_max_features_per_run', 20))
            except Exception as e:
                logger.debug(f"Failed to load leakage detection config: {e}, using defaults")
                cv_threshold = 0.99  # FALLBACK_DEFAULT_OK
                accuracy_threshold = 0.999  # FALLBACK_DEFAULT_OK
                r2_threshold = 0.999  # FALLBACK_DEFAULT_OK
                correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
                auto_fix_enabled = True  # FALLBACK_DEFAULT_OK
                auto_fix_min_confidence = 0.8  # FALLBACK_DEFAULT_OK
                auto_fix_max_features = 20  # FALLBACK_DEFAULT_OK
        else:
            # FALLBACK_DEFAULT_OK: Fallback defaults (config not available)
            cv_threshold = 0.99  # FALLBACK_DEFAULT_OK
            accuracy_threshold = 0.999  # FALLBACK_DEFAULT_OK
            r2_threshold = 0.999  # FALLBACK_DEFAULT_OK
            correlation_threshold = 0.999  # FALLBACK_DEFAULT_OK
            auto_fix_enabled = True  # FALLBACK_DEFAULT_OK
            auto_fix_min_confidence = 0.8
            auto_fix_max_features = 20  # FALLBACK_DEFAULT_OK
        
        # Check if auto-fixer is enabled
        if not auto_fix_enabled:
            logger.debug("Auto-fixer is disabled in config")
            should_auto_fix = False
        else:
            should_auto_fix = False
            
            # Check 1: Perfect CV scores (cross-validation)
            # CRITICAL: Use actual CV scores from model_scores (primary_scores), not model_metrics
            # model_metrics may contain training scores, but model_scores contains CV scores
            max_cv_score = None
            if primary_scores:
                # primary_scores contains CV scores from cross_val_score
                valid_cv_scores = [s for s in primary_scores.values() if s is not None and not np.isnan(s)]
                if valid_cv_scores:
                    max_cv_score = max(valid_cv_scores)
            
            # Fallback: try to extract from model_metrics if primary_scores unavailable
            # But be careful - model_metrics['accuracy'] etc. should now contain CV scores after our fix above
            if max_cv_score is None and model_metrics:
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict):
                        # Get CV score (should be CV after our fix, but double-check it's not training_accuracy)
                        cv_score_val = metrics.get('roc_auc') or metrics.get('r2') or metrics.get('accuracy')
                        # Exclude training scores explicitly
                        if cv_score_val is not None and not np.isnan(cv_score_val):
                            # Skip if this looks like a training score (training_accuracy exists and matches)
                            if 'training_accuracy' in metrics and abs(cv_score_val - metrics['training_accuracy']) < 0.001:
                                continue  # This is likely a training score, skip it
                            if max_cv_score is None or cv_score_val > max_cv_score:
                                max_cv_score = cv_score_val
            
            if max_cv_score is not None and max_cv_score >= cv_threshold:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect CV scores detected (max_cv={max_cv_score:.4f} >= {cv_threshold:.1%}) - enabling auto-fix mode")
            
            # Check 2: Perfect in-sample training accuracy with suspicion score gating
            # Use suspicion score to distinguish overfit noise from real leakage
            if not should_auto_fix and model_metrics:
                logger.debug(f"Checking model_metrics for perfect scores: {list(model_metrics.keys())}")
                
                # Compute suspicion score for each model with perfect train accuracy
                for model_name, metrics in model_metrics.items():
                    if isinstance(metrics, dict):
                        logger.debug(f"  {model_name} metrics: {list(metrics.keys())}")
                        
                        # Get train and CV scores
                        train_acc = metrics.get('training_accuracy')
                        cv_acc = metrics.get('accuracy')  # CV accuracy
                        train_r2 = metrics.get('training_r2')
                        cv_r2 = metrics.get('r2')  # CV R²
                        
                        # Check classification
                        if train_acc is not None and train_acc >= accuracy_threshold:
                            logger.debug(f"    {model_name} training_accuracy: {train_acc:.4f}")
                            
                            # Compute suspicion score
                            suspicion = _compute_suspicion_score(
                                train_score=train_acc,
                                cv_score=cv_acc,
                                feature_importances=feature_importances.get(model_name, {}) if feature_importances else {},
                                task_type='classification'
                            )
                            
                            # Only auto-fix if suspicion score crosses threshold
                            suspicion_threshold = 0.5  # Load from config if available
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training accuracy in {model_name} "
                                            f"(train={train_acc:.1%}, cv={cv_acc_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                # Overfit noise - log once at INFO level
                                cv_acc_str = f"{cv_acc:.3f}" if cv_acc is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_acc:.1%}, "
                                         f"cv={cv_acc_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_acc is not None and cv_acc >= accuracy_threshold:
                            # CV accuracy alone is suspicious
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV accuracy detected in {model_name} "
                                        f"({cv_acc:.1%} >= {accuracy_threshold:.1%}) - enabling auto-fix mode")
                            break
                        
                        # Check regression
                        if train_r2 is not None and train_r2 >= r2_threshold:
                            logger.debug(f"    {model_name} training_r2 (correlation): {train_r2:.4f}")
                            
                            # Compute suspicion score
                            suspicion = _compute_suspicion_score(
                                train_score=train_r2,
                                cv_score=cv_r2,
                                feature_importances=feature_importances.get(model_name, {}) if feature_importances else {},
                                task_type='regression'
                            )
                            
                            suspicion_threshold = 0.5
                            if suspicion >= suspicion_threshold:
                                should_auto_fix = True
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.warning(f"🚨 Suspicious perfect training correlation in {model_name} "
                                            f"(train={train_r2:.4f}, cv={cv_r2_str}, "
                                            f"suspicion={suspicion:.2f}) - enabling auto-fix mode")
                                break
                            else:
                                cv_r2_str = f"{cv_r2:.4f}" if cv_r2 is not None else "N/A"
                                logger.info(f"⚠️  {model_name} memorized training data (train={train_r2:.4f}, "
                                         f"cv={cv_r2_str}, suspicion={suspicion:.2f}). "
                                         f"Ignoring; check CV metrics.")
                        
                        elif cv_r2 is not None and cv_r2 >= r2_threshold:
                            # CV R² alone is suspicious
                            should_auto_fix = True
                            logger.warning(f"🚨 Perfect CV R² detected in {model_name} "
                                        f"({cv_r2:.4f} >= {r2_threshold:.4f}) - enabling auto-fix mode")
                            break
            
            # Check 3: Models that triggered perfect correlation warnings (fallback check)
            # Note: _perfect_correlation_models is populated inside train_and_evaluate_models,
            # but we check model_metrics above which covers the same cases, so this is just a safety check
            if not should_auto_fix and _perfect_correlation_models:
                should_auto_fix = True
                logger.warning(f"🚨 Perfect correlation detected in models: {', '.join(_perfect_correlation_models)} (>= {correlation_threshold:.1%}) - enabling auto-fix mode")
        
        if should_auto_fix:
            try:
                from TRAINING.common.leakage_auto_fixer import LeakageAutoFixer
                
                logger.info("🔧 Auto-fixing detected leaks...")
                logger.info(f"   Initializing LeakageAutoFixer (backups disabled)...")
                # Backups are disabled by default - no backup directory will be created
                fixer = LeakageAutoFixer(backup_configs=False, output_dir=output_dir)
                
                # Convert X to DataFrame if needed (auto-fixer expects DataFrame)
                if not isinstance(X, pd.DataFrame):
                    X_df = pd.DataFrame(X, columns=feature_names)
                else:
                    X_df = X
                
                # Convert y to Series if needed
                if not isinstance(y, pd.Series):
                    y_series = pd.Series(y)
                else:
                    y_series = y
                
                # Aggregate feature importances across all models
                aggregated_importance = {}
                if feature_importances:
                    # Sort model names for deterministic order (ensures reproducible aggregations)
                    for model_name in sorted(feature_importances.keys()):
                        importances = feature_importances[model_name]
                        if isinstance(importances, dict):
                            for feat, imp in importances.items():
                                if feat not in aggregated_importance:
                                    aggregated_importance[feat] = []
                                aggregated_importance[feat].append(imp)
                
                # Average importance across models (sort features for deterministic order)
                avg_importance = {feat: np.mean(imps) for feat, imps in sorted(aggregated_importance.items())} if aggregated_importance else {}
                
                # Get actual training accuracy from model_metrics (not CV scores)
                # This is critical - we detected perfect training accuracy, so pass that value
                actual_train_score = None
                if model_metrics:
                    for model_name, metrics in model_metrics.items():
                        if isinstance(metrics, dict):
                            # For classification, prefer training_accuracy (in-sample), fall back to CV accuracy
                            if 'training_accuracy' in metrics and metrics['training_accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['training_accuracy']
                                logger.debug(f"Using training accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'accuracy' in metrics and metrics['accuracy'] >= accuracy_threshold:
                                actual_train_score = metrics['accuracy']
                                logger.debug(f"Using CV accuracy {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            # For regression, prefer training_r2 (in-sample correlation), fall back to CV R²
                            elif 'training_r2' in metrics and metrics['training_r2'] >= r2_threshold:
                                actual_train_score = metrics['training_r2']
                                logger.debug(f"Using training correlation {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                            elif 'r2' in metrics and metrics['r2'] >= r2_threshold:
                                actual_train_score = metrics['r2']
                                logger.debug(f"Using CV R² {actual_train_score:.4f} from {model_name} for auto-fixer")
                                break
                
                # Fallback to CV score if no perfect training score found
                # CRITICAL: Use the same max_cv_score we computed above for consistency
                if actual_train_score is None:
                    if max_cv_score is not None:
                        actual_train_score = max_cv_score
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from model_metrics)")
                    else:
                        actual_train_score = max(primary_scores.values()) if primary_scores else None
                        logger.debug(f"Using CV score {actual_train_score:.4f} as fallback for auto-fixer (from primary_scores)")
                
                # Log what we're passing to auto-fixer (enhanced visibility)
                # CRITICAL: Clarify which feature set is being used for scanning vs training
                train_feature_set_size = len(feature_names)  # Features used for training (after pruning)
                scan_feature_set_size = len(safe_columns) if 'safe_columns' in locals() else len(feature_names)  # Features available for scanning
                scan_scope = "full_safe" if scan_feature_set_size > train_feature_set_size else "trained_only"
                
                train_score_str = f"{actual_train_score:.4f}" if actual_train_score is not None else "None"
                logger.info(f"🔧 Auto-fixer inputs: train_score={train_score_str}, "
                           f"train_feature_set_size={train_feature_set_size}, "
                           f"scan_feature_set_size={scan_feature_set_size}, "
                           f"scan_scope={scan_scope}, "
                           f"model_importance keys={len(avg_importance)}")
                if avg_importance:
                    top_5 = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                    logger.debug(f"   Top 5 features by importance: {', '.join([f'{f}={imp:.4f}' for f, imp in top_5])}")
                else:
                    logger.warning(f"   ⚠️  No aggregated importance available! feature_importances keys: {list(feature_importances.keys()) if feature_importances else 'None'}")
                
                # Detect leaks
                detections = fixer.detect_leaking_features(
                    X=X_df, y=y_series, feature_names=feature_names,
                    target_column=target_column,
                    symbols=pd.Series(symbols_array) if symbols_array is not None else None,
                    task_type='classification' if task_type == TaskType.BINARY_CLASSIFICATION or task_type == TaskType.MULTICLASS_CLASSIFICATION else 'regression',
                    data_interval_minutes=detected_interval,
                    model_importance=avg_importance if avg_importance else None,
                    train_score=actual_train_score,
                    test_score=None  # CV scores are already validation scores
                )
                
                if detections:
                    logger.warning(f"🔧 Auto-detected {len(detections)} leaking features")
                    # Apply fixes (with high confidence threshold to avoid false positives)
                    updates, autofix_info = fixer.apply_fixes(
                        detections, 
                        min_confidence=auto_fix_min_confidence, 
                        max_features=auto_fix_max_features,
                        dry_run=False,
                        target_name=target_name
                    )
                    if autofix_info.modified_configs:
                        logger.info(f"✅ Auto-fixed leaks. Configs updated.")
                        logger.info(f"   Updated: {len(updates.get('excluded_features_updates', {}).get('exact_patterns', []))} exact patterns, "
                                  f"{len(updates.get('excluded_features_updates', {}).get('prefix_patterns', []))} prefix patterns")
                        logger.info(f"   Rejected: {len(updates.get('feature_registry_updates', {}).get('rejected_features', []))} features in registry")
                    else:
                        logger.warning("⚠️  Auto-fix detected leaks but no configs were modified")
                        logger.warning("   This usually means all detections were below confidence threshold")
                        logger.warning(f"   Check logs above for confidence distribution details")
                    # Log backup info if available
                    if autofix_info.backup_files:
                        logger.info(f"📦 Backup created: {len(autofix_info.backup_files)} backup file(s)")
                else:
                    logger.info("🔍 Auto-fix detected no leaks (may need manual review)")
                    # Still create backup even when no leaks detected (to preserve state history)
                    # This ensures we have a backup whenever auto-fix mode is triggered
                    # But only if backup_configs is enabled
                    backup_files = []
                    if fixer.backup_configs:
                        try:
                            backup_files = fixer._backup_configs(
                                target_name=target_name,
                                max_backups_per_target=None  # Use instance config
                            )
                            if backup_files:
                                logger.info(f"📦 Backup created (no leaks detected): {len(backup_files)} backup file(s)")
                        except Exception as backup_error:
                            logger.warning(f"Failed to create backup when no leaks detected: {backup_error}")
            except Exception as e:
                logger.warning(f"Auto-fix failed: {e}", exc_info=True)
        
        # Ensure primary_scores is a dict
        if primary_scores is None:
            logger.warning(f"primary_scores is None, skipping")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        if not isinstance(primary_scores, dict):
            logger.warning(f"primary_scores is not a dict (got {type(primary_scores)}), skipping")
            return TargetPredictabilityScore(
                target_name=target_name,
                target_column=target_column,
                task_type=task_type,
                mean_score=-999.0,
                std_score=1.0,
                mean_importance=0.0,
                consistency=0.0,
                n_models=0,
                model_scores={}
            )
        
        all_model_scores.append(primary_scores)
        all_importances.append(importance)
        
        scores_str = ", ".join([f"{k}={v:.3f}" for k, v in primary_scores.items()])
        logger.info(f"Scores: {scores_str}, importance={importance:.2f}")
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        tb_str = traceback.format_exc()
        logger.warning(f"Failed: {error_msg}")
        logger.error(f"Full traceback:\n{tb_str}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=task_type,
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    if not all_model_scores:
        logger.warning(f"No successful evaluations for {target_name} (skipping)")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default, will be updated if target succeeds
            mean_score=-999.0,  # Flag for degenerate/failed targets
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={}
        )
    
    # Aggregate across models (skip NaN scores)
    # Note: With cross-sectional data, we only have one evaluation, not per-symbol
    all_scores_by_model = defaultdict(list)
    all_fold_scores = []  # Collect all fold scores across all models for distributional analysis
    for scores_dict in all_model_scores:
        # Defensive check: skip None or non-dict entries
        if scores_dict is None or not isinstance(scores_dict, dict):
            logger.warning(f"Skipping invalid scores_dict: {type(scores_dict)}")
            continue
        for model_name, score in scores_dict.items():
            if not (np.isnan(score) if isinstance(score, (float, np.floating)) else False):
                all_scores_by_model[model_name].append(score)
                # If score is from a single fold (not aggregated), add to fold_scores
                # Note: We'll collect actual fold scores separately if available
    
    # Calculate statistics (only from models that succeeded)
    model_means = {model: np.mean(scores) for model, scores in all_scores_by_model.items() if scores}
    if not model_means:
        logger.warning(f"No successful model evaluations for {target_name}")
        return TargetPredictabilityScore(
            target_name=target_name,
            target_column=target_column,
            task_type=TaskType.REGRESSION,  # Default
            mean_score=-999.0,
            std_score=1.0,
            mean_importance=0.0,
            consistency=0.0,
            n_models=0,
            model_scores={},
            leakage_flag="OK",
            suspicious_features=None
        )
    
    mean_score = np.mean(list(model_means.values()))
    std_score = np.std(list(model_means.values())) if len(model_means) > 1 else 0.0
    mean_importance = np.mean(all_importances)
    consistency = 1.0 - (std_score / (abs(mean_score) + 1e-6))
    
    # Determine task type (already inferred from data above)
    final_task_type = task_type
    
    # Get metric name for logging
    if final_task_type == TaskType.REGRESSION:
        metric_name = "R²"
    elif final_task_type == TaskType.BINARY_CLASSIFICATION:
        metric_name = "ROC-AUC"
    else:  # MULTICLASS_CLASSIFICATION
        metric_name = "Accuracy"
    
    # Composite score (normalize scores appropriately)
    composite, composite_def, composite_ver = calculate_composite_score(
        mean_score, std_score, mean_importance, len(all_scores_by_model), final_task_type
    )
    
    # Detect potential leakage (use task-appropriate thresholds)
    leakage_flag = detect_leakage(mean_score, composite, mean_importance, 
                                  target_name=target_name, model_scores=model_means, task_type=final_task_type)
    
    # Build detailed leakage flags for auto-rerun logic
    leakage_flags = {
        "perfect_train_acc": len(_perfect_correlation_models) > 0,  # Any model hit 100% training accuracy
        "high_auc": mean_score > 0.95 if final_task_type == TaskType.BINARY_CLASSIFICATION else False,
        "high_r2": mean_score > 0.80 if final_task_type == TaskType.REGRESSION else False,
        "suspicious_flag": leakage_flag != "OK"
    }
    
    # CRITICAL: Build LeakageAssessment to prevent contradictory reason strings
    from TRAINING.utils.leakage_assessment import LeakageAssessment
    
    # Determine CV suspicious flag (CV score too high suggests leakage, not just overfitting)
    cv_suspicious = False
    if primary_scores:
        valid_cv_scores = [s for s in primary_scores.values() if s is not None and not np.isnan(s)]
        if valid_cv_scores:
            max_cv_score = max(valid_cv_scores)
            # CV score >= 0.85 is suspicious (too good to be true)
            cv_suspicious = max_cv_score >= 0.85
    
    # Determine overfit_likely flag (perfect train but low CV = classic overfitting)
    overfit_likely = False
    if model_metrics:
        for model_name, metrics in model_metrics.items():
            if isinstance(metrics, dict):
                train_acc = metrics.get('training_accuracy')
                cv_acc = metrics.get('accuracy')
                train_r2 = metrics.get('training_r2')
                cv_r2 = metrics.get('r2')
                
                # Check if perfect train but low CV (classic overfitting)
                if train_acc is not None and train_acc >= 0.99:
                    if cv_acc is not None and cv_acc < 0.75:
                        overfit_likely = True
                        break
                if train_r2 is not None and train_r2 >= 0.99:
                    if cv_r2 is not None and cv_r2 < 0.50:
                        overfit_likely = True
                        break
    
    # Find models with AUC > 0.90
    auc_too_high_models = []
    if final_task_type == TaskType.BINARY_CLASSIFICATION and model_means:
        for model_name, score in model_means.items():
            if score is not None and not np.isnan(score) and score > 0.90:
                auc_too_high_models.append(model_name)
    
    # Build assessment
    assessment = LeakageAssessment(
        leak_scan_pass=not summary_leaky_features if 'summary_leaky_features' in locals() else True,
        cv_suspicious=cv_suspicious,
        overfit_likely=overfit_likely,
        auc_too_high_models=auc_too_high_models
    )
    
    # Determine status: SUSPICIOUS targets should be excluded from rankings
    # High AUC/R² after auto-fix suggests structural leakage (target construction issue)
    if leakage_flag in ["SUSPICIOUS", "HIGH_SCORE"]:
        # If we have very high scores, this is likely structural leakage, not just feature leakage
        if final_task_type == TaskType.BINARY_CLASSIFICATION and mean_score > 0.95:
            final_status = "SUSPICIOUS_STRONG"
        elif final_task_type == TaskType.REGRESSION and mean_score > 0.80:
            final_status = "SUSPICIOUS_STRONG"
        else:
            final_status = "SUSPICIOUS"
    else:
        final_status = "OK"
    
    # Collect fold scores if available (from model evaluations)
    # Note: This is a simplified collection - actual per-fold scores would require
    # storing them during cross_val_score calls, which can be enhanced later
    aggregated_fold_scores = None
    if all_fold_scores and len(all_fold_scores) > 0:
        aggregated_fold_scores = [float(s) for s in all_fold_scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
        if len(aggregated_fold_scores) == 0:
            aggregated_fold_scores = None
    
    result = TargetPredictabilityScore(
        target_name=target_name,
        target_column=target_column,
        task_type=final_task_type,
        mean_score=mean_score,
        std_score=std_score,
        mean_importance=mean_importance,
        consistency=consistency,
        n_models=len(all_scores_by_model),
        model_scores=model_means,
        composite_score=composite,
        composite_definition=composite_def,
        composite_version=composite_ver,
        leakage_flag=leakage_flag,
        suspicious_features=all_suspicious_features if all_suspicious_features else None,
        fold_timestamps=fold_timestamps,
        fold_scores=aggregated_fold_scores,
        leakage_flags=leakage_flags,
        autofix_info=autofix_info if 'autofix_info' in locals() else None,
        status=final_status,
        attempts=1
    )
    
    # Log canonical summary block (one block that can be screenshot for PR comments)
    # Use detected_interval from evaluate_target_predictability scope (defined at line ~2276)
    summary_interval = detected_interval if 'detected_interval' in locals() else None
    summary_horizon = target_horizon_minutes if 'target_horizon_minutes' in locals() else None
    summary_safe_features = len(safe_columns) if 'safe_columns' in locals() else 0
    summary_leaky_features = leaky_features if 'leaky_features' in locals() else []
    
    # Extract CV splitter info for logging
    splitter_name = None
    n_splits_val = None
    purge_minutes_val = None
    embargo_minutes_val = None
    max_lookback_val = None
    
    if 'resolved_config' in locals() and resolved_config:
        purge_minutes_val = resolved_config.purge_minutes
        embargo_minutes_val = resolved_config.embargo_minutes
        # CRITICAL: Use the FINAL lookback from resolved_config (should match POST_PRUNE recompute)
        # If there's a mismatch, the invariant check should have caught it
        max_lookback_val = resolved_config.feature_lookback_max_minutes
        splitter_name = "PurgedTimeSeriesSplit"  # Default for time-series CV
        n_splits_val = cv_folds if 'cv_folds' in locals() else None
        
        # SANITY CHECK: Verify resolved_config lookback matches what we computed at POST_PRUNE
        if 'computed_lookback' in locals() and computed_lookback is not None:
            if abs(max_lookback_val - computed_lookback) > 1.0:
                logger.error(
                    f"🚨 SUMMARY MISMATCH: resolved_config.feature_lookback_max_minutes={max_lookback_val:.1f}m "
                    f"but POST_PRUNE computed_lookback={computed_lookback:.1f}m. "
                    f"Using POST_PRUNE value for summary."
                )
                max_lookback_val = computed_lookback  # Use the correct value
        
        # Log lookback_budget_minutes cap status for auditability
        try:
            from CONFIG.config_loader import get_cfg
            budget_cap_raw = get_cfg("safety.leakage_detection.lookback_budget_minutes", default="auto", config_name="safety_config")
            if budget_cap_raw != "auto" and isinstance(budget_cap_raw, (int, float)):
                logger.info(f"📊 lookback_budget_minutes cap: {float(budget_cap_raw):.1f}m (active)")
            else:
                logger.info(f"📊 lookback_budget_minutes cap: auto (no cap, using actual max)")
        except Exception:
            pass
    
    _log_canonical_summary(
        target_name=target_name,
        target_column=target_column,
        symbols=symbols,
        time_vals=time_vals,
        interval=summary_interval,
        horizon=summary_horizon,
        rows=len(X) if X is not None else 0,
        features_safe=summary_safe_features,
        features_pruned=actual_pruned_feature_count if 'actual_pruned_feature_count' in locals() else (len(feature_names) if feature_names else 0),
        leak_scan_verdict="PASS" if not summary_leaky_features else "FAIL",
        auto_fix_verdict="SKIPPED" if not should_auto_fix else ("RAN" if autofix_info and autofix_info.modified_configs else "NO_CHANGES"),
        auto_fix_reason=assessment.auto_fix_reason() if 'assessment' in locals() else None,
        cv_metric=f"{metric_name}={mean_score:.3f}±{std_score:.3f}",
        composite=composite,
        leakage_flag=leakage_flag,
        cohort_path=None,  # Will be set by reproducibility tracker
        splitter_name=splitter_name,
        purge_minutes=purge_minutes_val,
        embargo_minutes=embargo_minutes_val,
        max_feature_lookback_minutes=max_lookback_val,
        n_splits=n_splits_val
    )
    
    # Legacy summary line (backward compatibility)
    leakage_indicator = f" [{leakage_flag}]" if leakage_flag != "OK" else ""
    logger.debug(f"Legacy summary: {metric_name}={mean_score:.3f}±{std_score:.3f}, "
               f"importance={mean_importance:.2f}, composite={composite:.3f}{leakage_indicator}")
    
    # Store suspicious features in result for summary report
    result.suspicious_features = all_suspicious_features if all_suspicious_features else None
    
    # Log top features actually used (for leakage diagnosis)
    # This helps identify if high-scoring models are using leaky features
    if 'feature_importances' in locals() and feature_importances and leakage_flag != "OK":
        logger.info("=" * 60)
        logger.info("TOP FEATURES USED (for leakage diagnosis)")
        logger.info("=" * 60)
        for model_name, importance_dict in feature_importances.items():
            if isinstance(importance_dict, dict) and importance_dict:
                # Sort by importance
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                top_20 = sorted_features[:20]
                logger.info(f"{model_name}: Top 20 features by importance:")
                for feat_name, importance in top_20:
                    logger.info(f"  {feat_name}: {importance:.4f}")
        logger.info("=" * 60)
    
    # Track reproducibility: compare to previous target ranking run
    # This runs regardless of which entry point calls this function
    if output_dir and result.mean_score != -999.0:
        try:
            from TRAINING.utils.reproducibility_tracker import ReproducibilityTracker
            
            # Use module-specific directory for reproducibility log
            # output_dir might be: output_dir_YYYYMMDD_HHMMSS/target_rankings/ or just output_dir_YYYYMMDD_HHMMSS
            # We want to store in target_rankings/ subdirectory for this module
            if output_dir.name == 'target_rankings':
                # Already in target_rankings subdirectory
                module_output_dir = output_dir
            elif (output_dir.parent / 'target_rankings').exists():
                # output_dir is parent, use target_rankings subdirectory
                module_output_dir = output_dir.parent / 'target_rankings'
            else:
                # Fallback: use output_dir directly (for standalone runs)
                module_output_dir = output_dir
            
            tracker = ReproducibilityTracker(
                output_dir=module_output_dir,
                search_previous_runs=True  # Search for previous runs in parent directories
            )
            
            # Automated audit-grade reproducibility tracking using RunContext
            try:
                from TRAINING.utils.run_context import RunContext
                
                # Build RunContext from available data
                # Prefer symbols_array (from prepare_cross_sectional_data_for_ranking) over symbols list
                symbols_for_ctx = None
                if 'cohort_context' in locals() and cohort_context:
                    symbols_for_ctx = cohort_context.get('symbols_array')
                    if symbols_for_ctx is None:
                        symbols_for_ctx = cohort_context.get('symbols')
                elif 'symbols_array' in locals():
                    symbols_for_ctx = symbols_array
                elif 'symbols' in locals():
                    symbols_for_ctx = symbols
                
                # Use resolved_config values if available (single source of truth)
                # CRITICAL: Use feature_lookback_max_minutes from resolved_config (computed from FINAL feature set)
                if 'resolved_config' in locals() and resolved_config:
                    purge_minutes_val = resolved_config.purge_minutes
                    embargo_minutes_val = resolved_config.embargo_minutes
                    # Use actual computed lookback from final features (post gatekeeper + pruning)
                    feature_lookback_max = resolved_config.feature_lookback_max_minutes
                elif 'purge_minutes_val' not in locals() or purge_minutes_val is None:
                    # Fallback: compute from purge_time if available
                    if 'purge_time' in locals() and purge_time is not None:
                        try:
                            if hasattr(purge_time, 'total_seconds'):
                                purge_minutes_val = purge_time.total_seconds() / 60.0
                                embargo_minutes_val = purge_minutes_val  # Assume same
                        except Exception:
                            pass
                
                # Fallback: if resolved_config not available, try to compute from final feature_names
                if 'feature_lookback_max' not in locals() or feature_lookback_max is None:
                    # Try to compute from final feature_names if available
                    if 'feature_names' in locals() and feature_names and 'data_interval_minutes' in locals() and data_interval_minutes:
                        from TRAINING.utils.leakage_budget import compute_budget
                        try:
                            # Get horizon for budget calculation
                            horizon = target_horizon_minutes if 'target_horizon_minutes' in locals() else 60.0
                            budget, _, _ = compute_budget(feature_names, data_interval_minutes, horizon, stage="run_context_budget")
                            feature_lookback_max = budget.max_feature_lookback_minutes
                        except Exception:
                            # Fallback: conservative estimate
                            feature_lookback_max = None
                    else:
                        feature_lookback_max = None
                
                # Get seed from config for reproducibility
                try:
                    from CONFIG.config_loader import get_cfg
                    seed_value = get_cfg("pipeline.determinism.base_seed", default=42)
                except Exception:
                    seed_value = 42
                
                # Build RunContext
                ctx = RunContext(
                    X=cohort_context.get('X') if 'cohort_context' in locals() and cohort_context else None,
                    y=cohort_context.get('y') if 'cohort_context' in locals() and cohort_context else None,
                    feature_names=feature_names if 'feature_names' in locals() else None,
                    symbols=symbols_for_ctx,
                    time_vals=cohort_context.get('time_vals') if 'cohort_context' in locals() and cohort_context else None,
                    target_column=target_column,
                    target_name=target_name,
                    min_cs=cohort_context.get('min_cs') if 'cohort_context' in locals() and cohort_context else (min_cs if 'min_cs' in locals() else None),
                    max_cs_samples=cohort_context.get('max_cs_samples') if 'cohort_context' in locals() and cohort_context else (max_cs_samples if 'max_cs_samples' in locals() else None),
                    mtf_data=cohort_context.get('mtf_data') if 'cohort_context' in locals() and cohort_context else None,
                    cv_method="purged_kfold",
                    cv_folds=cv_folds if 'cv_folds' in locals() else None,
                    horizon_minutes=target_horizon_minutes if 'target_horizon_minutes' in locals() else None,
                    purge_minutes=purge_minutes_val,
                    fold_timestamps=fold_timestamps if 'fold_timestamps' in locals() else None,
                    feature_lookback_max_minutes=feature_lookback_max,
                    data_interval_minutes=data_interval_minutes if 'data_interval_minutes' in locals() else None,
                    stage="target_ranking",
                    output_dir=output_dir,
                    seed=seed_value
                )
                # Add view and symbol to RunContext if available
                if 'view' in locals():
                    ctx.view = view
                if 'symbol' in locals() and symbol:
                    ctx.symbol = symbol
                
                # Build metrics dict with regression features
                # FIX: Remove redundancy - use n_features_post_prune (more descriptive) and drop features_final
                n_features_final = len(feature_names) if 'feature_names' in locals() and feature_names else None
                
                # Start with base metrics
                metrics_dict = {
                    "metric_name": metric_name,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "mean_importance": result.mean_importance,
                    "composite_score": result.composite_score,
                    "n_models": result.n_models,
                    "task_type": result.task_type.name if hasattr(result.task_type, 'name') else str(result.task_type),
                    # Regression features: feature counts
                    "n_features_pre": features_safe if 'features_safe' in locals() else None,
                    "n_features_post_prune": n_features_final,  # Final feature count after pruning
                    "features_safe": features_safe if 'features_safe' in locals() else None,  # Count of safe features before pruning
                }
                
                # Add composite score definition and version
                if result.composite_definition:
                    metrics_dict["composite_definition"] = result.composite_definition
                if result.composite_version:
                    metrics_dict["composite_version"] = result.composite_version
                
                # Add fold scores and distributional stats if available
                if result.fold_scores and len(result.fold_scores) > 0:
                    import numpy as np
                    valid_scores = [s for s in result.fold_scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
                    if valid_scores:
                        metrics_dict["fold_scores"] = [float(s) for s in valid_scores]
                        metrics_dict["min_score"] = float(np.min(valid_scores))
                        metrics_dict["max_score"] = float(np.max(valid_scores))
                        metrics_dict["median_score"] = float(np.median(valid_scores))
                
                # Add enhanced leakage info (use to_dict to get the structured format)
                result_dict = result.to_dict()
                if 'leakage' in result_dict:
                    metrics_dict['leakage'] = result_dict['leakage']
                # Also include legacy leakage_flag for backward compatibility
                metrics_dict['leakage_flag'] = result.leakage_flag
                
                # Add pos_rate if available (from y)
                if 'y' in locals() and y is not None:
                    try:
                        import numpy as np
                        if len(y) > 0:
                            pos_count = np.sum(y == 1) if hasattr(y, '__iter__') else 0
                            pos_rate = pos_count / len(y) if len(y) > 0 else None
                            if pos_rate is not None:
                                metrics_dict["pos_rate"] = float(pos_rate)
                    except Exception:
                        pass
                
                # Add view and symbol to RunContext if available (for dual-view target ranking)
                if 'view' in locals():
                    ctx.view = view
                if 'symbol' in locals() and symbol:
                    ctx.symbol = symbol
                
                # Use automated log_run API
                audit_result = tracker.log_run(ctx, metrics_dict)
                
                # Log audit report summary if available
                if audit_result.get("audit_report"):
                    audit_report = audit_result["audit_report"]
                    if audit_report.get("violations"):
                        logger.warning(f"🚨 Audit violations detected: {len(audit_report['violations'])}")
                        for violation in audit_report['violations']:
                            logger.warning(f"  - {violation['message']}")
                    if audit_report.get("warnings"):
                        logger.info(f"⚠️  Audit warnings: {len(audit_report['warnings'])}")
                        for warning in audit_report['warnings']:
                            logger.info(f"  - {warning['message']}")
                
                # Log trend summary if available (already logged by log_run, but include in result)
                if audit_result.get("trend_summary"):
                    trend = audit_result["trend_summary"]
                    # Trend summary is already logged by log_run, but we can add additional context here if needed
                    pass
                
            except ImportError:
                # Fallback to legacy API if RunContext not available
                logger.warning("RunContext not available, falling back to legacy reproducibility tracking")
                from TRAINING.utils.cohort_metadata_extractor import extract_cohort_metadata, format_for_reproducibility_tracker
                
                if 'cohort_context' in locals() and cohort_context:
                    symbols_for_extraction = cohort_context.get('symbols_array') or cohort_context.get('symbols')
                    cohort_metadata = extract_cohort_metadata(
                        X=cohort_context.get('X'),
                        symbols=symbols_for_extraction,
                        time_vals=cohort_context.get('time_vals'),
                        y=cohort_context.get('y'),
                        mtf_data=cohort_context.get('mtf_data'),
                        min_cs=cohort_context.get('min_cs'),
                        max_cs_samples=cohort_context.get('max_cs_samples'),
                        compute_data_fingerprint=True,
                        compute_per_symbol_stats=True
                    )
                else:
                    cohort_metadata = extract_cohort_metadata(
                        symbols=symbols if 'symbols' in locals() else None,
                        mtf_data=mtf_data if 'mtf_data' in locals() else None,
                        min_cs=min_cs if 'min_cs' in locals() else None,
                        max_cs_samples=max_cs_samples if 'max_cs_samples' in locals() else None
                    )
                
                cohort_metrics, cohort_additional_data = format_for_reproducibility_tracker(cohort_metadata)
                metrics_with_cohort = {
                    "metric_name": metric_name,
                    "mean_score": result.mean_score,
                    "std_score": result.std_score,
                    "mean_importance": result.mean_importance,
                    "composite_score": result.composite_score,
                    # Regression features: feature counts
                    "n_features_pre": features_safe if 'features_safe' in locals() else None,
                    "n_features_post_prune": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    "features_safe": features_safe if 'features_safe' in locals() else None,
                    "features_final": len(feature_names) if 'feature_names' in locals() and feature_names else None,
                    **cohort_metrics
                }
                
                # Add pos_rate if available (from y)
                if 'y' in locals() and y is not None:
                    try:
                        import numpy as np
                        if len(y) > 0:
                            pos_count = np.sum(y == 1) if hasattr(y, '__iter__') else 0
                            pos_rate = pos_count / len(y) if len(y) > 0 else None
                            if pos_rate is not None:
                                metrics_with_cohort["pos_rate"] = float(pos_rate)
                    except Exception:
                        pass
                
                # NOTE: NaN drops are now tracked immediately after data prep (above), not here
                
                # NEW: Add dropped features summary to additional_data for telemetry
                if 'dropped_tracker' in locals() and dropped_tracker is not None and not dropped_tracker.is_empty():
                    cohort_additional_data['dropped_features'] = dropped_tracker.get_summary()
                
                # Add resolved_data_config (mode, loader contract) to additional_data for telemetry
                if 'resolved_data_config' in locals() and resolved_data_config:
                    cohort_additional_data['resolved_data_mode'] = resolved_data_config.get('resolved_data_mode')
                    cohort_additional_data['mode_reason'] = resolved_data_config.get('mode_reason')
                    cohort_additional_data['loader_contract'] = resolved_data_config.get('loader_contract')
                
                additional_data_with_cohort = {
                    "n_models": result.n_models,
                    "leakage_flag": result.leakage_flag,
                    "task_type": result.task_type.name if hasattr(result.task_type, 'name') else str(result.task_type),
                    **cohort_additional_data
                }
                
                # Add view and symbol for dual-view target ranking
                if 'view' in locals():
                    additional_data_with_cohort['view'] = view
                if 'symbol' in locals() and symbol:
                    additional_data_with_cohort['symbol'] = symbol
                
                # Add CV details manually (legacy path)
                if 'target_horizon_minutes' in locals() and target_horizon_minutes is not None:
                    additional_data_with_cohort['horizon_minutes'] = target_horizon_minutes
                if 'purge_time' in locals() and purge_time is not None:
                    try:
                        if hasattr(purge_time, 'total_seconds'):
                            # Use purge_minutes_val if available (single source of truth)
                            if 'purge_minutes_val' in locals() and purge_minutes_val is not None:
                                additional_data_with_cohort['purge_minutes'] = purge_minutes_val
                                additional_data_with_cohort['embargo_minutes'] = purge_minutes_val
                            else:
                                purge_minutes_val = purge_time.total_seconds() / 60.0
                                additional_data_with_cohort['purge_minutes'] = purge_minutes_val
                                additional_data_with_cohort['embargo_minutes'] = purge_minutes_val
                    except Exception:
                        pass
                if 'cv_folds' in locals() and cv_folds is not None:
                    additional_data_with_cohort['cv_folds'] = cv_folds
                if 'fold_timestamps' in locals() and fold_timestamps:
                    additional_data_with_cohort['fold_timestamps'] = fold_timestamps
                if 'feature_names' in locals() and feature_names:
                    additional_data_with_cohort['feature_names'] = feature_names
                if 'data_interval_minutes' in locals() and data_interval_minutes is not None:
                    additional_data_with_cohort['data_interval_minutes'] = data_interval_minutes
                    max_lookback_bars = 288
                    additional_data_with_cohort['feature_lookback_max_minutes'] = max_lookback_bars * data_interval_minutes
                
                # FIX: Ensure view and symbol are included in additional_data for proper telemetry scoping
                # This aligns target ranking telemetry with feature selection telemetry
                # Features are compared per-target, per-view, per-symbol (not across all targets/views/symbols)
                if 'view' in locals() and view:
                    additional_data_with_cohort['view'] = view
                if 'symbol' in locals() and symbol:
                    additional_data_with_cohort['symbol'] = symbol
                
                # Add seed for reproducibility tracking
                try:
                    from CONFIG.config_loader import get_cfg
                    seed = get_cfg("pipeline.determinism.base_seed", default=42)
                    additional_data_with_cohort['seed'] = seed
                except Exception:
                    # Fallback to default if config not available
                    additional_data_with_cohort['seed'] = 42
                
                # FIX: For TARGET_RANKING, view is used as route_type (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO)
                # This ensures directory structure aligns: TARGET_RANKING/{view}/{target}/{symbol}/cohort={cohort_id}/
                route_type_for_target_ranking = view if 'view' in locals() and view else None
                
                tracker.log_comparison(
                    stage="target_ranking",
                    item_name=target_name,  # FIX: item_name is just target (view/symbol handled by route_type/symbol params)
                    metrics=metrics_with_cohort,
                    additional_data=additional_data_with_cohort,
                    route_type=route_type_for_target_ranking,  # FIX: Use view as route_type for TARGET_RANKING (ensures alignment with feature selection)
                    symbol=symbol if 'symbol' in locals() and symbol else None  # FIX: Properly scoped by symbol (for SYMBOL_SPECIFIC/LOSO views)
                )
        except Exception as e:
            logger.warning(f"Reproducibility tracking failed for {target_name}: {e}")
            import traceback
            logger.debug(f"Reproducibility tracking traceback: {traceback.format_exc()}")
    
    return result


