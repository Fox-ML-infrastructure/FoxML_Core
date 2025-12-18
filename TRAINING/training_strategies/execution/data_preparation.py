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

 # ---- PATH BOOTSTRAP: ensure project root on sys.path in parent AND children ----
import os, sys
from pathlib import Path

# CRITICAL: Set LD_LIBRARY_PATH for conda CUDA libraries BEFORE any imports
# This must happen before TensorFlow tries to load CUDA libraries
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib = os.path.join(conda_prefix, "lib")
    conda_targets_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = []
    if conda_lib not in current_ld_path:
        new_paths.append(conda_lib)
    if conda_targets_lib not in current_ld_path:
        new_paths.append(conda_targets_lib)
    if new_paths:
        updated_ld_path = ":".join(new_paths + [current_ld_path] if current_ld_path else new_paths)
        os.environ["LD_LIBRARY_PATH"] = updated_ld_path

# Show TensorFlow warnings so user knows if GPU isn't working
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Removed - show warnings
# os.environ.setdefault("TF_LOGGING_VERBOSITY", "ERROR")  # Removed - show warnings

# project root: TRAINING/training_strategies/*.py -> parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure Python can import `common`, `model_fun`, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Propagate to spawned processes (spawned interpreter reads PYTHONPATH at startup)
os.environ.setdefault("PYTHONPATH", str(_PROJECT_ROOT))

# Set up all paths using centralized utilities
# Note: setup_all_paths already adds CONFIG to sys.path
from TRAINING.common.utils.path_setup import setup_all_paths
_PROJECT_ROOT, _TRAINING_ROOT, _CONFIG_DIR = setup_all_paths(_PROJECT_ROOT)

# Import config loader (CONFIG is already in sys.path from setup_all_paths)
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    # Only log at debug level to avoid misleading warnings
    logging.getLogger(__name__).debug("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
from TRAINING.common.utils.process_cleanup import setup_loky_cleanup_from_config
setup_loky_cleanup_from_config()

"""
Enhanced Training Script with Multiple Strategies - Full Original Functionality

Replicates ALL functionality from train_mtf_cross_sectional_gpu.py but with:
- Modular architecture
- 3 training strategies (single-task, multi-task, cascade)
- All 20 model families from original script
- GPU acceleration
- Memory management
- Batch processing
- Cross-sectional training
- Target discovery
- Data validation
"""

# ANTI-DEADLOCK: Process-level safety (before importing TF/XGB/sklearn)
import time as _t
# Make thread pools predictable (also avoids weird deadlocks)


# Import the isolation runner (moved to TRAINING/common/isolation_runner.py)
# Paths are already set up above

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications - import from centralized constants
from TRAINING.common.family_constants import TF_FAMS, TORCH_FAMS, CPU_FAMS, TORCH_SEQ_FAMILIES

# Standard library imports
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any

# Third-party imports
import numpy as np
import pandas as pd
import polars as pl

# Import USE_POLARS - defined in utils.py, but we need it here
# Use environment variable directly to avoid circular import
import os
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"

# Import target router utilities from TRAINING root
from TRAINING.orchestration.routing.target_router import route_target
# safe_target_extraction may be in utils or defined elsewhere
try:
    from TRAINING.training_strategies.utils import safe_target_extraction
except ImportError:
    # Fallback definition
    def safe_target_extraction(df, target):
        if target in df.columns:
            return df[target], target
        # Try common variations
        for col in df.columns:
            if col.endswith(target) or target in col:
                return df[col], col
        raise ValueError(f"Target {target} not found in dataframe")

# Setup logger
logger = logging.getLogger(__name__)

"""Data preparation functions for training strategies."""

def prepare_training_data_cross_sectional(mtf_data: Dict[str, pd.DataFrame], 
                                       target: str, 
                                       feature_names: List[str] = None,
                                       min_cs: int = 10,
                                       max_cs_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Prepare cross-sectional training data with polars optimization for memory efficiency."""
    
    logger.info(f"🎯 Building cross-sectional training data for target: {target}")
    if max_cs_samples is None:
        # Load from config if available, otherwise use default
        if _CONFIG_AVAILABLE:
            max_cs_samples = get_cfg("pipeline.data_limits.max_cross_sectional_samples", default=1000)
            if max_cs_samples is None:
                max_cs_samples = 1000  # Config has null, use default
        else:
            max_cs_samples = 1000
        logger.info(f"📊 Using default aggressive sampling: max {max_cs_samples} samples per timestamp")
    else:
        logger.info(f"📊 Cross-sectional sampling: max {max_cs_samples} samples per timestamp")
    
    if USE_POLARS:
        return _prepare_training_data_polars(mtf_data, target, feature_names, min_cs, max_cs_samples)
    else:
        return _prepare_training_data_pandas(mtf_data, target, feature_names, min_cs, max_cs_samples)

def _prepare_training_data_polars(mtf_data: Dict[str, pd.DataFrame], 
                                 target: str, 
                                 feature_names: List[str] = None,
                                 min_cs: int = 10,
                                 max_cs_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Polars-based data preparation for memory efficiency with cross-sectional sampling."""
    
    # Initialize feature auditor for tracking feature drops
    from TRAINING.ranking.utils.feature_audit import FeatureAuditor
    auditor = FeatureAuditor(target=target)
    
    logger.info(f"🎯 Building cross-sectional training data (polars, memory-efficient) for target: {target}")
    
    # Harmonize schema across symbols to avoid width mismatches on concat
    import os
    align_cols = os.environ.get("CS_ALIGN_COLUMNS", "1") not in ("0", "false", "False")
    align_mode = os.environ.get("CS_ALIGN_MODE", "union").lower()
    ordered_schema = None
    if align_cols and mtf_data:
        first_df = next(iter(mtf_data.values()))
        if align_mode == "intersect":
            shared = None
            for _sym, _df in mtf_data.items():
                cols = list(_df.columns)
                shared = set(cols) if shared is None else (shared & set(cols))
            ordered_schema = [c for c in first_df.columns if c in (shared or set())]
            logger.info(f"🔧 [polars] Harmonized schema (intersect) with {len(ordered_schema)} columns")
        else:
            # union
            union = []
            seen = set()
            for c in first_df.columns:
                union.append(c); seen.add(c)
            for _sym, _df in mtf_data.items():
                for c in _df.columns:
                    if c not in seen:
                        union.append(c); seen.add(c)
            ordered_schema = union
            logger.info(f"🔧 [polars] Harmonized schema (union) with {len(ordered_schema)} columns")
    
    # Convert to polars for memory-efficient operations
    all_data_pl = []
    for symbol, df in mtf_data.items():
        if ordered_schema is not None:
            if align_mode == "intersect":
                df_use = df.loc[:, ordered_schema]
            else:
                df_use = df.reindex(columns=ordered_schema)
        else:
            df_use = df
        df_pl = pl.from_pandas(df_use)
        df_pl = df_pl.with_columns(pl.lit(symbol).alias("symbol"))
        all_data_pl.append(df_pl)
    
    # Combine using polars (memory efficient)
    combined_pl = pl.concat(all_data_pl)
    logger.info(f"Combined data shape (polars): {combined_pl.shape}")
    
    # Auto-discover features if not provided, then validate with registry
    if feature_names is None:
        all_cols = combined_pl.columns
        feature_names = [col for col in all_cols 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', 'timestamp', 'ts']]
    
    # Record requested features
    auditor.record_requested(feature_names)
    logger.info(f"📊 Feature audit [{target}]: {len(feature_names)} features requested")
    
    # Validate features with registry (if enabled)
    if feature_names:
        try:
            from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
            from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
            
            # Detect data interval for horizon conversion
            first_df = next(iter(mtf_data.values()))
            detected_interval = detect_interval_from_dataframe(first_df, timestamp_column='ts', default=5)
            # Ensure interval is valid (> 0)
            if detected_interval <= 0:
                detected_interval = 5
                logger.warning(f"  Invalid detected interval, using default: 5m")
            
            # Filter features using registry
            all_columns = list(combined_pl.columns)
            validated_features = filter_features_for_target(
                all_columns,
                target,
                verbose=True,  # Enable verbose to see what's being filtered
                use_registry=True,  # Enable registry validation
                data_interval_minutes=detected_interval
            )
            
            # Record registry filtering
            auditor.record_registry_allowed(validated_features, all_columns)
            logger.info(f"📊 Feature audit [{target}]: {len(validated_features)} features allowed by registry (from {len(all_columns)} total columns)")
            
            # Keep only features that are both in feature_names and validated
            feature_names = [f for f in feature_names if f in validated_features]
            
            if len(feature_names) < len([f for f in feature_names if f in validated_features]):
                logger.info(f"  Feature registry: Validated {len(feature_names)} features for target {target}")
        except Exception as e:
            logger.warning(f"  Feature registry validation failed: {e}. Using provided features as-is.")
    
    # Normalize time column name
    ts_name = "timestamp" if "timestamp" in combined_pl.columns else ("ts" if "ts" in combined_pl.columns else None)
    
    # Enforce min_cs: filter timestamps that don't meet cross-sectional size
    if ts_name:
        combined_pl = combined_pl.filter(
            pl.len().over(ts_name) >= min_cs
        )
    
    # Apply cross-sectional sampling if specified
    if max_cs_samples and ts_name:
        logger.info(f"📊 Applying cross-sectional sampling: max {max_cs_samples} samples per timestamp")
        
        # Use deterministic per-timestamp sampling with simple approach
        combined_pl = (
            combined_pl
            .sort([ts_name])
            .group_by(ts_name, maintain_order=True)
            .head(max_cs_samples)
        )
        
        logger.info(f"Cross-sectional sampling applied")
    
    # Extract target and features using polars
    try:
        # Get target column
        target_series_pl = combined_pl.select(pl.col(target))
        y = target_series_pl.to_pandas()[target].values
        
        # CRITICAL FIX: Track feature pipeline stages separately
        # Stage 1: requested → allowed (registry filtering - expected)
        # Stage 2: allowed → present (schema mismatch - NOT expected, indicates bug)
        # Stage 3: present → used (dtype/nan filtering - may be expected)
        
        # Get registry-allowed count (before schema check)
        registry_allowed_count = len(feature_names)  # feature_names is already registry-filtered at this point
        requested_count = len(auditor.requested_features) if hasattr(auditor, 'requested_features') and auditor.requested_features else registry_allowed_count
        
        # Check which allowed features are actually present in Polars frame
        available_features = [f for f in feature_names if f in combined_pl.columns]
        missing_in_polars = [c for c in feature_names if c not in combined_pl.columns]
        
        # CRITICAL FIX (Pitfall B): Diagnose missing allowed features with close matches
        if missing_in_polars:
            logger.error(f"🚨 CRITICAL [{target}]: {len(missing_in_polars)} registry-allowed features missing from polars frame")
            logger.error(f"  Missing features: {missing_in_polars[:20]}{'...' if len(missing_in_polars) > 20 else ''}")
            
            # Find close matches for missing features (helps diagnose name mismatches)
            from difflib import get_close_matches
            all_polars_cols = list(combined_pl.columns)
            close_matches = {}
            for missing_feat in missing_in_polars[:10]:  # Limit to first 10 for performance
                matches = get_close_matches(missing_feat, all_polars_cols, n=3, cutoff=0.6)
                if matches:
                    close_matches[missing_feat] = matches
            if close_matches:
                logger.error(f"  Close matches found in polars columns:")
                for missing, matches in close_matches.items():
                    logger.error(f"    '{missing}' → {matches}")
            
            if auditor:
                for feat in missing_in_polars:
                    auditor.record_drop(feat, "missing_from_polars", f"Feature not in polars frame columns")
        
        # Record features present in Polars
        auditor.record_present_in_polars(combined_pl, feature_names)
        present_count = len(auditor.present_in_polars) if hasattr(auditor, 'present_in_polars') else len(available_features)
        logger.info(
            f"📊 Feature audit [{target}]: "
            f"requested={requested_count} → allowed={registry_allowed_count} → present={present_count} "
            f"(allowed→present drop: {len(missing_in_polars)})"
        )
        
        # Build feature_cols with only existing columns
        feature_cols = [target] + available_features + ['symbol'] + ([ts_name] if ts_name and ts_name in combined_pl.columns else [])
        
        # CRITICAL FIX (Pitfall A): Check threshold on allowed → present (not requested → present)
        # This prevents false positives when registry intentionally prunes features
        if registry_allowed_count > 0 and present_count < registry_allowed_count * 0.5:
            error_msg = (
                f"🚨 CRITICAL [{target}]: Feature schema mismatch detected! "
                f"Registry allowed {registry_allowed_count} features, but only {present_count} exist in polars frame "
                f"(ratio={present_count/registry_allowed_count:.1%}). "
                f"This indicates a schema breach or feature name mismatch. "
                f"Missing allowed features: {missing_in_polars[:20]}{'...' if len(missing_in_polars) > 20 else ''}"
            )
            if close_matches:
                error_msg += f"\n  Close matches found (possible name mismatches): {close_matches}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        data_pl = combined_pl.select(feature_cols)
        
        # Convert to pandas for sklearn compatibility
        combined_df = data_pl.to_pandas()
        
        logger.info(f"Extracted target {target} from polars data")
        logger.info(f"🔍 Debug [{target}]: After polars→pandas conversion: combined_df shape={combined_df.shape}, "
                   f"feature_names count={len(feature_names)}, "
                   f"features in df={len([f for f in feature_names if f in combined_df.columns])}")
        
    except Exception as e:
        logger.error(f"Error extracting target {target}: {e}")
        return (None,)*8
    
    # Continue with pandas-based processing (pass auditor for tracking)
    result = _process_combined_data_pandas(combined_df, target, feature_names, auditor=auditor)
    
    # Write audit report if auditor was used
    if auditor and len(auditor.drop_records) > 0:
        try:
            # Try to get output directory from environment or use default
            import os
            output_dir = os.getenv("TRAINING_OUTPUT_DIR", "output")
            audit_dir = Path(output_dir) / "artifacts" / "feature_audits"
            audit_dir.mkdir(parents=True, exist_ok=True)
            auditor.write_report(audit_dir)
        except Exception as e:
            logger.warning(f"Failed to write feature audit report: {e}")
    
    return result

def _prepare_training_data_pandas(mtf_data: Dict[str, pd.DataFrame], 
                                 target: str, 
                                 feature_names: List[str] = None,
                                 min_cs: int = 10,
                                 max_cs_samples: int = None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Pandas-based data preparation (fallback)."""
    
    # Combine all symbol data
    all_data = []
    for symbol, df in mtf_data.items():
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Normalize time column name
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    
    # Enforce min_cs and apply sampling
    if time_col is not None:
        # enforce min_cs
        cs = combined_df.groupby(time_col)["symbol"].transform("size")
        combined_df = combined_df[cs >= min_cs]
        # per-timestamp deterministic sampling
        if max_cs_samples:
            combined_df["_rn"] = combined_df.groupby(time_col).cumcount()
            combined_df = (combined_df
                           .sort_values([time_col, "_rn"])
                           .groupby(time_col, group_keys=False)
                           .head(max_cs_samples)
                           .drop(columns="_rn"))
    
    # Auto-discover features, then validate with registry
    if feature_names is None:
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', time_col]]
    
    # Validate features with registry (if enabled)
    if feature_names:
        try:
            from TRAINING.ranking.utils.leakage_filtering import filter_features_for_target
            from TRAINING.ranking.utils.data_interval import detect_interval_from_dataframe
            
            # Detect data interval for horizon conversion
            detected_interval = detect_interval_from_dataframe(combined_df, timestamp_column=time_col or 'ts', default=5)
            # Ensure interval is valid (> 0)
            if detected_interval <= 0:
                detected_interval = 5
                logger.warning(f"  Invalid detected interval, using default: 5m")
            
            # Filter features using registry
            all_columns = combined_df.columns.tolist()
            validated_features = filter_features_for_target(
                all_columns,
                target,
                verbose=True,  # Enable verbose to see what's being filtered
                use_registry=True,  # Enable registry validation
                data_interval_minutes=detected_interval
            )
            
            # Keep only features that are both in feature_names and validated
            feature_names = [f for f in feature_names if f in validated_features]
            
            if len(feature_names) > 0:
                logger.info(f"  Feature registry: Validated {len(feature_names)} features for target {target}")
        except Exception as e:
            logger.warning(f"  Feature registry validation failed: {e}. Using provided features as-is.")
    
    return _process_combined_data_pandas(combined_df, target, feature_names)

def _process_combined_data_pandas(combined_df: pd.DataFrame, target: str, feature_names: List[str], auditor=None) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, List[str], Optional[np.ndarray], Dict[str, Any]]:
    """Process combined data using pandas."""
    
    # Route target to get task specification
    route_info = route_target(target)
    spec = route_info['spec']
    logger.info(f"[Router] Target {target} → {spec.task} task (objective={spec.objective})")
    
    # Extract target using safe extraction
    try:
        target_series, actual_col = safe_target_extraction(combined_df, target)
        # Sanitize target: replace inf/-inf with NaN
        target_series = target_series.replace([np.inf, -np.inf], np.nan)
        y = target_series.values
        logger.info(f"Extracted target {target} from column {actual_col}")
    except Exception as e:
        logger.error(f"Error extracting target {target}: {e}")
        return (None,)*8
    
    # Extract feature matrix - handle non-numeric columns
    # CRITICAL: Check if feature_names is empty or None
    if not feature_names:
        logger.error(f"❌ CRITICAL [{target}]: feature_names is empty or None! Cannot proceed.")
        return (None,)*8
    
    # Check which features actually exist in combined_df
    existing_features = [f for f in feature_names if f in combined_df.columns]
    missing_cols = [f for f in feature_names if f not in combined_df.columns]
    
    if not existing_features:
        logger.error(f"❌ CRITICAL [{target}]: NONE of the {len(feature_names)} selected features exist in combined_df!")
        logger.error(f"❌ [{target}]: Selected features: {feature_names[:20]}")
        logger.error(f"❌ [{target}]: Sample of combined_df columns: {list(combined_df.columns)[:20]}")
        return (None,)*8
    
    if missing_cols:
        logger.warning(f"🔍 Debug [{target}]: {len(missing_cols)} selected features missing from combined_df: {missing_cols[:10]}")
        logger.warning(f"🔍 Debug [{target}]: Using {len(existing_features)} existing features instead of {len(feature_names)}")
        feature_names = existing_features  # Use only existing features
    
    feature_df = combined_df[feature_names].copy()
    
    # Record features kept for training (before coercion)
    if auditor:
        auditor.record_kept_for_training(feature_df, feature_names)
        logger.info(f"📊 Feature audit [{target}]: {len(auditor.kept_for_training)} features kept for training (before coercion)")
    
    # DIAGNOSTIC: Log initial state before coercion
    logger.info(f"🔍 Debug [{target}]: Initial feature_df shape={feature_df.shape}, "
               f"feature_names count={len(feature_names)}, "
               f"columns in df={len([c for c in feature_names if c in feature_df.columns])}")
    
    # DIAGNOSTIC: Check NaN ratios BEFORE coercion
    if len(feature_df.columns) > 0:
        pre_coerce_nan_ratios = feature_df.isna().mean()
        all_nan_before = pre_coerce_nan_ratios[pre_coerce_nan_ratios == 1.0]
        if len(all_nan_before) > 0:
            logger.warning(f"🔍 Debug [{target}]: {len(all_nan_before)} features are ALL NaN BEFORE coercion: {list(all_nan_before.index)[:10]}")
    
    # Convert to numeric, coerce errors to NaN, and sanitize infinities
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # DIAGNOSTIC: Check NaN ratios AFTER coercion
    if len(feature_df.columns) > 0:
        post_coerce_nan_ratios = feature_df.isna().mean()
        all_nan_after = post_coerce_nan_ratios[post_coerce_nan_ratios == 1.0]
        if len(all_nan_after) > 0:
            logger.error(f"🔍 Debug [{target}]: {len(all_nan_after)} features became ALL NaN AFTER coercion: {list(all_nan_after.index)[:10]}")
            # Log sample of what these columns look like in raw data
            sample_cols = list(all_nan_after.index)[:5]
            for col in sample_cols:
                if col in combined_df.columns:
                    raw_sample = combined_df[col].head(10)
                    raw_dtype = combined_df[col].dtype
                    logger.error(f"🔍 Debug [{target}]: Column '{col}' (dtype={raw_dtype}) raw sample: {raw_sample.tolist()}")
    
    # Drop columns that are entirely NaN after coercion
    before_cols = feature_df.shape[1]
    dropped_cols = feature_df.columns[feature_df.isna().all()].tolist()
    feature_df = feature_df.dropna(axis=1, how='all')
    dropped_all_nan = before_cols - feature_df.shape[1]
    if dropped_all_nan:
        logger.warning(f"🔧 Dropped {dropped_all_nan} all-NaN feature columns after coercion")
        if auditor:
            auditor.record_dropped_all_nan(dropped_cols, combined_df)
        
        # CRITICAL: If ALL features were dropped, this is fatal
        if feature_df.shape[1] == 0:
            logger.error(f"❌ CRITICAL [{target}]: ALL {before_cols} selected features became all-NaN after coercion!")
            logger.error(f"❌ [{target}]: Selected features: {feature_names[:20]}...")
            logger.error(f"❌ [{target}]: This indicates a mismatch between feature_names and actual data columns/dtypes")
            
            # Write debug file
            try:
                import os
                debug_dir = Path("debug_feature_coercion")
                debug_dir.mkdir(exist_ok=True)
                debug_path = debug_dir / f"all_nan_features_{target.replace('/', '_')}.npz"
                np.savez_compressed(
                    debug_path,
                    feature_names=np.array(feature_names, dtype=object),
                    combined_df_columns=np.array(combined_df.columns.tolist(), dtype=object),
                    missing_cols=np.array(missing_cols, dtype=object) if missing_cols else np.array([], dtype=object),
                )
                logger.error(f"❌ [{target}]: Wrote debug file to {debug_path}")
            except Exception as e:
                logger.error(f"❌ [{target}]: Failed to write debug file: {e}")
            
            return (None,)*8
    
    # Ensure only numeric dtypes remain (guard against objects/arrays)
    numeric_cols = [c for c in feature_df.columns if pd.api.types.is_numeric_dtype(feature_df[c])]
    if len(numeric_cols) != feature_df.shape[1]:
        non_numeric_dropped = feature_df.shape[1] - len(numeric_cols)
        dropped_non_numeric = [c for c in feature_df.columns if c not in numeric_cols]
        feature_df = feature_df[numeric_cols]
        logger.info(f"🔧 Dropped {non_numeric_dropped} non-numeric feature columns")
        if auditor:
            auditor.record_dropped_non_numeric(dropped_non_numeric, combined_df)
    
    # Build float32 matrix safely
    X = feature_df.to_numpy(dtype=np.float32, copy=False)
    
    # Record features used in final X matrix
    if auditor:
        final_feature_names = feature_df.columns.tolist()
        auditor.record_used_in_X(final_feature_names, X)
        logger.info(f"📊 Feature audit [{target}]: {len(final_feature_names)} features used in final X matrix")
    
    # CRITICAL: Guard against empty feature matrix
    if X.shape[1] == 0:
        logger.error(f"❌ CRITICAL [{target}]: Feature matrix X has 0 columns after coercion and filtering!")
        logger.error(f"❌ [{target}]: Cannot proceed with training - no usable features")
        return (None,)*8
    
    # DIAGNOSTIC: Log X shape and feature stats
    logger.info(f"🔍 Debug [{target}]: X shape={X.shape}, y shape={y.shape}, "
               f"X NaN count={np.isnan(X).sum()}, y NaN count={np.isnan(y).sum()}")
    
    # Clean data - be more lenient with NaN values
    target_valid = ~np.isnan(y)
    
    # Compute feature NaN ratio safely (handle empty X case)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if X.shape[0] > 0 and X.shape[1] > 0:
            feature_nan_ratio = np.isnan(X).mean(axis=1)
        else:
            feature_nan_ratio = np.ones(X.shape[0])  # All invalid if empty
            logger.error(f"❌ [{target}]: X has zero columns or rows - cannot compute feature_nan_ratio")
            return (None,)*8
    
    feature_valid = feature_nan_ratio <= 0.5  # Allow up to 50% NaN in features
    
    # Treat inf in target as invalid as well
    y_is_finite = np.isfinite(y)
    valid_mask = target_valid & feature_valid & y_is_finite
    
    if not valid_mask.any():
        logger.error(f"❌ [{target}]: No valid data after cleaning")
        logger.error(f"❌ [{target}]: Target stats - total={len(y)}, valid={target_valid.sum()}, "
                    f"NaN={np.isnan(y).sum()}, inf={np.sum(~np.isfinite(y))}")
        logger.error(f"❌ [{target}]: Feature stats - rows={X.shape[0]}, cols={X.shape[1]}, "
                    f"valid_rows={feature_valid.sum()}, mean_NaN_ratio={feature_nan_ratio.mean():.2%}")
        return (None,)*8
    
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    symbols_clean = combined_df['symbol'].values[valid_mask]
    
    # Get final feature column names (after all filtering/dropping)
    # This should match the columns in X_clean (which comes from feature_df after filtering)
    final_feature_cols = list(feature_df.columns)  # Actual columns after filtering/dropping
    
    # Fill remaining NaN values with median (load strategy from config if available)
    from sklearn.impute import SimpleImputer
    if _CONFIG_AVAILABLE:
        try:
            imputation_strategy = get_cfg("preprocessing.imputation.strategy", default="median", config_name="preprocessing_config")
        except Exception:
            imputation_strategy = "median"
    else:
        imputation_strategy = "median"
    imputer = SimpleImputer(strategy=imputation_strategy)
    X_clean = imputer.fit_transform(X_clean)
    
    logger.info(f"Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features")
    logger.info(f"Removed {len(X) - len(X_clean)} rows due to cleaning")
    
    # Determine time column and extract time values
    time_col = "timestamp" if "timestamp" in combined_df.columns else ("ts" if "ts" in combined_df.columns else None)
    time_vals = combined_df[time_col].values[valid_mask] if time_col else None
    
    # Apply routing-based label preparation
    y_prepared, sample_weights, group_sizes, routing_meta = route_info['prepare_fn'](y_clean, time_vals)
    
    # Store routing metadata for trainer
    routing_meta['target_name'] = target
    routing_meta['spec'] = spec
    routing_meta['sample_weights'] = sample_weights
    routing_meta['group_sizes'] = group_sizes
    
    logger.info(f"[Routing] Prepared {spec.task} task: y_shape={y_prepared.shape}, has_weights={sample_weights is not None}, has_groups={group_sizes is not None}")
    
    # Return with prepared labels instead of raw labels
    # Note: We return routing_meta in the imputer slot (slot 7) for now - trainer can extract it
    # feat_cols should be the actual column names after filtering (numeric_cols), not the input feature_names
    final_feature_cols = list(feature_df.columns)  # Actual columns after filtering/dropping
    return X_clean, y_prepared, feature_names, symbols_clean, np.arange(len(X_clean)), final_feature_cols, time_vals, routing_meta

