"""
Feature Alignment Utilities

As-of join alignment for multi-interval features.
Aligns features with different native intervals onto a base training grid
using availability timestamps (feature_ts + embargo) to prevent lookahead bias.
"""

import pandas as pd
import polars as pl
from typing import Dict, List, Optional, Tuple, Any
import logging

from TRAINING.utils.feature_time_meta import FeatureTimeMeta

logger = logging.getLogger(__name__)


def align_features_asof(
    base_df: pd.DataFrame,
    feature_dfs: Dict[str, pd.DataFrame],
    feature_time_meta_map: Dict[str, FeatureTimeMeta],
    base_interval_minutes: float,
    symbol_column: str = "symbol",
    timestamp_column: str = "ts",
    max_staleness_minutes: Optional[float] = None
) -> pd.DataFrame:
    """
    Align features with different native intervals onto base training grid using as-of join.
    
    Key concept: Features are aligned by "availability timestamp" (feature_ts + embargo),
    not by feature timestamp. This prevents leakage when mixing intervals.
    
    Args:
        base_df: Base training grid DataFrame with columns [symbol_column, timestamp_column]
        feature_dfs: Dictionary mapping feature_name -> DataFrame with feature data
        feature_time_meta_map: Dictionary mapping feature_name -> FeatureTimeMeta
        base_interval_minutes: Base training grid interval in minutes
        symbol_column: Name of symbol column (default: "symbol")
        timestamp_column: Name of timestamp column (default: "ts")
        max_staleness_minutes: Optional cap on forward-fill (None = no cap)
    
    Returns:
        DataFrame with base grid columns + aligned feature columns
    
    Algorithm:
    1. Group features by (native_interval, embargo, publish_offset, max_staleness)
    2. For each group, compute availability_ts = feature_ts + publish_offset + embargo
    3. As-of join base_df with feature group on (symbol, availability_ts <= base_ts)
    4. Apply staleness cap if max_staleness_minutes is set
    """
    if not feature_dfs:
        return base_df.copy()
    
    # Group features by alignment parameters (same group = same join strategy)
    feature_groups: Dict[Tuple[float, float, float, Optional[float]], List[str]] = {}
    for feat_name, meta in feature_time_meta_map.items():
        if feat_name not in feature_dfs:
            logger.warning(f"Feature {feat_name} in meta_map but not in feature_dfs, skipping")
            continue
        
        # Use native interval or default to base
        native_interval = meta.native_interval_minutes or base_interval_minutes
        
        # Group key: (native_interval, embargo, publish_offset, max_staleness)
        group_key = (
            native_interval,
            meta.embargo_minutes,
            meta.publish_offset_minutes,
            meta.max_staleness_minutes or max_staleness_minutes
        )
        
        if group_key not in feature_groups:
            feature_groups[group_key] = []
        feature_groups[group_key].append(feat_name)
    
    # Start with base grid
    result_df = base_df.copy()
    
    # Process each group (one join per group for efficiency)
    for group_key, feat_names in feature_groups.items():
        native_interval, embargo, publish_offset, group_max_staleness_minutes = group_key
        
        # Skip as-of join if all features in group use base interval, zero embargo, zero publish_offset, and no staleness cap
        # (already aligned, no forward-fill needed)
        if (native_interval == base_interval_minutes and 
            embargo == 0.0 and 
            publish_offset == 0.0 and
            group_max_staleness_minutes is None):
            # Features are already on base grid, just merge directly
            for feat_name in feat_names:
                if feat_name in feature_dfs:
                    feat_df = feature_dfs[feat_name]
                    # Simple merge on symbol and timestamp (assuming same timeline)
                    result_df = result_df.merge(
                        feat_df[[symbol_column, timestamp_column, feat_name]],
                        on=[symbol_column, timestamp_column],
                        how="left"
                    )
            continue
        
        # For non-base-interval features, do as-of join
        logger.debug(
            f"Aligning {len(feat_names)} features with native_interval={native_interval}m, "
            f"embargo={embargo}m, publish_offset={publish_offset}m"
        )
        
        # Combine all features in this group into one DataFrame
        group_feat_dfs = []
        for feat_name in feat_names:
            if feat_name not in feature_dfs:
                continue
            feat_df = feature_dfs[feat_name].copy()
            
            # Compute availability timestamp
            if timestamp_column in feat_df.columns:
                # Convert to datetime if needed
                if not pd.api.types.is_datetime64_any_dtype(feat_df[timestamp_column]):
                    feat_df[timestamp_column] = pd.to_datetime(feat_df[timestamp_column])
                
                # availability_ts = feature_ts + publish_offset + embargo
                feat_df['_availability_ts'] = (
                    pd.to_datetime(feat_df[timestamp_column]) +
                    pd.Timedelta(minutes=publish_offset + embargo)
                )
            else:
                logger.warning(f"Feature {feat_name} missing timestamp column {timestamp_column}, skipping")
                continue
            
            group_feat_dfs.append(feat_df)
        
        if not group_feat_dfs:
            continue
        
        # Combine features in this group
        group_df = pd.concat(group_feat_dfs, ignore_index=True)
        
        # Sort by symbol and availability_ts for as-of join
        group_df = group_df.sort_values([symbol_column, '_availability_ts'])
        
        # Ensure base_df timestamp is datetime
        if timestamp_column in result_df.columns:
            if not pd.api.types.is_datetime64_any_dtype(result_df[timestamp_column]):
                result_df[timestamp_column] = pd.to_datetime(result_df[timestamp_column])
        
        # As-of join: for each base row, find latest feature row with availability_ts <= base_ts
        # Use merge_asof for efficient as-of join
        # CRITICAL: Sort both DataFrames before merge_asof
        result_df_sorted = result_df.sort_values([symbol_column, timestamp_column])
        group_df_sorted = group_df.sort_values([symbol_column, '_availability_ts'])
        
        # Select feature columns to join
        feature_cols = [f for f in feat_names if f in group_df_sorted.columns]
        if not feature_cols:
            continue
        
        joined_df = pd.merge_asof(
            result_df_sorted,
            group_df_sorted[[symbol_column, '_availability_ts'] + feature_cols],
            left_on=timestamp_column,
            right_on='_availability_ts',
            by=symbol_column,
            direction='backward'  # Latest available value at or before base timestamp
        )
        
        # INVARIANT CHECK: Verify availability time semantics (available_ts <= base_ts)
        # This ensures no lookahead bias
        if '_availability_ts' in joined_df.columns and timestamp_column in joined_df.columns:
            # Check that all non-null availability timestamps are <= base timestamp
            mask_valid = joined_df['_availability_ts'].notna()
            if mask_valid.any():
                availability_violations = (joined_df.loc[mask_valid, '_availability_ts'] > 
                                          joined_df.loc[mask_valid, timestamp_column])
                if availability_violations.any():
                    n_violations = availability_violations.sum()
                    logger.error(
                        f"🚨 ALIGNMENT INVARIANT VIOLATION: {n_violations} rows have availability_ts > base_ts "
                        f"(lookahead leak detected). This should never happen with backward merge_asof."
                    )
                    # In debug mode, raise error; in production, log and continue
                    import os
                    if os.environ.get('STRICT_ALIGNMENT_CHECKS', '0') == '1':
                        raise ValueError(f"Alignment invariant violation: {n_violations} rows with availability_ts > base_ts")
        
        # Update result_df with joined feature columns
        for feat_name in feature_cols:
            if feat_name in joined_df.columns:
                result_df[feat_name] = joined_df[feat_name].values
        
        # Apply staleness cap if set (use group-specific or global)
        staleness_cap = group_max_staleness_minutes if group_max_staleness_minutes is not None else max_staleness_minutes
        if staleness_cap is not None and '_availability_ts' in joined_df.columns:
            # Staleness = base_ts - source_ts (how old is the feature value we're using)
            # Since availability_ts = source_ts + publish_offset + embargo,
            # we have: source_ts = availability_ts - (publish_offset + embargo)
            # Therefore: staleness = base_ts - (availability_ts - publish_offset - embargo)
            #           = base_ts - availability_ts + publish_offset + embargo
            staleness = (
                (result_df[timestamp_column] - joined_df['_availability_ts']).dt.total_seconds() / 60.0 +
                publish_offset + embargo
            )
            mask_stale = staleness > staleness_cap
            n_stale = mask_stale.sum()
            if n_stale > 0:
                logger.debug(f"  Staleness cap: nulling {n_stale} rows where staleness > {staleness_cap}m")
            for feat_name in feature_cols:
                if feat_name in result_df.columns:
                    result_df.loc[mask_stale, feat_name] = None  # Null out stale values
        
        # Drop temporary column
        if '_availability_ts' in result_df.columns:
            result_df = result_df.drop(columns=['_availability_ts'])
        
        # INVARIANT CHECK: Row count and uniqueness
        # Base index (symbol, timestamp) must remain unique and sorted after merge
        if symbol_column in result_df.columns and timestamp_column in result_df.columns:
            n_rows_before = len(result_df_sorted)
            n_rows_after = len(result_df)
            if n_rows_before != n_rows_after:
                logger.warning(
                    f"⚠️  Alignment changed row count: {n_rows_before} → {n_rows_after} "
                    f"(should be unchanged). This may indicate a merge issue."
                )
            
            # Check uniqueness
            duplicates = result_df.duplicated(subset=[symbol_column, timestamp_column])
            if duplicates.any():
                n_dupes = duplicates.sum()
                logger.error(
                    f"🚨 ALIGNMENT INVARIANT VIOLATION: {n_dupes} duplicate (symbol, timestamp) pairs after merge. "
                    f"This should never happen."
                )
                import os
                if os.environ.get('STRICT_ALIGNMENT_CHECKS', '0') == '1':
                    raise ValueError(f"Alignment invariant violation: {n_dupes} duplicate rows")
    
    return result_df


def build_base_grid(
    symbols: List[str],
    timestamps: pd.Series,
    symbol_column: str = "symbol",
    timestamp_column: str = "ts"
) -> pd.DataFrame:
    """
    Build base training grid from symbols and timestamps.
    
    Args:
        symbols: List of symbols
        timestamps: Series of timestamps (will be expanded to all symbols)
        symbol_column: Name of symbol column
        timestamp_column: Name of timestamp column
    
    Returns:
        DataFrame with columns [symbol_column, timestamp_column] representing base grid
    """
    # Create cartesian product of symbols and timestamps
    base_grid = pd.MultiIndex.from_product(
        [symbols, timestamps],
        names=[symbol_column, timestamp_column]
    ).to_frame(index=False)
    
    return base_grid
