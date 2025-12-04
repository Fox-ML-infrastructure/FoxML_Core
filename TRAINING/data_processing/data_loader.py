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

"""Data processing utilities extracted from original 5K line file."""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def load_mtf_data(data_dir: str, symbols: List[str], interval: str = "5m", max_rows_per_symbol: int = None) -> Dict[str, pd.DataFrame]:
    """Load MTF data for specified symbols and interval.

    
    Args:
        data_dir: Directory containing parquet files
        symbols: List of symbols to load
        interval: Data interval (e.g., "5m")
        max_rows_per_symbol: Optional limit to prevent OOM on large datasets
    """
    if not USE_POLARS:
        return _load_mtf_data_pandas(data_dir, symbols, interval, max_rows_per_symbol)
    
    mtf_data = {}
    
    for symbol in symbols:
        # Try new directory structure first: interval={interval}/symbol={symbol}/{symbol}.parquet
        new_path = Path(data_dir) / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.parquet"
        legacy_path = Path(data_dir) / f"{symbol}_mtf.parquet"
        
        file_path = new_path if new_path.exists() else legacy_path
        if not file_path.exists():
            logger.warning(f"File not found for {symbol} at {new_path} or {legacy_path}")
            continue
            
        try:
            # Lazy scan - won't materialize until collect()
            lf = pl.scan_parquet(str(file_path))
            # Detect/standardize time column
            tcol = _resolve_time_col_polars(lf.collect_schema().names())
            # Use tolerant cast instead of strptime (handles both string and datetime columns)
            lf = lf.with_columns(pl.col(tcol).cast(pl.Datetime, strict=False).alias(tcol))\
                   .drop_nulls([tcol])
            if max_rows_per_symbol:
                lf = lf.tail(max_rows_per_symbol)  # Keep most recent
            df = lf.collect(streaming=True)
            # Hand back pandas for compatibility with the rest of your code
            mtf_data[symbol] = df.to_pandas(use_pyarrow_extension_array=False)
            logger.info(f"Loaded {symbol}: {len(mtf_data[symbol]):,} rows, {len(mtf_data[symbol].columns)} cols")
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    return mtf_data



def prepare_training_data_cross_sectional(
    mtf_data: Dict[str, pd.DataFrame],
    target: str,
    common_features: List[str],
    min_cs: int,
    max_samples_per_symbol: int = None,  # ignore per-symbol sampling for CS
    all_targets: set = None  # Set of all discovered target columns for precise filtering
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.Series, List[str]]:
    """
    Prepare TRUE cross-sectional training data with timestamp-based grouping.
    Clean implementation using long-format schema to prevent column collisions.
    
    Args:
        mtf_data: Dictionary of symbol -> DataFrame
        target: Target column name
        common_features: List of common feature columns
        max_samples_per_symbol: Ignored for CS training (we need all data for grouping)
    
    Returns:
        X: Feature matrix
        y: Target vector
        symbols: Symbol array
        groups: Group sizes for ranking
        ts_index: Timestamp index
        feat_cols: Actual feature columns used (may differ from common_features)
    """
    if not USE_POLARS:
        return _prepare_training_data_cross_sectional_pandas(mtf_data, target, common_features, min_cs, max_samples_per_symbol)

    logger.info("🎯 Building TRUE cross-sectional training data (polars, clean)…")
    if not mtf_data:
        return None, None, None, None, None, None

    # Check if target exists in any of the datasets
    target_found = False
    for sym, pdf in mtf_data.items():
        if target in pdf.columns:
            target_found = True
            break
    
    if not target_found:
        logger.error(f"Target '{target}' not found in any dataset. Available targets: {[col for col in next(iter(mtf_data.values())).columns if col.startswith('fwd_ret_')]}")
        return None, None, None, None, None, None

    # Detect time col
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    SYM = SYMBOL_COL

    # Build CS table in long format (one row per symbol-timestamp)
    lfs = []
    processed_symbols = 0
    for sym, pdf in mtf_data.items():
        cols = pdf.columns.tolist()
        
        # Pick exactly one target-like column (handles suffixes: _x/_y/_SYM/etc)
        tgt_in = _pick_one(cols, target)
        if tgt_in is None:
            # no target in this symbol → skip
            continue
        
        # ✅ NEW SAFE PATTERN: Strip ALL targets from features
        have_feats = [c for c in strip_targets(common_features, all_targets) if c in cols]
        if not have_feats:
            continue
        
        # Build selection: features + exactly one target + metadata
        sel = [time_col] + have_feats + [tgt_in]
        logger.info(f"Symbol {sym}: Selecting columns: {len(sel)} total")
        logger.info(f"Symbol {sym}: Target column '{tgt_in}' in selection: {tgt_in in sel}")
        df_subset = pdf[sel].copy()
        
        # Rename target column to canonical name
        if tgt_in != target:
            df_subset = df_subset.rename(columns={tgt_in: target})
        
        # ✅ NEW SAFE PATTERN: Collapse duplicate columns safely
        try:
            df_subset = collapse_identical_duplicate_columns(df_subset)
        except ValueError as e:
            logger.error(f"Symbol {sym}: {e}")
            continue
        
        logger.info(f"Symbol {sym}: After selection - shape: {df_subset.shape}, target column: {target in df_subset.columns}")
        
        # ✅ NEW SAFE PATTERN: Final contract validation
        if target in df_subset.columns:
            try:
                from target_resolver import safe_target_extraction
                target_series, actual_col = safe_target_extraction(df_subset, target)
                logger.info(f"Symbol {sym}: Target values sample: {target_series.head(3).tolist()}")
                
                # ✅ Contract validation: X = features only, y = exactly one target
                feature_cols = [c for c in df_subset.columns if c not in (target, time_col, "symbol")]
                assert target not in feature_cols, f"Target '{target}' leaked into features!"
                assert len(feature_cols) > 0, "No features found after target filtering"
                
            except Exception as e:
                logger.error(f"Symbol {sym}: Target extraction failed for '{target}': {e}")
                continue
        
        try:
            # ✅ Make sure target is numeric BEFORE Polars (see: all-NULL issue)
            df_subset[target] = pd.to_numeric(df_subset[target], errors="coerce")
            nnz = int(df_subset[target].notna().sum())
            if nnz == 0:
                logger.warning(f"Symbol {sym}: target '{target}' is all-NaN in pandas (before Polars); skipping")
                continue

            # Convert to Polars
            lf = pl.from_pandas(df_subset)
            logger.info(f"Symbol {sym}: After pandas->polars conversion: {lf.shape}")
            
            # Debug: check target column immediately after conversion
            try:
                target_after_conv = lf.select(pl.col(target).count()).item()
                target_nulls_after_conv = lf.select(pl.col(target).null_count()).item()
                logger.info(f"Symbol {sym}: After pandas->polars - target count: {target_after_conv}, nulls: {target_nulls_after_conv}")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not check target after conversion: {e}")

            # --- Timestamp handling (do NOT overwrite ts with year etc.) ---
            if lf[time_col].dtype in (pl.Int64, pl.UInt64, pl.Int32, pl.UInt32):
                # infer unit by span and convert with from_epoch
                span = max(abs(int(lf[time_col].min() or 0)), abs(int(lf[time_col].max() or 0)))
                

def _apply_cs_transforms_polars(df, feat_cols, time_col, p, ddof):
    """Polars-based CS transforms with multi-threading and streaming."""
    logger.info("🔧 CS transforms (polars, multi-threaded)…")
    ldf = pl.from_pandas(df[[time_col, *feat_cols]]).lazy()

    # stats per timestamp; choose winsor method
    if CS_WINSOR.lower() == "quantile":
        aggs = []
        for c in feat_cols:
            aggs += [
                pl.col(c).quantile(p).alias(f"{c}__lo"),
                pl.col(c).quantile(1 - p).alias(f"{c}__hi"),
                pl.col(c).mean().alias(f"{c}__mu"),
                pl.col(c).std(ddof=ddof).fill_nan(1e-8).fill_null(1e-8).alias(f"{c}__sd"),
            ]
        stats = ldf.group_by(time_col).agg(aggs)
        exprs = []
        for c in feat_cols:
            lo, hi, mu, sd = (pl.col(f"{c}__lo"), pl.col(f"{c}__hi"),
                              pl.col(f"{c}__mu"), pl.col(f"{c}__sd"))
            clipped = pl.when(pl.col(c) < lo).then(lo)\
                        .when(pl.col(c) > hi).then(hi)\
                        .otherwise(pl.col(c))
            exprs.append(((clipped - mu) / (sd + 1e-8)).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(c))
        out = (ldf.join(stats, on=time_col, how="left")
                 .with_columns(exprs)
                 .select([time_col, *feat_cols])
                 .collect(streaming=True))
    else:
        # streaming-friendly: k-sigma winsorization (approx quantile)
        # p=0.01 ≈ k=2.33 for normal
        import math
        from math import erf, sqrt
        # inverse CDF approx: use scipy if present, else 2.33 fallback
        try:
            from scipy.stats import norm
            k = float(norm.ppf(1 - p))
        except Exception:
            k = 2.33 if abs(p - 0.01) < 1e-6 else 2.0
        aggs = []
        for c in feat_cols:
            aggs += [
                pl.col(c).cast(pl.Float64).mean().alias(f"{c}__mu"),
                pl.col(c).cast(pl.Float64).std(ddof=ddof).fill_nan(1e-8).fill_null(1e-8).alias(f"{c}__sd"),
            ]
        stats = ldf.group_by(time_col).agg(aggs)
        exprs = []
        for c in feat_cols:
            mu, sd = (pl.col(f"{c}__mu"), pl.col(f"{c}__sd"))
            lo = (mu - k * sd)
            hi = (mu + k * sd)
            clipped = pl.when(pl.col(c).cast(pl.Float64) < lo).then(lo)\
                        .when(pl.col(c).cast(pl.Float64) > hi).then(hi)\
                        .otherwise(pl.col(c).cast(pl.Float64))
            exprs.append(((clipped - mu) / (sd + 1e-8)).fill_nan(0.0).fill_null(0.0).cast(pl.Float32).alias(c))
        out = (ldf.join(stats, on=time_col, how="left")
                 .with_columns(exprs)
                 .select([time_col, *feat_cols])
                 .collect(streaming=True))

    # Write back into the original pandas df as float32
    df[feat_cols] = out.to_pandas(use_pyarrow_extension_array=False)[feat_cols].astype("float32")
    return df



def _resolve_time_col_polars(cols):
    """Resolve time column name for Polars."""
    for c in ("ts","timestamp","time","datetime","ts_pred"):
        if c in cols:
            return c
    raise KeyError(f"No time column in {cols}")



def targets_for_interval(interval: str, exec_cadence: str, horizons_min: List[int], mtf_data: Dict[str, pd.DataFrame] = None) -> tuple[List[str], set]:
    """
    If interval matches exec cadence, discover ALL available targets in the data.
    Otherwise fall back to INTERVAL_TO_TARGET (single target).
    """
    if interval == exec_cadence:
        if mtf_data:
            # Discover all available targets in the data
            sample_symbol = list(mtf_data.keys())[0]
            all_targets = [col for col in mtf_data[sample_symbol].columns 
                          if (col.startswith('fwd_ret_') or 
                              col.startswith('will_peak') or 
                              col.startswith('will_valley') or
                              col.startswith('y_will_') or
                              col.startswith('y_first_touch') or
                              col.startswith('p_up') or
                              col.startswith('p_down') or
                              col.startswith('mfe') or
                              col.startswith('mdd'))]
            
            # Count how many symbols have each target
            target_counts = {}
            for target in all_targets:
                count = sum(1 for df in mtf_data.values() if target in df.columns)
                target_counts[target] = count
            
            # Sort by count (most common first) then alphabetically
            sorted_targets = sorted(target_counts.items(), key=lambda x: (-x[1], x[0]))
            common_targets = [target for target, count in sorted_targets]
            
            logger.info(f"🎯 Discovered {len(common_targets)} targets in data:")
            for target, count in sorted_targets[:10]:  # Show top 10
                logger.info(f"  {target}: {count}/{len(mtf_data)} symbols")
            if len(sorted_targets) > 10:
                logger.info(f"  ... and {len(sorted_targets) - 10} more targets")
            
            return common_targets, set(all_targets)
        else:
            # Fallback to horizons_min if no data provided
            fallback_targets = [f"fwd_ret_{h}m" for h in horizons_min]
            return fallback_targets, set(fallback_targets)
    if interval in INTERVAL_TO_TARGET:
        single_target = [INTERVAL_TO_TARGET[interval]]
        return single_target, set(single_target)
    raise KeyError(f"No target mapping for interval '{interval}'. "
                   f"Either run with --exec-cadence={interval} or populate INTERVAL_TO_TARGET.")

def cs_transform_live(df_snapshot: pd.DataFrame, feature_cols: List[str], p=0.01, ddof=1, method="quantile"):
    """
    Apply cross-sectional transforms to live data snapshot.
    This replicates the per-timestamp winsorize + z-score from training.
    Uses the same parameters (p, ddof, method) as saved in model metadata for parity.
    """
    available_cols = [c for c in feature_cols if c in df_snapshot.columns]
    if not available_cols:
        return df_snapshot
    
    s = df_snapshot[available_cols]
    
    if method.lower() == "ksigma":
        # k-sigma winsorization (matches Polars k-sigma path)
        import math
        try:
            from scipy.stats import norm
            k = float(norm.ppf(1 - p))
        except Exception:
            k = 2.33 if abs(p - 0.01) < 1e-6 else 2.0
        
        mu = s.mean()
        sd = s.std(ddof=ddof)
        lo = mu - k * sd
        hi = mu + k * sd
        s = s.clip(lo, hi, axis=1)
        df_snapshot[available_cols] = (s - mu) / (sd + 1e-8)
    else:
        # quantile winsorization (

