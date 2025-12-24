# MIT License - see LICENSE file

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


"""Data loading and preparation utilities for specialized models."""

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

def _load_mtf_data_pandas(data_dir: str, symbols: List[str], interval: str = "5m", max_rows_per_symbol: int = None) -> Dict[str, pd.DataFrame]:
    """Original pandas-based MTF loading."""
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
            df = pd.read_parquet(file_path)
            # Normalize timestamp dtype immediately
            tcol = resolve_time_col(df)
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            if df[tcol].isna().any():
                df = df.dropna(subset=[tcol])
            
            # Apply row limit if specified (keep most recent data)
            if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                df = df.tail(max_rows_per_symbol)
                logger.info(f"📊 Limited {symbol} to {len(df):,} rows (most recent)")
            
            mtf_data[symbol] = df
            logger.info(f"Loaded {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    return mtf_data

def _resolve_time_col_polars(cols):
    """Resolve time column name for Polars."""
    for c in ("ts","timestamp","time","datetime","ts_pred"):
        if c in cols:
            return c
    raise KeyError(f"No time column in {cols}")

def get_common_feature_columns(mtf_data: Dict[str, pd.DataFrame]) -> List[str]:
    """Get common feature columns across all symbols."""
    if not USE_POLARS:
        return _get_common_feature_columns_pandas(mtf_data)
    
    if not mtf_data:
        return []

    PROBLEMATIC = {'fractal_high','fractal_low','cmf_ema','chaikin_oscillator',
                   'ad_line_ema','williams_accumulation','ad_line','order_flow_imbalance'}
    # Detect time col from first item
    first = next(iter(mtf_data.values()))
    time_col = resolve_time_col(first)
    leak_cols = {'ts','timestamp','time','date','symbol','interval','source','ticker','id','index', time_col}

    # Intersect names across symbols using Polars schemas
    name_sets = []
    for df in mtf_data.values():
        name_sets.append(set(df.columns))
    common = set.intersection(*name_sets)

    # Exclude target columns (forward returns)
    target_cols = {col for col in first.columns if col.startswith('fwd_ret_') or col.startswith('will_peak') or col.startswith('will_valley')}
    
    # Keep numeric/bool, drop leaks/problematic/targets
    keep = [c for c in common
            if c not in leak_cols and c not in PROBLEMATIC and c not in target_cols
            and (pd.api.types.is_numeric_dtype(first[c]) or pd.api.types.is_bool_dtype(first[c]))]
    keep.sort()

    # Compute global NaN rates with pandas (NaN + None) - Polars null_count() only counts None
    counts = {c: 0 for c in keep}
    total = 0
    for df in mtf_data.values():
        s = df[keep]
        for c in keep:
            counts[c] += int(s[c].isna().sum())
        total += len(s)
    good = [c for c in keep if (counts[c] / max(1, total)) <= 0.30]
    logger.info(f"Found {len(good)} common features across {len(mtf_data)} symbols (filtered {len(keep)-len(good)} high-NaN)")
    return good

def _get_common_feature_columns_pandas(mtf_data: Dict[str, pd.DataFrame]) -> List[str]:
    """Original pandas-based common feature discovery."""
    # Define problematic indicators that have high NaN rates due to lookback/lookahead requirements
    PROBLEMATIC_INDICATORS = {
        'fractal_high', 'fractal_low',  # Require lookahead (shift(-1), shift(-2))
        'cmf_ema', 'chaikin_oscillator', 'ad_line_ema',  # Require long warmup periods
        'williams_accumulation', 'ad_line', 'order_flow_imbalance'  # Volume-based with warmup
    }
    
    # Get feature columns for each symbol
    # Detect time column to exclude it from features
    if not mtf_data:
        return []
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    
    # Comprehensive leak protection - exclude common time/symbol leakage columns
    leak_cols = {'ts', 'timestamp', 'time', 'date', 'symbol', 'interval', 'source', 'ticker', 'id', 'index', 'Unnamed: 0'}
    leak_cols.add(time_col)  # Add the detected time column
    
    all_feature_cols = []
    for symbol, df in mtf_data.items():
        feature_cols = [col for col in df.columns 
                       if not col.startswith('fwd_ret') and 
                       not col.startswith('will_peak') and 
                       not col.startswith('will_valley') and
                       col not in leak_cols and  # Comprehensive leak protection
                       col not in PROBLEMATIC_INDICATORS and  # Filter out problematic indicators
                       (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))]
        all_feature_cols.append(set(feature_cols))
    
    # Find intersection of all feature sets
    if not all_feature_cols:
        return []
    
    common_features = all_feature_cols[0]
    for feature_set in all_feature_cols[1:]:
        common_features = common_features.intersection(feature_set)
    
    common_features = sorted(list(common_features))
    
    # Filter out features with >30% NaN across the union (streaming approach)
    if common_features:
        # Compute NaN rates streaming per symbol to avoid huge concat
        counts = pd.Series(0, index=common_features, dtype="int64")
        nrows = 0
        for df in mtf_data.values():
            s = df[common_features]
            counts += s.isnull().sum()
            nrows += len(s)
        nan_rates = counts / max(1, nrows)
        good_features = [col for col in common_features if nan_rates[col] <= 0.3]
        
        if len(good_features) < len(common_features):
            logger.info(f"Filtered out {len(common_features) - len(good_features)} features with >30% NaN")
            common_features = good_features
    
    logger.info(f"Found {len(common_features)} common features across {len(mtf_data)} symbols")
    
    return common_features

def load_global_feature_list(feature_list_path: str) -> List[str]:
    """Load global feature list from JSON file."""
    import json
    with open(feature_list_path, 'r') as f:
        return json.load(f)

def save_global_feature_list(features: List[str], output_path: str = "features_all.json"):
    """Save global feature list to JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    logger.info(f"Saved global feature list to {output_path}")

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
        # quantile winsorization (default, matches pandas path)
        lo = s.quantile(p)
        hi = s.quantile(1-p)
        s = s.clip(lo, hi, axis=1)
        df_snapshot[available_cols] = (s - s.mean()) / (s.std(ddof=ddof) + 1e-8)
    
    return df_snapshot

def prepare_sequence_cs(mtf_data, feature_cols, target_cols, lookback=64, min_cs=10,
                        val_start_ts=None, train_ratio=0.8):
    """
    Build temporal sequences for neural network training.
    Creates (batch, lookback, features) sequences with causal structure.
    """
    # 1) detect time col
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    SYM = SYMBOL_COL

    # 2) concat (keep only time, sym, feats, targets)
    dfs = []
    for s, df in mtf_data.items():
        have = [c for c in feature_cols if c in df.columns]
        t_have = [t for t in target_cols if t in df.columns]
        if not t_have: 
            continue
        d = df[[time_col, *have, *t_have]].copy()
        d[SYM] = s
        d = d.sort_values(time_col, kind="mergesort").dropna(subset=t_have)
        dfs.append(d)
    if not dfs:
        raise ValueError("no data for requested targets")

    big = pd.concat(dfs, ignore_index=True)

    # 3) keep timestamps with enough cross-section (so CS eval still makes sense)
    counts = big.groupby(time_col)[SYM].nunique()
    keep_ts = counts[counts >= min_cs].index
    big = big[big[time_col].isin(keep_ts)].copy()

    # 4) build rolling sequences per-symbol (causal: last step is "t", predicts targets at "t")
    Xseq, ymat, ts_last = [], [], []
    for s, g in big.groupby(SYM, sort=False):
        g = g.sort_values(time_col, kind="mergesort")
        F = g[feature_cols].to_numpy(dtype=np.float32)
        T = g[target_cols].to_numpy(dtype=np.float32)
        ts = g[time_col].to_numpy()
        if len(g) < lookback: 
            continue
        for i in range(lookback, len(g)):  # window [i-lookback, i)
            Xseq.append(F[i-lookback:i])
            ymat.append(T[i])              # predict forward returns anchored at time ts[i]
            ts_last.append(ts[i])
    Xseq = np.stack(Xseq)           # (N, L, F)
    ymat = np.stack(ymat)           # (N, n_tasks)
    ts_last = np.asarray(ts_last)   # (N,)

    # 5) time-aware split (no leakage)
    if val_start_ts is not None:
        try:
            cut = np.datetime64(val_start_ts)
        except Exception:
            cut = val_start_ts  # numeric timestamps still ok
        tr_idx = ts_last < cut
        va_idx = ~tr_idx
        train_ts = set(np.unique(ts_last[tr_idx]))
        val_ts   = set(np.unique(ts_last[va_idx]))
    else:
        tr_idx, va_idx, train_ts, val_ts = create_time_aware_split(ts_last, train_ratio=train_ratio)

    pack = {
        "X_tr": Xseq[tr_idx], "X_va": Xseq[va_idx],
        "y_tr": ymat[tr_idx], "y_va": ymat[va_idx],
        "ts_tr": ts_last[tr_idx], "ts_va": ts_last[va_idx],
        "feature_cols": list(feature_cols),
        "task_names": list(target_cols),
        "lookback": lookback, "time_col": time_col,
        "train_ts": train_ts, "val_ts": val_ts
    }
    return pack

def _pick_one(colnames, base):
    """
    Pick exactly one column matching the base name (exact preferred, else shortest suffix).
    
    ✅ ENHANCED: Better error handling for ambiguous targets to prevent tolist() crashes.
    """
    # Exact match first (highest priority)
    if base in colnames:
        return base
    
    # Look for prefix matches
    candidates = [c for c in colnames if c.startswith(base + "_")]
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Multiple candidates - this causes the tolist() crash!
    logger.error(f"❌ AMBIGUOUS TARGET '{base}': {len(candidates)} matches found")
    logger.error(f"   Candidates: {candidates}")
    logger.error(f"   This will cause tolist() crash when df['{base}'] returns 2-D DataFrame")
    logger.error(f"   Solution: Use exact column name or fix duplicate columns")
    
    # Fail fast instead of silently picking first
    raise ValueError(f"Ambiguous target '{base}': {len(candidates)} matches found: {candidates}. Use exact column name.")

def strip_targets(cols, all_targets=None):
    """
    Remove ALL target-like columns from feature list.
    
    Args:
        cols: List of column names
        all_targets: Set of all discovered target columns (if None, uses heuristics)
    
    Returns:
        List of feature columns only (no targets, no symbol/timestamp)
    """
    if all_targets is None:
        # Fallback to heuristics if all_targets not provided
        EXCLUDE_PREFIXES = ("fwd_ret_", "will_peak", "will_valley", "mdd_", "mfe_", "y_will_")
        return [c for c in cols if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in ("symbol", "timestamp")]
    else:
        # Use explicit target list for precise filtering
        return [c for c in cols if c not in all_targets and c not in ("symbol", "timestamp")]

def collapse_identical_duplicate_columns(df):
    """
    Collapse identical duplicate columns, raise error if conflicting.
    
    Args:
        df: DataFrame with potentially duplicate columns
        
    Returns:
        DataFrame with unique columns, duplicates removed
    """
    if len(df.columns) == len(set(df.columns)):
        return df  # No duplicates
    
    # Group columns by name
    from collections import defaultdict
    col_groups = defaultdict(list)
    for i, col in enumerate(df.columns):
        col_groups[col].append(i)
    
    # Check for conflicts and remove exact duplicates
    new_cols = []
    for col_name, indices in col_groups.items():
        if len(indices) == 1:
            new_cols.append(col_name)
        else:
            # Multiple columns with same name - check if identical
            first_idx = indices[0]
            is_identical = all(df.iloc[:, idx].equals(df.iloc[:, first_idx]) for idx in indices[1:])
            
            if is_identical:
                # Keep one copy
                new_cols.append(col_name)
            else:
                # Conflicting data - this is the root cause of our crashes
                logger.error(f"❌ CONFLICTING DUPLICATE COLUMNS: '{col_name}' has {len(indices)} copies with different data")
                logger.error(f"   This will cause tolist() crashes when df['{col_name}'] returns 2-D DataFrame")
                raise ValueError(f"Conflicting duplicate columns found: '{col_name}' has {len(indices)} copies with different data")
    
    return df[new_cols]

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
                def _unit_from_span(v: int) -> str:
                    if v >= 1_000_000_000_000_000_000: return "ns"
                    if v >= 1_000_000_000_000_000:     return "us"
                    if v >= 1_000_000_000_000:         return "ms"
                    return "s"
                unit = _unit_from_span(span)
                lf = lf.with_columns(pl.from_epoch(pl.col(time_col).cast(pl.Int64), time_unit=unit).alias(time_col))
            elif lf[time_col].dtype == pl.Datetime:
                lf = lf.with_columns(pl.col(time_col).dt.cast_time_unit("us").alias(time_col))
            else:
                lf = lf.with_columns(pl.col(time_col).str.strptime(pl.Datetime, strict=False).alias(time_col))

            # Debug: check timestamp range before filtering
            try:
                ts_epoch = lf.select(pl.col(time_col).dt.epoch("us").alias("__t_us"))
                ts_min = ts_epoch.select(pl.col("__t_us").min()).item()
                ts_max = ts_epoch.select(pl.col("__t_us").max()).item()
                logger.info(f"Symbol {sym}: Timestamp range: {ts_min} to {ts_max} (μs)")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not check timestamp range: {e}")
                ts_min = 0
                ts_max = 0
            
            # Clamp timestamps to [1970, 2100] (μs) and add symbol
            MIN_US = 0
            MAX_US = 4102444800000000  # 2100-01-01 UTC in μs
            # But if timestamps are in nanoseconds, adjust the range
            if ts_min > 1_000_000_000_000_000_000:  # If timestamps are in nanoseconds
                MIN_US = MIN_US * 1000  # Convert to nanoseconds
                MAX_US = MAX_US * 1000  # Convert to nanoseconds
                logger.info(f"Symbol {sym}: Adjusted filter range for nanoseconds: {MIN_US} to {MAX_US}")
            else:
                logger.info(f"Symbol {sym}: Using microsecond filter range: {MIN_US} to {MAX_US}")
            
            lf = (
                lf.with_columns(pl.col(time_col).dt.epoch("us").alias("__t_us"))
                  .filter(pl.col("__t_us").is_between(MIN_US, MAX_US))
                  .with_columns(pl.from_epoch(pl.col("__t_us"), time_unit="us").alias(time_col))
                  .drop(["__t_us"])
                  .with_columns(pl.lit(sym).alias("symbol"))
            )
            
            # Debug: check target column after timestamp processing
            try:
                target_after_ts = lf.select(pl.col(target).count()).item()
                target_nulls_after_ts = lf.select(pl.col(target).null_count()).item()
                logger.info(f"Symbol {sym}: After timestamp processing - target count: {target_after_ts}, nulls: {target_nulls_after_ts}")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not check target after timestamp processing: {e}")

            # ✅ Robust target filter: cast to float and keep *finite* values
            # Debug: check target before filtering
            try:
                target_before = lf.select(pl.col(target).count()).item()
                target_nulls = lf.select(pl.col(target).null_count()).item()
                logger.info(f"Symbol {sym}: Before finite filter - total: {target_before}, nulls: {target_nulls}")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not get target debug info: {e}")
            
            t = pl.col(target).cast(pl.Float64, strict=False).alias(target)
            finite_mask = t.is_not_null() & t.is_finite()
            lf = lf.with_columns(t).filter(finite_mask)

            if lf.height == 0:
                logger.warning(f"Symbol {sym}: all rows dropped after finite target filter; schema={lf.schema}")
                # Debug: check what happened to the target values
                try:
                    # ✅ FIX: Safe target extraction to prevent tolist() crash on 2-D DataFrames
                    from target_resolver import safe_target_extraction
                    target_series, actual_col = safe_target_extraction(df_subset, target)
                    logger.info("Sample target values (raw pandas): %s", target_series.astype("string").head(5).tolist())
                    # Check if the issue is with the cast
                    import numpy as np
                    test_cast = target_series.astype('float64')
                    logger.info(f"Pandas cast to float64: {test_cast.head(5).tolist()}")
                    logger.info(f"Pandas finite check: {np.isfinite(test_cast).sum()} / {len(test_cast)}")
                except Exception as e:
                    logger.warning(f"Debug failed: {e}")
                continue

            lfs.append(lf)
            processed_symbols += 1
        except Exception as e:
            logger.warning(f"Symbol {sym}: Failed to process datetime column: {e}")
            continue

    logger.info(f"Processed {processed_symbols} symbols with target {target}")
    if not lfs:
        logger.error("No data with target present across symbols")
        return None, None, None, None, None, None

    # Vertical concat = one target column, not one per symbol
    big = pl.concat(lfs, how="vertical")
    logger.info(f"After concat: {big.shape}")

    # Filter timestamps with enough cross-section
    cs_counts = big.group_by(time_col).agg(pl.col("symbol").n_unique().alias("_n"))
    big = (big.join(cs_counts, on=time_col, how="left")
              .filter(pl.col("_n") >= min_cs)
              .drop("_n")
              .sort(time_col))
    logger.info(f"After CS filter: {big.shape}")

    # Sort by time (finite target filtering already done per-symbol)
    big = big.sort(time_col)
    logger.info(f"After sort: {big.shape}")

    # Timestamps are already sanitized per-symbol, no need to re-sanitize
    logger.info(f"Final big shape: {big.shape}")

    # Decide time cut using epoch microseconds (no Python datetime conversion)
    ts_us = big.select(pl.col(time_col).dt.epoch("us").alias("t")).unique().sort("t")["t"]
    logger.info(f"Found {len(ts_us)} unique timestamps")
    if len(ts_us) < 2:
        logger.error(f"Not enough distinct timestamps: {len(ts_us)}")
        return None, None, None, None, None, None
    
    cut_idx = int(0.8 * len(ts_us))
    t_cut_us = ts_us[cut_idx]
    
    def _collect(ldf):
        # Check if it's a LazyFrame or DataFrame
        if hasattr(ldf, 'collect'):
            # LazyFrame
            return (ldf.with_columns([pl.col(c).cast(pl.Float32) for c in common_features + [target]])
                        .collect(streaming=True)
                        .to_pandas(use_pyarrow_extension_array=False))
        else:
            # DataFrame
            return (ldf.with_columns([pl.col(c).cast(pl.Float32) for c in common_features + [target]])
                        .to_pandas(use_pyarrow_extension_array=False))
    
    pdf_tr = _collect(big.filter(pl.col(time_col).dt.epoch("us") < t_cut_us))
    pdf_va = _collect(big.filter(pl.col(time_col).dt.epoch("us") >= t_cut_us))
    
    # stitch back if your caller still expects a single arrays pack
    pdf = pd.concat([pdf_tr, pdf_va], ignore_index=True)
    
    if pdf.empty:
        logger.error("No timestamps with sufficient cross-section after filtering")
        return None, None, None, None, None, None
    
    # Sanity checks to add (cheap, helpful)
    logger.info(
        "Final target health: not-null ratio=%.3f, min=%s, max=%s",
        float(pd.notna(pdf[target]).mean()),
        pd.Series(pdf[target]).min(skipna=True),
        pd.Series(pdf[target]).max(skipna=True),
    )

    # Final min_cs check (cheap)
    counts = pdf.groupby(time_col)["symbol"].nunique()
    keep_ts = counts[counts >= min_cs].index
    pdf = pdf[pdf[time_col].isin(keep_ts)]

    # Features - strip targets and make unique
    common_features = strip_targets(list(dict.fromkeys(common_features)))
    feat_cols = [c for c in common_features if c in pdf.columns and pd.api.types.is_numeric_dtype(pdf[c])]
    leak_cols = {'ts','timestamp','time','date','symbol','interval','source','ticker','id','index', time_col}
    feat_cols = [c for c in feat_cols if c not in leak_cols]
    if not feat_cols:
        logger.error("No numeric common features present after leak protection")
        return None, None, None, None, None, None

    # Arrays for learners
    import numpy as np
    X = pdf[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = pdf[target].to_numpy(dtype=np.float32, copy=False)
    symbols = pdf["symbol"].astype('U32').to_numpy()
    ts_index = pdf[time_col]  # Pandas Series is fine for your split code

    # Group sizes (RLE on sorted timestamps)
    sizes = (pdf[time_col].ne(pdf[time_col].shift()).cumsum()
             .groupby(pdf[time_col].ne(pdf[time_col].shift()).cumsum()).size().tolist())

    logger.info(f"✅ CS data: rows={len(ts_index)}, features={len(feat_cols)}, times={len(sizes)}")
    return X, y, symbols, sizes, ts_index, feat_cols

def _prepare_training_data_cross_sectional_pandas(
    mtf_data: Dict[str, pd.DataFrame],
    target: str,
    common_features: List[str],
    min_cs: int,
    max_samples_per_symbol: int = None  # ignore per-symbol sampling for CS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.Series, List[str]]:
    """Original pandas-based CS dataset assembly."""
    logger.info("🎯 Building TRUE cross-sectional training data...")

    # 1) Detect time column from first dataframe
    if not mtf_data:
        logger.error("No data provided")
        return None, None, None, None, None, None
    first_df = next(iter(mtf_data.values()))
    time_col = resolve_time_col(first_df)
    logger.info(f"Using time column: {time_col}")

    # 2) concat all symbols
    dfs = []
    for sym, df in mtf_data.items():
        if target not in df.columns:
            logger.warning(f"Target {target} not found in {sym}, skipping")
            continue
        
        # Verify time column exists in this symbol
        if time_col not in df.columns:
            logger.warning(f"Time column {time_col} not found in {sym}, skipping")
            continue
            
        d = df[[time_col, target] + [c for c in common_features if c in df.columns]].copy()
        d[SYMBOL_COL] = sym
        d = d.dropna(subset=[target])        # targets must be finite
        dfs.append(d)
    
    if not dfs:
        logger.error("No data with target present across symbols")
        return None, None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)
    
    # Convert symbol column to category for memory efficiency
    df[SYMBOL_COL] = df[SYMBOL_COL].astype('category')
    
    logger.info(f"Concatenated data shape: {df.shape}")
    logger.info(f"Concatenated columns: {len(df.columns)}")
    
    # Remove duplicate columns if any
    if df.columns.duplicated().any():
        logger.warning(f"Removing duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # CRITICAL: Apply row capping to prevent OOM
    MAX_ROWS_PER_BATCH = int(os.environ.get("MAX_ROWS_PER_BATCH", "8000000"))  # 8M rows max
    if len(df) > MAX_ROWS_PER_BATCH:
        logger.warning(f"⚠️  Dataset too large ({len(df)} rows), capping to {MAX_ROWS_PER_BATCH} rows")
        # Sample uniformly by timestamp to preserve cross-sectional structure
        unique_ts = df[time_col].unique()
        avg_per_ts = max(1, len(df) // max(1, len(unique_ts)))
        n_ts_needed = max(1, MAX_ROWS_PER_BATCH // avg_per_ts)
        if n_ts_needed < len(unique_ts):
            rng = np.random.default_rng(42)  # Deterministic sampling
            selected_ts = rng.choice(unique_ts, size=n_ts_needed, replace=False)
            df = df[df[time_col].isin(selected_ts)].copy()
            logger.info(f"📊 After capping: {len(df)} rows, {len(df[time_col].unique())} timestamps")
        logger.info(f"📊 Final dataset shape: {df.shape}")
    # enforce numeric features
    feat_cols = [c for c in common_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    # Final leak protection - exclude common time/symbol columns
    leak_cols = {'ts', 'timestamp', 'time', 'date', 'symbol', 'interval', 'source', 'ticker', 'id', 'index'}
    leak_cols.add(time_col)  # Add the detected time column
    feat_cols = [c for c in feat_cols if c not in leak_cols]
    
    if not feat_cols:
        logger.error("No numeric common features present after leak protection")
        return None, None, None, None, None, None

    # 2) keep timestamps with enough symbols (CS)
    # Ensure we get a Series, not DataFrame
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]
    counts = df.groupby(time_col)[SYMBOL_COL].nunique()
    keep_ts = counts[counts >= min_cs].index
    df = df[time_series.isin(keep_ts)]
    if df.empty:
        logger.error("No timestamps with sufficient cross-section")
        return None, None, None, None, None, None

    # 3) sort by time - NO preprocessing yet (avoid leakage)
    df = df.sort_values(time_col, kind="mergesort")
    # Update time_series after filtering and sorting
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]

    # 4) remove rows with NaN targets / infinities
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    
    # Convert targets to float32 for memory efficiency
    df[target] = df[target].astype(np.float32)
    
    # Refresh time_series after any row filtering
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]
    
    # 5) Final min_cs check after all filtering operations
    counts = df.groupby(time_col)[SYMBOL_COL].nunique()
    keep_ts = counts[counts >= min_cs].index
    df = df[df[time_col].isin(keep_ts)]
    if df.empty:
        logger.error("No timestamps with sufficient cross-section after all filtering")
        return None, None, None, None, None, None

    # Refresh time_series after final filtering to ensure alignment
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]

    # 6) build groups array efficiently (rows per timestamp)
    logger.info("🔧 Building group sizes...")
    grp_sizes = df.groupby(time_col).size().tolist()
    ts_index = time_series

    # 6) Assert group integrity
    n = len(df)
    assert sum(grp_sizes) == n, f"group size mismatch {sum(grp_sizes)} != {n}"
    assert min(grp_sizes) >= min_cs, f"min CS {min(grp_sizes)} < min_cs={min_cs}"

    # 7) assemble arrays with memory optimization
    logger.info("🔧 Assembling feature matrix (memory-safe)...")
    
    # Memory monitoring
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        logger.info(f"💾 RSS before assemble: {rss:.1f} GiB")
    except Exception:
        pass
    
    # compact encodings first
    df[SYMBOL_COL] = df[SYMBOL_COL].astype('category')
    
    # cheap views for ids & time
    symbols = df[SYMBOL_COL].cat.codes.to_numpy(dtype=np.int32, copy=False)
    # Safe datetime handling - convert nanoseconds to datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        # Handle nanoseconds (divide by 1000 to get microseconds)
        if df[time_col].dtype == 'int64' and df[time_col].max() > 1e15:  # Likely nanoseconds
            df[time_col] = pd.to_datetime(df[time_col] / 1000, unit='us', errors="coerce")
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if getattr(df[time_col].dt, "tz", None) is not None:
        df[time_col] = df[time_col].dt.tz_convert(None)
    
    # Filter out invalid timestamps (year > 2100 or < 1900)
    valid_years = (df[time_col].dt.year >= 1900) & (df[time_col].dt.year <= 2100)
    df = df[valid_years].copy()
    
    ts_index = df[time_col].astype('int64', copy=False).to_numpy(copy=False)
    
    # one extraction, no temporary float64 frames
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[target].to_numpy(dtype=np.float32, copy=False)
    
    # spill to file if huge (keeps peak RAM flat)
    if int(os.getenv("USE_MEMMAP", "1")) and X.nbytes > 8_000_000_000:  # ~8 GB
        mm_dir = Path(os.getenv("MEMMAP_DIR", "/tmp/mmap"))
        mm_dir.mkdir(parents=True, exist_ok=True)
        mm_path = mm_dir / f"X_{X.shape[0]}x{X.shape[1]}.f32.mmap"
        X_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=X.shape)
        X_mm[:] = X  # one pass copy
        X = X_mm
    
    # convert back to original format for compatibility
    ts = df[time_col].to_numpy()
    
    # free the big frame promptly
    del df
    import gc; gc.collect()
    
    # Memory monitoring after assembly
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        logger.info(f"💾 RSS after assemble: {rss:.1f} GiB")
    except Exception:
        pass
    
    # Additional memory cleanup for 10M rows (optional)
    try:
        import psutil
        process = psutil.Process()
        logger.info(f"Memory before cleanup: {process.memory_info().rss / 1024**3:.1f} GB")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        logger.info(f"Memory after cleanup: {process.memory_info().rss / 1024**3:.1f} GB")
    except ImportError:
        # psutil not available, just do basic cleanup
        for _ in range(3):
            gc.collect()
        logger.info("🧹 Basic memory cleanup completed (psutil not available)")

    # Memory management is handled by symbol batching (--batch-size)
    # No need for additional row capping that would throw away data
    logger.info(f"📊 Dataset loaded: {len(ts_index)} rows, {len(pd.unique(ts_index))} timestamps")
    
    # Note: max_samples_per_symbol is ignored in cross-sectional mode
    # (we use all available data per timestamp for CS training)

    # 9) Log group size distribution
    g = np.array(grp_sizes)
    logger.info(f"✅ CS data: rows={len(ts_index)}, features={len(feat_cols)}, times={len(grp_sizes)}")
    logger.info(f"📊 CS group sizes mean={g.mean():.1f}, p5={np.percentile(g,5):.0f}, p95={np.percentile(g,95):.0f}")
    

    return X, y, symbols, grp_sizes, ts_index, feat_cols


