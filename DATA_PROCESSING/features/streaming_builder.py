#!/usr/bin/env python3

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
Streaming Feature Builder using Polars Lazy API
Handles large datasets without materializing everything in memory
"""

import polars as pl
import polars.selectors as cs
import yaml
import json
import numpy as np
import sys
from pathlib import Path
import argparse
from typing import List, Dict, Any
import logging
import gc
import time
import psutil
from enum import Enum
from collections import Counter
import re
import glob
from datetime import time, datetime
import pandas as pd
import exchange_calendars as xc

def _to_i8_array(x):
    """
    Return int64 nanoseconds since epoch as a NumPy array from:
      - pandas Series / Index with datetime64[ns] (tz-aware or naive)
      - numpy arrays of datetime64[ns]
      - plain arrays of ints/datetimes
    """
    # pandas objects
    if hasattr(x, "dtype"):
        # If it's pandas datetime-like, cast to int64 ns safely
        try:
            if pd.api.types.is_datetime64_any_dtype(x):
                return x.astype("int64").to_numpy()
        except Exception:
            pass
        # Generic pandas object → numpy int64
        return np.asarray(x).astype("int64")
    # already numpy or list
    arr = np.asarray(x)
    # if datetime64 ns, view as int64; else cast
    return arr.view("int64") if np.issubdtype(arr.dtype, np.datetime64) else arr.astype("int64")

def normalize_bars(df: pl.DataFrame, symbol: str, interval: str = "15m", 
                   source: str = "prod", adjusted: bool = False,
                   config: dict = None) -> pl.DataFrame:
    """
    Unified bar normalization for all intervals (5m, 15m, 30m, 1h, 1d).
    Ensures consistent schema and types for feature building.
    """
    if config is None:
        config = {}
    
    # 1) Timestamps: ensure timezone-aware UTC Datetime
    ts_schema = df.schema.get("ts")
    if ts_schema != pl.Datetime("ns", "UTC"):
        # Coerce to UTC datetime regardless of input type
        if ts_schema == pl.Utf8:
            # String column - parse as datetime
            df = df.with_columns([
                pl.col("ts").str.strptime(pl.Datetime, strict=False)
                .dt.replace_time_zone("UTC")
                .alias("ts")
            ])
        else:
            # Other types - cast to datetime and ensure UTC timezone
            df = df.with_columns([
                pl.col("ts").cast(pl.Datetime)
                .dt.replace_time_zone("UTC")
                .alias("ts")
            ])

    # 2) Ensure symbol column is present & typed
    if "symbol" not in df.columns:
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
    else:
        df = df.with_columns(pl.col("symbol").cast(pl.Utf8))

    # 3) Core OHLCV columns - ensure proper types
    core_columns = ["open", "high", "low", "close", "volume"]
    for col in core_columns:
        if col in df.columns:
            if col == "volume":
                df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Float64))

    # 4) VWAP calculation (OHLC4) if missing
    if "vwap" not in df.columns:
        vwap_expr = (pl.col("open") + pl.col("high") + pl.col("low") + pl.col("close")) / 4.0
        df = df.with_columns(
            pl.when(pl.col("volume") > 0).then(vwap_expr).otherwise(None)
              .cast(pl.Float64).alias("vwap")
        )

    # 5) Optional metadata columns with defaults
    optional_columns = {
        "n_trades": pl.lit(None, dtype=pl.Int64),
        "adjusted": pl.lit(adjusted).cast(pl.Boolean),
        "interval": pl.lit(interval).cast(pl.Utf8),
        "source": pl.lit(source).cast(pl.Utf8),
        "is_ext_hours": pl.lit(False).cast(pl.Boolean),
    }
    
    for col, expr in optional_columns.items():
        if col not in df.columns:
            df = df.with_columns(expr.alias(col))
        else:
            # Ensure proper type
            if col == "n_trades":
                df = df.with_columns(pl.col(col).cast(pl.Int64, strict=False))
            elif col == "adjusted" or col == "is_ext_hours":
                df = df.with_columns(pl.col(col).cast(pl.Boolean))
            else:
                df = df.with_columns(pl.col(col).cast(pl.Utf8))

    # 6) Add dollar volume for consistency
    if "dollar_volume" not in df.columns:
        df = df.with_columns(
            (pl.col("close") * pl.col("volume")).cast(pl.Float64).alias("dollar_volume")
        )

    # 7) Deterministic column ordering
    core_order = ["ts", "symbol", "open", "high", "low", "close", "volume", "vwap"]
    optional_order = ["n_trades", "adjusted", "interval", "source", "is_ext_hours", "dollar_volume"]
    
    # Get existing columns in order
    ordered_cols = []
    for col in core_order + optional_order:
        if col in df.columns:
            ordered_cols.append(col)
    
    # Add any remaining columns
    remaining = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining)
    
    return df.select(ordered_cols)

# Add project root to path for centralized utilities
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml.memory import MemoryManager
from scripts.simple_feature_computation import simple_feature_computer
from scripts.logging_manager import CentralLoggingManager
from scripts.io_safe_scan import safe_scan_parquet, write_features_strict, dedupe_symbols

# Timeframe enum to prevent string mismatches
class TF(str, Enum):
    M5 = "5m"
    M15 = "15m" 
    M30 = "30m"
    H1 = "1h"
    D1 = "1d"

# Resample mapping
RESAMPLE_MAP = {
    (TF.M5, TF.M15): "15m",
    (TF.M5, TF.M30): "30m", 
    (TF.M5, TF.H1): "1h",
    (TF.M5, TF.D1): "1d",
    # Identity mappings
    (TF.M15, TF.M15): None,
    (TF.M30, TF.M30): None,
    (TF.H1, TF.H1): None,
    (TF.D1, TF.D1): None,
}

def expect_minutes(tf: TF) -> int:
    """Expected minutes for each timeframe"""
    return {TF.M5: 5, TF.M15: 15, TF.M30: 30, TF.H1: 60, TF.D1: 1440}[tf]

def assert_step(df: pl.DataFrame, minutes: int, tag: str = ""):
    """Assert that dataframe has correct time intervals"""
    if len(df) < 2:
        return
    
    # Convert timestamps to nanoseconds for comparison
    ts_col = df["ts"]
    if ts_col.dtype == pl.Datetime:
        # Convert datetime to nanoseconds
        ts_ns = ts_col.dt.timestamp("ns")
    else:
        ts_ns = ts_col
    
    # Calculate differences in nanoseconds
    diffs = ts_ns.diff().drop_nulls()
    if len(diffs) == 0:
        return
        
    # Convert to minutes
    diffs_minutes = diffs / (60 * 1_000_000_000)  # ns to minutes
    
    # Get actual interval distribution
    actual_intervals = diffs_minutes.to_list()[:10]  # Sample first 10
    interval_counts = Counter([round(x) for x in actual_intervals])
    
    # Check if we have the expected interval
    if minutes not in interval_counts:
        raise AssertionError(f"{tag} Grid != {minutes}m; actual intervals: {dict(interval_counts)}")
    
    # Allow some tolerance for rounding and gaps
    tolerance = 0.1  # 0.1 minute = 6 seconds
    gap_tolerance = 60  # 60 minutes = 1 hour for overnight gaps
    
    for diff in diffs_minutes:
        if abs(diff - minutes) > tolerance and diff < gap_tolerance:
            raise AssertionError(f"{tag} Found {diff:.1f}m interval, expected {minutes}m")
        elif diff >= gap_tolerance:
            # Skip large gaps (overnight, weekends, etc.)
            continue

def dbg(tag: str, df: pl.DataFrame):
    """Debug logging for dataframe state"""
    if len(df) == 0:
        print(f"DBG {tag}: EMPTY")
        return
    
    print(f"DBG {tag}: id={id(df)} rows={len(df)}")
    if "ts" in df.columns:
        ts_sample = df["ts"].head(3).to_list()
        print(f"DBG {tag}: ts_sample={ts_sample}")
        
        # Calculate time differences
        if len(df) > 1:
            ts_col = df["ts"]
            if ts_col.dtype == pl.Datetime:
                ts_ns = ts_col.dt.timestamp("ns")
            else:
                ts_ns = ts_col
            diffs = ts_ns.diff().drop_nulls()
            if len(diffs) > 0:
                diffs_minutes = diffs / (60 * 1_000_000_000)
                interval_counts = Counter([round(x) for x in diffs_minutes.to_list()[:5]])
                print(f"DBG {tag}: intervals={dict(interval_counts)}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Symbol aliases for disk vs display names
SYMBOL_ALIASES = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
    "BRK-B": "BRK.B",  # reverse mapping
    "BF-B": "BF.B",
}

# --- Eager/Lazy helpers for audits ---
from typing import Union as _Union
Frame = _Union[pl.DataFrame, pl.LazyFrame]

def _nrows(frame: Frame) -> int:
    if isinstance(frame, pl.LazyFrame):
        return int(frame.select(pl.len().alias("n")).collect(streaming=True)["n"][0])
    return int(frame.height)

def _count_where(frame: Frame, predicate: pl.Expr) -> int:
    if isinstance(frame, pl.LazyFrame):
        return int(frame.filter(predicate).select(pl.len().alias("n")).collect(streaming=True)["n"][0])
    return int(frame.filter(predicate).height)

def _min_when(frame: Frame, predicate: pl.Expr, ts_col: str) -> object:
    if isinstance(frame, pl.LazyFrame):
        out = (
            frame.filter(predicate)
                 .select(pl.col(ts_col).min().alias("__ts__"))
                 .collect(streaming=True)["__ts__"]
        )
        return out[0] if len(out) else None
    df = frame.filter(predicate)
    if df.is_empty():
        return None
    return df.select(pl.col(ts_col).min().alias("__ts__")).item()

# --- RTH helpers (UTC schedule-overlap; version-agnostic) ---
def _infer_bar_label(df_idx: pd.DatetimeIndex) -> str:
    try:
        et = df_idx.tz_convert("America/New_York")
        at_open = ((et.hour == 9) & (et.minute == 30)).sum()
        at_close = ((et.hour == 16) & (et.minute == 0)).sum()
        return "right" if at_close >= at_open else "left"
    except Exception:
        return "right"

def _assert_15m_rth_structure(df_idx: pd.DatetimeIndex):
    """Validate 15m RTH structure - expect ~26 bars per normal session."""
    et = df_idx.tz_convert("America/New_York")
    counts = pd.Series(et.date).value_counts()
    if len(counts) == 0:
        raise AssertionError("RTH slice empty")
    med = float(counts.median())
    # 15m bars: 6.5h * 4 bars/hour = 26 bars per normal session
    # Allow some tolerance for early closes and DST transitions
    if not (24.0 <= med <= 28.0):
        raise AssertionError(f"Unexpected 15m RTH median bars/day: {med} (expected ~26)")

def _get_schedule_df(cal, start, end) -> pd.DataFrame:
    """Get schedule DataFrame using instance methods (not class methods)."""
    try:
        # Use instance schedule property, not class method
        sched = cal.schedule.loc[start:end]
        if not isinstance(sched, pd.DataFrame):
            raise TypeError(f"calendar.schedule returned {type(sched).__name__}, expected DataFrame")
        if not {"open","close"}.issubset(set(sched.columns)):
            raise AssertionError("calendar schedule missing open/close columns")
        return sched
    except Exception as e:
        raise RuntimeError(f"Failed to get schedule for {start} to {end}: {e}")

def _calculate_expected_bars_per_session(sched_df: pd.DataFrame, timeframe_minutes: int) -> pd.Series:
    """Calculate expected bars per session based on session duration and timeframe."""
    # Calculate session duration in minutes
    session_duration_min = (sched_df["close"] - sched_df["open"]).dt.total_seconds() / 60
    # Calculate expected bars (with tolerance for clock drift)
    expected_bars = (session_duration_min / timeframe_minutes).round().astype(int)
    return expected_bars

def _rollup_5m_to_15m(df_5m: pl.DataFrame) -> pl.DataFrame:
    """Roll up 5m RTH data to 15m by aggregating 3 consecutive 5m bars."""
    return (
        df_5m
        .with_columns(
            # Anchor to 15m buckets (900 seconds = 15 minutes)
            (pl.col("ts").cast(pl.Int64) // 900 * 900).alias("ts15")
        )
        .group_by("ts15")
        .agg([
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"), 
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            # Keep other columns by taking mean (or could recompute features later)
            *[pl.exclude(["ts", "open", "high", "low", "close", "volume"]).mean()]
        ])
        .with_columns(pl.col("ts15").cast(pl.Datetime("us")).alias("ts"))
        .drop("ts15")
        .sort("ts")
    )

# ---------------- DTYPE GUARDS & HELPERS ----------------
BASE_EXCLUDE_COLS = {
    "ts", "symbol", "open", "high", "low", "close",
    "volume", "volume_float", "vwap", "n_trades",
    "interval", "source", "adjusted", "date",
}

def enforce_base_schema_df(df: pl.DataFrame) -> pl.DataFrame:
    """Enforce base dtypes for critical columns right before write.
    - Keep volume as Int64 (rounded from Float if needed)
    - Keep volume_float as Float64 if present
    - Leave ts as currently used in store (Int64 epoch ns)
    """
    exprs = []
    if "volume_float" in df.columns:
        exprs.append(pl.col("volume_float").cast(pl.Float64, strict=False).alias("volume_float"))
    if "volume" in df.columns:
        exprs.append(
            pl.col("volume")
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False)
            .alias("volume")
        )
    # Ensure symbol as Utf8 (already handled elsewhere too)
    if "symbol" in df.columns:
        exprs.append(pl.col("symbol").cast(pl.Utf8))
    return df.with_columns(exprs) if exprs else df

def downcast_feature_floats_df(df: pl.DataFrame, target_col: str | None = None) -> pl.DataFrame:
    """Downcast float features to Float32, excluding base columns and optional target.
    Idempotent and safe (strict=False).
    """
    exclude = set(BASE_EXCLUDE_COLS)
    if target_col:
        exclude.add(target_col)
    float_feats = [c for c, d in df.schema.items() if c not in exclude and isinstance(d, pl.Float64)]
    if not float_feats:
        return df
    return df.with_columns([pl.col(c).cast(pl.Float32, strict=False) for c in float_feats])

def _schema_of(frame: Frame) -> dict:
    return (frame.schema if isinstance(frame, pl.DataFrame) else frame.collect_schema())

def assert_no_drift(stage_name: str, before_schema: dict, after_frame: Frame, logger: logging.Logger):
    after_schema = _schema_of(after_frame)
    drift = {}
    for col in ("volume", "volume_float"):
        b = before_schema.get(col)
        a = after_schema.get(col)
        if b != a:
            drift[col] = (b, a)
    if drift:
        logger.error(f"[DTYPE DRIFT] {stage_name}: {drift}")

def get_symbol_on_disk(symbol: str) -> str:
    """Get the actual symbol name as stored on disk."""
    return SYMBOL_ALIASES.get(symbol, symbol)

def build_symbol_glob(input_tpl: str, symbol: str) -> str:
    """Turn a template like .../symbol=*/date=*/*.parquet into a concrete glob for one symbol."""
    if "symbol=*" in input_tpl:
        return input_tpl.replace("symbol=*", f"symbol={symbol}")
    if "symbol=" in input_tpl:
        return re.sub(r"symbol=[^/]+", f"symbol={symbol}", input_tpl)
    return input_tpl  # no partition; assume caller gave a per-symbol path

def scan_symbol_lazy(input_tpl: str, sym: str, volume_policy: str = "strict") -> pl.LazyFrame:
    """Scan symbol data using safe parquet scanning with canonical dtypes."""
    try:
        sym_on_disk = get_symbol_on_disk(sym)
        glob = build_symbol_glob(input_tpl, sym_on_disk)
        
        # Load canonical schema
        canonical_schema_path = project_root / "config" / "canonical_schema.yaml"
        with open(canonical_schema_path, 'r') as f:
            schema_config = yaml.safe_load(f)
        
        canonical_dtypes = schema_config["base_dtypes"]
        
        logger.info(f"{sym}: scanning with volume_policy='{volume_policy}'")
        
        # Use safe scanner with canonical dtypes and volume policy
        return safe_scan_parquet(glob, canonical_dtypes, volume_policy=volume_policy)
    except Exception as e:
        logger.error(f"Failed to scan parquet for {sym}: {e}")
        return None

def add_partitions(df: pl.DataFrame, sym: str, interval: str = "1h") -> pl.DataFrame:
    """Add required partition columns for hive-style partitioning."""
    # Handle both datetime and int64 timestamp columns
    if df.schema.get("ts") == pl.Int64:
        # Convert int64 nanoseconds back to datetime for date extraction
        ts_expr = pl.from_epoch(pl.col("ts"), time_unit="ns").dt.convert_time_zone("America/New_York").dt.date().cast(pl.Utf8)
    else:
        # Handle datetime columns
        ts_expr = pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().cast(pl.Utf8)
    
    return (
        df
        .with_columns([
            pl.lit(sym).alias("symbol"),
            pl.lit(interval).alias("interval"),
            ts_expr.alias("date"),
        ])
    )

def preflight_write(df: pl.DataFrame, sym: str):
    """Preflight check before writing to catch zero-row issues."""
    n_rows = df.height
    n_cols = df.width
    if n_rows == 0:
        raise RuntimeError(f"[WRITE-GUARD] ZERO rows for {sym} just before sink")
    req = {"ts","open","high","low","close","volume","symbol","interval","date"}
    missing = req - set(df.columns)
    if missing:
        raise RuntimeError(f"[WRITE-GUARD] Missing required cols {missing} for {sym}")
    logger.info(f"About to write {sym}: rows={n_rows} cols={n_cols}")

def has_sources(input_tpl: str, sym: str) -> bool:
    """Check if parquet sources exist for a symbol to avoid scan errors."""
    pat = re.sub(r"symbol=\*", f"symbol={sym}", input_tpl)
    return len(glob.glob(pat)) > 0

def symbols_from_input_glob(input_tpl: str, universe: list[str]) -> list[str]:
    """Extract symbols to process from input glob pattern."""
    m = re.search(r"symbol=([^/*]+)", input_tpl)
    if m and "*" not in m.group(1):
        # explicit symbol in the path → process only that one
        return [m.group(1)]
    return universe

def extract_timeframe_from_path(input_pattern: str) -> str:
    """Extract timeframe from input pattern (e.g., 'interval=5m' -> '5m')."""
    import re
    match = re.search(r'interval=([^/]+)', input_pattern)
    if match:
        return match.group(1)
    return "1h"  # default fallback

def resample_5m_to_timeframe(df_5m: pl.DataFrame, target_timeframe: str) -> pl.DataFrame:
    """
    Properly resample 5m data to target timeframe (15m, 30m, 1h, 1d).
    
    Args:
        df_5m: 5m data with columns [ts, open, high, low, close, volume, ...]
        target_timeframe: '15m', '30m', '1h', or '1d'
    
    Returns:
        Resampled data with correct intervals
    """
    if df_5m.is_empty():
        return df_5m
    
    # Ensure ts is datetime
    if df_5m.schema.get("ts") != pl.Datetime:
        df_5m = df_5m.with_columns(pl.col("ts").cast(pl.Datetime))
    
    # Get timeframe in minutes
    timeframe_minutes = {
        "15m": 15,
        "30m": 30, 
        "1h": 60,
        "1d": 1440
    }.get(target_timeframe, 15)
    
    # Use Polars' built-in resampling
    df_resampled = df_5m.group_by_dynamic(
        "ts",
        every=f"{timeframe_minutes}m",
        closed="left",
        label="left"
    ).agg([
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        # Handle other columns by taking first
        *[pl.exclude(["ts", "open", "high", "low", "close", "volume"]).first()]
    ]).sort("ts")
    
    return df_resampled

def process_data_by_timeframe(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Process data based on the specific timeframe."""
    # Convert string to enum
    try:
        target_tf = TF(timeframe)
    except ValueError:
        raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of: {[tf.value for tf in TF]}")
    
    # Sort by timestamp
    df = df.sort("ts")

    # Deduplicate rows with same timestamp (keep first)
    df = df.unique(subset=["ts"], keep="first")
    
    dbg("before_process", df)
    
    # For 5m data, the data is already properly filtered and in UTC
    # No additional session filtering needed
    if target_tf == TF.M5:
        dbg("5m_no_resample", df)
        return df
    
    # For other timeframes, we need to resample from 5m data
    # First, check if we have 5m data by looking at time differences
    if len(df) > 1:
        time_diffs = df["ts"].diff().drop_nulls()
        min_diff_seconds = time_diffs.min().total_seconds()
        # Check if we have 5-minute intervals (300 seconds = 5 minutes)
        if min_diff_seconds <= 300:
            # We have higher frequency data, resample to target timeframe
            print(f"DEBUG: Resampling from 5m to {timeframe}")
            df_resampled = resample_5m_to_timeframe(df, timeframe)
            dbg("after_resample", df_resampled)
            # Assert the resampling worked
            assert_step(df_resampled, expect_minutes(target_tf), f"post_resample_{timeframe}")
        else:
            # Data is already at target frequency, just filter for trading hours
            print(f"DEBUG: Data already at {timeframe} frequency, no resampling needed")
            df_resampled = df
    else:
        df_resampled = df

    # For resampled data, skip additional session filtering as it's already RTH-only
    # The 5m data is already properly filtered, so resampled data should be too
    if target_tf != TF.M5:
        # Data is already RTH-only from 5m source, no additional filtering needed
        session_bars = df_resampled
    else:
        # Apply timezone conversion and session filtering for non-resampled data
        ts_dtype = df_resampled.schema.get("ts")
        if ts_dtype == pl.Datetime and hasattr(ts_dtype, 'time_zone') and ts_dtype.time_zone is None:
            # If timezone-naive, assume it's in America/New_York
            df_resampled = df_resampled.with_columns(pl.col("ts").dt.replace_time_zone("America/New_York"))
        elif hasattr(ts_dtype, 'time_zone') and ts_dtype.time_zone == "UTC":
            # Convert from UTC to America/New_York for session filtering
            df_resampled = df_resampled.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York"))

        # Filter for trading hours (9:30 AM to 4:00 PM ET)
        session_bars = df_resampled.filter(
            pl.col("ts").dt.time().is_between(time(9,30), time(16,0), closed="both")
        )

        # Convert to UTC for storage
        session_bars = session_bars.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))
    
    dbg("after_session_filter", session_bars)
    # Final assertion before return
    assert_step(session_bars, expect_minutes(target_tf), f"final_{timeframe}")

    return session_bars

def get_expected_bars_per_day(timeframe: str) -> tuple[int, int]:
    """Get expected min/max bars per day for validation."""
    if timeframe == "1h":
        return 6, 7  # 09:00-15:00 = 6-7 bars
    elif timeframe == "30m":
        return 12, 13  # 09:00-15:00 every 30min = 12-13 bars
    elif timeframe == "15m":
        return 24, 25  # 09:00-15:00 every 15min = 24-25 bars
    elif timeframe == "5m":
        return 75, 80  # 09:30-15:55 every 5min = 78 bars (6.5 hours * 12 bars/hour)
    else:
        return 1, 1000  # Unknown timeframe, be permissive


class StreamingFeatureBuilder:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.engine_config = self.config.get('engine', {})
        self.feature_config = self.config.get('features', {})
        self.io_config = self.config.get('io', {})
        self.output_config = self.config.get('output', {})
        
        # Initialize memory manager for monitoring (singleton semantics in manager)
        self.memory_manager = MemoryManager()
        
        # Track batch scaling
        self._last_memory_check = 0
        self._consecutive_low_usage = 0
        
        # Polars streaming is enabled by default in newer versions
        # No need to explicitly enable it
        
    def build_features(self, input_paths: List[str], output_dir: str, universe_config: str, input_pattern: str = ""):
        """Build features using streaming approach"""
        logger.info(f"Building features for {len(input_paths)} input files")
        logger.info(f"Output directory: {output_dir}")
        
        # Store input pattern for use in processing
        self.input_pattern = input_pattern
        
        # Extract timeframe from config file instead of input pattern
        self.timeframe = self.config.get('resample', {}).get('interval', '15m')
        logger.info(f"Using timeframe from config: {self.timeframe}")
        
        # Load universe config 
        with open(universe_config, 'r') as f:
            universe = yaml.safe_load(f)
        
        universe_symbols = universe.get('universe', [])
        
        # Restrict symbols based on input glob pattern
        symbols = symbols_from_input_glob(input_pattern, universe_symbols)
        
        # Deduplicate symbols to prevent double processing
        symbols = dedupe_symbols(symbols)
        
        logger.info(f"Processing {len(symbols)} symbols (restricted from {len(universe_symbols)} universe)")
        logger.info(f"Symbols to process: {symbols}")
        self._single_symbol_mode = (len(symbols) == 1)
        
        # Create output directory with timeframe
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each symbol separately to avoid memory issues
        processed_count = 0
        for i, symbol in enumerate(symbols):
            try:
                self._process_symbol(symbol, input_paths, output_path)
                processed_count += 1
                
                # Memory check every 10 symbols
                if (i + 1) % 10 == 0:
                    self.memory_manager.check_memory_with_cleanup(f"feature building batch {(i+1)//10}")
                
                # Progress update every 50 symbols
                if (i + 1) % 50 == 0 or (i + 1) == len(symbols):
                    progress = (i + 1) / len(symbols) * 100
                    print(f"✅ Processed {i+1}/{len(symbols)} symbols ({progress:.1f}%)")
                    
            except Exception as e:
                import traceback
                logger.error(f"Failed to process symbol {symbol}: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                continue
                
        print(f"✅ Feature building completed! Processed {processed_count}/{len(symbols)} symbols")
        logger.info("Feature building completed")
        
        # Write schema manifest and run metrics
        self._write_schema_and_metrics(output_path, symbols, processed_count)
    
    def _process_symbol(self, symbol: str, input_paths: List[str], output_path: Path):
        """Process a single symbol with streaming using dataset scanning"""
        print(f"🔄 Processing symbol: {symbol}")
        logger.info(f"Processing symbol: {symbol}")
        
        # Resume support: skip if per-symbol output already exists and is readable
        try:
            existing_out = self._get_symbol_output_path(symbol, output_path)
            if existing_out.exists() and existing_out.is_file() and existing_out.stat().st_size > 0:
                logger.info(f"⏭️  Skipping {symbol}: output exists → {existing_out}")
                return
        except Exception as e:
            logger.warning(f"Resume check failed for {symbol}: {e}")

        # Check if sources exist before scanning to avoid "expected at least 1 source" errors
        sym_on_disk = get_symbol_on_disk(symbol)
        if not has_sources(self.input_pattern, sym_on_disk):
            logger.warning(f"No parquet sources for {symbol} at this path; skipping")
            return
        
        # Get volume policy for this symbol
        volume_policy = self._get_volume_policy(symbol)
        logger.info(f"{symbol}: using volume policy '{volume_policy}' for scanning")
        
        # Use Polars-safe parquet scanning with volume policy
        scan = scan_symbol_lazy(self.input_pattern, symbol, volume_policy)
        if scan is None:
            logger.warning(f"No data found for symbol {symbol}")
            return
        
        # Collect with streaming to avoid memory issues
        # Safe scanner should handle dtype mismatches at scan time
        try:
            # Use streaming collection (no len() on LazyFrame)
            logger.info(f"Processing {symbol} with streaming collection")
            try:
                df = scan.collect(engine="streaming")  # Polars ≥1.25
            except TypeError:
                df = scan.collect(streaming=True)  # Fallback for older versions
            
            logger.info(f"Loaded {len(df)} rows for {symbol}")
        except Exception as e:
            logger.error(f"Failed to collect data for {symbol}: {e}")
            return
        
        if len(df) == 0:
            logger.warning(f"No data found for symbol {symbol}")
            return
        
        # Process data based on timeframe with proper timezone handling
        try:
            df_processed = process_data_by_timeframe(df, self.timeframe)
            logger.info(f"Processed to {len(df_processed)} {self.timeframe} bars for {symbol}")
        except Exception as e:
            import traceback
            logger.error(f"Failed to process {self.timeframe} data for {symbol}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return
        
        if len(df_processed) == 0:
            logger.warning(f"No {self.timeframe} bars after processing for {symbol}")
            return
        
        # Volume pipeline: apply policy-based volume handling
        logger.info(f"Volume columns available for {symbol}: {[col for col in df_processed.columns if 'volume' in col.lower()]}")
        
        # Get volume policy for this symbol
        volume_policy = self._get_volume_policy(symbol)
        logger.info(f"{symbol}: using volume policy '{volume_policy}'")
        
        # Step 1: Create canonical raw volume column (preserve integer semantics)
        if "volume_raw" in df_processed.columns:
            # Use existing volume_raw
            df_processed = df_processed.with_columns([
                pl.col("volume_raw").cast(pl.Int64, strict=False).alias("volume_raw")
            ])
        elif "volume" in df_processed.columns:
            # Create volume_raw from existing volume
            df_processed = df_processed.with_columns([
                pl.col("volume").cast(pl.Int64, strict=False).alias("volume_raw")
            ])
        else:
            logger.error(f"No volume column found for {symbol}. Available columns: {df_processed.columns}")
            raise ValueError(f"No volume column found for {symbol}")
        
        # Step 2: Apply volume policy
        if volume_policy == "strict":
            # Fail fast on fractional volumes
            non_int_count = df_processed.select(
                ((pl.col("volume_raw").cast(pl.Float64) - pl.col("volume_raw")).abs() > 0)
                .sum()
            ).item()
            if non_int_count > 0:
                logger.error(f"{symbol}: fractional volume detected in raw ingest ({non_int_count} rows); fix vendor/aggregation")
                raise ValueError(f"{symbol}: fractional volume detected in raw ingest; fix vendor/aggregation")
            
            # Maintain legacy volume column
            df_processed = df_processed.with_columns([
                pl.col("volume_raw").alias("volume")
            ])
            
        elif volume_policy == "separate":
            # Separate fractional volumes: keep int volume, create volume_adj
            df_processed = self._separate_volume(df_processed, symbol)
            
        elif volume_policy == "coerce":
            # Coerce near-integer volumes to integers
            df_processed = self._coerce_volume(df_processed, symbol)
            
        elif volume_policy == "quarantine":
            # Quarantine rows with fractional volumes
            df_processed, bad_rows = self._quarantine_volume(df_processed, symbol)
            if len(bad_rows) > 0:
                # TODO: Save quarantined rows to quarantine/AA.csv
                logger.warning(f"{symbol}: {len(bad_rows)} rows quarantined due to fractional volumes")
        else:
            raise ValueError(f"Unknown volume policy: {volume_policy}")
        
        # Step 3: Validate volume integrity
        nan_ratio = df_processed.select(pl.col("volume").is_null().mean()).item()
        if nan_ratio > 0.001:  # More than 0.1% NaN
            logger.error(f"{symbol}: volume NaN ratio too high: {nan_ratio:.2%}")
            raise ValueError(f"{symbol}: volume NaN ratio too high: {nan_ratio:.2%}")
        
        logger.info(f"{symbol}: volume pipeline OK - dtype: {df_processed.schema['volume']}, nan_ratio: {nan_ratio:.4%}")

        # Normalize schema to ensure all required columns are present
        df_processed = normalize_bars(df_processed, symbol, interval=self.timeframe, source="prod", adjusted=False)
        
        # Remove extra columns to match 5m schema (296 columns)
        columns_to_remove = ["ny_date", "__index_level_0__", "is_ext_hours"]
        df_processed = df_processed.drop([col for col in columns_to_remove if col in df_processed.columns])

        # Session handling: RTH filtering and is_ext_hours flag (interval-agnostic)
        try:
            sess_cfg = (self.config.get('session') if hasattr(self, 'config') else None) or {}
            session_mode = str(sess_cfg.get('mode', 'RTH')).upper()
            completeness_cfg = sess_cfg.get('completeness', {})
            completeness_method = str(completeness_cfg.get('method', 'empirical_mode')).lower()
            tolerance_bars = int(completeness_cfg.get('tolerance_bars', 1))
            clamp_min = int(completeness_cfg.get('clamp_min', 26))
            clamp_max = int(completeness_cfg.get('clamp_max', 36))
            min_complete_ratio = float(completeness_cfg.get('min_complete_ratio', 0.90))
            # Normalize timezone for session filtering
            # If ts is timezone-naive (common for derived 15m with NY-local buckets), assume America/New_York
            ts_dtype = df_processed.schema.get("ts")
            if ts_dtype == pl.Datetime and hasattr(ts_dtype, 'time_zone') and ts_dtype.time_zone is None:
                df_processed = df_processed.with_columns([
                    pl.col("ts").dt.replace_time_zone("America/New_York")
                ])
            # If ts is tz-aware but not NY, convert a copy to NY for masking
            df_ny = df_processed.with_columns([
                pl.col("ts").dt.convert_time_zone("America/New_York").alias("ts_ny")
            ])
            
            # Normalize schema for non-RTH path
            df_ny = normalize_bars(df_ny, symbol, interval=self.timeframe, source="prod", adjusted=False)
            
            # Remove extra columns to match 5m schema (296 columns)
            columns_to_remove = ["ny_date", "__index_level_0__", "is_ext_hours"]
            df_ny = df_ny.drop([col for col in columns_to_remove if col in df_ny.columns])
            # Build time-of-day mask (minutes since midnight)
            df_ny = df_ny.with_columns([
                (pl.col("ts_ny").dt.hour() * 60 + pl.col("ts_ny").dt.minute()).alias("tod_min")
            ])
            # Use UTC calendar-based RTH mask to avoid tz/label ambiguity
            pre_rows = df_ny.height
            if session_mode == "RTH" and str(self.timeframe) == "15m":
                try:
                    # Check if we have validated 5m RTH data to roll up from
                    rollup_5m_path = output_path.parent / f"5m_comprehensive_features_final/interval=5m/{symbol}.parquet"
                    if rollup_5m_path.exists():
                        logger.info(f"{symbol}: Using 5m rollup approach for 15m RTH")
                        try:
                            df_5m = pl.read_parquet(rollup_5m_path)
                            # Ensure 5m data is RTH-only (has is_ext_hours column)
                            if "is_ext_hours" in df_5m.columns:
                                df_5m_rth = df_5m.filter(pl.col("is_ext_hours") == False)
                                if len(df_5m_rth) > 0:
                                    df_15m_rolled = _rollup_5m_to_15m(df_5m_rth)
                                    # Convert to expected format
                                    df_ny = df_15m_rolled.with_columns([
                                        pl.col("ts").dt.convert_time_zone("America/New_York").alias("ts_ny"),
                                        pl.lit(False).alias("is_ext_hours")
                                    ])
                                    post_rows = df_ny.height
                                    logger.info(f"{symbol}: 15m rollup from 5m RTH: {len(df_5m_rth)} 5m bars → {post_rows} 15m bars")
                                else:
                                    raise RuntimeError("5m RTH data is empty")
                            else:
                                raise RuntimeError("5m data missing is_ext_hours column")
                        except Exception as e_rollup:
                            logger.warning(f"{symbol}: 5m rollup failed: {e_rollup}, falling back to direct 15m RTH")
                            raise e_rollup
                    else:
                        logger.info(f"{symbol}: No 5m RTH data found, using direct 15m RTH processing")
                        raise RuntimeError("No 5m rollup data available")
                        
                except Exception as e_rollup:
                    # Fallback to direct 15m RTH processing
                    try:
                        # Convert to pandas with UTC index
                        df_pd = df_processed.select(["ts","open","high","low","close","volume"]).to_pandas()
                        idx = pd.DatetimeIndex(df_pd["ts"]) 
                        if idx.tz is None:
                            idx = idx.tz_localize("UTC")
                        else:
                            idx = idx.tz_convert("UTC")
                        df_pd = df_pd.drop(columns=["ts"]).set_index(idx)
                        
                        # Ensure DatetimeIndex[UTC], unique, monotonic
                        if not isinstance(df_pd.index, (pd.DatetimeIndex, pd.PeriodIndex)):
                            raise TypeError("RTH requires DatetimeIndex or PeriodIndex")
                        
                        if isinstance(df_pd.index, pd.PeriodIndex):
                            df_pd.index = df_pd.index.to_timestamp(how="end")
                        
                        # localize/convert to UTC
                        if df_pd.index.tz is None:
                            df_pd.index = df_pd.index.tz_localize("UTC")
                        else:
                            df_pd.index = df_pd.index.tz_convert("UTC")
                        
                        df_pd = df_pd[~df_pd.index.duplicated(keep="last")]
                        df_pd = df_pd.sort_index()
                        
                        # Time-of-day minutes in UTC (int16 is plenty)
                        tod_min = (df_pd.index.hour * 60 + df_pd.index.minute).astype("int16")
                        df_pd["tod_min"] = tod_min
                        
                        # Session date key in UTC (no tz). Useful for per-session counts & joins.
                        df_pd["session_utc"] = df_pd.index.tz_convert("UTC").date

                        from scripts.bootstrap import load_cal_guarded
                        cal = load_cal_guarded("XNYS")
                        
                        # Unified RTH slicer using reference minute inside bar
                        delta = pd.to_timedelta("15min")
                        ts_end = df_pd.index
                        ts_start = ts_end - delta
                        ts_ref = ts_end - pd.Timedelta(minutes=1)

                        # Get schedule using instance methods
                        start_date = (ts_ref[0] - pd.Timedelta(days=2)).date()
                        end_date = (ts_ref[-1] + pd.Timedelta(days=2)).date()
                        sessions = cal.sessions_in_range(start_date, end_date)
                        sched_df = _get_schedule_df(cal, sessions[0], sessions[-1])

                        # Calculate expected bars per session for validation
                        expected_bars_per_session = _calculate_expected_bars_per_session(sched_df, 15)
                        # Log compact summary instead of giant list
                        vals, counts = np.unique(expected_bars_per_session, return_counts=True)
                        logger.info(f"{symbol}: Expected 15m bars/session distribution: {dict(zip(vals.tolist(), counts.tolist()))}")

                        # Use proper Polars coercion to get epoch nanoseconds
                        open_i8 = sched_df["open"].dt.epoch("ns").to_numpy()
                        close_i8 = sched_df["close"].dt.epoch("ns").to_numpy()
                        ref_i8 = ts_ref.dt.epoch("ns").to_numpy()

                        idx = np.searchsorted(open_i8, ref_i8, side="right") - 1
                        idx = np.clip(idx, 0, len(open_i8) - 1)
                        valid = (ref_i8 >= open_i8[idx]) & (ref_i8 <= close_i8[idx])

                        ts_start_i8 = ts_start.dt.epoch("ns").to_numpy()
                        ts_end_i8 = ts_end.dt.epoch("ns").to_numpy()
                        bar_len_i8 = (ts_end_i8 - ts_start_i8)

                        open_for_bar_i8 = open_i8[idx]
                        close_for_bar_i8 = close_i8[idx]

                        ov_start = np.maximum(ts_start_i8, open_for_bar_i8)
                        ov_end = np.minimum(ts_end_i8, close_for_bar_i8)
                        ov_i8 = np.maximum(ov_end - ov_start, 0)

                        keep = valid & (ov_i8 >= bar_len_i8)  # strict inside RTH
                        df_pd_rth = df_pd.loc[keep]

                        # Sanity log with improved validation
                        try:
                            et_dates = pd.Series(df_pd_rth.index.tz_convert("America/New_York").date)
                            per_day = et_dates.value_counts()
                            mean_bd = float(per_day.mean()) if len(per_day) else 0.0
                            logger.info(f"{symbol}: RTH_DIRECT pre={len(df_pd)} post={len(df_pd_rth)} freq=15min mean_bars/day≈{mean_bd:.2f}")
                            
                            # Validate against expected bars using session-based approach
                            if len(per_day) > 0:
                                # Create expected map from schedule
                                expected_map = dict(zip(sched_df.index.date, expected_bars_per_session))
                                
                                # Count actual bars per session
                                counts = df_pd_rth.groupby("session_utc").size()
                                expected = pd.Series({d: expected_map.get(d, 26) for d in counts.index})
                                
                                # % complete averaged across sessions
                                session_pct = (counts / expected).clip(upper=1.0)
                                overall_pct = session_pct.mean() * 100.0
                                complete_sessions = int((session_pct == 1.0).sum())
                                total_sessions = len(session_pct)
                                
                                logger.info(f"{symbol}: RTH sessions complete ≈{overall_pct:.1f}% ({complete_sessions}/{total_sessions} sessions ≥ expected)")
                        except Exception:
                            pass

                        if len(df_pd_rth) == 0:
                            hour_hist = pd.Series(df_pd.index.tz_convert("America/New_York").hour).value_counts().sort_index().to_dict()
                            raise RuntimeError(f"{symbol}: RTH EMPTY — check freq/label. ET hour histogram: {hour_hist}")

                        # Convert back to Polars with is_ext_hours=false for RTH rows
                        df_pd_rth = df_pd_rth.copy()
                        df_pd_rth["ts"] = df_pd_rth.index.tz_convert("UTC")
                        
                        # Add VWAP calculation (OHLC4) if missing
                        if "vwap" not in df_pd_rth.columns:
                            df_pd_rth["vwap"] = (df_pd_rth["open"] + df_pd_rth["high"] + df_pd_rth["low"] + df_pd_rth["close"]) / 4.0
                            # Set VWAP to NaN where volume is 0
                            df_pd_rth.loc[df_pd_rth["volume"] == 0, "vwap"] = np.nan
                        
                        # Add dollar volume for consistency
                        if "dollar_volume" not in df_pd_rth.columns:
                            df_pd_rth["dollar_volume"] = df_pd_rth["close"] * df_pd_rth["volume"]
                        
                        df_rth_pl = pl.from_pandas(df_pd_rth.reset_index(drop=True))
                        df_rth_pl = df_rth_pl.with_columns(
                            pl.col("ts").cast(pl.Datetime(time_unit="us")).dt.convert_time_zone("UTC")
                        )
                        
                        # Normalize schema for RTH path
                        df_rth_pl = normalize_bars(df_rth_pl, symbol, interval=self.timeframe, source="prod", adjusted=False)
                        
                        # Remove extra columns to match 5m schema (296 columns)
                        columns_to_remove = ["ny_date", "__index_level_0__", "is_ext_hours"]
                        df_rth_pl = df_rth_pl.drop([col for col in columns_to_remove if col in df_rth_pl.columns])
                        df_ny = df_rth_pl.with_columns([
                            pl.col("ts").dt.convert_time_zone("America/New_York").alias("ts_ny"),
                            pl.lit(False).alias("is_ext_hours")
                        ])
                        # Guard: median bars/day sanity (non-strict, avoids false positives on small spans)
                        try:
                            _assert_15m_rth_structure(df_rth_pl.select("ts").to_pandas()["ts"])
                        except Exception as _e:
                            logger.warning(f"{symbol}: RTH sanity check: {_e}")
                        post_rows = df_ny.height
                        logger.info(f"{symbol}: applied RTH filter via unified UTC schedule overlap mask")
                        
                    except Exception as e_cal:
                        logger.error(f"{symbol}: RTH UTC schedule mask failed: {e_cal}")
                        raise RuntimeError(f"RTH mask failed hard: {e_cal}")
            else:
                # RTH_EXT or other intervals: keep EXT with flag using right-labeled window
                rth_mask_right = (pl.col("tod_min") >= 585) & (pl.col("tod_min") <= 960)
                df_ny = df_ny.with_columns((~rth_mask_right).alias("is_ext_hours"))
                post_rows = df_ny.height
            # Convert back to UTC (drop helper columns except is_ext_hours)
            df_ny = df_ny.with_columns([
                pl.col("ts_ny").dt.convert_time_zone("UTC").alias("ts_utc")
            ]).drop(["ts"]).rename({"ts_utc": "ts"})
            df_processed = df_ny.drop(["ts_ny", "tod_min"])  # keep is_ext_hours
            
            # Remove extra columns to match 5m schema (296 columns)
            columns_to_remove = ["ny_date", "__index_level_0__", "is_ext_hours"]
            df_processed = df_processed.drop([col for col in columns_to_remove if col in df_processed.columns])
            # Emit per-day bars summary and completeness using post-slice data (best-effort)
            try:
                per_day_df = df_ny.group_by(pl.col("ts_ny").dt.date().alias("date")).agg(pl.len().alias("n")).sort("date")
                mean_bars = per_day_df.select(pl.col("n").mean()).item()
                p10 = per_day_df.select(pl.col("n").quantile(0.10, interpolation='nearest')).item()
                p90 = per_day_df.select(pl.col("n").quantile(0.90, interpolation='nearest')).item()
                # Calculate completeness using per-session expected bars
                complete_ratio = 0.0
                expected_emp = 0
                try:
                    if session_mode == "RTH" and per_day_df.height > 0:
                        # Get actual session dates and calculate per-session completeness
                        actual_dates = per_day_df.select("date").to_series().to_list()
                        actual_bars = per_day_df.select("n").to_series().to_list()
                        
                        # Load calendar to get expected bars per session
                        from scripts.bootstrap import load_cal_guarded
                        cal = load_cal_guarded("XNYS")
                        
                        if len(actual_dates) > 0:
                            start_date = min(actual_dates)
                            end_date = max(actual_dates)
                            sessions = cal.sessions_in_range(start_date, end_date)
                            sched_df = _get_schedule_df(cal, sessions[0], sessions[-1])
                            
                            # Calculate expected bars per session
                            expected_bars_per_session = _calculate_expected_bars_per_session(sched_df, int(self.timeframe.total_seconds() / 60))
                            
                            # Match actual dates to expected bars
                            session_to_expected = dict(zip(sched_df.index.date, expected_bars_per_session))
                            expected_for_actual = [session_to_expected.get(date, 26) for date in actual_dates]  # fallback to 26 for 15m
                            
                            # Calculate completeness
                            if expected_for_actual:
                                completeness_values = [actual / expected for actual, expected in zip(actual_bars, expected_for_actual)]
                                mean_completeness = sum(completeness_values) / len(completeness_values)
                                complete_ratio = mean_completeness
                                expected_emp = int(sum(expected_for_actual) / len(expected_for_actual))
                                
                                # Log ext_hours fraction if available
                                if "is_ext_hours" in df_ny.columns:
                                    ext_fraction = df_ny.select(pl.col("is_ext_hours").mean()).item()
                                    logger.info(f"{symbol}: ext_hours fraction: {ext_fraction:.1%}")
                                    
                except Exception as e_cal:
                    logger.warning(f"{symbol}: Could not calculate per-session completeness: {e_cal}")
                    # Fallback to empirical calculation
                    try:
                        import pandas as _pd
                        n_series = _pd.Series(per_day_df["n"].to_list())
                        mode_bars = int(n_series.mode().iloc[0]) if len(n_series) > 0 else 0
                    except Exception:
                        mode_bars = int(round(mean_bars)) if mean_bars is not None else 0
                    expected_emp = max(clamp_min, min(clamp_max, mode_bars))
                    # Completeness evaluation
                    if per_day_df.height > 0 and expected_emp > 0:
                        complete_ratio = (
                            per_day_df.filter(pl.col("n") >= expected_emp - tolerance_bars).height / per_day_df.height
                        )
                
                logger.info(
                    f"{symbol}: RTH_SLICE pre={pre_rows} post={post_rows} mean_bars_per_day≈{mean_bars:.2f} expected≈{expected_emp} p10/p90={p10}/{p90} complete_ratio={complete_ratio:.2%}"
                )
                # Optional debug: ET-hour histogram once per symbol
                debug_hist = bool(completeness_cfg.get('debug_hour_hist', False))
                if debug_hist:
                    hour_hist = (
                        df_ny.group_by(pl.col("ts_ny").dt.hour().alias("hour")).agg(pl.len().alias("n")).sort("hour")
                    )
                    logger.info(f"{symbol}: ET-hour histogram (bars): {hour_hist.to_dict(as_series=False)}")
            except Exception:
                logger.info(f"{symbol}: RTH_SLICE pre={pre_rows} post={post_rows}")
        except Exception as e:
            logger.warning(f"{symbol}: session filtering setup failed; proceeding without RTH filter: {e}")
        
        # Build features lazily
        features = self._build_feature_pipeline(df_processed.lazy(), symbol)
        
        # Validate schema and coverage before processing
        if not self._validate_symbol_data(features, symbol):
            return
        
        # Loud zero-row guard
        out_rows = len(df_processed)
        if out_rows == 0:
            head_ts = df.select(pl.col("ts").min()).collect().item() if len(df) > 0 else None
            tail_ts = df.select(pl.col("ts").max()).collect().item() if len(df) > 0 else None
            glob_path = build_symbol_glob(self.input_pattern, get_symbol_on_disk(symbol))
            logger.error("ZERO rows after %s processing for %s. ts[min]=%s ts[max]=%s path=%s",
                         self.timeframe, symbol, head_ts, tail_ts, glob_path)
            raise RuntimeError(f"No {self.timeframe} data produced for {symbol}; check TZ/session/offset")
        
        # Validate bar counts before processing
        if not self._validate_bar_counts(df_processed, symbol):
            logger.warning(f"Skipping {symbol} due to bar count validation failure")
            return
        
        # Process with optimized batch size
        self._process_with_batch_optimization(features, symbol, output_path, raw_rows=len(df_processed))
        
        # Force cleanup after each symbol
        gc.collect()
        
        # Dynamic batch scaling check
        self._check_and_scale_batch_size()
        
        print(f"✅ {symbol}: features built successfully")
    
    def _validate_symbol_data(self, features: pl.LazyFrame, symbol: str) -> bool:
        """Validate schema and coverage for a symbol"""
        from scripts.schema_validator import create_schema_expectations_from_features, validate_schema
        
        try:
            # Sample a small amount of data for validation (but get full span info)
            sample = features.limit(1000).collect()
            
            # Get full span info from the complete dataset
            full_span = features.select([
                pl.col("ts").min().alias("first_ts"),
                pl.col("ts").max().alias("last_ts")
            ]).collect()
            
            if len(sample) == 0:
                logger.warning(f"No data found for {symbol} after processing")
                return False
            
            # Check for required columns (date is added later during partitioning)
            required_cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in sample.columns]
            if missing_cols:
                logger.error(f"Missing required columns for {symbol}: {missing_cols}")
                return False
            
            # Use the actual computed features for schema validation
            # Only validate against features that were actually computed by the simple feature computer
            expected_features = simple_feature_computer.get_all_features()
            
            # Add core columns
            core_columns = {
                "symbol", "ts", "open", "high", "low", "close", "volume", "vwap", "n_trades",
                "interval", "source", "adjusted", "fwd_ret_1d"
            }
            expected_features.extend(core_columns)
            
            # Create schema expectations and validate with non-strict interactions
            spec = create_schema_expectations_from_features(expected_features)
            warnings = validate_schema(sample.columns, spec, strict_interactions=False)
            if warnings:
                logger.warning(f"Schema validation warnings for {symbol}: {warnings}")
            
            # Check for excessive NaN values in critical columns
            critical_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in critical_cols:
                if col in sample.columns:
                    nan_ratio = sample[col].null_count() / len(sample)
                    if nan_ratio > 0.1:  # More than 10% NaN
                        logger.warning(f"High NaN ratio in {col} for {symbol}: {nan_ratio:.2%}")
            
            # Check date coverage (actual trading days, not calendar span)
            if 'ts' in sample.columns:
                coverage_info = self._get_date_coverage_info(sample, symbol)
                if coverage_info['unique_sessions'] < 30:
                    logger.warning(f"Limited trading sessions for {symbol}: {coverage_info['unique_sessions']} sessions")
                else:
                    logger.info(f"Date coverage: {coverage_info['unique_sessions']} trading sessions "
                              f"({coverage_info['first_date']} to {coverage_info['last_date']})")
            
            logger.info(f"✅ Schema validation passed for {symbol}")
            
            # Add comprehensive session completeness analysis using full dataset
            completeness_info = self._get_session_completeness_info(features.collect(), symbol)
            logger.info(f"ALL SESSIONS: {completeness_info['sessions_all']} days | {completeness_info['pct_complete_all']}% complete | avg {completeness_info['avg_bars_per_day_all']} bars/day")
            logger.info(f"RECENT {completeness_info['recent_n']} SESSIONS: {completeness_info['pct_complete_recent']}% complete")
            
            # Use full span info instead of sample
            if full_span.height > 0:
                first_ts = full_span.item(0, 0)
                last_ts = full_span.item(0, 1)
                logger.info(f"SPAN: {first_ts} → {last_ts}")
            else:
                logger.info(f"SPAN: {completeness_info['first_ts']} → {completeness_info['last_ts']}")
            
            # Flag low completeness symbols for microstructure experiments
            if completeness_info['pct_complete_all'] < 90:
                logger.warning(f"LOW INTRADAY COMPLETENESS: {symbol} has {completeness_info['pct_complete_all']}% complete sessions - consider excluding from microstructure-heavy experiments")
            
            # Add volume quality analysis
            volume_info = self._get_volume_quality_info(sample, symbol)
            if volume_info['pct_fractional'] > 0:
                buckets_str = ", ".join([f"{k}={v}" for k, v in volume_info['buckets'].items() if v > 0])
                logger.warning(f"Volume DQ: {volume_info['pct_fractional']}% fractional rows ({buckets_str})")
            
            # Legacy bar count validation (keep for compatibility)
            self._validate_bar_counts(sample, symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Schema validation failed for {symbol}: {e}")
            return False

    def _check_and_scale_batch_size(self):
        """Check memory usage and dynamically scale batch size."""
        import time
        
        current_time = time.time()
        
        # Only check every 10 seconds to avoid overhead
        if current_time - self._last_memory_check < 10:
            return
            
        self._last_memory_check = current_time
        
        try:
            # Get current memory usage
            sys_usage = self.memory_manager.get_system_memory_usage()
            memory_percent = sys_usage["system_percent"]
            
            # Scale up if memory usage is low
            if memory_percent < 0.6:  # Less than 60% memory usage
                self._consecutive_low_usage += 1
                if self._consecutive_low_usage >= 2:  # 2 consecutive low usage checks
                    # Scale up batch size
                    new_batch = self.memory_manager._current_batch_size * 2
                    max_batch = 25000000  # Cap at 25M
                    if new_batch <= max_batch:
                        self.memory_manager._current_batch_size = new_batch
                        self._consecutive_low_usage = 0
                        logger.info(f"Scaled batch size up to {self.memory_manager._current_batch_size:,} rows (memory: {memory_percent:.1%})")
            else:
                self._consecutive_low_usage = 0
                
            # Scale down if memory usage is high
            if memory_percent > 0.85:
                new_batch = self.memory_manager._current_batch_size // 2
                min_batch = 10000000  # Don't go below 10M
                if new_batch >= min_batch:
                    self.memory_manager._current_batch_size = new_batch
                    logger.warning(f"Scaled batch size down to {self.memory_manager._current_batch_size:,} rows (memory: {memory_percent:.1%})")
                    
        except Exception as e:
            logger.warning(f"Dynamic scaling failed: {e}")

    def _get_volume_policy(self, symbol: str) -> str:
        """Get volume policy for a symbol (with overrides)"""
        try:
            # Load config to get volume policy settings
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            volume_policy = config.get('volume_policy', {})
            default_policy = volume_policy.get('default', 'strict')
            symbol_overrides = volume_policy.get('symbol_overrides', {})
            
            return symbol_overrides.get(symbol, default_policy)
        except Exception as e:
            logger.warning(f"Failed to load volume policy for {symbol}, using strict: {e}")
            return "strict"

    def _separate_volume(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Separate fractional volumes: keep raw int volume, create volume_adj for analytics"""
        try:
            volf = pl.col("volume").cast(pl.Float64)
            voli = volf.round(0).cast(pl.Int64)
            
            # Create both columns
            df_separated = df.with_columns([
                voli.alias("volume"),           # canonical int volume
                volf.alias("volume_adj"),       # keep float for analytics
            ])
            
            # Log fractional volume metrics
            frac_delta = df_separated.select(
                (pl.col("volume_adj") - pl.col("volume")).abs().max()
            ).item()
            
            logger.info(f"{symbol}: separated volume - max_delta: {frac_delta:.6f}")
            
            return df_separated
            
        except Exception as e:
            logger.error(f"Volume separation failed for {symbol}: {e}")
            raise

    def _coerce_volume(self, df: pl.DataFrame, symbol: str, eps: float = 1e-6) -> pl.DataFrame:
        """Coerce near-integer volumes to integers, fail on large fractions"""
        try:
            volf = pl.col("volume").cast(pl.Float64)
            voli = volf.round(0).cast(pl.Int64)
            frac = (volf - voli.cast(pl.Float64)).abs()
            
            # Coerce only if fraction is tiny
            df_coerced = df.with_columns([
                pl.when(frac <= eps).then(voli)
                  .otherwise(pl.lit(None, dtype=pl.Int64))
                  .alias("volume")
            ])
            
            # Check if any rows were set to None (large fractions)
            null_count = df_coerced.select(pl.col("volume").is_null().sum()).item()
            if null_count > 0:
                raise ValueError(f"{symbol}: {null_count} rows have fractional volumes > {eps}")
            
            logger.info(f"{symbol}: coerced volume - {len(df_coerced)} rows processed")
            return df_coerced
            
        except Exception as e:
            logger.error(f"Volume coercion failed for {symbol}: {e}")
            raise

    def _quarantine_volume(self, df: pl.DataFrame, symbol: str, eps: float = 1e-6) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Quarantine rows with fractional volumes"""
        try:
            volf = pl.col("volume").cast(pl.Float64)
            voli = volf.round(0).cast(pl.Int64)
            frac = (volf - voli.cast(pl.Float64)).abs()
            
            # Split into good and bad rows
            bad_rows = df.filter(frac > eps)
            good_rows = df.filter(frac <= eps).with_columns(voli.alias("volume"))
            
            # Log quarantine info
            logger.info(f"{symbol}: quarantined {len(bad_rows)} rows with fractional volumes")
            
            return good_rows, bad_rows
            
        except Exception as e:
            logger.error(f"Volume quarantine failed for {symbol}: {e}")
            raise

    def _assert_volume_ok(self, df: pl.DataFrame, symbol: str):
        """Assert volume column integrity before writing"""
        try:
            # Check integer type
            volume_dtype = df.schema["volume"]
            if volume_dtype not in (pl.Int64, pl.UInt64):
                raise ValueError(f"{symbol}: volume dtype must be int, got {volume_dtype}")
            
            # Check no nulls
            has_nulls = df.select(pl.col("volume").is_null().any()).item()
            if has_nulls:
                raise ValueError(f"{symbol}: volume contains nulls")
            
            # Check no negative volumes
            has_negative = df.select((pl.col("volume") < 0).any()).item()
            if has_negative:
                raise ValueError(f"{symbol}: volume contains negative values")
                
            logger.info(f"{symbol}: volume assertions passed - dtype: {volume_dtype}, rows: {len(df)}")
            
        except Exception as e:
            logger.error(f"Volume assertion failed for {symbol}: {e}")
            raise

    def _get_symbol_output_path(self, symbol: str, output_dir: Path) -> Path:
        """Construct proper file path for symbol output"""
        # Create partitioned directory structure: interval=5m/symbol=SYMBOL/symbol.parquet
        symbol_dir = output_dir / f"interval={self.timeframe}" / f"symbol={symbol}"
        symbol_dir.mkdir(parents=True, exist_ok=True)
        return symbol_dir / f"{symbol}.parquet"

    def _process_with_batch_optimization(self, features: pl.LazyFrame, symbol: str, output_path: Path, raw_rows: int):
        """Process features with batch size optimization"""
        try:
            # Get current batch size from memory manager
            batch_size = self.memory_manager._current_batch_size
            
            # Construct proper file path for this symbol
            symbol_output_path = self._get_symbol_output_path(symbol, output_path)
            
            # If dataset is smaller than batch size, process normally
            if raw_rows <= batch_size:
                logger.info(f"Processing {symbol} normally ({raw_rows:,} rows <= {batch_size:,} batch size)")
                self._write_features(features, symbol, symbol_output_path)
            else:
                # Process in chunks
                logger.info(f"Processing {symbol} in chunks ({raw_rows:,} rows > {batch_size:,} batch size)")
                self._process_in_chunks(features, symbol, symbol_output_path, batch_size)
                
        except Exception as e:
            logger.error(f"Batch optimization failed for {symbol}: {e}")
            # Fallback to normal processing
            symbol_output_path = self._get_symbol_output_path(symbol, output_path)
            self._write_features(features, symbol, symbol_output_path)

    def _process_in_chunks(self, features: pl.LazyFrame, symbol: str, output_path: Path, batch_size: int):
        """Process large datasets in chunks using date-based batching"""
        try:
            # For now, process normally without chunking to avoid LazyFrame len() issues
            # TODO: Implement date-based batching for very large datasets
            logger.info(f"Processing {symbol} normally (chunking disabled to avoid LazyFrame len() issues)")
            self._write_features(features, symbol, output_path)
                
        except Exception as e:
            logger.error(f"Chunk processing failed for {symbol}: {e}")
            # Fallback to normal processing
            self._write_features(features, symbol, output_path)

    def _write_schema_manifest(self, output_dir: Path, schema_info: dict):
        """Write schema manifest to output directory"""
        try:
            schema_path = output_dir / "schema.json"
            with open(schema_path, 'w') as f:
                json.dump(schema_info, f, indent=2, default=str)
            logger.info(f"📋 Schema manifest written → {schema_path}")
        except Exception as e:
            logger.warning(f"Failed to write schema manifest: {e}")

    def _write_run_metrics(self, output_dir: Path, metrics: list):
        """Write run metrics CSV to output directory"""
        try:
            import pandas as pd
            metrics_path = output_dir / "run_metrics.csv"
            df_metrics = pd.DataFrame(metrics)
            df_metrics.to_csv(metrics_path, index=False)
            logger.info(f"📊 Run metrics written → {metrics_path}")
        except Exception as e:
            logger.warning(f"Failed to write run metrics: {e}")

    def _write_schema_and_metrics(self, output_dir: Path, symbols: list, processed_count: int):
        """Write schema manifest and run metrics after feature building"""
        try:
            # Collect schema info from first processed symbol
            schema_info = {
                "version": "1.0",
                "timeframe": self.timeframe,
                "total_symbols": len(symbols),
                "processed_symbols": processed_count,
                "created_at": datetime.now().isoformat(),
                "config": {
                    "feature_categories": self.feature_config.get("features", []),
                    "volume_policy": self.config.get("volume_policy", {}),
                }
            }
            
            # Try to get schema from first available symbol
            for symbol in symbols:
                symbol_path = output_dir / f"interval={self.timeframe}" / f"symbol={symbol}" / f"{symbol}.parquet"
                if symbol_path.exists():
                    try:
                        df_sample = pl.read_parquet(str(symbol_path), n_rows=1)
                        schema_info["columns"] = {col: str(dtype) for col, dtype in df_sample.schema.items()}
                        schema_info["feature_count"] = len([col for col in df_sample.columns if col not in ["ts", "symbol", "open", "high", "low", "close", "volume", "volume_raw", "volume_adj"]])
                        break
                    except Exception as e:
                        logger.warning(f"Failed to read schema from {symbol}: {e}")
                        continue
            
            # Write schema manifest
            self._write_schema_manifest(output_dir, schema_info)
            
            # Collect run metrics
            metrics = []
            for symbol in symbols:
                symbol_path = output_dir / f"interval={self.timeframe}" / f"symbol={symbol}" / f"{symbol}.parquet"
                if symbol_path.exists():
                    try:
                        df_info = pl.read_parquet(str(symbol_path), n_rows=1)
                        # Get basic info without loading full dataset
                        row_count = pl.scan_parquet(str(symbol_path)).select(pl.len()).collect().item()
                        
                        metric = {
                            "symbol": symbol,
                            "rows": row_count,
                            "columns": len(df_info.columns),
                            "has_volume_adj": "volume_adj" in df_info.columns,
                            "volume_policy": self._get_volume_policy(symbol),
                            "file_size_mb": symbol_path.stat().st_size / (1024 * 1024),
                        }
                        metrics.append(metric)
                    except Exception as e:
                        logger.warning(f"Failed to collect metrics for {symbol}: {e}")
                        metrics.append({
                            "symbol": symbol,
                            "rows": 0,
                            "columns": 0,
                            "has_volume_adj": False,
                            "volume_policy": "unknown",
                            "file_size_mb": 0,
                        })
            
            # Write run metrics
            if metrics:
                self._write_run_metrics(output_dir, metrics)
                
        except Exception as e:
            logger.warning(f"Failed to write schema and metrics: {e}")

    def _write_features(self, features: pl.LazyFrame, symbol: str, output_path: Path):
        """Write features to output path"""
        try:
            # Create output directory
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure we have a file path, not directory
            assert output_path.suffix == ".parquet", f"Expected .parquet file, got {output_path}"
            assert not output_path.is_dir(), f"Expected file path, got directory: {output_path}"
            
            # Collect data for volume assertions
            df = features.collect(engine="streaming")
            
            # DEBUG: Check intervals before writing
            dbg(f"pre_write_{symbol}", df)
            assert_step(df, expect_minutes(TF(self.timeframe)), f"pre_write_{symbol}")
            
            # Assert volume integrity before writing
            self._assert_volume_ok(df, symbol)
            
            # Write parquet file
            df.write_parquet(
                str(output_path),
                compression="zstd",
                compression_level=7,
                row_group_size=1000000,
                use_pyarrow=True,
            )
            logger.info(f"📦 Wrote {symbol} rows={len(df)} cols={len(df.columns)} → {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to write features for {symbol}: {e}")
            raise
    
    def _validate_bar_counts(self, df: pl.DataFrame, symbol: str):
        """Validate bars/day completeness with support for empirical mode and RTH-only gating."""
        try:
            sess_cfg = (self.config.get('session') if hasattr(self, 'config') else None) or {}
            mode = str(sess_cfg.get('mode', 'RTH')).upper()
            expected_map = sess_cfg.get('expected_bars', {})
            tol = float(sess_cfg.get('completeness_tolerance', 0.02))
            completeness_cfg = sess_cfg.get('completeness', {})
            method = str(completeness_cfg.get('method', 'empirical_mode')).lower()
            tolerance_bars = int(completeness_cfg.get('tolerance_bars', 1))
            min_complete_ratio = float(completeness_cfg.get('min_complete_ratio', 0.90))
            clamp_min = int(completeness_cfg.get('clamp_min', 26))
            clamp_max = int(completeness_cfg.get('clamp_max', 36))

            # Fallback expected bars if map missing
            def fallback_expected(tf: str, mode: str) -> tuple[int, int]:
                defaults = {
                    ('RTH','5m'):(76,80), ('RTH','15m'):(25,27), ('RTH','30m'):(12,14), ('RTH','1d'):(1,1),
                    ('RTH_EXT','5m'):(90,110), ('RTH_EXT','15m'):(33,37), ('RTH_EXT','30m'):(16,19), ('RTH_EXT','1d'):(1,1),
                }
                return defaults.get((mode, tf), (76, 80))

            min_bars, max_bars = None, None
            try:
                rng = expected_map.get(mode, {}).get(str(self.timeframe))
                if isinstance(rng, list) and len(rng) == 2:
                    min_bars, max_bars = int(rng[0]), int(rng[1])
            except Exception:
                pass
            if min_bars is None or max_bars is None:
                min_bars, max_bars = fallback_expected(str(self.timeframe), mode)
            
            if len(df) == 0:
                logger.warning(f"No data to validate for {symbol}")
                return False
            
            # Convert to date and count bars per day
            df_with_date = df.with_columns(
                pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date")
            )
            
            bars_per_day = df_with_date.group_by("date").agg(pl.len().alias("count")).sort("date")
            total_days = bars_per_day.height
            if total_days == 0:
                logger.warning(f"No trading days found for {symbol}")
                return False

            if method == 'rth_only':
                # Gate on RTH counts only: derive expected for RTH, ignore EXT
                rth_expected = {
                    '5m': (76, 80), '15m': (25, 27), '30m': (12, 14), '1h': (6, 7)
                }.get(str(self.timeframe), (1, 1000))
                full_days = bars_per_day.filter(pl.col("count").is_between(rth_expected[0], rth_expected[1])).height
                pct_full = full_days / total_days * 100
                threshold = min_complete_ratio * 100.0
                if pct_full < threshold:
                    enforce = bool(sess_cfg.get('enforce', {}).get(str(self.timeframe), self.timeframe == '5m'))
                    if not enforce:
                        logger.warning(
                            f"RTH-only completeness below threshold for {symbol}: {pct_full:.1f}% in [{rth_expected[0]},{rth_expected[1]}] — proceeding (warn-only)"
                        )
                        return True
                    logger.warning(
                        f"Bar count validation failed (RTH-only) for {symbol}: {self.timeframe}: {pct_full:.1f}% complete"
                    )
                    return False
                else:
                    logger.info(f"✅ {symbol}: RTH-only completeness {pct_full:.1f}% ({full_days}/{total_days} days)")
                    return True
            else:
                # Empirical expected bars per day with clamping and tolerance
                try:
                    import pandas as _pd
                    n_series = _pd.Series(bars_per_day["count"].to_list())
                    mode_bars = int(n_series.mode().iloc[0]) if len(n_series) > 0 else 0
                except Exception:
                    mode_bars = int(bars_per_day.select(pl.col("count").mean()).item()) if total_days > 0 else 0
                expected_emp = max(clamp_min, min(clamp_max, mode_bars))
                complete_days = bars_per_day.filter(pl.col("count") >= expected_emp - tolerance_bars).height
                complete_ratio = complete_days / total_days
                pct_full = complete_ratio * 100.0
                if complete_ratio < min_complete_ratio:
                    enforce = bool(sess_cfg.get('enforce', {}).get(str(self.timeframe), self.timeframe == '5m'))
                    if not enforce:
                        logger.warning(
                            f"Empirical completeness below threshold for {symbol}: {pct_full:.1f}% (expected≈{expected_emp}, tol={tolerance_bars}) — proceeding (warn-only)"
                        )
                        return True
                    logger.warning(
                        f"Bar count validation failed (empirical) for {symbol}: {self.timeframe}: {pct_full:.1f}% complete (expected≈{expected_emp})"
                    )
                    return False
                else:
                    logger.info(
                        f"✅ {symbol}: completeness {pct_full:.1f}% using expected≈{expected_emp} (tol={tolerance_bars}) ({complete_days}/{total_days} days)"
                    )
                    return True
                
        except Exception as e:
            logger.warning(f"Bar count validation failed for {symbol}: {e}")
            return False
    
    def _get_date_coverage_info(self, df: pl.DataFrame, symbol: str) -> dict:
        """Get comprehensive date coverage information"""
        try:
            # Convert timestamp to date and get unique sessions
            df_with_date = df.with_columns(
                pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date")
            )
            
            unique_dates = df_with_date.select("date").unique().sort("date")
            unique_sessions = unique_dates.height
            
            if unique_sessions > 0:
                first_date = unique_dates.item(0, 0)
                last_date = unique_dates.item(-1, 0)
            else:
                first_date = last_date = None
            
            return {
                'unique_sessions': unique_sessions,
                'first_date': first_date,
                'last_date': last_date
            }
        except Exception as e:
            logger.warning(f"Date coverage analysis failed for {symbol}: {e}")
            return {'unique_sessions': 0, 'first_date': None, 'last_date': None}
    
    def _get_session_completeness_info(self, df: pl.DataFrame, symbol: str) -> dict:
        """Get comprehensive session completeness metrics (all vs recent) with early-close awareness"""
        try:
            # Convert to date and count bars per day
            df_with_date = df.with_columns(
                pl.col("ts").dt.convert_time_zone("America/New_York").dt.date().alias("date")
            )
            
            bars_per_day = df_with_date.group_by("date").agg(pl.len().alias("count")).sort("date")
            
            if bars_per_day.height == 0:
                return {
                    'sessions_all': 0, 'pct_complete_all': 0, 'recent_n': 0, 'pct_complete_recent': 0, 
                    'avg_bars_per_day_all': 0, 'first_ts': None, 'last_ts': None
                }
            
            # Early-close aware completeness function
            def is_complete_polars(count):
                # 78±2 normal day; 54±2 early close (1pm ET)
                return ((count >= 76) & (count <= 80)) | ((count >= 52) & (count <= 56))
            
            # All sessions completeness
            sessions_all = bars_per_day.height
            complete_mask = bars_per_day.filter(is_complete_polars(pl.col("count")))
            all_complete = complete_mask.height
            pct_complete_all = (all_complete / sessions_all * 100) if sessions_all > 0 else 0
            avg_bars_per_day_all = bars_per_day.select(pl.col("count").mean()).item(0, 0)
            
            # Recent N sessions completeness
            recent_n = min(20, sessions_all)
            recent_bars = bars_per_day.tail(recent_n)
            recent_complete = recent_bars.filter(is_complete_polars(pl.col("count"))).height
            pct_complete_recent = (recent_complete / recent_n * 100) if recent_n > 0 else 0
            
            # Get timestamp range
            first_ts = df.select(pl.col("ts").min()).item(0, 0)
            last_ts = df.select(pl.col("ts").max()).item(0, 0)
            
            return {
                'sessions_all': sessions_all,
                'pct_complete_all': round(pct_complete_all, 1),
                'recent_n': recent_n,
                'pct_complete_recent': round(pct_complete_recent, 1),
                'avg_bars_per_day_all': round(avg_bars_per_day_all, 2),
                'first_ts': str(first_ts),
                'last_ts': str(last_ts)
            }
        except Exception as e:
            logger.warning(f"Session completeness analysis failed for {symbol}: {e}")
            return {
                'sessions_all': 0, 'pct_complete_all': 0, 'recent_n': 0, 'pct_complete_recent': 0, 
                'avg_bars_per_day_all': 0, 'first_ts': None, 'last_ts': None
            }
    
    def _get_volume_quality_info(self, df: pl.DataFrame, symbol: str) -> dict:
        """Get volume data quality metrics"""
        try:
            if 'volume' not in df.columns:
                return {'pct_fractional': 0, 'buckets': {}}
            
            # Convert volume to float and check for fractional values
            volume_series = df.select(pl.col("volume").cast(pl.Float64)).to_series()
            
            # Check for fractional volumes (not whole numbers)
            fractional_mask = (volume_series % 1) != 0
            pct_fractional = (fractional_mask.sum() / len(volume_series) * 100) if len(volume_series) > 0 else 0
            
            # Count common fractional patterns
            buckets = {}
            if pct_fractional > 0:
                buckets = {
                    "half": ((volume_series % 1).abs() - 0.5).abs() < 1e-6,
                    "third": ((volume_series % 1).abs() - 1/3).abs() < 1e-6,
                    "quarter": ((volume_series % 1).abs() - 0.25).abs() < 1e-6,
                }
                buckets = {k: v.sum() for k, v in buckets.items()}
                buckets["other_frac"] = fractional_mask.sum() - sum(buckets.values())
            
            return {
                'pct_fractional': round(pct_fractional, 2),
                'buckets': buckets,
                'total_rows': len(volume_series)
            }
        except Exception as e:
            logger.warning(f"Volume quality analysis failed for {symbol}: {e}")
            return {'pct_fractional': 0, 'buckets': {}, 'total_rows': 0}
    
    def _apply_session_normalization(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        """Apply session normalization to ensure proper RTH filtering and minute grid"""
        try:
            from utils.session_normalize import normalize_interval
            
            # Determine interval from input pattern
            interval = "5m"  # Default
            if hasattr(self, 'input_pattern'):
                if 'interval=15m' in str(self.input_pattern):
                    interval = "15m"
                elif 'interval=30m' in str(self.input_pattern):
                    interval = "30m"
                elif 'interval=1h' in str(self.input_pattern):
                    interval = "1h"
            
            # Convert ts back to datetime for normalization
            df_with_dt = df.with_columns(
                pl.col('ts').cast(pl.Datetime(time_unit='ns')).dt.replace_time_zone('UTC')
            )
            
            # Apply session normalization
            normalized = normalize_interval(df_with_dt, interval)
            
            # Convert back to int64 nanoseconds
            normalized = normalized.with_columns(
                pl.col('ts').dt.timestamp('ns').cast(pl.Int64)
            )
            
            logger.info(f"Applied session normalization for {symbol} ({interval}): {df.height} → {normalized.height} bars")
            
            # Guardrail: catch zero rows after normalization
            if normalized.height == 0:
                logger.error(f"ZERO rows after session normalization for {symbol}; dumping sample timestamps")
                sample_pre = df.head(3)["ts"].to_list() if len(df) > 0 else []
                sample_post = normalized.head(3)["ts"].to_list() if len(normalized) > 0 else []
                logger.error(f"Pre-normalization samples: {sample_pre}")
                logger.error(f"Post-normalization samples: {sample_post}")
                raise RuntimeError(f"Normalization nuked all rows for {symbol}; check TZ/offset/session")
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Session normalization failed for {symbol}: {e}")
            return df
    
    def _should_exclude_microstructure(self, symbol: str) -> bool:
        """Check if symbol should be excluded from microstructure features due to low completeness"""
        try:
            # Load config to get quality thresholds
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            quality_threshold = config.get('quality_gates', {}).get('microstructure_completeness_threshold', 0.95)
            
            # This would need to be called with actual completeness data
            # For now, return False (include all features)
            return False
        except Exception as e:
            logger.warning(f"Failed to check microstructure exclusion for {symbol}: {e}")
            return False

    def _build_feature_pipeline(self, scan: pl.LazyFrame, symbol: str = None) -> pl.LazyFrame:
        """Build the feature computation pipeline using central feature computer"""
        try:
            # Get features from config
            config_features = []
            if hasattr(self, 'config') and 'features' in self.config:
                features_config = self.config['features']
                if isinstance(features_config, list):
                    # New format: list of category names
                    config_features = features_config
                elif isinstance(features_config, dict):
                    # Old format: dictionary of categories
                    for category in features_config.values():
                        if isinstance(category, list):
                            for item in category:
                                # Split comma-separated features
                                features = [f.strip() for f in item.split(',')]
                                config_features.extend(features)
            
            # Quality gate: exclude microstructure features for low-completeness symbols
            if symbol and self._should_exclude_microstructure(symbol):
                logger.warning(f"{symbol}: Excluding microstructure features due to low session completeness")
                # Filter out microstructure features
                config_features = [f for f in config_features if f != 'microstructure']
            
            # Use central feature computer
            return simple_feature_computer.compute_features(scan, config_features)
        except Exception as e:
            logger.error(f"Error building feature pipeline: {e}")
            raise
    
    def _rsi(self, close: pl.Expr, window: int) -> pl.Expr:
        """Calculate RSI"""
        delta = close.diff()
        gain = delta.clip(0, None).rolling_mean(window)
        loss = (-delta.clip(None, 0)).rolling_mean(window)
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _macd(self, close: pl.Expr, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = close.ewm_mean(span=fast)
        ema_slow = close.ewm_mean(span=slow)
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm_mean(span=signal)
        histogram = macd_line - signal_line
        
        return [
            macd_line.cast(pl.Float32).alias("macd_12_26"),
            signal_line.cast(pl.Float32).alias("macd_12_26_signal"),
            histogram.cast(pl.Float32).alias("macd_12_26_hist")
        ]
    
    def _bollinger_bands(self, close: pl.Expr, window: int = 20, std: float = 2.0):
        """Calculate Bollinger Bands"""
        ma = close.rolling_mean(window)
        std_dev = close.rolling_std(window)
        upper = ma + (std_dev * std)
        lower = ma - (std_dev * std)
        width = upper - lower
        percent_b = (close - lower) / width
        
        return [
            upper.cast(pl.Float32).alias("bb_upper_20"),
            lower.cast(pl.Float32).alias("bb_lower_20"),
            width.cast(pl.Float32).alias("bb_width_20"),
            percent_b.cast(pl.Float32).alias("bb_percent_b_20")
        ]
    
    def _process_monthly_chunks(self, features: pl.LazyFrame, symbol: str, output_path: Path, raw_rows: int):
        """Process features in monthly chunks"""
        # For now, process all data at once to avoid date complexity
        # In production, you'd want proper monthly chunking
        try:
            before_collect_schema = features.collect_schema()
            df = features.collect()
            assert_no_drift("collect", before_collect_schema, df, logger)
            if len(df) > 0:
                # Add forward returns only if configured; keep store non-destructive by default
                df_before_label = df.clone()
                include_label = bool(self.output_config.get("include_label", False))
                if include_label:
                    df = self._add_forward_returns(df)
                
                # Skip session normalization for data that's already properly filtered
                if self.timeframe in ["1h", "30m", "15m", "5m"]:
                    logger.info(f"Skipping session normalization for {symbol} ({self.timeframe} data already filtered)")
                else:
                    # Apply session normalization to ensure proper RTH filtering and minute grid
                    df = self._apply_session_normalization(df, symbol)
                
                # Drop helper that should not be in feature parquet
                if "price_col_used" in df.columns:
                    df = df.drop("price_col_used")

                # Ensure symbol is plain Utf8 (no categorical/dictionary)
                if "symbol" in df.columns:
                    df = df.with_columns(pl.col("symbol").cast(pl.Utf8))

                # Remove any residual categoricals
                df = df.with_columns(cs.categorical().cast(pl.Utf8))

                # Downcast feature floats only; exclude base/target columns
                df = downcast_feature_floats_df(df, target_col=("fwd_ret_1d" if "fwd_ret_1d" in df.columns else None))

                # Normalize ts to int64 epoch ns for downstream consumers
                if "ts" in df.columns and df.schema.get("ts") in (pl.Datetime, pl.Date):
                    # Handle timezone-aware datetimes correctly
                    ts_dtype = df.schema.get("ts")
                    if hasattr(ts_dtype, 'time_zone') and ts_dtype.time_zone is not None:
                        # For timezone-aware, convert to UTC first, then to nanoseconds
                        df = (df.with_columns(pl.col("ts").dt.convert_time_zone("UTC").dt.timestamp("ns").alias("ts_ns"))
                                .drop("ts").rename({"ts_ns":"ts"})
                                .with_columns(pl.col("ts").cast(pl.Int64)))
                    else:
                        # For naive datetimes, assume UTC
                        df = (df.with_columns(pl.col("ts").dt.timestamp("ns").alias("ts_ns"))
                                .drop("ts").rename({"ts_ns":"ts"})
                                .with_columns(pl.col("ts").cast(pl.Int64)))

                # Drop non-feature helper columns that may confuse downstream SVM writers
                for helper_col in ["ny_date"]:
                    if helper_col in df.columns:
                        df = df.drop(helper_col)
                
                # Feature selection - get features from config
                config_features = []
                if hasattr(self, 'config') and 'features' in self.config:
                    features_config = self.config['features']
                    if isinstance(features_config, list):
                        # New format: list of category names - pass directly to feature computer
                        config_features = features_config
                    elif isinstance(features_config, dict):
                        # Old format: dictionary of categories
                        for category in features_config.values():
                            if isinstance(category, list):
                                for item in category:
                                    # Split comma-separated features
                                    features = [f.strip() for f in item.split(',')]
                                    config_features.extend(features)
                
                # For new format (category names), we don't need to do feature selection
                # because the feature computer handles the categories internally
                if isinstance(self.config.get('features', []), list):
                    # New format: keep all columns (feature computer handles categories)
                    keep = df.columns
                else:
                    # Old format: do feature selection
                    base_columns = {"ts", "open", "high", "low", "close", "volume", "vwap", "n_trades", "interval", "source", "adjusted", "symbol"}
                    selected = set(config_features)
                    present = [c for c in df.columns if c in selected and c not in base_columns]
                    missing = sorted(list(selected - set(present) - base_columns))
                    
                    if not present:
                        logger.error("0/%d selected features present for %s. Missing %d (e.g. %s). Check interval namespace.",
                                     len(selected), symbol, len(missing), missing[:5])
                        # Fallback: keep OHLCV so rows still write
                        keep = ["ts","open","high","low","close","volume","vwap","fwd_ret_1d"]
                    else:
                        # Keep OHLCV + present features
                        keep = ["ts","open","high","low","close","volume","vwap","fwd_ret_1d"] + present
                
                # Project to selected columns
                df = df.select([c for c in keep if c in df.columns])

                # Snapshot right after projection (label present; no NaN trim yet)
                df_after_projection = df.clone()

                # Define model feature set (post-projection, excluding core/labels)
                feature_cols = [
                    c for c in df.columns
                    if c not in {"ts","symbol","interval","date","open","high","low","close","volume","vwap","fwd_ret_1d","units","price_col_used","adjusted_used","horizon_bars"}
                ]

                # Thresholded readiness mask on features BEFORE NaN trim and AFTER label added
                global_frac_threshold = 0.90
                if feature_cols:
                    non_null_sum = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.UInt16) for c in feature_cols])
                    need = int(max(1, round(global_frac_threshold * len(feature_cols))))
                    df_ready = df.select([
                        pl.col("ts"),
                        (non_null_sum >= pl.lit(need)).alias("___ready___"),
                        *feature_cols
                    ])
                    # df_ready is a DataFrame; no collect() needed
                    first_ready_ts_series = (
                        df_ready
                        .filter(pl.col("___ready___"))
                        .select(pl.col("ts").min().alias("__first_ts__"))
                    )["__first_ts__"]
                    first_ready_ts = first_ready_ts_series[0] if len(first_ready_ts_series) else None
                else:
                    df_ready = df.select([pl.col("ts")]).with_columns(pl.lit(True).alias("___ready___"))
                    first_ready_ts = df["ts"].min() if len(df) else None

                # Row trim policy: default none (do not drop rows because of feature nulls)
                policy = str(self.output_config.get("row_trim_policy", "none")).lower()
                if feature_cols:
                    pre_nan_trim_rows = len(df)
                    if policy == "none" or policy == "training_only":
                        # Keep rows; compute post count without dropping
                        post_nan_trim_rows = len(df)
                    elif policy == "minimal":
                        df = df.drop_nulls(["ts","symbol","close"])  # only fatal columns
                        post_nan_trim_rows = len(df)
                    else:
                        post_nan_trim_rows = len(df)
                else:
                    pre_nan_trim_rows = post_nan_trim_rows = len(df)
                
                # Enforce base schema just before adding partitions
                before_enforce = df.schema
                df = enforce_base_schema_df(df)
                assert_no_drift("enforce_base_schema", before_enforce, df, logger)

                # Add partition columns
                interval = self.timeframe  # Use the detected timeframe
                
                df = add_partitions(df, symbol, interval)
                
                # Preflight check
                preflight_write(df, symbol)

                # Store-integrity assert: rows in == rows out for feature-store builds
                rows_in = int(len(df_before_label))
                rows_out = int(len(df))
                policy = str(self.output_config.get("row_trim_policy", "none")).lower()
                if policy == "none":
                    assert rows_out == rows_in, f"{symbol}: store row mismatch {rows_in}→{rows_out}"

                # Deterministic drop accounting using thresholded readiness
                raw_len = int(raw_rows)
                if first_ready_ts is None:
                    raise RuntimeError(
                        f"Readiness threshold {int(global_frac_threshold*100)}% not met for {symbol}; consider lowering threshold or pruning laggards."
                    )
                lookback = _count_where(df_before_label.select(pl.col("ts")), pl.col("ts") < pl.lit(first_ready_ts))
                total_after_ready = _count_where(df_after_projection, pl.col("ts") >= pl.lit(first_ready_ts))
                if include_label and "fwd_ret_1d" in df_after_projection.columns:
                    label_valid_len = int(
                        df_after_projection
                        .filter(pl.col("ts") >= pl.lit(first_ready_ts))
                        .select(pl.col("fwd_ret_1d").is_not_null().sum().alias("n"))
                        .to_dict()["n"][0]
                    )
                    drop_label = max(0, total_after_ready - label_valid_len)
                else:
                    drop_label = 0
                final_len = int(len(df))
                total_drop = max(0, raw_len - final_len)
                # If policy is none/training_only, do not attribute drops to NaN trimming
                drop_nan = 0 if policy in ("none","training_only") else max(0, total_drop - lookback - drop_label)
                approx_days = int(round(raw_len / (6.5 if self.timeframe == "1h" else 1.0)))
                looks_like_days = abs(drop_label - approx_days) <= 3
                logger.info(
                    f"{symbol}: dropped {total_drop:,} rows (lookback={lookback:,}; label_shift={drop_label:,}; NaN_trim={drop_nan:,}; threshold={int(global_frac_threshold*100)}%; req_cols={len(feature_cols)})."
                )
                if looks_like_days:
                    logger.warning(f"{symbol}: label_shift drop ({drop_label:,}) ≈ days ({approx_days:,}). Check label horizon units (bars vs days).")

                # Coverage report (post-ready) and laggards (metrics only)
                if feature_cols:
                    laggards = []
                    for c in feature_cols[:200]:
                        try:
                            nnull = int(
                                df_before_label.filter(pl.col("ts") < pl.lit(first_ready_ts))
                                .select(pl.col(c).is_null().sum().alias("n")).to_dict()["n"][0]
                            )
                        except Exception:
                            nnull = 0
                        if nnull > 0:
                            laggards.append((c, nnull))
                    if laggards:
                        laggards.sort(key=lambda x: x[1], reverse=True)
                        logger.info(f"{symbol}: readiness laggards (nulls before first ready): {laggards[:10]}")
                
                # Write to partitioned parquet with proper storage structure using safe writer
                # Format: storage/features/<feature_name>/interval=<interval>/symbol=<symbol>/features.parquet
                feature_name = output_path.name if output_path.name != "feature_store" else "default_features"
                
                # Use safe writer with strict path validation
                output_file = write_features_strict(
                    df, 
                    "storage/features", 
                    f"{feature_name}/interval={interval}", 
                    symbol
                )

                # Append per-symbol manifest row (simple KPIs)
                try:
                    manifest_path = Path("storage/features") / feature_name / "_manifest.parquet"
                    manifest_row = pl.DataFrame([
                        {
                            "symbol": symbol,
                            "rows_in": rows_in,
                            "rows_out": rows_out,
                            "cols_out": len(df.columns),
                            "lookback": lookback,
                            "label_shift": drop_label,
                            "nan_trim": drop_nan,
                        }
                    ])
                    if manifest_path.exists():
                        pl.concat([pl.read_parquet(manifest_path), manifest_row]).write_parquet(manifest_path)
                    else:
                        manifest_row.write_parquet(manifest_path)
                except Exception as _e:
                    logger.warning(f"Manifest append failed for {symbol}: {_e}")
            
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _add_forward_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add same-session forward returns with hard cap ±0.20 for realistic intraday distribution"""
        # Adjust horizon based on data timeframe
        if self.timeframe == "15m":
            H = 1  # 15 minutes in 15-minute bars (minimal lookahead)
        elif self.timeframe == "1h":
            H = 1  # 1 hour in 1-hour bars (1 day forward return)
        elif self.timeframe == "30m":
            H = 2  # 1 hour in 30-minute bars
        elif self.timeframe == "5m":
            H = 12  # 1 hour in 5-minute bars
        else:
            H = 1  # Default fallback
        
        # Use adjusted close if available, otherwise regular close
        price_col = "close_adj" if "close_adj" in df.columns else "close"
        
        # Add ny_date for same-session filtering
        df = df.with_columns([
            pl.col("ts").dt.date().alias("ny_date")
        ])
        
        # Get future close with same-session check using shift
        df = df.with_columns([
            pl.col("ts").shift(-H).over("symbol").alias("ts_fut"),
            pl.col(price_col).shift(-H).over("symbol").alias("close_tpH"),
            pl.col("ny_date").shift(-H).over("symbol").alias("ny_date_tpH")
        ])
        
        # Calculate same-session forward return with hard cap
        df = df.with_columns([
            pl.col(price_col).alias("close_t"),
            pl.col("ny_date").eq(pl.col("ny_date_tpH")).alias("same_session"),
            ((pl.col("close_tpH") / pl.col(price_col)) - 1.0).alias("y_raw")
        ])
        
        # Filter to same-session only and apply hard cap
        df = df.filter(pl.col("same_session")).with_columns([
            pl.col("y_raw").clip(-0.20, 0.20).alias("fwd_ret_1d"),
            pl.lit("fraction").alias("units"),
            pl.lit(price_col).alias("price_col_used"),
            pl.lit(price_col == "close_adj").alias("adjusted_used"),
            pl.lit(H).alias("horizon_bars")
        ])
        
        # Clean up temporary columns
        df = df.drop(["ts_fut", "ny_date_tpH", "same_session", "y_raw"])
        
        return df.with_columns([
            pl.col("close_t").cast(pl.Float32),
            pl.col("close_tpH").cast(pl.Float32),
            pl.col("fwd_ret_1d").cast(pl.Float32),
            pl.col("horizon_bars").cast(pl.Int32),
            pl.col("units"),  # Already Utf8
            pl.col("adjusted_used"),  # Already Boolean
            pl.col("price_col_used")  # Already Utf8
        ])

def main():
    # Initialize logging manager (memory manager used from builder to avoid duplicate init banners)
    logging_manager = CentralLoggingManager()
    logger = logging_manager.setup_script_logging("streaming_feature_builder")
    
    try:
        parser = argparse.ArgumentParser(description="Streaming Feature Builder")
        parser.add_argument("--config", default="config/feature_store_comprehensive.yaml", help="Feature store config")
        parser.add_argument("--universe", default="config/universe_subset_200.yaml", help="Universe config")
        parser.add_argument("--input", default="data/polygon/bars/interval=5m/symbol=*/date=*/*.parquet", help="Input pattern")
        parser.add_argument("--output", default="liquid_1h_features", help="Feature name (will create features/<name>/ structure)")
        
        args = parser.parse_args()
        
        # Find input files
        input_paths = list(Path(".").glob(args.input))
        if not input_paths:
            logger.error(f"No input files found matching pattern: {args.input}")
            return 1
        
        # Build features
        builder = StreamingFeatureBuilder(args.config)

        # Single memory manager usage (avoid duplicate init logs)
        memory_manager = builder.memory_manager
        if not memory_manager.check_memory("feature_building"):
            logger.error("Insufficient memory available. Exiting.")
            return 1
        logger.info(f"Memory status at start: {memory_manager.get_system_memory_usage()}")
        logger.info(f"System memory usage: {memory_manager.get_system_memory_usage()}")
        builder.build_features([str(p) for p in input_paths], args.output, args.universe, args.input)
        
        logger.info(f"✅ Features built successfully!")
        logger.info(f"📁 Output location: storage/features/{args.output}/")
        logger.info(f"🔍 Check: ls -la storage/features/{args.output}/")
        
        logger.info("Feature building completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error during feature building: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
