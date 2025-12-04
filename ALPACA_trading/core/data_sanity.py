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
Data Sanity Layer
Comprehensive data validation and repair for all market data sources.
© 2025 Jennifer — Canary ID: aurora.lab:57c2a0f3
"""


import hashlib
import logging
import weakref
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from core.data_sanity.columnmap import map_ohlcv
from core.data_sanity.group import enforce_groupwise_time_order

logger = logging.getLogger(__name__)


# Error codes for data sanity validation
NEGATIVE_PRICES = "NEGATIVE_PRICES"
OHLC_INVARIANT = "OHLC_INVARIANT"
LOOKAHEAD = "LOOKAHEAD"
FUTURE_DATA = "FUTURE_DATA"
NONFINITE = "NONFINITE_VALUES"
DATETIME_INDEX_INVALID = "DATETIME_INDEX_INVALID"
REPAIRS_APPLIED = "REPAIRS_APPLIED"
WF_NEGATIVE = "WF_NEGATIVE"
WF_SEED_DIFF = "WF_SEED_DIFF"


class DataSanityError(Exception):
    """Exception raised for data sanity violations."""
    pass


def estring(code: str, detail: str) -> str:
    """Create consistent error string format for tests to regex."""
    return f"{code}: {detail}"


# columns the pipeline is allowed to create post-split (never in raw features)
_ALLOWED_PIPELINE_COLS = frozenset({"Returns", "Label", "Target", "y"})

# Standard OHLCV columns that should never be flagged as lookahead
_STANDARD_OHLCV_COLS = frozenset({"Open", "High", "Low", "Close", "Volume", "Adj Close"})


def _has_future_shift(col: pd.Series) -> bool:
    """Heuristic: detect if a column contains future-shifted data."""
    s = pd.to_numeric(col, errors="coerce")
    if s.isna().all() or len(s) < 10:
        return False
    
    # Look for specific patterns that indicate future leakage:
    # 1. Perfect correlation with future values (but not just monotonic sequences)
    fwd = s.shift(-1)
    corr_fwd = s.corr(fwd)
    
    # Only flag if there's perfect correlation AND the values aren't just sequential
    if corr_fwd is not None and corr_fwd > 0.999:
        # Check if this is just a sequential/monotonic series (which is normal for prices)
        is_sequential = s.diff().std() < 1e-10  # Very low variance in differences
        is_monotonic = s.is_monotonic_increasing or s.is_monotonic_decreasing
        
        # Only flag if it's NOT a simple sequential/monotonic pattern
        if not (is_sequential or is_monotonic):
            return True
    
    # 2. Look for values that match exactly at different time offsets
    for offset in [1, 2, 3]:  # Check 1-3 period shifts
        shifted = s.shift(-offset)
        exact_matches = (s == shifted).sum()
        if exact_matches > len(s) * 0.5:  # More than 50% exact matches
            return True
    
    return False


def detect_lookahead(df: pd.DataFrame, *, feature_cols=None) -> list[str]:
    """Detect lookahead contamination in DataFrame columns."""
    cols = list(feature_cols) if feature_cols else list(df.columns)
    offenders = []
    for c in cols:
        # Skip allowed pipeline columns and standard OHLCV columns
        if c in _ALLOWED_PIPELINE_COLS or c in _STANDARD_OHLCV_COLS:
            continue
        # obvious patterns that suggest future/lookahead data
        import re
        if re.search(r"(t\+1|lead|future|next|_tp1|_tplus1|lookahead|forward)", c, re.I):
            offenders.append(c)  # keep scanning others
            continue
        try:
            if _has_future_shift(df[c]):
                offenders.append(c)
        except Exception:
            # non-numeric/short series; ignore
            pass
    return offenders


def canonicalize_datetime_index(df: pd.DataFrame, ts_col: str | None = None, coerce_utc: bool = True) -> pd.DataFrame:
    """Canonicalize and repair datetime index."""
    if ts_col:
        idx = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            # try to interpret index
            idx = pd.to_datetime(df.index, errors="coerce", utc=True)
        else:
            idx = df.index.tz_localize(None).tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")

    if isinstance(idx, pd.DatetimeIndex) and not idx.isna().all():
        # clean/repair
        idx = pd.DatetimeIndex(idx)
        # drop NaT rows
        keep = ~idx.isna()
        df = df.loc[keep].copy()
        idx = idx[keep]
        # sort, dedupe
        df = df.iloc[idx.argsort()].copy()
        idx = idx.sort_values()
        # drop duplicates (keep first)
        if idx.has_duplicates:
            firsts = ~idx.duplicated(keep="first")
            df = df.iloc[firsts.values].copy()
            idx = idx[firsts]
        df.index = pd.DatetimeIndex(idx, tz="UTC", name=df.index.name or "timestamp")
        return df

    raise DataSanityError(estring(DATETIME_INDEX_INVALID, "No valid datetime index found"))


def safe_tz(index) -> str | None:
    """Safely get timezone from index."""
    return getattr(index, "tz", None).zone if hasattr(getattr(index, "tz", None), "zone") else None


def assert_ohlc_invariants(df: pd.DataFrame, profile) -> pd.DataFrame:
    """Assert and optionally repair OHLC invariants."""
    mapping = map_ohlcv(df)
    required = ("Open", "High", "Low", "Close")
    if not all(k in mapping for k in required):
        return df  # Not enough OHLC data to validate
    
    o, h, l, c = (df[mapping["Open"]], df[mapping["High"]], df[mapping["Low"]], df[mapping["Close"]])
    neg = (o < 0).any() or (h < 0).any() or (l < 0).any() or (c < 0).any()
    
    if neg and profile.strict:
        raise DataSanityError(estring(NEGATIVE_PRICES, "OHLC invariant violation: negative values present"))
    
    if neg and profile.allow_repairs:
        df = df.copy()
        df[mapping["Open"]] = df[mapping["Open"]].clip(lower=0.0)
        df[mapping["High"]] = df[mapping["High"]].clip(lower=0.0)
        df[mapping["Low"]] = df[mapping["Low"]].clip(lower=0.0)
        df[mapping["Close"]] = df[mapping["Close"]].clip(lower=0.0)
    
    # High >= max(Open,Close); Low <= min(Open,Close)
    bad = (h < o.combine(c, max)) | (l > o.combine(c, min))
    if bad.any():
        if profile.strict:
            raise DataSanityError(estring(OHLC_INVARIANT, "OHLC relation violated"))
        if profile.allow_repairs:
            df = df.copy()
            df[mapping["High"]] = df[[mapping["High"], mapping["Open"], mapping["Close"]]].max(axis=1)
            df[mapping["Low"]] = df[[mapping["Low"], mapping["Open"], mapping["Close"]]].min(axis=1)
    
    return df


def repair_nonfinite_ohlc(df: pd.DataFrame, profile) -> pd.DataFrame:
    """Repair non-finite values in OHLC data."""
    mapping = map_ohlcv(df)
    cols = [mapping[k] for k in ("Open", "High", "Low", "Close", "Volume") if k in mapping]
    if not cols:
        return df
    
    # Ensure numeric before checking isfinite
    df_numeric = df.copy()
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df_numeric[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df_numeric[col] = df[col]
    
    bad = ~np.isfinite(df_numeric[cols]).all(axis=None)
    if bad and not profile.allow_repairs:
        raise DataSanityError(estring(NONFINITE, f"Non-finite values in {cols}"))
    
    if bad:
        df = df.copy()
        df[cols] = df_numeric[cols].ffill().bfill()
    
    return df


@dataclass
class ValidationResult:
    """Result of data validation with detailed information."""

    repairs: list[str]  # List of repairs performed
    flags: list[str]  # List of flags raised (e.g., ["lookahead_detected"])
    outliers: int  # Number of outliers detected
    rows_in: int  # Number of input rows
    rows_out: int  # Number of output rows
    profile: str  # Profile used for validation
    validation_time: float  # Time taken for validation


@dataclass(frozen=True)
class SanityProfile:
    name: str = "default"
    strict: bool = False
    allow_repairs: bool = True
    # e.g. {"price_limits", "guard_validation"}
    fail_on: set[str] = field(default_factory=set)


def should_raise(profile: SanityProfile, flags: set[str], unrepaired_issues: set[str]) -> bool:
    if profile.strict and (flags or unrepaired_issues):
        return True
    if profile.fail_on & (flags | unrepaired_issues):
        return True
    return False


def _convert_flags_to_set(flags) -> set[str]:
    """Convert flags to set, handling both list and set inputs."""
    if isinstance(flags, set):
        return flags
    elif isinstance(flags, list):
        return set(flags)
    else:
        return set()


def _iter_chunks(n, chunk_size=1_000_000):
    start = 0
    while start < n:
        end = min(n, start + chunk_size)
        yield slice(start, end)
        start = end


def fast_isfinite_nd(values) -> bool:
    # values is a 2D float ndarray
    try:
        return np.isfinite(values).all()
    except MemoryError:
        for slc in _iter_chunks(values.shape[0]):
            if not np.isfinite(values[slc]).all():
                return False
        return True


def validate_index(idx: pd.Index):
    if not pd.api.types.is_datetime64_any_dtype(idx):
        raise DataSanityError(estring(DATETIME_INDEX_INVALID, "No valid datetime index found"))
    # normalize tz; sort for monotonicity; re-check uniqueness/monotonic
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    if not idx.is_monotonic_increasing or not idx.is_unique:
        # offer non-strict repair path
        return False, idx
    return True, idx


def compute_returns(close: pd.Series) -> pd.Series:
    # Ensure float and aligned; first element NaN by definition
    c = pd.to_numeric(close, errors="coerce").astype("float64")
    r = c.pct_change()
    return r


def verify_returns_identity(close: pd.Series, r: pd.Series, atol=1e-9):
    # (1+r).prod()-1 == close[-1]/close[0]-1  after dropping NaNs
    r_eff = (1.0 + r.dropna()).prod() - 1.0
    valid = close.dropna()
    if len(valid) < 2:
        return True  # degenerate; don't fail the property on tiny samples
    c_eff = float(valid.iloc[-1]) / float(valid.iloc[0]) - 1.0
    return np.isfinite(r_eff) and np.isfinite(c_eff) and abs(r_eff - c_eff) <= atol


NUMERIC_PRICE_COLS = ("Open", "High", "Low", "Close", "Volume")


def coerce_numeric_cols(df: pd.DataFrame, profile: SanityProfile) -> pd.DataFrame:
    # smart coercion for non-strict; strict raises on any coercion need
    coerced = df.copy()
    need_coercion = set()
    for c in NUMERIC_PRICE_COLS:
        if c in coerced:
            if not pd.api.types.is_numeric_dtype(coerced[c]):
                need_coercion.add(c)
    if need_coercion and profile.strict:
        raise DataSanityError(f"Non-numeric in numeric cols (strict): {sorted(need_coercion)}")
    if need_coercion and profile.allow_repairs:
        for c in need_coercion:
            coerced[c] = pd.to_numeric(coerced[c], errors="coerce")
    return coerced


def synth_missing_cols(df: pd.DataFrame, profile: SanityProfile) -> pd.DataFrame:
    # already doing this, but ensure determinism (no RNG)
    out = df.copy()
    need = [c for c in ["Open", "High", "Low", "Close"] if c not in out]
    if need and profile.strict:
        raise DataSanityError(f"Missing required columns (strict): {need}")
    if "Close" in out:
        base = pd.to_numeric(out["Close"], errors="coerce")
    elif "Open" in out:
        base = pd.to_numeric(out["Open"], errors="coerce")
    else:
        # make a constant baseline to stay deterministic
        base = pd.Series(1.0, index=out.index, dtype="float64")
    for c in need:
        out[c] = base  # deterministic fill; later checks can enforce O/H/L bounds
    return out


# Lightweight CI-facing result (non-breaking: new types and methods)
@dataclass
class SanityViolation:
    code: str
    details: str


@dataclass
class SanityCheckResult:
    mode: str
    violations: list[SanityViolation]
    ok: bool

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "ok": self.ok,
            "violations": [vi.__dict__ for vi in self.violations],
        }

    def summary(self) -> str:
        if self.ok or not self.violations:
            return "data_sanity_ok"
        v = self.violations[0]
        return f"data_sanity_violation[{v.code}]: {v.details}"


class DataSanityGuard:
    """
    Runtime guard to ensure DataFrames are validated before use.

    This guard tracks DataFrame objects and ensures they've been validated
    before being consumed by downstream components.
    """

    def __init__(self, df: pd.DataFrame):
        """Initialize guard for a DataFrame."""
        self._df_id = id(df)
        self._df_hash = self._compute_hash(df)
        self._validated = False
        self._validation_time = None
        self._symbol = None

    def _compute_hash(self, df: pd.DataFrame) -> str:
        """Compute a hash of the DataFrame for tracking."""
        # Hash the shape, dtypes, and first/last few values
        hash_data = f"{df.shape}_{df.dtypes.to_dict()}"
        if len(df) > 0:
            hash_data += f"_{df.iloc[0].to_dict()}_{df.iloc[-1].to_dict()}"
        return hashlib.md5(hash_data.encode()).hexdigest()[:8]

    def mark_validated(self, df: pd.DataFrame, symbol: str = None):
        """Mark this DataFrame as validated."""
        if id(df) == self._df_id:
            self._validated = True
            self._validation_time = datetime.now()
            self._symbol = symbol
            logger.debug(
                f"DataSanityGuard: Marked DataFrame {self._df_id} as validated for {symbol}"
            )
        else:
            logger.warning("DataSanityGuard: DataFrame ID mismatch during validation")

    def assert_validated(self, context: str = "unknown"):
        """Assert that this DataFrame has been validated."""
        if not self._validated:
            raise DataSanityError(
                f"DataSanityGuard: DataFrame used before validation in {context}. "
                f"ID: {self._df_id}, Hash: {self._df_hash}"
            )
        logger.debug(f"DataSanityGuard: DataFrame {self._df_id} validated for {context}")

    def get_status(self) -> dict:
        """Get guard status information."""
        return {
            "df_id": self._df_id,
            "df_hash": self._df_hash,
            "validated": self._validated,
            "validation_time": self._validation_time,
            "symbol": self._symbol,
        }


# Global registry of guards
_guard_registry = weakref.WeakValueDictionary()


def attach_guard(df: pd.DataFrame) -> DataSanityGuard:
    """Attach a DataSanityGuard to a DataFrame."""
    guard = DataSanityGuard(df)
    _guard_registry[id(df)] = guard
    df._sanity_guard = guard
    return guard


def get_guard(df: pd.DataFrame) -> DataSanityGuard | None:
    """Get the DataSanityGuard for a DataFrame."""
    return getattr(df, "_sanity_guard", None)


def assert_validated(df: pd.DataFrame, context: str = "unknown"):
    """Assert that a DataFrame has been validated."""
    guard = get_guard(df)
    if guard:
        guard.assert_validated(context)
    else:
        # If no guard, create one and mark as validated (backward compatibility)
        logger.warning(f"DataSanityGuard: No guard found for DataFrame in {context}, creating one")
        attach_guard(df)


class DataSanityValidator:
    """
    Comprehensive data validation and repair for market data.

    Validates:
    - Time series integrity (monotonic, UTC, no duplicates)
    - Price data sanity (finite, positive, reasonable bounds)
    - OHLC consistency (low <= {open,close} <= high)
    - Volume data validity
    - Outlier detection and repair
    """

    def __init__(self, config_path: str = "config/data_sanity.yaml", profile: str = "default"):
        """
        Initialize DataSanityValidator.

        Args:
            config_path: Path to configuration file
            profile: Profile to use ("default" or "strict")
        """
        self.config = self._load_config(config_path)
        
        self.profile = profile
        self.profile_config = self._get_profile_config(profile)
        
        # Update profile if repair_mode changes
        self._update_profile_for_repair_mode()
        self.repair_count = 0
        self.outlier_count = 0
        self.validation_failures = []
        
        # Track original user columns to avoid scanning derived columns for lookahead
        self.original_columns = set()
        self.ignore_lookahead_columns = {"Returns"}  # Columns to ignore in lookahead detection

        logger.info(
            f"Initialized DataSanityValidator with profile: {profile}, mode: {self.profile_config['mode']}"
        )

    def _update_profile_for_repair_mode(self):
        """Update profile if repair_mode is 'fail'."""
        if self.config.get("repair_mode") == "fail" and self.profile != "strict":
            self.profile = "strict"
            self.profile_config = self._get_profile_config("strict")
            logger.info("Repair mode is 'fail', switching to strict profile")

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded data sanity config from {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}")
            return self._get_default_config()

    def _get_profile_config(self, profile: str) -> dict:
        """Get profile-specific configuration."""
        profiles = self.config.get("profiles", {})
        if profile not in profiles:
            logger.warning(f"Profile '{profile}' not found, using default")
            profile = "default"

        return profiles.get(profile, {})

    @staticmethod
    def in_ci() -> bool:
        import os

        return str(os.getenv("CI", "")).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_tz(idx: pd.DatetimeIndex, tz: str) -> pd.DatetimeIndex:
        if getattr(idx, "tz", None) is None:
            return idx.tz_localize(tz)
        return idx.tz_convert(tz)

    def validate_dataframe_fast(self, df: pd.DataFrame, profile: str) -> SanityCheckResult:
        import numpy as np

        vio: list[SanityViolation] = []
        cfg_raw = self._get_profile_config(profile)
        mode = cfg_raw.get("mode", "warn")
        enforced = mode == "enforce"
        if df is None or len(df) == 0:
            return SanityCheckResult(
                mode=mode, violations=[SanityViolation("EMPTY_SERIES", "no rows")], ok=False
            )
        cfg = cfg_raw
        # timezone
        tz = cfg.get("tz")
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else None
        if tz and isinstance(idx, pd.DatetimeIndex):
            try:
                new_idx = self._normalize_tz(idx, tz)
                if not new_idx.equals(idx):
                    df = df.copy()
                    df.index = new_idx
                    idx = new_idx
            except Exception:
                pass
        # monotonic & duplicates
        require_monotonic = cfg.get("require_monotonic_dates", False) or enforced
        if require_monotonic and idx is not None and not idx.is_monotonic_increasing:
            vio.append(SanityViolation("NON_MONO_INDEX", "index not non-decreasing"))
        forbid_duplicates = cfg.get("forbid_duplicates", False) or enforced
        if forbid_duplicates and idx is not None and idx.has_duplicates:
            n_dupes = int(idx.duplicated().sum())
            vio.append(SanityViolation("DUP_TS", f"{n_dupes} duplicate stamps"))
        # numeric inf/nan
        nums = df.select_dtypes(include=[np.number])
        arr = (
            np.asarray(nums.to_numpy(dtype="float64"), dtype="float64")
            if nums.shape[1]
            else np.empty((len(df), 0))
        )
        forbid_inf = cfg.get("forbid_infinite", False) or enforced
        if forbid_inf and arr.size and np.isinf(arr).any():
            vio.append(SanityViolation("INF_VALUES", "infinite values present"))
        max_nan = cfg.get("max_nan_pct", (0.0 if enforced else None))
        if max_nan is not None and arr.size:
            nan_pct = float(np.isnan(arr).mean())
            if nan_pct > max_nan:
                vio.append(
                    SanityViolation("NAN_VALUES", f"NaN fraction {nan_pct:.4f} > {max_nan:.4f}")
                )
        return SanityCheckResult(mode=mode, violations=vio, ok=(len(vio) == 0))

    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "profiles": {
                "default": {
                    "mode": "warn",
                    "price_max": 1000000.0,
                    "allow_repairs": True,
                    "allow_winsorize": True,
                    "allow_clip_prices": True,
                    "allow_fix_ohlc": True,
                    "allow_drop_dupes": True,
                    "allow_ffill_nans": True,
                    "tolerate_outliers_after_repair": True,
                    "fail_on_lookahead_flag": False,
                    "fail_if_any_repair": False,
                },
                "strict": {
                    "mode": "fail",
                    "price_max": 1000000.0,
                    "allow_repairs": False,
                    "allow_winsorize": False,
                    "allow_clip_prices": False,
                    "allow_fix_ohlc": False,
                    "allow_drop_dupes": False,
                    "allow_ffill_nans": False,
                    "tolerate_outliers_after_repair": False,
                    "fail_on_lookahead_flag": True,
                    "fail_if_any_repair": True,
                },
            },
            "price_limits": {
                "max_price": 1000000.0,
                "min_price": 0.01,
                "max_daily_return": 0.3,
                "max_volume": 1000000000000,
            },
            "ohlc_validation": {
                "max_high_low_spread": 0.4,
                "require_ohlc_consistency": True,
                "allow_zero_volume": False,
            },
            "outlier_detection": {
                "z_score_threshold": 4.0,
                "mad_threshold": 3.0,
                "min_obs_for_outlier": 20,
            },
            "repair_mode": "warn",
            "winsorize_quantile": 0.01,
            "time_series": {
                "require_monotonic": True,
                "require_utc": True,
                "max_gap_days": 30,
                "allow_duplicates": False,
            },
            "returns": {
                "method": "log_close_to_close",
                "min_periods": 2,
                "fill_method": "forward",
            },
            "logging": {
                "log_repairs": True,
                "log_outliers": True,
                "log_validation_failures": True,
                "summary_level": "INFO",
            },
        }

    def validate_and_repair(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN"
    ) -> tuple[pd.DataFrame, ValidationResult]:
        """
        Validate and repair market data with strict invariants.

        Args:
            data: DataFrame with OHLCV data
            symbol: Symbol name for logging

        Returns:
            Tuple of (cleaned DataFrame, ValidationResult)
        """
        import time

        start_time = time.time()
        
        # Update profile if repair_mode changed
        self._update_profile_for_repair_mode()

        if data.empty:
            if self.profile_config.get("allow_repairs", True):
                logger.warning(f"Empty data for {symbol}")
                result = ValidationResult(
                    repairs=[],
                    flags=[],
                    outliers=0,
                    rows_in=0,
                    rows_out=0,
                    profile=self.profile,
                    validation_time=time.time() - start_time,
                )
                return data, result
            else:
                raise DataSanityError(f"{symbol}: Empty data not allowed in strict mode")

        original_shape = data.shape
        logger.info(f"Validating {symbol}: {original_shape[0]} rows, {original_shape[1]} columns")

        # Initialize validation result
        repairs = []
        flags = []
        outliers = 0

        # Reset counters
        self.repair_count = 0
        self.outlier_count = 0
        self.validation_failures = []

        # Capture original user columns before any modifications
        self.original_columns = set(data.columns)
        
        # Make a copy to avoid modifying original
        clean_data = data.copy()

        # Transfer guard from original to copy if it exists
        original_guard = get_guard(data)
        if original_guard:
            # Transfer the guard and update its DataFrame ID to match the copy
            clean_data._sanity_guard = original_guard
            original_guard._df_id = id(clean_data)  # Update to new DataFrame ID
            original_guard._df_hash = hash(str(clean_data.values.tobytes()))
        else:
            # Attach guard if not present
            attach_guard(clean_data)

        # 1. Validate and repair time series
        if self.profile_config.get("mode") == "fail":
            clean_data, time_repairs, time_flags = self._validate_time_series_strict(clean_data, symbol)
        else:
            clean_data, time_repairs, time_flags = self._validate_time_series(clean_data, symbol)
        repairs.extend(time_repairs)
        flags.extend(time_flags)

        # 1.5. Enforce groupwise time order (for symbol-grouped data)
        symbol_col = self.profile_config.get("symbol_column")
        if symbol_col and symbol_col in clean_data.columns:
            clean_data = enforce_groupwise_time_order(clean_data, symbol_col)
            repairs.append("enforced_groupwise_time_order")

        # 2. Validate and repair price data
        if self.profile_config.get("mode") == "fail":
            clean_data, price_repairs, price_flags = self._validate_price_data_strict(
                clean_data, symbol
            )
        else:
            clean_data, price_repairs, price_flags = self._validate_price_data(
                clean_data, symbol
            )
        repairs.extend(price_repairs)
        flags.extend(price_flags)

        # 3. Validate and repair OHLC consistency
        if self.profile_config.get("mode") == "fail":
            clean_data, ohlc_repairs, ohlc_flags = self._validate_ohlc_consistency_strict(
                clean_data, symbol
            )
        else:
            clean_data, ohlc_repairs, ohlc_flags = self._validate_ohlc_consistency(
                clean_data, symbol
            )
        repairs.extend(ohlc_repairs)
        flags.extend(ohlc_flags)

        # 4. Validate and repair volume data
        if self.profile_config.get("mode") == "fail":
            clean_data, volume_repairs, volume_flags = self._validate_volume_data_strict(
                clean_data, symbol
            )
        else:
            clean_data, volume_repairs, volume_flags = self._validate_volume_data(
                clean_data, symbol
            )
        repairs.extend(volume_repairs)
        flags.extend(volume_flags)

        # 5. Detect and repair outliers
        if self.profile_config.get("mode") == "fail":
            (
                clean_data,
                outlier_repairs,
                outlier_count,
            ) = self._detect_and_repair_outliers_strict(clean_data, symbol)
        else:
            (
                clean_data,
                outlier_repairs,
                outlier_count,
            ) = self._detect_and_repair_outliers(clean_data, symbol)
        repairs.extend(outlier_repairs)
        outliers = outlier_count

        # 6. Calculate clean returns
        if self.profile_config.get("mode") == "fail":
            clean_data, return_repairs, return_flags = self._calculate_returns_strict(
                clean_data, symbol
            )
        else:
            clean_data, return_repairs, return_flags = self._calculate_returns(
                clean_data, symbol
            )
        repairs.extend(return_repairs)
        flags.extend(return_flags)

        # 7. Final validation checks
        if self.profile_config.get("mode") == "fail":
            clean_data, final_repairs, final_flags = self._final_validation_checks_strict(
                clean_data, symbol
            )
        else:
            clean_data, final_repairs, final_flags = self._final_validation_checks(
                clean_data, symbol
            )
        repairs.extend(final_repairs)
        flags.extend(final_flags)

        # Check for lookahead contamination
        lookahead_detected = self._detect_lookahead_contamination(clean_data)
        if lookahead_detected:
            flags.append("lookahead_detected")
            if self.profile_config.get("fail_on_lookahead_flag", False):
                raise DataSanityError(f"{symbol}: Lookahead contamination detected")

        # In strict mode: fail if any repairs occurred
        if self.profile_config.get("fail_if_any_repair", False) and repairs:
            raise DataSanityError(f"{symbol}: Repairs occurred in strict mode: {repairs}")

        # Also fail if repairs are not allowed but repairs were attempted
        if not self.profile_config.get("allow_repairs", True) and repairs:
            raise DataSanityError(
                f"{symbol}: Repairs not allowed but repairs were attempted: {repairs}"
            )

        # Mark as validated - use the guard from the clean_data
        guard = get_guard(clean_data)
        if guard:
            guard.mark_validated(clean_data, symbol)
        else:
            # If no guard exists, create one and mark as validated
            attach_guard(clean_data)
            guard = get_guard(clean_data)
            guard.mark_validated(clean_data, symbol)

        # Create validation result
        validation_time = time.time() - start_time
        result = ValidationResult(
            repairs=repairs,
            flags=flags,
            outliers=outliers,
            rows_in=original_shape[0],
            rows_out=len(clean_data),
            profile=self.profile,
            validation_time=validation_time,
        )

        # Log summary
        self._log_validation_summary(symbol, original_shape, clean_data.shape)

        return clean_data, result

    def _validate_time_series_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate time series with strict profile rules."""
        repairs = []
        flags = []

        # Check for datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            if self.profile_config.get("allow_repairs", True):
                # Try to convert to datetime
                try:
                    data.index = pd.to_datetime(data.index)
                    repairs.append("converted_to_datetime_index")
                except Exception:
                    raise DataSanityError(f"{symbol}: No valid datetime index found") from None
            else:
                raise DataSanityError(f"{symbol}: No valid datetime index found")

        # Check for monotonic index and duplicates separately
        if data.index.has_duplicates:
            if self.profile_config.get("allow_drop_dupes", True):
                # Remove duplicates and sort
                data = data[~data.index.duplicated(keep="first")].sort_index()
                repairs.append("dropped_duplicates_and_sorted")
            else:
                raise DataSanityError(f"{symbol}: Duplicate timestamps detected")

        # Check for monotonic index after duplicate handling
        if not data.index.is_monotonic_increasing:
            if self.profile_config.get("allow_drop_dupes", True):
                # Sort the index
                data = data.sort_index()
                repairs.append("sorted_index")
            else:
                raise DataSanityError(f"{symbol}: Index is not monotonic")

        # Check for timezone consistency
        if data.index.tz is None:
            if self.profile_config.get("allow_repairs", True):
                data.index = data.index.tz_localize(UTC)
                repairs.append("localized_to_utc")
            else:
                raise DataSanityError(f"{symbol}: Naive timezone not allowed in strict mode")
        elif data.index.tz != UTC:
            if self.profile_config.get("allow_repairs", True):
                try:
                    data.index = data.index.tz_convert(UTC)
                    repairs.append("converted_to_utc")
                except Exception as e:
                    # Handle mixed timezone data by localizing to UTC
                    logger.warning(
                        f"Timezone conversion failed for {symbol}: {e}, localizing to UTC"
                    )
                    data.index = data.index.tz_localize(UTC)
                    repairs.append("localized_mixed_timezone_to_utc")
            else:
                raise DataSanityError(f"{symbol}: Non-UTC timezone not allowed in strict mode")

        return data, repairs, flags

    def _validate_time_series(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate and repair time series integrity."""
        repairs = []
        flags = []
        ts_config = self.config["time_series"]

        # Ensure we have a proper index
        if not isinstance(data.index, pd.DatetimeIndex):
            if "Date" in data.columns:
                data = data.set_index("Date")
            else:
                self._handle_validation_failure(f"{symbol}: No valid datetime index found")
                return data, repairs, flags

        # Check for monotonic timestamps
        if ts_config["require_monotonic"]:
            if not data.index.is_monotonic_increasing:
                if ts_config["allow_duplicates"]:
                    # Sort and keep duplicates
                    data = data.sort_index()
                else:
                    # Remove duplicates and sort
                    data = data[~data.index.duplicated(keep="first")].sort_index()
                    repairs.append("removed_duplicate_timestamps")

            # Ensure monotonic after any repairs
            if not data.index.is_monotonic_increasing:
                data = data.sort_index()

        # Check for UTC timezone
        if ts_config["require_utc"]:
            if data.index.tz is None:
                # Assume UTC if no timezone
                data.index = data.index.tz_localize("UTC")
                repairs.append("localized_timestamps_to_utc")
            elif data.index.tz != UTC:
                # Convert to UTC
                data.index = data.index.tz_convert("UTC")
                repairs.append("converted_timestamps_to_utc")

        # Check for large gaps
        if len(data) > 1:
            gaps = data.index.to_series().diff().dt.days
            large_gaps = gaps > ts_config["max_gap_days"]
            if large_gaps.any():
                gap_count = large_gaps.sum()
                max_gap = gaps.max()
                flags.append(f"large_gaps_detected_{gap_count}")

        return data, repairs, flags

    def _validate_price_data_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], set[str]]:
        """Validate price data with strict profile rules."""
        repairs = []
        flags = set()
        unrepaired = set()

        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Convert string data types to numeric if needed
            if series.dtype == "object":
                try:
                    series = pd.to_numeric(series, errors="coerce")
                    data[col] = series
                    if self.profile_config.get("allow_repairs", True):
                        repairs.append(f"converted_string_to_numeric_in_{col}")
                    else:
                        raise DataSanityError(
                            f"{symbol}: String data type in {col} not allowed in strict mode"
                        )
                except Exception:
                    if not self.profile_config.get("allow_repairs", True):
                        raise DataSanityError(
                            f"{symbol}: Cannot convert string data to numeric in {col}"
                        ) from None
                    else:
                        # Try to extract numeric values from strings
                        series = pd.to_numeric(
                            series.str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
                        )
                        data[col] = series
                        repairs.append(f"extracted_numeric_from_string_in_{col}")

            # Check for negative prices
            negative_prices = series <= 0
            if negative_prices.any():
                flags.add("price_limits")
                if self.profile_config.get("allow_clip_prices", True) and not self.profile_config.get("strict", False):
                    # Clip negative prices to minimum
                    min_price = self.config["price_limits"]["min_price"]
                    data.loc[negative_prices, col] = min_price
                    repairs.append(f"clipped_negative_prices_in_{col}")
                    # Recheck if any remain negative
                    if (data[col] <= 0).any():
                        unrepaired.add("price_limits")
                else:
                    unrepaired.add("price_limits")

            # Check for extreme prices
            max_price = self.profile_config.get(
                "price_max", self.config["price_limits"]["max_price"]
            )
            extreme_prices = series > max_price
            if extreme_prices.any():
                if self.profile_config.get("allow_clip_prices", True):
                    # Clip extreme prices
                    data.loc[extreme_prices, col] = max_price
                    repairs.append(f"clipped_extreme_prices_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: Prices > {max_price} in {col}")

            # Check for non-finite values (only for numeric data)
            if not pd.api.types.is_numeric_dtype(series):
                # Always fail for strict validation - no coercion allowed
                raise DataSanityError(f"{symbol}: Column {col} has non-numeric dtype: {series.dtype}")
            
            non_finite = ~np.isfinite(series)
            if non_finite.any():
                if self.profile_config.get("allow_ffill_nans", True):
                    # Forward fill non-finite values
                    data[col] = series.ffill().bfill()
                    repairs.append(f"forward_filled_non_finite_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: Non-finite values in {col}")

        # Check if we should raise based on profile settings
        profile = SanityProfile(
            name=self.profile,
            strict=self.profile_config.get("strict", False),
            allow_repairs=self.profile_config.get("allow_repairs", True),
            fail_on=set(self.profile_config.get("fail_on", []))
        )
        
        flags_set = _convert_flags_to_set(flags)
        if should_raise(profile, flags_set, unrepaired):
            raise DataSanityError(
                estring(NEGATIVE_PRICES, f"{symbol}: violations={sorted(flags_set)} unrepaired={sorted(unrepaired)}")
            )
        
        return data, repairs, flags

    def _validate_price_data(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate and repair price data."""
        repairs = []
        flags = []
        price_config = self.config["price_limits"]

        # Standardize column names
        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Check for non-finite values (only for numeric data)
            if not pd.api.types.is_numeric_dtype(series):
                # Try to coerce to numeric if repair mode allows
                if self.config["repair_mode"] == "fail":
                    raise DataSanityError(f"{symbol}: Column {col} has non-numeric dtype: {series.dtype}")
                else:
                    # Try to coerce to numeric
                    try:
                        numeric_series = pd.to_numeric(series, errors='coerce')
                        data[col] = numeric_series
                        series = numeric_series
                        repairs.append(f"coerced_{col}_to_numeric")
                    except Exception:
                        raise DataSanityError(f"{symbol}: Cannot coerce column {col} to numeric: {series.dtype}")
            
            non_finite = ~np.isfinite(series)
            if non_finite.any():
                count = non_finite.sum()
                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(f"{symbol}: {count} non-finite values in {col}")
                elif self.config["repair_mode"] == "drop":
                    data = data[~non_finite]
                    repairs.append(f"dropped_non_finite_values_from_{col}")
                else:
                    # Forward fill non-finite values
                    # Handle edge cases where NaN might be at the beginning or end
                    filled_series = series.ffill().bfill()
                    # If there are still NaN values, fill with a reasonable default
                    if filled_series.isna().any():
                        # Use median of non-NaN values as default
                        median_val = series.dropna().median()
                        if pd.isna(median_val):
                            median_val = 100.0  # Fallback default
                        filled_series = filled_series.fillna(median_val)
                        repairs.append(f"used_median_to_fill_nan_in_{col}")
                    data[col] = filled_series
                    repairs.append(f"forward_filled_non_finite_in_{col}")

            # Check price bounds
            too_high = series > price_config["max_price"]
            too_low = series < price_config["min_price"]

            if too_high.any() or too_low.any():
                high_count = too_high.sum()
                low_count = too_low.sum()

                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(
                        f"{symbol}: {high_count} prices > {price_config['max_price']}, "
                        f"{low_count} prices < {price_config['min_price']} in {col}"
                    )
                elif self.config["repair_mode"] == "drop":
                    invalid_mask = too_high | too_low
                    data = data.loc[~invalid_mask]
                    repairs.append(f"dropped_invalid_prices_from_{col}")
                else:
                    # Winsorize extreme values
                    if self.config["repair_mode"] == "winsorize":
                        data[col] = series.clip(
                            lower=price_config["min_price"],
                            upper=price_config["max_price"],
                        )
                        repairs.append(f"winsorized_extreme_prices_in_{col}")
                    else:
                        # Default warn mode - clip extreme values
                        data[col] = series.clip(
                            lower=price_config["min_price"],
                            upper=price_config["max_price"],
                        )
                        self._log_repair(
                            f"{symbol}: Clipped {high_count + low_count} extreme prices in {col}"
                        )

        # Final check: ensure all price columns are finite after all repairs
        for col in price_cols:
            if col in data.columns:
                series = data[col]
                # Check for non-finite values (only for numeric data)
                if not pd.api.types.is_numeric_dtype(series):
                    # Final cleanup - always fail if non-numeric at this stage
                    raise DataSanityError(f"{symbol}: Column {col} has non-numeric dtype at final cleanup: {series.dtype}")
                
                non_finite = ~np.isfinite(series)
                if non_finite.any():
                    # Use median of finite values as fallback
                    finite_series = series[np.isfinite(series)]
                    median_val = finite_series.median() if len(finite_series) > 0 else 100.0  # Ultimate fallback

                    data.loc[non_finite, col] = median_val
                    repairs.append(f"final_cleanup_filled_non_finite_in_{col}")

        return data, repairs, flags

    def _validate_ohlc_consistency_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate OHLC consistency with strict profile rules."""
        repairs = []
        flags = []

        if not all(col in data.columns for col in ["Open", "High", "Low", "Close"]):
            return data, repairs, flags

        # Check OHLC invariants
        bad_high = (data["High"] < data[["Open", "Close"]].max(axis=1)).sum()
        bad_low = (data["Low"] > data[["Open", "Close"]].min(axis=1)).sum()

        if bad_high > 0 or bad_low > 0:
            if self.profile_config.get("allow_fix_ohlc", True):
                # Fix OHLC inconsistencies
                data = self._repair_ohlc_inconsistencies(data)
                repairs.append(f"fixed_ohlc_inconsistencies:{bad_high + bad_low}")
            else:
                raise DataSanityError(
                    f"{symbol}: OHLC invariant violation (High < max(Open,Close): {bad_high}, Low > min(Open,Close): {bad_low})"
                )

        return data, repairs, flags

    def _validate_ohlc_consistency(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate and repair OHLC consistency."""
        repairs = []
        flags = []
        ohlc_config = self.config["ohlc_validation"]

        if not ohlc_config["require_ohlc_consistency"]:
            return data, repairs, flags

        price_cols = self._get_price_columns(data)
        if len(price_cols) < 4:  # Need OHLC
            return data, repairs, flags

        # Check OHLC relationships
        inconsistencies = []

        # High should be >= Open, Close
        if "High" in data.columns and "Open" in data.columns:
            high_open_violations = data["High"] < data["Open"]
            if high_open_violations.any():
                inconsistencies.append(f"High < Open: {high_open_violations.sum()}")

        if "High" in data.columns and "Close" in data.columns:
            high_close_violations = data["High"] < data["Close"]
            if high_close_violations.any():
                inconsistencies.append(f"High < Close: {high_close_violations.sum()}")

        # Low should be <= Open, Close
        if "Low" in data.columns and "Open" in data.columns:
            low_open_violations = data["Low"] > data["Open"]
            if low_open_violations.any():
                inconsistencies.append(f"Low > Open: {low_open_violations.sum()}")

        if "Low" in data.columns and "Close" in data.columns:
            low_close_violations = data["Low"] > data["Close"]
            if low_close_violations.any():
                inconsistencies.append(f"Low > Close: {low_close_violations.sum()}")

        # Check high-low spread
        if "High" in data.columns and "Low" in data.columns and "Close" in data.columns:
            spread = (data["High"] - data["Low"]) / data["Close"]
            excessive_spread = spread > ohlc_config["max_high_low_spread"]
            if excessive_spread.any():
                inconsistencies.append(f"Excessive spread: {excessive_spread.sum()}")

        if inconsistencies:
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(
                    f"{symbol}: OHLC inconsistencies: {', '.join(inconsistencies)}"
                )
            else:
                # Repair OHLC inconsistencies
                data = self._repair_ohlc_inconsistencies(data)
                repairs.append("repaired_ohlc_inconsistencies")

        return data, repairs, flags

    def _validate_volume_data_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate volume data with strict profile rules."""
        repairs = []
        flags = []

        volume_col = self._get_volume_column(data)
        if volume_col not in data.columns:
            return data, repairs, flags

        volume = data[volume_col]

        # Check for negative volume
        negative_volume = volume < 0
        if negative_volume.any():
            if self.profile_config.get("allow_clip_prices", True):
                # Make negative volume positive
                data.loc[negative_volume, volume_col] = volume[negative_volume].abs()
                repairs.append("made_negative_volume_positive")
            else:
                raise DataSanityError(f"{symbol}: Negative volume values")

        # Check for excessive volume
        max_volume = self.config["price_limits"]["max_volume"]
        excessive_volume = volume > max_volume
        if excessive_volume.any():
            if self.profile_config.get("allow_clip_prices", True):
                # Cap excessive volume
                data.loc[excessive_volume, volume_col] = max_volume
                repairs.append("capped_excessive_volume")
            else:
                raise DataSanityError(f"{symbol}: Excessive volume values > {max_volume}")

        # Check for zero volume
        if not self.config["ohlc_validation"]["allow_zero_volume"]:
            zero_volume = volume == 0
            if zero_volume.any():
                if self.profile_config.get("allow_repairs", True):
                    # Replace with median volume
                    median_volume = volume[volume > 0].median()
                    if pd.isna(median_volume):
                        median_volume = 1000000  # Default
                    data.loc[zero_volume, volume_col] = median_volume
                    repairs.append("replaced_zero_volume_with_median")
                else:
                    raise DataSanityError(f"{symbol}: Zero volume values")

        return data, repairs, flags

    def _validate_volume_data(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Validate and repair volume data."""
        repairs = []
        flags = []
        volume_col = "Volume"
        if volume_col not in data.columns:
            return data, repairs, flags

        volume = data[volume_col]

        # Check for negative volume
        negative_volume = volume < 0
        if negative_volume.any():
            count = negative_volume.sum()
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(f"{symbol}: {count} negative volume values")
            else:
                data[volume_col] = volume.abs()
                repairs.append("made_negative_volume_positive")

        # Check for excessive volume
        max_volume = self.config["price_limits"]["max_volume"]
        excessive_volume = volume > max_volume
        if excessive_volume.any():
            count = excessive_volume.sum()
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(
                    f"{symbol}: {count} excessive volume values > {max_volume}"
                )
            elif self.config["repair_mode"] == "drop":
                data = data.loc[~excessive_volume]
                repairs.append("dropped_excessive_volume_values")
            else:
                data[volume_col] = volume.clip(upper=max_volume)
                repairs.append("capped_excessive_volume_values")

        # Check for zero volume
        if not self.config["ohlc_validation"]["allow_zero_volume"]:
            zero_volume = volume == 0
            if zero_volume.any():
                count = zero_volume.sum()
                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(f"{symbol}: {count} zero volume values")
                else:
                    # Replace with median volume
                    median_volume = volume[volume > 0].median()
                    if pd.isna(median_volume):
                        median_volume = 1000000  # Default
                    data.loc[zero_volume, volume_col] = median_volume
                    repairs.append("replaced_zero_volume_with_median")

        return data, repairs, flags

    def _detect_and_repair_outliers_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], int]:
        """Detect and repair outliers with strict profile rules."""
        repairs = []
        outlier_count = 0

        outlier_config = self.config["outlier_detection"]

        if len(data) < outlier_config["min_obs_for_outlier"]:
            return data, repairs, outlier_count

        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Use log prices for outlier detection
            log_prices = np.log(series)

            # Calculate robust statistics
            median = np.median(log_prices)
            mad = np.median(np.abs(log_prices - median))

            if mad == 0:
                continue

            # Calculate z-scores using MAD
            z_scores = 0.6745 * (log_prices - median) / mad

            # Detect outliers
            outliers = np.abs(z_scores) > outlier_config["z_score_threshold"]

            if outliers.any():
                outlier_count += outliers.sum()

                if self.profile_config.get("allow_winsorize", True):
                    # Winsorize outliers
                    q_low = self.config["winsorize_quantile"]
                    q_high = 1 - q_low
                    lower_bound = series.quantile(q_low)
                    upper_bound = series.quantile(q_high)
                    data[col] = series.clip(lower=lower_bound, upper=upper_bound)
                    repairs.append(f"winsorized_outliers_in_{col}")
                else:
                    raise DataSanityError(f"{symbol}: {outliers.sum()} outliers detected in {col}")

        return data, repairs, outlier_count

    def _detect_and_repair_outliers(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], int]:
        """Detect and repair outliers using robust statistics."""
        repairs = []
        outlier_count = 0
        outlier_config = self.config["outlier_detection"]

        if len(data) < outlier_config["min_obs_for_outlier"]:
            return data, repairs, outlier_count

        price_cols = self._get_price_columns(data)

        for col in price_cols:
            if col not in data.columns:
                continue

            series = data[col]

            # Use log prices for outlier detection
            log_prices = np.log(series)

            # Calculate robust statistics
            median = np.median(log_prices)
            mad = np.median(np.abs(log_prices - median))

            if mad == 0:
                continue

            # Calculate z-scores using MAD
            z_scores = 0.6745 * (log_prices - median) / mad

            # Detect outliers
            outliers = np.abs(z_scores) > outlier_config["z_score_threshold"]

            if outliers.any():
                outlier_count += outliers.sum()

                if self.config["repair_mode"] == "fail":
                    self._handle_validation_failure(
                        f"{symbol}: {outliers.sum()} outliers detected in {col}"
                    )
                elif self.config["repair_mode"] == "drop":
                    data = data.loc[~outliers]
                    self._log_repair(f"{symbol}: Dropped {outlier_count} outliers from {col}")
                elif self.config["repair_mode"] == "winsorize":
                    # Winsorize outliers
                    q_low = self.config["winsorize_quantile"]
                    q_high = 1 - q_low
                    lower_bound = series.quantile(q_low)
                    upper_bound = series.quantile(q_high)
                    data[col] = series.clip(lower=lower_bound, upper=upper_bound)
                    repairs.append(f"winsorized_outliers_in_{col}")

        return data, repairs, outlier_count

    def _calculate_returns_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Calculate returns with strict profile rules."""
        repairs = []
        flags = []

        returns_config = self.config["returns"]

        if "Close" not in data.columns:
            return data, repairs, flags

        close_prices = data["Close"]

        if len(close_prices) < returns_config["min_periods"]:
            return data, repairs, flags

        # Calculate log returns
        returns = np.log(close_prices / close_prices.shift(1)) if returns_config["method"] == "log_close_to_close" else close_prices.pct_change()

        # Handle missing values
        returns = returns.ffill().bfill() if returns_config["fill_method"] == "forward" else returns.fillna(0)

        # Check for extreme returns
        max_return = self.config["price_limits"]["max_daily_return"]
        extreme_returns = np.abs(returns) > max_return

        if extreme_returns.any():
            if self.profile_config.get("allow_winsorize", True):
                # Winsorize extreme returns
                returns = returns.clip(lower=-max_return, upper=max_return)
                repairs.append("winsorized_extreme_returns")
            else:
                raise DataSanityError(
                    f"{symbol}: {extreme_returns.sum()} extreme returns > {max_return}"
                )

        # Add returns to data
        data["Returns"] = returns

        return data, repairs, flags

    def _calculate_returns(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Calculate clean returns from close prices."""
        repairs = []
        flags = []
        returns_config = self.config["returns"]

        if "Close" not in data.columns:
            return data, repairs, flags

        close_prices = data["Close"]

        if len(close_prices) < returns_config["min_periods"]:
            return data, repairs, flags

        # Calculate log returns
        returns = np.log(close_prices / close_prices.shift(1)) if returns_config["method"] == "log_close_to_close" else close_prices.pct_change()

        # Handle missing values
        returns = returns.ffill().bfill() if returns_config["fill_method"] == "forward" else returns.fillna(0)

        # Check for extreme returns
        max_return = self.config["price_limits"]["max_daily_return"]
        extreme_returns = np.abs(returns) > max_return

        if extreme_returns.any():
            count = extreme_returns.sum()
            if self.config["repair_mode"] == "fail":
                self._handle_validation_failure(f"{symbol}: {count} extreme returns > {max_return}")
            elif self.config["repair_mode"] == "drop":
                data = data.loc[~extreme_returns]
                repairs.append("dropped_extreme_returns")
            else:
                # Winsorize extreme returns
                returns = returns.clip(lower=-max_return, upper=max_return)
                repairs.append("winsorized_extreme_returns")

        # Add returns to data
        data["Returns"] = returns

        return data, repairs, flags

    def _final_validation_checks_strict(
        self, data: pd.DataFrame, symbol: str
    ) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Perform final validation checks with strict profile rules."""
        repairs = []
        flags = []

        # 1. Ensure minimum data requirements
        if len(data) < 1:
            raise DataSanityError(f"{symbol}: Insufficient data (need >= 1 row, got {len(data)})")

        # 2. Verify column schema
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise DataSanityError(f"{symbol}: Missing required columns: {missing_cols}")

        # 3. Verify data types
        expected_dtypes = {
            "Open": np.floating,
            "High": np.floating,
            "Low": np.floating,
            "Close": np.floating,
            "Volume": (np.floating, np.integer),  # Allow both float and int for volume
        }

        for col, expected_type in expected_dtypes.items():
            if col in data.columns:
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not any(np.issubdtype(data[col].dtype, t) for t in expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )
                else:
                    # Single allowed type
                    if not np.issubdtype(data[col].dtype, expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )

        # 4. Final finite check
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in data.columns:
                non_finite = ~np.isfinite(data[col])
                if non_finite.any():
                    raise DataSanityError(
                        f"{symbol}: {non_finite.sum()} non-finite values in {col} after validation"
                    )

        # 5. Verify index integrity
        if not data.index.is_monotonic_increasing:
            raise DataSanityError(f"{symbol}: Index is not monotonic after validation")

        if data.index.has_duplicates:
            raise DataSanityError(f"{symbol}: Index has duplicates after validation")

        # 6. Check for corporate actions consistency (if Adj Close present)
        if "Adj Close" in data.columns and "Close" in data.columns:
            adj_returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
            close_returns = np.log(data["Close"] / data["Close"].shift(1))

            # Adj returns should be similar to close returns (within tolerance)
            diff = np.abs(adj_returns - close_returns)
            large_diff = diff > 0.1  # 10% tolerance

            if large_diff.any():
                repairs.append(f"large_adj_close_differences:{large_diff.sum()}")

        return data, repairs, flags

    def _detect_lookahead_contamination(self, data: pd.DataFrame) -> bool:
        """Detect potential lookahead contamination."""
        # Check user-provided columns (including user-provided Returns if any)
        columns_to_scan = self.original_columns.copy()
        
        # If Returns was in original data (not created by us), check it for lookahead
        # If we created Returns during validation, ignore it
        if "Returns" in columns_to_scan:
            # Check if this Returns column has obvious lookahead contamination
            returns_series = data["Returns"]
            
            # Look for actual lookahead: values that match future values exactly
            # This would catch cases like: data.loc[i, "Returns"] = data.loc[i+1, "Returns"]
            if len(returns_series) > 1:
                for offset in [1, 2]:  # Check 1-2 period offsets
                    future_series = returns_series.shift(-offset)
                    exact_matches = (returns_series == future_series) & pd.notna(returns_series) & pd.notna(future_series)
                    
                    # If we find exact matches with future values, that's suspicious
                    if exact_matches.any():
                        match_count = exact_matches.sum()
                        logger.warning(f"Potential lookahead contamination in Returns column: {match_count} exact matches with future values at offset {offset}")
                        return True
        
        # Remove pipeline columns we trust (but not user-provided Returns which we checked above)
        user_feature_cols = columns_to_scan - {"Label", "Target", "y"}
        
        # Skip lookahead detection if no user columns to scan
        if not user_feature_cols:
            return False
            
        # Use centralized lookahead detection on user columns
        offenders = detect_lookahead(data, feature_cols=user_feature_cols)
        if offenders:
            for col in offenders:
                logger.warning(f"Lookahead contamination detected in column {col}: future-referencing patterns")
            return True

        return False

    def _final_validation_checks(self, data: pd.DataFrame, symbol: str) -> tuple[pd.DataFrame, list[str], list[str]]:
        """Perform final validation checks."""
        repairs = []
        flags = []

        # 1. Ensure minimum data requirements
        if len(data) < 1:
            raise DataSanityError(f"{symbol}: Insufficient data (need >= 1 row, got {len(data)})")

        # 2. Check for lookahead contamination (basic check)
        if "Returns" in data.columns:
            # Returns should not have future information
            future_returns = data["Returns"].shift(-1).notna()
            if future_returns.any():
                flags.append("potential_lookahead_contamination")

        # 3. Verify column schema
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            if self.config["repair_mode"] == "fail":
                raise DataSanityError(f"{symbol}: Missing required columns: {missing_cols}")
            else:
                # Try to repair by creating synthetic columns based on available data
                if "Close" not in data.columns and "Open" in data.columns:
                    # Use Open as Close if Close is missing
                    data["Close"] = data["Open"]
                    repairs.append("synthesized_close_from_open")
                if "High" not in data.columns and "Close" in data.columns:
                    # Use Close as High if High is missing
                    data["High"] = data["Close"] * 1.01  # Small markup
                    repairs.append("synthesized_high_from_close")
                if "Low" not in data.columns and "Close" in data.columns:
                    # Use Close as Low if Low is missing
                    data["Low"] = data["Close"] * 0.99  # Small markdown
                    repairs.append("synthesized_low_from_close")
                if "Volume" not in data.columns:
                    # Use default volume if missing
                    data["Volume"] = 1000000
                    repairs.append("synthesized_default_volume")
                
                # Re-check after repairs
                still_missing = [col for col in required_cols if col not in data.columns]
                if still_missing:
                    raise DataSanityError(f"{symbol}: Cannot repair missing columns: {still_missing}")

        # 4. Verify data types
        expected_dtypes = {
            "Open": np.floating,
            "High": np.floating,
            "Low": np.floating,
            "Close": np.floating,
            "Volume": (np.floating, np.integer),  # Allow both float and int for volume
        }

        for col, expected_type in expected_dtypes.items():
            if col in data.columns:
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not any(np.issubdtype(data[col].dtype, t) for t in expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )
                else:
                    # Single allowed type
                    if not np.issubdtype(data[col].dtype, expected_type):
                        raise DataSanityError(
                            f"{symbol}: Column {col} has wrong dtype: {data[col].dtype}"
                        )

        # 5. Final finite check
        price_cols = ["Open", "High", "Low", "Close"]
        for col in price_cols:
            if col in data.columns:
                non_finite = ~np.isfinite(data[col])
                if non_finite.any():
                    raise DataSanityError(
                        f"{symbol}: {non_finite.sum()} non-finite values in {col} after validation"
                    )

        # 6. Verify index integrity
        if not data.index.is_monotonic_increasing:
            if self.config["repair_mode"] == "fail":
                raise DataSanityError(f"{symbol}: Index is not monotonic after validation")
            else:
                # Try to repair by normalizing timezones and sorting
                if isinstance(data.index, pd.DatetimeIndex):
                    # Normalize mixed timezones
                    if hasattr(data.index, 'tz') and data.index.tz is not None:
                        # If timezone-aware, convert to UTC for consistency
                        data.index = data.index.tz_convert('UTC')
                    else:
                        # If mixed naive/aware, convert all to naive
                        if any(hasattr(ts, 'tz') and ts.tz is not None for ts in data.index):
                            data.index = data.index.tz_localize(None, level=0)
                    
                    # Sort by index to ensure monotonicity
                    data = data.sort_index()
                    repairs.append("normalized_timezones_and_sorted_index")
                    
                    # Re-check
                    if not data.index.is_monotonic_increasing:
                        raise DataSanityError(f"{symbol}: Cannot repair non-monotonic index")

        if data.index.has_duplicates:
            raise DataSanityError(f"{symbol}: Index has duplicates after validation")

        # 7. Check for corporate actions consistency (if Adj Close present)
        if "Adj Close" in data.columns and "Close" in data.columns:
            adj_returns = np.log(data["Adj Close"] / data["Adj Close"].shift(1))
            close_returns = np.log(data["Close"] / data["Close"].shift(1))

            # Adj returns should be similar to close returns (within tolerance)
            diff = np.abs(adj_returns - close_returns)
            large_diff = diff > 0.1  # 10% tolerance

            if large_diff.any():
                repairs.append(f"large_adj_close_differences:{large_diff.sum()}")

        logger.info(f"{symbol}: Final validation checks passed")
        return data, repairs, flags

    def _get_price_columns(self, data: pd.DataFrame) -> list[str]:
        """Get standardized price column names."""
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns - keep just the first level (OHLCV names)
            data.columns = [col[0] for col in data.columns]

        # Map common column name variations
        column_mapping = {
            "open": "Open",
            "OPEN": "Open",
            "Open": "Open",
            "high": "High",
            "HIGH": "High",
            "High": "High",
            "low": "Low",
            "LOW": "Low",
            "Low": "Low",
            "close": "Close",
            "CLOSE": "Close",
            "Close": "Close",
            "volume": "Volume",
            "VOLUME": "Volume",
            "Volume": "Volume",
        }

        # Standardize column names
        data.columns = [column_mapping.get(str(col).lower(), col) for col in data.columns]

        # Return price columns
        price_cols = []
        for col in ["Open", "High", "Low", "Close"]:
            if col in data.columns:
                price_cols.append(col)

        return price_cols

    def _get_volume_column(self, data: pd.DataFrame) -> str:
        """Get standardized volume column name."""
        # Handle MultiIndex columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Flatten multi-level columns - keep just the first level (OHLCV names)
            data.columns = [col[0] for col in data.columns]

        # Map common column name variations
        column_mapping = {"volume": "Volume", "VOLUME": "Volume", "Volume": "Volume"}

        # Standardize column names
        data.columns = [column_mapping.get(str(col).lower(), col) for col in data.columns]

        # Return volume column
        if "Volume" in data.columns:
            return "Volume"
        else:
            return "Volume"  # Default fallback

    def _repair_ohlc_inconsistencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Repair OHLC inconsistencies."""
        # Ensure High >= max(Open, Close)
        if all(col in data.columns for col in ["High", "Open", "Close"]):
            data["High"] = data[["High", "Open", "Close"]].max(axis=1)

        # Ensure Low <= min(Open, Close)
        if all(col in data.columns for col in ["Low", "Open", "Close"]):
            data["Low"] = data[["Low", "Open", "Close"]].min(axis=1)

        # Additional check: ensure High >= Low
        if all(col in data.columns for col in ["High", "Low"]):
            # If High < Low, set High = Low + small amount
            high_low_violations = data["High"] < data["Low"]
            if high_low_violations.any():
                data.loc[high_low_violations, "High"] = data.loc[high_low_violations, "Low"] + 0.01

        return data

    def _handle_validation_failure(self, message: str):
        """Handle validation failure based on repair mode."""
        self.validation_failures.append(message)

        if self.config["repair_mode"] == "fail":
            raise DataSanityError(message)
        else:
            logger.warning(f"Validation failure: {message}")

    def _log_repair(self, message: str):
        """Log a repair action."""
        self.repair_count += 1
        if self.config["logging"]["log_repairs"]:
            logger.info(f"Data repair: {message}")

    def _log_validation_summary(self, symbol: str, original_shape: tuple, final_shape: tuple):
        """Log validation summary."""
        summary_msg = (
            f"{symbol} validation complete: "
            f"{original_shape[0]} → {final_shape[0]} rows, "
            f"{self.repair_count} repairs, "
            f"{self.outlier_count} outliers"
        )

        if self.validation_failures:
            summary_msg += f", {len(self.validation_failures)} failures"

        log_level = getattr(logging, self.config["logging"]["summary_level"])
        logger.log(log_level, summary_msg)

    def coerce_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Convert string data types to numeric for OHLCV columns.

        Args:
            data: DataFrame with potentially string data

        Returns:
            DataFrame with numeric data types
        """
        result = data.copy()

        # Define columns to convert
        price_cols = ["Open", "High", "Low", "Close"]
        volume_cols = ["Volume"]

        # Convert price columns
        for col in price_cols:
            if col in result.columns and result[col].dtype == "object":
                try:
                    # Remove currency symbols and whitespace
                    cleaned_series = (
                        result[col].astype(str).str.replace(r"[\$€£¥₹]", "", regex=True)
                    )
                    cleaned_series = cleaned_series.str.strip()

                    # Handle European decimal notation (comma as decimal separator)
                    # Convert 100,02 to 100.02
                    cleaned_series = cleaned_series.str.replace(
                        r"(\d+),(\d+)", r"\1.\2", regex=True
                    )

                    # Remove remaining commas (thousands separators)
                    cleaned_series = cleaned_series.str.replace(r",", "", regex=False)

                    # Handle scientific notation and currency suffixes
                    cleaned_series = cleaned_series.str.replace(
                        r"\s*[A-Z]{3}$", "", regex=True
                    )  # Remove USD, EUR, etc.

                    # Convert to numeric
                    result[col] = pd.to_numeric(cleaned_series, errors="coerce")

                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")

        # Convert volume columns
        for col in volume_cols:
            if col in result.columns and result[col].dtype == "object":
                try:
                    # Remove thousands separators and whitespace
                    cleaned_series = result[col].astype(str).str.replace(r",", "", regex=False)
                    cleaned_series = cleaned_series.str.strip()

                    # Convert to numeric
                    result[col] = pd.to_numeric(cleaned_series, errors="coerce")

                except Exception as e:
                    logger.warning(f"Failed to convert {col} to numeric: {e}")

        return result

    def compute_returns(self, close_prices: pd.Series) -> pd.Series:
        """
        Compute returns from close prices.

        Args:
            close_prices: Series of close prices

        Returns:
            Series of returns (first value is 0)
        """
        if len(close_prices) == 0:
            return pd.Series(dtype=float)

        # Calculate percentage change
        returns = close_prices.pct_change()

        # Fill NaN with 0 (first observation)
        returns = returns.fillna(0.0)

        return returns


class DataSanityWrapper:
    """
    Wrapper that applies data sanity validation to all data sources.
    """

    def __init__(self, config_path: str = "config/data_sanity.yaml", profile: str = "default"):
        """Initialize DataSanityWrapper."""
        self.validator = DataSanityValidator(config_path, profile)
        self.profile = profile
        logger.info(f"Initialized DataSanityWrapper with profile: {profile}")

    def load_and_validate(self, filepath: str, symbol: str = None) -> pd.DataFrame:
        """
        Load data from file and apply validation.

        Args:
            filepath: Path to data file
            symbol: Symbol name (inferred from filename if None)

        Returns:
            Validated DataFrame
        """
        if symbol is None:
            symbol = Path(filepath).stem

        # Load data based on file extension
        if filepath.endswith(".pkl"):
            data = pd.read_pickle(filepath)
        elif filepath.endswith(".csv"):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif filepath.endswith(".parquet"):
            data = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        # Apply validation
        clean_data, result = self.validator.validate_and_repair(data, symbol)
        return clean_data

    def validate_dataframe(
        self, data: pd.DataFrame, symbol: str = "UNKNOWN", **kwargs
    ) -> pd.DataFrame:
        """
        Validate an existing DataFrame with back-compat support.

        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging (backward compatibility)
            **kwargs: Additional arguments (profile, etc.)

        Returns:
            Validated DataFrame
        """
        profile = kwargs.get("profile", "default")

        # Create validator with specified profile
        validator = DataSanityValidator(profile=profile)
        clean_data, result = validator.validate_and_repair(data, symbol)

        # Accumulate stats from this validation
        self.validator.repair_count += len(result.repairs)
        self.validator.outlier_count += result.outliers
        if result.repairs or result.flags:
            self.validator.validation_failures.append(f"{symbol}: {result.repairs}")

        # Handle mode-based behavior
        mode = validator.profile_config.get("mode", "warn")
        if mode == "error" and (result.repairs or result.flags):
            raise DataSanityError(
                f"{symbol}: Validation failed with mode='error': repairs={result.repairs}, flags={result.flags}"
            )

        return clean_data

    def validate(self, data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
        """
        Alias for validate_dataframe for compatibility.

        Args:
            data: DataFrame to validate
            symbol: Symbol name for logging

        Returns:
            Validated DataFrame
        """
        return self.validate_dataframe(data, symbol=symbol)

    def get_validation_stats(self) -> dict:
        """Get validation statistics."""
        return {
            "repair_count": self.validator.repair_count,
            "outlier_count": self.validator.outlier_count,
            "validation_failures": self.validator.validation_failures,
        }


# Global instance for easy access
_data_sanity_wrapper = None


def get_data_sanity_wrapper(
    config_path: str = "config/data_sanity.yaml", profile: str = "default"
) -> DataSanityWrapper:
    """Get global DataSanityWrapper instance."""
    global _data_sanity_wrapper
    if _data_sanity_wrapper is None:
        _data_sanity_wrapper = DataSanityWrapper(config_path, profile)
    return _data_sanity_wrapper


def validate_market_data(data: pd.DataFrame, symbol: str = "UNKNOWN") -> pd.DataFrame:
    """
    Convenience function to validate market data.

    Args:
        data: DataFrame with market data
        symbol: Symbol name for logging

    Returns:
        Validated DataFrame
    """
    wrapper = get_data_sanity_wrapper()
    return wrapper.validate_dataframe(data, symbol)
