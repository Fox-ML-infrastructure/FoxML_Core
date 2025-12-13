"""
Duration Parsing and Canonicalization

Provides robust parsing of time period strings (minutes, hours, days, weeks, compound durations)
and support for bar-based lookbacks. All durations are converted to a canonical Duration type
before comparison or computation.

**Core Invariant**: Everything becomes a Duration before comparison/computation.
This eliminates unit ambiguity and string comparison issues.

**Architecture**:
1. Parse input → Duration (canonical representation)
2. Enforce rules using Duration arithmetic
3. Format Duration → string only at the edge (logging/UI)

Usage:
    from TRAINING.utils.duration_parser import (
        parse_duration, 
        parse_duration_bars,  # Preferred for bar-based lookbacks
        enforce_purge_audit_rule, 
        format_duration
    )
    
    # Parse various formats
    d1 = parse_duration("85.0m")
    d2 = parse_duration("1h30m")
    d3 = parse_duration_bars(20, "5m")  # 20 bars at 5m = 100m (explicit, no ambiguity)
    
    # Enforce audit rule
    purge_out, min_purge, changed = enforce_purge_audit_rule(
        "85.0m",
        "100.0m",
        interval="5m",
        buffer_frac=0.01,
        strict_greater=True
    )
"""

import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Union

DurationLike = Union[str, int, float, timedelta, "Duration"]

# Fixed-time units only (no months/years; those are calendar-relative).
_UNIT_TO_SECONDS = {
    "ns": 1e-9,
    "us": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "m": 60.0,
    "min": 60.0,
    "mins": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
    "days": 86400.0,
    "w": 604800.0,
    "wk": 604800.0,
    "wks": 604800.0,
}

_TOKEN_RE = re.compile(r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?P<unit>[a-zA-Z]+)")

# Bar unit patterns (for explicit bar parsing)
_BAR_UNIT_PATTERNS = ["b", "bar", "bars"]


def parse_duration_bars(
    bars: Union[int, float, str],
    interval: DurationLike
) -> "Duration":
    """
    Parse bar-based duration explicitly (removes ambiguity).
    
    This is the preferred way to handle bar-based lookbacks, as it makes the
    intent explicit and avoids the ambiguity of "is this number seconds or bars?"
    
    Args:
        bars: Number of bars (int, float, or string like "20", "20b", "20bars")
        interval: Data interval (string like "5m", timedelta, Duration, or float seconds)
    
    Returns:
        Duration object
    
    Examples:
        >>> parse_duration_bars(20, "5m")  # 20 bars at 5m = 100m
        >>> parse_duration_bars("20b", "5m")  # Same, explicit format
        >>> parse_duration_bars(288, "5m")  # 288 bars at 5m = 1440m (1 day)
    
    Raises:
        ValueError: If parsing fails
    """
    interval_d = parse_duration(interval)
    
    # Handle string formats like "20b" or "20bars"
    if isinstance(bars, str):
        s = bars.strip().lower()
        # Remove bar unit suffix if present
        for suffix in _BAR_UNIT_PATTERNS:
            if s.endswith(suffix):
                s = s[:-len(suffix)].strip()
                break
        try:
            bars = float(s)
        except ValueError:
            raise ValueError(f"Could not parse bar count from '{bars}'")
    
    if not isinstance(bars, (int, float)):
        raise TypeError(f"Bars must be int, float, or string, got {type(bars)}")
    
    return interval_d * float(bars)


@dataclass(frozen=True)
class Duration:
    """
    Canonical duration type stored as integer microseconds for stable arithmetic.
    
    This ensures all duration comparisons and computations use a single canonical
    representation, avoiding unit ambiguity and rounding errors.
    """
    # Store as integer microseconds for stable arithmetic & logging friendliness
    microseconds: int

    @staticmethod
    def from_seconds(seconds: float) -> "Duration":
        """Create Duration from seconds (float)."""
        return Duration(int(round(seconds * 1_000_000)))

    @staticmethod
    def from_timedelta(td: timedelta) -> "Duration":
        """Create Duration from timedelta."""
        return Duration(int(td.total_seconds() * 1_000_000))

    def to_seconds(self) -> float:
        """Convert to seconds (float)."""
        return self.microseconds / 1_000_000

    def to_minutes(self) -> float:
        """Convert to minutes (float)."""
        return self.microseconds / (60 * 1_000_000)

    def to_timedelta(self) -> timedelta:
        """Convert to timedelta."""
        return timedelta(microseconds=self.microseconds)

    def __add__(self, other: "Duration") -> "Duration":
        """Add two durations."""
        return Duration(self.microseconds + other.microseconds)

    def __mul__(self, k: float) -> "Duration":
        """Multiply duration by scalar."""
        return Duration(int(round(self.microseconds * k)))

    def __rmul__(self, k: float) -> "Duration":
        """Right multiplication (scalar * duration)."""
        return self.__mul__(k)

    def __lt__(self, other: "Duration") -> bool:
        """Less than comparison."""
        return self.microseconds < other.microseconds

    def __le__(self, other: "Duration") -> bool:
        """Less than or equal comparison."""
        return self.microseconds <= other.microseconds

    def __gt__(self, other: "Duration") -> bool:
        """Greater than comparison."""
        return self.microseconds > other.microseconds

    def __ge__(self, other: "Duration") -> bool:
        """Greater than or equal comparison."""
        return self.microseconds >= other.microseconds

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Duration):
            return False
        return self.microseconds == other.microseconds

    def __repr__(self) -> str:
        """String representation."""
        return f"Duration({format_duration(self)})"


def parse_duration(
    x: DurationLike,
    *,
    interval: Optional[DurationLike] = None,
    assume_bars_if_number: bool = False
) -> Duration:
    """
    Parse duration from various input types.
    
    Supports:
      - Strings: "85.0m", "1h30m", "2d", "90s", "1h30m15s"
      - timedelta objects
      - Numbers (int/float):
          * if assume_bars_if_number=True, interpret as bars and multiply by interval (must be provided)
          * else interpret as seconds
    
    **Domain Constraints:**
    - Negative durations are disallowed (raises ValueError)
    - Empty strings are disallowed
    
    **Bar-based durations (ambiguity warning):**
    The `assume_bars_if_number=True` parameter creates ambiguity: "is this number seconds or bars?"
    Best practice: Use explicit string formats like "20b" or "20bars" (if supported by your config schema),
    or use separate fields (`lookback_bars` vs `lookback_duration`) in your config.
    
    For explicit bar parsing, use `parse_duration_bars()` instead.
    
    Args:
        x: Duration input (string, timedelta, int, float, or Duration)
        interval: Optional interval for bar-based interpretation
        assume_bars_if_number: If True and x is numeric, interpret as bars (requires interval)
                              **Warning**: This creates ambiguity. Prefer explicit bar formats.
    
    Returns:
        Duration object (non-negative)
    
    Raises:
        ValueError: If parsing fails, required parameters are missing, or duration is negative
        TypeError: If input type is unsupported
    """
    """
    Parse duration from various input types.
    
    Supports:
      - Strings: "85.0m", "1h30m", "2d", "90s", "1h30m15s"
      - timedelta objects
      - Numbers (int/float):
          * if assume_bars_if_number=True, interpret as bars and multiply by interval (must be provided)
          * else interpret as seconds
    
    **Bar-based durations (ambiguity warning):**
    The `assume_bars_if_number=True` parameter creates ambiguity: "is this number seconds or bars?"
    Best practice: Use explicit string formats like "20b" or "20bars" (if supported by your config schema),
    or use separate fields (`lookback_bars` vs `lookback_duration`) in your config.
    
    For explicit bar parsing, use `parse_duration_bars()` instead.
    
    Args:
        x: Duration input (string, timedelta, int, float, or Duration)
        interval: Optional interval for bar-based interpretation
        assume_bars_if_number: If True and x is numeric, interpret as bars (requires interval)
                              **Warning**: This creates ambiguity. Prefer explicit bar formats.
    
    Returns:
        Duration object
    
    Raises:
        ValueError: If parsing fails or required parameters are missing
        TypeError: If input type is unsupported
    """
    if isinstance(x, Duration):
        return x

    if isinstance(x, timedelta):
        return Duration.from_timedelta(x)

    if isinstance(x, (int, float)):
        if assume_bars_if_number:
            if interval is None:
                raise ValueError("interval is required when interpreting numbers as bars.")
            interval_d = parse_duration(interval)
            if interval_d.microseconds <= 0:
                raise ValueError(f"Interval must be positive when interpreting bars, got: {interval!r}")
            result = interval_d * float(x)
            if result.microseconds < 0:
                raise ValueError(f"Duration cannot be negative: {x} bars at {interval!r}")
            return result
        seconds = float(x)
        if seconds < 0:
            raise ValueError(f"Duration cannot be negative: {x}")
        return Duration.from_seconds(seconds)

    if not isinstance(x, str):
        raise TypeError(f"Unsupported duration type: {type(x)}")

    s = x.strip().lower()
    if not s:
        raise ValueError("Empty duration string.")

    # Support compound tokens like "1h30m" or "1.5h"
    total_seconds = 0.0
    pos = 0
    for m in _TOKEN_RE.finditer(s):
        if m.start() != pos and s[pos:m.start()].strip():
            raise ValueError(f"Unparseable duration chunk: {s[pos:m.start()]!r} in {x!r}")
        pos = m.end()

        val = float(m.group("value"))
        unit = m.group("unit").lower()
        if unit not in _UNIT_TO_SECONDS:
            raise ValueError(f"Unknown duration unit {unit!r} in {x!r}")
        total_seconds += val * _UNIT_TO_SECONDS[unit]

    if pos != len(s) and s[pos:].strip():
        raise ValueError(f"Unparseable duration tail: {s[pos:]!r} in {x!r}")

    if total_seconds < 0:
        raise ValueError(f"Duration cannot be negative: {x!r}")
    
    return Duration.from_seconds(total_seconds)


def ceil_to_interval(d: Duration, interval: Optional[DurationLike]) -> Duration:
    """
    Ceil duration up to a multiple of interval.
    
    If interval is None or invalid, return duration unchanged.
    
    Args:
        d: Duration to ceil
        interval: Interval to round up to (parsed if string/number)
    
    Returns:
        Ceiled duration
    """
    if interval is None:
        return d
    
    interval_d = parse_duration(interval)
    if interval_d.microseconds <= 0:
        return d
    
    q, r = divmod(d.microseconds, interval_d.microseconds)
    if r == 0:
        return d
    return Duration((q + 1) * interval_d.microseconds)


def format_duration(d: Duration) -> str:
    """
    Pretty format with adaptive units (s, m, h, d) for logs.
    
    Args:
        d: Duration to format
    
    Returns:
        Formatted string (e.g., "1.5h", "85.0m", "2d")
    """
    us = d.microseconds
    if us % (86400 * 1_000_000) == 0:
        return f"{us / (86400 * 1_000_000):.0f}d"
    if us % (3600 * 1_000_000) == 0:
        return f"{us / (3600 * 1_000_000):.0f}h"
    if us % (60 * 1_000_000) == 0:
        return f"{us / (60 * 1_000_000):.1f}m"
    return f"{us / 1_000_000:.1f}s"


def enforce_purge_audit_rule(
    purge: DurationLike,
    feature_lookback_max: DurationLike,
    *,
    interval: Optional[DurationLike] = None,
    buffer_frac: float = 0.01,
    strict_greater: bool = True,
) -> tuple[Duration, Duration, bool]:
    """
    Enforce purge audit rule: purge >= feature_lookback_max (with optional strict >).
    
    This is the generalized version that works with any time period strings,
    timedeltas, or bar-based lookbacks.
    
    **Rule Logic:**
    - If `interval` is provided: `min_purge = ceil_to_interval(lookback_max, interval) + interval`
      This guarantees `purge > lookback_max` at data resolution (the correct general solution).
    - If `interval` is None: `min_purge = lookback_max * (1 + buffer_frac)`
      This is a fallback when interval is unknown (less precise but safe).
    
    **Policy Note:**
    This rule treats feature lookback as requiring purge coverage. This is a **conservative policy**
    to prevent rolling window leakage, not a mathematical necessity. If features are strictly
    past-only (no lookahead), they don't cause train/test overlap leakage by themselves.
    Purge/embargo is primarily driven by label horizon / overlap of label windows.
    
    **Domain Constraints (Fail Fast):**
    - Negative durations are disallowed
    - Zero or negative intervals are disallowed
    - If `strict_greater=True`, `interval` must be provided (otherwise policy switches implicitly)
    
    Args:
        purge: Current purge value (string, timedelta, number, or Duration)
        feature_lookback_max: Maximum feature lookback (string, timedelta, number, or Duration)
        interval: Optional data interval for strict rounding (string, timedelta, number, or Duration)
                 If provided, this is the PRIMARY mechanism for strictness (interval-aware rounding)
                 **Required if strict_greater=True** (enforced)
        buffer_frac: Safety buffer fraction (default: 0.01 = 1%)
                    Only used as fallback when interval is None
        strict_greater: If True, ensure purge > lookback_max (not just >=)
                       When interval is provided, this is enforced via interval rounding
                       **Requires interval to be provided** (raises ValueError if not)
    
    Returns:
        (purge_out, min_purge, changed) tuple where:
        - purge_out: Adjusted purge duration (may be increased)
        - min_purge: Minimum required purge duration
        - changed: True if purge was increased
    
    Raises:
        ValueError: If duration parsing fails, negative durations, zero/negative interval,
                   or strict_greater=True without interval (fail closed - no silent fallbacks)
        TypeError: If input types are unsupported
    """
    # Parse all inputs to Duration (canonical representation)
    # This is the invariant: everything becomes Duration before comparison
    try:
        interval_d = parse_duration(interval) if interval is not None else None
        purge_d = parse_duration(purge)
        lb_d = parse_duration(feature_lookback_max)
    except (ValueError, TypeError) as e:
        # Fail closed: parsing errors are configuration errors, not runtime errors
        raise ValueError(
            f"Failed to parse durations for audit rule enforcement: {e}. "
            f"This indicates a configuration error that must be fixed. "
            f"purge={purge!r}, lookback={feature_lookback_max!r}, interval={interval!r}"
        ) from e
    
    # Domain constraints: fail fast on invalid inputs
    if purge_d.microseconds < 0:
        raise ValueError(f"Purge duration cannot be negative: {purge!r}")
    if lb_d.microseconds < 0:
        raise ValueError(f"Feature lookback duration cannot be negative: {feature_lookback_max!r}")
    if interval_d is not None:
        if interval_d.microseconds <= 0:
            raise ValueError(f"Interval must be positive, got: {interval!r}")
    
    # Enforce policy constraint: strict_greater requires interval
    if strict_greater and interval_d is None:
        raise ValueError(
            f"strict_greater=True requires interval to be provided. "
            f"This prevents implicit policy switching. "
            f"Either provide interval or set strict_greater=False."
        )

    # PRIMARY: Interval-aware strictness (the correct general solution)
    if strict_greater and interval_d is not None:
        # Rule: min_purge = ceil_to_interval(lookback_max, interval) + interval
        # This guarantees purge > lookback_max at data resolution
        min_purge = ceil_to_interval(lb_d, interval_d) + interval_d
    else:
        # FALLBACK: Buffer-based when interval is unknown
        min_purge = lb_d * (1.0 + buffer_frac)
        # If strict_greater but no interval, we can't guarantee strict >, so use buffer

    # Round min_purge to interval boundary if interval is known
    if interval_d is not None:
        min_purge = ceil_to_interval(min_purge, interval_d)

    changed = False
    if purge_d < min_purge:
        purge_d = min_purge
        changed = True

    return purge_d, min_purge, changed
