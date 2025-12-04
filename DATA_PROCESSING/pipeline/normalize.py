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

# utils/session_normalize.py
import polars as pl
from datetime import time

NY = "America/New_York"

# Expected bars per full RTH day (6.5h)
GRID = {
    "5m":  {"minutes": {0,5,10,15,20,25,30,35,40,45,50,55}, "bars": 78, "start": time(9,30), "end": time(16,0)},
    "15m": {"minutes": {0,15,30,45},                         "bars": 24, "start": time(9,30), "end": time(16,0)},
    "30m": {"minutes": {0,30},                               "bars": 13, "start": time(9,30), "end": time(16,0)},
    # 1h convention: top-of-hour bars 10:00..16:00 → 6 bars
    "1h":  {"minutes": {0},                                  "bars":  6, "start": time(10,0), "end": time(16,0)},
}

def normalize_interval(df: pl.DataFrame, interval: str) -> pl.DataFrame:
    """
    Enforce NYSE RTH and on-grid timestamps for 5m/15m/30m/1h.
    Assumes df has a UTC 'ts' column (tz-naive or tz=UTC).
    Returns 'ts' back in UTC.
    """
    if interval not in GRID:
        raise ValueError(f"Unsupported interval: {interval}")

    g = GRID[interval]
    # Ensure tz-aware, convert to NY
    df = df.with_columns(
        pl.col("ts").dt.replace_time_zone("UTC").dt.convert_time_zone(NY).alias("ts_ny")
    )

    # RTH time window
    df = df.filter(
        (pl.col("ts_ny").dt.time() >= g["start"]) &
        (pl.col("ts_ny").dt.time() <  g["end"])
    )

    # Minute grid
    df = df.filter(pl.col("ts_ny").dt.minute().is_in(list(g["minutes"])))

    # De-dup + sort
    df = df.unique(subset=["ts_ny"]).sort("ts_ny")

    # Back to UTC
    return df.with_columns(pl.col("ts_ny").dt.convert_time_zone("UTC").alias("ts")).drop("ts_ny")

def assert_bars_per_day(df: pl.DataFrame, interval: str, min_full_day_frac: float = 0.90):
    """
    Checks what fraction of trading days have the expected full-day bar count for the interval.
    Tolerates holidays/early-closes by using a fraction threshold.
    """
    g = GRID[interval]
    counts = (
        df.with_columns(
            pl.col("ts").dt.replace_time_zone("UTC").dt.convert_time_zone(NY).dt.truncate("1d").alias("day")
        )
        .group_by("day").len().rename({"len": "bars"})
    )
    if counts.is_empty():
        raise AssertionError("No data after normalization.")
    frac_full = (counts["bars"] == g["bars"]).mean()
    if frac_full < min_full_day_frac:
        raise AssertionError(
            f"{interval}: only {frac_full:.1%} of days have full {g['bars']} bars. "
            f"Likely a filtering/clock issue."
        )
