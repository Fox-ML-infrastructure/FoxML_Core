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

import polars as pl
from pathlib import Path
from typing import Iterable, Mapping, Dict, List, Literal
import glob
import logging

log = logging.getLogger(__name__)
DType = pl.datatypes.DataType
Policy = Literal["coerce", "fail", "separate"]

POLARS_DTYPES: Dict[str, DType] = {
    "utf8": pl.Utf8,
    "bool": pl.Boolean,
    "int64": pl.Int64,
    "float64": pl.Float64,
    "datetime[ns]": pl.Datetime("ns"),
}

def _map_yaml_dtypes(yaml_map: Mapping[str, str]) -> Dict[str, DType]:
    out = {}
    for k, v in yaml_map.items():
        out[k] = POLARS_DTYPES[v]
    return out

def canonicalize_base_dtypes(lf: pl.LazyFrame,
                             canonical: Mapping[str, DType],
                             volume_policy: Policy = "coerce",
                             eps: float = 1e-9) -> pl.LazyFrame:
    exprs: List[pl.Expr] = []

    # Volume handling first (it's the troublemaker)
    if "volume" in lf.collect_schema().names():
        vol = pl.col("volume")
        if volume_policy == "coerce":
            # Dual-column policy: keep original as volume_float; provide rounded Int64 canonical volume
            exprs.append(vol.cast(pl.Float64, strict=False).alias("volume_float"))
            exprs.append(
                (vol
                 .cast(pl.Float64, strict=False)
                 .fill_null(0.0)
                 .clip(lower_bound=0.0)
                 .round(0)
                 .cast(pl.Int64, strict=False)
                ).alias("volume")
            )
        elif volume_policy == "separate":
            # Keep original as float; provide int view as volume
            exprs.append(vol.cast(pl.Float64, strict=False).alias("volume_adj"))
            exprs.append(
                vol.cast(pl.Float64, strict=False)
                .round(0)
                .cast(pl.Int64, strict=False)
                .alias("volume")
            )
        else:  # "fail"
            frac = lf.select(
                ((vol.cast(pl.Float64, strict=False) - vol.cast(pl.Float64, strict=False).floor()).abs() > eps)
                .sum()
            ).collect().item()
            if frac and frac > 0:
                raise ValueError(f"Fractional volume detected in {frac} rows; set volume_policy=coerce|separate.")
            exprs.append(vol.cast(pl.Int64, strict=False).alias("volume"))

    # Other base columns: cast to canonical if present
    for col, dt in canonical.items():
        if col == "volume":  # handled above
            continue
        if col in lf.collect_schema().names():
            exprs.append(pl.col(col).cast(dt, strict=False).alias(col))

    return lf.with_columns(exprs) if exprs else lf

def safe_scan_parquet(
    paths_glob: str | Iterable[str],
    canonical_dtypes: Mapping[str, str],
    volume_policy: Policy = "coerce",
) -> pl.LazyFrame:
    """
    Deterministic, lazy per-file normalization:
      - Scan each file lazily
      - Canonicalize base dtypes (with explicit volume policy)
      - Concatenate lazily
      - Emit a single concise warning if lossy volume rounding occurred
    """
    # Resolve files
    if isinstance(paths_glob, str):
        paths = sorted(glob.glob(paths_glob))
    else:
        paths = sorted(paths_glob)
    if not paths:
        raise FileNotFoundError(f"No parquet sources at {paths_glob}")

    overrides = _map_yaml_dtypes(canonical_dtypes)
    # Store volume policy separately since it's not a dtype
    volume_policy_override = volume_policy

    lfs: List[pl.LazyFrame] = []
    lossy_volume_rows_total = 0
    total_rows = 0

    # Best-effort symbol extraction from the first path (expects .../symbol=SYMBOL/...)
    symbol_hint = None
    try:
        first = paths[0]
        parts = Path(first).parts
        for seg in parts:
            if seg.startswith("symbol="):
                symbol_hint = seg.split("=", 1)[1]
                break
        if symbol_hint is None:
            # Fallback: filename stem without extension if layout differs
            symbol_hint = Path(first).stem
    except Exception:
        symbol_hint = "<unknown>"

    for p in paths:
        # Per-file scan
        lfi = pl.scan_parquet(p)

        # Compute lossy metric for volume if target is Int64 and source can be float
        names = set(lfi.collect_schema().names())
        if volume_policy in ("coerce", "fail") and "volume" in names and overrides.get("volume") == pl.Int64:
            try:
                # Count total rows and non-integer volumes (pre-cast)
                stats = (
                    pl.scan_parquet(p)
                    .select(
                        total=pl.len(),
                        rounded=(
                            pl.col("volume").cast(pl.Float64, strict=False).is_not_null()
                            & ((pl.col("volume").cast(pl.Float64, strict=False) % 1) != 0)
                        ).sum(),
                    )
                ).collect(streaming=True)
                total_rows += int(stats["total"][0])
                lossy_volume_rows_total += int(stats["rounded"][0])
            except Exception:
                # Metric is best-effort; skip if unsupported
                pass

        # Canonicalize columns/dtypes lazily - preserve raw volume as int, use config volume policy
        canonical_dtypes_mapped = _map_yaml_dtypes(canonical_dtypes)
        lfi = canonicalize_base_dtypes(lfi, canonical_dtypes_mapped, volume_policy=volume_policy_override)
        lfs.append(lfi)

    # Concatenate lazily (dtypes aligned by canonicalization)
    lf = pl.concat(lfs, how="vertical")

    # Trigger planning cheaply to surface issues early
    _ = lf.select(pl.len()).fetch(1)

    if lossy_volume_rows_total > 0 and total_rows > 0:
        pct = lossy_volume_rows_total / total_rows
        # Gate warning at 2%; otherwise log at info
        level = log.warning if pct >= 0.02 else log.info
        msg = (
            "Rounded %s non-integer 'volume' rows for %s (%0.2f%% of %s bars)."
            % (f"{lossy_volume_rows_total:,}", symbol_hint, pct * 100.0, f"{total_rows:,}")
        )
        # If elevated rate, add split-factor fingerprint buckets (~1/2, ~1/3, ~1/4, other)
        if pct >= 0.02:
            try:
                bucket_lfs: List[pl.LazyFrame] = []
                for p in paths:
                    f = pl.col("volume").cast(pl.Float64, strict=False)
                    frac = (f % 1).abs()
                    is_frac = f.is_not_null() & (frac != 0)
                    bucket = (
                        pl.when((frac - 0.5).abs() < 0.01).then(pl.lit("~1/2"))
                        .when((frac - (1/3)).abs() < 0.01).then(pl.lit("~1/3"))
                        .when((frac - 0.25).abs() < 0.01).then(pl.lit("~1/4"))
                        .otherwise(pl.lit("other"))
                    )
                    bucket_lfs.append(
                        pl.scan_parquet(p)
                        .filter(is_frac)
                        .select(bucket.alias("bucket"))
                        .group_by("bucket").len()
                    )
                buckets_df = pl.concat(bucket_lfs, how="vertical").group_by("bucket").agg(pl.col("len").sum().alias("n")).collect(streaming=True)
                b = {r[0]: int(r[1]) for r in buckets_df.iter_rows()}
                msg += f" Buckets: ~1/2={b.get('~1/2',0)}, ~1/3={b.get('~1/3',0)}, ~1/4={b.get('~1/4',0)}, other={b.get('other',0)}."
            except Exception:
                # Fingerprinting is best-effort
                pass
        level(msg)

    return lf

def write_features_strict(df_or_lf, root: str, feature_set: str, symbol: str) -> str:
    """Write features with strict path validation and assertions."""
    df = df_or_lf.collect() if isinstance(df_or_lf, pl.LazyFrame) else df_or_lf
    # 'root' should be the storage root, e.g. "storage/features". Do not append another "features".
    out_dir = Path(root) / feature_set / "symbol" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}.parquet"
    df.write_parquet(out_path, compression="zstd", statistics=True)
    # double-assert
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"Write failed or empty: {out_path}")
    log.info("📦 Wrote %s rows=%d cols=%d → %s", symbol, df.height, df.width, out_path)
    return str(out_path)

def dedupe_symbols(symbols: list[str]) -> list[str]:
    """Remove duplicate symbols from the list."""
    seen, out = set(), []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
