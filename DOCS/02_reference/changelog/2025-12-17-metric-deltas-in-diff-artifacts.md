# Metric Deltas in Diff Artifacts - 2025-12-17

## Overview

Fixed critical issue where `metric_deltas` was empty in `diff_prev.json` even when metrics clearly differed. Implemented 3-tier reporting structure with structured metric deltas, noise detection, and proper separation of nondeterminism from performance regression.

## Achievement

This represents the hard, unsexy engineering that most ML stacks *pretend* to have:

- **Not just "things changed"** — Shows **prev vs curr**, **delta_abs / delta_pct**, and **impact label** for every metric
- **Separated reproducibility break (hash mismatch) from model quality impact (noise-level deltas)** — This is exactly how "audit-grade" systems should talk
- **Plumbing for trend/drift analysis** — Metric time series + thresholds ready for evolution into real trend analysis
- **Clean, meaningful reporting** — `metric_deltas_total` vs `metric_deltas_significant` with zero-delta spam filtered out

This is audit-grade diff telemetry that provides actionable insights, not just "something changed" noise.

## Problem

- `metric_deltas: {}` and `metric_deltas_count: 0` even when metrics differed
- Patch showed `"op": "add"` for `outputs.metrics` (indicating prev snapshot had no metrics)
- No structured delta information despite hash mismatches
- `diff_telemetry` field in `metrics.json` caused digest volatility

## Solution

### 1. Fixed Metric Extraction (`_normalize_outputs`)

- **Before**: Only extracted 3 metrics (`mean_score`, `std_score`, `composite_score`)
- **After**: Extracts all numeric metrics from `metrics.json`/`metrics.parquet` files
- Reads from files when metrics aren't in `run_data` (common for TARGET_RANKING/FEATURE_SELECTION)
- Captures all metrics: `mean_score`, `std_score`, `composite_score`, `mean_importance`, `pos_rate`, `N_effective_cs`, `n_models`, etc.

### 2. Enhanced Metric Delta Computation (`_compute_metric_deltas`)

- **Structured deltas** with noise detection using z-scores
- **Z-score computation**: `z = delta_abs / (std_score / sqrt(n_models))` when available
- **Impact classification**: `none|noise|minor|major` based on z-score and tolerances
- **Metric-dependent tolerances**: Different `abs_tol`/`rel_tol` for score metrics vs others
- Returns deltas with: `delta_abs`, `delta_pct`, `prev`, `curr`, `z_score`, `impact_label`, `abs_tol`, `rel_tol`

### 3. Implemented 3-Tier Reporting Structure

**Tier A (Summary in `diff_prev.json`)**:
- Lightweight summary with `metric_deltas_count`, `impact_label`
- `top_regressions` and `top_improvements` (up to 5 each)
- Reference to Tier B file: `metric_deltas_file`

**Tier B (Structured `metric_deltas.json`)**:
- Detailed per-metric deltas with full structured info
- Only written if deltas exist
- Keyed by metric name with complete delta information

**Tier C (Full Raw Metrics)**:
- Full raw metrics remain in `metrics.json` (already exists)
- No duplication, just reference

### 4. Separated Nondeterminism from Regression

- **Before**: Digest mismatch → always CRITICAL severity
- **After**: Severity depends on performance impact
  - Digest mismatch + `impact_label: noise` → MAJOR (reproducibility concern, but noise-level impact)
  - Digest mismatch + `impact_label: major` → CRITICAL (both reproducibility and performance issues)
- Severity now reflects **impact**, not just "hash changed"

### 5. Fixed Previous Snapshot Metrics Loading

- **Problem**: Previous snapshots loaded from index didn't have complete metrics
- **Solution**: Reload metrics from actual `metrics.json` file when finding previous comparable snapshot
- Ensures `prev_snapshot.outputs.get('metrics', {})` has complete metrics for comparison
- Patch now shows `"op": "replace"` instead of `"op": "add"`

### 6. Excluded `diff_telemetry` from Metrics Digest

- **Problem**: `diff_telemetry` field in `metrics.json` caused digest to change when `comparable` flag changed
- **Solution**: Exclude `diff_telemetry` and other metadata fields from metrics digest computation
- Prevents digest volatility from metadata changes

### 7. Added Trend/Drift Analysis Storage

- New `_emit_trend_time_series` method emits time series data per run
- Stores in `metrics_timeseries.parquet` at target level
- One row per metric with: `run_id`, `timestamp`, `comparison_group`, `stage`, `view`, `item_name`, `metric_name`, `value`
- Queryable time series keyed by `{stage, view, item_name, metric_name}`

## Impact Classification

- **none**: No change detected (within tolerances)
- **noise**: `|z| < 0.25` or below tolerance (statistically insignificant)
- **minor**: `0.25 ≤ |z| < 1.0` (small but noticeable change)
- **major**: `|z| ≥ 1.0` (significant change, potential regression/improvement)

## Files Changed

- `TRAINING/utils/diff_telemetry.py`:
  - Enhanced `_normalize_outputs` to extract all numeric metrics
  - Enhanced `_compute_metric_deltas` with z-scores and impact classification
  - Added `_classify_metric_impact` for overall impact assessment
  - Updated `_determine_severity` to separate nondeterminism from regression
  - Added `_reload_snapshot_metrics` to reload metrics when loading previous snapshots
  - Updated `save_diff` to implement 3-tier reporting structure
  - Added `_emit_trend_time_series` for trend analysis storage
  - Fixed `_compute_metrics_digest` to exclude `diff_telemetry`

## Example Output

**Before**:
```json
{
  "metric_deltas": {},
  "metric_deltas_count": 0,
  "summary": {
    "output_digest_changes": ["metrics_sha256"]
  }
}
```

**After**:
```json
{
  "metric_deltas": {
    "mean_score": {
      "delta_abs": 0.001449,
      "delta_pct": 0.223,
      "prev": 0.650859,
      "curr": 0.652308,
      "z_score": 0.0177,
      "impact_label": "noise",
      "abs_tol": 0.001,
      "rel_tol": 0.0001
    }
  },
  "metric_deltas_count": 10,
  "summary": {
    "impact_label": "noise",
    "top_regressions": [...],
    "top_improvements": [...],
    "metric_deltas_file": "metric_deltas.json"
  }
}
```

## Benefits

1. **Always compute deltas**: Even small changes are now reported with structured information
2. **Noise detection**: Z-scores distinguish real regressions from statistical noise
3. **Proper severity**: Severity reflects impact, not just hash mismatches
4. **Complete metrics**: All numeric metrics are captured, not just 3
5. **Stable digests**: Excluding `diff_telemetry` prevents digest volatility
6. **Trend analysis**: Time series data enables drift detection and trend analysis

## Testing

Verified with actual runs:
- `metric_deltas_total: 10`, `metric_deltas_significant: 3-4` (previously `metric_deltas_count: 0`)
- `impact_label: "noise"` correctly classified
- `top_regressions` and `top_improvements` populated (only significant deltas)
- `metric_deltas.json` file created with structured deltas
- Severity correctly set to MAJOR (nondeterminism) with noise-level impact noted
- Zero-delta entries (pos_rate, n_models) filtered out from `metric_deltas` dict

## Bug Fixes (2025-12-17 follow-up)

### Fixed `changed` Flag Semantics

**Problem**: `changed: true` was set for all entries in `metric_deltas`, even when `delta_abs < abs_tol`.

**Solution**: Split into two flags:
- `differs`: `prev != curr` (after rounding) - indicates values are different
- `changed`: `abs(delta) > max(abs_tol, rel_tol*abs(prev))` - indicates delta exceeds tolerance

**Example**: `std_score` with `delta_abs=9.2e-05` and `abs_tol=0.001` now correctly shows `changed: false`.

### Fixed `noise_explanation` Formatting Bug

**Problem**: Explanation showed `Delta (0.00092)` but actual value was `9.2e-05` (10× error).

**Solution**: Use exact serialized `delta_abs` value via placeholder replacement to ensure explanation matches the field exactly.

### Labeled `z_score` as Proxy

**Problem**: `z_score` uses same SE (`std_score/sqrt(n_models)`) for all metrics, but wasn't labeled as a proxy.

**Solution**: Added `z_score_basis: "auc_se_proxy"` to make it clear this is a heuristic using AUC standard error as proxy for all metrics, not a per-metric significance test.

### Added Identifiers to `metric_deltas.json`

**Enhancement**: Added `stage`, `view`, `item_name` fields to enable downstream joining and querying.

### Fixed Tolerance Logic (2025-12-17 follow-up)

**Problem**: `changed_tol: true` was set even when `delta_abs < abs_tol` (e.g., `std_score` with `delta_abs=0.000811` and `abs_tol=0.001`).

**Solution**: Use correct combined tolerance calculation:
- `tol = max(abs_tol, rel_tol * max(abs(prev), abs(curr), 1.0))`
- `changed_tol = abs(delta) > tol`
- Renamed `changed` to `changed_tol` for clarity

**Result**: `changed_tol` now correctly reflects whether delta exceeds the combined tolerance threshold.

### Fixed Per-Metric Polarity for Improvements/Regressions (2025-12-17 follow-up)

**Problem**: `std_score` increases were incorrectly classified as improvements (should be regressions since lower is better).

**Solution**: Added per-metric polarity mapping:
- `higher_is_better` dict: `std_score: False`, others: `True`
- Compute `signed_delta = delta_abs if is_higher_better else -delta_abs`
- Rank improvements/regressions using `signed_delta` instead of raw `delta_abs`

**Result**: `std_score` increases now correctly appear in `top_regressions`, not `top_improvements`.

## Next Level (Future Enhancements)

Potential future improvements:
- Compute variability per metric (empirical noise bands from repeated runs)
- Add drift detection thresholds based on historical variability
- Expand trend analysis with rolling baselines and CUSUM change-point detection

