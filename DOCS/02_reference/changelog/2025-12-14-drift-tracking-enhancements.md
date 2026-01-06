# Enhanced Drift Tracking with Fingerprints and Sanity Checks (2025-12-14)

## Summary

Enhanced the telemetry drift tracking system with fingerprints, drift tiers, critical metrics tracking, and sanity checks. This makes drift tracking bulletproof: can now answer "What changed between baseline and current, and was it data, config, code, or stochasticity?"

---

## New Features

### 1. Fingerprint Tracking

**Problem:** Need to prove baseline is actually different from current run, not just comparing to itself or wrong baseline.

**Solution:** Track fingerprints for both baseline and current runs:
- `git_commit` - Code version (baseline + current)
- `config_hash` - Configuration hash (baseline + current)
- `data_fingerprint` - Data fingerprint computed from date range, N_effective, n_symbols, universe_id (baseline + current)
- `timestamp` - Run timestamp (baseline + current)

**Implementation:**
- `_compute_data_fingerprint()`: Computes SHA256 hash from data metadata
- `_get_git_commit()`: Gets current git commit hash
- Loads baseline metadata to extract fingerprints
- Stores fingerprints in `telemetry_drift.json` under `fingerprints.baseline` and `fingerprints.current`

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`_write_telemetry_drift()`, `_compute_data_fingerprint()`, `_get_git_commit()`)

**Result:** Can definitively prove what changed: code, config, data, or all identical (stochasticity).

---

### 2. Fixed rel_delta Handling for Zeros

**Problem:** `rel_delta = null` when baseline is zero is ambiguous - is it undefined or both zero?

**Solution:** Explicit `rel_delta_status` field:
- `"undefined_zero_baseline"` - Baseline is zero, current is not (rel_delta undefined)
- `"both_zero"` - Both baseline and current are zero (rel_delta = 0.0)
- `"defined"` - Normal calculation applies (rel_delta = delta / baseline)
- `baseline_zero: true/false` flag for clarity

**Implementation:**
- Explicit zero handling in drift calculation
- `rel_delta_status` field added to all drift metrics
- Parquet files include `rel_delta_status` column

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`_write_telemetry_drift()`)

**Result:** No more ambiguous nulls - humans can triage zero cases clearly.

---

### 3. Drift Tiers (OK/WARN/ALERT)

**Problem:** Binary STABLE/DRIFTING/DIVERGED doesn't provide severity levels for triage.

**Solution:** Three-tier system with configurable thresholds:
- **OK**: Within acceptable range (no action needed)
- **WARN**: Exceeds warning threshold (investigate)
- **ALERT**: Exceeds alert threshold (urgent attention)

**Thresholds:**
- **Normal metrics**: 5% = WARN, 20% = ALERT
- **Critical metrics**: 1% = WARN, 3% = ALERT (stricter)

**Implementation:**
- `_classify_drift_tier()`: Classifies drift into OK/WARN/ALERT
- Stricter thresholds for critical metrics (label_window, horizon, leakage flags, etc.)
- Backward compatible: legacy `status` field still included (STABLE/DRIFTING/DIVERGED)

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`_classify_drift_tier()`, `_write_telemetry_drift()`)

**Result:** Severity-based triage - know what needs immediate attention vs. monitoring.

---

### 4. Critical Metrics Tracking

**Problem:** Silent killers (label_window, horizon, leakage flags) change outcomes without obvious metric shifts.

**Solution:** Automatically track critical metrics from metadata:
- `label_window`, `horizon`, `lookahead_guard_state`
- `cv_scheme_id`, `fold_count`, `purge_gap`
- `leakage_flag`, `leakage_events_count`
- `missingness_rate`, `winsorization_clip`, `outlier_rate`

**Implementation:**
- Pulls from metadata if not in metrics dict
- Marked with `is_critical: true`
- Uses stricter drift thresholds (1% WARN, 3% ALERT)
- Included in drift comparison even if not in standard metrics

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`_write_telemetry_drift()`)

**Result:** Catch silent configuration changes that affect outcomes.

---

### 5. Sanity Checks

**Problem:** "All STABLE" could be a bug (self-comparison, wrong baseline key, masking changes).

**Solution:** Sanity checks detect suspicious patterns:
- **SELF_COMPARISON**: `run_id == baseline_run_id` (comparing to itself)
- **SUSPICIOUSLY_IDENTICAL**: All fingerprints identical but run_id differs (deterministic run or wrong baseline)
- **OK**: Baseline and current are different (normal case)

**Implementation:**
- Compares fingerprints and run_ids
- Sets `sanity_check.status` and `sanity_check.message`
- Included in Parquet files for querying

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`_write_telemetry_drift()`)

**Result:** Can't accidentally report false "all stable" - system flags suspicious cases.

---

### 6. Parquet Files for Queryable Data

**Problem:** JSON files are human-readable but inefficient for queries across many runs.

**Solution:** Write Parquet files alongside JSON at all levels:
- **Per-cohort**: `telemetry_metrics.parquet`, `telemetry_drift.parquet` (long format)
- **View-level**: `telemetry_rollup.parquet` (flattened, queryable)
- **Stage-level**: `telemetry_rollup.parquet` (flattened, queryable)

**Implementation:**
- All drift metrics flattened to long format
- Includes fingerprints, sanity checks, tiers, critical flags
- Efficient queries without JSON parsing explosion

**Files Changed:**
- `TRAINING/utils/telemetry.py` (all write methods)

**Result:** Efficient cross-run queries while keeping JSON for human readability.

---

## Enhanced Drift Results Structure

### JSON Structure

```json
{
  "current_run_id": "2025-12-14_12-00-00",
  "baseline_run_id": "2025-12-13_12-00-00",
  "baseline_key": "TARGET_RANKING:CROSS_SECTIONAL:fwd_ret_1d",
  "timestamp": "2025-12-14T12:00:00",
  "view": "CROSS_SECTIONAL",
  "target": "fwd_ret_1d",
  "symbol": null,
  "fingerprints": {
    "baseline": {
      "git_commit": "abc123",
      "config_hash": "def456",
      "data_fingerprint": "789abc",
      "timestamp": "2025-12-13T12:00:00"
    },
    "current": {
      "git_commit": "xyz789",
      "config_hash": "ghi012",
      "data_fingerprint": "345def",
      "timestamp": "2025-12-14T12:00:00"
    }
  },
  "sanity_check": {
    "status": "OK",
    "message": "Baseline and current runs are different"
  },
  "drift_metrics": {
    "mean_score": {
      "current": 0.751,
      "baseline": 0.750,
      "delta": 0.001,
      "rel_delta": 0.0013,
      "rel_delta_status": "defined",
      "baseline_zero": false,
      "tier": "OK",
      "is_critical": false,
      "status": "STABLE"
    },
    "pos_rate": {
      "current": 0.0,
      "baseline": 0.0,
      "delta": 0.0,
      "rel_delta": 0.0,
      "rel_delta_status": "both_zero",
      "baseline_zero": true,
      "tier": "OK",
      "is_critical": false,
      "status": "STABLE"
    },
    "label_window": {
      "current": 60,
      "baseline": 60,
      "delta": 0,
      "rel_delta": 0.0,
      "rel_delta_status": "defined",
      "baseline_zero": false,
      "tier": "OK",
      "is_critical": true,
      "status": "STABLE"
    }
  }
}
```

### Parquet Structure

Long format with one row per metric:
- Dimensions: `current_run_id`, `baseline_run_id`, `view`, `target`, `symbol`, `metric_name`
- Values: `current_value`, `baseline_value`, `delta`, `rel_delta`, `rel_delta_status`
- Metadata: `tier`, `is_critical`, `baseline_zero`, `sanity_check_status`
- Fingerprints: `baseline_git_commit`, `current_git_commit`, `baseline_config_hash`, `current_config_hash`, `baseline_data_fingerprint`, `current_data_fingerprint`

---

## Files Changed

### Modified Files

---

## Usage

### Automatic (No Changes Needed)

Enhanced drift tracking is automatically applied when telemetry is enabled. All drift files now include:
- Fingerprints proving baseline is different
- Explicit zero handling
- Severity tiers (OK/WARN/ALERT)
- Critical metrics tracking
- Sanity checks

### Querying Parquet Files

```python
import pandas as pd

# Load drift data
df = pd.read_parquet("REPRODUCIBILITY/TARGET_RANKING/CROSS_SECTIONAL/fwd_ret_1d/cohort=.../telemetry_drift.parquet")

# Find all ALERT-tier drifts
alerts = df[df["tier"] == "ALERT"]

# Find drifts where fingerprints are identical (suspicious)
suspicious = df[df["sanity_check_status"] == "SUSPICIOUSLY_IDENTICAL"]

# Find critical metric changes
critical_changes = df[df["is_critical"] == True]
```

---

## Benefits

1. **Provenance**: Can definitively answer "what changed?" (code, config, data, or stochasticity)
2. **Zero Handling**: No ambiguous nulls - explicit status for zero cases
3. **Severity Triage**: OK/WARN/ALERT tiers for prioritized response
4. **Silent Killers**: Critical metrics tracked automatically
5. **Bug Prevention**: Sanity checks prevent false "all stable" reports
6. **Queryable**: Parquet files enable efficient cross-run analysis

---

## Related Documentation

- [Telemetry System](2025-12-14-telemetry-system.md) - Initial telemetry implementation
- [Reproducibility Structure](../../03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md) - Directory structure

---

**Last Updated:** 2025-12-14
