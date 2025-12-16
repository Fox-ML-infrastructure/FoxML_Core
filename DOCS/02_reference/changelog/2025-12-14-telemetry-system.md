# Metrics System Implementation (2025-12-14) [Renamed from Telemetry on 2025-12-15]

## Summary

Implemented a sidecar-based metrics system with view isolation. Metrics files live alongside existing artifacts (metadata.json, metrics.json, audit_report.json) in cohort directories, following the exact same directory structure. View isolation ensures CROSS_SECTIONAL drift only compares to CROSS_SECTIONAL baselines, and SYMBOL_SPECIFIC drift only compares to SYMBOL_SPECIFIC baselines.

**Note:** This system was renamed from "telemetry" to "metrics" on 2025-12-15. See [Metrics System Rename](2025-12-15-metrics-rename.md) for details. All references in this document use the original "telemetry" terminology for historical accuracy.

---

## New Features

### 1. Sidecar-Based Telemetry Structure

**Problem:** Need telemetry that matches existing artifact structure and enables per-target, per-symbol, and per-cross-sectional drift tracking.

**Solution:** Telemetry files written as sidecars in cohort directories:
- `telemetry_metrics.json` + `telemetry_metrics.parquet` - Telemetry facts for this cohort
- `telemetry_drift.json` + `telemetry_drift.parquet` - Drift comparison vs baseline (view-isolated)
- `telemetry_trend.json` - Optional trend analysis (after N runs)

**Files Changed:**
- `TRAINING/utils/metrics.py` (new file, originally named `telemetry.py`, renamed 2025-12-15)
- `TRAINING/utils/reproducibility_tracker.py` (integration)

**Structure:**
```
REPRODUCIBILITY/TARGET_RANKING/
  CROSS_SECTIONAL/
    y_will_swing_high_60m_0.05/
      cohort=cr_.../
        metadata.json
        metrics.json
        audit_report.json
        telemetry_metrics.json      # NEW
        telemetry_drift.json        # NEW
        telemetry_trend.json        # NEW (optional)
    telemetry_rollup.json           # NEW: aggregated across all CS targets
  SYMBOL_SPECIFIC/
    y_will_swing_high_60m_0.05/
      symbol=AAPL/
        cohort=sy_.../
          metadata.json
          metrics.json
          audit_report.json
          telemetry_metrics.json    # NEW
          telemetry_metrics.parquet  # NEW (queryable)
          telemetry_drift.json      # NEW
          telemetry_drift.parquet   # NEW (queryable)
          telemetry_trend.json      # NEW (optional)
    telemetry_rollup.json           # NEW: aggregated across all SS symbols/targets
    telemetry_rollup.parquet       # NEW (queryable)
  telemetry_rollup.json             # NEW: stage-level container
  telemetry_rollup.parquet          # NEW (queryable)
```

**Result:** Telemetry follows existing artifact structure exactly, making it easy to find and compare.

---

### 2. View Isolation Enforcement

**Problem:** Need to ensure drift comparisons never mix views (CROSS_SECTIONAL vs SYMBOL_SPECIFIC).

**Solution:** Baseline key format: `(stage, view, target[, symbol])`

**Implementation:**
- Baseline key format: `TARGET_RANKING:CROSS_SECTIONAL:y_will_swing_high_60m_0.05`
- For SYMBOL_SPECIFIC: `TARGET_RANKING:SYMBOL_SPECIFIC:y_will_swing_high_60m_0.05:AAPL`
- `_find_baseline_cohort()` validates view matches before comparison
- Drift comparison only proceeds if baseline_view == current_view

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`_find_baseline_cohort()`, `_write_telemetry_drift()`)

**Result:** View isolation guaranteed - no accidental cross-wiring between CROSS_SECTIONAL and SYMBOL_SPECIFIC.

---

### 3. Hierarchical Rollups

**Problem:** Need aggregated metrics at view-level and stage-level for high-level analysis.

**Solution:** Three-level rollup hierarchy:
1. **Cohort-level**: Sidecar files in each cohort directory
2. **View-level**: `CROSS_SECTIONAL/telemetry_rollup.json`, `SYMBOL_SPECIFIC/telemetry_rollup.json`
3. **Stage-level**: `TARGET_RANKING/telemetry_rollup.json` (container with both views)

**Implementation:**
- `generate_view_rollup()`: Aggregates metrics across all targets/symbols in a view
- `generate_stage_rollup()`: Container that references view-level rollups (no drift mixing)
- `generate_telemetry_rollups()`: Convenience method to generate all rollups for a stage

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`generate_view_rollup()`, `generate_stage_rollup()`)
- `TRAINING/utils/reproducibility_tracker.py` (`generate_telemetry_rollups()`)

**Result:** Can analyze drift at any level: per-target, per-symbol, per-view, or per-stage.

---

### 4. CROSS_SECTIONAL vs SYMBOL_SPECIFIC Scope

**Problem:** Need different telemetry scopes for different views.

**Solution:**
- **CROSS_SECTIONAL**: Per-target only (no per-symbol telemetry in CS view)
- **SYMBOL_SPECIFIC**: Per-symbol AND per-target-under-symbol (where leaf JSONs exist)

**Implementation:**
- `write_cohort_telemetry()` writes sidecar files based on view type
- View-level rollups aggregate appropriately (CS: across targets, SS: across symbols/targets)

**Files Changed:**
- `TRAINING/utils/telemetry.py` (`write_cohort_telemetry()`, `generate_view_rollup()`)

**Result:** Telemetry scope matches view semantics - CS is target-focused, SS is symbol-focused.

---

### 5. Configuration Integration

**Problem:** Need config-driven telemetry behavior.

**Solution:** Added `safety.telemetry` section to `safety.yaml`:
- `enabled`: Enable/disable telemetry
- `baselines`: Baseline configuration (previous_run, rolling_window_k, last_good_run)
- `drift`: Drift thresholds (psi_threshold, ks_threshold)

**Files Changed:**
- `CONFIG/pipeline/training/safety.yaml` (added `safety.telemetry` section)
- `TRAINING/utils/telemetry.py` (`load_telemetry_config()`)

**Result:** All telemetry behavior controlled by config, deterministic and reproducible.

---

## Integration

### Reproducibility Tracker Integration

**Automatic sidecar writing:**
- `_save_to_cohort()` now calls `telemetry.write_cohort_telemetry()` automatically
- Extracts view, target, symbol from route_type and item_name
- Generates baseline_key for drift comparison
- Writes sidecar files in same cohort directory

**Files Changed:**
- `TRAINING/utils/reproducibility_tracker.py` (`_save_to_cohort()`)

**Result:** Telemetry automatically recorded for all cohort saves, no manual intervention needed.

---

## Configuration

### New Config Section: `safety.telemetry`

```yaml
safety:
  telemetry:
    enabled: true
    baselines:
      previous_run: true
      rolling_window_k: 10
      last_good_run: true
    drift:
      psi_threshold: 0.2
      ks_threshold: 0.1
```

**Files Changed:**
- `CONFIG/pipeline/training/safety.yaml`

---

## Files Changed

### New Files
- `TRAINING/utils/telemetry.py` - Core telemetry system

### Modified Files
- `TRAINING/utils/reproducibility_tracker.py` - Integration with telemetry writer
- `CONFIG/pipeline/training/safety.yaml` - Added telemetry configuration section

---

## Usage

### Automatic (Recommended)

Telemetry is automatically written when `ReproducibilityTracker._save_to_cohort()` is called. No code changes needed.

### Manual Rollup Generation

After all cohorts for a stage are saved, generate rollups:

```python
tracker = ReproducibilityTracker(output_dir=Path("results"))
tracker.generate_telemetry_rollups(stage="TARGET_RANKING", run_id="2025-12-14_12-00-00")
```

This generates:
- `CROSS_SECTIONAL/telemetry_rollup.json`
- `SYMBOL_SPECIFIC/telemetry_rollup.json`
- `TARGET_RANKING/telemetry_rollup.json`

---

## Benefits

1. **Structure Matching**: Telemetry follows exact same directory structure as existing artifacts
2. **View Isolation**: No accidental drift comparisons across views
3. **Hierarchical Analysis**: Can analyze at any level (target, symbol, view, stage)
4. **Sidecar Placement**: Easy to find telemetry next to existing JSONs
5. **Config-Driven**: All behavior controlled by config, deterministic

---

## Related Documentation

- [Reproducibility Structure](03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md)
- [Reproducibility API](03_technical/implementation/REPRODUCIBILITY_API.md)

---

---

## Related Enhancements

- **[Enhanced Drift Tracking](2025-12-14-drift-tracking-enhancements.md)** - Fingerprints, drift tiers, critical metrics, sanity checks, Parquet files

---

**Last Updated:** 2025-12-14
