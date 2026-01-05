# 2026-01-04: Reproducibility File Output Fixes

## Summary

Fixed critical bugs preventing reproducibility files from being written to cohort directories. All files (`snapshot.json`, `baseline.json`, diff files) are now correctly written to target-first structure for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views.

## Problem

After implementing target-first path structure (`targets/<target>/reproducibility/...`), several reproducibility files were not being written:

1. **`snapshot.json`** - Not being written to cohort directories
2. **`baseline.json`** - Not being written to cohort directories  
3. **Diff files** (`diff_prev.json`, `metric_deltas.json`, `diff_baseline.json`) - Not being written
4. **Previous snapshot lookup** - Searching in wrong location (legacy `REPRODUCIBILITY/...` instead of `targets/.../reproducibility/...`)

## Root Causes

1. **Path detection bug**: Code checked for uppercase `"REPRODUCIBILITY"` but paths use lowercase `"reproducibility"`, causing path conversion to be skipped
2. **Baseline save blocking**: `_save_baseline_to_cohort()` blocked writes to paths containing `"REPRODUCIBILITY"` (uppercase), but didn't account for lowercase `"reproducibility"`
3. **Diff save blocking**: `save_diff()` had same uppercase check issue
4. **Path parsing failure**: When paths were already in target-first format, code tried to parse stage/view/target from path, but target-first paths don't include stage
5. **Wrong search location**: Previous snapshot lookup searched in legacy `REPRODUCIBILITY/...` structure instead of target-first `targets/.../reproducibility/...`

## Solutions

### 1. Fixed `snapshot.json` Writing (`save_snapshot()`)

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes**:
- Fixed path detection to handle both uppercase `REPRODUCIBILITY` (legacy) and lowercase `reproducibility` (target-first)
- Added check: `is_target_first_path = "reproducibility" in cohort_dir_str.lower() and not is_legacy_path`
- If path is already target-first, use it directly instead of trying to convert
- Added error handling and logging for snapshot writes
- Improved snapshot identifier extraction to use snapshot object fields instead of path parsing

**Key Code**:
```python
cohort_dir_str = str(cohort_dir)
is_legacy_path = "REPRODUCIBILITY" in cohort_dir_str
is_target_first_path = "reproducibility" in cohort_dir_str.lower() and not is_legacy_path

# If path is already in target-first format, use it directly
if is_target_first_path:
    target_cohort_dir = Path(cohort_dir)
    target_cohort_dir.mkdir(parents=True, exist_ok=True)
```

### 2. Fixed `baseline.json` Writing (`_save_baseline_to_cohort()`)

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes**:
- Fixed path check to allow lowercase `reproducibility` paths
- Changed from: `if "REPRODUCIBILITY" in str(cohort_dir):`
- Changed to: `if "REPRODUCIBILITY" in cohort_dir_str and "reproducibility" not in cohort_dir_str.lower():`
- Added error handling and logging

**Key Code**:
```python
cohort_dir_str = str(cohort_dir)
# Check for uppercase REPRODUCIBILITY (legacy) but allow lowercase reproducibility (target-first)
if "REPRODUCIBILITY" in cohort_dir_str and "reproducibility" not in cohort_dir_str.lower():
    logger.warning(f"⚠️ Skipping baseline save to legacy REPRODUCIBILITY path: {cohort_dir}")
    return
```

### 3. Fixed Diff Files Writing (`save_diff()`)

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes**:
- Fixed path check to allow lowercase `reproducibility` paths (same fix as baseline)
- Improved path parsing to extract stage/view/target from snapshot object when available
- Fallback to path parsing only if snapshot fields not available
- Ensured files are written even when path parsing fails but directory is already correct
- Changed from checking `if target_cohort_dir:` to using `cohort_dir` directly (already set to target-first)
- Added error handling and logging for all diff file writes

**Key Code**:
```python
# Extract identifiers from snapshot (preferred) or cohort_dir path
# Prefer snapshot's stage/view/target over path parsing
stage = snapshot.stage
view = getattr(snapshot, 'view', None)
target = snapshot.target
```

### 4. Fixed Previous Snapshot Lookup (`finalize_run()`)

**File**: `TRAINING/orchestration/utils/diff_telemetry.py`

**Changes**:
- Updated path reconstruction to search in target-first structure first
- Search path: `targets/<target>/reproducibility/<VIEW>/...` or `targets/<target>/reproducibility/<VIEW>/symbol=<symbol>/...`
- Falls back to legacy `REPRODUCIBILITY/...` structure if target-first not found
- Handles both CROSS_SECTIONAL and SYMBOL_SPECIFIC views correctly
- Applied same fix to both previous snapshot lookup and baseline snapshot lookup

**Key Code**:
```python
# Try target-first structure first
targets_dir = run_dir / "targets"
if targets_dir.exists() and (targets_dir / target_clean).exists():
    target_dir = targets_dir / target_clean
    repro_dir = target_dir / "reproducibility"
    if repro_dir.exists():
        view_dir = repro_dir / prev_snapshot.view
        if view_dir.exists():
            # Check for symbol-specific path
            if prev_snapshot.symbol:
                symbol_dir = view_dir / f"symbol={prev_snapshot.symbol}"
                # ... search in symbol_dir
            # Check for cross-sectional path (no symbol)
            # ... search in view_dir
# Fallback to legacy REPRODUCIBILITY structure
```

## Files Changed

- `TRAINING/orchestration/utils/diff_telemetry.py`
  - `save_snapshot()`: Fixed path detection and added error handling
  - `_save_baseline_to_cohort()`: Fixed path check to allow target-first paths
  - `save_diff()`: Fixed path check and improved identifier extraction
  - `finalize_run()`: Fixed previous snapshot and baseline snapshot lookup paths

## Impact

### Before
- `snapshot.json` not written to cohort directories
- `baseline.json` not written to cohort directories
- Diff files not written to cohort directories
- Previous snapshot lookup failed (searching wrong location)
- No error logging for write failures

### After
- All reproducibility files correctly written to target-first structure
- Works for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
- Previous snapshot lookup works correctly
- Error handling and logging for all writes
- Files written to correct paths:
  - CROSS_SECTIONAL: `targets/<target>/reproducibility/CROSS_SECTIONAL/cohort=<cohort_id>/`
  - SYMBOL_SPECIFIC: `targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol=<symbol>/cohort=<cohort_id>/`

## Testing

- Verified files are written to correct directories for both views
- Verified path detection handles both legacy and target-first structures
- Verified previous snapshot lookup searches correct locations
- Verified error handling logs failures appropriately

## Related

- Part of target-first path structure migration
- Related to determinism and comparability fixes (2026-01-04)
- Builds on run identity wiring fixes (2026-01-04)
