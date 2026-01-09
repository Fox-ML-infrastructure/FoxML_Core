# Audit Report: Snapshotting and Batch Universe Handling

**Date**: 2026-01-08  
**Scope**: All three stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING)  
**Focus**: Universe signature computation, batch handling, snapshot creation  
**Status**: ✅ **FIXES APPLIED** - All critical issues resolved

## Executive Summary

**CRITICAL ISSUES FOUND**: 3  
**WARNINGS**: 2  
**RECOMMENDATIONS**: 2

The audit identified critical issues with universe signature computation that could cause snapshots to be incorrectly scoped when batches have different universes. All issues are fixable without breaking existing functionality.

## Critical Issues

### Issue 1: Universe Signature Computed from Batch Instead of Full Run (CRITICAL)

**Severity**: CRITICAL  
**Impact**: Snapshots from different batches will have different `universe_sig` values, causing incorrect scoping and comparison group mismatches

**Locations**:
1. `TRAINING/training_strategies/execution/training.py:373` - Fallback identity creation
2. `TRAINING/training_strategies/execution/training.py:1118-1119` - Universe_sig computation

**Problem**:

1. **Fallback Identity Uses Batch Universe** (line 373):
```python
symbols = list(mtf_data.keys()) if mtf_data else []
effective_run_identity = create_stage_identity("TRAINING", symbols=symbols, ...)
```
   - If `mtf_data` is a batch subset (e.g., filtered by target availability), `run_identity` will have batch universe_sig in `dataset_signature`
   - This causes all snapshots created with this identity to be incorrectly scoped

2. **Universe_sig Computed from Batch** (line 1118-1119):
```python
full_universe = list(mtf_data.keys()) if mtf_data else [symbol]
universe_sig = compute_universe_signature(full_universe)
```
   - Should prefer `run_identity.dataset_signature` (full run universe) over `mtf_data.keys()` (batch subset)
   - Currently computes from batch, causing incorrect scoping for SYMBOL_SPECIFIC snapshots

**Root Cause**: 
- `intelligent_trainer.py:2761-2763` correctly creates identity with `self.symbols` (full universe)
- But `training.py:373` creates fallback with `mtf_data.keys()` (could be batch subset)
- And `training.py:1118` computes from `mtf_data.keys()` instead of using `run_identity.dataset_signature`

**Fix Required**:
- Always prefer `run_identity.dataset_signature` for universe_sig (full run universe)
- Only compute from `mtf_data.keys()` if `run_identity` is None AND no full universe available
- Add validation to ensure universe_sig matches run_identity when available

**Files to Fix**:
- `TRAINING/training_strategies/execution/training.py:373` - Use full universe if available
- `TRAINING/training_strategies/execution/training.py:1118-1119` - Prefer `run_identity.dataset_signature`

---

### Issue 2: Missing Universe Signature in Training Snapshots (CRITICAL)

**Severity**: CRITICAL  
**Impact**: Snapshots may be unscoped (legacy behavior), causing comparison groups to mismatch and run hash computation to be incorrect

**Location**: `TRAINING/training_strategies/reproducibility/schema.py:288-289`

**Problem**:
```python
if identity.get("universe_sig"):
    comparison_group["universe_sig"] = identity["universe_sig"]
```

- `RunIdentity.to_dict()` returns `dataset_signature`, NOT `universe_sig` (see `fingerprinting.py:340-357`)
- `TrainingSnapshot.from_training_result()` only checks `identity.get("universe_sig")` 
- The identity dict conversion does NOT map `dataset_signature` → `universe_sig`
- Result: `universe_sig` is never populated in comparison_group, even when `run_identity` has `dataset_signature`

**Evidence**:
- `reproducibility_tracker.py:1376-1378` already has fallback logic:
  ```python
  if full_metadata.get('universe_sig') is None and hasattr(run_identity, 'dataset_signature'):
      full_metadata['universe_sig'] = run_identity.dataset_signature[:12]
  ```
- But `TrainingSnapshot.from_training_result()` doesn't have this fallback

**Fix Required**:
- Check both `identity.get("universe_sig")` AND `identity.get("dataset_signature")`
- Add fallback: `identity.get("universe_sig") or identity.get("dataset_signature")`
- Optionally: Update `RunIdentity.to_dict()` to include `universe_sig` as alias for backward compatibility

**Files to Fix**:
- `TRAINING/training_strategies/reproducibility/schema.py:288-289` - Add dataset_signature fallback

---

### Issue 3: Inconsistent Universe Signature Extraction Across Stages (CRITICAL)

**Severity**: CRITICAL  
**Impact**: Different stages handle universe_sig differently, causing inconsistent scoping

**Locations**:
- TARGET_RANKING: `diff_telemetry.py:669-675` - Extracts from `additional_data.get('universe_sig')` or `cs_config`
- FEATURE_SELECTION: `feature_selector.py:1564-1577` - Uses `universe_sig` parameter with validation
- TRAINING: `training.py:1118-1119` - Computes from `mtf_data.keys()` (WRONG - should use run_identity)

**Problem**:
- TARGET_RANKING and FEATURE_SELECTION extract universe_sig from run_identity or parameters
- TRAINING computes from `mtf_data.keys()` which could be a batch subset
- This inconsistency causes snapshots to have different scoping behavior

**Fix Required**:
- Standardize: All stages should prefer `run_identity.dataset_signature` for universe_sig
- Add helper function: `extract_universe_sig_from_identity(identity)` that checks both `universe_sig` and `dataset_signature`
- Update TRAINING stage to use this helper

**Files to Fix**:
- `TRAINING/training_strategies/execution/training.py:1118-1119` - Use helper function
- Create helper in `TRAINING/orchestration/utils/reproducibility/utils.py` (or use existing `extract_universe_sig`)

---

## Warnings

### Warning 1: Aggregated Snapshot Cohort ID Uses First Symbol's Metadata

**Severity**: WARNING  
**Impact**: May cause incorrect cohort_id if symbols have different date ranges or sample counts

**Location**: `TRAINING/training_strategies/execution/training.py:1212-1236`

**Problem**:
- Uses first symbol's metadata as proxy for aggregated cohort_id
- If symbols have different date ranges or sample counts, cohort_id may be incorrect
- However, this may be acceptable if all symbols in a batch share the same date range

**Recommendation**:
- Verify if this is acceptable for your use case
- If not, aggregate metadata from all symbols before computing cohort_id

---

### Warning 2: Silent Failures in Snapshot Creation

**Severity**: WARNING  
**Impact**: Missing snapshots without clear indication, run hash computation may fail silently

**Locations**:
- `TRAINING/training_strategies/reproducibility/io.py:590-592` - Returns None on exception
- `TRAINING/training_strategies/execution/training.py:977-978` - Logs warning, continues
- `TRAINING/stability/feature_importance/hooks.py:285-286` - Logs error, continues

**Current Behavior**:
- Snapshot creation failures are logged but don't stop execution
- This is likely intentional (non-critical path)
- But may cause missing snapshots without clear indication

**Recommendation**:
- Verify logging levels are appropriate (WARNING vs ERROR)
- Consider adding validation that snapshots were created after pipeline completes
- Document that snapshot failures are non-fatal

---

## Recommendations

### Recommendation 1: Add Universe Signature Validation

**Priority**: HIGH  
**Impact**: Prevents unscoped snapshots

**Action**:
- Add validation in `TrainingSnapshot.from_training_result()` to ensure `universe_sig` is populated
- Log WARNING if `universe_sig` is missing (but don't fail)
- Document that unscoped snapshots are legacy behavior

---

### Recommendation 2: Standardize Universe Signature Extraction

**Priority**: HIGH  
**Impact**: Consistent behavior across all stages

**Action**:
- Create helper function: `extract_universe_sig_from_identity(identity)` 
- Use in all three stages for consistent extraction
- Helper should check: `identity.get("universe_sig") or identity.get("dataset_signature")`

---

## Fix Priority

1. **P0 (CRITICAL)**: Issue 1, Issue 2, Issue 3 - Fix immediately
2. **P1 (HIGH)**: Recommendation 1, Recommendation 2 - Fix soon
3. **P2 (MEDIUM)**: Warning 1 - Verify if acceptable
4. **P3 (LOW)**: Warning 2 - Document behavior

## Testing Requirements

After fixes, verify:
1. Snapshots from different batches have same `universe_sig` (full run universe)
2. Snapshots always have `universe_sig` populated in comparison_group
3. All three stages extract universe_sig consistently
4. Run hash computation works correctly with fixed snapshots
5. Backward compatibility: Old snapshots without universe_sig still work

## Backward Compatibility

All fixes maintain backward compatibility:
- Old snapshots without `universe_sig` will still work (legacy behavior)
- New snapshots will have `universe_sig` populated correctly
- No breaking changes to existing snapshot schema
