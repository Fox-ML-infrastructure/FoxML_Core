# Unified Lookback Cap Enforcement Structure

**Date**: 2025-12-13  
**Goal**: Ensure all phases follow the same structure for consistency and maintainability.

## Standard Structure (apply_lookback_cap)

All lookback cap enforcement phases should follow this structure:

```
1. Build canonical lookback map (or use provided)
2. Quarantine features exceeding cap
3. Compute budget from safe features
4. Validate invariants (hard-fail in strict mode)
5. Log summary (one-liner per stage)
6. Return result
```

## Current State

### ✅ Feature Selection (FS_PRE, FS_POST)
- **Uses**: `apply_lookback_cap()` directly
- **Structure**: Follows standard ✅
- **Stages**: `FS_PRE_{view}`, `FS_POST_{view}`

### ✅ Target Ranking Gatekeeper (_enforce_final_safety_gate)
- **Uses**: `apply_lookback_cap()` internally (refactored)
- **Structure**: Follows standard ✅ (uses `apply_lookback_cap()` for core structure)
- **Stages**: `GATEKEEPER`
- **Extra Logic**: X matrix manipulation, daily pattern heuristics, dropped_tracker (preserved)
- **Difference**: Returns `(X, feature_names)` tuple instead of `LookbackCapResult` (preserves existing API)

### ✅ Feature Sanitizer (auto_quarantine_long_lookback_features)
- **Uses**: `apply_lookback_cap()` internally (refactored)
- **Structure**: Follows standard ✅ (uses `apply_lookback_cap()` for core structure)
- **Stages**: `feature_sanitizer`
- **Difference**: Returns `(safe_features, quarantined_features, quarantine_report)` instead of `LookbackCapResult` (preserves existing API)

## Refactoring Status

### ✅ Completed: Refactored to use apply_lookback_cap internally

**Gatekeeper**:
- ✅ Calls `apply_lookback_cap()` internally
- ✅ Extracts `safe_features` and `quarantined_features` from result
- ✅ Preserves gatekeeper-specific logic (X matrix manipulation, daily pattern heuristics, dropped_tracker)
- ✅ Returns `(X, feature_names)` tuple (preserves existing API)

**Sanitizer**:
- ✅ Calls `apply_lookback_cap()` internally
- ✅ Extracts `safe_features` and `quarantined_features` from result
- ✅ Builds `quarantine_report` from result metadata
- ✅ Returns `(safe_features, quarantined_features, quarantine_report)` (preserves existing API)

## Benefits

1. ✅ True single source of truth - all phases use the same canonical map and quarantine logic
2. ✅ Reduced code duplication - core structure is shared
3. ✅ Impossible for phases to diverge - all use `apply_lookback_cap()`
4. ✅ Easier to maintain and test - changes to core logic propagate automatically
5. ✅ Existing APIs preserved - no breaking changes
