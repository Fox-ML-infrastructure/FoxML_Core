# POST_PRUNE Invariant Check

## Purpose

Add a hard-fail invariant check at POST_PRUNE to prevent regression: `max(lookback_map[features]) == actual_max_from_features`.

## Implementation

**File**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 1117-1160)

**Check**: After computing `lookback_result` at POST_PRUNE, verify that:
- `max(canonical_map[features]) == computed_lookback` (within 1.0 minute tolerance)

**Behavior**:
- **Strict mode**: Hard-fail with `RuntimeError` if invariant violated
- **Non-strict mode**: Log error but continue (for debugging)

## Why This Matters

This invariant ensures:
1. **Canonical map consistency**: The canonical map used for lookback computation matches the reported max
2. **Prevents regression**: If canonical map computation changes, this check will catch it immediately
3. **Single source of truth**: Confirms that `computed_lookback` is derived from the canonical map (not recomputed separately)

## Expected Behavior

### Success Case
```
✅ INVARIANT CHECK (POST_PRUNE): max(canonical_map[features])=150.0m == computed_lookback=150.0m ✓
```

### Failure Case (Strict Mode)
```
🚨 INVARIANT VIOLATION (POST_PRUNE): max(canonical_map[features])=1440.0m != computed_lookback=150.0m. 
This indicates canonical map inconsistency.
RuntimeError: 🚨 INVARIANT VIOLATION (POST_PRUNE): ...
```

## SYMBOL_SPECIFIC View

**Status**: ✅ Same behavior as CROSS_SECTIONAL

SYMBOL_SPECIFIC view uses the same code paths for:
- Feature sanitization (`auto_quarantine_long_lookback_features`)
- Gatekeeper enforcement (`_enforce_final_safety_gate`)
- POST_PRUNE invariant check

The only differences are:
- Data filtering (per-symbol vs pooled)
- Telemetry scoping (INDIVIDUAL vs CROSS_SECTIONAL)
- Output directory structure

**Verification**: The invariant check runs at POST_PRUNE regardless of view, ensuring consistency across both views.

## Related Fixes

- **Single Source of Truth**: `compute_feature_lookback_max()` uses canonical map directly
- **Sanitizer**: Uses canonical map from `compute_feature_lookback_max()`
- **Gatekeeper**: Uses canonical map from `compute_feature_lookback_max()`
- **POST_GATEKEEPER sanity check**: Hard-fails on mismatch in strict mode

## Definition of Done ✅

- [x] Invariant check added at POST_PRUNE
- [x] Hard-fail in strict mode
- [x] Log error in non-strict mode
- [x] 1.0 minute tolerance for floating-point differences
- [x] Works for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
