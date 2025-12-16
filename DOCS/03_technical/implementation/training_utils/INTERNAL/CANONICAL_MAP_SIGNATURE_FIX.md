# Canonical Map Signature Fix

**Date**: 2025-12-13  
**Issue**: `compute_feature_lookback_max() got an unexpected keyword argument 'canonical_lookback_map'`

## Problem

The wrapper function `compute_feature_lookback_max()` in `resolved_config.py` didn't accept `canonical_lookback_map` parameter, but callsites in `model_evaluation.py` were trying to pass it for SST reuse.

**Error**:
```
TypeError: compute_feature_lookback_max() got an unexpected keyword argument 'canonical_lookback_map'
```

## Root Cause

The wrapper in `resolved_config.py` was missing the `canonical_lookback_map` parameter, even though:
1. The underlying `leakage_budget.compute_feature_lookback_max()` accepts it
2. Callsites want to pass it for SST reuse (avoid recompute + guarantee same oracle)

## Fix

**Location**: `TRAINING/utils/resolved_config.py` lines 259-312

Added `canonical_lookback_map` parameter to wrapper signature and pass it through:

```python
def compute_feature_lookback_max(
    ...
    canonical_lookback_map: Optional[Dict[str, float]] = None  # NEW: Optional pre-computed canonical map (SST reuse)
) -> Tuple[Optional[float], List[Tuple[str, float]]]:
    ...
    result = leakage_budget.compute_feature_lookback_max(
        ...
        canonical_lookback_map=canonical_lookback_map  # NEW: Pass canonical map for SST reuse
    )
```

## Result

✅ **Wrapper accepts `canonical_lookback_map`** - Callsites can pass it for SST reuse

✅ **Passes through to underlying function** - No recompute, same oracle guaranteed

✅ **Backward compatible** - Parameter is optional, defaults to None

## Testing

Run should now proceed past POST_PRUNE without signature mismatch error. Canonical map from enforcement is reused, avoiding recompute and ensuring same oracle.
