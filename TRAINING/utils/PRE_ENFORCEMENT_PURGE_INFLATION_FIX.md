# Pre-Enforcement Purge Inflation Fix

**Date**: 2025-12-13  
**Issue**: `create_resolved_config()` inflating purge based on pre-enforcement lookback

## Problem

`create_resolved_config()` is called BEFORE enforcement (gatekeeper), so it sees features with long lookback (e.g., 60d = 86400m) and inflates purge to 86405m. But gatekeeper will drop those features, so the final purge should be much lower (155m).

**Logs showed**:
- `feature_lookback_max = 86400m` (because 60d features exist pre-gatekeeper)
- `purge forced to 86405m`
- Later CV splitter correctly uses final features and sets purge to `155m`

**Risk**: If anything downstream accidentally uses the `resolved_config.purge_minutes` before the final recompute, it will use the inflated value.

## Fix

**Location**: `TRAINING/utils/resolved_config.py` lines 541-556

**Before**: Used `feature_lookback_max_minutes` directly for purge bump calculation

**After**: Cap lookback used for purge bump to `lookback_budget_cap` if lookback exceeds cap:

```python
# CRITICAL: In pre-enforcement stages, cap lookback used for purge bump
# Pre-enforcement lookback includes long-lookback features that will be dropped by gatekeeper
# Don't inflate purge based on pre-enforcement max - cap it to lookback budget cap
lookback_budget_cap = None
# ... load from config ...

# If we have a cap and lookback exceeds it, cap lookback for purge bump
lookback_for_purge = feature_lookback_max_minutes
if lookback_budget_cap is not None and feature_lookback_max_minutes > lookback_budget_cap:
    lookback_for_purge = lookback_budget_cap
    logger.debug(
        f"📊 Pre-enforcement purge guard: feature_lookback_max={feature_lookback_max_minutes:.1f}m > "
        f"cap={lookback_budget_cap:.1f}m. Capping lookback used for purge bump to {lookback_budget_cap:.1f}m "
        f"(gatekeeper will drop long-lookback features, final purge will be recomputed at POST_PRUNE)."
    )

# Use capped lookback for purge bump calculation
lookback_in = lookback_for_purge  # Capped lookback (if pre-enforcement) or original
```

## Result

✅ **Pre-enforcement purge capped**: If lookback > cap, use cap for purge bump (not full lookback)

✅ **Final purge correct**: CV splitter recomputes from final featureset (POST_PRUNE), gets correct purge (155m)

✅ **No downstream risk**: Even if something uses `resolved_config.purge_minutes` early, it's capped to budget cap (240m), not inflated to 86405m

## Testing

1. **Run with 60d features**: Should see debug log about capping lookback for purge bump
2. **Check purge values**: 
   - `create_resolved_config()` purge should be capped (not 86405m)
   - CV splitter purge should be correct (155m from final featureset)

## Related Files

- `TRAINING/utils/resolved_config.py`: Pre-enforcement purge guard
