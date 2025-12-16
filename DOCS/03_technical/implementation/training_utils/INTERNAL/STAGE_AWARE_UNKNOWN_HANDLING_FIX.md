# Stage-Aware Unknown Lookback Handling Fix

**Date**: 2025-12-13  
**Issue**: `compute_budget()` hard-failing on unknowns in pre-enforcement stages

## Problem

`create_resolved_config()` calls `compute_feature_lookback_max()` → `compute_budget()` BEFORE enforcement runs. This is legitimate (needed to set up initial purge/embargo), but the new strict enforcement was hard-failing on unknowns.

**Error**:
```
🚨 compute_budget(create_resolved_config): 84 features have unknown lookback (inf). 
This indicates a bug: compute_budget() was called on features that should have been quarantined.
```

## Root Cause

`compute_budget()` was hard-failing on unknowns regardless of stage. But:
- **Pre-enforcement stages** (create_resolved_config, initial setup) are EXPECTED to see unknowns
- **Post-enforcement stages** (POST_PRUNE, POST_GATEKEEPER) should NEVER see unknowns

## Fix

Made `compute_budget()` stage-aware:

**Location**: `TRAINING/utils/leakage_budget.py` lines 974-1017

```python
# Stage-aware handling
is_pre_enforcement_stage = any(
    pre_stage in stage.lower() 
    for pre_stage in ["create_resolved_config", "pre_", "initial", "baseline"]
)
is_post_enforcement_stage = any(
    post_stage in stage.lower()
    for post_stage in ["post_prune", "post_gatekeeper", "post_", "gatekeeper_budget", "enforced"]
)

if is_post_enforcement_stage:
    # Hard-fail: unknowns should not reach compute_budget in post-enforcement stages
    if policy == "strict":
        raise RuntimeError(...)
elif is_pre_enforcement_stage:
    # Log DEBUG (not WARNING): This is expected in pre-enforcement stages
    logger.debug(f"📊 compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
                 f"This is expected in pre-enforcement stages. Enforcement will drop these later.")
else:
    # Unknown stage - log WARNING (conservative)
    logger.warning(...)
```

## Result

✅ **Pre-enforcement stages**: Log DEBUG (expected), don't hard-fail

✅ **Post-enforcement stages**: Hard-fail in strict mode (bug detection)

✅ **Unknown stages**: Log WARNING (conservative)

## Testing

Run should now proceed past `create_resolved_config()` without hard-failing. Unknowns will be logged at DEBUG level, and enforcement will drop them later at gatekeeper/POST_PRUNE.
