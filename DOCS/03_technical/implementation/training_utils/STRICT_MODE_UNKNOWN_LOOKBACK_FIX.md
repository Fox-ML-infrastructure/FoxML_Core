# Strict Mode Unknown Lookback Contract Fix

**Date**: 2025-12-13  
**Related**: [SST Enforcement Design](SST_ENFORCEMENT_DESIGN.md) | [Stage-Aware Unknown Handling](STAGE_AWARE_UNKNOWN_HANDLING_FIX.md)

## Problem

In strict mode, the system was logging warnings about unknown lookback features in pre-enforcement stages (like `create_resolved_config`), but the contract wasn't clear:

- Pre-enforcement stages see unknowns (expected)
- Post-enforcement stages (POST_GATEKEEPER, POST_PRUNE) should have ZERO unknowns
- But there was no hard-fail check to enforce this contract

## Solution

### 1. Pre-Enforcement: Log at INFO in Strict Mode

**Location**: `TRAINING/utils/leakage_budget.py` lines 1022-1035

**Change**: In strict mode, log unknown lookback at INFO (not DEBUG) to make contract visible:

```python
if policy == "strict":
    # In strict mode, log at INFO to make contract visible (not hidden in DEBUG)
    logger.info(
        f"📊 compute_budget({stage}): {len(unknown_features)} features have unknown lookback (inf). "
        f"This is expected in pre-enforcement stages. Enforcement (gatekeeper/sanitizer) will quarantine these. "
        f"Sample: {unknown_features[:5]}"
    )
```

**Rationale**: Makes the contract visible that unknowns exist and will be quarantined later.

### 2. Post-Enforcement: Hard-Fail on Unknowns

**Location**: 
- `TRAINING/ranking/predictability/model_evaluation.py` lines 4760-4780 (POST_GATEKEEPER)
- `TRAINING/ranking/predictability/model_evaluation.py` lines 1078-1095 (POST_PRUNE)

**Change**: Added hard-fail check that post-enforcement stages have ZERO unknowns:

```python
# CRITICAL: Hard-fail check: POST_GATEKEEPER/POST_PRUNE must have ZERO unknowns in strict mode
if len(enforced.unknown) > 0:
    if policy == "strict":
        error_msg = (
            f"🚨 POST_GATEKEEPER CONTRACT VIOLATION: {len(enforced.unknown)} features have unknown lookback (inf). "
            f"In strict mode, post-enforcement stages must have ZERO unknowns. "
            f"Gatekeeper should have quarantined these."
        )
        raise RuntimeError(f"{error_msg} (policy: strict - training blocked)")
```

**Rationale**: Enforces the contract that post-enforcement stages should never see unknowns.

### 3. Budget Caching to Reduce Log Noise

**Location**: `TRAINING/utils/leakage_budget.py` lines 45, 682-694, 1095-1107

**Change**: Added budget cache keyed by `(featureset_fingerprint, interval_minutes, horizon_minutes, cap_minutes, stage)`:

```python
# Budget cache: keyed by (featureset_fingerprint, interval_minutes, horizon_minutes, cap_minutes, stage)
_budget_cache: Dict[Tuple[str, float, float, Optional[float], str], Tuple[Any, str, str]] = {}

# Check cache first
cache_key = (set_fingerprint, interval_minutes, horizon_minutes, max_lookback_cap_minutes, stage)
if cache_key in _budget_cache and canonical_lookback_map is None:
    cached_budget, cached_fp, cached_order_fp = _budget_cache[cache_key]
    logger.debug(f"📋 Budget cache hit ({stage}): fingerprint={set_fingerprint[:8]}")
    return cached_budget, cached_fp, cached_order_fp

# ... compute budget ...

# Cache result
_budget_cache[cache_key] = (budget, set_fingerprint, order_fingerprint)
```

**Rationale**: Reduces log noise from repeated `compute_budget` calls at the same stage with the same featureset.

## Results

✅ **Pre-enforcement**: Logs at INFO in strict mode (contract visible)  
✅ **Post-enforcement**: Hard-fails on unknowns in strict mode (contract enforced)  
✅ **Log noise**: Reduced via budget caching (one-line summary per unique featureset/stage)

## Related Issues

- Shell/library readline error: Environment issue (not code). Check `which sh` and `echo $LD_LIBRARY_PATH`.
