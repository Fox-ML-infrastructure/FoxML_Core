# Resolved Config Fix: Consistent Logging and Purge/Embargo Derivation

## Problem Summary

The pipeline had inconsistent logging and multiple places computing purge/embargo values:

1. **min_cs inconsistency**: Logged as "min_cs=10" then "min_cs=5" without showing requested vs effective
2. **Feature count inconsistency**: "safe=307" then "pruned=304" without showing the full chain
3. **Purge/embargo drift**: Different modules computing different values (65m vs 85m)

## Solution

### 1. Created `ResolvedConfig` Object

**File**: `TRAINING/utils/resolved_config.py`

- Single source of truth for all resolved values
- Computes requested vs effective values once
- Centralized purge/embargo derivation
- Single authoritative log line

### 2. Centralized Purge/Embargo Derivation

**Function**: `derive_purge_embargo()`

Formula:
```python
base = horizon_minutes  # Feature lookback is NOT included (it's historical and safe)
buffer = purge_buffer_bars * interval_minutes
purge = embargo = base + buffer
```

**Note**: Feature lookback is historical data that doesn't need purging. Only the target's future window needs purging to prevent leakage.

**UPDATE (2025-12-12)**: The system now includes **Active Sanitization (Ghost Buster)** which proactively quarantines features with excessive lookback before training starts. This prevents "ghost feature" discrepancies where audit and auto-fix see different lookback values. See [Active Sanitization Guide](ACTIVE_SANITIZATION.md) for details.

**All modules now use this function** instead of local derivations.

### 3. Updated Logging

**Before**:
```
📊 Cross-sectional sampling: min_cs=10, max_cs_samples=1000
After min_cs=5 filter (requested 10, have 5 symbols): ...
```

**After**:
```
📊 Cross-sectional sampling: requested_min_cs=10 → effective_min_cs=5 (reason=only_5_symbols_loaded, n_symbols=5), max_cs_samples=1000
```

**Before**:
```
Filtered out 156 features (kept 307 safe features)
🔧 Dropped 3 all-NaN feature columns
features: safe=307 → pruned=304
```

**After**:
```
🔧 Features: safe=307 → drop_all_nan=3 → final=304
```

## Integration Points

### 1. `model_evaluation.py`

- Create `ResolvedConfig` at start of `evaluate_target_predictability()`
- Use centralized `derive_purge_embargo()` instead of local computation
- Log single authoritative summary using `resolved_config.log_summary()`
- Pass `resolved_config` to reproducibility tracker

### 2. `cross_sectional_data.py`

- Update logging to show requested vs effective min_cs
- Track feature counts (safe → dropped_nan → final)
- Return counts for resolved_config

### 3. `reproducibility_tracker.py`

- Use `resolved_config.purge_minutes` and `resolved_config.embargo_minutes`
- Store resolved_config values in metadata.json
- Ensure consistency with CV splitter

### 4. `RunContext`

- Use centralized `derive_purge_embargo()` in `compute_purge_embargo()`
- Remove local derivation logic

## Files Changed

1. **NEW**: `TRAINING/utils/resolved_config.py` - ResolvedConfig class and centralized derivation
2. **MODIFIED**: `TRAINING/ranking/predictability/model_evaluation.py` - Use resolved_config
3. **MODIFIED**: `TRAINING/utils/cross_sectional_data.py` - Update logging and return counts
4. **MODIFIED**: `TRAINING/utils/run_context.py` - Use centralized derivation
5. **MODIFIED**: `TRAINING/utils/reproducibility_tracker.py` - Use resolved_config values

## Verification

After fix, grep logs for a single target evaluation:

```bash
grep "Cross-sectional sampling\|Temporal safety\|Features:" logs/*.log
```

Should see **one** authoritative line for each category:
- One "Cross-sectional sampling" line with requested/effective
- One "Temporal safety" line with horizon/purge/embargo
- One "Features" line with safe→dropped→final chain

## Definition of Done

✅ Single authoritative log line per category per target evaluation
✅ Purge/embargo computed in one place and used everywhere
✅ Reproducibility artifacts store same purge/embargo as CV splitter
✅ All modules use `ResolvedConfig` or `derive_purge_embargo()`
