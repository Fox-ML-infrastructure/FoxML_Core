# Determinism Fixes Applied

## Issues Fixed

### 1. ✅ Plugin Load Order (Preserve Config Order)

**Problem:** Sorting plugins alphabetically changed meaning if config order expressed precedence.

**Fix:** Preserve config order (YAML lists are ordered, so this is deterministic).
- Config order may express plugin precedence ("load A then B")
- Only warn if list is not sorted (helps catch accidental misordering)
- No longer silently reorder plugins

**Changed:**
- `plugin_loader.py`: Removed alphabetical sorting, preserves config order
- Added debug log if list is not sorted (helps catch issues)

### 2. ✅ Trace Durations (Removed from Context)

**Problem:** Storing durations in `context['hook_traces']` breaks determinism claim.

**Fix:** Durations logged but NOT stored in context.
- Trace in context contains only deterministic fields: hook names, order, errors
- Durations are logged for observability but not in returned context
- Preserves determinism: same inputs = same context outputs

**Changed:**
- `pipeline_hooks.py`: Removed `duration_ms` and `total_duration_ms` from trace stored in context
- Durations still logged for debugging
- Trace structure: `{'hook_point', 'hooks_executed': [{'hook', 'status', 'error?'}], 'errors': [...]}`

### 3. ✅ disable_hook Mode (Documented Limitation)

**Problem:** `disable_hook` mode can make runs path-dependent if hooks fail intermittently.

**Fix:** Documented limitation and added warnings.
- Added comments explaining path-dependency risk
- Warns when hook is disabled that this can affect determinism
- Recommendation: Use only for non-critical "best effort" hooks
- For deterministic runs, prefer `raise` mode or `log_and_continue`

**Changed:**
- Added warnings in code comments
- Added warning message when hook is disabled
- Documented that `_disabled_hooks` is per-process state

## Determinism Guarantee (Updated)

**What we guarantee:**
- ✅ Deterministic hook execution order (given deterministic plugin load order)
- ✅ Deterministic context outputs (no timing data in returned context)
- ✅ Config order preserved (no silent reordering)

**What we cannot guarantee:**
- ❌ Deterministic model outputs (requires all hooks + training steps to be deterministic)
- ❌ Path-independent behavior with `disable_hook` mode (if hooks fail intermittently)

## Config Compatibility

**Won't break SST configs:**
- ✅ Hooks config is optional (defaults to empty list)
- ✅ "No plugins enabled" behaves exactly like today (no-op)
- ✅ No schema changes required (hooks config is additive)

## Summary

All three determinism footguns have been fixed:
1. ✅ Plugin order preserved (no silent reordering)
2. ✅ Durations removed from context (deterministic outputs)
3. ✅ `disable_hook` limitation documented (path-dependency risk)

The hook system now maintains determinism for hook execution order and context outputs, while preserving config semantics.
