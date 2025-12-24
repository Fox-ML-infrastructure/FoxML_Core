# Hook System Final Summary

## Status: ✅ **Production-Ready**

The hook system is now hardened, deterministic, and safe for FoxML going forward.

## Determinism Guarantees

### ✅ What We Guarantee

1. **Deterministic hook execution order**
   - Same code + same config + same plugin load order = same hook execution order
   - Sorting: `(priority, registration_index, stable_tiebreak)`

2. **Deterministic context outputs**
   - Durations logged but NOT stored in context
   - Trace contains only deterministic fields: hook names, order, errors

3. **Config order preserved**
   - Plugins loaded in config order (preserves intended precedence)
   - No silent reordering

### ❌ What We Cannot Guarantee

1. **Deterministic model outputs** - Requires all hooks + training steps to be deterministic
2. **Path-independent behavior with `disable_hook`** - If hooks fail intermittently

## SST Config Compatibility

**Won't break SST configs if:**
- ✅ Hooks config is optional (defaults to empty list)
- ✅ "No plugins enabled" behaves exactly like today (no-op)
- ✅ Config validator allows unknown keys OR schema updated to include `hooks` as optional

**May break if:**
- ❌ Config validator is strict about unknown keys AND schema not updated

## Governance Rules

### ✅ Hooks Are For:
- Optional cross-cutting concerns (leakage detection, stability analysis, reporting)
- Features that can fail gracefully without breaking the pipeline

### ❌ Hooks Are NOT For:
- Correctness-critical features (must be first-class stages)
- Required functionality (use `error_mode="raise"` in CI/production)

## Features

### 1. Deterministic Ordering
- Three-level sort: `(priority, registration_index, stable_tiebreak)`
- Config order preserved (no silent reordering)

### 2. Error Modes
- `log_and_continue` (default): Log error, continue
- `raise`: Fail fast (for CI/strict mode)
- `disable_hook`: Auto-disable after first error (documented path-dependency risk)

### 3. Hook Point Allowlist (CI Mode)
- Enforce that only documented hook points are used
- Prevents typos from silently disabling features
- Use in CI: `PipelineHooks.set_hook_point_allowlist(get_allowed_hook_points())`

### 4. Execution Trace
- Comprehensive trace with hook names, order, errors
- Durations logged but NOT stored in context (preserves determinism)
- Capped at 100 entries to prevent unbounded growth

### 5. Plugin Loading
- Explicit, controlled loading
- Config-driven: `load_plugins_from_config()`
- Preserves config order (no silent reordering)

## Usage

### Development
```python
# Default: log_and_continue (hooks don't break pipeline)
PipelineHooks.set_error_mode("log_and_continue")
```

### CI/Strict Runs
```python
# Fail fast on hook errors
PipelineHooks.set_error_mode("raise")

# Enforce allowlist
from TRAINING.common.pipeline_hook_points_list import get_allowed_hook_points
PipelineHooks.set_hook_point_allowlist(get_allowed_hook_points())
```

### Testing
```python
# Reset between tests
PipelineHooks.reset()
```

## Files

- `pipeline_hooks.py` - Core hook registry
- `plugin_loader.py` - Controlled plugin loading
- `pipeline_hook_points.md` - Hook point documentation
- `pipeline_hook_points_list.py` - Allowlist for CI
- `HOOK_POINT_ALLOWLIST.md` - Allowlist usage guide
- `DETERMINISM_GUIDE.md` - Determinism rules and patterns
- `DETERMINISM_FIXES.md` - Fixes applied
- `test_pipeline_hooks.py` - Integration tests

## Summary

The hook system is **production-ready** and maintains determinism for hook execution order and context outputs, while preserving config semantics. All determinism footguns have been fixed, and governance rules are documented.

**Next Steps:**
1. Add hook points to orchestrator (one-time setup)
2. Migrate existing optional features to hooks
3. Use `raise` mode + allowlist in CI
4. Document new hook points in `pipeline_hook_points.md` and `pipeline_hook_points_list.py`
