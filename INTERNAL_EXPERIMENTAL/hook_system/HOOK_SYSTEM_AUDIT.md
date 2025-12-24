# Hook System Determinism Audit

## Verdict: ✅ **Production-Viable** (with minor tweaks)

The implementation matches the design and is **directionally correct**. The architecture has the right pillars. A few minimal tweaks will make it bulletproof.

## What's Solid ✅

### 1. Dedupe Key (CORRECT)
**Line 81:** `callback_id = f"{callback.__module__}.{callback.__qualname__}"`
- ✅ Uses **fully-qualified name** (module + qualname)
- ✅ Prevents collisions across modules
- ✅ Correct implementation

### 2. Deterministic Ordering (CORRECT)
**Line 112:** `sort(key=lambda x: (x['priority'], x['registration_index'], x['stable_tiebreak']))`
- ✅ Three-level sort: priority → registration_index → stable_tiebreak
- ✅ `stable_tiebreak = (module_name, qualname)` provides final tiebreak
- ✅ Same code + same config = same order

### 3. Plugin Load Order (CORRECT)
**Line 59 (plugin_loader.py):** `sorted_plugins = sorted(plugins)`
- ✅ Sorts plugins alphabetically for deterministic load order
- ✅ Validates `isinstance(plugins, list)` before sorting
- ✅ Preserves order through `load_plugins()`

### 4. Error Modes (CORRECT)
**Lines 219-232:** Three error modes implemented
- ✅ `log_and_continue` (default)
- ✅ `raise` (for CI/strict mode)
- ✅ `disable_hook` (auto-quarantine)

### 5. Execution Trace (CORRECT)
**Lines 154-250:** Full execution trace
- ✅ Logs hook execution with timing
- ✅ Stores in `context['hook_traces']`
- ✅ Includes errors and durations

### 6. Registry Reset (CORRECT)
**Line 316:** `reset()` method
- ✅ Clears all state for testing
- ✅ Prevents cross-test leakage

## Remaining Risks (Minimal)

### 1. Return Value Validation (MINOR)
**Current:** Line 192 checks `if hook_result is not None`
**Risk:** Doesn't validate type. If hook returns wrong type (e.g., string instead of dict), it could cause downstream errors.

**Fix:** Add lightweight type check or document contract clearly.

### 2. Unknown Hook Point Policy (MINOR)
**Current:** No validation for typo'd hook points
**Risk:** Silent failures if someone registers to `'after_feature_selection'` (typo) instead of `'after_feature_selection'`

**Fix:** Add optional validation mode (warn on unknown hook points).

### 3. Trace Size Control (MINOR)
**Current:** Stores full trace objects in context
**Risk:** If many hooks run, `context['hook_traces']` could grow large

**Fix:** Cap trace length or store lightweight summaries.

### 4. Conditional Hook Registration (POLICY)
**Current:** Code doesn't prevent conditional registration
**Risk:** Hooks registering based on env vars/filesystem can break determinism

**Fix:** Document policy (hooks should register unconditionally on import).

## Minimal Tweaks (Non-Breaking)

### 1. Return Value Validation
Add lightweight check that return value is compatible with context type.

### 2. Unknown Hook Point Warning
Add optional validation mode to warn on unknown hook points (configurable).

### 3. Trace Size Control
Cap trace length or provide option to store lightweight traces.

### 4. Plugin Load Order Validation
Ensure config returns ordered list (already handled, but could add explicit check).

## Policy Decisions (Document)

1. **Hooks must be optional** - Required features shouldn't be hooks
2. **Use `raise` mode in CI** - Fail fast on hook errors in CI
3. **Hooks register unconditionally** - No env-var or filesystem-based conditional registration
4. **Seeded RNG only** - Hooks using randomness must use `context.get('run_seed')`

## Summary

**The implementation is solid.** The determinism guarantees are correct:
- ✅ Dedupe uses fully-qualified names
- ✅ Ordering is deterministic (priority + registration_index + stable_tiebreak)
- ✅ Plugin loading is sorted and explicit
- ✅ Error modes are implemented
- ✅ Execution trace is comprehensive

**Minor tweaks recommended:**
- Return value validation (lightweight)
- Unknown hook point warning (optional)
- Trace size control (optional)

**These are polish, not blockers.** The system is production-ready as-is.
