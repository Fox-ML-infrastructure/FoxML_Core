# Hook System Determinism Guide

## Determinism Guarantee

**Same inputs + same config + same code = same hook execution order + same outputs**

The hook system is designed to be fully deterministic. This document explains how determinism is guaranteed and what you must do to maintain it.

## How Determinism is Guaranteed

### 1. Deterministic Hook Ordering

Hooks are sorted by a **three-level sort key**:

1. **Priority** (lower = first)
2. **Registration Index** (order plugins were loaded)
3. **Stable Tiebreak** (`(module, qualname)` tuple)

This ensures:
- Same priority hooks execute in consistent order
- Order is reproducible across machines
- No dependency on import timing or filesystem order

### 2. Explicit Plugin Loading

Plugins are loaded from an **explicit, sorted list**:

```python
# Config-driven (automatically sorted)
load_plugins_from_config()  # Sorts plugins alphabetically

# Explicit (you should sort)
load_plugins(sorted(['TRAINING.common.feature_a', 'TRAINING.common.feature_b']))
```

**Never:**
- ❌ Use filesystem globs (`glob.glob('*.py')`)
- ❌ Auto-discover plugins via `pkgutil.iter_modules()`
- ❌ Rely on import side-effects without explicit loading

### 3. No Unordered Containers

The registry uses:
- ✅ `Dict[str, List[...]]` for hook storage (keys are sorted when iterating)
- ✅ `List` for hook execution (already sorted)
- ❌ Never `set` or unsorted `dict.items()` for hook iteration

### 4. Pure-ish Hook Functions

Hooks should follow functional patterns:

**✅ Good (deterministic):**
```python
@register_hook('after_feature_selection', priority=50)
def my_hook(context):
    # Read from context
    features = context.get('selected_features', [])
    
    # Use seeded RNG if needed
    import random
    rng = random.Random(context.get('run_seed', 42))
    
    # Return modified context
    context['modified_features'] = process_features(features, rng)
    return context
```

**❌ Bad (non-deterministic):**
```python
@register_hook('after_feature_selection', priority=50)
def bad_hook(context):
    # Time-based (non-deterministic)
    context['timestamp'] = time.time()
    
    # Random without seed (non-deterministic)
    context['random_value'] = random.random()
    
    # UUID (non-deterministic)
    context['id'] = str(uuid.uuid4())
    
    # Filesystem glob (order can vary)
    files = glob.glob('*.csv')
    
    return context
```

## Determinism Checklist

When writing hooks, ensure:

- [ ] Hook uses only `context` inputs (no global state, no filesystem reads)
- [ ] If using randomness: uses `context.get('run_seed')` for seeded RNG
- [ ] No `time.time()`, `uuid.uuid4()`, or other time-based values in outputs
- [ ] No filesystem globs or unordered iteration
- [ ] Returns modified context (functional pattern)
- [ ] No side effects that affect other hooks

## Testing Determinism

To verify determinism:

1. **Same config, same code, multiple runs:**
   ```python
   # Run 1
   context1 = PipelineHooks.execute('after_feature_selection', context)
   
   # Run 2 (same context, same config)
   context2 = PipelineHooks.execute('after_feature_selection', context)
   
   # Should be identical
   assert context1 == context2
   ```

2. **Check execution order:**
   ```python
   trace = context['hook_traces'][0]
   hooks_executed = [h['hook'] for h in trace['hooks_executed']]
   # Order should be consistent across runs
   ```

3. **Verify plugin load order:**
   ```python
   stats = PipelineHooks.get_execution_stats()
   # Plugin load order should be logged and consistent
   ```

## Common Failure Modes

### ❌ Auto-discovery via filesystem
```python
# BAD: Filesystem order can vary
plugins = glob.glob('TRAINING/common/*_hooks.py')
load_plugins(plugins)  # Order depends on filesystem
```

**Fix:** Use explicit, sorted list:
```python
# GOOD: Explicit, sorted
load_plugins(sorted(['TRAINING.common.hook_a', 'TRAINING.common.hook_b']))
```

### ❌ Import-time registration without explicit loading
```python
# BAD: Import order can vary
# In __init__.py
from TRAINING.common import feature_a  # Registers hooks
from TRAINING.common import feature_b  # Registers hooks
# Order depends on import order
```

**Fix:** Explicit plugin loading:
```python
# GOOD: Explicit, sorted
load_plugins(sorted(['TRAINING.common.feature_a', 'TRAINING.common.feature_b']))
```

### ❌ Same priority, no tiebreak
```python
# BAD: Order depends on registration timing
@register_hook('point', priority=100)  # Which runs first?
def hook_a(context): ...

@register_hook('point', priority=100)  # Unpredictable
def hook_b(context): ...
```

**Fix:** Use different priorities or rely on stable tiebreak (already implemented):
```python
# GOOD: Different priorities
@register_hook('point', priority=50)  # Runs first
def hook_a(context): ...

@register_hook('point', priority=100)  # Runs second
def hook_b(context): ...
```

### ❌ Randomness without seed
```python
# BAD: Non-deterministic
def hook(context):
    value = random.random()  # Different each run
    return context
```

**Fix:** Use seeded RNG:
```python
# GOOD: Deterministic
def hook(context):
    rng = random.Random(context.get('run_seed', 42))
    value = rng.random()  # Same each run with same seed
    return context
```

## Summary

**Determinism is guaranteed if you:**
1. ✅ Load plugins from explicit, sorted list
2. ✅ Use seeded RNG for randomness
3. ✅ Avoid time-based/UUID outputs
4. ✅ Follow functional patterns (read context, return modified context)
5. ✅ No filesystem globs or unordered iteration

**The hook system handles:**
- Deterministic sorting (priority + registration_index + stable_tiebreak)
- Explicit plugin loading (sorted order)
- Stable execution order (no unordered containers)

**You must handle:**
- Seeded randomness (use `context.get('run_seed')`)
- No time-based outputs
- Functional hook patterns
