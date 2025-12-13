# Silent Failures Audit

## Summary

Audit of potential silent failures from recent refactors, especially around config centralization.

## Potential Issues Found

### 1. Config Loading Returns Empty Dict on Failure ⚠️

**Location:** `CONFIG/config_loader.py`

**Issue:**
- `load_model_config()` returns `{}` if file not found or fails to load
- `load_defaults_config()` returns `{}` if file not found or fails to load
- `get_cfg()` returns `default` if config not found

**Impact:**
- Code continues with empty config instead of failing
- Trainers have `setdefault()` fallbacks, so this might be okay
- But could mask configuration errors

**Example:**
```python
config = load_model_config("lightgbm")  # Returns {} if file missing
# Code continues, uses setdefault() fallbacks
```

**Status:** ⚠️ **Potentially problematic** - Should at least log warnings (which it does)

### 2. Defaults Injection Fails Silently ⚠️

**Location:** `CONFIG/config_loader.py::inject_defaults()`

**Issue:**
- If `load_defaults_config()` returns `{}`, `inject_defaults()` just returns original config
- No error raised, no warning (only debug log)
- Code continues without defaults

**Code:**
```python
defaults = load_defaults_config()
if not defaults:
    return config  # Silent - no defaults injected
```

**Impact:**
- If `defaults.yaml` is missing or broken, defaults aren't injected
- But explicit values in configs still work
- Trainers have fallbacks, so might be okay

**Status:** ⚠️ **Low risk** - Has warning in `load_defaults_config()`, but injection is silent

### 3. Random State Fallback to 42 ⚠️

**Location:** `CONFIG/config_loader.py::inject_defaults()`

**Issue:**
- If pipeline_config.yaml can't be loaded, falls back to `random_state = 42`
- Exception is caught silently
- No warning logged

**Code:**
```python
try:
    # Load pipeline_config.yaml
    random_state = pipeline_config.get('pipeline', {}).get('determinism', {}).get('base_seed', 42)
except Exception:
    random_state = 42  # FALLBACK_DEFAULT_OK - silent!
```

**Impact:**
- Reproducibility might be affected if determinism config is broken
- But fallback is reasonable (42 is standard)

**Status:** ⚠️ **Low risk** - Should log warning but fallback is acceptable

### 4. Model Family Detection Might Miss Models ⚠️

**Location:** `CONFIG/config_loader.py::inject_defaults()`

**Issue:**
- Neural network detection uses string matching:
  ```python
  elif ('neural' in model_lower or 'mlp' in model_lower or 'lstm' in model_lower or 
        'cnn' in model_lower or 'transformer' in model_lower or 'multi_task' in model_lower or
        'vae' in model_lower or 'gan' in model_lower or 'meta_learning' in model_lower or
        'reward_based' in model_lower):
  ```
- If a new neural network model doesn't match these patterns, it won't get neural network defaults
- No warning logged

**Impact:**
- New neural network models might not get `dropout`, `activation`, `patience` defaults
- But explicit values in their configs still work

**Status:** ⚠️ **Low risk** - Explicit configs still work, but defaults won't apply

### 5. Config File Not Found Returns Empty Dict ⚠️

**Location:** `CONFIG/config_loader.py::load_model_config()`

**Issue:**
```python
if not config_file.exists():
    logger.warning(f"Config file not found: {config_file}, using empty config")
    return {}
```

**Impact:**
- If a model config file is missing, returns `{}`
- Code continues, uses trainer fallbacks
- Might mask missing config files

**Status:** ⚠️ **Low risk** - Warning is logged, fallbacks exist

## Recommendations

### High Priority
1. **Add warning when defaults injection fails** - Log when `defaults.yaml` is missing/broken
2. **Add warning when random_state fallback is used** - Log when determinism config can't be loaded

### Medium Priority
3. **Verify model family detection** - Check if all neural network models are detected correctly
4. **Add validation** - Warn if `load_model_config()` returns empty dict for known models

### Low Priority
5. **Consider raising errors** - For critical configs, consider raising instead of returning empty dict
6. **Add config validation** - Validate that required keys exist after loading

## Current Safety Mechanisms

✅ **Trainer fallbacks** - All trainers use `setdefault()` for critical parameters  
✅ **Warnings logged** - Most failures log warnings (though some are debug level)  
✅ **Explicit configs work** - Even if defaults fail, explicit values in configs still work  

## Conclusion

**Most issues are low-risk** because:
- Trainers have hardcoded fallbacks
- Warnings are logged (though some at debug level)
- Explicit config values still work

**Main concern:** Silent defaults injection failure - if `defaults.yaml` is broken, defaults won't be injected but code continues. Should add warning.
