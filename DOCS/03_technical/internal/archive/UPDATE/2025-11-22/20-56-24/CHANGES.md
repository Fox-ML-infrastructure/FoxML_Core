# Fix: NoneType .items() Error in Target Ranking Script

**Date:** 2025-11-22 20:56:24
**Issue:** `'NoneType' object has no attribute 'items'` error causing target evaluation failures

## Problem

The script was failing with `AttributeError: 'NoneType' object has no attribute 'items'` when evaluating targets. The traceback revealed the error was occurring in `SCRIPTS/utils/task_types.py` at line 242 in `_get_model_constructor()`.

### Root Cause

The issue was that `model_spec.get('config', {})` can return `None` if the key `'config'` exists in the YAML but has a `None` value. Python's `.get()` method only returns the default value when the key doesn't exist, not when the key exists but the value is `None`.

When `_get_model_constructor()` tried to call `config.items()`, it failed because `config` was `None`.

## Changes Made

### 1. Fixed `create_model_configs_from_yaml()` in `task_types.py`

**Location:** `SCRIPTS/utils/task_types.py:204-208`

Added defensive check before calling `_get_model_constructor()`:
```python
# Defensive check: ensure config is not None
model_config = model_spec.get('config')
if model_config is None or not isinstance(model_config, dict):
    model_config = {}
constructor = _get_model_constructor(model_name, task_type, model_config)
```

**Location:** `SCRIPTS/utils/task_types.py:215-217`

Fixed the same issue when creating `ModelConfig`:
```python
# Defensive check: ensure default_params is not None
default_params = model_spec.get('config')
if default_params is None or not isinstance(default_params, dict):
    default_params = {}
```

### 2. Fixed `_get_model_constructor()` in `task_types.py`

**Location:** `SCRIPTS/utils/task_types.py:248-251`

Added defensive check at the start of the function:
```python
# Defensive check: ensure config is a dict
if config is None or not isinstance(config, dict):
    logger.warning(f"Config for {model_name} is None or not a dict (got {type(config)}), using empty config")
    config = {}
```

### 3. Enhanced Error Logging in `rank_target_predictability.py`

**Location:** `SCRIPTS/rank_target_predictability.py:1464-1480`

Added detailed error logging to help diagnose future issues:
- Full traceback logging for `.items()` errors
- Logging of result object types and values
- Better error messages to identify the exact failure point

### 4. Additional Defensive Checks

Added defensive checks throughout the codebase:
- `get_model_config()` - validates config structure before returning
- `create_model_configs_from_yaml()` - validates `model_families` dict structure
- Model family loading loops - filters out None/non-dict configs
- Config usage in model training - validates configs are dicts before calling `.items()`

## Files Modified

1. `SCRIPTS/utils/task_types.py`
 - Added defensive checks in `create_model_configs_from_yaml()`
 - Added defensive check in `_get_model_constructor()`
 - Added logger import

2. `SCRIPTS/rank_target_predictability.py`
 - Enhanced error logging with full tracebacks
 - Added defensive checks in `get_model_config()`
 - Added defensive checks in model family loading
 - Added defensive checks in config cleaning operations

## Testing

The fix was verified by:
- Compiling both files successfully
- Testing with a config that has `None` values (handled gracefully)
- The script now handles malformed YAML configs without crashing

## Impact

- **Before:** Script would crash with `AttributeError` when encountering `None` config values
- **After:** Script handles `None` config values gracefully, logging warnings and using empty dicts as fallback

## Related Issues

This fix addresses the persistent `'NoneType' object has no attribute 'items'` error that was preventing target evaluation from completing successfully.

