# Hook Point Allowlist (CI Mode)

## Purpose

Prevent typos and unauthorized hook points from silently disabling features in CI/production.

## Usage

### Enable Allowlist (CI Mode)

```python
from TRAINING.common.pipeline_hooks import PipelineHooks

# Set allowlist from documented hook points
PipelineHooks.set_hook_point_allowlist([
    'before_target_ranking',
    'after_target_ranking',
    'before_feature_selection',
    'after_feature_selection',
    'before_training',
    'after_training',
    # ... add all documented hook points
])
```

### Disable Allowlist (Development)

```python
# Disable allowlist (default)
PipelineHooks.set_hook_point_allowlist(None)
```

## Behavior

- **With allowlist enabled:** Executing an unknown hook point raises `ValueError`
- **With allowlist disabled:** Hook points execute normally (no validation)

## Recommended Hook Points

See `pipeline_hook_points.md` for complete list. Common ones:

- `before_target_ranking`
- `after_target_ranking`
- `before_feature_selection`
- `after_feature_selection`
- `before_training`
- `after_training`

## CI Integration

Add to CI setup:

```python
# In CI setup
from TRAINING.common.pipeline_hooks import PipelineHooks
from TRAINING.common.pipeline_hook_points_list import get_allowed_hook_points

# Enforce allowlist in CI
PipelineHooks.set_hook_point_allowlist(get_allowed_hook_points())
```

This ensures only documented, intentional hook points are used.

## Updating the Allowlist

When adding new hook points:

1. Document in `pipeline_hook_points.md`
2. Add to `DOCUMENTED_HOOK_POINTS` in `pipeline_hook_points_list.py`
3. Update CI to use new list

This keeps the allowlist in sync with documentation.
