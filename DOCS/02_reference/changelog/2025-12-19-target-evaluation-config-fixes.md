# 2025-12-19: Target Evaluation Config Loading Fixes

## Problem

Two issues were identified with target evaluation configuration:

1. **Config Precedence Issue**: `max_targets_to_evaluate` from experiment config was not properly overriding test config values. When running with 'test' in the output directory name, test config values were taking precedence over experiment config values, even though experiment config should have higher priority.

2. **Missing Whitelist Support**: When `auto_targets: true`, there was no way to specify a specific list of targets to evaluate. The system only supported:
   - `manual_targets` (only works when `auto_targets: false`)
   - `exclude_target_patterns` (blacklist, works with `auto_targets: true`)
   - No whitelist option for "evaluate only these specific targets"

## Root Cause

### Issue 1: Config Precedence
The test config override logic was checking if experiment config had set values, but the check was happening before the experiment config values were fully loaded and tracked. The `config_sources` tracking dictionary was not being used to determine precedence correctly.

### Issue 2: Missing Whitelist
The `rank_targets_auto()` method only supported blacklisting via `exclude_target_patterns`. There was no mechanism to specify a whitelist of targets to evaluate when using auto-discovery.

## Solution

### Fix 1: Config Precedence
- Added `config_sources` dictionary to track where each config value comes from (`base_config`, `experiment_config`, `test_config`)
- Updated test config override logic to check `config_sources` to determine if experiment config already set the value
- Added debug logging to show config precedence chain and which values are being used
- Added config trace for `intelligent_training` section overrides in the config trace output

### Fix 2: Whitelist Support
- Added new config field `intelligent_training.targets_to_evaluate` that works with `auto_targets: true`
- Acts as a whitelist: if specified, only evaluate these targets (after pattern exclusions)
- If empty/not specified, evaluate all discovered targets (backward compatible)
- Implemented in both main config loading path and fallback path
- Added `targets_to_evaluate` parameter to `rank_targets_auto()` and `train_with_intelligence()` methods
- Filtering happens after pattern exclusions and before `max_targets_to_evaluate` limit
- Included whitelist in cache key to properly invalidate cache when whitelist changes

## Files Changed

- `TRAINING/orchestration/intelligent_trainer.py`:
  - Added `config_sources` tracking dictionary
  - Fixed test config override logic to respect experiment config precedence
  - Added `targets_to_evaluate` parameter to `rank_targets_auto()` and `train_with_intelligence()`
  - Added whitelist filtering logic in `rank_targets_auto()` (after pattern exclusions)
  - Added debug logging for config loading and precedence
  - Added config trace for `intelligent_training` section overrides
  - Updated cache key to include whitelist

## Impact

### Config Precedence
- Experiment config values now correctly override test config values
- Debug logging shows exactly where each config value comes from
- Config trace output now shows experiment config overrides for `intelligent_training` section

### Whitelist Support
- Users can now specify `targets_to_evaluate: [target1, target2, ...]` in experiment config
- Works seamlessly with `auto_targets: true`
- Provides fine-grained control over which targets to evaluate during ranking
- Maintains full backward compatibility (if not specified, all targets are evaluated)

## Example Usage

```yaml
intelligent_training:
  auto_targets: true
  max_targets_to_evaluate: 5
  targets_to_evaluate: [fwd_ret_60m, fwd_ret_120m, fwd_ret_30m]  # NEW: Whitelist
  exclude_target_patterns:
    - "will_peak"
    - "will_valley"
```

This will:
1. Auto-discover all targets from data
2. Exclude targets matching "will_peak" or "will_valley" patterns
3. Filter to only evaluate `fwd_ret_60m`, `fwd_ret_120m`, and `fwd_ret_30m` (whitelist)
4. Limit evaluation to top 5 targets (after whitelist filtering)

## Testing

- Verified `max_targets_to_evaluate: 5` from experiment config is respected when test config is also present
- Verified `targets_to_evaluate` whitelist works with `auto_targets: true`
- Verified precedence: experiment config > test config > base config
- Verified backward compatibility (no whitelist = all targets evaluated)

