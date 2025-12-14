# Feature Selection and Config Fixes (2025-12-14)

## Summary

Critical bug fixes for feature selection pipeline, experiment config loading, target exclusion, and lookback enforcement. These fixes resolve cascading failures that were preventing feature selection from running and blocking target evaluation.

---

## Critical Bug Fixes

### 1. UnboundLocalError for `np` in Feature Selection (11 model families failing)

**Problem:** Redundant local import of `numpy as np` at line 1235 in `multi_model_feature_selection.py` was causing Python to treat `np` as a local variable throughout the function, causing `UnboundLocalError` when `np` was used earlier (lines 934, 948, 1050, 1141, 1244, 1568, 1649, 1745, 1779, 1847, 1953, 2100).

**Impact:** 11 out of 13 model families were failing during feature selection (ridge, elastic_net, catboost, mutual_information, univariate_selection, rfe, boruta, stability_selection, xgboost, random_forest).

**Fix:**
- Removed redundant local import `import numpy as np` at line 1235
- Use module-level import (already present at line 45)
- Added comment explaining why local imports should not be used

**Files Changed:**
- `TRAINING/ranking/multi_model_feature_selection.py`

**Result:** All 13 model families now run successfully during feature selection.

---

### 2. Missing `parse_duration_minutes` Import

**Problem:** `feature_selector.py` was trying to import `parse_duration_minutes` which doesn't exist in `duration_parser.py`.

**Impact:** ImportError during feature selection, causing pipeline to fail.

**Fix:**
- Use `parse_duration()` and call `.to_minutes()` instead
- This is the correct API from `duration_parser.py`

**Files Changed:**
- `TRAINING/ranking/feature_selector.py`

**Result:** Feature selection completes without ImportError.

---

### 3. Unpacking Error in Shared Harness (7 values vs 6)

**Problem:** `train_and_evaluate_models` returns 7 values, but `feature_selector.py` was unpacking only 6, causing `ValueError: too many values to unpack (expected 6)`.

**Impact:** Shared harness failed, causing fallback to per-symbol processing (slower, less efficient).

**Fix:**
- Added `perfect_correlation_models` to unpacking in both places in `feature_selector.py`
- Updated return type annotation in `shared_ranking_harness.py`

**Files Changed:**
- `TRAINING/ranking/feature_selector.py`
- `TRAINING/ranking/shared_ranking_harness.py`

**Result:** Shared harness works correctly for feature selection, preventing fallback to per-symbol processing.

---

## Target Ranking and Routing Fixes

### 4. Honest Routing Reason Strings

**Problem:** Routing decisions showed misleading `"no strong symbol-specific signal"` even when symbol-specific evaluation never produced any results (`n_symbols_evaluated == 0`).

**Impact:** Confusing diagnostics - made it look like evaluation ran but was weak, when actually no symbols were evaluable.

**Fix:**
- Check if `n_symbols_evaluated == 0` before generating reason string
- Changed reason from `"no strong symbol-specific signal"` to `"symbol_eval=0 symbols evaluable"` when no symbols were evaluable
- Makes it clear when evaluation never ran vs when it ran but failed

**Files Changed:**
- `TRAINING/ranking/target_routing.py`

**Result:** Routing decisions now show honest reasons, making diagnostics clearer.

---

### 5. Per-Symbol Skip Reason Tracking

**Problem:** No visibility into why symbol-specific evaluation was returning 0 symbols. Couldn't tell if it was data constraints, evaluation failures, or configuration issues.

**Impact:** Difficult to diagnose why symbol-specific evaluation wasn't working.

**Fix:**
- Track skip reasons during symbol-specific evaluation (degenerate, exception, etc.)
- Store skip reasons in `results_sym._skip_reasons` dict structure
- Pass skip reasons to routing decisions function
- Add `symbol_skip_reasons` field to routing decisions JSON
- Added logging summary: `"Symbol-specific skip reasons for {target_name}: {n}/{total} symbols skipped"`

**Files Changed:**
- `TRAINING/ranking/target_ranker.py`
- `TRAINING/ranking/target_routing.py`

**Result:** Routing decisions JSON now includes per-symbol skip reasons with status, leakage_flag, mean_score, making it easy to diagnose why symbol-specific evaluation returns 0 symbols.

---

## Experiment Configuration Fixes

### 6. `max_targets_to_evaluate` Not Loaded from Experiment Config

**Problem:** `max_targets_to_evaluate` and `top_n_targets` were being loaded from base config but not extracted from experiment config YAML. The experiment config loader only extracted `manual_targets` and `auto_targets`.

**Impact:** Setting `max_targets_to_evaluate: 100` in experiment config had no effect - always used base config value.

**Fix:**
- Extract `max_targets_to_evaluate` from experiment config `intelligent_training` section
- Extract `top_n_targets` from experiment config `intelligent_training` section
- Both values now override base config when specified in experiment config
- Added logging to show when experiment config values are used
- Applied fix to both config loading code paths

**Files Changed:**
- `TRAINING/orchestration/intelligent_trainer.py`

**Result:** Per-experiment control of target evaluation limits now works correctly.

---

### 7. Test Config Overriding Experiment Config

**Problem:** When "test" was detected in output directory name, test config from `pipeline.yaml` (with `max_targets_to_evaluate: 23`, `top_n_targets: 23`) was overriding experiment config values, even when experiment config explicitly set different values.

**Impact:** Experiment config values were ignored when running in test mode.

**Fix:**
- Experiment config now takes priority over test config
- Test config only applies if experiment config doesn't set the value
- Added logging to show when experiment config overrides test config
- Applied fix to both config loading code paths

**Files Changed:**
- `TRAINING/orchestration/intelligent_trainer.py`

**Result:** Experiment config values (e.g., `max_targets_to_evaluate: 100`) now correctly override test config, allowing per-experiment control even in test runs.

---

## New Features

### 8. Target Pattern Exclusion (`exclude_target_patterns`)

**Problem:** No way to exclude specific target types (e.g., `will_peak`, `will_valley`) from evaluation without modifying discovery code.

**Impact:** Had to manually filter targets or modify code to exclude unwanted target types.

**Fix:**
- Added `exclude_target_patterns` option to `intelligent_training` section in experiment config
- Filter discovered targets based on pattern matching (substring match)
- Log excluded targets count and patterns used
- Patterns are matched as substrings (e.g., `"will_peak"` matches `"y_will_peak_60m_0.8"`)

**Files Changed:**
- `TRAINING/orchestration/intelligent_trainer.py`
- `CONFIG/experiments/e2e_ranking_test.yaml` (example)
- `CONFIG/experiments/e2e_full_targets_test.yaml` (example)

**Usage:**
```yaml
intelligent_training:
  exclude_target_patterns:
    - "will_peak"
    - "will_valley"
```

**Result:** Per-experiment control over which target types are evaluated, useful for excluding specific target families without modifying discovery code.

---

## Lookback Enforcement Fixes

### 9. `hour_of_day` Unknown Lookback Violation

**Problem:** `hour_of_day` is a time-of-day feature (known at time t, no historical data needed) but wasn't recognized as a calendar feature with 0 lookback. Gatekeeper treated it as unknown lookback (inf) and blocked training in strict mode.

**Impact:** RuntimeError: `"🚨 UNKNOWN LOOKBACK VIOLATION (GATEKEEPER): 1 features have unknown lookback (inf). Sample: ['hour_of_day']"`

**Fix:**
- Added `hour_of_day` to `CALENDAR_FEATURES` set (0m lookback)
- Added `minute_of_hour` to `CALENDAR_FEATURES` set (0m lookback)
- Added patterns for `hour_of_day` and `minute_of_hour` to `CALENDAR_ZERO_PATTERNS`

**Files Changed:**
- `TRAINING/utils/leakage_budget.py`

**Result:** `hour_of_day` now recognized as calendar feature with 0m lookback, preventing gatekeeper from blocking training.

---

## Commercial Documentation Updates

### 10. Pricing Anchor for Commercial Signaling

**Problem:** "For pricing information, please contact the maintainer" without context was suppressing inbound inquiries (the "shy trap").

**Impact:** Potential customers didn't know if inquiry was appropriate, reducing commercial inquiries.

**Fix:**
- Added pricing anchor sentence: "Commercial licenses typically start in the low five figures annually, depending on team size and deployment scope."
- Added to all commercial licensing documentation:
  - `README.md`
  - `LEGAL/SUBSCRIPTIONS.md`
  - `COMMERCIAL_LICENSE.md`
  - `LEGAL/COMMERCIAL_USE.md`

**Files Changed:**
- `README.md`
- `LEGAL/SUBSCRIPTIONS.md`
- `COMMERCIAL_LICENSE.md`
- `LEGAL/COMMERCIAL_USE.md`

**Result:** Better commercial signaling - provides order-of-magnitude context without committing to exact numbers, making inquiry feel appropriate rather than awkward.

---

## Files Changed Summary

### Core Code Changes
- `TRAINING/ranking/multi_model_feature_selection.py` - Fixed `np` UnboundLocalError
- `TRAINING/ranking/feature_selector.py` - Fixed import error and unpacking error
- `TRAINING/ranking/shared_ranking_harness.py` - Fixed return type annotation
- `TRAINING/ranking/target_ranker.py` - Added skip reason tracking
- `TRAINING/ranking/target_routing.py` - Fixed routing reason strings, added skip reasons to JSON
- `TRAINING/orchestration/intelligent_trainer.py` - Fixed config loading, added target exclusion
- `TRAINING/utils/leakage_budget.py` - Added calendar features

### Configuration Changes
- `CONFIG/experiments/e2e_ranking_test.yaml` - Added exclude_target_patterns example
- `CONFIG/experiments/e2e_full_targets_test.yaml` - Added exclude_target_patterns

### Documentation Changes
- `README.md` - Added pricing anchor
- `LEGAL/SUBSCRIPTIONS.md` - Added pricing anchor
- `COMMERCIAL_LICENSE.md` - Added pricing anchor
- `LEGAL/COMMERCIAL_USE.md` - Added pricing anchor

---

## Testing Status

- ✅ All 13 model families now run during feature selection
- ✅ Feature selection completes without ImportError
- ✅ Shared harness works correctly (no fallback to per-symbol processing)
- ✅ Routing decisions show honest reasons
- ✅ Skip reasons tracked and included in JSON output
- ✅ Experiment config values override test config
- ✅ Target exclusion patterns work correctly
- ✅ `hour_of_day` no longer causes unknown lookback violations

---

## Migration Notes

### For Users

1. **Feature Selection:** No changes needed - fixes are automatic
2. **Experiment Config:** You can now use `exclude_target_patterns` to filter targets:
   ```yaml
   intelligent_training:
     exclude_target_patterns:
       - "will_peak"
       - "will_valley"
   ```
3. **Target Limits:** `max_targets_to_evaluate` and `top_n_targets` in experiment config now work correctly
4. **Calendar Features:** `hour_of_day` and `minute_of_hour` are now recognized as calendar features (0 lookback)

### For Developers

- Avoid local imports of modules already imported at module level (causes UnboundLocalError)
- When adding new return values to functions, update all call sites
- Experiment config values take priority over test config
- Calendar features should be added to `CALENDAR_FEATURES` set and `CALENDAR_ZERO_PATTERNS` list

---

## Related Documentation

- [Look-Ahead Bias Fixes](2025-12-14-lookahead-bias-fixes.md) - Related fixes from same day
- [Experiment Config Guide](../../01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md) - How to use experiment configs
- [Auto Target Ranking](../../01_tutorials/training/AUTO_TARGET_RANKING.md) - Target discovery and ranking
- [Feature Selection Tutorial](../../01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - Feature selection workflow

---

## Commit History

- `544709f` - fix(feature_selection): Fix UnboundLocalError for np and missing parse_duration_minutes
- `1f70166` - fix(feature_selection): Fix unpacking error in shared harness call
- `a7eff05` - fix(routing): Add honest skip reason tracking and diagnostics
- `53f01e3` - fix(config): Load max_targets_to_evaluate and top_n_targets from experiment config
- `7aebca1` - fix(config): Also load max_targets_to_evaluate in second config loading path
- `4542f4a` - feat(config): Add exclude_target_patterns to filter discovered targets
- `7cdc583` - fix(config): Add exclude_target_patterns to e2e_full_targets_test and improve logging
- `0739d03` - fix(config): Make experiment config override test config for target limits
- `8e2443c` - fix(lookback): Add hour_of_day and minute_of_hour to calendar features (0 lookback)
- `54135f1` - docs(commercial): Add pricing anchor to COMMERCIAL_USE.md
- `0429253` - docs(commercial): Add pricing anchor for better commercial signaling
