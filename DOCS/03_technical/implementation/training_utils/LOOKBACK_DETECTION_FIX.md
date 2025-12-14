# Lookback Detection Precedence Fix

## Problem Identified

Features like `intraday_seasonality_15m` and `intraday_seasonality_30m` were being incorrectly tagged with 1440m (24h) lookback because the keyword heuristic (`.*day.*`) was being checked **before** explicit time suffixes (`_15m`, `_30m`).

This caused false positives where features with explicit short lookbacks were being dropped as if they required 24 hours of history.

## Root Cause

The lookback detection logic had incorrect precedence:

**Before (WRONG):**
1. Registry metadata ✅
2. Keyword heuristics (`.*day.*`) ❌ **Too early**
3. Explicit suffixes (`_15m`, `_30m`, `_1d`, etc.) ❌ **Too late**

**After (CORRECT):**
1. Registry metadata ✅
2. **Explicit time suffixes** (`_15m`, `_30m`, `_1d`, `_24h`, etc.) ✅ **First priority**
3. Keyword heuristics (`.*day.*`) ✅ **Fallback only**

## Fix Applied

### Files Modified

1. **`TRAINING/utils/resolved_config.py`** - `compute_feature_lookback_max()`
   - Reordered pattern matching to check explicit suffixes first
   - Keyword heuristics only used as fallback when no explicit suffix found

2. **`TRAINING/ranking/predictability/model_evaluation.py`** - `_enforce_final_safety_gate()`
   - Updated to use same precedence as `compute_feature_lookback_max()`
   - Checks for explicit suffixes before applying keyword heuristics

3. **`TRAINING/utils/target_conditional_exclusions.py`** - `compute_feature_lookback_minutes()`
   - Reordered to match the same precedence

### Precedence Order (Final)

1. **Registry metadata** (most reliable)
   - Explicit `lag_bars` from feature registry

2. **Explicit time suffixes** (highly reliable)
   - `_(\d+)m$` - Minute suffixes (e.g., `_15m`, `_30m`, `_1440m`)
   - `_(\d+)h` - Hour suffixes (e.g., `_12h`, `_24h`)
   - `_(\d+)d` - Day suffixes (e.g., `_1d`, `_3d`)

3. **Explicit daily patterns** (moderate reliability)
   - `_1d$`, `_24h$`, `^daily_`, `_daily$`, `_1440m`
   - `rolling.*daily`, `daily.*high`, `daily.*low`
   - `volatility.*day`, `vol.*day`, `volume.*day`

4. **Aggressive keyword fallback** (low reliability, last resort)
   - `.*day.*` - Only if no explicit suffix found

5. **Bar-based patterns** (for technical indicators)
   - `sma_200`, `rsi_14`, etc.

## Test Results

### Before Fix
```
❌ intraday_seasonality_15m -> 1440m (incorrectly tagged as "day" keyword)
❌ intraday_seasonality_30m -> 1440m (incorrectly tagged as "day" keyword)
```

### After Fix
```
✅ intraday_seasonality_15m -> 15.0m (minute_suffix) ✅ CORRECT
✅ intraday_seasonality_30m -> 30.0m (minute_suffix) ✅ CORRECT
✅ day_of_week -> 1440.0m (day_keyword) ✅ CORRECT
✅ mom_1d -> 1440.0m (day_suffix) ✅ CORRECT
✅ volatility_12h -> 720.0m (hour_suffix) ✅ CORRECT
```

## Impact

- **False positives eliminated**: Features with explicit short lookbacks (`_15m`, `_30m`) are no longer incorrectly tagged as 1440m
- **Consistency**: All three lookback detection functions now use the same precedence
- **Accuracy**: Explicit suffixes take precedence over keyword heuristics, preventing misclassification

## Additional Fix: Duplicate Warning Prevention

Added caching to prevent duplicate "audit violation prevention" warnings when `create_resolved_config()` is called multiple times with the same parameters.

## Verification

The fix ensures that:
1. ✅ Explicit time suffixes (`_15m`, `_30m`) are detected correctly
2. ✅ Keyword heuristics only apply when no explicit suffix exists
3. ✅ Features like `intraday_seasonality_15m` are no longer incorrectly dropped
4. ✅ All lookback detection functions use consistent precedence
