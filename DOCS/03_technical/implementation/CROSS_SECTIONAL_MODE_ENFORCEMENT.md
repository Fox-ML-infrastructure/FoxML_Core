# Cross-Sectional Mode Enforcement

## Overview

Cross-sectional ranking and single-symbol time series are **different problems**. If the system silently falls from N symbols to 1, it's no longer training what you think it's training, and safety + evaluation logic becomes misleading.

## The Problem

### Why Silent Degradation is Dangerous

1. **Target becomes ill-defined for ranking**
   - Cross-sectional ranking = "rank *symbols* at the same timestamp"
   - With 1 symbol, there is nothing to rank. The objective degenerates into "always rank NVDA #1"

2. **CV splitter assumptions break**
   - Cross-sectional CV groups by timestamp / "event time" and expects multiple symbols per group
   - With 1 symbol, many folds can become **degenerate** (empty validation groups, constant labels)

3. **Fake confidence**
   - Logs still say "CROSS_SECTIONAL," "min_cs=3," "ranking mode"
   - But the model is actually learning single-series patterns. Any "ranking performance" conclusion is junk

4. **Leakage controls are tuned for wrong geometry**
   - Purge/embargo logic in ranking mode is calibrated around cross-sectional sampling
   - In a single series, the same settings can be too strict (kills folds) or too lax (leaks)

5. **Downstream behavior changes**
   - Model trained on only NVDA can't generalize cross-sectionally
   - May overfit NVDA idiosyncrasies
   - Will rank nonsense when applied to AAPL/MSFT/etc.

## Solution: Hard-Stop Enforcement

### Implementation

**Location**: `TRAINING/utils/cross_sectional_data.py::prepare_cross_sectional_data_for_ranking()`

**Check**: After loading symbols, before building cross-sectional data

```python
if n_symbols_available < min_cs:
    raise ValueError(
        f"CROSS_SECTIONAL mode requires >= {min_cs} symbols, but only {n_symbols_available} loaded. "
        f"Loaded symbols: {list(mtf_data.keys())}. "
        f"This would silently degrade into single-symbol time series masquerading as cross-sectional ranking. "
        f"Use SYMBOL_SPECIFIC mode for single-symbol ranking, or ensure sufficient symbols are available."
    )
```

### View-Specific Behavior

1. **SYMBOL_SPECIFIC**: Uses `min_cs=1` (correct - single symbol doesn't need min_cs)
2. **CROSS_SECTIONAL**: Uses `min_cs=self.min_cs` (check will trigger)
3. **LOSO**: Uses `min_cs=self.min_cs` on training set (N-1 symbols)
   - Validation symbol loaded separately with `min_cs=1`
   - Check ensures training set has sufficient symbols for cross-sectional folds

### Symbol Load Report

**Always logged** (not conditional):

```
📦 Symbol load report:
   Requested: N symbols [list]
   Loaded: M symbols [list]
   Dropped: K symbols {symbol: reason}
```

This prevents confusion and makes regressions obvious.

## LOSO View Details

For LOSO (Leave-One-Symbol-Out):

- Training set: All symbols except validation symbol (N-1 symbols)
- Validation set: Single symbol (loaded separately)
- Check is applied to training set size (N-1), which is correct
- We need at least `min_cs` symbols loaded in the training set (global availability)
- Per-timestamp filtering ensures each timestamp has >= effective_min_cs symbols present
- The validation symbol is loaded separately with `min_cs=1` (see `evaluate_target_predictability`)

### Important Distinction

The hard-stop check enforces **"N symbols loaded overall"** (global availability), while per-timestamp filtering enforces **"effective cross-sectional width per timestamp"** (after intersection/missingness). 

- Hard-stop prevents: "only 1 symbol total" → would silently degrade
- Per-timestamp filtering prevents: "timestamps with < min_cs symbols" → would produce degenerate cross-sections

Both checks are necessary and complementary.

## AUTO Mode (Future Enhancement)

If AUTO mode is added, it must:

1. **Never silently downgrade**
   - Log WARN
   - Persist `resolved_data_mode` + `downgrade_reason`
   - Record `requested_mode=AUTO`, `resolved_mode=SINGLE_SYMBOL_TS`

2. **Telemetry must tag the run clearly**
   - Never compare "CROSS_SECTIONAL" runs to "AUTO→SINGLE_SYMBOL_TS" runs by mistake

3. **Output directory must include resolved mode**
   - `.../AUTO/RESOLVED_SINGLE_SYMBOL_TS/...` or
   - `.../SINGLE_SYMBOL_TS/...` with metadata saying `requested=AUTO`

**Recommendation**: Keep strict as default, add AUTO as explicit opt-in for:
- Fast experiments
- Notebooks
- "Just get me a model" workflows
- Smoke tests when symbol availability is flaky

## Error Message Format

Error messages include:

- Required vs actual symbol count
- List of loaded symbols
- Explanation of the problem
- Dropped symbols with reasons (if available)
- Missing symbols from requested list (if available)
- Guidance to use SYMBOL_SPECIFIC for single-symbol ranking

## Resolved Mode Tracking

The system tracks `resolved_data_mode`:

- `CROSS_SECTIONAL`: Full panel (n_symbols >= 10)
- `CROSS_SECTIONAL`: Small panel (n_symbols < 10)
- `SINGLE_SYMBOL_TS`: Should never occur for CROSS_SECTIONAL (due to check), but tracked for safety

This is logged and passed to telemetry for audit purposes.

