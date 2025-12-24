# Critical Fixes: Preprocessing Leak & Horizon Overlap Logic

**Date:** 2025-11-23 00:56:38

## Issue 1: Preprocessing Leak (FIXED)

### Problem
The code was applying `StandardScaler` and `SimpleImputer` to the **entire dataset** before cross-validation, causing "distribution leak":
- Model sees mean/std of future data during training
- Artificially inflates R² scores (e.g., 0.02 → 0.05)

### Solution
Replaced manual preprocessing with `sklearn.pipeline.Pipeline`:
- **Neural Network**: Now uses Pipeline with imputer → scaler → model
- **Lasso**: Now uses Pipeline with imputer → model
- All preprocessing happens **within each CV fold** (no leakage)

### Files Changed
- `SCRIPTS/rank_target_predictability.py`:
 - Neural Network block (lines ~807-879)
 - Lasso block (lines ~1020-1056)

### Impact
- More realistic R² scores
- Proper temporal causality in CV
- No distribution leakage

---

## Issue 2: Overly Aggressive Horizon Filtering (FIXED)

### Problem
The `target/4` rule was excluding **all** features with overlapping horizons, including:
- `rsi_15m` for a 60m target (legitimate predictor)
- `volatility_30m` for a 60m target (legitimate predictor)
- Standard technical indicators computed on past data

**This was too aggressive** - these features represent **causality**, not leakage.

### Solution
Relaxed the rule to only exclude **forward-looking features**:
- Only exclude if feature is `fwd_ret_*`, `y_*`, `p_*`, `barrier_*`, etc.
- **Keep** standard technical indicators (RSI, MA, volatility) regardless of horizon
- A 15m RSI computed from past data **should** be able to predict a 60m move

### Files Changed
- `SCRIPTS/utils/leakage_filtering.py`:
 - `_filter_for_barrier_target()` function (lines ~399-405)

### Logic
```python
# OLD (too aggressive):
if col_horizon >= target_horizon / 4:
    exclude()  # Excludes everything

# NEW (smart):
if col_horizon >= target_horizon / 4:
    if is_forward_looking:  # Only exclude fwd_ret_*, y_*, etc.
        exclude()
    # Keep standard indicators (RSI, MA, volatility)
```

### Impact
- More features available for prediction
- Better feature set (includes legitimate technical indicators)
- Still excludes actual leaks (forward-looking features)

---

## What's Still Safe

The following models don't need Pipeline fixes because they:
- Don't use cross_val_score with preprocessing (mutual_information, univariate_selection)
- Handle NaN natively (LightGBM, XGBoost, RandomForest, CatBoost)
- Use preprocessing only for feature selection, not final scoring (RFE, Boruta)

---

## Testing

After these fixes:
1. **Restart Python process** (clears config cache)
2. Re-run ranking script
3. Expect:
 - More realistic R² scores (0.10-0.20 for good targets)
 - More features available (~300+ instead of ~229)
 - No preprocessing leakage warnings

---

## Notes

- The horizon overlap rule now distinguishes between:
 - **Leakage**: Forward-looking features (fwd_ret_*, y_*, p_*, barrier_*)
 - **Causality**: Standard technical indicators (RSI, MA, volatility, momentum)

- This aligns with the user's feedback: "A 15-minute RSI *should* be able to predict a 60-minute move. That is causality, not leakage."
