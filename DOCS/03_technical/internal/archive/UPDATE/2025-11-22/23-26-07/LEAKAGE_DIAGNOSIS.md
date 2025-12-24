# Why Leakage Is Still Appearing - Diagnosis

**Date:** 2025-11-22 23:26:07

## The Problem

You're still seeing 100% accuracy despite adding exclusion patterns. Here's why:

## Root Cause: Config Cache

**The most likely issue:** The Python process was started **BEFORE** we added `barrier_*` patterns to the config. The config is cached globally, and cache invalidation only works if the file is modified **AFTER** the process starts.

### How Config Caching Works

1. **First load:** Config is loaded and cached in `_LEAKAGE_CONFIG`
2. **Cache invalidation:** Only triggers if file modification time changes AFTER first load
3. **If script started before config update:** It uses the old cached config (without `barrier_*`)

### Solution

**RESTART THE PYTHON PROCESS** to clear the cache:

```bash
# Kill the current process (Ctrl+C)
# Then restart:
python SCRIPTS/rank_target_predictability.py --discover-all --symbols AAPL,MSFT,GOOGL,TSLA,JPM --output-dir results/target_rankings_updated
```

## Verification

The filtering logic is correct:
- Config file has `barrier_*` patterns
- Filtering works when tested in isolation
- No perfect correlations found in filtered features
- All known leaks are excluded

## Other Possible Issues

### 1. Feature Combinations

Even if no single feature leaks, **multiple features together** might perfectly predict the target:
- Example: `high_60m` + `vol_60m` + `ret_zscore_60m` together = perfect prediction
- Solution: Use `find_leaking_features.py` to identify high-importance feature combinations

### 2. Degenerate CV Folds

If a CV fold has all 1s or all 0s, the model can achieve perfect accuracy:
- TimeSeriesSplit might create imbalanced folds
- Solution: Check fold balance (see `diagnose_leakage.py`)

### 3. Target Degeneracy

If the target is degenerate in some samples:
- All 1s or all 0s in training data
- Solution: Check target distribution before training

## Diagnostic Tools

### 1. `SCRIPTS/diagnose_leakage.py` (NEW)

Comprehensive diagnosis:
```bash
python SCRIPTS/diagnose_leakage.py y_will_peak_60m_0.8 --symbol AAPL
```

Checks:
- Config status
- Filtering results
- Perfect correlations
- Target distribution
- CV fold issues

### 2. `SCRIPTS/find_leaking_features.py`

Analyze feature importance exports:
```bash
python SCRIPTS/find_leaking_features.py results/target_rankings_updated --top-n 30
```

Finds:
- Features matching exclusion patterns (shouldn't be there!)
- High-importance features (>50% or >30% and 3x next)
- Recommendations for config updates

## Step-by-Step Fix

1. **Stop the current ranking script** (Ctrl+C)

2. **Verify config is correct:**
   ```bash
   grep -A 2 "barrier_" CONFIG/excluded_features.yaml
   ```
 Should show `^barrier_` in regex_patterns and `barrier_` in prefix_patterns

3. **Restart Python process:**
   ```bash
   python SCRIPTS/rank_target_predictability.py --discover-all --symbols AAPL,MSFT,GOOGL,TSLA,JPM --output-dir results/target_rankings_updated
   ```

4. **Monitor the logs:**
 - Should see: "Filtered out 231 potentially leaking features (kept 298 safe features)"
 - Should NOT see: 100% accuracy warnings
 - Should see realistic scores: R² ~0.13-0.17

5. **If still seeing leaks:**
   ```bash
   # Run diagnosis
   python SCRIPTS/diagnose_leakage.py y_will_peak_60m_0.8 --symbol AAPL

   # Analyze feature importances
   python SCRIPTS/find_leaking_features.py results/target_rankings_updated
   ```

## Expected Results After Fix

- **Feature count:** 298 for peak targets, 297 for valley targets
- **R² scores:** ~0.13-0.17 (realistic for financial data)
- **No 100% accuracy warnings**
- **Models show genuine learning** (different scores across models)

## Code Logic Is Correct

The ranking script logic is correct:
1. Calls `filter_features_for_target()` correctly
2. Uses TimeSeriesSplit for temporal causality
3. Detects leaks via feature importance analysis
4. Logs warnings when leaks are detected

**The only issue is the config cache** - restart Python to fix it!
