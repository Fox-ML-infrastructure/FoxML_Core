# Config-Driven Leakage Filtering Refactor

**Date:** 2025-11-22 21:45:00
**Topic:** Refactored leakage filtering to be fully config-driven, no hardcoded patterns

---

## Summary

Refactored the entire leakage filtering system to be **100% config-driven** with zero hardcoded patterns. The system now loads all exclusion rules from `CONFIG/excluded_features.yaml`, making it work with any dataset, features, and targets without code changes.

---

## Changes Made

### 1. **Refactored `SCRIPTS/utils/leakage_filtering.py`**
 - **Removed all hardcoded patterns** - no more hardcoded regex/prefix/keyword lists
 - **Added config loader** - loads patterns from `CONFIG/excluded_features.yaml`
 - **Support for 4 pattern types:**
 - `regex_patterns`: Python regex (e.g., `"^tth_"`)
 - `prefix_patterns`: String prefixes (e.g., `"tth_"`)
 - `keyword_patterns`: Substring matches (e.g., `"peak"`)
 - `exact_patterns`: Exact feature names
 - **Configurable target classification** - rules for identifying target types from config
 - **Configurable horizon extraction** - patterns for extracting time horizons from config
 - **Caching** - config is loaded once and cached for performance

### 2. **Restructured `CONFIG/excluded_features.yaml`**
 - **New structure:**
 - `always_exclude`: Patterns applied to all targets (regex, prefix, keyword, exact)
 - `target_type_rules`: Target-specific patterns (forward_return, barrier, first_touch)
 - `target_classification`: Rules for identifying target types
 - `horizon_extraction`: Patterns for extracting time horizons
 - `metadata_columns`: Non-feature columns to exclude
 - `config`: Feature flags
 - **Added missing patterns:**
 - `mfe_` and `mdd_` to `always_exclude -> prefix_patterns` (MFE/MDD raw values)
 - `fwd_ret_` to `barrier -> prefix_patterns` (forward returns for barrier targets)
 - **Fixed regex patterns** - corrected YAML escaping for `\d` in regex patterns

### 3. **Enhanced Feature Importance Export**
 - **Per-symbol exports** - saves CSV files after each symbol completes
 - **Aggregated exports** - averages importances across symbols
 - **Directory structure:**
     ```
     {output_dir}/feature_importances/
       {target_name}/
         {symbol}/
           lightgbm_importances.csv
           xgboost_importances.csv
           random_forest_importances.csv
           ...
         AGGREGATED/
           lightgbm_importances.csv  # Averaged across symbols
           ...
     ```
 - **CSV columns:**
 - `feature`: Feature name
 - `importance`: Raw importance score
 - `importance_pct`: Percentage of total importance
 - `cumulative_pct`: Cumulative percentage (sorted by importance)

### 4. **Created Analysis Script**
 - **`SCRIPTS/analyze_feature_importances.py`** - analyzes feature importance results
 - **Identifies leaking features** - scans all CSV files and finds features matching leak patterns
 - **Generates recommendations** - suggests config updates based on findings
 - **Usage:** `python SCRIPTS/analyze_feature_importances.py <output_dir>`

### 5. **Fixed Leakage Detection**
 - **Fixed `all_suspicious_features` initialization** - was causing `NameError`
 - **Updated return values** - `train_and_evaluate_models` now returns 5 values (added `feature_importances`)
 - **Enhanced leak detection** - better reporting of top features when leaks detected

---

## Files Changed

### Modified
- `SCRIPTS/utils/leakage_filtering.py` - Complete refactor to config-driven system
- `CONFIG/excluded_features.yaml` - Restructured with new pattern system
- `SCRIPTS/rank_target_predictability.py` - Added feature importance export, fixed return values

### Created
- `SCRIPTS/analyze_feature_importances.py` - Analysis tool for feature importance results

---

## Key Improvements

### 1. **Zero Hardcoding**
 - All exclusion patterns are in config
 - Add new patterns by editing YAML (no code changes)
 - Works with any dataset/target combination

### 2. **Flexible Pattern System**
 - 4 pattern types (regex, prefix, keyword, exact)
 - Target-type-specific rules
 - Configurable horizon overlap detection

### 3. **Better Leak Detection**
 - Feature importance exports show exactly which features models use
 - Analysis script automatically identifies leaks
 - Clear recommendations for config updates

### 4. **Analysis Workflow**
 - Run ranking script → generates feature importance CSVs
 - Run analysis script → identifies leaks and suggests fixes
 - Update config → re-run ranking → verify leaks are filtered

---

## Leaking Features Identified

From analysis of existing results for `y_will_peak_60m_0.8`:

1. **MFE features** (Max Favorable Excursion - requires future path):
 - `mfe_60m_0.001` - up to 4.07% importance
 - `mfe_60m_0.002` - up to 5.34% importance
 - `mfe_5m_0.002`, `mfe_5m_0.001`, `mfe_60m_0.005`, etc.

2. **MDD features** (Max Drawdown - requires future path):
 - `mdd_60m_0.002` - 1.52% importance
 - `mdd_60m_0.005` - 1.14% importance

3. **Forward return features** (overlapping with target):
 - `fwd_ret_15m`, `fwd_ret_30m`, `fwd_ret_60m`, `fwd_ret_5d`, `fwd_ret_20d`, `fwd_ret_120m`

**All of these are now filtered by the updated config.**

---

## Testing

### Verification Tests
```python
# Test filtering
from scripts.utils.leakage_filtering import filter_features_for_target

test_cols = ['mfe_60m_0.001', 'mfe_5m_0.002', 'mdd_5m_0.002',
             'fwd_ret_15m', 'fwd_ret_60m', 'ret_zscore_5m', 'feature1']
target = 'y_will_peak_60m_0.8'

safe = filter_features_for_target(test_cols, target, verbose=True)
# Result: Only 'ret_zscore_5m' and 'feature1' remain (all leaks excluded)
```

### Analysis Script
```bash
python SCRIPTS/analyze_feature_importances.py results/target_rankings_updated
```

---

## Migration Notes

### For Existing Users
- **No breaking changes** - existing configs still work
- **Recommended:** Review `CONFIG/excluded_features.yaml` and add any custom patterns
- **Re-run ranking** after updating config to verify leaks are filtered

### For New Datasets
- **Edit `CONFIG/excluded_features.yaml`** to add dataset-specific patterns
- **No code changes needed** - all filtering is config-driven
- **Use analysis script** to identify leaks in your results

---

## Next Steps

1. **Re-run ranking script** with updated config to verify leaks are filtered
2. **Review feature importance CSVs** to understand which features models use
3. **Add custom patterns** to config as needed for your specific dataset
4. **Use analysis script** regularly to catch new leaks

---

## Related Files

- `CONFIG/excluded_features.yaml` - Main configuration file
- `SCRIPTS/utils/leakage_filtering.py` - Filtering implementation
- `SCRIPTS/analyze_feature_importances.py` - Analysis tool
- `SCRIPTS/rank_target_predictability.py` - Ranking script with export

