# Leak Detection System Implementation

**Date:** 2025-11-22 21:30:00
**Feature:** Automatic detection of "Trojan Horse" features with suspiciously high importance

## Problem

The target ranking script was showing perfect scores (1.000) for multiple models, indicating data leakage. The existing regex-based filtering was successfully excluding 183 features with obvious names (like `tth_*`, `mfe_share_*`), but some features with "safe" names were still leaking future information.

### The "Trojan Horse" Problem

- Features with innocent names (e.g., `close`, `high`, `low`) can contain future information if:
 - They use centered moving averages (`center=True` uses `t+5` to calculate `t`)
 - They have backward shifts (`.shift(-1)` instead of `.shift(1)`)
 - They encode barrier information indirectly (e.g., current bar's High/Low matching the peak target definition)

- Tree-based models (LightGBM, XGBoost) can find perfect if/else splits that map directly to the target
- Statistical methods (Mutual Information, Univariate Selection) miss these because they look for broad correlations

## Solution: Feature Importance-Based Leak Detection

Implemented an automatic leak detection system that analyzes feature importance from trained models to identify suspicious features.

## Changes Made

### 1. Core Leak Detection Function (`_detect_leaking_features`)

**Location:** `SCRIPTS/rank_target_predictability.py:378-429`

Added a function that:
- Normalizes feature importances to sum to 1
- Flags features with >50% importance (likely leakage)
- Also flags features with >30% importance that are 3x larger than the next feature
- Logs immediate error/warning messages when leaks are detected

```python
def _detect_leaking_features(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    threshold: float = 0.50
) -> List[Tuple[str, float]]:
    """Detect features with suspiciously high importance (likely data leakage)."""
```

### 2. Integration into Model Training

**Locations:**
- `SCRIPTS/rank_target_predictability.py:700-707` - LightGBM
- `SCRIPTS/rank_target_predictability.py:745-750` - Random Forest
- `SCRIPTS/rank_target_predictability.py:870-876` - XGBoost

Added leak detection immediately after model training:
```python
importances = model.feature_importances_
suspicious_features = _detect_leaking_features(
    feature_names, importances, model_name='lightgbm', threshold=0.50
)
if suspicious_features:
    all_suspicious_features['lightgbm'] = suspicious_features
```

### 3. Leak Reporting Functions

**Location:** `SCRIPTS/rank_target_predictability.py:431-456`

#### `_log_suspicious_features()`
- Logs detailed per-symbol/target reports to `results/leak_detection_report.txt`
- Appends to file for each symbol evaluated
- Includes feature names and importance percentages

#### `save_leak_report_summary()`
**Location:** `SCRIPTS/rank_target_predictability.py:1430-1485`
- Creates final summary report at end of run
- Aggregates all detected leaks across all targets
- Includes recommendations for fixing leaks
- Saved to `{output_dir}/leak_detection_summary.txt`

### 4. Enhanced TargetPredictabilityScore

**Location:** `SCRIPTS/rank_target_predictability.py:72-105`

Added `suspicious_features` field to store leak detection results:
```python
@dataclass
class TargetPredictabilityScore:
    # ... existing fields ...
    suspicious_features: Dict[str, List[Tuple[str, float]]] = None
```

### 5. Updated Function Signatures

**Location:** `SCRIPTS/rank_target_predictability.py:379-387`

Updated `train_and_evaluate_models()` to:
- Accept `target_column` parameter for reporting
- Return 4 values instead of 3: `(model_metrics, model_scores, mean_importance, suspicious_features)`

### 6. Aggregation and Reporting

**Location:** `SCRIPTS/rank_target_predictability.py:1569-1587`

- Aggregates suspicious features across symbols
- Merges duplicate features, keeping max importance
- Logs to file after each symbol evaluation
- Stores in `TargetPredictabilityScore` for final summary

**Location:** `SCRIPTS/rank_target_predictability.py:1880-1884`

- Calls `save_leak_report_summary()` at end of main run
- Creates comprehensive summary of all detected leaks

## Output Files

1. **`results/leak_detection_report.txt`**
 - Detailed per-symbol/target report
 - Appended for each evaluation
 - Format:
     ```
     ================================================================================
     Target: y_will_peak_60m_0.8 | Symbol: AAPL
     ================================================================================

     LIGHTGBM - Suspicious Features:
     --------------------------------------------------------------------------------
       suspicious_feature_name                          | Importance: 95.2%
     ```

2. **`{output_dir}/leak_detection_summary.txt`**
 - Final summary of all detected leaks
 - Aggregated across all targets and symbols
 - Includes recommendations for fixing leaks

## Detection Criteria

### Primary Detection (>50% importance)
- Any feature with normalized importance ≥ 50% is flagged as a leak
- Indicates the model is using a single feature to make predictions
- Classic "Trojan Horse" pattern

### Secondary Detection (>30% and 3x larger)
- Features with ≥30% importance that are 3x larger than the next feature
- Catches cases where one feature dominates but doesn't exceed 50%
- Still indicates suspicious concentration of predictive power

## Expected Behavior

When running the script:
1. **During Training:** Immediate error/warning messages when leaks are detected:
   ```
    LEAK DETECTED: suspicious_feature has 95.2% importance in lightgbm (threshold: 50.0%) - likely data leakage!
   ```

2. **After Each Symbol:** Report saved to `leak_detection_report.txt`

3. **At End of Run:** Summary report created with:
 - Total targets with suspicious features
 - Total suspicious feature detections
 - Per-target breakdown
 - Recommendations for fixing

## Recommendations in Report

The summary report includes:
1. Review features with >50% importance - they likely contain future information
2. Check for:
 - Centered moving averages (center=True)
 - Backward shifts (.shift(-1) instead of .shift(1))
 - High/Low data that matches target definition
 - Features computed from the same barrier logic as the target
3. Add suspicious features to `leakage_filtering.py` exclusion list
4. Re-run ranking after fixing leaks

## Files Modified

1. `SCRIPTS/rank_target_predictability.py`
 - Added `_detect_leaking_features()` function
 - Added `_log_suspicious_features()` function
 - Added `save_leak_report_summary()` function
 - Updated `train_and_evaluate_models()` signature and return values
 - Integrated leak detection into LightGBM, Random Forest, XGBoost training
 - Updated `TargetPredictabilityScore` dataclass
 - Added aggregation and reporting logic

## Testing

The system was tested by:
- Compiling the file successfully (no syntax errors)
- Verifying function signatures match usage
- Ensuring all return values are properly handled

## Impact

- **Before:** Perfect scores (1.000) with no indication of which features were leaking
- **After:** Automatic detection and reporting of suspicious features with detailed logs and recommendations

## Next Steps

1. Run the script and review `leak_detection_report.txt` and `leak_detection_summary.txt`
2. Identify the specific features causing perfect scores
3. Add those features to `SCRIPTS/utils/leakage_filtering.py` exclusion patterns
4. Re-run ranking to verify scores are now realistic (0.13-0.17 for honest targets)

