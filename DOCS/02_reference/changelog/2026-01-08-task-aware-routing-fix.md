# Task-Aware Routing Fix

**Date**: 2026-01-08  
**Type**: Bug Fix, Enhancement  
**Impact**: Critical - Fixes incorrect routing for regression targets

## Problem

Routing logic used fixed thresholds (0.65/0.60/0.90/0.95) on a field named `auc`, assuming values in [0,1] range. This broke for regression targets where `auc` contains R² (can be negative), causing:
- Regression targets with negative R² but positive IC to route incorrectly or never route
- "Suspicious" thresholds (0.90/0.95) becoming nonsense for regression
- Inconsistent routing behavior between regression and classification targets

## Solution

Implemented task-aware routing using a unified `skill01` score that normalizes both regression IC and classification AUC-excess to [0,1] range.

### Key Changes

1. **Fixed IC Extraction Bug** (`model_evaluation.py`)
   - **Problem**: `primary_metric_mean_centered` was incorrectly set to R² for regression (line 7195)
   - **Fix**: Extract IC from `model_metrics` (already computed by `train_and_evaluate_models`)
   - **Implementation**: 
     - Extract IC values from all models in `model_metrics`
     - Compute mean IC for regression targets
     - Use IC for `primary_metric_mean_centered` (IC is already centered at 0)
     - Fallback to 0.0 if IC not available (with warning log)
   - **Location**: Lines 7168-7230

2. **Added skill01 Property** (`scoring.py`)
   - **New**: Computed property that normalizes `primary_metric_mean` to [0,1] range
   - **Formula**: `skill01 = 0.5 * (primary_metric_mean + 1.0)`
   - **Works for both**:
     - Regression IC: [-1, 1] → [0, 1]
     - Classification AUC-excess: [-0.5, 0.5] → [0, 1]
   - **Safety**: Clamps to [0,1] range (handles edge cases like failed targets)
   - **Location**: Lines 209-227

3. **Updated Routing Logic** (`target_routing.py`)
   - **Both functions updated**: `_compute_target_routing_decisions()` and `_compute_single_target_routing_decision()`
   - **Changes**:
     - Extract `skill01_cs` instead of `auc` for cross-sectional routing
     - Extract `symbol_skill01s` instead of `symbol_aucs` for symbol-specific routing
     - Use `skill01` thresholds from config (with backward compatibility)
     - Update all routing conditions to use `skill01`
     - Enhanced suspicious detection: high `skill01` + low `tstat` = suspicious (task-aware)
   - **Location**: Lines 40-209, 233-371

4. **Updated Config** (`configs.yaml`)
   - **New thresholds**: `skill01_threshold`, `symbol_skill01_threshold`, `suspicious_skill01`, `suspicious_symbol_skill01`
   - **Backward compatibility**: Old `auc_threshold` names still supported (maps to new names)
   - **Location**: `CONFIG/ranking/targets/configs.yaml` (routing section)

5. **Enhanced Suspicious Detection**
   - **Task-aware**: High `skill01` (≥0.90) triggers suspicious check
   - **Stability check**: If `primary_metric_tstat > 3.0`, signal is stable (not blocked)
   - **Prevents**: Blocking legitimate strong signals that are stable
   - **Location**: `target_routing.py` lines 140-158, 307-324

6. **Updated Output Artifacts**
   - **CSV** (`reporting.py`): Added `skill01` column
   - **YAML** (`reporting.py`): Added `skill01` to target prioritization
   - **Metrics dict** (`metrics_schema.py`): Added `skill01` to `primary_metric` group
   - **Routing decisions**: Include `skill01_cs` and `skill01_sym_mean` (kept `auc` for backward compat)

7. **Added Unit Tests** (`target_routing_test.py`)
   - Tests for regression with negative R² but positive IC
   - Tests for classification (unchanged behavior)
   - Tests for suspicious detection with tstat checks
   - Tests for skill01 normalization

8. **Added Plan Hashing** (`manifest.py`)
   - **New**: `_compute_plan_hashes()` function
   - Hashes `globals/routing_plan/routing_plan.json` → `routing_plan_hash`
   - Hashes `globals/training_plan/master_training_plan.json` → `training_plan_hash`
   - Added to manifest as `plan_hashes` section
   - Enables fast change detection between runs

## Bug Fixes

1. **Failed Target Handling** (classification)
   - **Issue**: If `auc = -999.0` (failed), `auc_excess_mean = -999.5`, making `skill01` negative
   - **Fix**: Check for `auc < -900.0` or `np.isnan(auc)` before computing excess
   - **Location**: `model_evaluation.py` lines 7215-7232

2. **skill01 Range Clamping**
   - **Issue**: `skill01` could be negative or >1 if `primary_metric_mean` is outside expected range
   - **Fix**: Added `max(0.0, min(1.0, normalized))` clamp
   - **Location**: `scoring.py` line 227

3. **None Comparison in Routing**
   - **Issue**: `symbol_skill01_iqr` could be `None` when used in comparison
   - **Fix**: Added `symbol_skill01_iqr is not None` check before comparison
   - **Location**: `target_routing.py` lines 175, 341

## Files Changed

### Core Implementation
- `TRAINING/ranking/predictability/model_evaluation.py` - IC extraction fix
- `TRAINING/ranking/predictability/scoring.py` - skill01 property
- `TRAINING/ranking/target_routing.py` - Routing logic update (both functions)
- `CONFIG/ranking/targets/configs.yaml` - Routing thresholds

### Output Artifacts
- `TRAINING/ranking/predictability/reporting.py` - CSV/YAML output
- `TRAINING/ranking/predictability/metrics_schema.py` - Metrics dict

### Testing & Documentation
- `TRAINING/ranking/target_routing_test.py` - Unit tests (new file)
- `TRAINING/orchestration/utils/manifest.py` - Plan hashing

## Backward Compatibility

- ✅ `auc` field kept in `TargetPredictabilityScore` (contains R² for regression, AUC for classification)
- ✅ `auc` kept in routing decision output (marked deprecated)
- ✅ Config supports both old (`auc_threshold`) and new (`skill01_threshold`) names
- ✅ Existing snapshots/artifacts remain readable (new fields are additive)
- ✅ Deprecated fields (`symbol_auc_mean`, etc.) still populated (now contain skill01 values)

## Determinism Impact

- **Hash changes expected**: Routing decisions change for regression targets (this is correct)
- **Snapshot schema**: Unchanged (additive fields only)
- **Run hashes**: Will change (routing is part of run identity)

## Benefits

1. **Correct Routing**: Regression targets now route correctly even when R² < 0
2. **Unified Thresholds**: Single set of thresholds works for both task types
3. **Task-Aware Suspicious Detection**: Prevents blocking legitimate strong signals
4. **Fast Change Detection**: Plan hashes enable quick comparison between runs
5. **Better Observability**: `skill01` provides clear, normalized skill score

## Testing

- ✅ Unit tests added for routing logic
- ✅ Edge cases handled (None values, failed targets, empty lists)
- ✅ Backward compatibility verified
- ✅ No linter errors (only expected import resolution warnings)

## Migration Notes

- No migration required - changes are backward compatible
- Old config files continue to work (backward compatibility mapping)
- Existing code reading `auc` field continues to work (field preserved)
- New code should use `skill01` for routing decisions

## Future Enhancements

- Consider removing deprecated `auc` field in future version
- Consider removing backward compatibility mapping after transition period
- Consider adding `skill01` to more output artifacts for consistency
