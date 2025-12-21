# Feature Selection Routing and Training View Tracking Fixes

**Date**: 2025-12-21  
**Type**: Bug Fix, Enhancement  
**Impact**: High - Ensures correct feature selection and training routing with proper view separation  
**Breaking**: No - Backward compatible

## Summary

Fixed multiple issues related to feature selection routing and training pipeline view tracking:

1. **Path resolution warning** - Fixed path resolution logic that was walking to root directory
2. **View tracking in feature selection routing** - Added view information (CROSS_SECTIONAL/SYMBOL_SPECIFIC) to `feature_selection_routing.json`
3. **Training reproducibility view tracking** - Added route/view information to training reproducibility tracking for proper output separation
4. **BOTH route feature selection** - Fixed BOTH route to use symbol-specific features for symbol-specific model training
5. **Per-target routing decision view** - Added view information to per-target `routing_decision.json` files

## Problem

### Issue 1: Path Resolution Warning

Path resolution in `feature_selection_reporting.py` was walking all the way to root directory (`/`) when it couldn't find run directory markers, causing warnings:
```
Path resolution failed - base_output_dir resolved to root or invalid: /. Using original output_dir
```

**Root Cause**:
- Path resolution started from `repro_dir` (deep in REPRODUCIBILITY structure) and walked up
- If it didn't find `targets/`, `globals/`, or `cache/` directories, it would walk all the way to root
- Should start from `output_dir` parameter when available (more reliable)

### Issue 2: Missing View Information in Feature Selection Routing

`feature_selection_routing.json` didn't track which view was used for feature selection, making it impossible to:
- Know which view was used for feature selection per target
- Compare feature selection results across views
- Ensure consistency between feature selection view and routing decision view

**Root Cause**:
- `save_target_routing_metadata` didn't accept or use `view` parameter
- Routing decision keys didn't include view information
- Global summary didn't track view per target

### Issue 3: Training Reproducibility Missing View Information

Training reproducibility tracking didn't pass `route_type` or `view` to `log_comparison()`, which meant:
- Reproducibility outputs might not be properly separated by view
- CROSS_SECTIONAL and SYMBOL_SPECIFIC outputs could be mixed
- Trend analysis and diff telemetry couldn't properly track view-specific changes

**Root Cause**:
- Training code didn't add `route` and `view` to `additional_data_with_cohort`
- `log_comparison()` calls didn't pass `route_type` parameter
- Reproducibility tracker had to infer view from symbol presence (not explicit)

### Issue 4: BOTH Route Not Using Symbol-Specific Features

BOTH route created both CS and symbol-specific features during feature selection, but training only used CS features for all models:
- Symbol-specific features were created but ignored
- Symbol-specific models used wrong features (CS instead of symbol-specific)
- Data accuracy issue - wrong features used for symbol-specific training

**Root Cause**:
- BOTH route handling in training only extracted CS features
- Symbol-specific training path wasn't triggered for BOTH route
- No logic to train symbol-specific models using symbol-specific features

### Issue 5: Per-Target Routing Decision Missing View

Per-target `routing_decision.json` files didn't include view information, making it unclear which view was used for feature selection.

## Solution

### Fix 1: Path Resolution Warning

**File**: `TRAINING/ranking/feature_selection_reporting.py` (lines 129-157)

**Changes**:
- Start path resolution from `output_dir` parameter when available (preferred method)
- Fall back to `repro_dir` walk-up only if `output_dir` method doesn't work
- Improved validation to prevent walking to root directory

**Before**:
```python
base_output_dir = repro_dir
for _ in range(10):
    if (base_output_dir / "targets").exists() or ...:
        break
    base_output_dir = base_output_dir.parent
```

**After**:
```python
# First, try to find run directory from output_dir (preferred method)
if output_dir and output_dir.exists():
    temp_dir = Path(output_dir)
    for _ in range(10):
        if (temp_dir / "targets").exists() or ...:
            base_output_dir = temp_dir
            break
        temp_dir = temp_dir.parent

# Fallback: If output_dir method didn't work, try from repro_dir
if base_output_dir is None or base_output_dir == Path('/'):
    # ... fallback logic
```

### Fix 2: View Tracking in Feature Selection Routing

**File**: `TRAINING/orchestration/target_routing.py` (lines 220-318)

**Changes**:
- Added `view` parameter to `save_target_routing_metadata` function
- Changed routing decision key to include view: `target_name:view` (e.g., `fwd_ret_5d:CROSS_SECTIONAL`)
- Added view information to routing decision metadata

**File**: `TRAINING/orchestration/intelligent_trainer.py` (line 1303)

**Changes**:
- Pass `view` parameter when calling `save_target_routing_metadata`

**Before**:
```python
save_target_routing_metadata(self.output_dir, target, conf, routing)
# Keys: {"target_name": {...}}
```

**After**:
```python
save_target_routing_metadata(self.output_dir, target, conf, routing, view=view)
# Keys: {"target_name:CROSS_SECTIONAL": {...}, "target_name:SYMBOL_SPECIFIC": {...}}
```

### Fix 3: Training Reproducibility View Tracking

**File**: `TRAINING/training_strategies/execution/training.py`

**Changes**:
- **CROSS_SECTIONAL route** (lines 1090-1120): Added `route` and `view` to `additional_data_with_cohort`, pass `route_type` to `log_comparison()`
- **SYMBOL_SPECIFIC route** (lines 756-770): Added `route` and `view` to `additional_data_with_cohort`, pass `route_type` to `log_comparison()`
- **BOTH route**: Added appropriate view information for both CS and symbol-specific training paths

**Before**:
```python
additional_data_with_cohort = {
    "strategy": strategy,
    "n_features": len(feature_names),
    "model_family": family,
    **cohort_additional_data
}
tracker.log_comparison(
    stage="model_training",
    item_name=f"{target}:{family}",
    metrics=metrics_with_cohort,
    additional_data=additional_data_adapted
)
```

**After**:
```python
additional_data_with_cohort = {
    "strategy": strategy,
    "n_features": len(feature_names),
    "model_family": family,
    "route": route,  # Add route information
    "view": "CROSS_SECTIONAL",  # Add view information
    **cohort_additional_data
}
tracker.log_comparison(
    stage="model_training",
    item_name=f"{target}:{family}",
    metrics=metrics_with_cohort,
    additional_data=additional_data_adapted,
    route_type=route  # Pass route_type for proper view separation
)
```

### Fix 4: BOTH Route Feature Selection

**File**: `TRAINING/training_strategies/execution/training.py` (lines 269-320, 1430-1520)

**Changes**:
- Extract both CS and symbol-specific features from BOTH route structure
- Check routing plan to determine which symbols need symbol-specific training
- Train symbol-specific models using symbol-specific features after CS training completes
- Each symbol-specific model uses its own symbol-specific features

**Before**:
```python
elif route == 'BOTH':
    selected_features = target_feat_data['cross_sectional']
    # Only uses CS features, ignores symbol-specific features
```

**After**:
```python
elif route == 'BOTH':
    # Extract CS features for CS training
    selected_features = target_feat_data['cross_sectional']
    
    # Extract symbol-specific features for symbol-specific training
    symbol_specific_features = target_feat_data.get('symbol_specific', {})
    # Check routing plan for winner_symbols
    # Train symbol-specific models using symbol-specific features
    # ... (after CS training completes)
```

### Fix 5: Per-Target Routing Decision View

**File**: `TRAINING/orchestration/target_routing.py` (lines 260-272)

**Changes**:
- Added `view` field to per-target routing decision data structure

**Before**:
```python
routing_data = {
    target_name: {
        'target_name': target_name,
        'confidence': conf,
        'routing': routing,
        ...
    }
}
```

**After**:
```python
routing_data = {
    target_name: {
        'target_name': target_name,
        'view': view_normalized,  # Add view information
        'confidence': conf,
        'routing': routing,
        ...
    }
}
```

## Files Changed

1. `TRAINING/ranking/feature_selection_reporting.py`
   - Lines 129-157: Fixed path resolution to start from output_dir when available

2. `TRAINING/orchestration/target_routing.py`
   - Lines 220-318: Added view parameter, updated routing keys to include view, added view to per-target files

3. `TRAINING/orchestration/intelligent_trainer.py`
   - Line 1303: Pass view parameter to save_target_routing_metadata

4. `TRAINING/training_strategies/execution/training.py`
   - Lines 756-770: Added route/view to SYMBOL_SPECIFIC reproducibility tracking
   - Lines 1090-1120: Added route/view to CROSS_SECTIONAL reproducibility tracking
   - Lines 269-320: Fixed BOTH route to extract and use symbol-specific features
   - Lines 1430-1520: Added symbol-specific training for BOTH route

## Testing/Verification

### Expected Results

1. **Path Resolution**:
   - No more warnings about path resolution walking to root
   - Path resolution should correctly find run directory from output_dir

2. **Feature Selection Routing**:
   - `globals/feature_selection_routing.json` should have keys like `target_name:CROSS_SECTIONAL` and `target_name:SYMBOL_SPECIFIC`
   - Per-target `routing_decision.json` should include `view` field

3. **Training Reproducibility**:
   - Reproducibility outputs should be properly separated:
     - CROSS_SECTIONAL: `targets/<target>/reproducibility/CROSS_SECTIONAL/cohort={id}/`
     - SYMBOL_SPECIFIC: `targets/<target>/reproducibility/SYMBOL_SPECIFIC/symbol={symbol}/cohort={id}/`

4. **BOTH Route**:
   - CS models should use CS features
   - Symbol-specific models should use symbol-specific features (not CS features)
   - Both types of models should be trained for BOTH route

### Verification Steps

1. Run feature selection with multiple targets and views
2. Check `globals/feature_selection_routing.json` - should have view-specific keys
3. Check per-target `routing_decision.json` - should include view field
4. Run training with BOTH route target
5. Verify CS models use CS features, symbol-specific models use symbol-specific features
6. Check reproducibility outputs are separated by view correctly

## Related Changes

- Preceded by: [2025-12-21-catboost-verbosity-and-reproducibility-fixes.md](2025-12-21-catboost-verbosity-and-reproducibility-fixes.md) - n_features fix
- Related to: [2025-12-21-run-comparison-fixes.md](2025-12-21-run-comparison-fixes.md) - Target-first structure fixes
- Related to: [2025-12-20-feature-selection-output-organization-and-elastic-net-fail-fast.md](2025-12-20-feature-selection-output-organization-and-elastic-net-fail-fast.md) - Output organization

## Risk Assessment

- **Risk**: Low - All changes are additive (adding view information) or fix clear bugs (path resolution, BOTH route)
- **Impact**: 
  - Feature selection routing now properly tracks view information
  - Training reproducibility outputs are properly separated by view
  - BOTH route now uses correct features for each model type
  - Path resolution warnings eliminated
- **Rollback**: Simple git revert if needed

