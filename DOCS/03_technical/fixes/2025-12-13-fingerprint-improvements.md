# Fingerprint Tracking Improvements

**Date**: 2025-12-13  
**Related**: [Lookback Fingerprint Tracking](2025-12-13-lookback-fingerprint-tracking.md) | [Lookback Result Migration](2025-12-13-lookback-result-dataclass-migration.md) | [Leakage Validation Fix](2025-12-13-leakage-validation-fix.md) | [Leakage Controls Evaluation](../architecture/LEAKAGE_CONTROLS_EVALUATION.md) | [Canary Test Guide](../testing/LEAKAGE_CANARY_TEST_GUIDE.md)

**Follow-up**: Improvements to fingerprint tracking implementation

## Improvements Made

### 1. ✅ MODEL_TRAIN_INPUT Fingerprint is Truly Final (POST_PRUNE)

**Fixed**: Fingerprint is now computed AFTER pruning, not just post-gatekeeper.

**Execution Order**:
1. Gatekeeper runs (drops features) → `POST_GATEKEEPER` stage logged
2. Pruning happens (in `train_and_evaluate_models`) → `POST_PRUNE` stage logged
3. MODEL_TRAIN_INPUT fingerprint computed → Uses final pruned feature set

**Changes**:
- Added explicit stage logging: `PRE_GATEKEEPER`, `POST_GATEKEEPER`, `POST_PRUNE`, `MODEL_TRAIN_INPUT`
- MODEL_TRAIN_INPUT fingerprint computed from feature_names AFTER pruning (line ~1128)
- All lookback computations validate against MODEL_TRAIN_INPUT fingerprint

### 2. ✅ Set-Invariant Fingerprints with Order-Change Detection

**Changed**: Fingerprints are now set-invariant (sorted) by default, with separate order-sensitive fingerprint for order-change detection.

**Implementation**:
- `_compute_feature_fingerprint()` now returns `(set_fingerprint, order_fingerprint)` tuple
- Set fingerprint: sorted feature list (for set equality checks)
- Order fingerprint: original order (for order-change detection)
- `_log_feature_set()` detects and logs order changes separately

**Benefits**:
- Same feature set with different order → same set fingerprint, separate order warning
- Prevents false alarms from column reordering
- Still detects actual feature set changes

### 3. ✅ LookbackResult Dataclass

**Replaced**: Tuple returns with `LookbackResult` dataclass.

**Before**:
```python
max_lookback, top_offenders, fingerprint = compute_feature_lookback_max(...)
```

**After**:
```python
result = compute_feature_lookback_max(...)
max_lookback = result.max_minutes
top_offenders = result.top_offenders
fingerprint = result.fingerprint
order_fingerprint = result.order_fingerprint
```

**Benefits**:
- Type-safe (prevents silent mis-wires)
- Self-documenting (clear field names)
- Extensible (easy to add fields)

### 4. ✅ Explicit Stage Logging

**Added**: Explicit stage markers in logs:
- `PRE_GATEKEEPER`: Before gatekeeper runs
- `POST_GATEKEEPER`: After gatekeeper drops features
- `POST_PRUNE`: After pruning selects final features
- `MODEL_TRAIN_INPUT`: Final feature set used in training

**Benefits**:
- Clear visibility into which stage each fingerprint represents
- Easier debugging when fingerprints don't match
- Audit trail of feature set evolution

### 5. ⏳ Integration Test (Pending)

**TODO**: Add integration test that:
- Starts with features containing `*_60d` (86400m lookback)
- Gatekeeper/pruner drops it
- Ensures computed max lookback becomes small
- Ensures "top offenders" list no longer mentions dropped feature
- Ensures fingerprint invariant passes end-to-end

## Files Modified

1. **`TRAINING/utils/leakage_budget.py`**
   - Added `LookbackResult` dataclass
   - Modified `_compute_feature_fingerprint()` to return both fingerprints
   - Modified `compute_budget()` to return `(LeakageBudget, set_fp, order_fp)`
   - Modified `compute_feature_lookback_max()` to return `LookbackResult`

2. **`TRAINING/utils/cross_sectional_data.py`**
   - Modified `_compute_feature_fingerprint()` to return both fingerprints
   - Added order-change detection in `_log_feature_set()`

3. **`TRAINING/ranking/predictability/model_evaluation.py`**
   - Added explicit stage logging (PRE_GATEKEEPER, POST_GATEKEEPER, POST_PRUNE)
   - Updated all fingerprint computations to use new tuple return
   - Updated all lookback computations to use `LookbackResult` dataclass
   - Fixed MODEL_TRAIN_INPUT to be truly POST_PRUNE

4. **`TRAINING/utils/resolved_config.py`**
   - Updated wrapper to handle `LookbackResult` dataclass

5. **`TRAINING/ranking/shared_ranking_harness.py`**
   - Updated to handle `LookbackResult` dataclass

6. **`TRAINING/utils/feature_sanitizer.py`**
   - Updated to handle `LookbackResult` dataclass

## Validation

### Before Fix

```
⚠️  Lookback mismatch: reported max=100.0m but actual max from features=86400.0m
Top lookback features: ..._60d(86400m), ..._20d(28800m)
```

### After Fix

```
📊 FEATURESET [PRE_GATEKEEPER]: n=300, fingerprint=abc12345
📊 FEATURESET [POST_GATEKEEPER]: n=285, fingerprint=def67890
📊 FEATURESET [POST_PRUNE]: n=108, fingerprint=ghi11111
📊 FEATURESET [MODEL_TRAIN_INPUT]: n=108, fingerprint=ghi11111 (POST_PRUNE)
📊 compute_budget(POST_GATEKEEPER): max_lookback=100.0m, n_features=285, fingerprint=def67890
✅ Fingerprint validated: all lookback computations use same feature set
```

## Key Invariants

1. **MODEL_TRAIN_INPUT fingerprint == POST_PRUNE fingerprint** (must be equal)
2. **All lookback computations use MODEL_TRAIN_INPUT fingerprint** (for validation)
3. **Set fingerprint is set-invariant** (sorted, for set equality)
4. **Order fingerprint detects order changes** (separate warning)

## Testing

Run with canary test config to validate:
```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "fingerprint_validation_test" \
  --experiment-config leakage_canary_test
```

Expected: No fingerprint mismatch warnings, consistent lookback values throughout, explicit stage logging.
