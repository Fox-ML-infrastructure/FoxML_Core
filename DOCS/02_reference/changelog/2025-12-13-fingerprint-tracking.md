# Changelog — 2025-12-13 (Fingerprint Tracking & Lookback Consistency)

**Fingerprint Tracking System for Lookback Computation Consistency**

**Related**: [SST Enforcement Design](../../../TRAINING/utils/SST_ENFORCEMENT_DESIGN.md) | [Fingerprint Improvements](../../03_technical/fixes/2025-12-13-fingerprint-improvements.md)

For a quick overview, see the [root changelog](../../../CHANGELOG.md).  
For other dates, see the [changelog index](README.md).

---

## Summary

Implemented comprehensive fingerprint tracking system to ensure lookback computations use the exact same feature set throughout the pipeline. This fixes the core issue where "reported max=100.0m but actual max from features=86400.0m" warnings indicated lookback was being computed on different feature sets at different stages.

---

## Added

### Fingerprint Tracking System

**Set-Invariant Fingerprints with Order-Change Detection**
- **Enhancement**: Added fingerprint tracking to all lookback computations
- **Implementation**:
  - `_compute_feature_fingerprint()` returns `(set_fingerprint, order_fingerprint)` tuple
  - Set fingerprint: sorted feature list (set-invariant, for equality checks)
  - Order fingerprint: original order (for order-change detection)
- **Benefits**:
  - Prevents false alarms from column reordering
  - Detects actual feature set changes vs. order changes
  - Enables invariant checks across pipeline stages
- **Files**:
  - `TRAINING/utils/leakage_budget.py` - Fingerprint computation and validation
  - `TRAINING/utils/cross_sectional_data.py` - Order-change detection in logging

### LookbackResult Dataclass

**Type-Safe Return Type for Lookback Computations**
- **Enhancement**: Replaced brittle tuple returns with `LookbackResult` dataclass
- **Structure**:
  ```python
  @dataclass(frozen=True)
  class LookbackResult:
      max_minutes: Optional[float]
      top_offenders: List[Tuple[str, float]]
      fingerprint: str
      order_fingerprint: str
  ```
- **Benefits**:
  - Type-safe (prevents silent mis-wires)
  - Self-documenting (clear field names)
  - Extensible (easy to add fields)
- **Files**:
  - `TRAINING/utils/leakage_budget.py` - Dataclass definition
  - All call sites updated to use dataclass

### Explicit Stage Logging

**Clear Visibility into Feature Set Evolution**
- **Enhancement**: Added explicit stage markers in logs
- **Stages**:
  - `PRE_GATEKEEPER`: Before gatekeeper runs
  - `POST_GATEKEEPER`: After gatekeeper drops features
  - `POST_PRUNE`: After pruning selects final features
  - `MODEL_TRAIN_INPUT`: Final feature set used in training
- **Benefits**:
  - Clear visibility into which stage each fingerprint represents
  - Easier debugging when fingerprints don't match
  - Audit trail of feature set evolution
- **Files**:
  - `TRAINING/ranking/predictability/model_evaluation.py` - Stage logging

### Leakage Canary Test Configuration

**Test Configuration for Pipeline Integrity Validation**
- **Enhancement**: Created dedicated test config for canary leaky targets
- **Purpose**: Validate guardrails (purge/embargo, split logic, feature alignment)
- **Features**:
  - Uses known-leaky targets as canaries
  - Strict policy for hard-stop validation
  - Minimal data limits for fast testing
  - Sequential execution for easier debugging
- **Files**:
  - `CONFIG/experiments/leakage_canary_test.yaml` - Test configuration
  - `DOCS/03_technical/testing/LEAKAGE_CANARY_TEST_GUIDE.md` - Usage guide

---

## Fixed

### Lookback Mismatch Warnings

**Root Cause**: Lookback computed on different feature sets at different stages
- **Symptom**: "reported max=100.0m but actual max from features=86400.0m"
- **Fix**: All lookback computations now validate against MODEL_TRAIN_INPUT fingerprint
- **Files**:
  - `TRAINING/utils/leakage_budget.py` - Fingerprint validation
  - `TRAINING/ranking/predictability/model_evaluation.py` - Invariant checks

### MODEL_TRAIN_INPUT Fingerprint Timing

**Root Cause**: Fingerprint computed before pruning, not after
- **Symptom**: Fingerprint didn't represent final feature set used in training
- **Fix**: Fingerprint now computed AFTER pruning (POST_PRUNE stage)
- **Files**:
  - `TRAINING/ranking/predictability/model_evaluation.py` - Timing fix

### Gatekeeper Missing Features

**Root Cause**: Gatekeeper using different feature→lookback mapping than analyzer
- **Symptom**: Features like `price_momentum_60d` not caught by gatekeeper
- **Fix**: Gatekeeper now uses same unified lookback calculator as analyzer
- **Files**:
  - `TRAINING/ranking/predictability/model_evaluation.py` - Unified calculator

### Tuple vs Dataclass Return Type Mismatch

**Root Cause**: Wrapper function returning tuple, code expecting dataclass
- **Symptom**: `AttributeError: 'tuple' object has no attribute 'max_minutes'`
- **Fix**: Wrapper now returns `LookbackResult` dataclass directly
- **Files**:
  - `TRAINING/utils/resolved_config.py` - Return type fix
  - `TRAINING/ranking/predictability/model_evaluation.py` - Backward compat handling

### NameError Fixes

**Multiple NameError fixes during implementation**:
1. `name 'fingerprint' is not defined` - Fixed variable name to `set_fingerprint`
2. `name 'model_train_input_fingerprint' is not defined` - Fixed to use `post_gatekeeper_fp`
3. `name 'Any' is not defined` - Added import from typing
- **Files**:
  - `TRAINING/utils/leakage_budget.py` - Variable name fixes
  - `TRAINING/ranking/predictability/model_evaluation.py` - Scope fixes

### TypeError: Unexpected Keyword Argument

**Root Cause**: Wrapper function didn't accept new parameters
- **Symptom**: `compute_feature_lookback_max() got an unexpected keyword argument 'expected_fingerprint'`
- **Fix**: Updated wrapper signature to accept and pass through new parameters
- **Files**:
  - `TRAINING/utils/resolved_config.py` - Parameter forwarding

---

## Changed

### Return Type Signatures

**All lookback computation functions now return consistent types**:
- `compute_budget()`: Returns `(LeakageBudget, set_fingerprint, order_fingerprint)` tuple
- `compute_feature_lookback_max()`: Returns `LookbackResult` dataclass
- Wrapper functions: Return dataclass directly (not tuple)

### Fingerprint Computation

**Changed from order-sensitive to set-invariant**:
- **Before**: Order-sensitive fingerprint (different order = different fingerprint)
- **After**: Set-invariant fingerprint (sorted) + separate order fingerprint
- **Benefits**: Prevents false alarms from column reordering

### Stage Logging

**Added explicit stage markers**:
- All feature set transitions now logged with stage names
- Fingerprints logged at each stage for traceability
- Order changes detected and logged separately

---

## Files Modified

### Core Implementation
- `TRAINING/utils/leakage_budget.py` - Fingerprint tracking, dataclass, validation
- `TRAINING/utils/cross_sectional_data.py` - Set-invariant fingerprints, order detection
- `TRAINING/ranking/predictability/model_evaluation.py` - Stage logging, invariant checks
- `TRAINING/utils/resolved_config.py` - Wrapper return type, parameter forwarding

### Call Sites Updated
- `TRAINING/ranking/shared_ranking_harness.py` - Dataclass handling
- `TRAINING/utils/feature_sanitizer.py` - Dataclass handling

### Configuration
- `CONFIG/experiments/leakage_canary_test.yaml` - New test config
- `CONFIG/experiments/README.md` - Updated with canary test info

### Documentation
- `DOCS/03_technical/fixes/2025-12-13-lookback-fingerprint-tracking.md` - Initial implementation
- `DOCS/03_technical/fixes/2025-12-13-fingerprint-improvements.md` - Improvements
- `DOCS/03_technical/fixes/2025-12-13-lookback-result-dataclass-migration.md` - Migration guide
- `DOCS/03_technical/testing/LEAKAGE_CANARY_TEST_GUIDE.md` - Canary test guide

---

## Related Documentation

- [Leakage Controls Evaluation](../03_technical/architecture/LEAKAGE_CONTROLS_EVALUATION.md)
- [Leakage Validation Fix](../03_technical/fixes/2025-12-13-leakage-validation-fix.md)
- [Fingerprint Improvements](../03_technical/fixes/2025-12-13-fingerprint-improvements.md)
- [Lookback Result Migration](../03_technical/fixes/2025-12-13-lookback-result-dataclass-migration.md)
- [Leakage Canary Test Guide](../03_technical/testing/LEAKAGE_CANARY_TEST_GUIDE.md)

---

## Testing

Run with canary test config to validate:
```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "fingerprint_validation_test" \
  --experiment-config leakage_canary_test
```

Expected: No fingerprint mismatch warnings, consistent lookback values throughout, explicit stage logging.
