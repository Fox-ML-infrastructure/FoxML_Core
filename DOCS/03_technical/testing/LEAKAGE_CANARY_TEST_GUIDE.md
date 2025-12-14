# Leakage Canary Test Guide

**Date**: 2025-12-13  
**Related**: [Leakage Controls Evaluation](../architecture/LEAKAGE_CONTROLS_EVALUATION.md) | [Leakage Validation Fix](../fixes/2025-12-13-leakage-validation-fix.md) | [Fingerprint Tracking](../fixes/2025-12-13-lookback-fingerprint-tracking.md) | [Fingerprint Improvements](../fixes/2025-12-13-fingerprint-improvements.md)

**Purpose**: Validate pipeline integrity using known-leaky targets as canaries

## Overview

Leaky-by-default targets are **useful canaries** that force your pipeline to walk into the exact traps you're trying to detect. They validate **guardrails**, not alpha.

## What Canary Targets Validate

### ✅ They Help Catch:

1. **Split / gap logic bugs**
   - Purge/embargo calculation errors
   - Horizon math mistakes
   - Interval conversion bugs

2. **Feature alignment bugs**
   - Accidentally using `t+1` features at `t`
   - Timestamp misalignment

3. **CV preprocessing leakage**
   - Global imputer/scaler fit on full data
   - Cross-fold contamination

4. **Schema/pattern filters failing**
   - Label columns slipping into X
   - Target patterns not excluded

5. **Cross-sectional mixing mistakes**
   - Timestamp joins that let info bleed
   - Panel data grouping errors

### ❌ They Don't Prove:

- Your "real" targets are clean
- Your model generalizes
- You have alpha

**If a known-canary target doesn't get blocked/flagged, your defenses are broken.**

## Test Configuration

### Config File: `CONFIG/experiments/leakage_canary_test.yaml`

```bash
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "leakage_canary_test_results" \
  --experiment-config leakage_canary_test
```

### Key Settings

- **Policy**: `strict` (hard-stop on violations)
- **Manual targets**: Known leaky targets (e.g., `y_will_swing_high_60m_0.05`)
- **Leakage diagnostics**: Enabled (`run_leakage_diagnostics: true`)
- **Minimal data**: Small limits for fast testing
- **Sequential execution**: Disabled parallel for easier debugging

## Expected Outcomes

### 1. Hard-Stop Scenarios (Policy=Strict)

If purge/embargo are insufficient:

```
🚨 LEAKAGE VIOLATION: purge (105.0m) < lookback_requirement (160.0m) [max_lookback=100.0m + buffer=5.0m]
RuntimeError: (policy: strict - training blocked)
```

**Expected**: ✅ Hard-stop prevents training

### 2. Auto-Exclude Scenarios

If targets are leaky but purge/embargo are sufficient:

```
LEAKAGE WARNING: 6 models have ROC-AUC > 0.90
Excluded y_will_swing_high_60m_0.05 (SUSPICIOUS)
```

**Expected**: ✅ Target marked as SUSPICIOUS, excluded from rankings

### 3. Negative Controls (Smoke Tests)

#### Permutation Test
- **Action**: Shuffle `y` within train folds
- **Expected**: AUC collapses to ~0.50
- **Failure threshold**: AUC > 0.58 → pipeline leak detected

#### Feature Shift Test
- **Action**: Shift all features by +1 bar
- **Expected**: AUC drops hard
- **Failure threshold**: AUC > 0.55 → leak detected

**Expected**: ✅ Both tests pass (AUC ~0.50 after shuffle, drops after shift)

### 4. Top Features Analysis

For suspicious targets, system logs top 20 features:

```
TOP FEATURES USED (for leakage diagnosis)
lightgbm: Top 20 features by importance:
  feature_name: 0.1234
  ...
```

**Check for**: Features that directly encode the label window or future information

## Validation Checklist

After running canary test, verify:

- [ ] **Hard-stop works**: Targets with insufficient purge/embargo are blocked
- [ ] **SUSPICIOUS flag works**: High-scoring targets (AUC>0.90) are flagged
- [ ] **Permutation test passes**: Shuffled labels → AUC ~0.50
- [ ] **Feature shift test passes**: Shifted features → AUC drops
- [ ] **Top features logged**: For suspicious targets, top 20 features are logged
- [ ] **Canaries excluded**: Canary targets don't appear in "top targets" rankings

## If Canaries Pass Without Flags

**This indicates a pipeline bug:**

1. Check purge/embargo calculation
2. Check CV splitter (is it time-safe?)
3. Check feature filtering (are leaky features being dropped?)
4. Check smoke tests (are they wired as gates?)
5. Review logs for warnings that were ignored

## Integration with CI/CD

Add to your test suite:

```bash
# Run canary test as part of CI
python -m TRAINING.orchestration.intelligent_trainer \
  --output-dir "ci_canary_test" \
  --experiment-config leakage_canary_test

# Assert: All canary targets should be SUSPICIOUS or hard-stop
# If any pass with high scores, fail the build
```

## Future Enhancements

1. **Automated assertions**: Parse results and assert expected outcomes
2. **Smoke tests as gates**: Wire permutation/shift tests to block "PASS" status
3. **Canary registry**: Maintain a list of known-leaky targets with expected behavior
4. **Regression detection**: Compare canary results across runs to catch regressions

## Related Documentation

- [Leakage Controls Evaluation](../architecture/LEAKAGE_CONTROLS_EVALUATION.md)
- [Leakage Validation Fix](../fixes/2025-12-13-leakage-validation-fix.md)
- [Safety Configuration](../../02_reference/configuration/SAFETY_CONFIG.md)
