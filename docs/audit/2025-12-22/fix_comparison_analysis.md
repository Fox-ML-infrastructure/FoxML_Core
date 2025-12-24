# Fix Comparison: Applied vs Plan Requirements

## Summary

This document compares the fixes just applied with the detailed plan requirements from the user's task specification.

---

## ✅ **FIXED - Issues Resolved**

### 1. Feature Selector Filtering ✅
**Plan Requirement:** "Don't keep selectors inside the training-family list"

**What Was Fixed:**
- Added `FEATURE_SELECTORS` filtering in `intelligent_trainer.py` (lines 2112-2120)
- Filters out: `random_forest`, `catboost`, `lasso`, `mutual_information`, `univariate_selection`, etc.
- Applied before passing families to training execution
- Final defensive filter added (line 2195) to catch any that slip through

**Status:** ✅ **COMPLETE** - Selectors are now filtered before training

---

### 2. Symbol-Specific Loop Using Validated Families ✅
**Plan Requirement:** "Your symbol-specific loop is using the unfiltered original families list, not the preflight 'valid trainer families'"

**What Was Fixed:**
- Line 580 in `training.py`: `families_to_train = validated_families`
- Symbol-specific training now uses `validated_families` from preflight (not original `target_families`)
- Preflight validation (lines 467-485) filters invalid families before any training loops

**Status:** ✅ **COMPLETE** - Symbol-specific loop uses validated families

---

### 3. Folder Structure Simplification ✅
**Plan Requirement:** "Models should be in `training_results/<family>/` not nested `training_results/training_results/`"

**What Was Fixed:**
- Removed nested `training_results/training_results/` path creation
- Changed model saving to: `output_dir / family /` (line 1341)
- Removed target subdirectories for model files
- Updated CS route, SYMBOL_SPECIFIC route, and BOTH route saving

**Status:** ✅ **COMPLETE** - Simplified folder structure implemented

---

### 4. Family Name Normalization ✅
**Plan Requirement:** "Family NeuralNetwork not found in TRAINER_MODULE_MAP (should be neural_network)"

**What Was Fixed:**
- Added normalization in `isolation_runner.py` (lines 489-503)
- Uses `normalize_family()` from `sst_contract.py` to convert "NeuralNetwork" → "neural_network"
- Applied before TRAINER_MODULE_MAP lookup

**Status:** ✅ **COMPLETE** - Family names normalized consistently

---

### 5. Reproducibility Path Handling ✅
**Plan Requirement:** "Reproducibility tracking stops throwing `'str' object has no attribute 'name'`"

**What Was Fixed:**
- Fixed Path/string conversion (lines 1220-1228)
- Ensures `output_dir` is Path object before operations
- Added `tracker_input_adapter` usage (lines 1307-1322) to handle Enum/string conversion

**Status:** ⚠️ **PARTIALLY COMPLETE** - Path issue fixed, but `.name` attribute issue may still exist if Enums are passed directly

---

## ❌ **NOT FIXED - Still Needs Implementation**

### 1. Training Plan 0 Jobs Handling ❌
**Plan Requirement:** 
- Option A: Hard-fail if `jobs==0` (no fallback)
- Option B: Downgrade to WARNING and make explicit `plan_disabled=True`

**Current State:**
- `training_plan_consumer.py` line 137-143: Still logs ERROR but returns fallback
- No explicit `plan_disabled` flag
- No hard-fail option

**What Needs Fix:**
```python
# Option A (hard-fail):
if not jobs:
    raise ValueError("Training plan has 0 jobs - cannot proceed. Check routing decisions.")

# Option B (explicit disable):
if not jobs:
    logger.warning("Training plan has 0 jobs - plan is disabled, using fallback")
    # Set explicit flag
```

**Status:** ❌ **NOT FIXED**

---

### 2. Routing Decisions Mismatch ❌
**Plan Requirement:** 
- "Routing decisions must be keyed by run-scoped fingerprint"
- "Don't load routing decisions from disk unless `--reuse-routing` is explicitly on"
- Enforce: `set(routing_decisions.targets) == set(filtered_targets)`

**Current State:**
- `intelligent_trainer.py` lines 2257-2260: Loads routing decisions without fingerprint check
- No `--reuse-routing` flag
- No validation that routing decisions match filtered targets

**What Needs Fix:**
```python
# Add fingerprint check:
run_fingerprint = compute_run_fingerprint(dataset, targets, config_hash)
if routing_decisions.get('fingerprint') != run_fingerprint:
    logger.warning("Routing decisions fingerprint mismatch - ignoring stale decisions")
    routing_decisions = {}

# Add target validation:
routing_targets = set(routing_decisions.keys())
filtered_targets_set = set(filtered_targets)
if not routing_targets.issubset(filtered_targets_set):
    logger.error(f"Routing decisions contain targets not in filtered list: {routing_targets - filtered_targets_set}")
```

**Status:** ❌ **NOT FIXED**

---

### 3. Reproducibility `.name` Attribute Issue ⚠️
**Plan Requirement:** "Replace `x.name` with `getattr(x, 'name', str(x))` or normalize inputs to Enums"

**Current State:**
- `tracker_input_adapter` is used (line 1309) but may not cover all cases
- Need to verify it handles all Enum/string conversions

**What Needs Fix:**
- Audit `tracker_input_adapter` to ensure it handles all Enum types
- Add defensive `getattr(x, 'name', str(x))` in reproducibility tracker itself

**Status:** ⚠️ **PARTIALLY FIXED** - Adapter exists but needs verification

---

### 4. Feature Registry Filtering Upstream ❌
**Plan Requirement:** 
- "Enforce registry constraints BEFORE ranking/selecting features"
- "Selection output should include: `selected_features_total=100`, `selected_features_registry_allowed=8`"

**Current State:**
- Feature selection happens before registry filtering
- Registry filtering happens at training time (too late)
- No visibility into why features were rejected

**What Needs Fix:**
- Move registry filtering into feature selection step
- Add metadata about rejected features and reasons
- Update feature selection output format

**Status:** ❌ **NOT FIXED**

---

### 5. Horizon→Bars Logic ❌
**Plan Requirement:** "Define horizon in terms of the same calendar your labels use (trading days, not 24/7)"

**Current State:**
- Horizon calculation assumes 24/7 trading (7200 minutes / 5 = 1440 bars)
- Doesn't account for market hours/trading days

**What Needs Fix:**
- Convert horizon to trading days/sessions
- Store `horizon_bars` directly in target spec instead of inferring

**Status:** ❌ **NOT FIXED**

---

### 6. Config Split: Training vs Selection Families ❌
**Plan Requirement:** "Split this in your SST config: `feature_selection_families` vs `training_families`"

**Current State:**
- Single `model_families` list contains both trainers and selectors
- Filtering happens at runtime, not at config level

**What Needs Fix:**
- Update config schema to have separate lists
- Update config loading to separate them
- Stop passing selectors into training at all

**Status:** ❌ **NOT FIXED** (but runtime filtering works)

---

## 📊 **Gap Analysis**

| Issue | Plan Priority | Status | Impact |
|-------|--------------|--------|--------|
| Feature selector filtering | Critical | ✅ Fixed | High - prevents training errors |
| Symbol-specific family loop | Critical | ✅ Fixed | High - prevents training errors |
| Folder structure | Medium | ✅ Fixed | Medium - organization |
| Family normalization | Medium | ✅ Fixed | Medium - prevents lookup errors |
| Training plan 0 jobs | Critical | ❌ Not Fixed | High - silent fallback |
| Routing mismatch | Critical | ❌ Not Fixed | High - stale data issues |
| Reproducibility `.name` | Medium | ⚠️ Partial | Medium - may still error |
| Registry filtering upstream | High | ❌ Not Fixed | High - feature count collapse |
| Horizon→bars logic | Medium | ❌ Not Fixed | Medium - incorrect calculations |
| Config split | Low | ❌ Not Fixed | Low - runtime filtering works |

---

## 🎯 **Recommended Next Steps**

### Immediate (Critical Bugs):
1. **Fix training plan 0 jobs** - Add hard-fail or explicit disable flag
2. **Fix routing decisions mismatch** - Add fingerprint validation
3. **Verify reproducibility `.name` fix** - Test with Enum inputs

### High Priority:
4. **Move registry filtering upstream** - Fix feature count collapse
5. **Add config split** - Separate training vs selection families in config

### Medium Priority:
6. **Fix horizon→bars logic** - Use trading days calendar

---

## ✅ **What Was Successfully Fixed**

The following critical issues from the plan were successfully resolved:

1. ✅ Feature selectors are filtered before training
2. ✅ Symbol-specific loop uses validated families
3. ✅ Folder structure simplified (no nested training_results)
4. ✅ Family names normalized consistently
5. ✅ Path handling fixed for reproducibility

These fixes address the most critical runtime errors (attempting to train selectors, using invalid families, folder structure issues).

---

## ⚠️ **Remaining Issues**

The following issues from the plan still need to be addressed:

1. ❌ Training plan 0 jobs handling (silent fallback)
2. ❌ Routing decisions mismatch (stale data)
3. ⚠️ Reproducibility `.name` (needs verification)
4. ❌ Feature registry filtering upstream
5. ❌ Horizon→bars calendar logic
6. ❌ Config split (nice-to-have, runtime filtering works)

---

## 📝 **Notes**

- The fixes applied address the **immediate runtime errors** (training selectors, invalid families, folder structure)
- The remaining issues are more about **data integrity and pipeline correctness** (stale routing, plan validation, upstream filtering)
- Runtime filtering works but config-level separation would be cleaner
- The reproducibility `.name` fix may be complete via `tracker_input_adapter`, but needs testing with actual Enum inputs

