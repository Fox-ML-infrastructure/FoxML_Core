# Known Issues & Failure Modes

**Documentation of potential errors and silent failure modes, and how they're handled.**

## ✅ Fixed Issues

### 1. IndexError in Common Families Calculation
**Issue:** If `filtered_targets[0]` doesn't exist in `target_families_map`, accessing it would cause IndexError.

**Fix:** Added check: `if filtered_targets and filtered_targets[0] in target_families_map`

**Location:** `TRAINING/orchestration/intelligent_trainer.py:723`

### 2. Empty Families List
**Issue:** If all model families are filtered out, `families_list` could be empty, causing training to fail.

**Fix:** Added validation with fallback to default families.

**Location:** `TRAINING/orchestration/intelligent_trainer.py:738-741`

### 3. All Targets Filtered Out
**Issue:** If training plan filters out all targets, training would proceed with empty list, causing confusing errors.

**Fix:** Added early return with clear error message.

**Location:** `TRAINING/orchestration/intelligent_trainer.py:747-755`

### 4. Derived Views Generation Failures
**Issue:** If JSON write fails during derived view generation, entire plan generation would crash.

**Fix:** Added try/except blocks around each view generation section.

**Location:** `TRAINING/orchestration/training_plan_generator.py:_generate_derived_views()`

### 5. Invalid Job Structure
**Issue:** If jobs in plan have missing or invalid fields, derived view generation could crash.

**Fix:** Added validation checks for required fields before processing.

**Location:** `TRAINING/orchestration/training_plan_generator.py:_generate_derived_views()`

### 6. Empty Targets List
**Issue:** If empty targets list is passed to filter functions, could cause issues.

**Fix:** Added validation at start of filter functions.

**Location:** `TRAINING/orchestration/training_plan_consumer.py:filter_targets_by_training_plan()`

### 7. Invalid Model Families Type
**Issue:** If `model_families` in job is not a list, could cause type errors.

**Fix:** Added type validation in `get_model_families_for_job()`.

**Location:** `TRAINING/orchestration/training_plan_consumer.py:get_model_families_for_job()`

## ⚠️ Remaining Edge Cases

### 0. Phase 3 Integration ✅ FIXED
**Status:** Phase 3 (Sequential Model Training) is now integrated with training plan.

**Current Behavior:** 
- ✅ Sequential models can be trained via `IntelligentTrainer` (full integration)
- ✅ Sequential models can be trained via `main.py` with `--training-plan-dir` (newly added)
- ✅ Training plan filtering works for sequential models

**Implementation:**
- Added `--training-plan-dir` argument to `main.py`
- Integrated training plan loading and filtering
- Sequential models (LSTM, Transformer, CNN1D) now respect training plan

### 1. Malformed Training Plan JSON
**Status:** Handled with try/except, but error message could be clearer.

**Current Behavior:** Returns None, logs warning.

**Recommendation:** Add validation function that's called before using plan.

### 2. Training Plan Stale (Older Than Routing Plan)
**Status:** Not currently checked.

**Current Behavior:** Uses whatever plan exists, even if stale.

**Recommendation:** Add timestamp comparison and warning if plan is older than routing plan.

### 3. Symbol Filtering Not Applied in Execution
**Status:** Symbol filtering is computed but not enforced in training execution.

**Current Behavior:** Symbols are filtered and logged, but training still uses all symbols.

**Recommendation:** Wire `filtered_symbols_by_target` into symbol-specific training loops when that feature is implemented.

### 4. Per-Target Families Not Fully Utilized
**Status:** Per-target families are computed but only used if all targets have different families.

**Current Behavior:** If targets have same families, uses global list. If different, uses union (not per-target).

**Recommendation:** Always use per-target families when available, even if they're the same.

### 5. Missing Validation on Training Plan Structure
**Status:** Basic validation exists but not comprehensive.

**Current Behavior:** `validate_training_plan()` exists but isn't called automatically.

**Recommendation:** Call validation after loading plan and log warnings.

## 🔍 Silent Failure Modes

### 1. Training Plan Load Fails Silently
**Location:** `intelligent_trainer.py:659`

**Current Behavior:** If `load_training_plan()` fails, `training_plan` is None, and system falls back to training all targets.

**Impact:** Low - System still works, just without filtering.

**Recommendation:** Log warning more prominently.

### 2. Model Families Filtering Fails Silently
**Location:** `intelligent_trainer.py:709-714`

**Current Behavior:** If `get_model_families_for_job()` returns None or empty list, silently falls back to default families.

**Impact:** Medium - Training proceeds with wrong families.

**Recommendation:** Log warning when families not found in plan.

### 3. Derived Views Generation Fails Silently
**Location:** `training_plan_generator.py:_generate_derived_views()`

**Current Behavior:** If view generation fails, logs warning but continues.

**Impact:** Low - Main plan still generated, just views missing.

**Recommendation:** Current behavior is acceptable.

## 🛡️ Error Handling Strategy

### Graceful Degradation
The system is designed to fail gracefully:
- If training plan missing → train all targets (backward compatible)
- If families not found → use default families
- If views fail → main plan still generated
- If validation fails → log warning but continue

### Logging Levels
- **ERROR:** Critical failures that prevent operation
- **WARNING:** Issues that affect behavior but don't prevent operation
- **INFO:** Normal operation messages
- **DEBUG:** Detailed diagnostic information

### Validation Points
1. **Plan Loading:** Validates JSON structure
2. **Plan Usage:** Validates required fields before use
3. **Filtering:** Validates inputs before filtering
4. **Execution:** Validates filtered results before training

## 📋 Recommendations for Future

1. **Add Plan Validation Hook:** Call `validate_training_plan()` automatically after loading
2. **Add Staleness Check:** Compare timestamps between routing plan and training plan
3. **Add Family Validation:** Warn if families in plan don't match available families
4. **Add Target Validation:** Warn if targets in plan don't match requested targets
5. **Add Symbol Validation:** Warn if symbols in plan don't match available symbols

## ✅ Current Safety Measures

- All file operations wrapped in try/except
- All dict/list access uses `.get()` with defaults
- Empty list/None checks before iteration
- Type validation for critical fields
- Early returns for invalid states
- Clear error messages for failures
- Backward compatibility maintained
