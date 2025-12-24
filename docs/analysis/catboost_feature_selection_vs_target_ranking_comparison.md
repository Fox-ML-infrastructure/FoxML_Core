# CatBoost Feature Selection vs Target Ranking Comparison

**Date**: 2025-12-21  
**Purpose**: Identify why CatBoost takes hours in feature selection but only ~10 minutes in target ranking

## Key Findings

### CRITICAL DIFFERENCE #1: Training Process

**Feature Selection** (`TRAINING/ranking/multi_model_feature_selection.py`):
1. **CV Phase**: Runs `cross_val_score` with `n_splits=3` (line 1828-1844)
   - Uses `PurgedTimeSeriesSplit` with 3 folds
   - If `cv_n_jobs > 1`, runs in parallel
2. **Final Fit Phase**: 
   - Does `train_test_split` (80/20) - line 1851
   - Fits with `eval_set=[(X_val_fit, y_val_fit)]` for early stopping - line 1932
   - **Total fits**: 3 (CV) + 1 (final with eval_set) = **4 fits**

**Target Ranking** (`TRAINING/ranking/predictability/model_evaluation.py`):
1. **CV Phase**: Runs `cross_val_score` with CV folds (line 3596)
   - Uses `PurgedTimeSeriesSplit`
   - Runs in parallel if `cv_n_jobs > 1`
2. **Final Fit Phase**:
   - Fits on **FULL dataset** (X, y) - line 3625
   - **NO train_test_split**
   - **NO eval_set** for early stopping
   - **Total fits**: CV folds + 1 (final on full data) = **4+ fits**

### CRITICAL DIFFERENCE #2: Early Stopping Configuration

**Feature Selection**:
- Config has: `od_type: "Iter"`, `od_wait: 20` (lines 1551-1556)
- Early stopping is **configured and used** in final fit with eval_set
- Iterations capped at 2000 (line 1560)

**Target Ranking**:
- Config has: `iterations: 300` but **NO od_type or od_wait** in config
- Early stopping **may not be configured** for final fit
- Uses default iterations (300)

### CRITICAL DIFFERENCE #3: Early Stopping Usage

**Feature Selection**:
- Final fit uses `eval_set=[(X_val_fit, y_val_fit)]` (line 1932)
- Early stopping **should trigger** if validation doesn't improve
- But if early stopping isn't working, it could run all 2000 iterations

**Target Ranking**:
- Final fit uses `model.fit(X, y)` with **NO eval_set** (line 3625)
- Early stopping **cannot work** without eval_set
- Runs full 300 iterations (but 300 is much less than 2000)

### CRITICAL DIFFERENCE #4: Iterations Cap

**Feature Selection**:
- Config: `iterations: 300`
- **Capped at 2000** for feature selection (line 1560)
- If early stopping fails, could run all 2000 iterations

**Target Ranking**:
- Config: `iterations: 300`
- **No cap applied**
- Runs 300 iterations (or less if early stopping works in CV)

## Hypothesis: Why Feature Selection is Slow

### Primary Suspect: Early Stopping Not Working

**Evidence**:
1. Feature selection configures early stopping (`od_type`, `od_wait`)
2. Feature selection uses `eval_set` in final fit
3. But if early stopping isn't actually triggering, it runs all 2000 iterations
4. 2000 iterations × slow GPU overhead = hours

**Why early stopping might not work**:
- `eval_set` might not be passed correctly through GPU wrapper
- Early stopping params might not be set on the model correctly
- GPU mode might have issues with early stopping

### Secondary Suspect: CV Overhead

**Evidence**:
1. Both stages do CV, but feature selection does CV + train_test_split + fit
2. CV with 3 folds = 3 fits
3. If each CV fold takes 10-20 minutes, that's 30-60 minutes just for CV
4. Then final fit with 2000 iterations = another 1-2 hours

**Why CV might be slow in feature selection**:
- CV might not be using early stopping (cross_val_score doesn't support eval_set)
- Each CV fold might run full iterations without early stopping
- GPU wrapper overhead in CV

### Tertiary Suspect: GPU Wrapper Overhead

**Evidence**:
1. Feature selection uses `CatBoostGPUWrapper` that converts arrays to Pool objects
2. Target ranking also uses wrapper, but might handle it differently
3. Pool conversion overhead × 2000 iterations = significant time

## Comparison Table

| Aspect | Feature Selection | Target Ranking |
|--------|------------------|----------------|
| **CV Usage** | Yes (3 folds) | Yes (CV folds) |
| **CV Early Stopping** | No (cross_val_score doesn't support) | No (cross_val_score doesn't support) |
| **Final Fit** | train_test_split (80/20) + eval_set | Full dataset, no eval_set |
| **Early Stopping Config** | od_type: "Iter", od_wait: 20 | Not in config |
| **Early Stopping Used** | Yes (eval_set in final fit) | No (no eval_set) |
| **Max Iterations** | 2000 (capped) | 300 (config default) |
| **GPU Wrapper** | Yes (CatBoostGPUWrapper) | Yes (CatBoostGPUWrapper) |
| **Total Fits** | 3 (CV) + 1 (final) = 4 | CV folds + 1 = 4+ |

## Recommended Fixes

### Fix 1: Skip CV in Feature Selection (Like Target Ranking Does for Final Fit)

**Option A**: Skip CV entirely for feature selection (use single fit like target ranking)
**Option B**: Keep CV but skip final fit (use CV results only)

### Fix 2: Ensure Early Stopping Works

- Verify `eval_set` is passed correctly through GPU wrapper
- Verify early stopping params are set on model before fit
- Add logging to confirm early stopping triggers

### Fix 3: Reduce Iterations Further

- If early stopping works, 2000 iterations should be fine
- If early stopping doesn't work, reduce to 300 (match target ranking)

### Fix 4: Match Target Ranking Approach

- Remove train_test_split
- Fit on full dataset like target ranking
- Remove eval_set (or verify it's needed for early stopping)

## Next Steps

1. Add timing logs to measure CV vs final fit duration
2. Verify early stopping actually triggers (log iteration count)
3. Test hypothesis: Apply target ranking's approach to feature selection
4. If CV is bottleneck, skip CV or reduce folds

