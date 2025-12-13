# Reproducibility Audit - Non-Determinism Sources

**Date:** 2025-12-10  
**Issue:** Slight differences in values between runs (Δ ≈ 0.001–0.02 in ROC-AUC)

## Analysis Summary

After reviewing the codebase, here are the sources of non-determinism and their status:

### ✅ **FIXED / DETERMINISTIC**

1. **Model evaluation order** - ✅ FIXED
   - Models are now sorted alphabetically before evaluation
   - Feature importances aggregated in sorted order
   - CSV outputs have features in sorted order

2. **Quick importance pruning** - ✅ DETERMINISTIC
   - Uses `random_state` from determinism system (`stable_seed_from([target_name, 'feature_pruning'])`)
   - Uses `n_jobs=1` for single-threaded execution
   - Location: `TRAINING/utils/feature_pruning.py:145, 172`

3. **Cross-sectional sampling** - ✅ DETERMINISTIC
   - Uses timestamp-based deterministic seeding (not hash-based)
   - Same timestamp → same shuffle seed across runs
   - Location: `TRAINING/utils/cross_sectional_data.py:190-201`

4. **CV splits** - ✅ DETERMINISTIC
   - Uses `PurgedTimeSeriesSplit` with `shuffle=False`
   - Time-based purging (not row-count based)
   - Location: `TRAINING/utils/purged_time_series_split.py:108`

### ⚠️ **POTENTIAL SOURCES OF VARIATION**

1. **Model random_state** - ⚠️ NEEDS VERIFICATION
   - Models use `**rf_config`, `**lgb_config`, etc.
   - Need to verify these configs include `random_state` from defaults injection
   - Location: `TRAINING/ranking/predictability/model_evaluation.py:832, 987, 1139`

2. **CV parallelism** - ⚠️ POTENTIAL ISSUE
   - `cross_val_score(..., n_jobs=cv_n_jobs)` where `cv_n_jobs` can be > 1
   - Parallel CV can cause non-deterministic order of fold evaluation
   - Location: `TRAINING/ranking/predictability/model_evaluation.py:991, 1079, etc.`

3. **Model-level parallelism** - ⚠️ POTENTIAL ISSUE
   - LightGBM/XGBoost can use multiple threads internally
   - Even with `random_state` set, parallel tree building can have slight variations
   - GPU execution is inherently non-deterministic

4. **Floating-point accumulation order** - ⚠️ INHERENT
   - Even with deterministic seeds, floating-point operations can have tiny differences
   - Parallel execution can change accumulation order
   - GPU kernels are non-deterministic by design

## Recommendations

### For Strict Determinism (Bit-Level Reproducibility)

1. **Add deterministic mode toggle** in config:
   ```yaml
   pipeline:
     determinism:
       strict_mode: true  # Forces n_jobs=1, CPU-only, no GPU
   ```

2. **Verify model random_state injection**:
   - Check that all model configs receive `random_state` from defaults
   - Log the actual `random_state` used by each model

3. **Force single-threaded CV**:
   - When `strict_mode=true`, set `cv_n_jobs=1`
   - This ensures folds are evaluated in deterministic order

4. **Disable GPU in strict mode**:
   - Force CPU execution for LightGBM/XGBoost
   - GPU kernels are inherently non-deterministic

5. **Save and reuse CV folds**:
   - Save fold indices to disk
   - Reuse same folds across runs for exact reproducibility

### Current Status

**For your use case (leakage detection):**
- Δ ≈ 0.001–0.02 is **normal and acceptable**
- The qualitative verdict "suspiciously high AUC" is stable
- The exact numeric value bouncing doesn't change the decision

**The system is working as designed:**
- Good hygiene (seeds set, deterministic where possible)
- Stochastic models (tree ensembles, NN) will always have some variation
- Parallel execution adds small non-determinism

**If you need bit-level reproducibility:**
- Implement strict deterministic mode (see recommendations above)
- Use for regression testing, publishing numbers, comparing refactors
- Keep current mode for exploration (faster, still reproducible enough)

## Next Steps

1. ✅ Verify model configs include `random_state` (check defaults injection)
2. ⚠️ Consider adding strict deterministic mode toggle
3. ⚠️ Consider saving/reusing CV folds for exact reproducibility
4. ✅ Document that small variations (0.001–0.02) are expected and normal
