# Additional Centralization Opportunities

## Summary

After the initial centralization, here are additional settings that could be centralized:

## Already Well-Centralized ✅

These are already in their proper config files and don't need duplication:
- `pipeline.data_limits.*` → `CONFIG/training_config/pipeline_config.yaml`
- `preprocessing.validation.*` → `CONFIG/training_config/preprocessing_config.yaml`
- `safety.*` → `CONFIG/training_config/safety_config.yaml`

## Potential Additional Centralizations

### 1. Data Sampling Limits (Lower Priority)

**Current State:**
- `max_samples_per_symbol: 50000` appears in multiple multi-model configs
- Already has default in `pipeline.data_limits.default_max_samples_feature_selection: 50000`
- Code already loads from pipeline_config as fallback

**Recommendation:** 
- Keep as-is (already centralized in pipeline_config)
- The explicit values in multi-model configs are workflow-specific overrides
- Could add to `defaults.yaml` under a `data_limits` section if desired

### 2. Neural Network Training Defaults (Already Partially Done)

**Current State:**
- `epochs: 50` appears in 6/9 neural network configs (66.7% coverage)
- `batch_size: 512` appears in 5/9 neural network configs (55.6% coverage)
- BUT: LSTM uses `epochs: 30, batch_size: 256` (intentional overrides)
- GAN uses `epochs: 100, batch_size: 256` (intentional overrides)
- RewardBased uses `epochs: 100, batch_size: 256` (intentional overrides)

**Recommendation:**
- **Don't centralize** - too much variation, legitimate model-specific needs
- The variations are intentional (LSTM needs smaller batches, GAN needs more epochs)

### 3. Model Family Settings in Multi-Model Configs (Workflow-Specific)

**Current State:**
- `model_families.catboost.config.depth: 6` (100% coverage in multi-model configs)
- `model_families.boruta.config.max_iter: 100` (100% coverage in multi-model configs)
- Many other model-family-specific settings

**Recommendation:**
- **Don't centralize globally** - these are specific to the multi-model feature selection workflow
- They're already consistent across the 3 multi-model config files
- Could create a shared base config for multi-model workflows, but lower priority

### 4. Preset Configurations (Lower Priority)

**Current State:**
- Preset configs (fast, balanced, comprehensive) are duplicated across 3 multi-model configs
- Same structure, same values

**Recommendation:**
- Could create a shared preset config file
- Lower priority - presets are meant to be explicit and visible

## What's Already Centralized ✅

### In `defaults.yaml`:
- ✅ Randomness (`random_state`, `random_seed`)
- ✅ Performance (`n_jobs`, `num_threads`, `device`, `verbose`)
- ✅ Tree models (learning_rate, max_depth, regularization, etc.)
- ✅ Neural networks (dropout, activation, patience)
- ✅ Linear models (max_iter, alpha)
- ✅ Cross-validation (cv_folds, n_jobs)
- ✅ Sampling (validation_split)
- ✅ Aggregation (consensus_threshold, cross_model_method, require_min_models)
- ✅ Output/Compute (save_metadata, use_gpu)
- ✅ SHAP/Explainability (max_samples, use_tree_explainer, etc.)

### In `pipeline_config.yaml`:
- ✅ Data limits (max_samples_per_symbol, max_cs_samples, etc.)
- ✅ Determinism (base_seed, random_state)

### In `preprocessing_config.yaml`:
- ✅ Validation splits (test_size, validation_split)
- ✅ Imputation, scaling, feature selection defaults

### In `safety_config.yaml`:
- ✅ Safety thresholds, clipping, guards

## Remaining Opportunities (Low Priority)

1. **Epochs/Batch Size** - Too much variation, not worth centralizing
2. **Model-Family Settings in Multi-Model** - Workflow-specific, already consistent
3. **Preset Configs** - Could share but lower priority

## Conclusion

**Most important settings are already centralized!** The remaining duplicates are either:
- Workflow-specific (multi-model feature selection)
- Intentionally varied (model-specific needs)
- Already have defaults in pipeline_config (just duplicated for clarity)

The system is now well-centralized. Further centralization would provide diminishing returns.
