# Config Usage Audit - Ranking Pipeline (INTERNAL)

**Date**: 2025-01-10  
**Purpose**: Verify all config values are loaded from YAML files (single source of truth)  
**Status**: INTERNAL TECHNICAL DOCUMENT - Not for public distribution

## Summary

This audit identifies all places where config values are being **used** (read from config) in the ranking pipeline, and verifies consistency with Phase 3 training.

## Config Loading Patterns Found

### ✅ Properly Loading from Config

1. **Model Hyperparameters** (via `load_model_config()`)
   - `multi_model_feature_selection.py`: LightGBM, XGBoost, MLP
   - `cross_sectional_feature_ranker.py`: LightGBM, XGBoost
   - Pattern: `load_model_config('lightgbm')` → returns hyperparameters directly

2. **Pipeline Data Limits** (via `get_cfg()`)
   - `default_max_samples_feature_selection`: 50000
   - `default_max_samples_ranking`: 10000
   - `default_max_rows_per_symbol_ranking`: 50000
   - `max_cs_samples`: 1000
   - `min_cross_sectional_samples`: 10

3. **Preprocessing Settings** (via `get_cfg()`)
   - `validation.test_size`: 0.2
   - `validation.time_aware_split_ratio`: 0.8
   - `validation.min_samples_for_split`: 10
   - `validation.early_stopping_rounds`: 50
   - `feature_pruning.*`: All thresholds and hyperparameters

4. **Multi-Model Feature Selection** (via `get_cfg()`)
   - `aggregation.*`: All aggregation settings
   - `model_weights.*`: Model weights
   - `rfe.*`: RFE settings
   - `boruta.*`: Boruta hyperparameters
   - `shap.kernel_explainer_sample_size`: 100
   - `random_forest.*`: Random Forest settings
   - `neural_network.validation_fraction`: 0.1

5. **Safety/Leakage Detection** (via `get_safety_config()`)
   - `leakage_detection.auto_fix_thresholds.*`
   - `leakage_detection.model_alerts.*`
   - `leakage_detection.importance.*`
   - `leakage_detection.model_evaluation.*`
   - `leakage_detection.ranking.*`
   - `leakage_detection.auto_rerun.*`

6. **Leakage/Purging** (via `get_cfg()`)
   - `pipeline.leakage.purge_buffer_bars`: 5
   - `pipeline.leakage.purge_time_minutes`: 85

### ⚠️ Potential Gaps

1. **Function Parameter Defaults**
   - `feature_selector.py`: `max_samples_per_symbol: int = 50000`
     - Currently: Only loads from config if `feature_selection_config.max_samples_per_symbol` exists
     - Should: Load from config when default value is used
   
   - `target_ranker.py`: 
     - `min_cs: int = 10` - Should load from config
     - `max_rows_per_symbol: int = 50000` - Should load from config
     - `max_cs_samples: Optional[int] = None` - Already loads from config ✅

2. **Cross-Sectional Ranking Settings**
   - `feature_selector.py`: Reads from `cs_config.get()` with hardcoded defaults:
     - `min_symbols=5`
     - `top_k_candidates=50`
     - `symbol_threshold=0.1`
     - `cs_threshold=0.1`
   - These should be in `preprocessing_config.yaml` under `multi_model_feature_selection.cross_sectional_ranking.*`

## Config Files Used

1. **`CONFIG/model_config/*.yaml`**
   - `lightgbm.yaml`
   - `xgboost.yaml`
   - `mlp.yaml`

2. **`CONFIG/training_config/pipeline_config.yaml`**
   - `pipeline.data_limits.*`
   - `pipeline.leakage.*`
   - `pipeline.determinism.*`

3. **`CONFIG/training_config/preprocessing_config.yaml`**
   - `preprocessing.validation.*`
   - `preprocessing.feature_pruning.*`
   - `preprocessing.multi_model_feature_selection.*`

4. **`CONFIG/training_config/safety_config.yaml`**
   - `safety.leakage_detection.*`

## Consistency with Phase 3

✅ **Consistent**:
- Model hyperparameters: Same `load_model_config()` pattern
- Config file structure: Same YAML files
- Default values: Match Phase 3 defaults when config unavailable

## Updates Completed (2025-01-10)

1. ✅ **Updated function parameter defaults** to load from config when default is used
2. ✅ **Added cross-sectional ranking settings** to `preprocessing_config.yaml`
3. ✅ **Added leakage sentinel thresholds** to `safety_config.yaml`
4. ✅ **Added auto-fixer settings** to `safety_config.yaml`
5. ✅ **Updated all strategies** to use determinism system for `random_state`
6. ✅ **Updated feature pruning** to load all parameters from config
7. ✅ **Updated unified training interface** to load seed and test_size from config

## Remaining Hardcoded Values (Intentional)

The following hardcoded values are **intentional** and appropriate:
- `set_global_determinism(base_seed: int = 42)` - Function parameter default (this function sets the seed)
- Numerical stability constants (1e-6, 1e-10) - Implementation details
- Fallback values in error cases (0.0 scores, etc.) - Appropriate defaults
- Test/experimental files - Not part of production pipeline

## Files Audited & Updated

### Ranking Pipeline
- `TRAINING/ranking/multi_model_feature_selection.py` ✅
- `TRAINING/ranking/cross_sectional_feature_ranker.py` ✅
- `TRAINING/ranking/predictability/model_evaluation.py` ✅
- `TRAINING/ranking/predictability/leakage_detection.py` ✅
- `TRAINING/ranking/predictability/data_loading.py` ✅
- `TRAINING/ranking/predictability/main.py` ✅
- `TRAINING/ranking/feature_selector.py` ✅ (updated)
- `TRAINING/ranking/target_ranker.py` ✅ (updated)

### Core Training Components
- `TRAINING/utils/feature_pruning.py` ✅ (updated)
- `TRAINING/common/leakage_auto_fixer.py` ✅ (updated)
- `TRAINING/common/leakage_sentinels.py` ✅ (updated)
- `TRAINING/unified_training_interface.py` ✅ (updated)
- `TRAINING/utils/data_preprocessor.py` ✅ (updated)
- `TRAINING/strategies/single_task.py` ✅ (updated)
- `TRAINING/strategies/cascade.py` ✅ (updated)
