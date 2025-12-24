# Configuration Audit - TRAINING Pipeline

**Date**: 2025-12-10  
**Status**: ✅ **COMPLETED** - All critical hardcoded values now use config  
**Goal**: Ensure all configurable values come from YAML config files (single source of truth)  
**Reference**: Phase 3 model training (`TRAINING/training_strategies/`) implementation pattern

## Reference Implementation (Phase 3)

Phase 3 uses this pattern:
```python
from config_loader import get_cfg, get_pipeline_config

# Load from config with fallback
value = get_cfg("path.to.config", default=fallback_value, config_name="config_file")
```

### Config Files Used by Phase 3:
- `pipeline_config.yaml` - Main pipeline settings
- `preprocessing_config.yaml` - Data preprocessing
- `system_config.yaml` - System paths and settings
- `threading_config.yaml` - Threading defaults
- `safety_config.yaml` - Safety thresholds

## Hardcoded Values Found

### 1. Ranking Pipeline (`TRAINING/ranking/`)

#### `predictability/model_evaluation.py`
- **Line 293**: `n_estimators=50` (feature pruning) → Should be in `preprocessing_config.yaml`
- **Line 290**: `cumulative_threshold=0.0001` (0.01% importance) → Should be in `preprocessing_config.yaml`
- **Line 291**: `min_features=50` → Should be in `preprocessing_config.yaml`
- **Line 385**: `purge_buffer_bars = 5` → Should be in `pipeline_config.yaml` or `safety_config.yaml`
- **Line 395**: `purge_time = pd.Timedelta(minutes=85)` → Should be in `pipeline_config.yaml`
- **Line 500-504**: `_correlation_threshold = 0.999`, `_suspicious_score_threshold = 0.99` → Should be in `safety_config.yaml`
- **Line 810**: `threshold=0.50` (importance threshold) → Should be in `safety_config.yaml`
- **Line 1951-2072**: `MIN_FEATURES_REQUIRED = 2`, `MIN_FEATURES_FOR_MODEL = 3` → Should be in `safety_config.yaml`
- **Line 188**: `early_stopping_rounds=50` → Should be in model config or `preprocessing_config.yaml`

#### `predictability/leakage_detection.py`
- **Line 475**: `n_estimators=50` (feature pruning) → Should be in `preprocessing_config.yaml`
- **Line 472**: `cumulative_threshold=0.0001` → Should be in `preprocessing_config.yaml`
- **Line 473**: `min_features=50` → Should be in `preprocessing_config.yaml`
- **Line 567**: `purge_buffer_bars = 5` → Should be in `pipeline_config.yaml`
- **Line 577**: `purge_time = pd.Timedelta(minutes=85)` → Should be in `pipeline_config.yaml`
- **Line 682-686**: `_correlation_threshold = 0.999`, `_suspicious_score_threshold = 0.99` → Should be in `safety_config.yaml`
- **Line 992, 1050, 1253**: `threshold=0.50` → Should be in `safety_config.yaml`
- **Line 157-168**: `min_match = 0.999`, `min_corr = 0.999`, `min_valid_pairs = 10` → Should be in `safety_config.yaml`

#### `predictability/data_loading.py`
- **Line 259**: `max_samples: int = 10000` (default) → Should be in `pipeline_config.yaml`

#### `multi_model_feature_selection.py`
- **Line 214**: `'max_samples_per_symbol': 50000` (default config) → Should be in `pipeline_config.yaml`
- **Line 908**: `max_samples: int = 50000` → Should be in `pipeline_config.yaml`
- **Line 266**: `max_samples: int = 1000` (SHAP sampling) → Should be in `pipeline_config.yaml`

#### `cross_sectional_feature_ranker.py`
- **Line 119-132**: Default model configs (n_estimators=100, max_depth=6, learning_rate=0.05) → Should be in model config files
- **Line 201**: `min_cs: int = 10` → Should be in `pipeline_config.yaml`
- **Line 202**: `max_cs_samples: int = 1000` → Should be in `pipeline_config.yaml`

### 2. Model Trainers (`TRAINING/model_fun/`)

#### `comprehensive_trainer.py`
- **Line 95-105**: Hardcoded model params (n_estimators=100, max_depth=6, learning_rate=0.1) → Should be in model config files

#### `feature_pruning.py`
- **Line 104-105, 120-121**: Hardcoded model params (max_depth=5, learning_rate=0.1) → Should be in `preprocessing_config.yaml`

### 3. Common Utilities (`TRAINING/common/`, `TRAINING/utils/`)

#### `utils/data_interval.py`
- **Line 228**: `MAX_REASONABLE_MINUTES = 1440.0` (1 day) → Should be in `pipeline_config.yaml`

#### `utils/feature_pruning.py`
- **Line 104-105**: Hardcoded pruning model params → Should be in `preprocessing_config.yaml`

### 4. Safety Thresholds (Should be in `safety_config.yaml`)

Currently hardcoded in multiple files:
- `MIN_FEATURES_REQUIRED = 2`
- `MIN_FEATURES_FOR_MODEL = 3`
- `MIN_FEATURES_AFTER_LEAK_REMOVAL = 2`
- `correlation_threshold = 0.999`
- `suspicious_score_threshold = 0.99`
- `importance_threshold = 0.50`
- `min_match = 0.999`
- `min_corr = 0.999`
- `min_valid_pairs = 10`

## Proposed Config Structure

### Update `pipeline_config.yaml`
```yaml
pipeline:
  # Data Processing Limits (existing, but add missing)
  data_limits:
    max_samples_per_symbol: null  # Already exists
    max_rows_train: null  # Already exists
    min_cross_sectional_samples: 10  # Already exists
    max_cross_sectional_samples: null  # Already exists
    max_cs_samples: 1000  # NEW: For cross-sectional sampling
    default_max_samples_ranking: 10000  # NEW: Default for ranking
    default_max_samples_feature_selection: 50000  # NEW: Default for feature selection
  
  # Leakage Detection Settings
  leakage:
    purge_buffer_bars: 5  # NEW: Safety buffer for purging
    purge_time_minutes: 85  # NEW: Fallback purge time
    data_interval_default: 5  # NEW: Default data interval (minutes)
  
  # Sequential Model Settings (existing)
  sequential:
    default_lookback: 64  # Already exists
```

### Update `preprocessing_config.yaml`
```yaml
preprocessing:
  # Feature Pruning
  feature_pruning:
    n_estimators: 50  # NEW: For quick pruning models
    max_depth: 5  # NEW: Shallow for speed
    learning_rate: 0.1  # NEW: For pruning models
    cumulative_threshold: 0.0001  # NEW: 0.01% cumulative importance
    min_features: 50  # NEW: Always keep at least 50
  
  # Validation Splits (existing, but verify)
  validation:
    test_size: 0.2  # Already exists
    early_stopping_rounds: 50  # NEW: Default early stopping
```

### Update `safety_config.yaml`
```yaml
safety:
  # Feature Requirements
  min_features:
    required: 2  # NEW: Minimum features required
    for_model: 3  # NEW: Minimum features for model training
    after_leak_removal: 2  # NEW: Minimum after leak removal
  
  # Leakage Detection Thresholds
  leakage:
    correlation_threshold: 0.999  # NEW: For near-copy detection
    suspicious_score_threshold: 0.99  # NEW: For suspicious feature detection
    importance_threshold: 0.50  # NEW: Flag if single feature has >50% importance
    min_match: 0.999  # NEW: Minimum match for near-copy
    min_corr: 0.999  # NEW: Minimum correlation for near-copy
    min_valid_pairs: 10  # NEW: Minimum valid pairs for detection
```

### Create/Update Model Config Files
- `model_config/lightgbm.yaml` - Should have default n_estimators, max_depth, learning_rate
- `model_config/xgboost.yaml` - Should have default params
- Cross-sectional ranker should use these configs

## Implementation Status

### ✅ Phase 1: Safety Thresholds (COMPLETED)
1. ✅ Added safety thresholds to `safety_config.yaml`
2. ✅ Updated `model_evaluation.py` to use `get_cfg("safety.leakage.*")`
3. ✅ Updated `leakage_detection.py` to use `get_cfg("safety.leakage.*")`
4. ✅ Updated all `MIN_FEATURES_*` to use `get_cfg("safety.leakage_detection.ranking.*")`

### ✅ Phase 2: Feature Pruning (COMPLETED)
1. ✅ Added feature pruning config to `preprocessing_config.yaml`
2. ✅ Updated `model_evaluation.py` to use `get_cfg("preprocessing.feature_pruning.*")`
3. ✅ Updated `leakage_detection.py` to use `get_cfg("preprocessing.feature_pruning.*")`
4. ✅ Updated `feature_pruning.py` to use config

### ✅ Phase 3: Data Limits (COMPLETED)
1. ✅ Added missing data limits to `pipeline_config.yaml`
2. ✅ Updated ranking pipeline to use `get_cfg("pipeline.data_limits.*")`
3. ✅ Updated feature selection to use `get_cfg("pipeline.data_limits.*")`
4. ✅ Updated `data_loading.py` to use config defaults

### ✅ Phase 4: Leakage Detection Settings (COMPLETED)
1. ✅ Added leakage settings to `pipeline_config.yaml`
2. ✅ Updated all `purge_buffer_bars` and `purge_time` to use config

### ✅ Phase 5: Model Defaults (COMPLETED)
1. ✅ Updated cross-sectional ranker to load from model configs (`load_model_config('lightgbm')`, `load_model_config('xgboost')`)
2. ✅ Model config files already exist with default params

### ✅ Phase 6: Additional Fixes (COMPLETED)
1. ✅ Updated `data_interval.py` to use `get_cfg("pipeline.data_interval.max_reasonable_minutes")`
2. ✅ Updated `model_evaluation.py` early_stopping_rounds to use `get_cfg("preprocessing.validation.early_stopping_rounds")`
3. ✅ Updated all importance thresholds to use config

## Code Pattern to Follow

### Before:
```python
n_estimators = 50
min_features = 50
threshold = 0.50
```

### After:
```python
# At top of file
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False

# In function/class
if _CONFIG_AVAILABLE:
    n_estimators = get_cfg("preprocessing.feature_pruning.n_estimators", default=50, config_name="preprocessing_config")
    min_features = get_cfg("preprocessing.feature_pruning.min_features", default=50, config_name="preprocessing_config")
    threshold = get_cfg("safety.leakage.importance_threshold", default=0.50, config_name="safety_config")
else:
    n_estimators = 50
    min_features = 50
    threshold = 0.50
```

## Files Updated ✅

### Completed:
1. ✅ `TRAINING/ranking/predictability/model_evaluation.py` - All hardcoded values now use config
2. ✅ `TRAINING/ranking/predictability/leakage_detection.py` - All hardcoded values now use config
3. ✅ `TRAINING/ranking/multi_model_feature_selection.py` - Data limits now use config
4. ✅ `TRAINING/ranking/cross_sectional_feature_ranker.py` - Model configs and data limits use config
5. ✅ `CONFIG/training_config/safety_config.yaml` - Added missing thresholds
6. ✅ `CONFIG/training_config/preprocessing_config.yaml` - Added feature pruning config
7. ✅ `CONFIG/training_config/pipeline_config.yaml` - Added data limits and leakage settings
8. ✅ `TRAINING/ranking/predictability/data_loading.py` - Default max_samples uses config
9. ✅ `TRAINING/utils/feature_pruning.py` - Model params use config
10. ✅ `TRAINING/utils/data_interval.py` - MAX_REASONABLE_MINUTES uses config

### Remaining (Lower Priority):
11. `TRAINING/model_fun/comprehensive_trainer.py` - Hardcoded model params (can use model configs)
12. Other model trainers - Can be updated to use model config files (already exist)

## Testing

After implementation:
1. Verify all values load from config
2. Test with config overrides
3. Verify fallback to defaults works
4. Ensure backward compatibility
