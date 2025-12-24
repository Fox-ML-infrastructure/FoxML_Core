# Configuration Locations Audit

**Purpose:** Track all configuration settings scattered throughout the TRAINING directory for future centralization.

**Status:** Documentation only — no code changes yet.

**Last Updated:** 2025-12-06

---

## Overview

This document catalogs all configuration values, default settings, hardcoded parameters, and environment variables found in the TRAINING directory. This audit will inform the future centralized configuration system (Phase 2 of roadmap).

---

## 1. Model-Specific Configurations

### 1.1 Model Trainer Defaults

All model trainers have hardcoded defaults in their `__init__` methods. These are marked as "DEPRECATED" and should eventually load from `CONFIG/model_config/{family}.yaml`.

#### Location Pattern: `TRAINING/model_fun/{family}_trainer.py`

**LightGBM** (`lightgbm_trainer.py`):
- `num_leaves`: 96
- `max_depth`: 8
- `min_data_in_leaf`: 200
- `min_child_weight`: 0.5
- `feature_fraction`: 0.75
- `bagging_fraction`: 0.75
- `bagging_freq`: 1
- `lambda_l1`: 0.1
- `lambda_l2`: 0.1
- `learning_rate`: 0.03
- `n_estimators`: 1000
- `early_stopping_rounds`: 50
- `num_threads`: from `OMP_NUM_THREADS` env or 4

**XGBoost** (`xgboost_trainer.py`):
- `max_depth`: 7
- `min_child_weight`: 0.5
- `subsample`: 0.75
- `colsample_bytree`: 0.75
- `gamma`: 0.3
- `reg_alpha`: 0.1
- `reg_lambda`: 0.1
- `eta` (learning_rate): 0.03
- `n_estimators`: 1000
- `early_stopping_rounds`: 50
- `num_threads`: from `OMP_NUM_THREADS` env or 4

**MLP** (`mlp_trainer.py`):
- `epochs`: 50
- `batch_size`: 512
- `hidden_layers`: [256, 128] (or `hidden` key)
- `dropout`: 0.2
- `learning_rate`: 1e-3
- `patience`: 10

**VAE** (`vae_trainer.py`):
- `epochs`: 50
- `batch_size`: 512
- `latent_dim`: 16 (or `z_dim` key)
- `hidden_dim`: 128
- `dropout`: 0.2
- `learning_rate`: 1e-3
- `beta`: 1.0 (KL weight)
- `patience`: 10

**GAN** (`gan_trainer.py`):
- `epochs`: 50
- `batch_size`: 512
- `generator_hidden_dim`: 256 (or `hidden_dim` key)
- `dropout`: 0.2
- `learning_rate_generator`: 1e-3 (or `learning_rate` key)
- `patience`: 10

**MetaLearning** (`meta_learning_trainer.py`):
- `epochs`: 50
- `batch_size`: 512
- `hidden_dim`: 128
- `dropout`: 0.2
- `learning_rate`: 1e-3
- `patience`: 10

**LSTM** (`lstm_trainer.py`):
- `epochs`: 30 (reduced from 50)
- `batch_size`: 256 (reduced from 512)
- `lstm_units`: 128 (or `units` key)
- `dropout`: 0.2
- `recurrent_dropout`: 0.1 (reduced from 0.2)
- `learning_rate`: 1e-3
- `patience`: 5 (reduced from 10)

**Transformer** (`transformer_trainer.py`):
- `epochs`: 50
- `batch_size`: 128 (reduced from 512, dynamically adjusted for large sequences)
- `d_model`: 128
- `heads`: 4 (reduced from 8)
- `ff_dim`: 256
- `dropout`: 0.1
- `learning_rate`: 1e-3
- `patience`: 10
- **Dynamic batch scaling**: For sequences > 200 features, batch size scales down automatically

**CNN1D** (`cnn1d_trainer.py`):
- `epochs`: 30 (reduced from 50)
- `batch_size`: 256 (reduced from 512)
- `filters`: [64, 64]
- `dropout`: 0.2
- `learning_rate`: 1e-3
- `patience`: 10

**TabCNN, TabLSTM, TabTransformer**:
- These are aliases for CNN1D, LSTM, Transformer respectively
- Use same configs as their base models

**MultiTask** (`multi_task_trainer.py`):
- `epochs`: 50
- `batch_size`: 512
- `hidden_dim`: 256 (shared hidden layer size)
- `dropout`: 0.2
- `learning_rate`: 3e-4 (1e-4 to 5e-4 range)
- `patience`: 10
- `use_multi_head`: None (auto-detect from y shape)
- `loss_weights`: None (dict mapping target names to weights)
- `target_names`: None (list of target names for multi-head)

**Ensemble** (`ensemble_trainer.py`):
- `num_threads`: from `OMP_NUM_THREADS` env or 12
- **HGB config:**
  - `hgb_max_iter`: 300
  - `hgb_max_depth`: 8
  - `hgb_learning_rate`: 0.05
  - `hgb_max_bins`: 255
  - `hgb_l2`: 1e-4
  - `hgb_early_stop`: True
- **RF config:**
  - `rf_n_estimators`: 300
  - `rf_max_depth`: 15
  - `rf_max_samples`: 0.7
  - `rf_max_features`: "sqrt"
- **Ridge config:**
  - `ridge_alpha`: 1.0
- **Stacking config:**
  - `use_stacking`: True
  - `stacking_cv`: 5
  - `final_estimator_alpha`: 1.0

**NGBoost** (`ngboost_trainer.py`):
- `n_estimators`: 700
- `learning_rate`: 0.03
- `clip`: 10.0
- `early_stopping_rounds`: 50
- `col_sample`: 0.6 (default in code)
- `seed`: 42 (default in code)

**GMMRegime** (`gmm_regime_trainer.py`):
- `n_components`: 3
- `reg_covar`: 1e-4
- `covariance_type`: "diag"
- `max_iter`: 100
- `ridge_alpha`: 1.0

**ChangePoint** (`change_point_trainer.py`):
- `n_regimes`: 2
- **Thread allocation**: Uses `max(12, n_threads - 2)` for BLAS-heavy operations

**FTRLProximal** (`ftrl_proximal_trainer.py`):
- `learning_rate`: 0.0001 (or `alpha` key)
- `l1_regularization_strength`: 0.15 (or `l1_ratio` key)
- `max_iter`: 2000
- `tol`: 1e-4
- `learning_rate_schedule`: "optimal"

**RewardBased** (`reward_based_trainer.py`):
- `alpha`: 1.0
- `reward_power`: 1.0 (weights ~ |y|^p)
- `max_iter`: 1000
- `tol`: 1e-4

**QuantileLightGBM** (`quantile_lightgbm_trainer.py`):
- `num_threads`: from `OMP_NUM_THREADS` env or 4 (clamped to 4-8 range)
- `alpha`: 0.5
- `learning_rate`: 0.05
- `n_estimators`: 2000
- `num_leaves`: 64
- `max_depth`: -1
- `min_data_in_leaf`: 2048
- `feature_fraction`: 0.7
- `bagging_fraction`: 0.7
- `lambda_l1`: 0.0
- `lambda_l2`: 2.0
- `early_stopping_rounds`: 100
- `max_bin`: 63
- `time_budget_sec`: 1800 (30 min default)

---

## 2. Training Pipeline Configurations

### 2.1 Main Training Script (`train_with_strategies.py`)

**Threading:**
- `DEFAULT_THREADS`: `max(1, (os.cpu_count() or 2) - 1)` (line 477)
- `THREADS`: from env `THREADS` or `DEFAULT_THREADS` (line 413)
- `MKL_THREADS_DEFAULT`: 1 (line 414)
- Thread allocation per family via `child_env` dict
- `OPENBLAS_NUM_THREADS`: "1" (hardcoded)
- `NUMEXPR_NUM_THREADS`: "1" (hardcoded)

**Data Processing:**
- `MAX_SPS` / `max_samples_per_symbol`: 20 (in test scripts), varies
- `EPOCHS`: 10 (in test scripts), 50-100 (defaults)
- `--epochs` CLI arg: default 50, help text suggests 1000 for production (line 2053)
- `min_cs`: 1 (minimum cross-sectional samples)
- `max_cs_samples`: None (max cross-sectional samples)
- `max_rows_train`: None (max training rows)

**Polars Settings:**
- `USE_POLARS`: from env `USE_POLARS` or "1"
- `POLARS_MAX_THREADS`: set to `DEFAULT_THREADS`
- `CS_ALIGN_MODE`: "union" (default)

**Determinism:**
- `PYTHONHASHSEED`: "42"
- `TF_DETERMINISTIC_OPS`: "1"
- `TF_CPP_MIN_LOG_LEVEL`: Removed (was "3", now shows warnings) (line 39)
- `TF_LOGGING_VERBOSITY`: Removed (was "ERROR", now shows warnings) (line 40)

**Model Selection:**
- `SEQUENTIAL_MODELS`: ['CNN1D', 'LSTM', 'Transformer', 'TabCNN', 'TabLSTM', 'TabTransformer']
- `CROSS_SECTIONAL_MODELS`: All other models
- `SEQ_BACKEND`: "torch" (default), can be "tf"

**Family Capabilities Map** (`FAMILY_CAPS`):
- Hardcoded dictionary with `nan_ok`, `needs_tf`, `backend`, `experimental`, `preprocess_in_family` flags
- Location: `train_with_strategies.py` line ~543

**Isolation Runner Settings:**
- `timeout_s`: 7200 (2 hours) default for `_run_family_isolated()` (line 222)
- Used for child process timeout detection

**Test Configuration:**
- `TEST_TARGETS`: ['fwd_ret_5m', 'fwd_ret_15m', 'mdd_5m_0.001', 'will_peak_5m']
- `MAX_SPS`: 20 (reduced for testing)
- `EPOCHS`: 10 (reduced for testing)

**Path & Environment:**
- `_JOBLIB_TMP`: `Path.home() / "trainer_tmp" / "joblib"` (line 77)
- `JOBLIB_TEMP_FOLDER`: Set to `_JOBLIB_TMP` (line 79)
- `PYTHONPATH`: Set to project root (line 50)
- `CONDA_PREFIX`: Used for CUDA library paths (lines 24-36)

---

## 3. Threading & Resource Management

### 3.1 Thread Configuration (`common/threads.py`)

**Default Threads:**
- `default_threads()`: `max(1, (os.cpu_count() or 8) - 1)`
- Used as fallback when `THREADS` env not set

**Environment Variables:**
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `OMP_DYNAMIC`: "false"
- `OMP_PROC_BIND`: "FALSE"
- `GOMP_CPU_AFFINITY`
- `KMP_BLOCKTIME`: "0"
- `MKL_THREADING_LAYER`: "GNU"

**Thread Planning:**
- `plan_for_family()`: Calculates optimal OMP/MKL threads per family
- `plan_for_estimator()`: Thread planning for specific estimators
- Location: `common/threads.py`

### 3.2 Family Thread Policy (`common/family_config.py`)

**Thread Policies:**
- `thread_policy`: "omp_heavy", "blas_heavy", "balanced", etc.
- Per-family configuration
- Location: `common/family_config.py`

---

## 4. GPU & CUDA Configuration

### 4.1 GPU Settings

**Environment Variables:**
- `CUDA_VISIBLE_DEVICES`: "0" (default), "-1" (hide GPU)
- `TF_GPU_ALLOCATOR`: "cuda_malloc_async"
- `TF_FORCE_GPU_ALLOW_GROWTH`: "true"
- `TF_NUM_INTRAOP_THREADS`: 1
- `TF_NUM_INTEROP_THREADS`: 1
- `TF_VISIBLE_DEVICE_LIST`: mirrors `CUDA_VISIBLE_DEVICES`
- `TF_CPP_MIN_LOG_LEVEL`: varies ("1", "2", "3")
- `TF_ENABLE_ONEDNN_OPTS`: can be "0" to disable

**GPU Memory Caps:**
- `cap_vram_mb`: 4096 (for TensorFlow GPU families)
- Location: `common/runtime_policy.py`

**GPU Detection:**
- `needs_gpu`: True/False per family
- Location: `common/runtime_policy.py`

---

## 5. Runtime Policy Configuration

### 5.1 Family Runtime Policies (`common/runtime_policy.py`)

**Policy Attributes:**
- `run_mode`: "process" (isolated) or "inproc" (parent)
- `needs_gpu`: True/False
- `backends`: {"tf"}, {"torch"}, {"xgb"}, etc.
- `omp_user_api`: "openmp", "blas", or None
- `cap_vram_mb`: VRAM limit in MB (4096 for TF GPU families)
- `force_isolation_reason`: String explaining isolation requirement

**Per-Family Policies:**
- Defined in `CROSS_SECTIONAL_POLICIES` and `SEQUENTIAL_POLICIES` dictionaries
- Location: `common/runtime_policy.py` lines ~71-250

---

## 6. Memory Management

### 6.1 Memory Settings (`memory/memory_manager.py`)

**Memory Limits:**
- `memory_threshold`: 0.8 (80% memory usage threshold) - line 38
- `chunk_size`: 1000000 (1M rows per chunk) - line 39
- `aggressive_cleanup`: True (default) - line 40
- High memory warning thresholds:
  - System memory: 90% (line 70)
  - Process RSS: 50GB (lines 72, 79)
- Location: `memory/memory_manager.py`

**Memory Cleanup:**
- `aggressive_cleanup()`: Function for memory cleanup
- `monitor_memory()`: Memory monitoring
- Location: `memory/memory_manager.py`

---

## 7. Data Processing Configuration

### 7.1 Cross-Sectional Processing (`processing/cross_sectional.py`)

**Sampling:**
- `max_samples_per_symbol`: 20 (default in test scripts)
- Cross-sectional alignment mode
- Polars optimization settings

### 7.2 Sequential Processing

**Lookback Windows:**
- Default lookback: 64 (hardcoded in `build_sequences_from_features()`)
- CLI arg `--seq-lookback`: default 64 (line 2051)
- Location: `train_with_strategies.py` lines 577, 1402, 2051

**Validation Splits (Sequential Dataset):**
- `val_ratio`: 0.15 (15% validation) - `datasets/seq_dataset.py` line 215
- `test_ratio`: 0.15 (15% test) - `datasets/seq_dataset.py` line 216
- Location: `datasets/seq_dataset.py`

**Reshaping:**
- Sequential data reshaping parameters
- Check sequential trainer files

---

## 8. Preprocessing Configuration

### 8.1 Data Preprocessing

**Imputation:**
- Strategy settings (median, mean, etc.)
- Location: `utils/data_preprocessor.py`

**Scaling:**
- StandardScaler, MinMaxScaler settings
- Location: `utils/data_preprocessor.py` and trainer files

**Feature Selection:**
- `n_features`: 50 (in feature selection config)
- `min_importance`: 0.001
- Location: `EXPERIMENTS/phase1_feature_engineering/feature_selection_config.yaml`

**Validation Splits:**
- `test_size`: 0.2 (20% test split) - hardcoded in most trainers
- `random_state`: 42 - hardcoded in most trainers
- Location: All trainer files using `train_test_split()`

**NaN Thresholds:**
- `nan_ratio > 0.1` (10% NaN threshold) - `utils/validation.py` lines 101, 150, 235
- Used for data validation warnings

**Time-Aware Splits:**
- `train_ratio`: 0.8 (80% training) - `utils/core_utils.py` line 119
- Location: `utils/core_utils.py`

---

## 9. Environment Variables Summary

### 9.1 Threading
- `THREADS`: Main thread count
- `OMP_NUM_THREADS`: OpenMP threads
- `MKL_NUM_THREADS`: MKL/BLAS threads
- `OPENBLAS_NUM_THREADS`: OpenBLAS threads
- `NUMEXPR_NUM_THREADS`: NumExpr threads (usually 1)
- `OMP_DYNAMIC`: "false"
- `OMP_PROC_BIND`: "FALSE"
- `GOMP_CPU_AFFINITY`: CPU affinity string
- `KMP_BLOCKTIME`: "0"
- `MKL_THREADING_LAYER`: "GNU"

### 9.2 GPU/CUDA
- `CUDA_VISIBLE_DEVICES`: GPU visibility
- `TF_GPU_ALLOCATOR`: "cuda_malloc_async"
- `TF_FORCE_GPU_ALLOW_GROWTH`: "true"
- `TF_NUM_INTRAOP_THREADS`: 1
- `TF_NUM_INTEROP_THREADS`: 1
- `TF_VISIBLE_DEVICE_LIST`: mirrors CUDA_VISIBLE_DEVICES
- `TF_CPP_MIN_LOG_LEVEL`: "1", "2", or "3"
- `TF_ENABLE_ONEDNN_OPTS`: can be "0"

### 9.3 TensorFlow/PyTorch Control
- `TRAINER_CHILD_NO_TF`: "0" or "1"
- `TRAINER_CHILD_NO_TORCH`: "0" or "1"
- `TRAINER_CHILD_FAMILY`: Family name for child process
- `TRAINER_CHILD_FORCE_OMP`: Override OMP threads
- `TRAINER_NO_ISOLATION`: "0" or "1"
- `TRAINER_FORCE_ISOLATION_FOR`: Comma-separated family list
- `TRAINER_GPU_IDS`: GPU IDs to use
- `TRAINER_TF_LOG_LEVEL`: TensorFlow log level

### 9.4 Data Processing
- `USE_POLARS`: "1" or "0"
- `POLARS_MAX_THREADS`: Polars thread limit
- `CS_ALIGN_MODE`: "union" (default)
- `PYTHONHASHSEED`: "42"
- `TF_DETERMINISTIC_OPS`: "1"

### 9.5 System/Shell
- `SHELL`: "/usr/bin/bash"
- `TERM`: "dumb"
- `INPUTRC`: "/dev/null"
- `JOBLIB_START_METHOD`: "spawn"
- `JOBLIB_TEMP_FOLDER`: Temp directory path
- `TRAINER_TMP`: Temp directory override
- `TRAINING_TMPDIR`: Temp directory override

---

## 10. Hardcoded Constants

### 10.1 Training Scripts

**`train_with_strategies.py`:**
- `DEFAULT_THREADS`: Calculated from CPU count
- `SEQUENTIAL_MODELS`: List of 6 sequential models
- `CROSS_SECTIONAL_MODELS`: List of cross-sectional models
- `FAMILY_CAPS`: Large dictionary of family capabilities
- `MODMAP`: Module mapping dictionary (two instances)
- `TF_FAMS`, `TORCH_FAMS`, `CPU_FAMS`: Family classifications
- `RISKY_MKL_FAMILIES`: {"ChangePoint", "GMMRegime"}

**Test Scripts:**
- `test_gpu_models.sh`: `MAX_SPS=20`, `EPOCHS=10`
- `train_all_symbols.sh`: Various batch sizes and limits

### 10.2 Timeouts & Limits

**Isolation Runner:**
- `timeout_s`: 7200 (2 hours) default
- Memory cap enforcement
- Location: `common/isolation_runner.py` and `train_with_strategies.py` line 222 (`_run_family_isolated` function)

**Memory Limits:**
- Check `memory_manager.py` for specific limits

**Sequential Dataset:**
- Default `batch_size`: 32 (in `datasets/seq_dataset.py` lines 176, 217)
- Used for PyTorch sequential model dataloaders

---

## 11. Configuration Files

### 11.1 Existing Config Files

**YAML Configs:**
- `EXPERIMENTS/phase1_feature_engineering/feature_selection_config.yaml`
  - Feature selection settings
  - Feature engineering settings (VAE, GMM)
  - LightGBM configuration

**Python Config:**
- `common/family_config.py`: Family thread policies and configurations

### 11.2 Centralized Config (Future)

**Target Location:** `CONFIG/model_config/{family}.yaml`
- Currently referenced but not always used
- Some trainers check `_USE_CENTRALIZED_CONFIG` flag
- Loaded via `config_loader.load_model_config()`

---

## 12. Strategy-Specific Configuration

### 12.1 Single Task Strategy (`strategies/single_task.py`)

**Settings:**
- Check file for strategy-specific configs

### 12.2 Multi Task Strategy (`strategies/multi_task.py`)

**Settings:**
- Check file for multi-task specific configs

### 12.3 Cascade Strategy (`strategies/cascade.py`)

**Settings:**
- Check file for cascade-specific configs

---

## 13. Preprocessing Configuration

### 13.1 Data Preprocessor (`utils/data_preprocessor.py`)

**Settings:**
- Imputation strategies
- Scaling methods
- Feature selection thresholds
- NaN handling

### 13.2 Sequential Preprocessor

**Settings:**
- Lookback window sizes
- Sequence building parameters
- Location: `preprocessing/mega_script_sequential_preprocessor.py`

---

## 14. Validation & Testing Configuration

### 14.1 Validation Settings (`utils/validation.py`)

**Settings:**
- Validation split ratios
- Cross-validation folds
- Test size defaults

### 14.2 Test Configuration

**Test Scripts:**
- `test_gpu_models.sh`: Test-specific settings
- `train_all_symbols.sh`: Batch processing settings

---

## 15. Logging Configuration

### 15.1 Log Levels

**Settings:**
- TensorFlow log levels: `TF_CPP_MIN_LOG_LEVEL`
- Python logging levels
- Location: Various files

---

## 16. File Paths & Directories

### 16.1 Data Paths

**Settings:**
- `DATA_DIR`: Data directory path
- Default: `data/data_labeled/interval=5m`
- Location: Scripts and `train_with_strategies.py`

### 16.2 Output Paths

**Settings:**
- `output_dir`: Model output directory
- Default patterns: `test_output_*`, `mtf_*`
- Location: `train_with_strategies.py`

### 16.3 Temp Paths

**Settings:**
- `TRAINER_TMP`: Temp directory
- `TRAINING_TMPDIR`: Temp directory override
- `JOBLIB_TEMP_FOLDER`: Joblib temp folder
- Default: `/tmp`

---

## 17. Model-Specific Special Settings

### 17.1 Ensemble Trainer (`ensemble_trainer.py`)

**Settings:**
- Base model configurations (HGB, RF, Ridge)
- Stacking parameters
- Ridge alpha
- RandomForest settings
- **Purge Overlap:** 17 bars (default for 60m target with 5m bars) - line 186
- Location: `model_fun/ensemble_trainer.py`

### 17.2 GMM Regime Trainer (`gmm_regime_trainer.py`)

**Settings:**
- Number of components
- Covariance type
- Location: `model_fun/gmm_regime_trainer.py`

### 17.3 Change Point Trainer (`change_point_trainer.py`)

**Settings:**
- Change point detection parameters
- Location: `model_fun/change_point_trainer.py`

---

## 18. Callback & Training Loop Settings

### 18.1 Early Stopping

**Settings:**
- `patience`: 10 (default for most models)
- `restore_best_weights`: True
- Location: Trainer files

### 18.2 Learning Rate Scheduling

**Settings:**
- `ReduceLROnPlateau`: patience=5, factor=0.5, min_lr=1e-6
- Location: Trainer files

### 18.3 Mixed Precision

**Settings:**
- Enabled for Ampere GPUs (compute capability 8.6+)
- Policy: "mixed_float16"
- Location: Trainer files (LSTM, etc.)

### 18.4 Gradient Clipping

**Settings:**
- `clipnorm`: 1.0 (hardcoded in all TensorFlow/Keras trainers)
- `max_norm`: 1.0 (hardcoded in PyTorch trainers)
- Location: All TensorFlow/Keras trainers (LSTM, Transformer, CNN1D, MetaLearning)
- Location: `seq_torch_base.py` lines 131, 138

### 18.5 Optimizer Defaults

**Settings:**
- Adam optimizer: `clipnorm=1.0` (TensorFlow)
- AdamW optimizer: `lr=1e-3`, `weight_decay=0.0` (PyTorch)
- Location: Trainer files and `seq_torch_base.py`

---

## 19. Security & Safety Settings

### 19.1 Readline Suppression

**Settings:**
- `SHELL`: "/usr/bin/bash"
- `TERM`: "dumb"
- `INPUTRC`: "/dev/null"
- Location: `common/threads.py`, `common/isolation_runner.py`

### 19.2 MKL Guard

**Settings:**
- `MKL_THREADING_LAYER`: "SEQUENTIAL" (for risky families)
- `SKLEARN_RIDGE_SOLVER`: "lsqr" (for risky families)
- Location: `common/isolation_runner.py`

### 19.3 Safety Thresholds (`common/safety.py`)

**Settings:**
- Feature clipping: `clip=1e3` (line 26) - clips features to [-1000, 1000]
- Target capping: `cap_sigma=15.0` (line 33) - caps targets using 15 MAD
- Safe exponential bounds: `lo=-40.0`, `hi=40.0` (line 75)
- Location: `common/safety.py`

### 19.4 Memory Cap Settings

**Settings:**
- `TRAINER_CHILD_MEMCAP_GB`: Memory cap in GB (default: "0" = disabled)
- `TRAINER_CHILD_MEMCAP_DISABLE`: "1" to disable cap check
- Location: `common/isolation_runner.py` lines 303-306, 315-340

### 19.5 VRAM Caps

**Settings:**
- `cap_vram_mb`: 4096 (4GB) for all TensorFlow GPU families
- Location: `common/runtime_policy.py` - hardcoded in all GPU family policies

---

## 20. Recommendations for Centralization

### 20.1 High Priority (Frequently Changed)

1. **Model Hyperparameters**
   - All trainer defaults should move to YAML
   - Currently partially implemented but not consistently used

2. **Threading Configuration**
   - Default thread counts
   - Per-family thread policies
   - OMP/MKL thread allocations

3. **GPU Settings**
   - CUDA_VISIBLE_DEVICES defaults
   - VRAM caps
   - TensorFlow GPU settings

4. **Data Processing**
   - Max samples per symbol
   - Batch sizes
   - Epochs
   - Cross-sectional sampling limits

### 20.2 Medium Priority (Occasionally Changed)

1. **Memory Management**
   - Memory caps
   - Cleanup thresholds
   - Batch size calculations

2. **Preprocessing**
   - Imputation strategies
   - Scaling methods
   - Feature selection thresholds

3. **Training Loop**
   - Early stopping patience
   - Learning rate schedules
   - Callback configurations

### 20.3 Low Priority (Rarely Changed)

1. **System Settings**
   - Shell/terminal settings
   - Temp directory paths
   - Log levels

2. **Family Classifications**
   - Model family lists
   - Capability maps
   - Runtime policies

---

## 21. Files Requiring Config Extraction

### 21.1 High Priority Files

1. `train_with_strategies.py` - Main training script with many hardcoded values
2. `common/threads.py` - Threading configuration
3. `common/runtime_policy.py` - Runtime policies
4. All `model_fun/*_trainer.py` files - Model defaults
5. `common/family_config.py` - Family configurations
6. `memory/memory_manager.py` - Memory settings

### 21.2 Medium Priority Files

1. `processing/cross_sectional.py` - Cross-sectional processing
2. `utils/data_preprocessor.py` - Preprocessing settings
3. `common/isolation_runner.py` - Isolation settings
4. Test scripts - Test-specific configurations

### 21.3 Low Priority Files

1. Strategy files - Strategy-specific settings
2. Utility files - Various utility settings

---

## 22. Configuration Categories

### 22.1 Model Hyperparameters
- Learning rates
- Batch sizes
- Epochs
- Hidden dimensions
- Dropout rates
- Regularization parameters
- Early stopping settings

### 22.2 System Resources
- Thread counts
- GPU settings
- Memory limits
- CPU affinity

### 22.3 Data Processing
- Sampling limits
- Batch sizes
- Cross-sectional settings
- Sequential settings

### 22.4 Training Pipeline
- Timeouts
- Isolation settings
- Process management
- Error handling

### 22.5 Preprocessing
- Imputation
- Scaling
- Feature selection
- Data cleaning

---

## 23. Next Steps

1. **Create Configuration Schema**
   - Define YAML structure for all config categories
   - Add validation layer
   - Create example templates

2. **Extract Hardcoded Values**
   - Systematically move values to config files
   - Maintain backward compatibility during transition
   - Update all references

3. **Implement Config Loader**
   - Enhance existing `config_loader.py`
   - Add validation
   - Add environment variable overrides
   - Add command-line argument support

4. **Documentation**
   - Create configuration guide
   - Document all available settings
   - Provide examples

---

## Notes

- Many trainers already have infrastructure for centralized config (`_USE_CENTRALIZED_CONFIG` flag)
- Some configs are environment-specific and should remain as env vars
- Some configs are runtime-calculated and may not need centralization
- Priority should be on user-facing, frequently-changed settings

---

## 24. Additional Config Settings Found

### 24.1 Callback Defaults (Common Across Trainers)

**Early Stopping:**
- `patience`: Varies by model (5-10)
- `restore_best_weights`: True (hardcoded in all TensorFlow trainers)
- `monitor`: "val_loss" (implicit default)

**Learning Rate Reduction:**
- `patience`: 5 (hardcoded in all trainers)
- `factor`: 0.5 (hardcoded)
- `min_lr`: 1e-6 (hardcoded)
- Location: All TensorFlow/Keras trainers

### 24.2 Transformer Dynamic Batch Scaling

**Location:** `TRAINING/model_fun/transformer_trainer.py` lines 89-99

**Logic:**
- Base batch size: 128
- For sequences > 200 features: `batch_size = max(32, base_batch_size * (200 / seq_len))`
- Prevents OOM errors with large attention matrices
- Logs warning when batch size is reduced

### 24.3 QuantileLightGBM Thread Clamping

**Location:** `TRAINING/model_fun/quantile_lightgbm_trainer.py` lines 95-98

**Logic:**
- Threads clamped to 4-8 range: `max(1, min(8, int(t)))`
- Optimized for quantile regression performance

### 24.4 ChangePoint BLAS Thread Allocation

**Location:** `TRAINING/model_fun/change_point_trainer.py` line 65

**Logic:**
- Uses `max(12, n_threads - 2)` for BLAS-heavy operations
- Applied to KMeans and Ridge fits

### 24.5 Ensemble Thread Allocation

**Location:** `TRAINING/model_fun/ensemble_trainer.py` lines 129-132

**Logic:**
- `hgb_omp`: T (all threads for HGB)
- `rf_jobs`: T (all threads for RF joblib workers)
- `rf_omp`: 1 (no OpenMP in RF to avoid oversubscription)

### 24.6 Joblib Temp Directory

**Location:** `TRAINING/train_with_strategies.py` lines 77-79

**Settings:**
- `_JOBLIB_TMP`: `Path.home() / "trainer_tmp" / "joblib"`
- Created with `mkdir(parents=True, exist_ok=True)`
- Set via `JOBLIB_TEMP_FOLDER` environment variable

### 24.7 CUDA Library Path Setup

**Location:** `TRAINING/train_with_strategies.py` lines 22-36

**Logic:**
- Checks `CONDA_PREFIX` environment variable
- Adds `{CONDA_PREFIX}/lib` to `LD_LIBRARY_PATH`
- Adds `{CONDA_PREFIX}/targets/x86_64-linux/lib` to `LD_LIBRARY_PATH`
- Must happen before TensorFlow imports

### 24.8 Sequential Dataset Defaults

**Location:** `TRAINING/datasets/seq_dataset.py` lines 176, 217

**Settings:**
- Default `batch_size`: 32 (for PyTorch dataloaders)
- Used in `create_seq_dataloader()` function

### 24.9 Random State & Seed Values

**Common Values:**
- `random_state`: 42 (hardcoded in most trainers and splits)
- `base_seed`: 42 (in `train_with_strategies.py` line 450, `train_crypto_models.py` line 34)
- `BASE_SEED`: 1234 (in `common/determinism.py` line 31)
- `seed`: 42 (fallback in `common/determinism.py` line 439)
- Location: Throughout codebase

### 24.10 Cross-Sectional Sampling Defaults

**Location:** `TRAINING/train_with_strategies.py` lines 648-657

**Settings:**
- `min_cs`: 10 (minimum cross-sectional samples) - line 648
- `max_cs_samples`: None (default), falls back to 1000 (aggressive sampling) - line 657
- Used in `_prepare_training_data()` function

### 24.11 Cascade Strategy Configs

**Location:** `TRAINING/strategies/cascade.py`

**Settings:**
- `random_state`: 42 (hardcoded in RandomForest models) - lines 187, 203, 344, 349, 352, 366, 373
- `n_estimators`: 100 (for RandomForest) - lines 190, 206

### 24.12 Single Task Strategy Configs

**Location:** `TRAINING/strategies/single_task.py`

**Settings:**
- `test_size`: 0.2 (validation split) - line 68
- `random_state`: 42 (validation split) - line 68
- `early_stopping_rounds`: 50 (default) - line 75

### 24.13 Multi Task Strategy Configs

**Location:** `TRAINING/strategies/multi_task.py`

**Settings:**
- `dropout`: 0.2 (hardcoded in PyTorch layers) - lines 264, 267

### 24.14 Family Config Defaults

**Location:** `TRAINING/common/family_config.py` lines 69-74

**Default Family Info:**
- `thread_policy`: "omp_heavy" (safe default)
- `needs_tf`: False
- `needs_torch`: False
- `ridge_solver`: "auto"

### 24.15 Early Stopping Improvement Threshold

**Location:** `TRAINING/model_fun/seq_torch_base.py` line 163

**Settings:**
- Improvement threshold: `1e-6` (used to detect validation loss improvement)
- `if va < best["loss"] - 1e-6:`

### 24.16 Safety & Numerical Guards

**Location:** `TRAINING/common/safety.py`

**Settings:**
- Feature clipping: `clip=1e3` (clips to [-1000, 1000])
- Target capping: `cap_sigma=15.0` (15 MAD multiplier)
- Safe exp bounds: `lo=-40.0`, `hi=40.0`
- NumPy error handling: `over='warn'`, `invalid='warn'`, `divide='warn'`, `under='ignore'`

### 24.17 VRAM Memory Caps

**Location:** `TRAINING/common/runtime_policy.py`

**Settings:**
- `cap_vram_mb`: 4096 (4GB) for all TensorFlow GPU families
- Applied to: MLP, GAN, VAE, MetaLearning, MultiTask, CNN1D, LSTM, Transformer, TabCNN, TabLSTM, TabTransformer
- Lines: 87, 96, 105, 114, 153, 204, 213, 222, 231, 240, 249

### 24.8 Sequential Dataset Defaults

**Location:** `TRAINING/datasets/seq_dataset.py` lines 176, 217

**Settings:**
- Default `batch_size`: 32 (for PyTorch dataloaders)
- Used in `create_seq_dataloader()` function

---

## 25. Summary of Config Locations by Priority

### High Priority (User-Facing, Frequently Changed)

1. **Model Hyperparameters** - All trainer `__init__` methods
   - Learning rates, batch sizes, epochs, dropout, etc.
   - Currently: Hardcoded defaults with YAML override option
   - Target: Full YAML-based configuration

2. **Training Pipeline Settings** - `train_with_strategies.py`
   - Timeouts (7200s default), thread allocation, data sampling limits
   - Currently: Hardcoded constants and env vars
   - Target: YAML config with env var overrides

3. **GPU Settings** - Multiple files
   - VRAM caps, CUDA device selection, TensorFlow GPU settings
   - Currently: Hardcoded in runtime policies
   - Target: YAML config with GPU-specific profiles

### Medium Priority (Occasionally Changed)

1. **Threading Policies** - `common/threads.py`, `common/family_config.py`
   - Per-family thread allocation, OMP/MKL settings
   - Currently: Calculated dynamically with hardcoded defaults
   - Target: YAML config with profiles (balanced, omp_heavy, blas_heavy)

2. **Memory Management** - `memory/memory_manager.py`
   - Memory caps, cleanup thresholds
   - Currently: Hardcoded values
   - Target: YAML config

3. **Preprocessing** - `utils/data_preprocessor.py`
   - Imputation strategies, scaling methods
   - Currently: Hardcoded in code
   - Target: YAML config

### Low Priority (Rarely Changed, System-Level)

1. **System Paths** - Multiple files
   - Temp directories (`trainer_tmp/joblib`), data paths, output paths
   - Currently: Hardcoded or env vars
   - Target: YAML config with env var fallbacks

2. **Family Classifications** - `train_with_strategies.py`, `common/runtime_policy.py`
   - Model family lists, capability maps, runtime policies
   - Currently: Hardcoded dictionaries
   - Target: YAML config (may remain code-based for type safety)

3. **Security/Shell Settings** - `common/threads.py`, `common/isolation_runner.py`
   - Readline suppression, shell settings
   - Currently: Hardcoded env vars
   - Target: May remain as env vars (system-level)

---

## 26. Audit Completeness Summary

### Coverage Status

**✅ Model Trainers (20+ families):**
- All trainer `__init__` methods scanned
- All `setdefault()` calls documented
- All hardcoded hyperparameters cataloged
- Dynamic config logic documented (Transformer batch scaling, etc.)

**✅ Training Pipeline:**
- Main script (`train_with_strategies.py`) fully scanned
- Threading configuration documented
- Data processing limits documented
- Timeout settings documented
- Path and environment setup documented

**✅ Common Utilities:**
- Threading (`common/threads.py`) - documented
- Runtime policies (`common/runtime_policy.py`) - documented
- Safety guards (`common/safety.py`) - documented
- Family config (`common/family_config.py`) - documented
- Isolation runner (`common/isolation_runner.py`) - documented
- Determinism (`common/determinism.py`) - documented

**✅ Strategies:**
- Single task strategy - documented
- Multi task strategy - documented
- Cascade strategy - documented

**✅ Data Processing:**
- Cross-sectional processing - documented
- Sequential processing - documented
- Preprocessing utilities - documented
- Memory management - documented

**✅ Additional Settings:**
- Callback defaults (early stopping, LR reduction) - documented
- Gradient clipping - documented
- Safety thresholds - documented
- VRAM caps - documented
- Random seeds - documented
- Validation splits - documented

### Potential Gaps (Low Priority)

1. **PyTorch Sequential Models** - Some torch-based trainers may have additional configs
   - Files: `*_trainer_torch.py`, `seq_torch_base.py`
   - Status: Partially documented (batch_size, lr, weight_decay found)

2. **Comprehensive Trainer** - May have additional model-specific configs
   - File: `model_fun/comprehensive_trainer.py`
   - Status: Needs deeper review

3. **Experiment Scripts** - May have experiment-specific configs
   - Location: `EXPERIMENTS/` directory
   - Status: Not fully scanned (lower priority)

4. **Test Scripts** - Test-specific configurations
   - Files: `test_*.py`, `*_test.py`, shell scripts
   - Status: Partially documented (test-specific values noted)

### Confidence Level

**High Confidence (>95%):**
- Model trainer hyperparameters
- Training pipeline main settings
- Threading and resource management
- GPU/CUDA settings
- Memory management thresholds

**Medium Confidence (80-95%):**
- Strategy-specific configs
- Preprocessing detailed settings
- PyTorch model configs

**Lower Confidence (<80%):**
- Experiment-specific configs
- Test script configs
- Utility function internal defaults

### Recommendations

1. **Phase 1 (High Priority):** Extract all model hyperparameters to YAML
2. **Phase 2 (Medium Priority):** Extract training pipeline settings, threading, GPU configs
3. **Phase 3 (Lower Priority):** Extract preprocessing, memory, and system-level settings
4. **Phase 4 (Future):** Review experiment scripts and test configs if needed

---

**End of Audit**

