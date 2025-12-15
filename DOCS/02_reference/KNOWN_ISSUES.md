# Known Issues & Limitations

This document tracks features that are **not yet fully functional**, have **known limitations**, or are **experimental** and should be used with caution.

**Last Updated**: 2025-12-12

---

## Experimental Features (Use with Caution)

### Decision-Making System ⚠️ EXPERIMENTAL
- **Status**: Under active testing
- **Location**: `TRAINING/decisioning/`
- **Components**:
  - Decision Policies (`policies.py`)
  - Decision Engine (`decision_engine.py`)
  - Bayesian Patch Policy (`bayesian_policy.py`)
- **Limitations**:
  - Requires 5+ runs in same cohort+segment before Bayesian recommendations
  - Thresholds may need adjustment based on data characteristics
  - Auto-apply mode (`apply_mode: "apply"`) should be used with extreme caution
- **Recommendation**: Keep `decisions.apply_mode: "off"` or `"dry_run"` until fully validated
- **See**: [TESTING_NOTICE.md](testing/TESTING_NOTICE.md) for details

### Stability Analysis ⚠️ EXPERIMENTAL
- **Status**: Under active testing
- **Location**: `TRAINING/stability/`
- **Limitations**:
  - Thresholds may need adjustment based on your data characteristics
  - Some model families may show false positives for instability
- **Recommendation**: Monitor stability reports and adjust thresholds in `stability_config.yaml` as needed

---

## GPU Acceleration

### Current Status
- **LightGBM**: ✅ Fully functional (CUDA and OpenCL support)
- **XGBoost**: ✅ Functional (XGBoost 3.1+ compatible, requires XGBoost built with GPU support)
- **CatBoost**: ✅ Functional (requires `task_type='GPU'` explicitly set, requires CatBoost GPU support)

### Known Limitations
- **GPU Detection**: Test models are created to verify GPU availability, which may add small startup overhead
- **Fallback Behavior**: If GPU test fails, system falls back to CPU silently (check logs for `⚠️` warnings)
- **XGBoost 3.1+ Compatibility**: `gpu_id` parameter removed in XGBoost 3.1+. System uses `device='cuda'` with `tree_method='hist'` (automatic fallback to legacy API for older versions)
- **CatBoost Quantization**: CatBoost does quantization on CPU first (20+ seconds for large datasets), then trains on GPU. Watch GPU memory allocation, not just utilization %, to verify GPU usage
- **Multi-GPU**: Currently configured for single GPU (device 0). Multi-GPU support not yet implemented

### Troubleshooting

**XGBoost 3.1+ Issues:**
- **Error**: `gpu_id has been removed since 3.1. Use device instead`
- **Fix**: System automatically uses `device='cuda'` (no `gpu_id` needed). Ensure `gpu.xgboost.device: "cuda"` in config
- **Verification**: Check logs for `✅ Using GPU (CUDA) for XGBoost`

**CatBoost Not Using GPU:**
- **Critical**: CatBoost **requires** `task_type='GPU'` to use GPU (devices alone is ignored)
- **Check**: Look for `✅ CatBoost GPU verified: task_type=GPU` in logs
- **Configuration**: Ensure `gpu.catboost.task_type: "GPU"` is set in `gpu_config.yaml`
- **Verification**: Watch GPU memory allocation with `watch -n 0.1 nvidia-smi` (quantization happens on CPU first)

**CPU Bottleneck (GPU Underutilization):**
- **Symptom**: CPU at 100% usage, GPU at low utilization (30-40%), slow training despite GPU being enabled
- **Common Causes** (ordered by likelihood):
  1. **Outer Parallelism** (Most Common): `thread_count` only limits threads inside a single `fit()`. If you have outer parallelism (CV folds in parallel, multiple symbols/targets in parallel, GridSearch `n_jobs`), CPU will still peg.
     - **Tell-tale**: Load average ≈ number of threads, multiple Python/CatBoost processes visible
     - **Fix**: Set all outer parallelism to 1: `GridSearchCV(..., n_jobs=1)`, don't parallelize across symbols/targets while also capping threads
     - **Check**: Look for `cv_n_jobs > 1` or parallel execution of multiple models
  2. **CatBoost Not Actually Using GPU**: GPU at ~7% while CPU is 100% usually means CPU training or CPU-side prep dominates
     - **Diagnosis**: Check `model.get_all_params()` after fit - verify `task_type="GPU"` and `devices="0"` are actually set
     - **Fix**: System automatically verifies GPU params post-fit and warns if GPU not active
     - **Check CatBoost logs**: Should explicitly say "GPU" in training logs
  3. **CPU Burn from Pool Building/Quantization/Metrics** (Not Training Threads): Even with `task_type="GPU"`, CatBoost can hammer CPU for:
     - pandas→numpy conversion
     - quantization/binarization (happens on CPU before GPU training)
     - frequent metric evaluation
     - **Fixes**:
       - Build `Pool` once and reuse it (don't rebuild each fold/target)
       - Reduce metric frequency: `metric_period=50` (or larger), `verbose=50` (system automatically sets this)
       - If using eval every iter + early stopping, that can dominate
  4. **BLAS/OpenMP Threads Exploding Outside CatBoost**: Even if CatBoost respects `thread_count`, NumPy/OpenBLAS/MKL can spawn threads during preprocessing
     - **Fix**: Set environment variables before importing numpy/pandas/catboost:
       ```python
       import os
       for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
           os.environ[k] = "2"  # Or appropriate limit
       ```
     - **Note**: System's threading utilities should handle this, but check if preprocessing is spawning extra threads
  5. **Small Dataset Overhead**: For datasets <100k-200k rows, CPU data preparation overhead can exceed GPU computation time
     - **Fix**: For datasets < 50k rows, consider using CPU instead of GPU
- **Diagnosis**: 
  - Check system monitor: CPU at 100%, load average > number of cores
  - GPU utilization < 50% despite `task_type='GPU'` being set
  - Check for multiple Python/CatBoost processes (outer parallelism)
  - Verify GPU is actually active: `model.get_all_params()` after fit
- **Solutions**:
  1. **Disable outer parallelism**: Set `cv_n_jobs=1` when using GPU training
  2. **CatBoost thread limiting**: CatBoost now limits CPU threads via `gpu.catboost.thread_count` in `gpu_config.yaml` (default: 8 threads)
  3. **Verify GPU usage**: System automatically checks `model.get_all_params()` post-fit and warns if GPU not active
  4. **Reduce metric frequency**: System automatically sets `metric_period=50` to reduce evaluation overhead
  5. **For datasets < 50k rows**: Consider using CPU instead of GPU (overhead exceeds benefit)
- **When to use GPU**: GPU acceleration is most beneficial for datasets > 100k-200k rows where the computation time exceeds the overhead

**CatBoost Slow Training (20+ minutes for 50k samples):**
- **Symptom**: CatBoost training takes 20+ minutes for 50k samples (should take seconds to a few minutes)
- **Most Likely Causes** (ordered by probability):
  1. **Text Features**: Raw text columns passed to CatBoost without `text_features` parameter
     - **Problem**: CatBoost treats text as categorical with unique values per row, exploding computation
     - **Fix**: Explicitly specify text columns: `text_features=['description_column', 'comments_column']`
     - **Check**: Look for columns with string/object dtype that aren't explicitly marked as text
  2. **High Cardinality Categoricals**: Categorical columns with thousands of unique values (e.g., `User_ID`, `IP_Address`)
     - **Problem**: One-hot encoding on 10k unique values creates 10k feature columns
     - **Important**: **Numeric columns with high cardinality are normal** (continuous features like `mid_price_vol`, `bid_ask_spread_est`). Only categorical columns with ID-like characteristics should be flagged.
     - **Auto-Detection Logic**: System only suggests dropping when multiple signals agree:
       - Column is treated as **categorical** (in `cat_features` or object/category dtype)
       - High unique ratio (>20% or >50% for strict)
       - Values mostly occur once (median count per value ≤ 2) OR unique ratio > 80%
       - ID-like name patterns (`_id`, `uuid`, `tx_`, `order_`, `session_`, `hash_`) OR near-perfect uniqueness (>95%)
     - **Fix**: 
       - For **categorical ID columns**: Drop or encode differently
       - For **numeric columns**: Keep them (high cardinality is normal for continuous features)
       - Verify columns are not accidentally in `cat_features` list if they should be numeric
     - **Check**: System automatically detects and warns about problematic categorical ID columns, but preserves numeric columns
  3. **Tree Depth Too High**: Depth > 8 causes exponential complexity ($2^d$)
     - **Problem**: Default depth is 6, but if set to 10+ training time explodes
     - **Fix**: Keep `depth: 6` (default) or `depth: 8` maximum
     - **Check**: Verify `depth` in CatBoost config is ≤ 8
  4. **Evaluation Overhead**: Calculating metrics after every tree is expensive
     - **Problem**: With `eval_set` and many trees, metric calculation happens every iteration
     - **Fix**: Set `metric_period=50` or `100` to calculate metrics every 50-100 trees instead of every tree
     - **Check**: If using early stopping with `eval_set`, add `metric_period` to `.fit()` call
  5. **Hardware Mismatch**: GPU overhead for small datasets, or CPU thread restrictions
     - **Problem**: GPU transfer overhead for small datasets, or CatBoost using only 1 thread on CPU
     - **Fix**: For <50k rows, use CPU with `thread_count=-1` (all cores), or switch to CPU entirely
- **Diagnostic Checklist**:
  1. Drop ID columns (`User_ID`, `Row_ID`, etc.)
  2. Define text features explicitly: `text_features=['col_name']`
  3. Check depth: Ensure `depth ≤ 8` (preferably 6)
  4. Add `metric_period=100` to `.fit()` call if using `eval_set`
  5. For small datasets (<50k), consider CPU instead of GPU

**General GPU Issues:**
- If GPU isn't being used, check logs for:
  - `✅ Using GPU (CUDA) for [Model]` - GPU is active
  - `✅ CatBoost GPU verified: task_type=GPU` - CatBoost GPU confirmed
  - `⚠️ [Model] GPU test failed` - GPU not available, using CPU
- Verify GPU config in `CONFIG/training_config/gpu_config.yaml`
- Ensure CUDA drivers and GPU-enabled libraries are installed

**Process Deadlock/Hang (readline library conflict):**
- **Symptom**: Process hangs for 10+ minutes on small datasets, CPU at 100%, error: `sh: symbol lookup error: sh: undefined symbol: rl_print_keybinding`
- **Cause**: Conda environment's `readline` library conflicts with system's `readline` library, causing shell command failures (e.g., `nvidia-smi` checks) to retry indefinitely
- **Fix**:
  1. Kill the hung process (`Ctrl+C` or `kill -9`)
  2. Repair Conda environment:
     ```bash
     conda install -c conda-forge readline=8.2
     # Or: conda update readline
     # If that doesn't work, also install:
     conda install -c conda-forge ncurses
     ```
  3. Verify fix: Run a quick test - training should complete in seconds, not minutes
- **Prevention**: The system already sets `TERM=dumb` and `SHELL=/usr/bin/bash` in isolation runner to mitigate readline issues, but Conda environment conflicts can still occur

---

## Target Ranking & Feature Selection

### Current Status
- **Target Ranking**: ✅ Fully functional
- **Feature Selection**: ✅ Fully functional
- **GPU Acceleration**: ✅ Enabled for LightGBM, XGBoost, CatBoost

### Known Limitations
- **Parallel Execution**: Currently disabled by default (`parallel_targets: false`). Enable in experiment config for faster execution
- **Large Target Lists**: Auto-discovery of 100+ targets may be slow. Use `max_targets_to_evaluate` to limit
- **Feature Selection Speed**: Multi-model feature selection can be slow on large feature sets. Consider reducing `top_m_features` for testing

---

## Configuration System

### Current Status
- **SST Compliance**: ✅ All hardcoded values removed from TRAINING pipeline
- **Config Loading**: ✅ Fully functional
- **Config Validation**: ✅ Smart validation based on context (e.g., `auto_targets`)

### Known Limitations
- **Config Overlays**: Some advanced overlay scenarios may not be fully tested
- **Environment Variables**: Not all config values can be overridden via environment variables yet

---

## Data Processing

### Current Status
- **Label Generation**: ✅ Functional
- **Feature Engineering**: ✅ Functional
- **Data Validation**: ✅ Functional

### Known Limitations
- **Barrier Targets**: All datasets generated before 2025-12-12 should be regenerated due to horizon unit fix
- **Versioned Datasets**: Version tracking is implemented but migration tools not yet available

---

## Model Training

### Current Status
- **All 20 Model Families**: ✅ Functional
- **GPU Acceleration**: ✅ Enabled for supported families
- **Training Routing**: ✅ Functional

### Known Limitations
- **Neural Networks**: Some GPU memory issues may occur with very large datasets (VRAM caps configured)
- **Sequential Models**: 3D preprocessing issues resolved, but edge cases may still exist
- **Model Isolation**: Some families require `TRAINER_NO_ISOLATION=1` for GPU support

---

## Reproducibility & Tracking

### Current Status
- **Reproducibility Tracking**: ✅ Fully functional
- **Trend Analysis**: ✅ Integrated across all stages
- **Cohort Organization**: ✅ Functional

### Known Limitations
- **Metadata Completeness**: Some runs from before 2025-12-12 may be missing metadata files (see `METADATA_MISSING_README.md` if present)
- **Trend Analysis**: Requires 3+ runs in same cohort for trend detection

---

## Documentation

### Current Status
- **4-Tier Documentation**: ✅ Complete
- **Cross-Linking**: ✅ Complete
- **Legal Documentation**: ✅ Complete

### Known Limitations
- **Some Legacy Docs**: May reference deprecated workflows. Check for `⚠️ **Legacy**` or `**DEPRECATED**` markers

---

## Reporting Issues

If you encounter issues not listed here:

1. Check [CHANGELOG.md](../../CHANGELOG.md) for recent changes
2. Review [Detailed Changelog](changelog/README.md) for technical details
3. Check [TESTING_NOTICE.md](testing/TESTING_NOTICE.md) for experimental features
4. Report with sufficient detail (config, error messages, environment)

---

**Note**: This document is updated as issues are discovered and resolved. Check regularly for updates.
