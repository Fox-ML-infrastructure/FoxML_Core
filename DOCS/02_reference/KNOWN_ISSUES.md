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
- **See**: [TESTING_NOTICE.md](../../../../TESTING_NOTICE.md) for details

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
- **XGBoost**: ✅ Functional (requires XGBoost built with GPU support)
- **CatBoost**: ✅ Functional (requires CatBoost GPU support)

### Known Limitations
- **GPU Detection**: Test models are created to verify GPU availability, which may add small startup overhead
- **Fallback Behavior**: If GPU test fails, system falls back to CPU silently (check logs for `⚠️` warnings)
- **XGBoost Legacy API**: System tries both new API (`device='cuda'`) and legacy API (`tree_method='gpu_hist'`) automatically
- **Multi-GPU**: Currently configured for single GPU (device 0). Multi-GPU support not yet implemented

### Troubleshooting
- If GPU isn't being used, check logs for:
  - `✅ Using GPU (CUDA) for [Model]` - GPU is active
  - `⚠️ [Model] GPU test failed` - GPU not available, using CPU
- Verify GPU config in `CONFIG/training_config/gpu_config.yaml`
- Ensure CUDA drivers and GPU-enabled libraries are installed

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

1. Check [CHANGELOG.md](../../../../CHANGELOG.md) for recent changes
2. Review [Detailed Changelog](changelog/README.md) for technical details
3. Check [TESTING_NOTICE.md](../../../../TESTING_NOTICE.md) for experimental features
4. Report with sufficient detail (config, error messages, environment)

---

**Note**: This document is updated as issues are discovered and resolved. Check regularly for updates.
