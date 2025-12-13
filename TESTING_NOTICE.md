# Testing Notice

## Testing & Development Data

**All testing and development is performed using 5-minute interval data.** This includes:
- All unit tests
- Integration tests
- End-to-end pipeline validation
- Performance benchmarks
- Feature validation
- Model training workflows

The software is designed to work with various data intervals, but all validation, testing, and development work uses 5-minute bar data as the standard reference.

---

## Highly Experimental Features (2025-12-12)

⚠️ **IMPORTANT**: The following features are **highly experimental** and are currently being **heavily tested**. These are new additions to the TRAINING system and should be used with caution until fully validated in your environment.

### Decision-Making System
- **Decision Policies** (`TRAINING/decisioning/policies.py`): Automated decision policies for feature instability, route instability, feature explosion decline, and class balance drift
- **Decision Engine** (`TRAINING/decisioning/decision_engine.py`): Evaluates regression/trend signals and produces actionable decisions
- **Config-driven thresholds** (`CONFIG/training_config/decision_policies.yaml`): All policy thresholds are configurable
- **Status**: Under active testing. Use with caution in production environments.

### Bayesian Patch Policy
- **BayesianPatchPolicy** (`TRAINING/decisioning/bayesian_policy.py`): Thompson sampling over discrete patch templates
- **Adaptive config tuning**: Learns from past run outcomes to recommend config patches
- **State persistence**: Bayesian state stored in `REPRODUCIBILITY/bayes_state/`
- **Status**: Under active testing. Requires 5+ runs in same cohort+segment before recommendations. Use `dry_run` mode first.

### Stability Analysis
- **Stability Config** (`CONFIG/training_config/stability_config.yaml`): Configurable thresholds for importance difference detection
- **Status**: Under active testing. Thresholds may need adjustment based on your data characteristics.

### Auto-Config Application
- **Apply-mode** (`--apply-decisions apply`): Automatically applies decision patches to config
- **Status**: **Use with extreme caution**. Always test with `dry_run` mode first. Verify receipts in `REPRODUCIBILITY/patches/` before enabling apply-mode.

**Recommendation**: For production use, keep `decisions.apply_mode: "off"` or `"dry_run"` until these features are fully validated in your environment.

---

**Status**: End-to-End Testing Underway  
**Date**: 2025-12-11

## Current Status

**Full end-to-end testing is currently underway** to validate the complete pipeline from target ranking → feature selection → training plan generation → model training.

Recent improvements:
- ✅ Complete SST config centralization (all hardcoded values moved to YAML)
- ✅ Full determinism (all random seeds use centralized system)
- ✅ Pipeline robustness fixes (syntax errors, import issues resolved)
- ✅ Complete F821 undefined name error elimination (194 errors fixed)
- ✅ **NEW**: GPU Acceleration for Target Ranking & Feature Selection (2025-12-12) - **Functional, under validation**
  - XGBoost, CatBoost, and LightGBM GPU acceleration
  - XGBoost 3.1+ compatibility (fixed `gpu_id` removal issue)
  - CatBoost GPU verification (explicit `task_type='GPU'` requirement)
  - CatBoost CPU bottleneck mitigation (thread_count limiting added, under investigation)
  - Automatic detection with graceful CPU fallback
  - All settings config-driven from `gpu_config.yaml` (SST)
- ✅ **NEW**: Training Routing & Planning System (2025-12-11) - **Currently being tested**
  - Config-driven routing decisions (cross-sectional vs symbol-specific)
  - Automatic training plan generation
  - 2-stage training pipeline (CPU → GPU)
  - One-command end-to-end flow

## What's Being Tested

- **Feature Selection Critical Fixes** (NEW - 2025-12-13):
  - **Shared harness robustness**: Tolerant unpack handling, fallback path consistency
  - **Dtype enforcement**: Hard guardrail preventing CatBoost from mis-typing numeric features
  - **Linear model implementations**: Ridge/ElasticNet with StandardScaler, proper l1_ratio handling
  - **RFE edge cases**: Clamping n_features_to_select to [1, n_features] for small feature sets
  - **Stability tracking**: Per-model-family snapshots with feature universe fingerprint (not cross-model mixing)
  - **Telemetry scoping**: Correct view→route_type mapping, symbol=None for CROSS_SECTIONAL, cohort_id filtering
  - **Consensus integrity**: Failed models excluded with skip reasons (no uniform fallback noise)
  - **Acceptance criteria**: CROSS_SECTIONAL and INDIVIDUAL runs compare only within matching (method, universe_id) buckets
- **Training Routing System** (2025-12-11):
  - One-command pipeline: target ranking → feature selection → training plan → training execution
  - 2-stage training (CPU models first, then GPU models)
  - Training plan auto-detection and filtering
  - All 20 models (sequential + cross-sectional)
- Full pipeline validation: target ranking → feature selection → model training
- Testing with 5 symbols (AAPL, MSFT, GOOGL, TSLA, NVDA)
- Validating all model families (20 families)
- Verifying config-driven reproducibility

## Known Issues & Workarounds

### Process Deadlock/Hang (readline library conflict)
- **Symptom**: Process hangs for 10+ minutes on small datasets, CPU at 100%, error: `sh: symbol lookup error: sh: undefined symbol: rl_print_keybinding`
- **Cause**: Conda environment's `readline` library conflicts with system's `readline` library
- **Fix**: 
  1. Kill hung process (`Ctrl+C` or `kill -9`)
  2. Repair environment: `conda install -c conda-forge readline=8.2` (or `conda update readline`)
  3. If needed: `conda install -c conda-forge ncurses`
- **Prevention**: System sets `TERM=dumb` and `SHELL=/usr/bin/bash` to mitigate, but Conda conflicts can still occur
- See [Known Issues](DOCS/02_reference/KNOWN_ISSUES.md) for details

### GPU Acceleration
- **XGBoost 3.1+**: If you see `gpu_id has been removed since 3.1` errors, ensure you're using the latest code (fixed 2025-12-12)
- **CatBoost GPU**: CatBoost requires `task_type='GPU'` explicitly set. Check logs for `✅ CatBoost GPU verified` to confirm GPU is being used
- **CatBoost Performance Issues** ⚠️ **UNDER INVESTIGATION** (2025-12-12):
  - **CPU Bottleneck**: CPU at 100% usage, GPU at low utilization (30-40%) during CatBoost GPU training, especially on small datasets (<100k rows)
    - **Cause**: CPU data preparation/quantization overhead exceeds GPU computation time for small datasets
    - **Current Fix**: `thread_count` limiting added (default: 8 threads) via `gpu.catboost.thread_count` in `gpu_config.yaml`
    - **Workaround**: Set `gpu.catboost.thread_count: 8` (or lower) in `gpu_config.yaml` to reduce CPU bottleneck
  - **Slow Training (20+ minutes for 50k samples)**: CatBoost training abnormally slow despite being generally fast
    - **Most Likely Causes**: Text features without `text_features` parameter, high cardinality categoricals (ID columns), depth > 8, or evaluation overhead
    - **Fixes Applied**: Added `metric_period: 50` to configs to reduce evaluation overhead
    - **Checklist**: Drop ID columns, define text features explicitly, keep depth ≤ 8, use `metric_period` if using eval_set
    - **Status**: Being actively investigated. See [Known Issues](DOCS/02_reference/KNOWN_ISSUES.md) for detailed troubleshooting
- **GPU Detection**: If GPU isn't being used, check logs for `⚠️ [Model] GPU test failed` messages and verify CUDA drivers are installed
- See [GPU Setup Guide](DOCS/01_tutorials/setup/GPU_SETUP.md) and [Known Issues](DOCS/02_reference/KNOWN_ISSUES.md) for detailed troubleshooting

## Reporting Issues

If you encounter issues:
1. Check `CHANGELOG.md` for recent changes
2. Review detailed changelog: `DOCS/02_reference/changelog/README.md`
3. Check [Known Issues & Limitations](DOCS/02_reference/KNOWN_ISSUES.md) for known problems and workarounds
4. Report with sufficient detail (config, error messages, environment, GPU status)

---

**Note**: This notice will be updated once testing is complete. For detailed change history, see `CHANGELOG.md` and `DOCS/02_reference/changelog/README.md`.
