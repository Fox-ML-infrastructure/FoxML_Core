# Testing Notice

**Status**: End-to-End Testing Underway  
**Date**: 2025-12-10 (Updated)

## Current Status

**Full end-to-end testing is currently underway** to validate the complete pipeline from target ranking → feature selection → model training. Recent improvements include:

- ✅ **Complete Single Source of Truth (SST) config centralization** (2025-12-10) — **ALL** hardcoded configuration values across the entire TRAINING pipeline moved to YAML files. Every model trainer (52+ files in `model_fun/` and `models/`) now loads hyperparameters, test splits, and random seeds from centralized config. Same config → same results across all pipeline stages.
- ✅ **Full determinism** (2025-12-10) — All `random_state` values now use centralized determinism system (`BASE_SEED`) instead of hardcoded values. Training strategies, feature selection, data splits, and all model initializations are now fully deterministic.
- ✅ **Pipeline robustness fixes** (2025-12-10) — Fixed critical syntax errors and variable initialization issues in config loading patterns. All config loading blocks now have proper fallbacks, preventing runtime errors.
- ✅ **Large file refactoring** (2025-12-09) — Split 3 monolithic files into modular components while maintaining 100% backward compatibility
- ✅ **Model family status tracking** — Added comprehensive debugging to identify which families succeed/fail and why
- ✅ **Interval detection robustness** — Fixed timestamp gap filtering to ignore outliers before computing median
- ✅ **Import fixes** — Resolved all missing import errors in refactored modules
- ✅ Target ranking and feature selection have consistent behavior
- ✅ Interval detection respects `data.bar_interval` from config (no spurious warnings)
- ✅ All sklearn models use shared preprocessing (`make_sklearn_dense_X`) for consistent NaN/dtype handling
- ✅ CatBoost auto-detects target type and sets correct loss function
- ✅ Ranking and selection pipelines are behaviorally identical

## What's Being Tested

- ✅ **Config centralization** — All configuration values load from YAML files (single source of truth)
- ✅ **Determinism** — All random seeds use centralized determinism system for reproducibility
- ✅ **Config loading robustness** — All config loading patterns verified to have proper fallbacks and no syntax errors
- ✅ Target ranking workflows — Working with unified interval handling
- ✅ Feature selection — Fixed sklearn NaN/dtype issues, CatBoost loss function, Boruta feature count mismatch
- ✅ Pipeline consistency — Ranking and selection now use same helpers and patterns
- ✅ Boruta gatekeeper — Fixed feature count mismatch, now functions as statistical gatekeeper without false failures
- 🔄 **Full end-to-end testing** — **CURRENTLY UNDERWAY**: Complete pipeline validation from target ranking → feature selection → model training
  - Testing with 5 symbols (AAPL, MSFT, GOOGL, TSLA, NVDA)
  - Validating all model families (8+ families being tested)
  - Verifying data flow through Phase 3 (model training)
  - Checking model family status tracking output
  - Verifying config-driven reproducibility (same config → same results)
  - Confirming no runtime errors from config loading patterns

## Recent Fixes

- **Config loading pattern robustness** (2025-12-10): Fixed critical syntax and runtime errors:
  - Fixed `SyntaxError` in `data_loading.py` - moved `if _CONFIG_AVAILABLE:` out of function parameter list
  - Fixed `SyntaxError` in `leakage_detection.py` - moved config loading out of function call parameters
  - Fixed `UnboundLocalError` in `model_evaluation.py` - added missing `else:` clauses for `MIN_FEATURES_FOR_MODEL` and `MIN_FEATURES_AFTER_LEAK_REMOVAL`
  - Comprehensive audit confirmed all `if _CONFIG_AVAILABLE:` blocks now have proper `else:` clauses
  - All variables are guaranteed to be initialized before use, preventing runtime errors
- **Complete Single Source of Truth (SST) config centralization** (2025-12-10): **ALL** hardcoded values across entire pipeline moved to YAML files:
  - **All model trainers** (52+ files): `test_size`, `random_state`, `n_estimators`, `max_depth`, `learning_rate`, `alpha` now load from config
  - Feature pruning thresholds and hyperparameters → `preprocessing_config.yaml`
  - Leakage detection thresholds → `safety_config.yaml` (`leakage_sentinels.*`)
  - Auto-fixer settings → `safety_config.yaml` (`auto_fixer.*`)
  - Training strategy parameters (test_size, random_state) → load from config
  - Model hyperparameters → `models.{family}.{param}` config paths
  - Neural network optimizers → `optimizer.learning_rate` config
  - All function defaults now use `Optional[Type] = None` and load from config when `None`
- **Determinism system** (2025-12-10): All `random_state=42` hardcoded values replaced with `BASE_SEED`:
  - Training strategies (`single_task.py`, `cascade.py`)
  - Feature pruning utilities
  - Data preprocessing splits
  - Model creation in strategies
- **Interval handling**: Wired `explicit_interval` through entire ranking call chain
- **Sklearn preprocessing**: Replaced ad-hoc imputers with shared `make_sklearn_dense_X()` helper
- **CatBoost configuration**: Auto-detects classification vs regression and sets appropriate loss function
- **Shared utilities**: Created `TRAINING/utils/target_utils.py` for consistent target type detection
- **Boruta feature count mismatch**: Fixed `ValueError: X has N features, but ExtraTreesClassifier is expecting M features` by using `train_score = math.nan` for Boruta (selector, not predictor). Added NaN handling in logging and checkpoint serialization. Boruta gatekeeper now functions properly without false "failed" status.

## Known Considerations

- Feature engineering may still require human review and validation
- Some configurations may require adjustment based on your specific use case
- Performance characteristics may vary depending on hardware and dataset size
- Edge cases and error handling are still being validated

## Reporting Issues

If you encounter issues during testing:
1. Check existing issues in the repository
2. Verify your configuration matches the expected format
3. Review recent changes in `CHANGELOG.md`
4. Report issues with sufficient detail (config, error messages, environment)

## Next Steps

- Continue end-to-end testing with multiple targets and model families
- Monitor for any remaining interval detection warnings
- Verify CatBoost runs successfully for both classification and regression targets
- Validate sklearn models handle edge cases (sparse data, extreme values, etc.)
- Verify Boruta gatekeeper produces expected confirmed/rejected/tentative feature labels
- Confirm Boruta gate effect is visible in `feature_importance_with_boruta_debug.csv` output

---

**Note**: This notice will be removed or updated once testing is complete and the changes are fully validated.

