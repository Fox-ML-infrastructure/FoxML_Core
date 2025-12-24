# Training Pipeline Fixes — Roadmap (2025-12-16 15:30)

**Prompt:** "Fix family registry mismatches, banner printing in child processes, and other training pipeline issues"

## Context

The training orchestrator was failing with multiple distinct issues:
1. **Family registry mismatch**: Config uses snake_case (`catboost`, `neural_network`) but registries use TitleCase (`CatBoost`, `NeuralNetwork`)
2. **Banner printing in child processes**: License banner prints during isolation runs
3. **Missing families**: `CatBoost`, `RandomForest`, `Lasso`, `MutualInformation`, `UnivariateSelection` not in trainer registry
4. **Reproducibility tracking bug**: `.name` called on strings instead of Enums
5. **Model saving bug**: `_pkg_ver` referenced before assignment
6. **Misleading success banner**: Shows "✅ completed successfully" even when families fail
7. **Feature count collapse**: Silent drop from 100 requested → 52 allowed → 12 used

## Plan (now)

1. ✅ Create canonical family normalization function
2. ✅ Apply normalization at all registry boundaries
3. ✅ Add preflight validation before training starts
4. ✅ Suppress banner in child processes via environment variables
5. ✅ Fix reproducibility `.name` attribute errors
6. ✅ Fix `_pkg_ver` referenced before assignment
7. ✅ Improve run summary with trained_ok/failed/skipped counts
8. ✅ Add feature count validation logging

## Success criteria

- No `Unknown family` warnings for configured families
- No `Family 'X' not found...` at runtime (either mapped or explicitly skipped)
- No banner printed during isolation runs
- No `.name` attribute errors in reproducibility tracking
- Run summary accurately reflects training outcomes
- Feature count collapse is logged and visible

