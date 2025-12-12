# Testing Notice

**Status**: End-to-End Testing Underway  
**Date**: 2025-12-11

## Current Status

**Full end-to-end testing is currently underway** to validate the complete pipeline from target ranking → feature selection → training plan generation → model training.

Recent improvements:
- ✅ Complete SST config centralization (all hardcoded values moved to YAML)
- ✅ Full determinism (all random seeds use centralized system)
- ✅ Pipeline robustness fixes (syntax errors, import issues resolved)
- ✅ Complete F821 undefined name error elimination (194 errors fixed)
- ✅ **NEW**: Training Routing & Planning System (2025-12-11) - **Currently being tested**
  - Config-driven routing decisions (cross-sectional vs symbol-specific)
  - Automatic training plan generation
  - 2-stage training pipeline (CPU → GPU)
  - One-command end-to-end flow

## What's Being Tested

- **Training Routing System** (NEW - 2025-12-11):
  - One-command pipeline: target ranking → feature selection → training plan → training execution
  - 2-stage training (CPU models first, then GPU models)
  - Training plan auto-detection and filtering
  - All 20 models (sequential + cross-sectional)
- Full pipeline validation: target ranking → feature selection → model training
- Testing with 5 symbols (AAPL, MSFT, GOOGL, TSLA, NVDA)
- Validating all model families (20 families)
- Verifying config-driven reproducibility

## Reporting Issues

If you encounter issues:
1. Check `CHANGELOG.md` for recent changes
2. Review detailed changelog: `DOCS/02_reference/changelog/README.md`
3. Report with sufficient detail (config, error messages, environment)

---

**Note**: This notice will be updated once testing is complete. For detailed change history, see `CHANGELOG.md` and `DOCS/02_reference/changelog/README.md`.
