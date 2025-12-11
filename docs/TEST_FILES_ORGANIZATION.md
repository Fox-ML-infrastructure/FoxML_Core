# Test Files Organization

## Summary

Moved 10 test files from `TRAINING/` to `SCRIPTS/tests/` (untracked directory).

## Files Moved

1. `test_auto_fixer_e2e.sh`
2. `test_e2e_pipeline.sh`
3. `test_feature_registry_system.py`
4. `test_gpu_models.sh`
5. `test_imports.py`
6. `test_intelligent_pipeline.sh`
7. `test_leakage_fixes.sh`
8. `test_leakage_quick.sh`
9. `test_leakage_system_comprehensive.py`
10. `test_lightgbm_gpu.py`

## Verification

- ✅ Files moved successfully
- ✅ `SCRIPTS/` already in `.gitignore` (untracked)
- ✅ No runtime imports broken
- ✅ No compilation errors
- ✅ Core modules still import correctly

## Usage

Test files are now in `SCRIPTS/tests/`. Run from repo root:

```bash
# Python tests
python3 SCRIPTS/tests/test_imports.py
python3 SCRIPTS/tests/test_feature_registry_system.py

# Shell scripts
bash SCRIPTS/tests/test_e2e_pipeline.sh
bash SCRIPTS/tests/test_auto_fixer_e2e.sh
```

## Note

These files are **not imported by runtime code** - they're standalone test scripts. Moving them doesn't affect the production codebase.
