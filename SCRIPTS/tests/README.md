# Test Scripts

This directory contains test and validation scripts that are not part of the core runtime.

## Files

- `test_auto_fixer_e2e.sh` - End-to-end test for auto-fixer
- `test_e2e_pipeline.sh` - End-to-end pipeline test
- `test_feature_registry_system.py` - Feature registry system test
- `test_gpu_models.sh` - GPU model testing
- `test_imports.py` - Import verification
- `test_intelligent_pipeline.sh` - Intelligent pipeline test
- `test_leakage_fixes.sh` - Leakage fix testing
- `test_leakage_quick.sh` - Quick leakage test
- `test_leakage_system_comprehensive.py` - Comprehensive leakage system test
- `test_lightgbm_gpu.py` - LightGBM GPU test

## Usage

These scripts are for development and testing purposes. They are not imported by the runtime codebase.

Run from repo root:
```bash
./SCRIPTS/tests/test_imports.py
bash SCRIPTS/tests/test_e2e_pipeline.sh
```

## Note

This directory is in `.gitignore` and is not tracked by git.
