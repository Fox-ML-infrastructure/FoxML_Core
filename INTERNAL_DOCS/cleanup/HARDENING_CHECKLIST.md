# Code Hardening Checklist

This document tracks systematic code quality improvements to prevent silent failures.

## ✅ Completed

### 1. Config Access Bugs (12 instances fixed)
- **Issue**: Code accessed `safety_config` without `safety.` prefix, causing silent fallback to defaults
- **Files Fixed**:
  - `TRAINING/ranking/predictability/model_evaluation.py` (6 instances)
  - `TRAINING/ranking/predictability/leakage_detection.py` (4 instances)
  - `TRAINING/ranking/target_ranker.py` (1 instance)
  - `TRAINING/common/leakage_auto_fixer.py` (5 instances)
  - `TRAINING/common/leakage_sentinels.py` (1 instance)
- **Impact**: Config values now read correctly from YAML files

### 2. Undefined Variables (3 instances fixed)
- **Issue**: `output_dir` used but not defined/passed
- **Files Fixed**:
  - `TRAINING/ranking/predictability/model_evaluation.py` (1 instance)
  - `TRAINING/ranking/multi_model_feature_selection.py` (2 instances)
- **Impact**: Prevents `NameError` at runtime

### 3. Syntax Errors (1 instance fixed)
- **Issue**: Indentation error in `unified_training_interface.py`
- **Impact**: Module now imports correctly

### 4. Infrastructure Created
- ✅ Config schema validation (`CONFIG/config_schemas.py`)
- ✅ Strict mode enforcement (`TRAINING/common/strict_mode.py`)
- ✅ Config integrity tests (`tests/test_config_integrity.py`)

## 🔄 In Progress

### Static Analysis Setup
- [ ] Run `ruff check TRAINING --select F821` regularly
- [ ] Set up mypy with `check_untyped_defs = True`
- [ ] Add to CI pipeline

### Config Validation
- [ ] Add schema validation to all config loaders
- [ ] Run `pytest tests/test_config_integrity.py` in CI
- [ ] Document expected config structure

### Strict Mode
- [ ] Add strict mode checks to critical paths
- [ ] Enable in test environment
- [ ] Document usage

## 📋 Remaining Tasks

### 1. Complete Static Analysis
```bash
# Run these regularly:
ruff check TRAINING --select F821
python -m compileall .
pytest tests/test_config_integrity.py
```

### 2. Add More Schema Validations
- [ ] System config schema
- [ ] Pipeline config schema
- [ ] Model config schemas

### 3. Expand Strict Mode Coverage
- [ ] Add to config access paths
- [ ] Add to critical data processing functions
- [ ] Add to model training entry points

### 4. Create Smoke Tests
- [ ] Minimal end-to-end test
- [ ] Test with minimal config
- [ ] Test with missing optional configs

## 🎯 Usage

### Enable Strict Mode
```bash
FOXML_STRICT_MODE=1 python your_script.py
```

### Run Config Tests
```bash
pytest tests/test_config_integrity.py -v
```

### Check for Undefined Variables
```bash
ruff check TRAINING --select F821
```

## 📊 Statistics

- **Total Issues Fixed**: 16
  - Config access bugs: 12
  - Undefined variables: 3
  - Syntax errors: 1
- **Files Modified**: 6
- **Infrastructure Created**: 3 files
