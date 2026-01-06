# Documentation Cross-References

This document tracks key cross-references between documentation files to ensure consistency.

## New Files (2026-01-04)

### RunIdentity Wiring and Path Organization
- **Files**:
  - `DOCS/02_reference/changelog/2026-01-04-run-identity-wiring-and-path-organization.md` - Detailed changelog
  - `CHANGELOG.md` - Root changelog entry
- **Updated Docs**:
  - `DOCS/00_executive/DETERMINISTIC_TRAINING.md` - Added strict determinism section and snapshot verification
  - `DOCS/02_reference/configuration/DETERMINISTIC_RUNS.md` - Added output directory structure and snapshot verification
  - `DOCS/02_reference/changelog/README.md` - Added index entry
- **References**:
  - `DOCS/02_reference/configuration/RUN_IDENTITY.md` - RunIdentity system reference
  - `bin/run_deterministic.sh` - Deterministic launcher script
- **Code Changes**:
  - `TRAINING/orchestration/utils/reproducibility_tracker.py` - `_save_to_cohort()` params
  - `TRAINING/ranking/predictability/model_evaluation.py` - `partial_identity` usage, seed fallback
  - `TRAINING/orchestration/utils/output_layout.py` - SYMBOL_SPECIFIC path simplification

## New Files (2025-12-14)

### Telemetry System
- **Files**: 
  - `TRAINING/utils/telemetry.py` (new)
  - `CONFIG/pipeline/training/safety.yaml` (added `safety.telemetry` section)
- **References**:
  - `CHANGELOG.md` - Root changelog entry
- **Structure**: Sidecar files in cohort directories, view-level and stage-level rollups

## New Files (2025-12-09)

### Refactoring Documentation
- **Files**: 
  - `DOCS/03_technical/refactoring/REFACTORING_SUMMARY.md`
  - `DOCS/03_technical/refactoring/SPECIALIZED_MODELS.md`
  - `DOCS/03_technical/refactoring/TARGET_PREDICTABILITY_RANKING.md`
  - `DOCS/03_technical/refactoring/TRAINING_STRATEGIES.md`
- **References**:
  - `INDEX.md` - Added to refactoring section
  - `01_tutorials/training/TRAINING_README.md` - Links to refactoring docs
  - `changelog/README.md` - Changelog index (refactoring note in general.md)
- **Module READMEs**:
  - `TRAINING/models/specialized/README.md` - Brief, links to detailed docs
  - `TRAINING/ranking/predictability/README.md` - Brief, links to detailed docs
  - `TRAINING/training_strategies/README.md` - Brief, links to detailed docs

## New Files (2025-12-08)

### Ranking and Selection Consistency
- **File**: `DOCS/01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md`
- **References**:
  - `INTELLIGENT_TRAINING_TUTORIAL.md` - Main pipeline guide
  - `MODULAR_CONFIG_SYSTEM.md` - Config structure
  - `USAGE_EXAMPLES.md` - Practical examples
  - `CONFIG_LOADER_API.md` - Logging config utilities
  - `MODULE_REFERENCE.md` - Utility API reference
  - `README.md` (configuration) - Config overview

### Logging Configuration
- **File**: `CONFIG/logging_config.yaml`
- **Documented in**:
  - `MODULAR_CONFIG_SYSTEM.md` - Section 5
  - `02_reference/configuration/CONFIG_README.md` - Directory structure
  - `README.md` (configuration) - Config file list
  - `CONFIG_LOADER_API.md` - API functions
  - `CONFIG_BASICS.md` - Example structure


### Training Pipeline
All training docs should reference:
- `INTELLIGENT_TRAINING_TUTORIAL.md` - Main tutorial
- `RANKING_SELECTION_CONSISTENCY.md` - Pipeline behavior
- `MODULAR_CONFIG_SYSTEM.md` - Config system

### API References
All API docs should reference:
- `MODULE_REFERENCE.md` - Python API
- `INTELLIGENT_TRAINER_API.md` - Trainer API
- `CONFIG_LOADER_API.md` - Config loading

## Broken References Fixed

- Removed reference to non-existent `COMPREHENSIVE_FEATURE_RANKING.md` in `FEATURE_IMPORTANCE_METHODOLOGY.md`
- Updated all references to point to newer unified pipeline docs

## New Files (2025-12-12)

### Cohort-Aware Reproducibility System
- **Files**:
  - `DOCS/03_technical/implementation/COHORT_AWARE_REPRODUCIBILITY.md` - Complete guide
  - `DOCS/03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md` - Directory structure guide
  - `DOCS/03_technical/implementation/REPRODUCIBILITY_API.md` - API reference
  - `DOCS/03_technical/implementation/REPRODUCIBILITY_ERROR_HANDLING.md` - Error handling guide
  - `DOCS/03_technical/implementation/REPRODUCIBILITY_IMPROVEMENTS.md` - Improvements summary
  - `DOCS/03_technical/implementation/REPRODUCIBILITY_SELF_TEST.md` - Self-test checklist
- **References**:
  - `INDEX.md` - Added to implementation section
  - `INTELLIGENT_TRAINING_TUTORIAL.md` - Updated output structure section
  - `CHANGELOG.md` - Added highlights section
  - `changelog/2025-12-12.md` - Complete detailed changelog
### Integrated Config Backups
- **New Location**: `RESULTS/{cohort_id}/{run_name}/backups/` (when `output_dir` provided)
- **Legacy Location**: `CONFIG/backups/` (backward compatible)
- **Documented in**:
  - `INTELLIGENT_TRAINING_TUTORIAL.md` - Output structure section
  - `changelog/2025-12-12.md` - Detailed backup integration notes
- **Updated References**:
  - `configuration/README.md` - Should mention both locations
  - `SAFETY_LEAKAGE_CONFIGS.md` - Should mention new location

## Preferred Documentation Order

When multiple docs cover similar topics, prefer:
1. **Newer unified docs** (RANKING_SELECTION_CONSISTENCY.md) over older scattered docs
2. **Modular config system** docs over legacy config references
3. **Intelligent training tutorial** over manual workflow docs
4. **Usage examples** with practical code over abstract descriptions
5. **Cohort-aware reproducibility** docs over legacy reproducibility tracking

## Legacy Documentation

**Deprecated files moved to `LEGACY/`:**
- `EXPERIMENTS_WORKFLOW.md` - Replaced by Intelligent Training Pipeline
- `EXPERIMENTS_QUICK_START.md` - Replaced by Intelligent Training Tutorial
- `EXPERIMENTS_IMPLEMENTATION.md` - Replaced by current implementation docs
- `STATUS_DEBUGGING.md` - Outdated debugging status (2025-12-09)

See `LEGACY/README.md` for migration guide.

## Cross-Reference Updates (2025-12-14)

### Feature Selection and Config Fixes
- **Files**:
  - `DOCS/02_reference/changelog/2025-12-14-feature-selection-and-config-fixes.md` - Complete detailed changelog
- **Code Changes**:
  - `TRAINING/ranking/multi_model_feature_selection.py` - Fixed UnboundLocalError for np
  - `TRAINING/ranking/feature_selector.py` - Fixed import and unpacking errors
  - `TRAINING/ranking/shared_ranking_harness.py` - Fixed return type annotation
  - `TRAINING/ranking/target_ranker.py` - Added skip reason tracking
  - `TRAINING/ranking/target_routing.py` - Fixed routing reason strings, added skip reasons
  - `TRAINING/orchestration/intelligent_trainer.py` - Fixed config loading, added target exclusion
  - `TRAINING/utils/leakage_budget.py` - Added calendar features (hour_of_day, minute_of_hour)
- **Config Changes**:
  - `CONFIG/experiments/e2e_ranking_test.yaml` - Added exclude_target_patterns example
  - `CONFIG/experiments/e2e_full_targets_test.yaml` - Added exclude_target_patterns
- **Documentation Updates**:
  - `CHANGELOG.md` - Added to Recent Highlights and Fixed sections
  - `DOCS/02_reference/changelog/README.md` - Added 2025-12-14 entry
  - `DOCS/01_tutorials/configuration/EXPERIMENT_CONFIG_GUIDE.md` - Added exclude_target_patterns documentation
  - `DOCS/01_tutorials/training/AUTO_TARGET_RANKING.md` - Added exclude_target_patterns to examples and table
  - `DOCS/CROSS_REFERENCES.md` - This section

### Look-Ahead Bias Fixes
- **Files**:
  - `DOCS/02_reference/changelog/2025-12-14-lookahead-bias-fixes.md`
- **References**:
  - `CHANGELOG.md` - Added to Recent Highlights
  - `CONFIG/pipeline/training/safety.yaml` - Added lookahead_bias_fixes config section

## Cross-Reference Updates (2025-12-12)

**Architecture Documentation Moved:**
- `TRAINING/stability/FEATURE_IMPORTANCE_STABILITY.md` → `DOCS/03_technical/implementation/FEATURE_IMPORTANCE_STABILITY.md`
- `TRAINING/common/PARALLEL_EXECUTION.md` → `DOCS/03_technical/implementation/PARALLEL_EXECUTION.md`

**All references updated in:**
- `INDEX.md` - Systems Reference, Research, Implementation sections
- `FEATURE_SELECTION_TUTORIAL.md` - Updated path reference
