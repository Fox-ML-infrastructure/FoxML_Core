# Documentation Cross-References

This document tracks key cross-references between documentation files to ensure consistency.

## New Files (2025-12-14)

### Telemetry System
- **Files**: 
  - `TRAINING/utils/telemetry.py` (new)
  - `CONFIG/pipeline/training/safety.yaml` (added `safety.telemetry` section)
- **References**:
  - `TRAINING/utils/reproducibility_tracker.py` - Integrated telemetry writer
  - `DOCS/02_reference/changelog/2025-12-14-telemetry-system.md` - Detailed changelog
  - `DOCS/03_technical/implementation/REPRODUCIBILITY_STRUCTURE.md` - Reproducibility structure (telemetry follows same structure)
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
  - `TRAINING/README.md` - Links to refactoring docs
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
  - `CONFIG/README.md` - Directory structure
  - `README.md` (configuration) - Config file list
  - `CONFIG_LOADER_API.md` - API functions
  - `CONFIG_BASICS.md` - Example structure

### Utility Modules
- **Files**: 
  - `TRAINING/utils/target_utils.py`
  - `TRAINING/utils/sklearn_safe.py`
- **Documented in**:
  - `RANKING_SELECTION_CONSISTENCY.md` - Usage guide
  - `MODULE_REFERENCE.md` - API reference

## Key Cross-Reference Patterns

### Configuration Files
All config files should reference:
- `MODULAR_CONFIG_SYSTEM.md` - Main config guide
- `README.md` (configuration) - Overview
- `USAGE_EXAMPLES.md` - Practical examples

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
  - `DOCS/03_technical/implementation/COHORT_AWARE_REPRODUCIBILITY_IMPLEMENTATION.md` - Implementation details
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
- **Code**:
  - `TRAINING/utils/cohort_metadata_extractor.py` - **NEW** - Unified metadata extraction utility
  - `TRAINING/utils/reproducibility_tracker.py` - Major refactor for cohort-aware mode
  - All pipeline modules use unified extractor

### RESULTS Directory Organization
- **Structure**: All runs organized in `RESULTS/{cohort_id}/{run_name}/`
- **Documented in**:
  - `INTELLIGENT_TRAINING_TUTORIAL.md` - Output structure section
  - `REPRODUCIBILITY_STRUCTURE.md` - Complete structure guide
  - `COHORT_AWARE_REPRODUCIBILITY.md` - Storage structure section

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

**Deprecated files moved to `DOCS/LEGACY/`:**
- `EXPERIMENTS_WORKFLOW.md` - Replaced by Intelligent Training Pipeline
- `EXPERIMENTS_QUICK_START.md` - Replaced by Intelligent Training Tutorial
- `EXPERIMENTS_IMPLEMENTATION.md` - Replaced by current implementation docs
- `STATUS_DEBUGGING.md` - Outdated debugging status (2025-12-09)

See `DOCS/LEGACY/README.md` for migration guide.

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
  - `DOCS/03_technical/fixes/LOOKAHEAD_BIAS_FIX_PLAN.md`
  - `DOCS/03_technical/fixes/LOOKAHEAD_BIAS_SAFE_IMPLEMENTATION.md`
  - `DOCS/02_reference/changelog/2025-12-14-lookahead-bias-fixes.md`
- **References**:
  - `CHANGELOG.md` - Added to Recent Highlights
  - `DOCS/INDEX.md` - Added to Fixes section
  - `DOCS/02_reference/changelog/README.md` - Added 2025-12-14 entry
  - `DOCS/03_technical/fixes/README.md` - Added look-ahead bias fixes section
  - `DOCS/03_technical/README.md` - Added to Fixes section
  - `CONFIG/pipeline/training/safety.yaml` - Added lookahead_bias_fixes config section
  - `CONFIG/experiments/e2e_ranking_test.yaml` - Added lookahead_bias_fixes config section

## Cross-Reference Updates (2025-12-13)

**SST Enforcement Design Implementation:**
- **Files**:
  - `TRAINING/utils/SST_ENFORCEMENT_DESIGN.md` - Complete design specification
  - `TRAINING/utils/SST_IMPLEMENTATION_COVERAGE.md` - Implementation coverage matrix
  - `TRAINING/utils/TYPE_BOUNDARY_WIRING_COMPLETE.md` - Type boundary wiring details
  - `TRAINING/utils/BOUNDARY_ASSERTIONS_COMPLETE.md` - Boundary assertions details
  - `DOCS/02_reference/changelog/2025-12-13-sst-enforcement-design.md` - Complete changelog
  - `DOCS/03_technical/fixes/2025-12-13-sst-enforcement-design.md` - Technical fix documentation
- **References**:
  - `INDEX.md` - Added to implementation and architecture sections
  - `03_technical/README.md` - Added to fixes section
  - `03_technical/implementation/README.md` - Added to core systems section
  - `CHANGELOG.md` - Added highlights section
  - `changelog/README.md` - Added index entry
  - `fixes/README.md` - Added to recent fixes
- **Related Documentation**:
  - Links to Single Source of Truth fix
  - Links to Fingerprint Tracking
  - Links to Feature Selection Unification

## Cross-Reference Updates (2025-12-12)

**Architecture Documentation Moved:**
- `TRAINING/stability/FEATURE_IMPORTANCE_STABILITY.md` → `DOCS/03_technical/implementation/FEATURE_IMPORTANCE_STABILITY.md`
- `TRAINING/common/PARALLEL_EXECUTION.md` → `DOCS/03_technical/implementation/PARALLEL_EXECUTION.md`

**All references updated in:**
- `INDEX.md` - Systems Reference, Research, Implementation sections
- `FEATURE_SELECTION_TUTORIAL.md` - Updated path reference
