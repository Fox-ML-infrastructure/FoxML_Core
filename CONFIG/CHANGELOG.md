# CONFIG Directory Changelog

## 2025-12-23: Scope Violation Firewall Config

### New Safety Config Option
- **Added**: `safety.output_layout.strict_scope_partitioning` flag
  - Location: `CONFIG/pipeline/training/safety.yaml`
  - Default: `false` (warnings only, legacy fallback allowed)
  - Set to `true` to enforce strict OutputLayout validation (hard errors on missing metadata)
  
### Purpose
- Controls behavior of `_save_to_cohort()` when cohort metadata is missing required fields (`view`, `universe_sig`, `target`)
- When `false`: Warns and falls back to legacy path construction (for gradual migration)
- When `true`: Hard error if OutputLayout invariants cannot be validated (for production)

### Related Changes
- New `OutputLayout` dataclass in `TRAINING/orchestration/utils/output_layout.py`
- Extended path helpers with optional `universe_sig` parameter

---

## 2025-12-18: Config Cleanup and Path Migration

### Config Cleanup
- **Removed duplicate file**: `multi_model_feature_selection.yaml` 
  - Duplicate of `ranking/features/multi_model.yaml`
  - Code already had fallback logic to use new location
- **Symlink audit**: Documented all 24 symlinks for backward compatibility
- **Directory structure**: Verified clean organization with no duplicates

### Path Migration
- **Replaced all hardcoded paths** in TRAINING directory with centralized config loader API
- **Files updated**:
  - `TRAINING/orchestration/intelligent_trainer.py` - 13 instances replaced
  - `TRAINING/ranking/predictability/model_evaluation.py` - 2 instances
  - `TRAINING/ranking/feature_selector.py` - 2 instances
  - `TRAINING/ranking/target_ranker.py` - 5 instances
  - `TRAINING/ranking/multi_model_feature_selection.py` - Uses centralized loader
  - `TRAINING/ranking/utils/leakage_filtering.py` - Enhanced path resolution

### New Config Loader Functions
- `get_experiment_config_path(exp_name: str) -> Path`
  - Get path to experiment config file
  - Centralized path resolution
  
- `load_experiment_config(exp_name: str) -> Dict[str, Any]`
  - Load experiment config by name
  - Proper precedence: experiment config overrides intelligent_training_config and defaults
  - Missing values fall back through precedence chain

- Enhanced `get_config_path()` to handle experiment configs automatically

### Validation Tools
- **Created**: `CONFIG/tools/validate_config_paths.py`
  - Scans TRAINING for remaining hardcoded `Path("CONFIG/...")` patterns
  - Validates config loader API access
  - Checks symlink validity
  - Can be run as part of CI/CD

### SST Compliance
- All config access goes through centralized loader
- Defaults automatically injected from `defaults.yaml` via `inject_defaults()`
- Experiment configs properly override defaults (top-level config)
- Fixed one SST issue: `config_loader.py` now uses `get_config_path()` instead of hardcoded path

### Documentation Updates
- Updated `CONFIG/README.md` with:
  - Complete symlink documentation
  - Migration guide with code examples
  - Config precedence clarification
  - Recent changes section
  
- Updated `CONFIG/tools/README.md` with validation script documentation

### Backward Compatibility
- All symlinks remain intact for backward compatibility
- Old paths still work via symlinks
- Config loader checks both new and old locations
- Fallback code paths preserved for when config loader unavailable

### Results
- ✅ No syntax errors
- ✅ All config files accessible via config loader API
- ✅ All symlinks valid
- ✅ Remaining hardcoded paths only in fallback code (acceptable)
- ✅ SST compliance maintained

