# CONFIG Folder Audit & Organization Plan

## Hardcoded Values Found in TRAINING (Should Move to CONFIG)

### 1. Decision Policies (`TRAINING/decisioning/policies.py`)
**Hardcoded thresholds:**
- `jaccard_topK < 0.5` (line 56)
- `jaccard_topK < recent.iloc[-2] * 0.8` (line 56)
- `route_entropy > 1.5` (line 73)
- `route_changed >= 3` in last 5 runs (line 76)
- `auc_trend < -0.01` AND `feature_trend > 10` (line 97)
- `pos_rate drift > 0.1` (line 117)

**Recommendation:** Move to `CONFIG/training_config/decision_policies.yaml`

### 2. Resolved Config (`TRAINING/utils/resolved_config.py`)
**Hardcoded value:**
- `default_purge_minutes = 85.0` (line 537 in model_evaluation.py - should be config-driven)

**Recommendation:** Move to `CONFIG/training_config/safety_config.yaml` (already exists, add purge defaults)

### 3. Memory Manager (`TRAINING/memory/memory_manager.py`)
**Hardcoded value:**
- `memory_threshold = 0.8` (line 63, 68) - fallback if not in config

**Status:** Already has config fallback, but default should be in config file

### 4. Importance Diff Detector (`TRAINING/common/importance_diff_detector.py`)
**Hardcoded thresholds:**
- `diff_threshold: float = 0.1` (line 61)
- `relative_diff_threshold: float = 0.5` (line 62)
- `min_importance_full: float = 0.01` (line 63)

**Recommendation:** Move to `CONFIG/training_config/stability_config.yaml` (new file)

### 5. Target Routing (`TRAINING/ranking/target_routing.py`)
**Hardcoded thresholds:**
- `cs_auc_threshold = 0.65` (default fallback)
- `frac_symbols_good_threshold = 0.5` (default fallback)

**Status:** Already has config fallback via `get_cfg()`, but defaults should be in config

## Config Files Status

### ✅ Active Config Files (Referenced in Code)

**Core Configs:**
- `CONFIG/defaults.yaml` - Global defaults (SST)
- `CONFIG/config_loader.py` - Config loader
- `CONFIG/feature_registry.yaml` - Feature registry
- `CONFIG/excluded_features.yaml` - Excluded features

**Training Configs (`CONFIG/training_config/`):**
- `intelligent_training_config.yaml` - ✅ Active (intelligent trainer)
- `system_config.yaml` - ✅ Active
- `pipeline_config.yaml` - ✅ Active
- `safety_config.yaml` - ✅ Active
- `preprocessing_config.yaml` - ✅ Active
- `optimizer_config.yaml` - ✅ Active
- `gpu_config.yaml` - ✅ Active
- `memory_config.yaml` - ✅ Active
- `threading_config.yaml` - ✅ Active
- `routing_config.yaml` - ✅ Active
- `callbacks_config.yaml` - ✅ Active
- `family_config.yaml` - ✅ Active
- `sequential_config.yaml` - ✅ Active
- `first_batch_specs.yaml` - ✅ Active

**Model Configs (`CONFIG/model_config/`):**
- All model YAML files - ✅ Active (loaded by model trainers)

**Feature Selection (`CONFIG/feature_selection/`):**
- `multi_model.yaml` - ✅ Active

**Routing (`CONFIG/routing/`):**
- `default.yaml` - ✅ Active

**Target Ranking (`CONFIG/target_ranking/`):**
- `multi_model.yaml` - ✅ Active

**Experiments (`CONFIG/experiments/`):**
- `e2e_ranking_test.yaml` - ✅ Active
- `fwd_ret_60m_test.yaml` - ✅ Active

### ⚠️ Deprecated/Unused Config Files

1. **`CONFIG/multi_model_feature_selection.yaml.deprecated`**
   - Status: Explicitly deprecated (has `.deprecated` extension)
   - Action: Can be deleted (moved to `CONFIG/feature_selection/multi_model.yaml`)

2. **`CONFIG/multi_model_feature_selection.yaml`**
   - Status: Check if still used (may be duplicate of `feature_selection/multi_model.yaml`)
   - Action: Verify usage, consolidate if duplicate

3. **`CONFIG/comprehensive_feature_ranking.yaml`**
   - Status: Unknown usage
   - Action: Check if referenced in code

4. **`CONFIG/fast_target_ranking.yaml`**
   - Status: Unknown usage
   - Action: Check if referenced in code

5. **`CONFIG/feature_selection_config.yaml`**
   - Status: Unknown usage
   - Action: Check if referenced in code

6. **`CONFIG/target_configs.yaml`**
   - Status: Unknown usage
   - Action: Check if referenced in code

7. **`CONFIG/feature_target_schema.yaml`**
   - Status: Unknown usage
   - Action: Check if referenced in code

8. **`CONFIG/feature_groups.yaml`**
   - Status: Unknown usage
   - Action: Check if referenced in code

9. **`CONFIG/training/models.yaml`**
   - Status: Unknown usage (may be superseded by `model_config/` files)
   - Action: Check if referenced in code

## Organization Recommendations

### 1. Create Missing Config Files

**New config files needed:**
- `CONFIG/training_config/decision_policies.yaml` - Decision policy thresholds
- `CONFIG/training_config/stability_config.yaml` - Stability analysis thresholds

### 2. Consolidate Duplicate Configs

**Potential duplicates:**
- `multi_model_feature_selection.yaml` vs `feature_selection/multi_model.yaml`
- `training/models.yaml` vs `model_config/*.yaml`

### 3. Archive/Remove Unused Files

**Files to verify and potentially remove:**
- `multi_model_feature_selection.yaml.deprecated` (explicitly deprecated)
- Any config files not referenced in codebase

### 4. Add Default Values to Existing Configs

**Configs that need default values added:**
- `safety_config.yaml` - Add `default_purge_minutes: 85.0`
- `routing_config.yaml` - Add default thresholds if missing
- `memory_config.yaml` - Ensure `memory_threshold: 0.8` is present

## Action Plan

1. **Phase 1: Create Missing Configs**
   - Create `decision_policies.yaml`
   - Create `stability_config.yaml`
   - Add defaults to existing configs

2. **Phase 2: Move Hardcoded Values**
   - Update `policies.py` to use config
   - Update `resolved_config.py` to use config
   - Update `importance_diff_detector.py` to use config

3. **Phase 3: Clean Up Unused Files**
   - Verify usage of unknown config files
   - Remove deprecated files
   - Consolidate duplicates

4. **Phase 4: Documentation**
   - Update CONFIG_README.md with new structure
   - Document all config files and their purposes
