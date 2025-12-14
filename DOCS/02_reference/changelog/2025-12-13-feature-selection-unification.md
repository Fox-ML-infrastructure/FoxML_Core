# Changelog — 2025-12-13 (Feature Selection Unification)

**Feature Selection Now Uses Same Harness, Config System, and Comprehensive Functionality as Target Ranking**

**Related**: [SST Enforcement Design](../../../TRAINING/utils/SST_ENFORCEMENT_DESIGN.md) | [Feature Selection Guide](../../03_technical/implementation/FEATURE_SELECTION_GUIDE.md)

For a quick overview, see the [root changelog](../../../CHANGELOG.md).  
For other dates, see the [changelog index](README.md).

---

## Added

### Shared Ranking Harness

**Unified Evaluation Contract for Target Ranking and Feature Selection**
- **Enhancement**: Created `RankingHarness` class that both target ranking and feature selection now use
- **Purpose**: Ensures identical evaluation contracts to prevent "in-sample-ish" mistakes from scaling
- **Features**:
  - Same split generator (PurgedTimeSeriesSplit with time-based purging)
  - Same scoring function + metric normalization
  - Same leakage-safe imputation policy
  - Same RunContext + reproducibility tracker payload
  - Same logging/artifact writer
- **Implementation**:
  - `TRAINING/ranking/shared_ranking_harness.py` - New shared harness class
  - `TRAINING/ranking/feature_selector.py` - Refactored to use shared harness
  - `TRAINING/ranking/predictability/model_evaluation.py` - Uses same harness methods
- **Benefits**:
  - ✅ Identical evaluation contracts (no divergence between target ranking and feature selection)
  - ✅ Same stability snapshot machinery (overlap, Kendall tau)
  - ✅ Same sanitization + dtype canonicalization (prevents CatBoost object column errors)
  - ✅ Same cleaning and audit checks (ghost busters, leak scan, target validation)

### Feature Selection Comprehensive Hardening

**Feature Selection Now Has Complete Parity with Target Ranking**
- **Enhancement**: Feature selection now includes all the same comprehensive checks and functionality as target ranking
- **Added Functionality**:
  - **Linear Models**: Lasso, Ridge, and ElasticNet now enabled in feature selection (same as target ranking)
  - **Ghost Busters**: Final gatekeeper enforcement (drops problematic features before training)
  - **Pre-training Leak Scan**: Detects near-copy features that are highly correlated with target
  - **Target-Conditional Exclusions**: Per-target exclusion lists tailored to target physics
  - **Duplicate Column Detection**: Hard-fails on duplicate feature names
  - **Target Validation**: Checks for degenerate targets and class imbalance
  - **Stability Tracking**: Per-model snapshots + aggregated consensus snapshots
  - **Leak Detection Summary**: Saves `leak_detection_summary.txt` (same format as target ranking)
  - **Stability Analysis Hook**: Calls `analyze_all_stability_hook()` at end of run
- **Files**:
  - `TRAINING/ranking/shared_ranking_harness.py` - Centralized cleaning and audit checks
  - `TRAINING/ranking/feature_selector.py` - Integrated shared harness for both CROSS_SECTIONAL and SYMBOL_SPECIFIC views
  - `CONFIG/ranking/features/multi_model.yaml` - Enabled Ridge and ElasticNet (Lasso was already enabled)
- **Benefits**:
  - ✅ Feature selection is now as hardened as target ranking
  - ✅ Same comprehensive safety checks prevent data leakage
  - ✅ Same stability tracking enables reproducibility analysis
  - ✅ Same leak detection enables proactive feature review

### Feature Selection Reporting Module

**Same Output Structure as Target Ranking**
- **Enhancement**: Created `feature_selection_reporting.py` module that saves results in same format as target ranking
- **Features**:
  - `save_feature_selection_rankings()` - Saves CSV and YAML files (same format as target ranking)
  - `save_dual_view_feature_selections()` - Saves dual-view structure in REPRODUCIBILITY/FEATURE_SELECTION/
  - `save_feature_importances_for_reproducibility()` - Saves feature importances (same structure as target ranking)
- **Output Structure**:
  ```
  RESULTS/{run_id}/
    feature_selections/
      {target_column}/
        feature_selection_rankings.csv          # Same format as target ranking
        feature_selection_rankings.yaml         # Same format as target ranking
        selected_features.txt                   # Same as target ranking
        feature_importance_multi_model.csv     # Detailed multi-model summary
        feature_importances/                    # Same structure as target ranking
          {target_column}/
            CROSS_SECTIONAL/
              {model_family}_importances.csv
            SYMBOL_SPECIFIC/
              {symbol}/
                {model_family}_importances.csv
        feature_exclusions/                     # Target-conditional exclusions
          {target_name}_exclusions.yaml
    REPRODUCIBILITY/
      FEATURE_SELECTION/                        # Same structure as TARGET_RANKING
        CROSS_SECTIONAL/
          {target_column}/
            cohort={cohort_id}/
              metrics.json
              metadata.json
        SYMBOL_SPECIFIC/
          {target_column}/
            symbol={symbol}/
              cohort={cohort_id}/
                metrics.json
                metadata.json
  ```
- **Files**:
  - `TRAINING/ranking/feature_selection_reporting.py` - New reporting module
  - `TRAINING/ranking/feature_selector.py` - Integrated reporting functions
- **Benefits**:
  - ✅ Consistent output structure across target ranking and feature selection
  - ✅ Same artifacts enable easy comparison and analysis
  - ✅ Same reproducibility structure enables cohort tracking

## Changed

### Feature Selection Config Integration

**Same Config-Driven Setup as Target Ranking**
- **Enhancement**: Feature selection now uses same config hierarchy and loading methods as target ranking
- **Config System**:
  - Uses `get_cfg()` from `CONFIG.config_loader` (same as target ranking)
  - Uses `get_safety_config()` for safety thresholds (same as target ranking)
  - Uses `create_resolved_config()` for purge/embargo derivation (same as target ranking)
  - Accepts `experiment_config` parameter (same as target ranking)
  - Uses same config paths: `pipeline_config`, `safety_config`, `preprocessing_config`
- **Safety Configs**:
  - `MIN_FEATURES_REQUIRED` (from `safety.leakage_detection.ranking.min_features_required`)
  - `MIN_FEATURES_AFTER_LEAK_REMOVAL` (from `safety.leakage_detection.ranking.min_features_after_leak_removal`)
  - `MIN_FEATURES_FOR_MODEL` (from `safety.leakage_detection.ranking.min_features_for_model`)
  - `default_purge_minutes` (from `safety.temporal.default_purge_minutes`)
- **Files**:
  - `TRAINING/ranking/shared_ranking_harness.py` - Uses same config loading methods
  - `TRAINING/ranking/feature_selector.py` - Loads configs same way as target ranking
- **Benefits**:
  - ✅ Single source of truth for all safety thresholds
  - ✅ Consistent config-driven behavior across ranking and selection
  - ✅ Easy to maintain and update safety rules

### Feature Selection Dual-View Support

**Maintains Cross-Sectional and Symbol-Specific Views**
- **Enhancement**: Feature selection now properly supports both CROSS_SECTIONAL and SYMBOL_SPECIFIC views using shared harness
- **Implementation**:
  - CROSS_SECTIONAL: Single harness instance for all symbols (pooled data)
  - SYMBOL_SPECIFIC: Loop through symbols, create harness instance per symbol
  - Both views use same harness methods (build_panel, split_policy, run_importance_producers)
- **Files**:
  - `TRAINING/ranking/feature_selector.py` - Refactored to use shared harness for both views
- **Benefits**:
  - ✅ View consistency: Target ranking → feature selection → training uses same view
  - ✅ Same evaluation contract regardless of view
  - ✅ Proper per-symbol processing for SYMBOL_SPECIFIC view

## Fixed

### Critical Feature Selection Fixes

**Comprehensive Hardening and Bug Fixes**
- **Shared Harness Unpack Crashes**: Fixed `KeyError` / "too many values to unpack" errors with tolerant unpack handling (length checking, logging, graceful fallback)
- **CatBoost Dtype Mis-typing**: Fixed CatBoost treating numeric features as text/categorical with hard dtype enforcement guardrail (explicit float32 casting, fail-fast checks, inf/-inf handling)
- **RFE Edge Cases**: Fixed `KeyError: 'n_features_to_select'` with safe defaults and clamping to `[1, n_features]` for small feature sets
- **Linear Model Failures**: Fixed Ridge/ElasticNet "Unknown model family" errors with full implementations (RidgeClassifier/LogisticRegression, StandardScaler in pipelines, proper l1_ratio handling)
- **Stability Cross-Model Mixing**: Fixed stability warnings from comparing heterogeneous model families by using per-model-family snapshots with feature universe fingerprint
- **Telemetry Scoping Issues**: Fixed incorrect comparisons across targets/views/symbols with proper view→route_type mapping, symbol=None for CROSS_SECTIONAL, cohort_id filtering
- **Uniform Importance Fallback**: Fixed polluting consensus by raising ValueError for invalid models (all-zero coefficients) instead of uniform fallback
- **Last-Mile Improvements**:
  - Failed model skip reasons in consensus summary (e.g., `ridge:zero_coefs`, `elastic_net:singular`)
  - Feature universe fingerprint for stability tracking (prevents comparing different candidate sets)
- **Files**:
  - `TRAINING/ranking/feature_selector.py` - Tolerant unpack, RunContext population, telemetry scoping
  - `TRAINING/ranking/multi_model_feature_selection.py` - Dtype enforcement, linear models, stability fingerprint, skip reasons
  - `TRAINING/ranking/shared_ranking_harness.py` - Hard dtype guardrail, inf handling
  - `TRAINING/ranking/predictability/leakage_detection.py` - RFE clamping
  - `TRAINING/utils/reproducibility_tracker.py` - Telemetry scoping, cohort filtering
- **Documentation**:
  - [Implementation Verification](03_technical/fixes/2025-12-13-implementation-verification.md) - Complete verification of all 6 critical checks + 2 last-mile improvements
  - [Critical Fixes](03_technical/fixes/2025-12-13-critical-fixes.md) - Detailed root-cause analysis and fixes
  - [Telemetry Scoping Fix](03_technical/fixes/2025-12-13-telemetry-scoping-fix.md) - Telemetry scoping implementation
  - [Sharp Edges Verification](03_technical/fixes/2025-12-13-sharp-edges-verification.md) - Verification against user checklist
- **Benefits**:
  - ✅ No more shared harness unpack crashes (tolerant handling)
  - ✅ CatBoost correctly treats all numeric features as numeric (hard guardrail)
  - ✅ RFE handles edge cases gracefully (clamping)
  - ✅ Linear models work correctly (full implementations with scaling)
  - ✅ Stability tracking is per-model-family (no cross-model mixing)
  - ✅ Telemetry compares only within correct scopes (target, view, symbol, cohort)
  - ✅ Consensus integrity maintained (failed models excluded with reasons)

### Feature Selection Stability Tracking

**Per-Model Snapshots Now Saved with Feature Universe Fingerprint**
- **Issue**: Feature selection was only saving aggregated consensus snapshots, not per-model snapshots, and stability was comparing across different candidate feature sets
- **Fix**: Now saves stability snapshots for each model family with feature universe fingerprint (prevents comparing different candidate sets)
- **Implementation**:
  - CROSS_SECTIONAL: Saves snapshots with `universe_id="ALL:{feature_universe_fingerprint}"`
  - SYMBOL_SPECIFIC: Saves snapshots with `universe_id="{symbol}:{feature_universe_fingerprint}"`
  - Fingerprint computed from sorted feature names (stable across runs)
- **Files**:
  - `TRAINING/ranking/feature_selector.py` - Added per-model snapshot saving
  - `TRAINING/ranking/multi_model_feature_selection.py` - Feature universe fingerprint computation
- **Benefits**:
  - ✅ Same stability tracking as target ranking
  - ✅ Can analyze stability per model family
  - ✅ Prevents comparing stability across different candidate feature sets (pruner/sanitizer differences)
  - ✅ Enables comprehensive stability analysis

### Feature Selection Leak Detection

**Leak Detection Summary Now Saved**
- **Issue**: Feature selection detected suspicious features but didn't save summary report
- **Fix**: Now saves `leak_detection_summary.txt` in same format as target ranking
- **Implementation**:
  - Collects suspicious features from all model families and symbols
  - Saves summary with recommendations (same format as target ranking)
- **Files**:
  - `TRAINING/ranking/feature_selector.py` - Added leak detection summary saving
- **Benefits**:
  - ✅ Easy to review suspicious features across all models
  - ✅ Same format enables comparison with target ranking results
  - ✅ Proactive leak detection and review

## Documentation

### Updated Feature Selection Documentation

- **TRAINING/ranking/feature_selection_reporting.py** - New module with comprehensive docstrings
- **TRAINING/ranking/shared_ranking_harness.py** - Comprehensive docstrings explaining shared contract
- **DOCS/02_reference/changelog/2025-12-13-feature-selection-unification.md** - This changelog

### Updated Target Ranking Documentation

- **DOCS/02_reference/target_ranking/README.md** - Updated to mention shared harness and feature selection integration

## Testing

### Verification

To verify feature selection uses same harness:
1. Run feature selection - should see "Using shared ranking harness" in logs
2. Check output structure - should match target ranking structure
3. Verify stability snapshots - should see per-model snapshots in `artifacts/feature_importance/`
4. Check leak detection summary - should see `leak_detection_summary.txt` in output directory

To verify config system:
1. Set safety thresholds in `safety_config.yaml`
2. Run feature selection - should use same thresholds as target ranking
3. Check logs - should show config trace with same config paths

## Migration Notes

### For Users

- **No action required** - Feature selection automatically uses shared harness
- **Recommended**: Review feature selection output structure - now matches target ranking
- **New**: Feature selection now saves stability snapshots and leak detection summaries

### For Developers

- **Shared harness**: Use `RankingHarness` class for any new ranking operations
- **Config system**: Use same config loading methods (`get_cfg()`, `get_safety_config()`)
- **Reporting**: Use `feature_selection_reporting.py` module for consistent output format

## Files Modified

### Core Ranking System
- `TRAINING/ranking/shared_ranking_harness.py` - New shared harness class, hard dtype guardrail, inf handling
- `TRAINING/ranking/feature_selector.py` - Refactored to use shared harness, tolerant unpack, RunContext population, telemetry scoping (lines 259-281, 371-404, 1144-1163, 1187-1202, 1218-1246)
- `TRAINING/ranking/multi_model_feature_selection.py` - Dtype enforcement, linear models, stability fingerprint, skip reasons (lines 1419-1455, 1486-1650, 2285-2325, 2447-2470, 2604-2630)
- `TRAINING/ranking/feature_selection_reporting.py` - New reporting module
- `TRAINING/ranking/predictability/leakage_detection.py` - RFE clamping (line 1635)
- `TRAINING/ranking/predictability/model_evaluation.py` - Telemetry scoping (lines 4540-4560)
- `TRAINING/utils/reproducibility_tracker.py` - Telemetry scoping, cohort filtering (lines 529-551, 1728-1750)
- `TRAINING/utils/run_context.py` - Added view and symbol fields (lines 83-84)
- `CONFIG/ranking/features/multi_model.yaml` - Enabled Ridge and ElasticNet

### Documentation
- `DOCS/02_reference/changelog/2025-12-13-feature-selection-unification.md` - This changelog
- `DOCS/02_reference/target_ranking/README.md` - Updated integration section
- `DOCS/03_technical/fixes/2025-12-13-implementation-verification.md` - Complete verification of all fixes
- `DOCS/03_technical/fixes/2025-12-13-critical-fixes.md` - Detailed root-cause analysis
- `DOCS/03_technical/fixes/2025-12-13-telemetry-scoping-fix.md` - Telemetry scoping implementation
- `DOCS/03_technical/fixes/2025-12-13-sharp-edges-verification.md` - Verification against user checklist
