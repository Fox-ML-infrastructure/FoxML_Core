# Feature Selection Execution Order

**Formalized hierarchy of feature selection operations in the training pipeline.**

This document defines the exact order of operations for multi-model feature selection to ensure consistent behavior and reproducible results.

## Execution Order (Chronological)

### Phase 1: Initialization and Configuration
1. **Load determinism system** (set global seeds, thread control)
2. **Load multi-model config** from `CONFIG/feature_selection/multi_model.yaml` or `FeatureSelectionConfig`
3. **Extract model families** (filter enabled families from config)
4. **Load aggregation config** (consensus method, weights, fallback settings)
5. **Validate view parameter** (CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO)
6. **Filter symbols** based on view (all symbols for CROSS_SECTIONAL, single symbol for SYMBOL_SPECIFIC)

**Location**: `TRAINING/ranking/feature_selector.py` - `select_features_for_target()` (lines ~86-183)

### Phase 2: Symbol Processing (Parallel or Sequential)
7. **Check parallel execution config** (`parallel_symbols` from config)
8. **For each symbol** (or parallel batch):
   - **Load symbol data** from `data_dir/symbol={symbol}/{symbol}.parquet`
   - **Validate data file exists**
   - **Call `_process_single_symbol()`** (see Phase 3)

**Location**: `TRAINING/ranking/feature_selector.py` (lines ~229-296)  
**Parallelization**: Uses `ProcessPoolExecutor` if `parallel_symbols=true` and `threading.parallel.enabled=true`

### Phase 3: Single Symbol Processing
9. **Load data** (with `max_samples_per_symbol` cap)
10. **Detect data interval** (auto-detect from timestamps or use explicit interval)
11. **Prepare features and target**:
    - Extract target column
    - Filter features (leakage filtering, registry filtering)
    - Handle missing values
12. **For each model family** (enabled families from config):
    - **Train model** with family-specific config
    - **Extract feature importance** (method depends on family):
      - Tree models: `feature_importances_` (gain/split)
      - Neural networks: SHAP TreeExplainer or permutation
      - Linear models: Absolute coefficients
    - **Normalize importance** (handle NaN/inf, pad/truncate to feature count)
    - **Apply fallback** if no signal (uniform importance)
    - **Store result** in `ImportanceResult` dataclass
13. **Return results** for all model families

**Location**: `TRAINING/ranking/multi_model_feature_selection.py` - `_process_single_symbol()` (lines ~1652-1975)  
**Key Function**: `train_model_and_get_importance()` (lines ~887-1648)

### Phase 4: Aggregation Across Models and Symbols
14. **Collect all results** from all symbols and model families
15. **Group by model family** (LightGBM, XGBoost, Random Forest, etc.)
16. **Normalize importance** per family (handle edge cases, fallbacks)
17. **Aggregate across symbols** (mean, median, or weighted mean per family)
18. **Compute consensus score** across model families:
    - **Weighted average** (family weights from config)
    - **Bonus for multi-family agreement** (features important across diverse architectures)
    - **Penalty for single-family importance** (avoid model-specific biases)
19. **Rank features** by consensus score (descending)
20. **Select top N features** (`top_n` from config)

**Location**: `TRAINING/ranking/multi_model_feature_selection.py` - `_aggregate_multi_model_importance()` (lines ~2000-2500)

### Phase 5: Cross-Sectional Ranking (Optional)
21. **Check if enabled** (`cross_sectional_ranking.enabled` from config)
22. **Check minimum symbols** (`min_symbols` threshold, default: 5)
23. **If enabled**:
    - **Select top K candidates** (`top_k_candidates`, default: 50)
    - **Compute cross-sectional importance** (rank features by cross-symbol predictive power)
    - **Tag features** by importance tier (high, medium, low)
    - **Merge with consensus scores** (cross-sectional bonus/penalty)
24. **Re-rank features** with cross-sectional adjustments

**Location**: `TRAINING/ranking/feature_selector.py` (lines ~372-450)  
**Module**: `TRAINING/ranking/cross_sectional_feature_ranker.py`

### Phase 6: Stability Tracking (Optional)
25. **Save stability snapshot** (if stability tracking enabled):
    - **Convert to importance dict** (feature → consensus_score)
    - **Call `save_snapshot_hook()`** (non-invasive hook)
    - **Store in `output_dir/stability/`** (if provided)
26. **Auto-analyze stability** (if `auto_analyze=true` in config)

**Location**: `TRAINING/ranking/feature_selector.py` (lines ~316-332)  
**Module**: `TRAINING/stability/feature_importance.py`

### Phase 7: Return Results
27. **Return selected features** (list of feature names)
28. **Return importance dataframe** (feature, consensus_score, per-family scores, cross-sectional tags)

**Location**: `TRAINING/ranking/feature_selector.py` - `select_features_for_target()` (return at line ~100)

## Key Principles

### 1. Multi-Model Consensus
- **Purpose**: Avoid model-specific biases by aggregating across diverse architectures
- **Method**: Weighted average with bonuses for multi-family agreement
- **Result**: Features that are important across tree models, neural networks, and linear models

### 2. Symbol-Level Aggregation
- **Purpose**: Find features that generalize across symbols
- **Method**: Mean/median aggregation per model family, then consensus across families
- **Result**: Features that work for AAPL, MSFT, GOOGL, etc.

### 3. Cross-Sectional Ranking (Optional)
- **Purpose**: Identify features with cross-symbol predictive power
- **Method**: Rank features by how well they predict across symbols simultaneously
- **Result**: Features that capture market-wide patterns, not just symbol-specific noise

### 4. Determinism and Reproducibility
- **Global determinism** set before any ML imports
- **Stable seeds** per symbol (using `stable_seed_from()`)
- **Reproducibility tracking** via stability snapshots
- **Unified threading control**: All models use `TRAINING/common/threads.py` utilities (`plan_for_family()`, `thread_guard()`) for GPU-aware thread management and optimal OMP/MKL allocation

### 5. Fallback Handling
- **No signal**: Uniform importance (1e-6 per feature) instead of failure
- **Invalid importance**: Normalize and pad/truncate to feature count
- **Model failure**: Log warning, continue with other models

## Configuration

### Multi-Model Config (`CONFIG/feature_selection/multi_model.yaml`)
```yaml
model_families:
  lightgbm:
    enabled: true
    importance_method: native
    weight: 1.0
  xgboost:
    enabled: true
    importance_method: native
    weight: 1.0
  # ... more families

aggregation:
  method: weighted_mean
  multi_family_bonus: 0.1
  single_family_penalty: 0.05
  fallback:
    uniform_importance: 1e-6
    normalize_after_fallback: true

cross_sectional_ranking:
  enabled: true
  min_symbols: 5
  top_k_candidates: 50
```

### Feature Selection Config (`FeatureSelectionConfig`)
- `target`: Target column name
- `symbols`: List of symbols to process
- `top_n`: Number of features to select
- `max_samples_per_symbol`: Data cap per symbol
- `model_families`: Per-family configs
- `aggregation`: Consensus method and weights

## Integration Points

### With Target Ranking
- Feature selection runs **before** target ranking
- Selected features are passed to `evaluate_target_predictability()`
- Reduces dimensionality for faster ranking

### With Training Pipeline
- Selected features are passed to `train_models_for_interval_comprehensive()`
- Per-target feature lists stored in `target_features` dict
- Training uses selected features instead of full feature set

### With Stability Tracking
- Stability snapshots saved after aggregation
- Tracks feature importance consistency across runs
- Enables regression detection (importance drift)

## Troubleshooting

### Issue: All features have zero importance

**Symptom**: Consensus scores are all zero or uniform

**Root Cause**: Models failed to train or extract importance

**Fix**:
1. Check model training logs for errors
2. Verify data has sufficient samples
3. Check target column has variance
4. Review fallback config (should use uniform importance, not zero)

### Issue: Selected features are model-specific

**Symptom**: Top features only important to one model family

**Root Cause**: Consensus weights favor single family, or multi-family bonus too low

**Fix**:
1. Increase `multi_family_bonus` in aggregation config
2. Increase `single_family_penalty` to discourage single-family features
3. Verify multiple model families are enabled

### Issue: Cross-sectional ranking not running

**Symptom**: No cross-sectional tags in importance dataframe

**Root Cause**: Disabled in config or insufficient symbols

**Fix**:
1. Check `cross_sectional_ranking.enabled=true`
2. Verify `len(symbols) >= min_symbols` (default: 5)
3. Check logs for "Cross-sectional ranking skipped" messages

### Issue: Parallel execution not working

**Symptom**: Symbols processed sequentially despite `parallel_symbols=true`

**Root Cause**: Global parallel execution disabled or only 1 symbol

**Fix**:
1. Check `threading.parallel.enabled=true` in threading config
2. Verify `parallel_symbols=true` in feature selection config
3. Ensure `len(symbols) > 1` (parallel only for multiple symbols)

## Current Implementation Status

✅ **Implemented**:
- Phase 1-4: Core multi-model feature selection
- Phase 5: Cross-sectional ranking (optional)
- Phase 6: Stability tracking (optional)
- Phase 7: Results return

🔧 **Future Enhancements**:
- Per-target feature selection (currently global)
- Adaptive feature selection (adjust top_n based on data size)
- Feature interaction detection (combine correlated features)

## Related Documentation

- [Feature Selection Tutorial](../../01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - User guide
- [Feature Selection Guide](FEATURE_SELECTION_GUIDE.md) - Technical details
- [Feature Filtering Execution Order](FEATURE_FILTERING_EXECUTION_ORDER.md) - Pre-selection filtering
- [Feature Pruning Execution Order](FEATURE_PRUNING_EXECUTION_ORDER.md) - Post-selection pruning
- [Parallel Execution](PARALLEL_EXECUTION.md) - Parallelization infrastructure
