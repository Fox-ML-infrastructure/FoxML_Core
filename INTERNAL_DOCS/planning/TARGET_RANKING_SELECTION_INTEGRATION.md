# Target Ranking & Selection Integration into Training Pipeline

**Status**: ✅ **COMPLETED**  
**Created**: 2025-12-07  
**Completed**: 2025-12-07  
**Goal**: Fold target ranking and feature selection into the main training pipeline for automated, intelligent model training

---

## Overview

Currently, target ranking (`SCRIPTS/rank_target_predictability.py`) and feature selection (`SCRIPTS/multi_model_feature_selection.py`) are separate scripts that must be run manually before training. This document outlines the plan to integrate these capabilities directly into `TRAINING/train_with_strategies.py` to create an end-to-end automated pipeline.

---

## Current State

### Target Ranking (`SCRIPTS/rank_target_predictability.py`)
- **Purpose**: Evaluates which of 63+ targets are most predictable
- **Method**: Trains multiple model families (LightGBM, XGBoost, Random Forest, Neural Networks) on sample data
- **Output**: Ranked list of targets with:
  - Composite predictability scores
  - Model R²/ROC-AUC scores (cross-validated)
  - Feature importance magnitude
  - Consistency across models
  - Leakage detection flags
- **Usage**: Standalone script, outputs JSON/CSV rankings
- **Config**: `CONFIG/fast_target_ranking.yaml`, `CONFIG/target_configs.yaml`

### Feature Selection (`SCRIPTS/multi_model_feature_selection.py`)
- **Purpose**: Selects predictive features using multi-model consensus
- **Method**: Trains multiple model families, extracts importance (native/SHAP/permutation), aggregates across models and symbols
- **Output**: Ranked feature list with:
  - Multi-model importance scores
  - Consensus rankings
  - Target-specific or target-agnostic rankings
- **Usage**: Standalone script, requires pre-selected targets
- **Config**: `CONFIG/multi_model_feature_selection.yaml`, `CONFIG/comprehensive_feature_ranking.yaml`

### Training Pipeline (`TRAINING/train_with_strategies.py`)
- **Current**: Takes targets and features as inputs (discovered or provided)
- **Target Discovery**: Auto-discovers targets from data if `interval == exec_cadence`
- **Feature Discovery**: Auto-discovers features (excludes target columns)
- **Limitation**: No intelligent ranking/selection - uses all discovered targets/features or user-provided lists

---

## Integration Goals

### Primary Objectives
1. **Automated Target Selection**: Run target ranking before training, select top N targets automatically
2. **Intelligent Feature Selection**: Run feature selection for each selected target, use top M features per target
3. **Configurable Workflow**: Make ranking/selection optional (can still provide manual targets/features)
4. **Performance Optimization**: Cache ranking/selection results to avoid re-computation
5. **Unified Configuration**: Centralize ranking/selection configs in `CONFIG/training_config/`

### Secondary Objectives
1. **Incremental Updates**: Re-rank targets periodically (e.g., weekly) without full re-run
2. **Target-Specific Features**: Allow different feature sets per target
3. **Cross-Validation Integration**: Use same CV folds for ranking, selection, and training
4. **Progress Reporting**: Show ranking/selection progress in training logs

---

## Architecture Design

### Phase 1: Basic Integration (MVP)

#### 1.1 New Training Modes
Add to `train_with_strategies.py`:
- `--auto-targets`: Enable automatic target ranking and selection
- `--auto-features`: Enable automatic feature selection per target
- `--top-n-targets`: Number of top targets to select (default: 5)
- `--top-m-features`: Number of top features per target (default: 100)

#### 1.2 Integration Points
```
train_with_strategies.py
├── [NEW] run_target_ranking()
│   ├── Calls SCRIPTS/rank_target_predictability.py logic
│   ├── Returns ranked target list
│   └── Caches results to disk
│
├── [NEW] run_feature_selection(target, symbols, data)
│   ├── Calls SCRIPTS/multi_model_feature_selection.py logic
│   ├── Returns ranked feature list for target
│   └── Caches results to disk
│
├── [MODIFY] train_models_for_interval_comprehensive()
│   ├── If --auto-targets: call run_target_ranking() first
│   ├── For each selected target:
│   │   ├── If --auto-features: call run_feature_selection()
│   │   └── Use selected features for training
│   └── Otherwise: use existing discovery logic
│
└── [NEW] load_cached_rankings()
    └── Check cache before re-running ranking/selection
```

#### 1.3 Configuration
Create `CONFIG/training_config/target_ranking_config.yaml`:
```yaml
target_ranking:
  enabled: false  # Set to true to enable auto-ranking
  top_n_targets: 5  # Number of top targets to select
  min_predictability_score: 0.1  # Minimum composite score
  model_families:
    - lightgbm
    - xgboost
    - random_forest
  max_samples_per_symbol: 10000
  cv_folds: 3
  cache_dir: "results/target_rankings"
  cache_ttl_days: 7  # Re-rank after 7 days

feature_selection:
  enabled: false  # Set to true to enable auto-selection
  top_m_features: 100  # Number of top features per target
  model_families:
    - lightgbm
    - random_forest
    - neural_network
  max_samples_per_symbol: 50000
  cache_dir: "results/feature_selections"
  cache_ttl_days: 7
```

#### 1.4 Code Structure
```
TRAINING/
├── train_with_strategies.py
│   └── [MODIFY] Add ranking/selection integration
│
├── ranking/
│   ├── __init__.py
│   ├── target_ranker.py  # [NEW] Wrapper around rank_target_predictability.py logic
│   └── feature_selector.py  # [NEW] Wrapper around multi_model_feature_selection.py logic
│
└── utils/
    └── ranking_cache.py  # [NEW] Cache management for rankings
```

### Phase 2: Advanced Integration

#### 2.1 Shared Data Loading
- Reuse same data loading logic across ranking, selection, and training
- Avoid loading data multiple times
- Implement data caching layer

#### 2.2 Cross-Validation Consistency
- Use same CV folds for ranking, selection, and training
- Prevents data leakage and ensures fair comparison
- Store fold indices in cache

#### 2.3 Incremental Updates
- Track when rankings were last computed
- Only re-rank if data is newer than cache
- Support "force refresh" flag

#### 2.4 Target-Specific Feature Sets
- Allow different feature sets per target
- Store feature rankings per target in cache
- Use target-specific features during training

### Phase 3: Optimization & Polish

#### 3.1 Performance
- Parallelize ranking across targets
- Parallelize feature selection across targets
- Optimize data loading and caching

#### 3.2 Monitoring
- Add progress bars for ranking/selection
- Log ranking/selection metrics
- Track cache hit rates

#### 3.3 Documentation
- Update `docs/03_technical/implementation/` with integration guide
- Add examples to `TRAINING/examples/`
- Update `ROADMAP.md` with completion status

---

## Implementation Plan

### Step 1: Extract Core Logic (Week 1)
- [ ] Extract target ranking logic from `SCRIPTS/rank_target_predictability.py` into `TRAINING/ranking/target_ranker.py`
- [ ] Extract feature selection logic from `SCRIPTS/multi_model_feature_selection.py` into `TRAINING/ranking/feature_selector.py`
- [ ] Create shared utilities for data loading and caching
- [ ] Write unit tests for extracted modules

### Step 2: Create Configuration (Week 1)
- [ ] Create `CONFIG/training_config/target_ranking_config.yaml`
- [ ] Update `CONFIG/config_loader.py` to load ranking configs
- [ ] Add config validation

### Step 3: Basic Integration (Week 2)
- [ ] Add `--auto-targets` and `--auto-features` flags to `train_with_strategies.py`
- [ ] Implement `run_target_ranking()` function
- [ ] Implement `run_feature_selection()` function
- [ ] Integrate into `train_models_for_interval_comprehensive()`
- [ ] Add basic caching (file-based)

### Step 4: Testing & Validation (Week 2)
- [ ] Test with small dataset (5 symbols, 3 targets)
- [ ] Verify rankings match standalone script outputs
- [ ] Verify feature selections match standalone script outputs
- [ ] Test cache hit/miss scenarios
- [ ] Test with existing manual target/feature lists (backward compatibility)

### Step 5: Advanced Features (Week 3)
- [ ] Implement shared CV folds
- [ ] Implement incremental updates
- [ ] Add target-specific feature sets
- [ ] Optimize data loading

### Step 6: Documentation & Polish (Week 3)
- [ ] Update documentation
- [ ] Add examples
- [ ] Update ROADMAP.md
- [ ] Performance benchmarking

---

## Technical Considerations

### Data Loading
- **Current**: Each script loads data independently
- **Proposed**: Shared data loader with caching
- **Challenge**: Memory usage with large datasets
- **Solution**: Lazy loading, chunking, or sampling

### Caching Strategy
- **Location**: `results/target_rankings/`, `results/feature_selections/`
- **Format**: JSON for metadata, CSV for rankings, pickle for complex objects
- **Invalidation**: Based on data timestamps, config changes, or TTL
- **Key**: `{symbols_hash}_{targets_hash}_{config_hash}`

### Backward Compatibility
- **Requirement**: Existing workflows must continue to work
- **Solution**: Make ranking/selection opt-in via flags/config
- **Default**: Disabled (current behavior)

### Performance Impact
- **Concern**: Ranking/selection adds significant time to training
- **Mitigation**: 
  - Cache results aggressively
  - Run ranking/selection in parallel
  - Use fast configs (fewer models, fewer samples)
  - Make it optional

### Memory Management
- **Concern**: Loading data for ranking + selection + training
- **Solution**:
  - Reuse same data objects
  - Clear intermediate results
  - Use memory-efficient data structures (Polars?)

---

## Configuration Examples

### Example 1: Full Auto Mode
```bash
python TRAINING/train_with_strategies.py \
    --interval 5m \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --auto-targets \
    --top-n-targets 5 \
    --auto-features \
    --top-m-features 100 \
    --families lightgbm,xgboost,mlp
```

### Example 2: Auto Targets, Manual Features
```bash
python TRAINING/train_with_strategies.py \
    --interval 5m \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --auto-targets \
    --top-n-targets 5 \
    --features feature1,feature2,feature3 \
    --families lightgbm,xgboost
```

### Example 3: Manual Targets, Auto Features
```bash
python TRAINING/train_with_strategies.py \
    --interval 5m \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --targets fwd_ret_5m,fwd_ret_15m \
    --auto-features \
    --top-m-features 50 \
    --families lightgbm
```

### Example 4: Config File Mode
```yaml
# CONFIG/training_config/target_ranking_config.yaml
target_ranking:
  enabled: true
  top_n_targets: 5
  min_predictability_score: 0.15

feature_selection:
  enabled: true
  top_m_features: 100
```

```bash
python TRAINING/train_with_strategies.py \
    --interval 5m \
    --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
    --config CONFIG/training_config/target_ranking_config.yaml
```

---

## Success Criteria

### Functional
- [ ] Target ranking runs automatically before training (when enabled)
- [ ] Feature selection runs automatically per target (when enabled)
- [ ] Selected targets/features are used for training
- [ ] Results match standalone script outputs
- [ ] Backward compatibility maintained (manual targets/features still work)

### Performance
- [ ] Caching reduces re-computation time by >80%
- [ ] Ranking/selection adds <30% overhead to total training time (with cache)
- [ ] Memory usage stays within acceptable limits

### Usability
- [ ] Clear progress indicators for ranking/selection
- [ ] Helpful error messages if ranking/selection fails
- [ ] Documentation is complete and accurate
- [ ] Examples work out of the box

---

## Risks & Mitigations

### Risk 1: Performance Degradation
- **Risk**: Ranking/selection adds too much time
- **Mitigation**: Aggressive caching, fast configs, optional feature

### Risk 2: Memory Issues
- **Risk**: Loading data multiple times causes OOM
- **Mitigation**: Shared data loading, lazy evaluation, sampling

### Risk 3: Breaking Changes
- **Risk**: Integration breaks existing workflows
- **Mitigation**: Opt-in design, extensive testing, backward compatibility

### Risk 4: Cache Invalidation
- **Risk**: Stale rankings/features used after data changes
- **Mitigation**: TTL-based invalidation, data hash checking, force refresh flag

---

## Dependencies

### External
- Existing `SCRIPTS/rank_target_predictability.py` (logic source)
- Existing `SCRIPTS/multi_model_feature_selection.py` (logic source)
- Config loader (`CONFIG/config_loader.py`)
- Data loading utilities (`TRAINING/data_processing/`)

### Internal
- Training pipeline (`TRAINING/train_with_strategies.py`)
- Model families (all trainers)
- Cross-sectional data utilities

---

## Related Documents

- `docs/03_technical/research/TARGET_DISCOVERY.md` - Target discovery methodology
- `docs/internal/research/TARGET_RECOMMENDATIONS.md` - Target recommendations
- `docs/internal/research/TARGET_MODEL_PIPELINE_ANALYSIS.md` - Pipeline analysis
- `CONFIG/fast_target_ranking.yaml` - Fast ranking config
- `CONFIG/multi_model_feature_selection.yaml` - Feature selection config
- `ROADMAP.md` - Project roadmap (update after completion)

---

## Notes

- This integration aligns with Phase 1 "Intelligent Training Framework" in ROADMAP.md
- Consider future integration with walk-forward validation
- May want to add ranking/selection metrics to training reports
- Consider A/B testing framework for ranking/selection strategies

---

**Last Updated**: 2025-12-07  
**Next Review**: After Phase 1 completion

