# Main Training Script Outline

**Status**: Planning  
**Created**: 2025-12-07  
**Goal**: Create a single entry point (`TRAINING/train.py`) that automatically ranks targets, selects features, and trains models in one seamless pipeline

---

## Overview

The main training script (`TRAINING/train.py`) will be the primary entry point for all training workflows. It intelligently orchestrates:
1. **Target Ranking** - Automatically identifies the most predictable targets
2. **Feature Selection** - Automatically selects the best features per target
3. **Model Training** - Trains all model families on the selected targets/features

All steps are automatic by default, fully configurable via YAML configs, and preserve all existing leakage-free behavior.

---

## Architecture

### Entry Point
```
TRAINING/train.py
    ↓
TRAINING/orchestration/intelligent_trainer.py (IntelligentTrainer class)
    ↓
    ├─→ TRAINING/ranking/target_ranker.py (target ranking)
    ├─→ TRAINING/ranking/feature_selector.py (feature selection)
    └─→ TRAINING/train_with_strategies.py (model training)
```

### File Structure
```
TRAINING/
├── train.py                          # [NEW] Main entry point
├── orchestration/
│   └── intelligent_trainer.py        # [EXISTS] Orchestrator class
├── ranking/
│   ├── target_ranker.py              # [EXISTS] Target ranking logic
│   └── feature_selector.py           # [EXISTS] Feature selection logic
└── train_with_strategies.py          # [EXISTS] Training pipeline (unchanged)
```

---

## Functionality

### Phase 1: Target Ranking (Automatic)

**When**: Always runs if `--auto-targets` (default: enabled)

**Process**:
1. Discover all available targets from data (or load from config)
2. Evaluate each target using multiple model families:
   - LightGBM, XGBoost, Random Forest, Neural Network
   - Cross-validated with PurgedTimeSeriesSplit (leakage-free)
   - Calculate composite predictability scores
3. Rank targets by composite score
4. Select top N targets (default: 5, configurable)
5. Cache results to avoid re-computation

**Output**:
- `output_dir/cache/target_rankings.json` - Cached rankings
- `output_dir/target_rankings/target_predictability_rankings.csv` - Full rankings
- List of selected target names

**Config**: `CONFIG/training_config/target_ranking_config.yaml`

**Key Functions**:
- `TRAINING/ranking/target_ranker.rank_targets()`
- `TRAINING/ranking/target_ranker.discover_targets()`

---

### Phase 2: Feature Selection (Automatic)

**When**: Always runs if `--auto-features` (default: enabled)

**Process**:
1. For each selected target:
   - Load cross-sectional data
   - Train multiple model families (LightGBM, Random Forest, Neural Network)
   - Extract feature importance (native/SHAP/permutation)
   - Aggregate importance across models and symbols
   - Rank features by consensus score
2. Select top M features per target (default: 100, configurable)
3. Cache results per target

**Output**:
- `output_dir/cache/feature_selections/{target}.json` - Cached selections
- `output_dir/feature_selections/{target}/selected_features.txt` - Feature list
- `output_dir/feature_selections/{target}/feature_importance_multi_model.csv` - Full rankings
- Dict mapping `{target: [feature_names]}`

**Config**: `CONFIG/multi_model_feature_selection.yaml`

**Key Functions**:
- `TRAINING/ranking/feature_selector.select_features_for_target()`

---

### Phase 3: Model Training (Automatic)

**When**: Always runs after ranking/selection

**Process**:
1. Prepare training data:
   - Load MTF data for all symbols
   - Use selected targets and features
   - Build cross-sectional datasets
2. Train all model families:
   - CPU families: LightGBM, XGBoost, NGBoost, etc.
   - GPU families: MLP, VAE, GAN, MetaLearning, MultiTask
   - Sequential families: CNN1D, LSTM, Transformer, etc.
3. Use existing training pipeline:
   - Isolation for GPU families
   - Threading control
   - Memory management
   - All existing safeguards

**Output**:
- `output_dir/training_results/` - Model artifacts, metrics, predictions
- Same structure as existing `train_with_strategies.py` output

**Key Functions**:
- `TRAINING/train_with_strategies.train_models_for_interval_comprehensive()`

---

## Command-Line Interface

### Basic Usage (Fully Automatic)
```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL
```

### With Overrides
```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT GOOGL \
    --top-n-targets 10 \
    --top-m-features 50 \
    --families lightgbm xgboost mlp \
    --strategy single_task
```

### Manual Mode (Disable Auto Features)
```bash
python TRAINING/train.py \
    --data-dir data/data_labeled/interval=5m \
    --symbols AAPL MSFT \
    --targets fwd_ret_5m fwd_ret_15m \
    --features feat1 feat2 feat3 \
    --no-auto-targets \
    --no-auto-features
```

### Arguments

**Required**:
- `--data-dir`: Data directory path
- `--symbols`: List of symbols to train on

**Target Selection**:
- `--auto-targets`: Enable automatic target ranking (default: True)
- `--no-auto-targets`: Disable automatic target ranking
- `--top-n-targets`: Number of top targets to select (default: 5, from config)
- `--targets`: Manual target list (overrides auto-targets if provided)

**Feature Selection**:
- `--auto-features`: Enable automatic feature selection (default: True)
- `--no-auto-features`: Disable automatic feature selection
- `--top-m-features`: Number of top features per target (default: 100, from config)
- `--features`: Manual feature list (overrides auto-features if provided)

**Training**:
- `--families`: Model families to train (default: all enabled, from config)
- `--strategy`: Training strategy - single_task, multi_task, cascade (default: single_task)
- `--output-dir`: Output directory (default: intelligent_output)

**Cache Control**:
- `--force-refresh`: Force refresh of cached rankings/selections
- `--no-cache`: Disable caching entirely

**Config Files**:
- `--target-ranking-config`: Path to target ranking config (default: CONFIG/training_config/target_ranking_config.yaml)
- `--multi-model-config`: Path to feature selection config (default: CONFIG/multi_model_feature_selection.yaml)

**Pass-through Arguments**:
- All other arguments from `train_with_strategies.py` are supported:
  - `--min-cs`, `--max-rows-train`, `--threads`, `--cpu-only`, etc.

---

## Configuration

### Default Config Files

1. **Target Ranking**: `CONFIG/training_config/target_ranking_config.yaml`
   ```yaml
   target_ranking:
     enabled: true
     top_n_targets: 5
     min_predictability_score: 0.1
     model_families:
       - lightgbm
       - xgboost
       - random_forest
     max_samples_per_symbol: 10000
     cv_folds: 3
   ```

2. **Feature Selection**: `CONFIG/multi_model_feature_selection.yaml`
   ```yaml
   model_families:
     lightgbm:
       enabled: true
     random_forest:
       enabled: true
     neural_network:
       enabled: true
   aggregation:
     require_min_models: 2
   ```

3. **Training Pipeline**: `CONFIG/training_config/pipeline_config.yaml`
   - Already exists, used for training parameters

### Config Loading Priority

1. Command-line arguments (highest priority)
2. Config file specified via `--config` flag
3. Default config files in `CONFIG/` directory
4. Hardcoded defaults (lowest priority)

---

## Implementation Details

### Main Script Structure (`TRAINING/train.py`)

```python
#!/usr/bin/env python3
"""
Main Training Script

Automatically ranks targets, selects features, and trains models.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from TRAINING.orchestration import IntelligentTrainer

def main():
    parser = argparse.ArgumentParser(
        description='Intelligent Training Pipeline: Auto-rank targets, select features, train models',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--data-dir', type=Path, required=True, ...)
    parser.add_argument('--symbols', nargs='+', required=True, ...)
    
    # Target selection arguments
    parser.add_argument('--auto-targets', action='store_true', default=True, ...)
    parser.add_argument('--no-auto-targets', dest='auto_targets', action='store_false', ...)
    parser.add_argument('--top-n-targets', type=int, ...)
    parser.add_argument('--targets', nargs='+', ...)
    
    # Feature selection arguments
    parser.add_argument('--auto-features', action='store_true', default=True, ...)
    parser.add_argument('--no-auto-features', dest='auto_features', action='store_false', ...)
    parser.add_argument('--top-m-features', type=int, ...)
    parser.add_argument('--features', nargs='+', ...)
    
    # Training arguments
    parser.add_argument('--families', nargs='+', ...)
    parser.add_argument('--strategy', choices=['single_task', 'multi_task', 'cascade'], ...)
    parser.add_argument('--output-dir', type=Path, default=Path('intelligent_output'), ...)
    
    # Cache control
    parser.add_argument('--force-refresh', action='store_true', ...)
    parser.add_argument('--no-cache', action='store_true', ...)
    
    # Config files
    parser.add_argument('--target-ranking-config', type=Path, ...)
    parser.add_argument('--multi-model-config', type=Path, ...)
    
    # Pass-through arguments (from train_with_strategies.py)
    parser.add_argument('--min-cs', type=int, ...)
    parser.add_argument('--max-rows-train', type=int, ...)
    parser.add_argument('--threads', type=int, ...)
    # ... etc
    
    args = parser.parse_args()
    
    # Create orchestrator
    trainer = IntelligentTrainer(
        data_dir=args.data_dir,
        symbols=args.symbols,
        output_dir=args.output_dir
    )
    
    # Run intelligent training
    results = trainer.train_with_intelligence(
        auto_targets=args.auto_targets,
        top_n_targets=args.top_n_targets,
        auto_features=args.auto_features,
        top_m_features=args.top_m_features,
        targets=args.targets,
        features=args.features,
        families=args.families,
        strategy=args.strategy,
        force_refresh=args.force_refresh,
        # Pass through other args
        min_cs=args.min_cs,
        max_rows_train=args.max_rows_train,
        threads=args.threads,
        # ... etc
    )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### Integration with Existing Code

**No changes needed to**:
- `TRAINING/train_with_strategies.py` - Called as-is
- `TRAINING/ranking/target_ranker.py` - Already extracted
- `TRAINING/ranking/feature_selector.py` - Already extracted
- All model trainers - Unchanged
- All common utilities - Unchanged

**Only additions**:
- `TRAINING/train.py` - New main entry point
- Complete `IntelligentTrainer.train_with_intelligence()` method - Currently placeholder

---

## Output Structure

```
intelligent_output/
├── cache/
│   ├── target_rankings.json              # Cached target rankings
│   └── feature_selections/
│       ├── fwd_ret_5m.json
│       ├── fwd_ret_15m.json
│       └── y_will_peak_60m_0.8.json
│
├── target_rankings/
│   ├── target_predictability_rankings.csv
│   ├── target_predictability_rankings.yaml
│   └── feature_importances/              # Per-model feature importances
│
├── feature_selections/
│   ├── fwd_ret_5m/
│   │   ├── selected_features.txt
│   │   ├── feature_importance_multi_model.csv
│   │   └── model_agreement_matrix.csv
│   └── fwd_ret_15m/
│       └── ...
│
└── training_results/
    ├── models/                           # Trained model artifacts
    ├── metrics/                          # Performance metrics
    ├── predictions/                      # Prediction outputs
    └── logs/                             # Training logs
```

---

## Error Handling

### Target Ranking Failures
- If ranking fails for a target → log warning, skip that target
- If all targets fail → raise error and exit
- If some targets fail → continue with successful targets

### Feature Selection Failures
- If selection fails for a target → log warning, use all available features
- If selection fails for all targets → raise error and exit

### Training Failures
- If a model family fails → log error, continue with other families
- If all families fail → raise error and exit
- Use existing error handling from `train_with_strategies.py`

---

## Caching Strategy

### Cache Keys
- Target rankings: `{symbols_hash}_{config_hash}`
- Feature selections: `{target}_{symbols_hash}_{config_hash}`

### Cache Invalidation
- **Time-based**: Configurable TTL (default: 7 days)
- **Data-based**: Check if data files are newer than cache
- **Config-based**: Hash config files, invalidate if changed
- **Force refresh**: `--force-refresh` flag

### Cache Benefits
- Ranking 50 targets: ~30 minutes → ~5 seconds (if cached)
- Feature selection per target: ~10 minutes → ~1 second (if cached)
- Total time savings: Hours → Seconds for repeated runs

---

## Testing Strategy

### Unit Tests
- Test `IntelligentTrainer` class methods individually
- Test cache loading/saving
- Test config loading and defaults

### Integration Tests
- Test full pipeline with small dataset (3 symbols, 2 targets)
- Test caching behavior
- Test error handling

### End-to-End Tests
- Test with real data (5 symbols, 5 targets)
- Verify output structure
- Verify model artifacts are created

---

## Migration Path

### Phase 1: Complete Orchestrator
- [ ] Complete `IntelligentTrainer.train_with_intelligence()` method
- [ ] Integrate with `train_with_strategies.py` functions
- [ ] Add pass-through argument handling

### Phase 2: Create Main Script
- [ ] Create `TRAINING/train.py`
- [ ] Add all argument parsing
- [ ] Add config loading logic
- [ ] Add error handling

### Phase 3: Testing
- [ ] Unit tests for orchestrator
- [ ] Integration tests for full pipeline
- [ ] End-to-end tests with real data

### Phase 4: Documentation
- [ ] Update main README
- [ ] Add usage examples
- [ ] Document config files
- [ ] Add troubleshooting guide

---

## Success Criteria

### Functional
- [ ] Single command runs full pipeline (ranking → selection → training)
- [ ] All existing functionality preserved
- [ ] Caching works correctly
- [ ] Config files are respected
- [ ] Error handling is robust

### Performance
- [ ] Caching reduces re-computation time by >80%
- [ ] Total pipeline time is reasonable (<2 hours for 5 symbols, 5 targets)
- [ ] Memory usage stays within limits

### Usability
- [ ] Simple default usage (just data-dir and symbols)
- [ ] Clear progress indicators
- [ ] Helpful error messages
- [ ] Complete documentation

---

## Related Documents

- `docs/internal/planning/TARGET_RANKING_SELECTION_INTEGRATION.md` - Detailed integration plan
- `TRAINING/orchestration/intelligent_trainer.py` - Orchestrator implementation
- `TRAINING/ranking/` - Ranking and selection modules
- `CONFIG/training_config/` - Configuration files

---

**Last Updated**: 2025-12-07  
**Next Steps**: Complete `IntelligentTrainer.train_with_intelligence()` method to integrate with training pipeline

