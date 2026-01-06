# Feature Selection Guide

Feature selection methodology and implementation.

**NEW (2025-12-13)**: Feature selection now uses the same shared ranking harness as target ranking, ensuring identical evaluation contracts, comprehensive hardening, and consistent output structure. See [Feature Selection Unification Changelog](../../02_reference/changelog/2025-12-13-feature-selection-unification.md) for details.

## Key Features

### Shared Ranking Harness
- **Same Evaluation Contract**: Uses `RankingHarness` class shared with target ranking
- **Same Split Policy**: PurgedTimeSeriesSplit with time-based purging
- **Same Cleaning Checks**: Ghost busters, leak scan, target validation, duplicate detection
- **Same Config System**: Uses same config hierarchy and loading methods as target ranking
- **Same Stability Tracking**: Per-model snapshots + aggregated consensus snapshots
- **Same Leak Detection**: Saves `leak_detection_summary.txt` in same format as target ranking

### Comprehensive Hardening
- **Linear Models**: Lasso, Ridge, and ElasticNet enabled (same as target ranking)
- **Pre-training Leak Scan**: Detects near-copy features before training
- **Target-Conditional Exclusions**: Per-target exclusion lists tailored to target physics
- **Final Gatekeeper**: Drops problematic features before training starts
- **SST Enforcement**: Uses `EnforcedFeatureSet` contract for provably split-brain free enforcement
- **Stability Analysis**: Calls `analyze_all_stability_hook()` at end of run

SST Enforcement Design ensures feature selection uses the same `EnforcedFeatureSet` contract as target ranking, with immediate X slicing and boundary assertions.

### Output Structure
- **Same Format**: CSV and YAML files match target ranking format
- **Same Reproducibility**: REPRODUCIBILITY/FEATURE_SELECTION/ structure matches TARGET_RANKING/
- **Same Artifacts**: Feature importances, stability snapshots, leak detection summaries

## Implementation Status

### Step 1: SingleTaskStrategy with Early Stopping
Location: `TRAINING/strategies/single_task.py`

Validation split and early stopping implemented:

```python
# Automatic validation split and early stopping
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50, verbose=False)]
)
logger.info(f"Early stopping at iteration {model.best_iteration_}")
```

Configuration (from `first_batch_specs.yaml`):
```yaml
lightgbm:
  max_depth: 8              # Prevents deep, overfit trees
  subsample: 0.75           # Row sampling (was 1.0)
  colsample_bytree: 0.75    # Feature sampling (was 1.0)
  reg_alpha: 0.1            # L1 regularization
  reg_lambda: 0.1           # L2 regularization
  learning_rate: 0.03       # Slow learning
  n_estimators: 1000        # Will stop early
  early_stopping_rounds: 50
```

### Step 2: MultiTaskStrategy with Active Dropout
Location: `TRAINING/strategies/multi_task.py`

Dropout active during training, disabled during inference:

```python
# Training: Dropout ACTIVE
self.shared_model.train()
for batch_X, batch_y in dataloader:
    predictions = self.shared_model(batch_X)  # Dropout drops units

# Inference: Dropout DISABLED
self.shared_model.eval()
with torch.no_grad():
    predictions = self.shared_model(X_test)  # All units active
```

Early stopping implemented:
```python
patience = 10
if patience_counter >= patience:
    logger.info(f"Early stopping at epoch {epoch}")
    model.load_state_dict(best_model_state)  # Restore best
    break
```

## Step 3: Feature Selection

### Reduce from 421 to 50 Features

```python
from strategies.single_task import SingleTaskStrategy
from utils.feature_selection import get_feature_importance_from_strategy, select_top_features

# 1. Train on ALL features to get importance
config = {
    'models': {'lightgbm': {'max_depth': 8, 'learning_rate': 0.03}},
    'early_stopping_rounds': 50
}

strategy = SingleTaskStrategy(config)
strategy.train(X, {'fwd_ret_5m': y}, feature_names)

# 2. Get feature importance (sorted)
sorted_features, sorted_importances = get_feature_importance_from_strategy(
    strategy, 'fwd_ret_5m', feature_names
)

# 3. Select top 50 features
top_50_features = sorted_features[:50]
top_50_indices = [feature_names.index(f) for f in top_50_features]
X_selected = X[:, top_50_indices]

# 4. Retrain on selected features
strategy_final = SingleTaskStrategy(config)
strategy_final.train(X_selected, y_dict, top_50_features)
```

### Complete Workflow

See `TRAINING/examples/feature_selection_workflow.py` for the full workflow:

```python
from examples.feature_selection_workflow import feature_selection_workflow

# Run complete workflow
X_selected, selected_features, strategy = feature_selection_workflow(
    X, y_dict, feature_names, config,
    n_features=50,
    primary_target='fwd_ret_5m'
)

# Now use X_selected for all models
```

## Step 4: Simplified Workflow

### Phase 1: Core Predictors

```python
# 1. Feature selection (reduce 421 → 50)
X_selected, selected_features, _ = feature_selection_workflow(
    X, y_dict, feature_names, config, n_features=50
)

# 2. (Optional) Add GMM regime as feature 51
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(X_selected[:, :5])  # Train on first 5 features
regime = gmm.predict(X_selected).reshape(-1, 1)

X_final = np.column_stack([X_selected, regime])
features_final = selected_features + ['gmm_regime']

# 3. Train LightGBM (Single-Task)
from strategies.single_task import SingleTaskStrategy

config_lgbm = {
    'models': {
        'lightgbm': {
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
    },
    'early_stopping_rounds': 50,
}

strategy_lgbm = SingleTaskStrategy(config_lgbm)
results_lgbm = strategy_lgbm.train(X_final, y_dict, features_final)

# 4. Train MultiTask NN (correlated targets only)
from strategies.multi_task import MultiTaskStrategy

# ONLY use highly correlated targets
y_correlated = {
    'mfe_5m': y_dict['mfe_5m'],
    'mdd_5m': y_dict['mdd_5m'],
    'tth_5m': y_dict['tth_5m'],
    'fwd_ret_5m': y_dict['fwd_ret_5m'],
}

config_mtl = {
    'model': {'shared_dim': 128, 'learning_rate': 0.001},
    'batch_size': 32,
    'n_epochs': 100,
    'patience': 10,
}

strategy_mtl = MultiTaskStrategy(config_mtl)
results_mtl = strategy_mtl.train(X_final, y_correlated, features_final)
```

### Phase 2: Advanced Models

Proceed after Phase 1 is working.

```python
# Ensemble (uses LightGBM as a base)
from model_fun.ensemble_trainer import EnsembleTrainer

ensemble = EnsembleTrainer(config)
ensemble.train(X_final, y_dict['fwd_ret_5m'])

# QuantileLightGBM (prediction ranges)
from model_fun.quantile_lightgbm_trainer import QuantileLightGBMTrainer

# Train TWO models: lower and upper bounds
config_q05 = {'alpha': 0.05}  # 5th percentile
config_q95 = {'alpha': 0.95}  # 95th percentile

q05 = QuantileLightGBMTrainer(config_q05)
q05.train(X_final, y_dict['fwd_ret_5m'])

q95 = QuantileLightGBMTrainer(config_q95)
q95.train(X_final, y_dict['fwd_ret_5m'])

# Prediction range: [q05.predict(X), q95.predict(X)]
```

### Phase 3: Feature Engineering

Defer until Phases 1-2 are solid:
- VAE: Complex, requires research
- GAN: Complex, requires research
- NGBoost: Slow, use after LightGBM works
- FTRL, RewardBased: Specialized, different framework

## Backward Compatibility

All existing functionality preserved.

### Option 1: Use with ALL features (existing behavior)
```python
strategy = SingleTaskStrategy(config)
strategy.train(X, y_dict, feature_names)  # Works as before
```

### Option 2: Use with SELECTED features
```python
# Feature selection is OPTIONAL
X_selected, selected_features, strategy = feature_selection_workflow(...)
strategy.train(X_selected, y_dict, selected_features)
```

### Option 3: Manual feature selection
```python
# You can still manually select features
selected_indices = [0, 5, 10, ...]  # Your choice
X_manual = X[:, selected_indices]
strategy.train(X_manual, y_dict, [feature_names[i] for i in selected_indices])
```

## Verification Checklist

### 1. Overfitting Fixed?
```python
# After training, compare scores
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
print(f"Train: {train_score:.4f}, Val: {val_score:.4f}")

# GOOD: Scores should be close (within 0.05)
# BAD: train_score >> val_score means overfitting
```

### 2. Early Stopping Working?
```python
# Check logs for messages like:
# " fwd_ret_5m: Early stopping at iteration 234"

# Check model iteration count
if hasattr(model, 'best_iteration_'):
    print(f"Stopped at iteration {model.best_iteration_} (out of {model.n_estimators})")
    # GOOD: Should stop well before n_estimators
```

### 3. Dropout Active?
```python
# For PyTorch models
model.train()  # Should enable Dropout
print(model.encoder.training)  # Should be True

model.eval()   # Should disable Dropout
print(model.encoder.training)  # Should be False
```

### 4. Feature Selection Effective?
```python
# Compare performance before/after feature selection
# With 421 features vs 50 features

# GOOD indicators:
# - Validation score improves or stays similar
# - Training time decreases significantly
# - Model complexity reduced (fewer overfitting risks)
```

## Implementation Files

1. `TRAINING/utils/feature_selection.py`
   - `select_top_features()`: Select top N features by importance
   - `get_feature_importance_from_strategy()`: Extract importance from trained model
   - `create_feature_report()`: Generate CSV report with rankings
   - `auto_select_features()`: End-to-end feature selection

2. `TRAINING/examples/feature_selection_workflow.py`
   - Complete workflow demonstrating Steps 3-4
   - Shows how to reduce from 421 to 50 features
   - Includes optional GMM regime feature

3. `TRAINING/strategies/base.py` (enhanced)
   - `get_feature_importance()`: Now supports all targets or specific target
   - Returns dict mapping targets to importance arrays

## Related Documentation

- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior (includes sklearn preprocessing)
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Automated feature selection workflow
- [Feature Selection Tutorial](../../01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md) - Manual feature selection tutorial
- [Training Optimization Guide](TRAINING_OPTIMIZATION_GUIDE.md) - Optimization guide
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Config system guide
