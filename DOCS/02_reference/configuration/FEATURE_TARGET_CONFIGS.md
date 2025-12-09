# Feature & Target Configuration Guide

Complete guide to configuring features and targets in FoxML Core.

## Overview

Feature and target configuration files control what data is used for training, which features are safe for which targets, and which targets are enabled.

## Configuration Files

### `excluded_features.yaml`

**Purpose:** Defines patterns for features that are always excluded from training.

**When to use:** When you need to permanently exclude leaky features or feature patterns.

**Structure:**
```yaml
always_exclude:
  regex_patterns:
    - "^y_"           # All target columns
    - "^fwd_ret_"     # Forward returns (future info)
    - "^barrier_"     # Barrier-related features
  prefix_patterns:
    - "barrier_"
    - "mdd_"
  exact_patterns:
    - "my_leaky_feature"
```

**Key Patterns:**
- `^y_*` - All target columns
- `^fwd_ret_*` - Forward returns (future information)
- `^barrier_*` - Barrier-related features
- `^mfe_*`, `^mdd_*` - Maximum favorable/adverse excursion

**Auto-Fixer Integration:** Auto-fixer automatically adds patterns here when leakage is detected.

**Example: Excluding a Leaky Feature**

```yaml
always_exclude:
  exact_patterns:
    - "future_price"  # Exact feature name
```

**Example: Excluding a Pattern**

```yaml
always_exclude:
  regex_patterns:
    - "^future_"  # All features starting with "future_"
```

---

### `feature_registry.yaml`

**Purpose:** Defines temporal metadata for features to prevent leakage based on target horizon.

**When to use:** When adding new features or adjusting which features are safe for which target horizons.

**Structure:**
```yaml
features:
  ret_1:
    source: price
    lag_bars: 1
    allowed_horizons: [1, 2, 3, 5, 12, 24, 60]
    description: "1-bar lagged return"
  ret_5:
    source: price
    lag_bars: 5
    allowed_horizons: [5, 12, 24, 60]
    description: "5-bar lagged return"
```

**Key Fields:**
- `source` - Data source (price, volume, etc.)
- `lag_bars` - Number of bars lagged (must be ≥ 0, cannot be negative)
- `allowed_horizons` - List of target horizons this feature is safe for
- `rejected` - If true, feature is rejected (with reason)
- `description` - Human-readable description

**How It Works:**
- Features are filtered based on target horizon
- Only features with `allowed_horizons` including the target's horizon are used
- Features with `rejected: true` are always excluded

**Example: Adding a New Feature**

```yaml
features:
  my_custom_feature:
    source: price
    lag_bars: 3
    allowed_horizons: [5, 12, 24]  # Safe for 5, 12, 24-bar horizons
    description: "3-bar momentum indicator"
```

**Example: Rejecting a Feature**

```yaml
features:
  leaky_feature:
    rejected: true
    reason: "Contains future information"
```

**Important Rules:**
- `lag_bars` must be ≥ 0 (negative values indicate future information)
- `allowed_horizons` must include horizons where `lag_bars < horizon`
- Features without `allowed_horizons` are rejected by default

---

### `feature_target_schema.yaml`

**Purpose:** Explicitly defines which columns are metadata, targets, or features.

**When to use:** When you need to control how columns are classified or adjust ranking vs. training mode rules.

**Structure:**
```yaml
# Metadata columns - always excluded from features
metadata_columns:
  - symbol
  - interval
  - source
  - ts
  - timestamp

# Target column patterns - these are targets, not features
target_patterns:
  - "^y_will_peak"
  - "^y_will_valley"
  - "^fwd_ret_"

# Feature families with mode-specific rules
feature_families:
  ohlcv:
    ranking_mode:
      always_include: true  # Always include in ranking
    training_mode:
      strict_filtering: true  # Apply all filters in training
```

**Modes:**
- **Ranking Mode:** More permissive rules for target ranking
  - Allows basic OHLCV/TA features even if in `always_exclude`
  - Registry is advisory (unknown features allowed if they pass pattern filtering)
  
- **Training Mode:** Strict rules for actual training
  - Enforces all leakage filters strictly
  - Registry is required (unknown features rejected)

**Example: Adding Metadata Column**

```yaml
metadata_columns:
  - symbol
  - ts
  - my_custom_metadata_column  # Add new metadata column
```

**Example: Adding Target Pattern**

```yaml
target_patterns:
  - "^y_will_peak"
  - "^my_custom_target_"  # New target pattern
```

---

### `target_configs.yaml`

**Purpose:** Defines all available targets (63 total) and their settings.

**When to use:** When enabling/disabling targets or adjusting target-specific settings.

**Structure:**
```yaml
targets:
  peak_60m:
    target_column: "y_will_peak_60m_0.8"
    description: "Predict upward barrier hits (peaks) at 60m horizon"
    use_case: "Long entry signals, profit target optimization"
    top_n: 60
    method: "mean"
    enabled: true
```

**Key Fields:**
- `target_column` - Column name in dataset
- `description` - What the target predicts
- `use_case` - Trading use case
- `top_n` - Number of top features to select
- `method` - Feature selection method
- `enabled` - Enable/disable flag

**Target Categories:**
- **Triple Barrier:** peak, valley, first_touch
- **Swing High/Low:** swing_high, swing_low
- **MFE:** Maximum Favorable Excursion
- **MDD:** Maximum Drawdown

**Example: Enabling a Target**

```yaml
targets:
  swing_high_15m:
    enabled: true  # Change from false to true
    top_n: 50
    method: "mean"
```

**Example: Adjusting Target Settings**

```yaml
targets:
  peak_60m:
    top_n: 100  # Increase from 60
    method: "weighted"  # Change selection method
```

---

### `multi_model_feature_selection.yaml` (Feature Selection)

**Purpose:** Configures multi-model consensus for feature selection.

**When to use:** When adjusting which models participate in feature selection or their weights.

**Location:**
- **NEW:** `CONFIG/feature_selection/multi_model.yaml` (preferred)
- **LEGACY:** `CONFIG/multi_model_feature_selection.yaml` (deprecated, still works)

**Note:** For new projects, use experiment configs (see [Modular Config System](MODULAR_CONFIG_SYSTEM.md)) instead of editing this file directly. The config builder automatically merges experiment config overrides with this module config.

**Structure:**
```yaml
model_families:
  lightgbm:
    enabled: true
    importance_method: "native"  # native/SHAP/permutation
    weight: 1.0
    config:
      n_estimators: 300
      learning_rate: 0.05
      # ... model hyperparameters
```

**Model Families:**
- **Tree-based:** LightGBM, XGBoost, Random Forest, CatBoost
- **Neural:** MLP, Transformer, LSTM, CNN1D
- **Statistical/Wrapper:** Lasso, Mutual Information, Univariate Selection, RFE, Boruta (gatekeeper), Stability Selection
- **Ensemble:** Ensemble, MultiTask

**Note on Boruta:**
Boruta is implemented as a **statistical gatekeeper**, not just another importance scorer. It:
- Uses ExtraTrees (more random than RF) with stability-oriented hyperparams (`n_estimators: 500`, `max_depth: 6`, `perc: 95`)
- Is excluded from base consensus calculation
- Modifies final consensus via bonuses/penalties (confirmed: +0.2, rejected: -0.3, tentative: neutral)
- Provides separate columns in output: `consensus_score_base`, `consensus_score` (final), `boruta_gate_effect`
- Uses `train_score = math.nan` (not a numeric score) since it's a selector, not a predictor - prevents feature count mismatch errors
- See [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md#boruta-statistical-gatekeeper) for details

**Example: Adjusting Model Weights**

Edit `CONFIG/feature_selection/multi_model.yaml` (or use experiment config override):

```yaml
model_families:
  lightgbm:
    enabled: true
    weight: 1.5  # Increase weight (more influence)
  random_forest:
    enabled: true
    weight: 1.0
  neural_network:
    enabled: false  # Disable this model family
```

**Or override in experiment config:**
```yaml
# CONFIG/experiments/my_experiment.yaml
feature_selection:
  model_families: [lightgbm, xgboost]  # Override enabled families
```

**Example: Changing Importance Method**

```yaml
model_families:
  lightgbm:
    importance_method: "SHAP"  # Use SHAP instead of native
```

**Target Confidence & Routing:**

The multi-model config includes automatic target quality assessment and routing:

```yaml
confidence:
  # Confidence thresholds (HIGH/MEDIUM/LOW)
  high:
    boruta_confirmed_min: 5
    agreement_ratio_min: 0.4
    mean_score_min: 0.05
    model_coverage_min: 0.7
  
  # Score tier thresholds (signal strength, orthogonal to confidence)
  score_tier:
    high:
      mean_strong_score_min: 0.08
      max_score_min: 0.70
    medium:
      mean_strong_score_min: 0.03
      max_score_min: 0.55
  
  # Routing rules (confidence + score_tier → operational buckets)
  routing:
    experimental:
      confidence: "LOW"
      low_confidence_reason: "boruta_zero_confirmed"
    core:
      confidence: "HIGH"
    candidate:
      confidence: "MEDIUM"
      score_tier_min: "MEDIUM"
```

**Output Artifacts:**
- Per-target: `target_confidence.json`, `target_routing.json`
- Run-level: `target_confidence_summary.json`, `target_confidence_summary.csv`

See [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md#target-confidence-and-routing) for details.

---

### `feature_selection_config.yaml`

**Purpose:** General feature selection settings.

**When to use:** When adjusting feature selection criteria or methods.

**Settings:**
- Feature importance aggregation methods
- Selection criteria
- Minimum feature requirements

---

### `feature_groups.yaml`

**Purpose:** Defines feature groups for organization and analysis.

**When to use:** When organizing features into logical groups.

---

### `comprehensive_feature_ranking.yaml` & `fast_target_ranking.yaml`

**Purpose:** Alternative ranking configurations for different use cases.

**When to use:**
- `comprehensive_feature_ranking.yaml` - Full ranking with all models
- `fast_target_ranking.yaml` - Faster ranking with fewer models

---

## Common Workflows

### Workflow 1: Adding a New Feature

1. **Add to `feature_registry.yaml`:**
```yaml
features:
  my_new_feature:
    source: price
    lag_bars: 3
    allowed_horizons: [5, 12, 24, 60]
    description: "3-bar momentum"
```

2. **Verify not excluded:**
   - Check `excluded_features.yaml` - ensure no patterns match
   - Check `feature_target_schema.yaml` - ensure not in metadata/target patterns

3. **Feature is now available** for targets with horizons 5, 12, 24, or 60 bars

### Workflow 2: Excluding a Leaky Feature

1. **Add to `excluded_features.yaml`:**
```yaml
always_exclude:
  exact_patterns:
    - my_leaky_feature
```

2. **Or add pattern if multiple features:**
```yaml
always_exclude:
  regex_patterns:
    - "^future_"
```

### Workflow 3: Enabling More Targets

1. **Edit `target_configs.yaml`:**
```yaml
targets:
  swing_high_15m:
    enabled: true  # Change from false
```

2. **Targets are automatically discovered** and ranked in the intelligent training pipeline

### Workflow 4: Adjusting Feature Selection

1. **Edit `multi_model_feature_selection.yaml`:**
```yaml
model_families:
  lightgbm:
    weight: 1.5  # Increase influence
  neural_network:
    enabled: false  # Disable
```

2. **Feature selection will use updated weights** in next run

---

## Best Practices

1. **Always specify `lag_bars` and `allowed_horizons`** for new features
2. **Use `exact_patterns` over `regex_patterns`** when possible (more precise)
3. **Check horizon compatibility** - feature's `allowed_horizons` must include target's horizon
4. **Review excluded features** before adding new features
5. **Document custom features** with clear descriptions
6. **Test feature availability** by checking logs during target ranking

---

## Related Documentation

- **[Modular Config System](MODULAR_CONFIG_SYSTEM.md)** - Complete guide to modular configs (includes `logging_config.yaml`)
- [Configuration System Overview](README.md) - Main configuration overview (includes `logging_config.yaml` documentation)
- [Usage Examples](USAGE_EXAMPLES.md) - Practical configuration examples (includes interval config and CatBoost examples)
- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior guide
- [Training Pipeline Configs](TRAINING_PIPELINE_CONFIGS.md) - Training configuration
- [Safety & Leakage Configs](SAFETY_LEAKAGE_CONFIGS.md) - Leakage detection settings
- [Leakage Analysis](../../03_technical/research/LEAKAGE_ANALYSIS.md) - Detailed leakage documentation
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - Complete pipeline guide

