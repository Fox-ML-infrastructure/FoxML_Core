# Intelligence Layer Overview

Complete technical overview of the intelligent training pipeline's decision-making and automation systems.

## What is the Intelligence Layer?

The "intelligence layer" is the automated decision-making system that orchestrates target ranking, feature selection, leakage detection, and model training. It replaces manual, error-prone workflows with automated, reproducible pipelines.

**Core Components:**
1. **Target Ranking System** - Multi-model consensus ranking of prediction targets
2. **Feature Selection System** - Multi-model consensus feature importance aggregation
3. **Leakage Detection & Auto-Fix** - Automatic detection and remediation of data leakage
4. **Auto-Rerun Mechanism** - Automatic re-evaluation after fixes
5. **Config Backup System** - Automatic backups before modifications
6. **Caching System** - Intelligent caching for faster iterative development

## Decision Flow

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────┐
│                   1. Target Discovery                        │
│              (Scan data for available targets)               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   2. Target Ranking                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Pre-Scan    │→ │  Model       │→ │  Leakage     │     │
│  │  (Leak       │  │  Training    │  │  Detection   │     │
│  │  Detection)  │  │  (Multi-     │  │  & Auto-Fix  │     │
│  │              │  │   Model)     │  │              │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
│                       │                                     │
│                       ▼                                     │
│              ┌──────────────┐                              │
│              │  Auto-Rerun  │ (if leakage detected)        │
│              │  Loop        │                              │
│              └──────────────┘                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   3. Feature Selection                       │
│              (Per target, multi-model consensus)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   4. Model Training                          │
│              (Train all models with selected features)      │
└─────────────────────────────────────────────────────────────┘
```

### Detailed Decision Flow: Target Ranking

For each target, the system follows this decision tree:

```
1. Load target data
   ↓
2. Filter features (leakage-safe filtering)
   ├─→ Insufficient features? → Mark as INSUFFICIENT_FEATURES, skip
   ↓
3. Pre-training leak scan
   ├─→ Near-copy features found? → Remove them
   ├─→ Too few features remain? → Mark as LEAKAGE_DETECTED, skip
   ↓
4. Train models (LightGBM, RF, NN)
   ├─→ Perfect training accuracy? → Trigger auto-fix
   ├─→ High AUC/R²? → Mark as SUSPICIOUS
   ↓
5. Evaluate results
   ├─→ Leakage detected? → Auto-fix → Auto-rerun (if enabled)
   ├─→ No leakage? → Calculate predictability score
   ↓
6. Rank and return
```

## Auto-Rerun Mechanism

The auto-rerun mechanism automatically re-evaluates targets after leakage fixes are applied.

### When Auto-Rerun Triggers

Auto-rerun is enabled by default and triggers when:

1. **Leakage detected** during target evaluation
2. **Auto-fixer modifies configs** (excluded_features.yaml or feature_registry.yaml)
3. **Config changes detected** (backup created, files modified)

### Auto-Rerun Flow

```
Target Evaluation
   ↓
Leakage Detected?
   ├─→ No → Return result (OK)
   ↓
   Yes
   ↓
Auto-Fixer Enabled?
   ├─→ No → Mark as LEAKAGE_UNRESOLVED, skip
   ↓
   Yes
   ↓
Detect Leaking Features
   ↓
Apply Fixes (update configs, create backup)
   ↓
Configs Modified?
   ├─→ No → Mark as LEAKAGE_UNRESOLVED, skip
   ↓
   Yes
   ↓
Reload Feature Configs
   ↓
Re-run Target Evaluation (attempt N+1)
   ↓
Max Reruns Reached?
   ├─→ Yes → Mark as LEAKAGE_UNRESOLVED_MAX_RETRIES, skip
   ↓
   No
   ↓
Check Result
   ├─→ Still leaking? → Loop back to "Detect Leaking Features"
   ├─→ No leakage? → Return result (OK)
   ├─→ Suspicious? → Mark as SUSPICIOUS/SUSPICIOUS_STRONG, skip
```

### Configuration

Auto-rerun is configured in `training_config/safety_config.yaml` (see [Safety & Leakage Configs](../../02_reference/configuration/SAFETY_LEAKAGE_CONFIGS.md)):

```yaml
leakage_detection:
  auto_rerun:
    enabled: true                    # Enable/disable auto-rerun
    max_reruns: 3                    # Maximum reruns per target
    rerun_on_perfect_train_acc: true # Rerun on perfect training accuracy
    rerun_on_high_auc_only: false    # Rerun on high AUC alone (default: false)
```

### Status Codes

After auto-rerun, targets are marked with status codes:

- `OK` - No leakage detected, target is safe
- `SUSPICIOUS` - High AUC/R² but no obvious leakage (manual review recommended)
- `SUSPICIOUS_STRONG` - Very high AUC/R² (≥0.95), likely structural leakage
- `LEAKAGE_DETECTED` - Leakage found but not fixed (pre-scan removal)
- `LEAKAGE_UNRESOLVED` - Leakage detected but auto-fixer couldn't fix it
- `LEAKAGE_UNRESOLVED_MAX_RETRIES` - Leakage persists after max reruns
- `INSUFFICIENT_FEATURES` - Too few features after filtering

## Leakage Detection & Auto-Fix

### Detection Methods

1. **Pre-Training Scan** (before model training):
   - Binary classification: Features matching target with ≥99.9% accuracy
   - Regression: Features with ≥99.9% correlation with target
   - Automatically removes detected leaky features

2. **During Training** (behavioral signals):
   - Perfect CV scores (≥99%)
   - Perfect training accuracy (≥99.9%)
   - Perfect R² scores (≥99.9%)
   - Perfect correlation between predictions and targets

3. **Feature Importance Analysis**:
   - High importance in perfect-score models
   - Known leaky patterns (e.g., `y_*`, `fwd_ret_*`)

### Auto-Fix Process

1. **Detect** leaking features using multiple methods
2. **Filter** by confidence (default: ≥80%)
3. **Limit** to top N features per run (default: 20)
4. **Backup** config files (before modification)
5. **Update** `excluded_features.yaml` and `feature_registry.yaml`
6. **Reload** configs for next evaluation

### Backup System

Config backups are automatically created in:
```
CONFIG/backups/{target_name}/{timestamp}/
├── excluded_features.yaml
├── feature_registry.yaml
└── manifest.json
```

**Manifest includes:**
- Backup version
- Source (auto_fix_leakage)
- Target name
- Timestamp
- Git commit hash
- List of backed-up files
- Original config paths

**Retention Policy:**
- Keeps last N backups per target (default: 20)
- Automatic pruning of old backups
- Configurable via `system_config.yaml`

## Multi-Model Consensus

### Target Ranking

Targets are ranked using consensus across multiple model families:

1. **LightGBM** - Gradient boosting (GPU-accelerated)
2. **Random Forest** - Ensemble method
3. **Neural Network** - Deep learning

**Scoring:**
- Each model produces a score (ROC-AUC for classification, R² for regression)
- Scores are averaged across models
- Feature importance is aggregated
- Consistency across models is measured
- Final composite score combines all factors

### Feature Selection

Features are selected using multi-model importance aggregation:

1. **Train multiple models** on the target
2. **Extract feature importance** (native/SHAP/permutation)
3. **Aggregate importance** across models and symbols
4. **Rank features** by consensus score
5. **Select top M** features

## Caching Strategy

### Cache Structure

```
output_dir/cache/
├── target_rankings.json          # Cached target rankings
└── feature_selections/
    ├── {target1}.json
    ├── {target2}.json
    └── ...
```

### Cache Keys

Cache keys are generated from:
- Symbol list (sorted)
- Model families used
- Configuration hash (MD5 of relevant configs)

**Same symbols + same configs = cache hit**

### Cache Invalidation

Cache is automatically invalidated when:
- Symbol list changes
- Model families change
- Configuration files change (detected via hash)

Manual invalidation:
- `--force-refresh` - Force re-computation
- `--no-cache` - Disable caching entirely

## Configuration

All intelligence layer behavior is configurable via YAML configs:

### Safety Config (`training_config/safety_config.yaml`)

```yaml
leakage_detection:
  # Pre-scan thresholds
  pre_scan:
    min_match: 0.999      # Binary classification threshold
    min_corr: 0.999       # Regression correlation threshold
    min_valid_pairs: 10   # Minimum pairs for correlation
  
  # Feature requirements
  ranking:
    min_features_required: 2        # Minimum for ranking
    min_features_for_model: 3       # Minimum for training
    min_features_after_leak_removal: 2
  
  # Auto-fixer thresholds
  auto_fix_thresholds:
    cv_score: 0.99
    training_accuracy: 0.999
    training_r2: 0.999
    perfect_correlation: 0.999
  
  # Auto-fixer settings
  auto_fix_min_confidence: 0.8
  auto_fix_max_features_per_run: 20
  auto_fix_enabled: true
  
  # Auto-rerun settings
  auto_rerun:
    enabled: true
    max_reruns: 3
    rerun_on_perfect_train_acc: true
    rerun_on_high_auc_only: false
```

### System Config (`training_config/system_config.yaml`)

```yaml
system:
  backup:
    max_backups_per_target: 20  # Backup retention policy
    enable_retention: true
```

## Troubleshooting

### All Targets Skipped

**Symptoms:** All targets marked as `INSUFFICIENT_FEATURES` or `LEAKAGE_DETECTED`

**Possible Causes:**
1. Feature registry too strict for short-horizon targets
2. Too many features excluded in `excluded_features.yaml`
3. Pre-scan removing too many features

**Solutions:**
1. Check `CONFIG/feature_registry.yaml` for allowed horizons
2. Review `CONFIG/excluded_features.yaml` for overly broad patterns
3. Adjust `min_features_required` in `safety_config.yaml`
4. Check logs for specific reasons targets were skipped

### Auto-Rerun Looping Forever

**Symptoms:** Target keeps re-running, never resolves

**Possible Causes:**
1. Structural leakage (not fixable by auto-fixer)
2. Auto-fixer not finding the right features
3. Max reruns too high

**Solutions:**
1. Check target construction logic (may have inherent leakage)
2. Review auto-fixer detections in logs
3. Reduce `max_reruns` to fail faster
4. Manually review and fix configs

### High False Positive Rate

**Symptoms:** Many targets marked as `SUSPICIOUS` or `SUSPICIOUS_STRONG`

**Possible Causes:**
1. Legitimately high-performing targets
2. Overfitting (not true leakage)
3. Thresholds too strict

**Solutions:**
1. Review `SUSPICIOUS` targets manually - they may be valid
2. Check CV scores vs. training scores (overfitting indicator)
3. Adjust `warning_thresholds` in `safety_config.yaml`
4. Consider target construction - some targets may be inherently easier to predict

### Backup Directory Growing Too Large

**Symptoms:** `CONFIG/backups/` consuming too much disk space

**Solutions:**
1. Reduce `max_backups_per_target` in `system_config.yaml`
2. Manually prune old backups: `CONFIG/backups/{target}/`
3. Set `enable_retention: true` (already default)

## Related Documentation

- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md) - How to use the pipeline
- [Ranking and Selection Consistency](../../01_tutorials/training/RANKING_SELECTION_CONSISTENCY.md) - Unified pipeline behavior (interval handling, sklearn preprocessing, CatBoost configuration)
- [Intelligent Trainer API](../../02_reference/api/INTELLIGENT_TRAINER_API.md) - API reference
- [Modular Config System](../../02_reference/configuration/MODULAR_CONFIG_SYSTEM.md) - Config system guide (includes `logging_config.yaml`)
- [Leakage Analysis](LEAKAGE_ANALYSIS.md) - Detailed leakage detection documentation
- [Target Discovery](TARGET_DISCOVERY.md) - How targets are discovered
- [Feature Importance Methodology](FEATURE_IMPORTANCE_METHODOLOGY.md) - Feature importance research
- [Feature Importance Methodology](FEATURE_IMPORTANCE_METHODOLOGY.md) - How feature importance is calculated

