# Metadata Required for Run Comparability

Runs are only comparable (can be diffed) if they have **exactly the same outcome-influencing metadata**. This document explains what fields must match.

## Core Comparability Checks

Two runs are comparable if they pass ALL of these checks:

1. **Same fingerprint schema version** - Fingerprint computation must be compatible
2. **Same stage** - TARGET_RANKING, FEATURE_SELECTION, or TRAINING
3. **Same view** - CROSS_SECTIONAL, SYMBOL_SPECIFIC, LOSO, etc.
4. **Same target** (if specified) - Target name must match
5. **Same comparison group** - **This is the primary check** (see below)

## Comparison Group Fields

The comparison group is the key mechanism - runs are only comparable if they have **identical comparison group keys**. The comparison group includes:

### Required for All Stages

- **`n_effective`** (CRITICAL) - Exact sample size must match exactly
  - Example: A run with N=5000 can only compare to other runs with N=5000
  - Not: N=5000 vs N=10000 (different sample sizes = different outcomes)

- **`dataset_signature`** - Hash of:
  - Universe ID
  - Date range (start/end)
  - `min_cs` (minimum cross-sectional samples)
  - `max_cs_samples` (maximum cross-sectional samples)
  
- **`task_signature`** - Hash of:
  - Target name
  - Horizon (minutes)
  - Objective (e.g., "binary_classification", "regression")

- **`routing_signature`** - Hash of routing/view configuration
  - View type (CROSS_SECTIONAL, SYMBOL_SPECIFIC, etc.)

- **`experiment_id`** (optional) - If tracked, must match

### Stage-Specific Fields

#### TARGET_RANKING
- **Does NOT include:**
  - `model_family` (not applicable)
  - `feature_signature` (not applicable)

#### FEATURE_SELECTION
- **Includes:**
  - `feature_signature` - Hash of feature set (CRITICAL: different features = different outcomes)
  - `hyperparameters_signature` - Hash of all hyperparameters (CRITICAL: different HPs = different features selected)
  - `train_seed` - Training seed (CRITICAL: different seeds = different features selected)
  - `library_versions_signature` - Hash of library versions (CRITICAL: different versions = different features selected)
- **Does NOT include:**
  - `model_family` (not applicable - feature selection uses multiple model families)

#### TRAINING
- **Includes:**
  - `model_family` - Model family (CRITICAL: different families = different outcomes)
  - `feature_signature` - Hash of feature set (CRITICAL: different features = different outcomes)
  - `hyperparameters_signature` - Hash of all hyperparameters (CRITICAL: different HPs = different outcomes)
  - `train_seed` - Training seed (CRITICAL: different seeds = different outcomes)
  - `library_versions_signature` - Hash of library versions (CRITICAL: different versions = different outcomes)

## Comparison Group Key Format

The comparison group key is built as:
```
exp={experiment_id}|data={dataset_hash[:8]}|task={task_hash[:8]}|route={routing_hash[:8]}|n={n_effective}|family={model_family}|features={feature_hash[:8]}|hps={hyperparameters_hash[:8]}|seed={train_seed}|libs={library_versions_hash[:8]}
```

Example:
```
exp=e2e_test|data=a09c1bbf|task=6e999f55|route=36e32497|n=4940|family=lightgbm|features=abe71c8f|hps=3f2a1b9c|seed=42|libs=a1b2c3d4
```

## What Does NOT Affect Comparability

These fields are **not outcome-influencing** - they can differ between comparable runs:

- **Run ID** - Just an identifier, format differences don't matter
- **Timestamp** - Just an identifier

**NOTE:** All outcome-influencing factors are **now part of the comparison group** and **DO affect comparability**:
- Hyperparameters (learning_rate, max_depth, etc.)
- Train seed
- Library versions (pandas, numpy, scikit-learn, etc.)

Runs with different hyperparameters, train_seed, or library versions will NOT be comparable.

## Required Metadata Fields (for snapshot creation)

To create a snapshot, these fields must be present in `resolved_metadata`:

### TARGET_RANKING
- `stage`
- `run_id`
- `cohort_id`
- `date_range_start`
- `date_range_end`
- `n_symbols`
- `n_effective`
- `target_name`
- `view`
- `min_cs`
- `max_cs_samples`

### FEATURE_SELECTION
- All TARGET_RANKING fields, plus:
- `n_features`
- `hyperparameters` (from training section)
- `train_seed` (from training section)

### TRAINING
- All FEATURE_SELECTION fields, plus:
- `model_family`

## Example: Why Runs Are Not Comparable

**Run A:**
- Target: `y_will_swing_high_60m_0.05`
- N_effective: 5000
- View: CROSS_SECTIONAL
- Dataset: 2025Q3, min_cs=3, max_cs=1000

**Run B:**
- Target: `y_will_swing_high_60m_0.05`
- N_effective: 5000
- View: CROSS_SECTIONAL
- Dataset: 2025Q3, min_cs=3, max_cs=1000
- **Model family: lightgbm** (TRAINING stage)

**Result:** Not comparable - Run B has `model_family` set (TRAINING stage), Run A doesn't (TARGET_RANKING stage). Different stages = not comparable.

**Run C:**
- Target: `y_will_swing_high_60m_0.05`
- N_effective: **10000** (different!)
- View: CROSS_SECTIONAL
- Dataset: 2025Q3, min_cs=3, max_cs=1000

**Result:** Not comparable - Different `n_effective` (5000 vs 10000) = different comparison group.

## Summary

**Key principle:** Only runs with **exactly the same outcome-influencing metadata** are comparable. The comparison group key captures all of this - if two runs have the same comparison group key, they're comparable.

**What matters for reproducibility:**
- Exact sample size (`n_effective`)
- Same dataset (universe, dates, min_cs, max_cs)
- Same task (target, horizon, objective)
- Same routing/view
- Same model family (TRAINING only)
- Same feature set (FEATURE_SELECTION, TRAINING only)
- **Same hyperparameters** (TRAINING only - CRITICAL: different HPs = different outcomes)
- **Same train_seed** (TRAINING only - CRITICAL: different seeds = different outcomes)
- **Same library versions** (CRITICAL: different versions = different outcomes)

**What doesn't matter:**
- Run ID format (just an identifier)
- Timestamp (just an identifier)

