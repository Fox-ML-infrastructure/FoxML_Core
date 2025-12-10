# Target Predictability Ranking Module

This document describes the `TRAINING/ranking/predictability/` module structure, split from the original 3,454-line `rank_target_predictability.py` file for better maintainability.

## Purpose

Evaluates which targets are most predictable by training multiple model families on sample data and calculating composite predictability scores. Helps prioritize compute by training models on high-predictability targets first.

## Module Structure

### Core Modules

- **`scoring.py`** (59 lines): Scoring class definition
  - `TargetPredictabilityScore`: Dataclass for storing predictability assessment results
    - Contains: mean_score, std_score, mean_importance, consistency, model_scores, leakage flags, etc.

- **`composite_score.py`** (61 lines): Composite score calculation
  - `calculate_composite_score`: Combines multiple metrics into a single predictability score

- **`data_loading.py`** (360 lines): Configuration and data loading utilities
  - `load_target_configs`: Load target configurations from YAML
  - `discover_all_targets`: Discover all available targets for a symbol
  - `load_sample_data`: Load sample data for evaluation
  - `prepare_features_and_target`: Prepare features and target for training
  - `load_multi_model_config`: Load multi-model configuration
  - `get_model_config`: Get configuration for a specific model

- **`leakage_detection.py`** (1,964 lines): Leakage detection and feature analysis
  - `find_near_copy_features`: Find features that are near-copies of the target
  - `_detect_leaking_features`: Internal leakage detection logic
  - `detect_leakage`: Detect potential data leakage based on scores
  - `_save_feature_importances`: Save feature importance data
  - `_log_suspicious_features`: Log suspicious features for review

- **`model_evaluation.py`** (2,542 lines): Model training and evaluation
  - `train_and_evaluate_models`: Train multiple models and evaluate performance
  - `evaluate_target_predictability`: Main function to evaluate a single target's predictability
    - Handles data loading, leakage filtering, model training, score aggregation
    - Includes auto-fix logic for detected leakage
    - Returns `TargetPredictabilityScore` object

- **`reporting.py`** (267 lines): Report generation and output utilities
  - `save_leak_report_summary`: Save leakage detection summary report
  - `save_rankings`: Save ranked target list with recommendations
  - `_get_recommendation`: Generate recommendation text for a target

- **`main.py`** (334 lines): Main entry point
  - `main`: CLI entry point
  - Handles argument parsing, target discovery, evaluation loop, ranking output

## Usage

### Direct Import (Recommended)

```python
from TRAINING.ranking.predictability.model_evaluation import evaluate_target_predictability
from TRAINING.ranking.predictability.scoring import TargetPredictabilityScore
from TRAINING.ranking.predictability.data_loading import load_target_configs
```

### Backward Compatible Import

```python
# Still works - imports from predictability/ modules
from TRAINING.ranking.rank_target_predictability import (
    evaluate_target_predictability,
    TargetPredictabilityScore,
    load_target_configs
)
```

### CLI Usage

```bash
# Rank all enabled targets
python -m TRAINING.ranking.predictability.main

# Test on specific symbols
python -m TRAINING.ranking.predictability.main --symbols AAPL,MSFT,GOOGL

# Rank specific targets
python -m TRAINING.ranking.predictability.main --targets peak_60m,valley_60m
```

## Workflow

1. **Data Loading**: Load target configs and sample data for symbols
2. **Leakage Filtering**: Apply leakage detection rules to filter unsafe features
3. **Pre-Training Scan**: Detect near-copy features before model training
4. **Model Training**: Train multiple model families on cross-sectional data
5. **Evaluation**: Calculate metrics (R², ROC-AUC, accuracy) per model
6. **Leakage Detection**: Check for suspiciously high scores indicating leakage
7. **Auto-Fix**: Automatically fix detected leakage (if enabled)
8. **Aggregation**: Aggregate scores across models
9. **Ranking**: Calculate composite scores and rank targets
10. **Reporting**: Generate reports and save rankings

## Configuration

- **Target configs**: `CONFIG/target_configs.yaml`
- **Multi-model config**: `CONFIG/target_ranking/multi_model.yaml` (or feature_selection fallback)
- **Leakage detection**: `CONFIG/leakage_detection.yaml`
- **Feature registry**: `CONFIG/feature_registry.yaml`

## Output

- **Rankings**: `output/target_rankings_YYYYMMDD_HHMMSS.json`
- **Leak reports**: `output/leak_reports/`
- **Feature importances**: `output/feature_importances/`

## Notes

- All modules maintain backward compatibility with the original `rank_target_predictability.py` interface
- The original file is now a thin wrapper (56 lines) that re-exports everything
- Original file archived at `TRAINING/archive/original_large_files/rank_target_predictability.py.original`
- Large modules (`model_evaluation.py`, `leakage_detection.py`) are cohesive subsystems with clear responsibilities

## Related Documentation

- **[Refactoring & Wrappers Guide](../../01_tutorials/REFACTORING_AND_WRAPPERS.md)** - Complete explanation of how wrappers work and import patterns
- **[Refactoring Summary](REFACTORING_SUMMARY_INTERNAL.md)** - Overall refactoring details
