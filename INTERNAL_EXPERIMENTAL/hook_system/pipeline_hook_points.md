# Pipeline Hook Points

Standard hook points for extending the pipeline without refactoring.

## Hook Point Naming Convention

Format: `{stage}_{event}`

- `stage`: Pipeline stage (target_ranking, feature_selection, training, etc.)
- `event`: When hook fires (before, after, on_error)

## Available Hook Points

### Target Ranking
- `before_target_ranking` - Before target ranking starts
- `after_target_ranking` - After target ranking completes (receives rankings list)
- `on_target_ranking_error` - If target ranking fails

### Feature Selection
- `before_feature_selection` - Before feature selection starts (receives target name)
- `after_feature_selection` - After feature selection completes (receives selected_features list)
- `on_feature_selection_error` - If feature selection fails

### Training
- `before_training` - Before model training starts
- `after_training` - After training completes (receives training results)
- `on_training_error` - If training fails

### Data Processing
- `before_data_load` - Before loading data
- `after_data_load` - After data loaded (receives data dict)
- `before_data_preprocessing` - Before preprocessing
- `after_data_preprocessing` - After preprocessing

## Context Object

Hooks receive a context object (dict) with:
- `stage`: Current stage name
- `output_dir`: Output directory
- `data_dir`: Data directory
- `symbols`: List of symbols
- `target`: Target name (if applicable)
- `features`: Feature list (if applicable)
- `results`: Stage results (if applicable)
- `metadata`: Additional metadata

Hooks can modify context and return it, or return None to keep original.

## Priority

Lower priority = executes first. Default priority = 100.

- Priority 0-50: Critical hooks (validation, safety checks)
- Priority 51-100: Standard hooks
- Priority 101+: Post-processing hooks (logging, reporting)
