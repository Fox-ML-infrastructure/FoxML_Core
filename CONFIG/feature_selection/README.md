# Feature Selection Module Configs

This directory contains configuration files for the feature selection pipeline.

## Files

- `multi_model.yaml` - Multi-model feature selection configuration
  - Model families (lightgbm, xgboost, random_forest, etc.)
  - Aggregation strategies
  - Sampling settings
  - SHAP/permutation importance settings
  - **Target confidence thresholds** (HIGH/MEDIUM/LOW confidence requirements)
  - **Score tier thresholds** (signal strength metrics)
  - **Routing rules** (confidence + score_tier → operational buckets: core/candidate/experimental)

## Usage

Feature selection configs are loaded by `CONFIG/config_builder.py` and merged with experiment configs.

## Target Confidence & Routing

The `confidence` section in `multi_model.yaml` configures:
- **Confidence thresholds**: Requirements for HIGH/MEDIUM/LOW confidence buckets
- **Score tier thresholds**: Signal strength metrics (orthogonal to confidence)
- **Routing rules**: How confidence + score_tier map to operational buckets

See [Feature & Target Configs](../../DOCS/02_reference/configuration/FEATURE_TARGET_CONFIGS.md#target-confidence--routing) for complete documentation.

## Migration

The legacy `CONFIG/multi_model_feature_selection.yaml` will be moved here during Phase 2 of the config refactor.

