# TODO / Follow-ups

- [ ] **Verify banner suppression works**: Run isolation training and confirm no banner prints
- [ ] **Test family normalization**: Run with config containing `catboost`, `neural_network`, `random_forest` variants
- [ ] **Investigate feature count collapse**: Why 100 requested → 52 allowed → 12 used? Check feature registry filtering logic
- [ ] **Add missing families to registry**: If `CatBoost`, `RandomForest`, `Lasso` are real trainers, add to MODMAP/TRAINER_MODULE_MAP
- [ ] **Monitor reproducibility tracking**: Watch for any remaining `.name` errors in production runs
- [ ] **Update documentation**: Document canonical family ID format and normalization rules

(Links: see ../15-30-00_training_pipeline_fixes/)

