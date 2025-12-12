# Feature Filtering Execution Order

**Formalized hierarchy of feature filtering operations in the training pipeline.**

This document defines the exact order of operations for feature filtering to ensure consistent behavior and prevent race conditions.

## Execution Order (Chronological)

### Phase 1: Data Loading
1. **Load raw data** from data files
2. **Extract all columns** from dataframe
3. **Detect data interval** (5m, 15m, etc.)

### Phase 2: Target-Conditional Exclusions (Pre-Processing)
4. **Load existing exclusion list** from `RESULTS/{cohort}/{run}/feature_exclusions/{target}_exclusions.yaml` (if exists)
5. **OR Generate new exclusion list** based on:
   - Target horizon (Rule 1: Horizon Safety - exclude features with lookback > horizon * multiplier)
   - Target semantics (Rule 2: Semantic Safety - exclude repainting indicators for peak/valley targets)
6. **Apply target-conditional exclusions** to column list
7. **Save exclusion list** to RESULTS directory for future runs

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` (line ~2514-2555)

### Phase 3: Global Feature Filtering
8. **Metadata exclusion** (hardcoded safety net: `symbol`, `interval`, `source`, `ts`)
9. **Pattern-based exclusion** (`excluded_features.yaml` - always_exclude patterns)
10. **Feature registry filtering** (structural rules based on temporal metadata)
11. **Target-specific filtering** (forward_return, barrier, first_touch rules)
12. **Registry final filter** (explicitly rejected features)

**Location**: `TRAINING/utils/leakage_filtering.py` - `filter_features_for_target()`

### Phase 4: Ranking Mode Schema Merge
13. **Schema-based safe features** (OHLCV/TA families from `feature_target_schema.yaml`)
14. **Hardcoded safe features** (fallback patterns)
15. **Merge with filtered features** (add schema features that were excluded earlier)

**Location**: `TRAINING/utils/leakage_filtering.py` (line ~777-858)

### Phase 5: Active Sanitization (Ghost Buster)
16. **Compute lookback for all features** using `compute_feature_lookback_max()`
17. **Quarantine features** with lookback > `max_safe_lookback_minutes` (default: 240m)
18. **Log quarantined features** and reasons

**Location**: `TRAINING/utils/leakage_filtering.py` (line ~859-876)  
**CRITICAL**: Runs AFTER schema merge to catch ghost features that sneak in

### Phase 6: Data Preparation
19. **Prepare cross-sectional data** (pool samples across symbols)
20. **Drop all-NaN features** (features that are NaN for all samples)
21. **Clean data** (handle remaining NaNs, preserve for CV-safe imputation)

**Location**: `TRAINING/utils/cross_sectional_data.py`

### Phase 7: Pre-Training Leak Scan
22. **Detect near-copy features** (features matching target with ≥99.9% accuracy/correlation)
23. **Remove leaky features** from dataframe
24. **Update feature list** and counts

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` (line ~2765-2795)

### Phase 8: Final Gatekeeper (Last Mile)
25. **Compute lookback for ALL features** using `compute_feature_lookback_max()` (SAME as audit)
26. **Calculate safe_lookback_max** = purge_limit * 0.99 (1% safety buffer)
27. **Drop features** that violate purge limit:
   - Explicit daily/24h naming patterns
   - Calculated lookback > safe_lookback_max
28. **Physically remove** from X array (numpy array column deletion)
29. **Return filtered X and feature_names**

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` (line ~268-377, called at line ~3047)  
**CRITICAL**: This is the absolute last check before data touches the model

### Phase 9: Feature Pruning
30. **Quick importance pruning** (select top features by importance)
31. **Update feature list** and counts

**Location**: `TRAINING/utils/feature_pruning.py`

### Phase 10: Model Training
32. **Train models** with final feature set
33. **Evaluate models** on validation set

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` - `train_and_evaluate_models()`

### Phase 11: Audit Validation
34. **Compute feature_lookback_max_minutes** from final feature set (SAME calculation as Final Gatekeeper)
35. **Validate purge_minutes >= feature_lookback_max_minutes**
36. **Log violations** if purge < lookback

**Location**: `TRAINING/utils/audit_enforcer.py` - `_validate_feature_lookback()`

## Key Principles

### 1. Single Source of Truth for Lookback Calculation
- **ALL** lookback calculations use `TRAINING/utils/resolved_config.py` - `compute_feature_lookback_max()`
- This ensures Final Gatekeeper and Audit see the same values
- No duplicate logic or pattern matching inconsistencies

### 2. Execution Order is Critical
- Target-conditional exclusions run FIRST (before global filtering)
- Active sanitization runs AFTER schema merge (to catch ghost features)
- Final Gatekeeper runs LAST (absolute final check before model)

### 3. Each Phase Has a Purpose
- **Target-conditional**: Tailor features to target physics (short-term vs long-term)
- **Global filtering**: Apply universal safety rules (leakage patterns, registry)
- **Schema merge**: Add safe baseline features for ranking mode
- **Active sanitization**: Catch long lookback features that slip through
- **Final Gatekeeper**: Last-mile enforcement (handles race conditions)

### 4. No Feature Can Escape
- Features are checked at multiple stages
- Final Gatekeeper is the absolute last check
- If a feature violates purge limit, it's removed before model training

## Troubleshooting

### Issue: Audit sees different lookback than Final Gatekeeper

**Symptom**: Final Gatekeeper drops features, but audit still sees 1440m lookback

**Root Cause**: Final Gatekeeper and audit using different lookback calculations

**Fix**: Both must use `compute_feature_lookback_max()` from `resolved_config.py`

### Issue: Ghost feature still appears after Final Gatekeeper

**Symptom**: Final Gatekeeper runs, but audit violation still occurs

**Root Cause**: Final Gatekeeper not using same calculation as audit, or feature name doesn't match patterns

**Fix**: 
1. Ensure Final Gatekeeper uses `compute_feature_lookback_max()` (same as audit)
2. Check that feature_lookback_dict includes ALL features, not just top 10
3. Verify feature names match between Final Gatekeeper and audit

### Issue: Features added after Final Gatekeeper

**Symptom**: Final Gatekeeper drops features, but they reappear later

**Root Cause**: Feature list modified after Final Gatekeeper runs

**Fix**: Final Gatekeeper must run immediately before `train_and_evaluate_models()` - no feature modifications after that point

## Current Implementation Status

✅ **Implemented**:
- Phase 1-7: All implemented and working
- Phase 8: Final Gatekeeper implemented (needs fix to use same calculation as audit)
- Phase 9-10: Model training working
- Phase 11: Audit validation working

🔧 **Needs Fix**:
- Final Gatekeeper must use `compute_feature_lookback_max()` for ALL features (not just top 10)
- Must ensure feature_lookback_dict includes every feature in the final list

## Related Documentation

- [Active Sanitization Guide](ACTIVE_SANITIZATION.md) - Phase 5 details
- [Target-Conditional Exclusions](../utils/target_conditional_exclusions.py) - Phase 2 details
- [Resolved Config Fix](RESOLVED_CONFIG_FIX.md) - Lookback calculation details
- [Leakage Analysis](../../research/LEAKAGE_ANALYSIS.md) - Overall leakage prevention system
