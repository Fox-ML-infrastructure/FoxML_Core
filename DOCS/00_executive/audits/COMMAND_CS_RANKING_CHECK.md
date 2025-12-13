# Command Analysis: Cross-Sectional Ranking Test Coverage

**Command**: 
```bash
python TRAINING/train.py \
  --data-dir "data/data_labeled/interval=5m" \
  --symbols AAPL MSFT GOOGL TSLA NVDA \
  --output-dir "test_e2e_ranking_unified" \
  --auto-targets --top-n-targets 23 --max-targets-to-evaluate 23 \
  --auto-features --top-m-features 50 \
  --min-cs 3 --max-rows-per-symbol 5000 --max-rows-train 10000 \
  --families lightgbm xgboost random_forest catboost neural_network lasso mutual_information univariate_selection
```

## ✅ YES - Cross-Sectional Ranking WILL Run

### Conditions Met:

1. **`--auto-features` flag**: ✅ Present
   - This triggers `select_features_for_target()` which includes CS ranking

2. **Symbol count**: ✅ 5 symbols (AAPL, MSFT, GOOGL, TSLA, NVDA)
   - Config requires: `min_symbols: 2` (currently set to 2 for testing)
   - **5 >= 2** ✅

3. **Config enabled**: ✅ `CONFIG/feature_selection/multi_model.yaml:233`
   ```yaml
   cross_sectional_ranking:
     enabled: true  # ✅ Enabled
   ```

4. **Feature selection flow**: ✅
   - `train.py` → `intelligent_trainer.py` → `select_features_auto()` → `select_features_for_target()` → CS ranking

### What Will Happen:

1. **Per-target feature selection** (for each of 23 targets):
   - Per-symbol importance computation
   - Multi-model consensus aggregation
   - **Cross-sectional ranking** (panel model training)
   - Feature categorization (CORE/SYMBOL_SPECIFIC/CS_SPECIFIC/WEAK)

2. **CS Ranking Details**:
   - Uses top 50 features from per-symbol selection as candidates (`top_k_candidates: 50`)
   - Trains LightGBM panel model across all 5 symbols simultaneously
   - Requires `min_cs: 10` per timestamp (but adjusts to 5 if needed - see note below)
   - Max 1000 samples per timestamp (`max_cs_samples: 1000`)

### Important Notes:

#### `--min-cs 3` Does NOT Affect CS Ranking

**⚠️ Important**: The `--min-cs 3` argument affects **training data preparation**, NOT cross-sectional ranking.

- **Training uses**: `min_cs=3` (from CLI arg)
- **CS ranking uses**: `min_cs=10` (from config: `aggregation.cross_sectional_ranking.min_cs`)

However, the CS ranking code has a safety check:
```python
# From TRAINING/utils/cross_sectional_data.py:167
effective_min_cs = min(min_cs, len(mtf_data))
```
So with 5 symbols, even if config says `min_cs: 10`, it will use `effective_min_cs = 5`.

#### Potential Issues:

1. **Insufficient cross-sectional size**: If timestamps don't have at least 5 symbols with data, CS ranking might return zero importance (but won't crash)

2. **Data loading failures**: If `load_mtf_data_for_ranking()` fails, CS ranking will skip silently

3. **Model training failures**: If LightGBM fails to train, CS ranking will skip that model family

### What to Look For in Logs:

**Success indicators**:
```
🔍 Computing cross-sectional importance for 50 candidate features...
   Panel data: X samples, 50 features
   Training lightgbm panel model...
   lightgbm: top feature = feature_name (0.XXXX)
   ✅ Cross-sectional importance computed: top feature = feature_name (0.XXXX)
   ✅ Cross-sectional ranking complete
      CORE: X features
      SYMBOL_SPECIFIC: X features
      CS_SPECIFIC: X features
      WEAK: X features
```

**Skip indicators**:
```
Skipping cross-sectional ranking: only X symbols (min: 2)
Cross-sectional ranking disabled in config
```

**Failure indicators**:
```
Cross-sectional ranking failed: <exception>
Failed to prepare cross-sectional data, returning zero importance
All panel models failed, returning zero importance
```

### Expected Output:

The feature selection output will include:
- `cs_importance_score`: Cross-sectional importance (0-1 normalized)
- `feature_category`: CORE/SYMBOL_SPECIFIC/CS_SPECIFIC/WEAK/UNKNOWN

These will be saved in:
```
test_e2e_ranking_unified/feature_selections/<target>/feature_importance_summary.csv
```

## Summary

✅ **YES, your command WILL test cross-sectional ranking and selection**

The command will:
1. ✅ Run automatic target ranking (23 targets)
2. ✅ Run automatic feature selection per target (50 features each)
3. ✅ **Run cross-sectional ranking for each target** (panel model across 5 symbols)
4. ✅ Train models with selected features

**To verify it ran**: Check logs for `"🔍 Computing cross-sectional importance"` or check output CSV files for `cs_importance_score` and `feature_category` columns.
