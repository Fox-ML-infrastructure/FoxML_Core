# Model Family Status Tracking

**Added**: 2025-12-09  
**Purpose**: Debug why only 8/14 model families are being aggregated in multi-model feature selection

## Problem

When running multi-model feature selection, logs show:
- `NVDA: Completed 8/14 models`
- Aggregation only includes 8 families
- No visibility into which families failed and why

## Solution

Added structured status tracking that:

1. **Tracks success/failure per family per symbol** with detailed error information
2. **Logs clear summaries** showing which families succeeded/failed
3. **Persists status to JSON** for post-run analysis
4. **Logs excluded families** during aggregation

## What You'll See

### Per-Symbol Logs

Instead of just:
```
✅ NVDA: Completed 8/14 models
```

You now get:
```
✅ NVDA: Completed 8/14 model families
   ✅ Success: lightgbm, xgboost, random_forest, neural_network, catboost, lasso, mutual_information, rfe
   ❌ Failed: boruta, stability_selection, univariate_selection, some_experimental
      - boruta: ImportError: No module named 'boruta'
      - stability_selection: ValueError: all NaNs in target
      - univariate_selection: InvalidImportance: Model returned None importance
```

### Aggregation Logs

```
⚠️  3 model families excluded from aggregation (no results): boruta, stability_selection, univariate_selection
   - boruta: Failed for 5 symbol(s) (AAPL, MSFT, GOOGL, TSLA, NVDA) with error types: ImportError
   - stability_selection: Failed for 2 symbol(s) (TSLA, NVDA) with error types: ValueError
✅ Aggregating 8 model families with results: lightgbm, xgboost, random_forest, neural_network, catboost, lasso, mutual_information, rfe
```

## Output Files

### `model_family_status.json`

Saved in the feature selection output directory with:

1. **Summary** (per-family statistics):
   ```json
   {
     "summary": {
       "boruta": {
         "total_runs": 5,
         "success": 0,
         "failed": 5,
         "symbols_success": [],
         "symbols_failed": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
         "error_types": ["ImportError"],
         "errors": [
           {
             "symbol": "AAPL",
             "error_type": "ImportError",
             "error": "No module named 'boruta'"
           }
         ]
       }
     },
     "detailed": [...]
   }
   ```

2. **Detailed** (per-family per-symbol status):
   ```json
   {
     "detailed": [
       {
         "status": "success",
         "family": "lightgbm",
         "symbol": "AAPL",
         "score": 0.8234,
         "top_feature": "volatility_20d",
         "top_feature_score": 0.1234,
         "error": null,
         "error_type": null
       },
       {
         "status": "failed",
         "family": "boruta",
         "symbol": "AAPL",
         "score": null,
         "top_feature": null,
         "top_feature_score": null,
         "error": "No module named 'boruta'",
         "error_type": "ImportError"
       }
     ]
   }
   ```

## How to Use

### 1. Run Feature Selection

```bash
python TRAINING/train.py \
    --data-dir "data/data_labeled/interval=5m" \
    --symbols AAPL MSFT GOOGL TSLA NVDA \
    --auto-features \
    --top-m-features 50
```

### 2. Check Logs

Look for the per-symbol summaries and aggregation warnings.

### 3. Analyze Status File

```python
import json

with open("output_dir/model_family_status.json") as f:
    status = json.load(f)

# Which families failed?
failed = [f for f, s in status['summary'].items() if s['failed'] > 0]
print(f"Failed families: {failed}")

# Why did boruta fail?
if 'boruta' in status['summary']:
    print(status['summary']['boruta']['errors'])
```

### 4. Fix or Disable Broken Families

Based on the error types:

- **ImportError**: Missing dependency → `pip install <package>` or disable in config
- **ValueError**: Data issue (e.g., all NaNs) → Check data quality or disable for that target type
- **InvalidImportance**: Model returned None/zeros → Check model implementation or disable

Update `CONFIG/feature_selection/multi_model.yaml`:

```yaml
model_families:
  boruta:
    enabled: false  # Disable until dependency is installed
  stability_selection:
    enabled: false  # Disable for binary targets (known issue)
```

## Implementation Details

### Status Tracking

Each family execution is wrapped in try/except that captures:
- Success/failure status
- Exception type and message
- Score and top feature (if successful)
- Symbol name

### Status Collection

Statuses are collected per symbol and passed to:
1. `aggregate_multi_model_importance()` - for logging excluded families
2. `save_multi_model_results()` - for saving to JSON

### Backward Compatibility

- All existing code paths continue to work
- Status tracking is optional (gracefully handles missing status info)
- No breaking changes to APIs

## Next Steps

1. **Run your pipeline** and check `model_family_status.json`
2. **Identify broken families** from the summary
3. **Fix or disable** based on error types
4. **Re-run** to verify all enabled families succeed

## Example Debugging Workflow

```bash
# 1. Run feature selection
python TRAINING/train.py --auto-features ...

# 2. Check status file
cat output_dir/model_family_status.json | jq '.summary | keys'

# 3. See which failed
cat output_dir/model_family_status.json | jq '.summary | to_entries | map(select(.value.failed > 0)) | map({family: .key, errors: .value.errors})'

# 4. Fix one family at a time
# Edit CONFIG/feature_selection/multi_model.yaml
# Re-run to verify
```

---

**See Also:**
- [Multi-Model Feature Selection](../../01_tutorials/training/FEATURE_SELECTION_TUTORIAL.md)
- [Intelligent Training Tutorial](../../01_tutorials/training/INTELLIGENT_TRAINING_TUTORIAL.md)
