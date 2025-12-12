# Dual-View Target Ranking: Reproducibility Integration

## Overview

The dual-view target ranking system is **fully integrated** with the existing reproducibility suite. All view and symbol metadata is properly tracked and stored.

## Integration Points

### 1. RunContext Integration

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` (lines 3246-3250)

```python
# Build RunContext
ctx = RunContext(...)

# Add view and symbol to RunContext if available
if 'view' in locals():
    ctx.view = view
if 'symbol' in locals() and symbol:
    ctx.symbol = symbol
```

**Status**: ✅ **Working** - View and symbol are set on RunContext before reproducibility tracking

### 2. Directory Structure

**Location**: `TRAINING/utils/reproducibility_tracker.py` (lines 719-730)

The `_get_cohort_dir()` method automatically creates view-based directory structure:

```
REPRODUCIBILITY/
  TARGET_RANKING/
    {view}/                    # CROSS_SECTIONAL, SYMBOL_SPECIFIC, or LOSO
      {target_name}/
        symbol={symbol}/       # Only for SYMBOL_SPECIFIC and LOSO
          cohort={cohort_id}/
            metadata.json
            metrics.json
```

**Status**: ✅ **Working** - Directory structure includes view subdirectory

### 3. Metadata Storage

**Location**: `TRAINING/utils/reproducibility_tracker.py` (lines 860, 2165)

View metadata is stored in two places:

1. **`full_metadata`** (line 860):
   ```python
   "view": getattr(ctx, 'view', None) if stage_normalized == "TARGET_RANKING" else None
   ```

2. **`additional_data`** (line 2165):
   ```python
   if ctx.stage == "target_ranking" and hasattr(ctx, 'view') and ctx.view:
       additional_data["view"] = ctx.view
   ```

**Status**: ✅ **Working** - View is stored in both metadata locations

### 4. Route Type Mapping

**Location**: `TRAINING/utils/reproducibility_tracker.py` (lines 812, 2226)

For TARGET_RANKING, `view` is used as `route_type`:

```python
# In log_run()
if stage_normalized == "TARGET_RANKING" and hasattr(ctx, 'view') and ctx.view:
    route_type = ctx.view  # Use view as route_type for TARGET_RANKING

# In log_comparison()
if ctx.stage == "target_ranking" and hasattr(ctx, 'view') and ctx.view:
    route_type_for_log = ctx.view
```

**Status**: ✅ **Working** - View is correctly mapped to route_type

### 5. Symbol Extraction

**Location**: `TRAINING/utils/reproducibility_tracker.py` (lines 829-831)

Symbol is extracted from RunContext for SYMBOL_SPECIFIC/LOSO views:

```python
if stage_normalized == "TARGET_RANKING" and hasattr(ctx, 'symbol') and ctx.symbol:
    symbol = ctx.symbol  # Override symbol from RunContext
```

**Status**: ✅ **Working** - Symbol is correctly extracted

### 6. Index Updates

**Location**: `TRAINING/utils/reproducibility_tracker.py` (lines 1154-1173)

The index.parquet file includes view information:

```python
new_row = {
    "phase": phase,
    "mode": mode,  # For TARGET_RANKING, this is the view
    "target": item_name,
    "symbol": symbol,  # For SYMBOL_SPECIFIC/LOSO
    ...
}
```

**Status**: ✅ **Working** - Index includes view/symbol for querying

## Verification Checklist

To verify reproducibility integration is working:

### ✅ RunContext Population
- [x] `ctx.view` is set before `log_run()` is called
- [x] `ctx.symbol` is set for SYMBOL_SPECIFIC/LOSO views
- [x] View defaults to "CROSS_SECTIONAL" if not specified

### ✅ Directory Structure
- [x] `REPRODUCIBILITY/TARGET_RANKING/{view}/` directories are created
- [x] `symbol={symbol}/` subdirectories for SYMBOL_SPECIFIC/LOSO
- [x] `cohort={cohort_id}/` directories contain metadata.json

### ✅ Metadata Storage
- [x] `metadata.json` includes `"view"` field
- [x] `metadata.json` includes `"symbol"` field (when applicable)
- [x] `metadata.json` includes `"route_type"` (set to view for TARGET_RANKING)

### ✅ Trend Analysis
- [x] Trend analyzer can group runs by view/symbol
- [x] Series keys include view/symbol for proper grouping
- [x] Cross-view comparisons work correctly

### ✅ Index.parquet
- [x] Index includes `mode` (view) for TARGET_RANKING
- [x] Index includes `symbol` for SYMBOL_SPECIFIC/LOSO
- [x] Queries can filter by view/symbol

## Example Metadata Structure

### Cross-Sectional View

```json
{
  "schema_version": 1,
  "cohort_id": "cs_..._f2849563",
  "run_id": "2025-12-12_14-30-00",
  "stage": "TARGET_RANKING",
  "route_type": "CROSS_SECTIONAL",
  "view": "CROSS_SECTIONAL",
  "target": "y_will_peak_60m_0.8",
  "symbol": null,
  "N_effective": 24925,
  "n_symbols": 5,
  "symbols": ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"],
  ...
}
```

### Symbol-Specific View

```json
{
  "schema_version": 1,
  "cohort_id": "cs_..._a1b2c3d4",
  "run_id": "2025-12-12_14-30-00",
  "stage": "TARGET_RANKING",
  "route_type": "SYMBOL_SPECIFIC",
  "view": "SYMBOL_SPECIFIC",
  "target": "y_will_peak_60m_0.8",
  "symbol": "AAPL",
  "N_effective": 4985,
  "n_symbols": 1,
  "symbols": ["AAPL"],
  ...
}
```

## Testing

To test reproducibility integration:

1. **Run target ranking with dual views**:
   ```bash
   python -m TRAINING.ranking.rank_target_predictability \
       --data-dir "data/data_labeled/interval=5m" \
       --symbols AAPL MSFT GOOGL \
       --output-dir "test_repro"
   ```

2. **Check directory structure**:
   ```bash
   ls -la test_repro/REPRODUCIBILITY/TARGET_RANKING/
   # Should see: CROSS_SECTIONAL/, SYMBOL_SPECIFIC/
   
   ls -la test_repro/REPRODUCIBILITY/TARGET_RANKING/SYMBOL_SPECIFIC/y_will_peak_60m_0.8/
   # Should see: symbol=AAPL/, symbol=MSFT/, symbol=GOOGL/
   ```

3. **Check metadata.json**:
   ```bash
   cat test_repro/REPRODUCIBILITY/TARGET_RANKING/CROSS_SECTIONAL/y_will_peak_60m_0.8/cohort=.../metadata.json | jq '.view'
   # Should output: "CROSS_SECTIONAL"
   
   cat test_repro/REPRODUCIBILITY/TARGET_RANKING/SYMBOL_SPECIFIC/y_will_peak_60m_0.8/symbol=AAPL/cohort=.../metadata.json | jq '.view, .symbol'
   # Should output: "SYMBOL_SPECIFIC" and "AAPL"
   ```

4. **Check index.parquet**:
   ```python
   import pandas as pd
   df = pd.read_parquet("test_repro/REPRODUCIBILITY/index.parquet")
   print(df[df['phase'] == 'TARGET_RANKING'][['mode', 'target', 'symbol']])
   # Should show view in 'mode' column, symbol in 'symbol' column
   ```

## Known Issues

None currently. All integration points are working correctly.

## Backward Compatibility

✅ **Fully backward compatible**:
- If `view` is not set, defaults to "CROSS_SECTIONAL"
- Existing runs without view metadata continue to work
- Directory structure gracefully handles missing view (defaults to CROSS_SECTIONAL)
- Metadata includes view field only for TARGET_RANKING stage
