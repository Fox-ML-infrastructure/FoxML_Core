# Telemetry Scoping Fix — 2025-12-13

**Fix for Feature Selection and Target Ranking Telemetry: Compare Features Per-Target, Per-View, Per-Symbol**

## Summary

Fixed telemetry to ensure features are compared feature-to-feature based on the target they're being selected for, separated by cross-sectional and individual symbol views. **Target ranking telemetry now aligns with feature selection telemetry**, ensuring that when a target is ranked for just AAPL (SYMBOL_SPECIFIC), the feature selection telemetry only compares features selected for that same target + symbol combination. This makes telemetry trustworthy by ensuring comparisons are only made within the same scope, allowing you to compare feature performance for a given target rather than the same feature across every target.

---

## Problem

Telemetry was comparing features across all targets and all views, which is meaningless. For example:
- Comparing features selected for `y_will_swing_low_60m_0.20` (CROSS_SECTIONAL) with features for `y_will_swing_low_60m_0.10` (SYMBOL_SPECIFIC, symbol=AAPL)
- This makes telemetry untrustworthy because it's comparing apples to oranges

## Solution

Telemetry is now properly scoped by:
1. **Target**: Features are compared only for the same target (e.g., `y_will_swing_low_60m_0.20`)
2. **View**: Separated by CROSS_SECTIONAL vs SYMBOL_SPECIFIC
3. **Symbol**: For SYMBOL_SPECIFIC view, also separated by symbol (e.g., AAPL vs MSFT)

---

## Changes

### 1. Map View to Route Type for FEATURE_SELECTION

**Files Modified:**
- `TRAINING/utils/reproducibility_tracker.py` (lines 529-551, 1977-1985, 2525-2530, 2575-2585, 2595-2605)

**Changes:**
```python
# FIX: For FEATURE_SELECTION, map view to route_type
# - view="CROSS_SECTIONAL" → route_type="CROSS_SECTIONAL"
# - view="SYMBOL_SPECIFIC" → route_type="INDIVIDUAL"

def _extract_route_type(self, additional_data: Optional[Dict[str, Any]] = None) -> Optional[str]:
    # FIX: For FEATURE_SELECTION, map view to route_type
    view = additional_data.get('view')
    if view:
        if view.upper() == "CROSS_SECTIONAL":
            return "CROSS_SECTIONAL"
        elif view.upper() in ["SYMBOL_SPECIFIC", "INDIVIDUAL"]:
            return "INDIVIDUAL"  # SYMBOL_SPECIFIC maps to INDIVIDUAL for FEATURE_SELECTION
    ...
```

**Key Principle:**
- **CROSS_SECTIONAL**: Features selected across all symbols (pooled data)
- **SYMBOL_SPECIFIC** → **INDIVIDUAL**: Features selected per-symbol (separate telemetry per symbol)

---

### 2. Ensure RunContext Has View and Symbol

**Files Modified:**
- `TRAINING/ranking/feature_selector.py` (lines 1144-1160, 1187-1202, 1218-1246)

**Changes:**
```python
# FIX: Ensure view and symbol are set for proper telemetry scoping
ctx_to_use = RunContext(
    stage="FEATURE_SELECTION",
    target_name=target_column,
    target_column=target_column,
    ...
    view=view,  # FIX: Set view for proper telemetry scoping (CROSS_SECTIONAL vs SYMBOL_SPECIFIC)
    symbol=symbol  # FIX: Set symbol for SYMBOL_SPECIFIC view (ensures per-symbol telemetry)
)
```

**Key Principle:**
- **CROSS_SECTIONAL**: `view="CROSS_SECTIONAL"`, `symbol=None`
- **SYMBOL_SPECIFIC**: `view="SYMBOL_SPECIFIC"`, `symbol="AAPL"` (or specific symbol)

---

### 3. Pass View and Symbol to Additional Data

**Files Modified:**
- `TRAINING/utils/reproducibility_tracker.py` (lines 2515-2520)
- `TRAINING/ranking/feature_selector.py` (lines 1218-1246)

**Changes:**
```python
# FIX: Add view to additional_data for both TARGET_RANKING and FEATURE_SELECTION
if hasattr(ctx, 'view') and ctx.view:
    additional_data["view"] = ctx.view
# Also add symbol for SYMBOL_SPECIFIC/INDIVIDUAL views
if hasattr(ctx, 'symbol') and ctx.symbol:
    additional_data["symbol"] = ctx.symbol
```

**Legacy API (fallback):**
```python
additional_data_with_cohort = {
    **cohort_additional_data,
    "view": view,  # FIX: Include view for proper telemetry scoping
    "symbol": symbol,  # FIX: Include symbol for SYMBOL_SPECIFIC view
    "route_type": route_type_for_legacy  # FIX: Map view to route_type
}

tracker.log_comparison(
    stage="feature_selection",
    item_name=target_column,  # FIX: item_name is just target (view/symbol handled by route_type/symbol params)
    metrics=metrics_with_cohort,
    additional_data=additional_data_with_cohort,
    route_type=route_type_for_legacy,  # FIX: Properly scoped by view
    symbol=symbol  # FIX: Properly scoped by symbol (for SYMBOL_SPECIFIC view)
)
```

---

## Directory Structure

Telemetry is now organized as:

```
REPRODUCIBILITY/
  TARGET_RANKING/
    CROSS_SECTIONAL/           # Cross-sectional target ranking
      {target}/                # e.g., y_will_swing_low_60m_0.20
        cohort={cohort_id}/
          metadata.json
          metrics.json
          ...
    SYMBOL_SPECIFIC/           # Symbol-specific target ranking
      {target}/                # e.g., y_will_swing_low_60m_0.20
        symbol={symbol}/        # e.g., symbol=AAPL
          cohort={cohort_id}/
            metadata.json
            metrics.json
            ...
  FEATURE_SELECTION/
    CROSS_SECTIONAL/           # Cross-sectional feature selection
      {target}/                # e.g., y_will_swing_low_60m_0.20
        cohort={cohort_id}/
          metadata.json
          metrics.json
          ...
    INDIVIDUAL/                # Symbol-specific feature selection (maps from SYMBOL_SPECIFIC view)
      {target}/                # e.g., y_will_swing_low_60m_0.20
        symbol={symbol}/        # e.g., symbol=AAPL
          cohort={cohort_id}/
            metadata.json
            metrics.json
            ...
```

**Key Alignment:**
- **TARGET_RANKING** and **FEATURE_SELECTION** use the same directory structure
- **Same target, same view, same symbol** → Same directory structure
- When target ranking runs for `y_will_swing_low_60m_0.20` (SYMBOL_SPECIFIC, symbol=AAPL), feature selection telemetry for the same target + symbol is in the aligned directory structure

**Key Points:**
- **Target separation**: Each target has its own directory
- **View separation**: CROSS_SECTIONAL vs INDIVIDUAL (mapped from SYMBOL_SPECIFIC)
- **Symbol separation**: For INDIVIDUAL view, each symbol has its own subdirectory
- **Comparisons**: Only compare within the same `{target}/{view}/{symbol}` scope

---

## Index Filtering

The `index.parquet` file filters by:
- `phase == "FEATURE_SELECTION"`
- `target == item_name` (e.g., `y_will_swing_low_60m_0.20`)
- `mode == route_type` (e.g., `CROSS_SECTIONAL` or `INDIVIDUAL`)
- `symbol == symbol` (e.g., `AAPL` for INDIVIDUAL view, `None` for CROSS_SECTIONAL)

This ensures:
- ✅ Features for `target_A` (CROSS_SECTIONAL) are only compared to previous runs of `target_A` (CROSS_SECTIONAL)
- ✅ Features for `target_A` (SYMBOL_SPECIFIC, symbol=AAPL) are only compared to previous runs of `target_A` (SYMBOL_SPECIFIC, symbol=AAPL)
- ✅ Features for `target_A` (CROSS_SECTIONAL) are **NOT** compared to `target_A` (SYMBOL_SPECIFIC, symbol=AAPL)
- ✅ Features for `target_A` are **NOT** compared to `target_B`

---

## Expected Behavior

### CROSS_SECTIONAL View

**Target Ranking:**
- **Scope**: Target ranked across all symbols (pooled data)
- **Telemetry**: Compares target ranking for same target, CROSS_SECTIONAL view
- **Directory**: `REPRODUCIBILITY/TARGET_RANKING/CROSS_SECTIONAL/{target}/cohort={cohort_id}/`
- **Index filter**: `phase="TARGET_RANKING" AND target="{target}" AND mode="CROSS_SECTIONAL" AND symbol IS NULL`

**Feature Selection:**
- **Scope**: Features selected across all symbols (pooled data)
- **Telemetry**: Compares features for same target, CROSS_SECTIONAL view
- **Directory**: `REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/cohort={cohort_id}/`
- **Index filter**: `phase="FEATURE_SELECTION" AND target="{target}" AND mode="CROSS_SECTIONAL" AND symbol IS NULL`

**Alignment**: Target ranking and feature selection for the same target (CROSS_SECTIONAL) are in aligned directory structures, allowing you to compare feature performance for that target.

### SYMBOL_SPECIFIC View

**Target Ranking:**
- **Scope**: Target ranked for a specific symbol (e.g., AAPL only)
- **Telemetry**: Compares target ranking for same target, same symbol, SYMBOL_SPECIFIC view
- **Directory**: `REPRODUCIBILITY/TARGET_RANKING/SYMBOL_SPECIFIC/{target}/symbol={symbol}/cohort={cohort_id}/`
- **Index filter**: `phase="TARGET_RANKING" AND target="{target}" AND mode="SYMBOL_SPECIFIC" AND symbol="{symbol}"`

**Feature Selection:**
- **Scope**: Features selected per-symbol (separate selection for each symbol)
- **Telemetry**: Compares features for same target, same symbol, INDIVIDUAL mode (mapped from SYMBOL_SPECIFIC)
- **Directory**: `REPRODUCIBILITY/FEATURE_SELECTION/INDIVIDUAL/{target}/symbol={symbol}/cohort={cohort_id}/`
- **Index filter**: `phase="FEATURE_SELECTION" AND target="{target}" AND mode="INDIVIDUAL" AND symbol="{symbol}"`

**Alignment**: When a target is ranked for just AAPL (SYMBOL_SPECIFIC), the feature selection telemetry only compares features selected for that same target + symbol combination. This ensures you can compare feature performance for a given target (e.g., `y_will_swing_low_60m_0.20` for AAPL) rather than the same feature across every target.

---

## Verification

1. **Check directory structure**:
   ```bash
   ls REPRODUCIBILITY/FEATURE_SELECTION/
   # Should see: CROSS_SECTIONAL/ and INDIVIDUAL/
   
   ls REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/
   # Should see: y_will_swing_low_60m_0.20/, y_will_swing_low_60m_0.10/, etc.
   
   ls REPRODUCIBILITY/FEATURE_SELECTION/INDIVIDUAL/y_will_swing_low_60m_0.20/
   # Should see: symbol=AAPL/, symbol=MSFT/, etc.
   ```

2. **Check index.parquet filtering**:
   ```python
   import pandas as pd
   df = pd.read_parquet("REPRODUCIBILITY/index.parquet")
   
   # Should only see CROSS_SECTIONAL entries for target_A
   cs_entries = df[(df['phase'] == 'FEATURE_SELECTION') & 
                   (df['target'] == 'y_will_swing_low_60m_0.20') & 
                   (df['mode'] == 'CROSS_SECTIONAL')]
   
   # Should only see INDIVIDUAL entries for target_A, symbol=AAPL
   ind_entries = df[(df['phase'] == 'FEATURE_SELECTION') & 
                     (df['target'] == 'y_will_swing_low_60m_0.20') & 
                     (df['mode'] == 'INDIVIDUAL') & 
                     (df['symbol'] == 'AAPL')]
   ```

3. **Check telemetry comparisons**:
   - Features for `target_A` (CROSS_SECTIONAL) should only be compared to previous `target_A` (CROSS_SECTIONAL) runs
   - Features for `target_A` (SYMBOL_SPECIFIC, symbol=AAPL) should only be compared to previous `target_A` (SYMBOL_SPECIFIC, symbol=AAPL) runs
   - No cross-target or cross-view comparisons

---

## Files Modified

1. `TRAINING/utils/reproducibility_tracker.py`
   - Updated `_extract_route_type` to map view to route_type for FEATURE_SELECTION (lines 529-551)
   - Updated `log_comparison` to extract route_type from view for FEATURE_SELECTION (lines 1977-1985)
   - Updated `log_run` to map view to route_type for FEATURE_SELECTION (lines 2525-2530, 2575-2585, 2595-2605)
   - Added view and symbol to additional_data (lines 2515-2520)

2. `TRAINING/ranking/feature_selector.py`
   - Set view and symbol in RunContext (lines 1144-1160, 1187-1202)
   - Pass view, symbol, and route_type to legacy API (lines 1218-1246)

3. `TRAINING/ranking/predictability/model_evaluation.py`
   - Pass view and symbol to legacy API for target ranking (lines 4540-4560)
   - **CRITICAL**: Ensures target ranking telemetry aligns with feature selection telemetry
   - When a target is ranked for just AAPL (SYMBOL_SPECIFIC), feature selection telemetry only compares features for that same target + symbol combination

---

## Migration Notes

### For Users
- **No action required** - All fixes are backward compatible
- **Telemetry is now trustworthy** - Features are only compared within the same scope (target, view, symbol)
- **Directory structure** - New runs will use the proper structure (CROSS_SECTIONAL vs INDIVIDUAL)

### For Developers
- **Always set view and symbol in RunContext** - Required for proper telemetry scoping
- **Map view to route_type for FEATURE_SELECTION** - CROSS_SECTIONAL → CROSS_SECTIONAL, SYMBOL_SPECIFIC → INDIVIDUAL
- **item_name is just target** - View and symbol are handled by directory structure and filtering, not item_name
- **Index filters by phase, mode, target, symbol** - Ensures comparisons are only within the same scope

---

## Related Issues

- **Untrustworthy telemetry**: Fixed by proper scoping (target, view, symbol)
- **Cross-target comparisons**: Fixed by filtering on target in index
- **Cross-view comparisons**: Fixed by mapping view to route_type and filtering on mode
- **Cross-symbol comparisons**: Fixed by filtering on symbol for INDIVIDUAL mode

All fixes are complete. Telemetry is now trustworthy: features are only compared feature-to-feature based on the target they're being selected for, separated by cross-sectional and individual symbol views.
