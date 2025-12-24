# Forward Return 20-Day Leakage Analysis

## The Problem

**R² = 0.60-0.70 for predicting 20-day forward returns is suspiciously high.**

These scores indicate **data leakage** - features containing information about the 20-day future return.

## Why These Scores Are Suspicious

### Expected Performance for Forward Returns

| R² Range | Interpretation | Status |
|----------|----------------|--------|
| **0.30-0.50** | **Excellent** - honest alpha | **Realistic!** |
| **0.20-0.30** | **Good** - tradeable | Acceptable |
| **0.10-0.20** | **Decent** - needs risk mgmt | Acceptable |
| **0.50-0.60** | ️ **Borderline suspicious** | Investigate |
| **> 0.60** | **LIKELY LEAKAGE** | **Halt & investigate** |

**Your scores: 0.60-0.70 = LEAKAGE**

### Why Forward Returns Are Hard to Predict

1. **Market Efficiency**: Returns are close to random walks
2. **Noise Dominance**: Signal-to-noise ratio is very low
3. **Mean Reversion vs Momentum**: Competing forces cancel out
4. **Regime Changes**: Market behavior shifts unpredictably

**R² > 0.60 means the model is "cheating" - using future information.**

## What's Likely Leaking

### 1. Temporal Overlap Features

Features calculated over **20-day windows** create autocorrelation with `fwd_ret_20d`:

```
 ret_20d, returns_20d        # 20-day returns (direct overlap)
 vol_20d, volatility_20d      # 20-day volatility (clustering)
 sma_20d, ema_20d            # 20-day moving averages
 rsi_20d, macd_20d            # 20-day indicators
 max_return_20d, min_return_20d  # 20-day extremes
```

**Why they leak:**
- Past 20-day return → predicts next 20-day return (momentum)
- Past 20-day volatility → predicts next 20-day volatility (clustering)
- Moving averages over 20 days → autocorrelated with 20-day forward return

### 2. Features with Matching Horizons

Any feature with a **20-day window** will have temporal overlap:

```
 Any feature ending in _20d
 Any feature with window = 20 days
 Any feature calculated over 20 trading days
```

### 3. Long-Window Features That Overlap

Features with windows **close to 20 days** (e.g., 15d, 18d, 22d, 25d) may also leak:

```
️ ret_15d, ret_18d, ret_22d, ret_25d
️ vol_15d, vol_18d, vol_22d, vol_25d
```

## Current Filtering Issue

The current feature filtering (`CONFIG/excluded_features.yaml`) is **target-agnostic**:

- Excludes 30m+ windows for 60m targets
- **Does NOT exclude 20d windows for 20d targets**
- **Does NOT adapt to target horizon**

**Result:** Features with 20-day windows are still being used to predict 20-day forward returns → **LEAKAGE**

## The Fix Needed

### Option 1: Target-Aware Filtering (Recommended)

Make `filter_leaking_features.py` target-aware:

```python
def filter_features_for_target(
    all_columns: List[str],
    target_name: str,
    config: dict = None
) -> List[str]:
    """
    Filter features based on target horizon.

    For fwd_ret_20d: exclude features with 20d windows
    For fwd_ret_60m: exclude features with 60m windows
    etc.
    """
    # Extract target horizon
    if target_name.startswith('fwd_ret_'):
        horizon = extract_horizon(target_name)  # e.g., "20d" -> 20 days

        # Exclude features with matching/similar horizons
        excluded = get_temporal_overlap_features(horizon)

    return safe_features
```

### Option 2: Add 20-Day Overlap Exclusion

Add to `CONFIG/excluded_features.yaml`:

```yaml
temporal_overlap_20d_plus:
  - ret_20d
  - returns_20d
  - vol_20d
  - volatility_20d
  - sma_20d
  - ema_20d
  - rsi_20d
  - macd_20d
  # ... all 20-day window features
```

Then exclude these when target is `fwd_ret_20d`.

## Immediate Action

1. **Flag these targets as LEAKED** in the output
2. **Don't use them for training** until filtering is fixed
3. **Investigate which features are causing the high scores**
4. **Implement target-aware filtering**

## Expected Scores After Fix

After excluding 20-day overlap features:

| Target | Current R² | Expected R² (honest) |
|--------|------------|----------------------|
| `fwd_ret_20d` | **0.60-0.70** | **0.15-0.35** |

The honest scores will be much lower, but they'll be **realistic and tradeable**.

