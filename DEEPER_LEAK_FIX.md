# 🔍 Deeper Leak Investigation - Round 2

## The Problem

After first leakage fix, R² only dropped from **0.72 → 0.70** (not enough!).

You were **100% correct** to be suspicious - we found **76 MORE leaking features**.

---

## What We Missed: Temporal Overlap Leaks

The original 90 features were **obvious forward-looking** metrics.  
The NEW 76 features are **subtle temporal overlap** metrics.

### Issue: Matching Time Windows

When predicting **60 minutes forward**, features using **60-minute windows** create spurious correlations:

| Feature | Why It Leaks |
|---------|--------------|
| `ret_60m` | Past 60m return correlates with next 60m (momentum) |
| `vol_60m` | Past 60m volatility clusters → predicts next 60m vol |
| `hit_direction_60m_0.3` | **Which barrier hit first** - direct future info! |
| `max_return_60m_*` | Maximum return in 60m window - requires future path |
| `min_return_60m_*` | Minimum return in 60m window - requires future path |
| `vol_at_t_60m_0.8` | Volatility "at time of hit" - requires knowing when hit occurs |

---

## New Exclusions (76 Features)

### 1. Max/Min Return Features (30 features)
```
max_return_5m_0.001, max_return_5m_0.002, max_return_5m_0.005
min_return_5m_0.001, min_return_5m_0.002, min_return_5m_0.005
... (same for 10m, 15m, 30m, 60m horizons)
```

**Why**: Require knowing the entire future price path.

### 2. Hit Direction Features (15 features)
```
hit_direction_5m_0.3, hit_direction_5m_0.5, hit_direction_5m_0.8
... (same for 10m, 15m, 30m, 60m horizons)
```

**Why**: Direct answer to "which barrier hits first?" (-1, 0, +1).

### 3. Volatility "At Time of Hit" (15 features)
```
vol_at_t_5m_0.3, vol_at_t_5m_0.5, vol_at_t_5m_0.8
... (same for 10m, 15m, 30m, 60m horizons)
```

**Why**: Requires knowing *when* the barrier hit occurs.

### 4. Range Compression (16 features)
```
range_compression_5m, range_compression_10m, ...
```

**Why**: Uses future high/low to calculate compression ratio.

---

## Updated Feature Count

|  | Before | After Round 2 | Change |
|--|--------|---------------|--------|
| **Total columns** | 531 | 531 | - |
| **Safe features** | 388 | **312** | **-76** |
| **Excluded** | 143 | **219** | **+76** |

---

## What's Safe Now?

✅ **OHLCV data** (open, high, low, close, volume)  
✅ **Technical indicators** using PAST data only (RSI, MACD, etc.)  
✅ **Lagged features** (returns, volatility from completed bars)  
✅ **Intraday patterns** (time of day, day of week)  
✅ **Market microstructure** (bid-ask, order flow - if from past)

❌ **Any feature with matching time horizon** (60m feature for 60m target)  
❌ **Any "at time of X" metrics** (require knowing when X occurs)  
❌ **Any max/min over future windows**

---

## Expected R² After This Fix

With 76 more leaking features removed:

| Scenario | Expected R² | Status |
|----------|-------------|--------|
| Before (0.70) | 0.65-0.75 | 🚨 Too high - still leaking |
| **After this fix** | **0.35-0.55** | ✅ **Realistic alpha** |
| If still > 0.60 | 0.60-0.70 | ⚠️ More investigation needed |

---

## Next Step: Re-Run Ranking

```bash
conda activate trader_env
cd /home/Jennifer/trader

# Run with ALL 54 valid targets
python scripts/rank_target_predictability.py \
  --discover-all \
  --symbols AAPL,MSFT,GOOGL \
  --output-dir results/round2_clean_baseline
```

**Expected time**: 30-60 minutes (54 targets × 3 symbols × 3 model families)

---

## If R² Is STILL > 0.60...

Then we investigate:

1. **Rolling statistics with look-forward bias**:
   - `vol_60m` might be calculated incorrectly
   - Check if ANY indicator uses future bars

2. **Target calculation itself**:
   - Verify `y_will_peak_60m_0.8` uses ONLY future data
   - No information from current bar should be in target

3. **Autocorrelation from daily patterns**:
   - Market opens/closes create predictable patterns
   - This is LEGAL but inflates R²
   - Solution: Add time-of-day controls

---

## Key Insight

> **"Match your feature horizon to your prediction horizon - then EXCLUDE those features"**

For a 60m prediction:
- ✅ Use 5m, 10m, 15m features (shorter windows)
- ✅ Use 1d, 5d, 20d features (much longer windows)
- ❌ DON'T use 60m features (temporal overlap)
- ❌ DON'T use 30m features (too close, autocorrelation)

---

## Files Updated

- ✅ `CONFIG/excluded_features.yaml` (+76 exclusions)
- ✅ `scripts/rank_target_predictability.py` (added `--discover-all` flag)

---

## Summary

**Round 1**: Removed 90 obvious forward-looking features  
**Round 2**: Removed 76 subtle temporal-overlap features  
**Total removed**: **166 leaking features** (31% of dataset)

Now we should see **honest** predictive power. 🎯

