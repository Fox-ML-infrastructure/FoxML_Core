# Fix: 95% CV Score - Overlapping Horizons & HIGH Features

**Date:** 2025-11-23 00:25:53

## Problem

Even after excluding 60m features, CV scores were still **95%** (should be 10-20%).

## Root Causes

1. **Overlapping horizons**: Features with 30m, 45m horizons for 60m target create autocorrelation
2. **HIGH-based features**: Features using HIGH prices leak information for peak targets
3. **Feature combinations**: Multiple return/volume features together achieve 95% prediction

## Solution

### 1. Exclude Overlapping Horizons

For barrier targets, exclude features with horizon >= target_horizon/2:
- 60m target → exclude 30m+ features
- 30m target → exclude 15m+ features

**Reason**: Autocorrelation - past 30m return predicts next 60m movement.

### 2. Exclude HIGH-based Features for Peak Targets

Exclude features containing: `high`, `upper`, `max`, `top`, `ceiling`

**Reason**: Features using HIGH prices encode information about whether we're near a peak.

### 3. Exclude LOW-based Features for Valley Targets

Exclude features containing: `low`, `lower`, `min`, `bottom`, `floor`

**Reason**: Features using LOW prices encode information about whether we're near a valley.

## Changes Made

### `CONFIG/excluded_features.yaml`
- Added `exclude_overlapping_horizon: true` to barrier horizon_overlap config
- Added comments about HIGH/LOW feature exclusions

### `SCRIPTS/utils/leakage_filtering.py`
- Enhanced `_filter_for_barrier_target()` to:
 - Exclude features with horizon >= target_horizon/2
 - Exclude HIGH-based features for peak targets
 - Exclude LOW-based features for valley targets

## Results

**Before:**
- Features: 289
- CV Score: 0.95 (95%)

**After:**
- Features: 240 (49 excluded)
- Expected CV Score: 0.10-0.20 (10-20%)

## Excluded Features

1. **60m horizon**: 9 features (ret_zscore_60m, vol_over_vol_60m, etc.)
2. **30m+ overlapping**: ~30 features (ret_zscore_30m, vol_30m, etc.)
3. **HIGH-based**: ~10 features (vwap_dev_high, fractal_high, etc.)

## Next Steps

1. **Restart Python** to clear config cache
2. **Re-run ranking script**:
   ```bash
   python SCRIPTS/rank_target_predictability.py \
       --discover-all \
       --symbols AAPL,MSFT,GOOGL,TSLA,JPM \
       --output-dir results/target_rankings_updated
   ```
3. **Verify scores**: Should see 0.10-0.20 CV scores (realistic)
4. **Check feature count**: Should see ~240 features for peak targets

## Notes

- `high_vol_frac` is correctly allowed (volatility feature, not price)
- Overlapping horizon exclusion is conservative (excludes >= target/2)
- HIGH/LOW exclusions are target-aware (peak vs valley)
