# Target Discovery Update - Added Forward Return Targets

## What Changed

Updated `SCRIPTS/rank_target_predictability.py` to discover and rank **additional target types**:

### Before
- Only discovered `y_*` targets (63 total)
- Ranked 53 valid `y_*` targets
- Skipped 9 degenerate + 1 leaked target

### After
- Discovers `y_*` targets (barrier, swing, MFE/MDD)
- **Also discovers `fwd_ret_*` targets (forward returns)**
- Will rank **63 total targets** (53 y_* + 10 fwd_ret_*)

---

## New Targets Added

### Forward Return Targets (`fwd_ret_*`)
All 10 forward return targets are now discovered and ranked:

1. `fwd_ret_5m` - 5-minute forward return
2. `fwd_ret_10m` - 10-minute forward return
3. `fwd_ret_15m` - 15-minute forward return
4. `fwd_ret_30m` - 30-minute forward return
5. `fwd_ret_60m` - 60-minute forward return
6. `fwd_ret_120m` - 120-minute forward return
7. `fwd_ret_1d` - 1-day forward return
8. `fwd_ret_5d` - 5-day forward return
9. `fwd_ret_20d` - 20-day forward return
10. `fwd_ret_oc_same_day` - Open-to-close same day return

**All validated:**
- Non-degenerate (multiple unique values)
- Non-zero variance
- Sufficient samples (~188k each)

---

## Validation Logic

The updated `discover_all_targets()` function now:

1. **Discovers both target types:**
 - `y_*` targets (existing)
 - `fwd_ret_*` targets (new)

2. **Validates targets:**
 - Checks for single class (degenerate)
 - For `fwd_ret_*`: Also checks variance (must be > 1e-6)
 - Skips `first_touch` targets (leaked)

3. **Categorizes correctly:**
 - `y_*` targets: Classification (≤10 unique) or Regression (>10 unique)
 - `fwd_ret_*` targets: Always Regression (continuous)

---

## Feature Filtering (Already Working)

The script already correctly filters leaking features:

 **291 safe features** used for training
 **152 excluded features** (leaking/metadata)
- Definite leaks: `tth_*`, `mfe_share_*`, `time_in_profit_*`, etc.
- Temporal overlap: `ret_30m`, `ret_60m`, `vol_30m`, `vol_60m` (for 60m targets)
- Metadata: `ts`, `datetime`, `symbol`, etc.

**No changes needed** - feature filtering is working correctly.

---

## Usage

Run the updated script exactly as before:

```bash
conda activate trader_env
cd /home/Jennifer/trader

python SCRIPTS/rank_target_predictability.py \
  --discover-all \
  --output-dir results/target_ranking_2
```

**Expected output:**
```
 Auto-discovering ALL targets from data...
  Discovered 63 valid targets
    - y_* targets: 53
    - fwd_ret_* targets: 10
  Skipped 9 degenerate targets (single class/zero variance)
  Skipped 1 first_touch targets (leaked)
```

---

## Why Forward Returns Matter

Forward return targets (`fwd_ret_*`) are often **more predictable** than binary barrier targets:

1. **Continuous values** - More information than binary (0/1)
2. **Commonly used** - Standard in quantitative trading
3. **Directly tradeable** - Predict return magnitude, not just direction
4. **Better for regression** - Models can learn magnitude relationships

**Expected performance:**
- Short horizons (5m-15m): R² = 0.01-0.05 (very challenging)
- Medium horizons (30m-60m): R² = 0.05-0.15 (moderate)
- Long horizons (1d-20d): R² = 0.10-0.25 (more predictable)

---

## Summary

| Category | Count | Status |
|----------|-------|--------|
| **y_* targets** | 53 | Being ranked |
| **fwd_ret_* targets** | 10 | **Now being ranked** |
| **Total targets** | **63** | **All discovered** |
| **Safe features** | 291 | Used for training |
| **Excluded features** | 152 | Correctly filtered |

---

## Next Steps

1. **Re-run target ranking** to get scores for all 63 targets
2. **Compare performance** between `y_*` and `fwd_ret_*` targets
3. **Focus on top targets** for feature selection
4. **Consider ensemble** - combine best `y_*` and `fwd_ret_*` targets

