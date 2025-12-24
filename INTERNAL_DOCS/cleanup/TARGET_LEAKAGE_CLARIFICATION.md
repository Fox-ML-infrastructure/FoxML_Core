# Target Leakage Clarification

## The Confusion: Features vs Targets

You're right to be concerned, but there's an important distinction:

### **LEAKING FEATURES** (excluded from training)
These are **features** (inputs) that leak future information:
- `tth_*` - time-to-hit (knows when barrier will be hit)
- `mfe_share_*` - fraction of time in profit (requires future path)
- `time_in_profit_*` - duration in profit (requires future path)
- `hit_direction_*` - which barrier hits first (correlated with `first_touch`)

**These are correctly excluded** - you can't use them as features to predict targets.

### **TARGETS** (what we're predicting)
These are **targets** (outputs) - what we want to predict:
- `y_will_peak_*` - will price hit upper barrier?
- `y_will_valley_*` - will price hit lower barrier?
- `y_will_swing_*` - will there be a swing high/low?

**These are fine to use as targets** - they're what we're trying to predict!

### **LEAKED TARGETS** (excluded from ranking)
Only **1 target** is explicitly skipped:
- `y_first_touch_60m_0.8` - This is perfectly correlated (r=1.0) with the feature `hit_direction_60m_0.8`, making it a "feature disguised as a target"

---

## Current Status

From your data:
- **63 total `y_*` targets** found
- **53 valid targets** being ranked
- **9 degenerate targets** (single class) - skipped
- **1 leaked target** (`first_touch`) - skipped

**All your other targets are being ranked!** The script is working correctly.

---

## Additional Targets You Have (Not Being Ranked)

You also have these columns that **could** be targets but aren't being ranked:

### 1. **Forward Returns** (`fwd_ret_*`)
```
fwd_ret_1d, fwd_ret_5d, fwd_ret_20d
fwd_ret_15m, fwd_ret_30m, fwd_ret_60m
fwd_ret_120m, fwd_ret_240m, fwd_ret_480m, fwd_ret_1440m
```

**These are valid targets!** They're forward returns - commonly used in trading.

**Why not ranked?** The script only looks for `y_*` columns. These start with `fwd_ret_*`.

**Should we add them?** Yes! These are often more predictable than barrier targets.

---

### 2. **Barrier Levels** (`barrier_*`)
```
barrier_up_60m_0.8
barrier_down_60m_0.8
```

**These are NOT targets** - they're barrier price levels (features), not labels.

---

### 3. **Zigzag Labels** (`zigzag_*`)
```
zigzag_high
zigzag_low
```

**These could be targets** - they indicate swing points.

**Why not ranked?** The script only looks for `y_*` columns.

---

### 4. **Probabilities** (`p_*`)
```
p_up_60m_0.8
p_down_60m_0.8
```

**These are NOT targets** - they're probability estimates (could be features or metadata).

---

## Recommendation: Add Forward Return Targets

The **forward return targets** (`fwd_ret_*`) are the most valuable addition:

1. **They're commonly used** in trading strategies
2. **They're often more predictable** than binary barrier targets
3. **They're already in your data** - just need to rank them

### How to Add Them

We can modify `rank_target_predictability.py` to also discover `fwd_ret_*` columns:

```python
# In discover_all_targets():
# Find all y_ columns
all_targets = [c for c in df.columns if c.startswith('y_')]

# ALSO find forward return targets
fwd_ret_targets = [c for c in df.columns if c.startswith('fwd_ret_')]
all_targets.extend(fwd_ret_targets)
```

This would add **10 more targets** to rank!

---

## Summary

| Category | Count | Status | Action |
|----------|-------|--------|--------|
| `y_*` targets | 63 | 53 being ranked | Working correctly |
| `fwd_ret_*` targets | 10 | Not ranked | **Should add!** |
| `zigzag_*` targets | 2 | Not ranked | Could add |
| Leaked targets | 1 | Skipped | Correctly excluded |
| Degenerate targets | 9 | Skipped | Correctly excluded |

---

## Next Steps

1. **Current run is fine** - ranking 53 valid `y_*` targets
2. **After it finishes**, we can add `fwd_ret_*` targets to the ranking script
3. **Re-run** to get rankings for forward returns too

Would you like me to update the script to also rank `fwd_ret_*` targets?

