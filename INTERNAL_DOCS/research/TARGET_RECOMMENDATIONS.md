# Target Recommendations & Analysis

## Current Status

 **Script is working correctly!** The warnings about degenerate targets are **expected behavior** - those targets are being skipped because they have insufficient data for cross-validation.

## Performance Summary (from your run)

### **High-Performing Targets** (R² > 0.10)
- `peak_60m_0.8`: R² = 0.123 ± 0.040 (predictable!)
- `valley_60m_0.8`: R² = 0.157 ± 0.040 (best so far!)

### **Marginal Targets** (R² ≈ 0.0)
- `swing_high_30m_0.05`: R² = -0.014 ± 0.002 (essentially random)
- `swing_low_30m_0.05`: R² = -0.014 ± 0.001 (essentially random)

### **Poor Targets** (R² < -0.30)
- Most `swing_high_10m_*` and `swing_low_10m_*`: R² ≈ -0.40 (worse than random!)
- `peak_mfe_5m_0.001`: R² = -0.504 ± 0.060 (very unpredictable)
- `valley_mdd_5m_0.001`: R² = -0.495 ± 0.062 (very unpredictable)

## Recommended Additional Targets to Try

### 1. **Excess Return Targets** (Market-Adjusted)
**Why:** Removes market beta, focuses on stock-specific alpha.

```python
# These would need to be generated in your pipeline:
- y_excess_ret_15m: Beta-adjusted 15m forward return
- y_excess_ret_30m: Beta-adjusted 30m forward return
- y_excess_ret_60m: Beta-adjusted 60m forward return
- y_excess_ret_class_15m: 3-class classification (down/neutral/up)
```

**Expected Performance:** Often better than raw returns because they remove market noise.

---

### 2. **Cross-Sectional Ranking Targets**
**Why:** Relative performance is often more predictable than absolute.

```python
# Rank-based targets (percentile ranks across universe):
- y_xrank_ret_15m: Cross-sectional return rank (0-100 percentile)
- y_xrank_ret_30m: Cross-sectional return rank
- y_xrank_idio_15m: Idiosyncratic return rank
```

**Expected Performance:** Can achieve R² = 0.15-0.25 if you have good cross-sectional features.

---

### 3. **Time-to-Hit (TTH) Targets**
**Why:** Predicts *when* a barrier will be hit, not just *if*.

```python
# Regression targets (minutes until barrier hit):
- y_tth_peak_60m_0.8: Minutes until upper barrier hit
- y_tth_valley_60m_0.8: Minutes until lower barrier hit
```

**Expected Performance:** R² = 0.10-0.20 if timing is predictable.

**Note:** These might already exist in your data as `tth_*` columns, but check for leakage!

---

### 4. **Regime-Conditional Targets**
**Why:** Different targets work in different market regimes.

```python
# Regime classification (3-5 classes):
- y_regime_trend_15m: Trending (0), Choppy (1), Volatile (2)
- y_regime_vol_15m: Low vol (0), Medium (1), High (2)
- y_regime_liq_15m: Low liquidity (0), Medium (1), High (2)

# Then create regime-conditional barrier targets:
- y_will_peak_60m_0.8_trending: Only in trending regime
- y_will_valley_60m_0.8_choppy: Only in choppy regime
```

**Expected Performance:** Can improve R² by 0.05-0.10 if regimes are well-defined.

---

### 5. **Ordinal/Multi-Class Targets**
**Why:** More granular than binary, less noisy than regression.

```python
# Ordinal classification (7 classes: -3 to +3):
- y_ret_ord_15m: Ordinal return buckets
- y_mfe_ord_15m: Ordinal MFE buckets
- y_mdd_ord_15m: Ordinal MDD buckets
```

**Expected Performance:** R² = 0.10-0.20 if classes are well-balanced.

---

### 6. **Path Quality Targets** (if not leaked!)
**Why:** Predicts trade quality, not just direction.

```python
# ️ WARNING: Check for leakage first!
- y_mfe_share_15m: Fraction of time in profit (0-1)
- y_time_in_profit_15m: Minutes in profit
- y_flipcount_15m: Number of profit/loss flips
```

**Expected Performance:** R² = 0.15-0.25 if not leaked, but many of these are already excluded.

---

### 7. **Asymmetric Barrier Targets**
**Why:** Different thresholds for up vs. down barriers.

```python
# Asymmetric barriers (e.g., 0.8 up, 0.5 down):
- y_hit_asym_60m_0.8_0.5: Up barrier 0.8, down barrier 0.5
- y_hit_asym_60m_1.0_0.6: Up barrier 1.0, down barrier 0.6
```

**Expected Performance:** Similar to symmetric barriers, but can be better for directional strategies.

---

## Targets to **AVOID** (Based on Your Results)

 **Swing targets with tight thresholds (0.05)**: Too noisy, negative R²
 **Very short horizons (5m) with small thresholds (0.001)**: Essentially random
 **First-touch targets**: Already identified as leaked (correlated with `hit_direction`)

## Next Steps

1. **Wait for current run to complete** - Check `results/target_ranking_2/target_rankings.csv`

2. **Focus on top 10-20 targets** by composite score for feature ranking

3. **Consider generating new targets**:
 - Start with excess returns (easiest to add)
 - Then cross-sectional ranks (requires universe data)
 - Then regime-conditional targets (requires regime detection)

4. **Re-run ranking** with new targets to see if they outperform

## Implementation Priority

**High Priority (Easy to Add):**
1. Excess return targets (just need SPY/SPX data for beta adjustment)
2. Asymmetric barrier targets (modify existing barrier code)

**Medium Priority (Moderate Effort):**
3. Cross-sectional ranking targets (need universe aggregation)
4. Time-to-hit targets (if not already generated, check for leakage)

**Low Priority (More Complex):**
5. Regime-conditional targets (need regime detection first)
6. Ordinal targets (need to define buckets)

---

## Questions to Consider

1. **Do you have SPY/SPX data?** → Needed for excess return targets
2. **Do you have universe-level data?** → Needed for cross-sectional ranks
3. **Are regime features already computed?** → Check `DATA_PROCESSING/features/regime_features.py`
4. **What's your primary trading style?** → This determines which targets matter most
 - Scalping (5-15m) → Focus on short-horizon targets
 - Day trading (15-60m) → Focus on medium-horizon targets
 - Swing trading (60m+) → Focus on long-horizon targets

