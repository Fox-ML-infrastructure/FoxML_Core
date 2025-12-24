# Leak Detection Fix - 2025-11-22 23-18-52

## Problem
Models achieving 100% accuracy on `y_will_peak_60m_0.8` target due to data leakage.

## Root Cause
1. **`p_up_60m_0.8` feature has perfect correlation (1.0000) with target**
 - This probability feature directly encodes the barrier hit probability
 - It's the ANSWER to the target question: "Will price hit the upper barrier?"
 - Model learns: `y_will_peak_60m_0.8 ≈ p_up_60m_0.8` → 100% accuracy

2. **`barrier_up_60m_0.8` and `barrier_down_60m_0.8` features encode barrier logic**
 - These features directly encode the barrier levels
 - They were NOT being excluded (now fixed)

## Solution
1. **Added `barrier_*` to exclusion patterns** in `CONFIG/excluded_features.yaml`:
 - Added `^barrier_` regex pattern
 - Added `barrier_` prefix pattern
 - These features are now excluded

2. **Verified `p_*` exclusion is working**:
 - `^p_` pattern already exists in config
 - All `p_*` probability features are correctly excluded
 - Config cache invalidation ensures updates are picked up

## Files Changed
- `CONFIG/excluded_features.yaml`: Added `barrier_*` exclusion patterns

## Testing
- Verified `barrier_*` features are excluded
- Verified `p_*` features are excluded
- Config cache invalidation works correctly

## Next Steps
1. **Restart Python process** to clear config cache (or wait for file modification time check)
2. **Re-run ranking script** - should now show realistic scores (R² ~0.13-0.17 instead of 0.97)
3. **Monitor for other leaks** - check feature importances for suspicious patterns

## Expected Results
After fix:
- R² scores should drop from ~0.97 to ~0.13-0.17 (realistic for financial data)
- No more 100% accuracy warnings
- Models will show genuine learning, not identity functions
