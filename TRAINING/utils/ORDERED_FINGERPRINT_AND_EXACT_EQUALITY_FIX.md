# Ordered Fingerprint + Exact Equality Checks

**Date**: 2025-12-13  
**Goal**: Implement ordered fingerprint + exact list equality checks to prevent subtle drift

## Changes Implemented

### 1. EnforcedFeatureSet: Both Set and Ordered Fingerprints ✅

**Location**: `TRAINING/utils/lookback_cap_enforcement.py` lines 34-60

**Before**: Only `fingerprint: str` (set-invariant)

**After**: Both fingerprints stored:
```python
@dataclass
class EnforcedFeatureSet:
    fingerprint_set: str        # hash(sorted(features)) - for cache keys
    fingerprint_ordered: str    # hash(tuple(features)) - for validation
    features: List[str]  # Safe, ordered feature list (the truth)
    
    @property
    def fingerprint(self) -> str:
        """Backward compatibility: return set fingerprint."""
        return self.fingerprint_set
```

**Impact**: Can detect order divergence, not just set membership changes

### 2. Enhanced `assert_featureset_fingerprint()` with Exact Equality ✅

**Location**: `TRAINING/utils/lookback_policy.py` lines 125-220

**Before**: Only checked hash equality + set diff

**After**: 
- **Exact list equality check first** (not just hash)
- **Order divergence detection** (first index of divergence + window)
- **Actionable diff** (added/removed features, order divergence)
- **Stage/cap/policy in error message** (pin blame instantly)

**Key improvements**:
```python
# 1. Check exact list equality first (not just hash)
if actual_features == expected.features:
    return  # Exact match - pass

# 2. Find first index where order diverges
for i in range(min_len):
    if expected.features[i] != actual_features[i]:
        order_divergence_idx = i
        # Show window around divergence (5 before, 5 after)
        break

# 3. Build actionable error with stage/cap/policy
error_parts = [
    f"🚨 FEATURESET MIS-WIRE ({label}):",
    f"   Stage: {expected.stage} → {label}",
    f"   Cap: {expected.cap_minutes:.1f}m" if expected.cap_minutes else "   Cap: None",
    f"   Added features ({len(added)}): {list(added)[:10]}",
    f"   Order divergence at index {order_divergence_idx}: ..."
]
```

### 3. Order Drift Clamping After Transforms ✅

**Location**: 
- `TRAINING/utils/cross_sectional_data.py` lines 493-503 (AFTER_CLEANING)
- `TRAINING/ranking/predictability/model_evaluation.py` lines 4527-4534 (AFTER_LEAK_REMOVAL)

**Before**: DataFrame columns could be reordered after cleaning/leak removal, causing "(order changed)" warnings

**After**: Reindex DataFrame columns to match `feature_names` order exactly:
```python
# After cleaning
if isinstance(feature_df, pd.DataFrame):
    # Reindex columns to match feature_names order exactly
    feature_df = feature_df.loc[:, [f for f in feature_names if f in feature_df.columns]]

# After leak removal
# Note: For numpy arrays, column order is implicit via feature_names list
# The feature_names list IS the authoritative order - X columns must match it
```

**Impact**: Prevents "(order changed)" warnings, ensures deterministic column alignment

### 4. Updated Invariant Check in model_evaluation.py ✅

**Location**: `TRAINING/ranking/predictability/model_evaluation.py` lines 1339-1380

**Before**: Only hash comparison

**After**: 
- Uses `assert_featureset_fingerprint()` helper if `EnforcedFeatureSet` available
- Fallback: Exact list equality check first, then hash
- Order divergence detection in fallback

## Result

✅ **Ordered fingerprint stored** - Can detect order divergence

✅ **Exact list equality checked** - Not just hash (catches all mismatches)

✅ **Actionable diffs** - Shows first divergence index, added/removed features

✅ **Order drift clamped** - DataFrame columns reindexed after transforms

✅ **Stage/cap/policy in errors** - Pin blame instantly

## Testing

1. **Run with order drift**: Should see reindexing prevent "(order changed)" warnings
2. **Run with featureset mis-wire**: Should see exact equality check catch it immediately with actionable diff
3. **Check logs**: Should see both set and ordered fingerprints in EnforcedFeatureSet

## Related Files

- `TRAINING/utils/lookback_cap_enforcement.py`: EnforcedFeatureSet with both fingerprints
- `TRAINING/utils/lookback_policy.py`: Enhanced assert_featureset_fingerprint()
- `TRAINING/utils/cross_sectional_data.py`: Order drift clamping after cleaning
- `TRAINING/ranking/predictability/model_evaluation.py`: Updated invariant check
