# Single Source of Truth (SST) Enforcement Design

**Date**: 2025-12-13  
**Goal**: Prevent split-brain by defining a shared contract and centralized policy resolution

## Problem

Current implementation has "patches at callsites" that allow split-brain:
- Gatekeeper passes, POST_PRUNE fails (featureset mis-wire)
- Unknown features get `1440m` in one place, `inf` in another (lookback oracle mismatch)
- `roc_50` (250m) not caught at gatekeeper (pattern missing)
- Policy resolution scattered ("strict implies drop" footgun)

## Solution: SST Design

### 1. Shared Contract: `EnforcedFeatureSet`

**Location**: `TRAINING/utils/lookback_cap_enforcement.py`

After any stage that can change features, the pipeline MUST produce an `EnforcedFeatureSet`, and downstream code must take THAT, not raw `feature_names` lists.

```python
@dataclass
class EnforcedFeatureSet:
    features: List[str]  # Safe, ordered feature list (the truth)
    fingerprint: str  # Set-invariant fingerprint (for validation)
    cap_minutes: Optional[float]  # Cap that was enforced
    actual_max_minutes: float  # Actual max lookback from safe features
    canonical_map: Dict[str, float]  # Canonical lookback map (SST)
    quarantined: Dict[str, float]  # Feature → lookback for quarantined
    unknown: List[str]  # Features with unknown lookback (inf)
    stage: str  # Stage name
    budget: Any  # LeakageBudget object
```

**Rule**: After enforcement, slice X immediately:
```python
enforced = apply_lookback_cap(...).to_enforced_set(stage="GATEKEEPER", cap_minutes=cap)
X = X.loc[:, enforced.features]  # NO rediscovery from X.columns
```

### 2. Centralized Policy Resolution

**Location**: `TRAINING/utils/lookback_policy.py`

Resolve policy ONCE from config, pass it everywhere:

```python
policy = resolve_lookback_policy(resolved_config)
# policy.over_budget_action: "hard_stop" | "drop" | "warn"
# policy.unknown_lookback_action: "hard_stop" | "drop" | "warn"
# policy.unknown_policy: "drop" (inf) | "conservative" (1440m)
```

**Key invariant**: ranking + selection must call the same `resolve_lookback_policy()`.

**No more**: "strict implies drop" - policy is explicit and deterministic.

### 3. Reusable Invariant Check

**Location**: `TRAINING/utils/lookback_policy.py`

```python
assert_featureset_fingerprint(
    label="MODEL_TRAIN_INPUT",
    expected=enforced_post_prune,
    actual_features=feature_names
)
```

Call at **every boundary**:
- After cleaning (`SAFE_CANDIDATES`)
- After leak removal (`AFTER_LEAK_REMOVAL`)
- After gatekeeper (`POST_GATEKEEPER`)
- After pruning (`POST_PRUNE`)
- Before model training (`MODEL_TRAIN_INPUT`)
- FS pre/post enforcement

### 4. Fixed Immediate Issues

#### A) Gatekeeper treats `inf` as violation ✅

**Location**: `TRAINING/utils/lookback_cap_enforcement.py` lines 138-181

- Unknown (`inf`) lookback now quarantined EXACTLY like `lookback > cap`
- Logs show `unknown=N` count separately
- Top offenders show `(inf)` for unknown features

#### B) Added `roc` pattern to inference ✅

**Location**: `TRAINING/utils/leakage_budget.py` lines 497, 172

- Pattern: `r'^(rsi|cci|mfi|atr|adx|macd|bb|mom|std|var|roc)_(\d+)$'`
- `roc_50` → 50 bars × 5m = 250m (correctly caught at gatekeeper)

#### C) Unknown policy consistency ✅

**Location**: `TRAINING/utils/leakage_budget.py` lines 1118-1155

- In strict mode: `unknown_policy="drop"` → unknown features get `inf`
- Gatekeeper and POST_PRUNE use same policy (no 1440m vs inf mismatch)

## Migration Path

### Phase 1: Core Infrastructure ✅ (Done)

- [x] `EnforcedFeatureSet` dataclass
- [x] `LookbackPolicy` and `resolve_lookback_policy()`
- [x] `assert_featureset_fingerprint()` helper
- [x] Gatekeeper treats `inf` as violation
- [x] `roc` pattern added
- [x] Unknown policy consistency

### Phase 2: Update Enforcement (TODO)

- [ ] Update `apply_lookback_cap()` to return `EnforcedFeatureSet` directly (or via `.to_enforced_set()`)
- [ ] Update gatekeeper to use `EnforcedFeatureSet` and slice X immediately
- [ ] Update POST_PRUNE to use `EnforcedFeatureSet`
- [ ] Update feature selection (FS_PRE, FS_POST) to use `EnforcedFeatureSet`

### Phase 3: Add Invariant Checks (TODO)

- [ ] Add `assert_featureset_fingerprint()` at SAFE_CANDIDATES
- [ ] Add `assert_featureset_fingerprint()` at AFTER_LEAK_REMOVAL
- [ ] Add `assert_featureset_fingerprint()` at POST_GATEKEEPER
- [ ] Add `assert_featureset_fingerprint()` at POST_PRUNE
- [ ] Add `assert_featureset_fingerprint()` at MODEL_TRAIN_INPUT (already done)
- [ ] Add `assert_featureset_fingerprint()` at FS_PRE/FS_POST

### Phase 4: Canonical Map Cache (TODO)

- [ ] Implement `get_or_build_canonical_map(fingerprint, features, policy, interval)`
- [ ] Cache in `RunContext` or `resolved_config`
- [ ] Hard-stop if same fingerprint produces different `actual_max`

## Definition of Done

After full migration:

✅ **No featureset mis-wire**: All stages use `EnforcedFeatureSet`, invariant checks at boundaries

✅ **No lookback oracle mismatch**: Same `unknown_policy` everywhere, canonical map cached by fingerprint

✅ **No late surprises**: `roc_50` caught at gatekeeper, unknown features quarantined early

✅ **Consistent policy**: One `resolve_lookback_policy()` call, explicit actions (no "strict implies drop")

✅ **No rediscovery**: X sliced immediately after enforcement, never use `X.columns` later

## Testing

1. **Run with strict mode**:
   - Gatekeeper should show `unknown=37` (if suffixless features present)
   - POST_PRUNE should never see those 37 features
   - `roc_50` should be quarantined at gatekeeper (250m > 240m cap)

2. **Check invariant checks**:
   - All boundary checks should pass (or fail with clear error)
   - Fingerprint mismatches caught immediately

3. **Verify policy consistency**:
   - Same `unknown_policy` used in gatekeeper and POST_PRUNE
   - Unknown features get `inf` (not 1440m) in strict mode

## Related Files

- `TRAINING/utils/lookback_cap_enforcement.py`: `EnforcedFeatureSet`, updated quarantine logic
- `TRAINING/utils/lookback_policy.py`: Policy resolution, invariant check helper
- `TRAINING/utils/leakage_budget.py`: `roc` pattern, unknown policy consistency
- `TRAINING/ranking/predictability/model_evaluation.py`: Invariant check (MODEL_TRAIN_INPUT)
