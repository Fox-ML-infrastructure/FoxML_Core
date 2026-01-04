# 2026-01-03: Deterministic Run Identity System

## Summary

Implemented a comprehensive deterministic run identity system to ensure stability analysis and diff telemetry only compare truly equivalent runs. This replaces ad-hoc grouping with cryptographically robust identity keys derived from canonical payloads.

## Problem Statement

Prior to this change, stability analysis could incorrectly compare runs with:
- Different feature sets (same names, different implementations)
- Different hyperparameters
- Different data splits
- Different universe/symbol configurations

This led to misleading stability metrics and comparison deltas.

## Solution

### Core Components

1. **`RunIdentity` SST Dataclass** (`fingerprinting.py`)
   - Two-phase construction: partial (early pipeline) → final (after features locked)
   - `is_final` flag enforced at snapshot save time
   - `finalize(feature_signature)` method validates required fields before computing keys

2. **Identity Keys**
   - `strict_key`: Full 64-char SHA256 including `train_seed` (for diff telemetry)
   - `replicate_key`: Excludes `train_seed` (for cross-seed stability analysis)
   - `debug_key`: Human-readable summary for logging

3. **Component Signatures** (all 64-char SHA256)
   - `dataset_signature`: data_dir, symbols, max_samples_per_symbol
   - `split_signature`: CV method, fold boundaries, fold row counts, purge/embargo config
   - `target_signature`: target column, task type, horizon
   - `feature_signature`: per-feature registry metadata with provenance markers
   - `hparams_signature`: per-model-family hyperparameters
   - `routing_signature`: view, symbol (if SS), contracted routing payload

4. **Canonicalization** (`config_hashing.py`)
   - `canonical_json()`: Deterministic JSON with sorted keys, UTC timestamps, no None values
   - `sha256_full()`: 64-character hex digest
   - `sha256_short()`: Truncated for debug display only

### Feature Identity Enhancement

Feature fingerprinting now includes:
- Registry metadata per feature (lag_bars, source, allowed_horizons, version)
- Explicit provenance markers:
  - `registry_explicit`: All features have explicit registry entries
  - `registry_mixed`: Some explicit, some auto-inferred
  - `registry_inferred`: All auto-inferred from patterns
  - `names_only_degraded`: Registry unavailable
  - `empty`: No features

### Configurable Enforcement

New `CONFIG/identity_config.yaml`:

```yaml
identity:
  mode: strict  # strict | relaxed | legacy

stability:
  filter_mode: replicate
  allow_legacy_snapshots: false

feature_identity:
  mode: registry_resolved
```

| Mode | Missing Signature | Partial Identity |
|------|-------------------|------------------|
| strict | Fail (raise) | Fail (raise) |
| relaxed | Log ERROR, continue | Log ERROR, continue |
| legacy | Log warning | Log warning |

### Hash-Based Storage

Snapshots stored at identity-keyed paths:
```
replicate/<replicate_key>/<strict_key>.json
```

Benefits:
- No collisions (64-char hashes are identity)
- Fast grouping (glob replicate directory)
- Backfill friendly (regenerate strict without rewriting group)

## Files Changed

### New Files
- `CONFIG/identity_config.yaml` - Identity enforcement configuration

### Modified Files
- `TRAINING/common/utils/fingerprinting.py`
  - Added `RunIdentity` dataclass with two-phase construction
  - Added `resolve_feature_specs_from_registry()` for rich feature specs
  - Added `compute_feature_fingerprint_from_specs()` with provenance markers
  - Added `compute_routing_fingerprint()` for routing identity
  - Added `get_identity_config()` and `get_identity_mode()` config loaders

- `TRAINING/common/utils/config_hashing.py`
  - Added `canonicalize()` function with type-specific handling
  - Added `canonical_json()` wrapper
  - Added `sha256_full()` and `sha256_short()` hash functions
  - Updated `compute_config_hash()` to use new helpers

- `TRAINING/stability/feature_importance/hooks.py`
  - Added identity mode checking from config
  - Strict mode: raise on partial/missing identity
  - Relaxed mode: log error, continue
  - Legacy mode: log warning

- `TRAINING/ranking/feature_selector.py`
  - Partial identity creation at start
  - Identity propagation to all snapshot call sites
  - Per-model-family identity finalization
  - Exception handling respects identity mode

- `TRAINING/ranking/predictability/model_evaluation.py`
  - Full identity computation for target ranking
  - Per-model-family hparams/feature signatures
  - Identity passed to `save_feature_importances()`

- `TRAINING/ranking/cross_sectional_feature_ranker.py`
  - CS identity computation and finalization
  - Identity passed to `save_snapshot_from_series_hook()`

- `TRAINING/ranking/multi_model_feature_selection.py`
  - Per-family identity computation
  - Identity passed to snapshot hooks

- `TRAINING/orchestration/intelligent_trainer.py`
  - Partial identity created at pipeline start
  - Passed to feature selection

## Testing

### Verify Config Loading
```python
from TRAINING.common.utils.fingerprinting import get_identity_config, get_identity_mode
print(get_identity_mode())  # Should print: strict
```

### Verify Canonicalization
```python
from TRAINING.common.utils.config_hashing import canonical_json, sha256_full
payload = {"b": 2, "a": 1}
print(canonical_json(payload))  # {"a":1,"b":2}
print(sha256_full(canonical_json(payload)))  # 64-char hash
```

### Verify Identity Creation
```python
from TRAINING.common.utils.fingerprinting import RunIdentity

# Partial identity
partial = RunIdentity(
    dataset_signature="abc123...",
    split_signature=None,  # Not yet computed
    target_signature="def456...",
    hparams_signature="ghi789...",
    routing_signature="jkl012...",
    train_seed=42,
    is_final=False
)
print(partial.is_final)  # False
print(partial.replicate_key)  # None (not computed yet)

# Finalize
final = partial.finalize(
    feature_signature="mno345...",
    split_signature="pqr678..."
)
print(final.is_final)  # True
print(final.replicate_key)  # 64-char hash
print(final.strict_key)  # 64-char hash (includes seed)
```

## Backward Compatibility

- Legacy snapshots (without identity) are ignored in replicate/strict modes
- Set `identity.mode: legacy` to allow legacy snapshots during migration
- Set `stability.allow_legacy_snapshots: true` to include legacy in analysis

## Impact

1. **Stability analysis** only compares runs with matching identity signatures
2. **Diff telemetry** uses strict_key for same-seed comparisons
3. **No silent degradation** in strict mode - missing signatures fail fast
4. **Feature changes detected** - different registry metadata = different signature
5. **Cross-seed stability** works via replicate_key grouping
