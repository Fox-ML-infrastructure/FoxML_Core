# Run Identity System

The run identity system ensures that stability analysis and diff telemetry only compare truly equivalent runs. It provides cryptographically robust identity keys derived from canonical payloads.

## Quick Start

### Default Behavior (Production)

By default, the system runs in **strict mode**:
- Snapshots require a finalized `RunIdentity` with all required signatures
- Missing signatures cause the snapshot save to fail
- Stability analysis only groups runs with matching identity keys

No configuration needed - this is the safe default.

### Configuration

Identity enforcement is configured in `CONFIG/identity_config.yaml`:

```yaml
identity:
  mode: strict  # strict | relaxed | legacy

stability:
  filter_mode: replicate  # replicate | strict | legacy
  allow_legacy_snapshots: false
  min_snapshots: 2

feature_identity:
  mode: registry_resolved  # registry_resolved | names_only
```

## Core Concepts

### Identity Keys

Every finalized run has two identity keys:

| Key | Includes Seed | Use Case |
|-----|---------------|----------|
| `replicate_key` | No | Cross-seed stability analysis (grouping runs with different seeds) |
| `strict_key` | Yes | Diff telemetry (comparing runs with same seed) |

Both keys are 64-character SHA256 hashes of canonical payloads.

### Component Signatures

Identity is built from these component signatures:

| Signature | Computed From | When Available |
|-----------|---------------|----------------|
| `dataset_signature` | data_dir, symbols, max_samples_per_symbol | Pipeline start |
| `target_signature` | target column, task type, horizon | Target selection |
| `routing_signature` | view, symbol (if SS), routing config | Routing decision |
| `split_signature` | CV method, fold boundaries, row counts | Fold creation |
| `hparams_signature` | Model hyperparameters (per family) | Model setup |
| `feature_signature` | Final feature specs with registry metadata | After feature selection |

### Two-Phase Construction

Identity is built in two phases:

1. **Partial Identity** (early pipeline)
   - Created in `IntelligentTrainer` before feature selection
   - Has `is_final=False`, no keys computed
   - Missing `feature_signature` (features not yet selected)

2. **Finalized Identity** (after features locked)
   - Created via `partial.finalize(feature_signature)`
   - Has `is_final=True`, keys computed
   - Required for snapshot saving

```python
# Phase 1: Create partial identity
partial = RunIdentity(
    dataset_signature="abc123...",
    split_signature="def456...",
    target_signature="ghi789...",
    hparams_signature="jkl012...",
    routing_signature="mno345...",
    train_seed=42,
    is_final=False
)

# Phase 2: Finalize after features are selected
final = partial.finalize(feature_signature="pqr678...")

# Now keys are available
print(final.replicate_key)  # 64-char hash (excludes seed)
print(final.strict_key)     # 64-char hash (includes seed)
```

## Enforcement Modes

### Strict Mode (Production Default)

```yaml
identity:
  mode: strict
```

Behavior:
- **Partial identity** → `ValueError` raised, snapshot not saved
- **Missing identity** → `ValueError` raised, snapshot not saved
- **Training continues** but snapshot is skipped (fail-fast for identity)

Use for: Production runs where you need guaranteed reproducibility.

### Relaxed Mode (Development)

```yaml
identity:
  mode: relaxed
```

Behavior:
- **Partial identity** → ERROR logged, snapshot saved anyway
- **Missing identity** → ERROR logged, snapshot saved with legacy path

Use for: Development and debugging when you want to see all outputs.

### Legacy Mode (Migration Only)

```yaml
identity:
  mode: legacy
```

Behavior:
- All snapshots saved regardless of identity status
- Warnings logged for missing identity

Use for: Migrating from old snapshot format or backward compatibility.

## Stability Analysis Modes

### Replicate Mode (Default)

Groups runs by `replicate_key` (excludes seed):

```yaml
stability:
  filter_mode: replicate
```

Same replicate_key means:
- Same dataset
- Same features
- Same hyperparameters
- Same target
- Same splits

Different seeds produce same replicate_key → grouped for stability analysis.

### Strict Mode

Groups runs by `strict_key` (includes seed):

```yaml
stability:
  filter_mode: strict
```

Only compares runs with identical configurations including seed.

### Legacy Mode

Best-effort grouping by universe_sig only:

```yaml
stability:
  filter_mode: legacy
  allow_legacy_snapshots: true
```

## Feature Identity

### Registry-Resolved Mode (Recommended)

```yaml
feature_identity:
  mode: registry_resolved
```

Feature fingerprint includes per-feature registry metadata:
- `lag_bars` - lookback window
- `source` - feature source
- `allowed_horizons` - valid horizons
- `version` - registry version
- `rejected` - rejection status

Provenance markers indicate how features were resolved:
- `registry_explicit` - All features have explicit registry entries
- `registry_mixed` - Some explicit, some auto-inferred
- `registry_inferred` - All auto-inferred from patterns
- `names_only_degraded` - Registry unavailable

### Names-Only Mode (Deprecated)

```yaml
feature_identity:
  mode: names_only
```

Only hashes feature names. Not recommended - different implementations of same-named features would have identical signatures.

## Hash-Based Storage

Snapshots are stored under identity-keyed paths:

```
feature_importance_snapshots/
  replicate/<replicate_key>/
    <strict_key>.json
```

Benefits:
- **No collisions** - 64-char hashes are unique
- **Fast grouping** - Glob replicate directory for stability
- **Debuggable** - `debug_key` stored inside snapshot

## API Reference

### RunIdentity

```python
from TRAINING.common.utils.fingerprinting import RunIdentity

@dataclass
class RunIdentity:
    # Required signatures
    dataset_signature: Optional[str]
    split_signature: Optional[str]
    target_signature: Optional[str]
    hparams_signature: Optional[str]
    routing_signature: Optional[str]
    feature_signature: Optional[str]  # Set during finalize()
    
    # Seed (included in strict_key only)
    train_seed: Optional[int]
    
    # State
    is_final: bool = False
    
    # Computed keys (only when is_final=True)
    replicate_key: Optional[str]  # computed
    strict_key: Optional[str]     # computed
    debug_key: Optional[str]      # computed
    
    def finalize(self, feature_signature: str) -> 'RunIdentity':
        """Create finalized identity with computed keys."""
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
```

### Config Functions

```python
from TRAINING.common.utils.fingerprinting import (
    get_identity_config,
    get_identity_mode,
)

# Get full config
config = get_identity_config()
print(config["identity"]["mode"])  # "strict"

# Get just the mode
mode = get_identity_mode()  # "strict"
```

### Fingerprint Functions

```python
from TRAINING.common.utils.fingerprinting import (
    resolve_feature_specs_from_registry,
    compute_feature_fingerprint_from_specs,
    compute_hparams_fingerprint,
    compute_split_fingerprint,
    compute_target_fingerprint,
    compute_routing_fingerprint,
)

# Feature fingerprinting
specs = resolve_feature_specs_from_registry(["rsi_14", "sma_20"])
feature_sig = compute_feature_fingerprint_from_specs(specs)

# Hyperparameter fingerprinting
hparams_sig = compute_hparams_fingerprint(
    model_family="lightgbm",
    hyperparameters={"n_estimators": 100, "learning_rate": 0.1}
)
```

## Troubleshooting

### "Cannot save snapshot without run_identity"

**Cause**: Snapshot hook called without identity in strict mode.

**Fix**: Ensure `run_identity` is passed to `save_snapshot_hook()`.

### "Cannot save snapshot with partial RunIdentity"

**Cause**: Trying to save with `is_final=False`.

**Fix**: Call `run_identity.finalize(feature_signature)` before saving.

### Runs not being compared

**Cause**: Different identity keys (different features, params, etc.).

**Check**: Compare `replicate_key` values in snapshot files.

### Legacy snapshots ignored

**Cause**: `allow_legacy_snapshots: false` and old snapshots lack identity.

**Fix**: Set `allow_legacy_snapshots: true` during migration.

## See Also

- [Deterministic Training](../../00_executive/DETERMINISTIC_TRAINING.md) - Overall reproducibility architecture
- [Diff Telemetry](../../03_technical/telemetry/DIFF_TELEMETRY.md) - Run comparison system
- [Changelog](../changelog/2026-01-03-deterministic-run-identity.md) - Implementation details
