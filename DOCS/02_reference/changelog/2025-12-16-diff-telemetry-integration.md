# Diff Telemetry Integration - 2025-12-16

## Overview

This changelog documents the integration of diff telemetry into metadata and metrics outputs, making change tracking available across all pipeline stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING).

## Changes

### Diff Telemetry Integration

**Problem**: Diff telemetry was computed but not persisted in metadata/metrics outputs, making it difficult to audit changes and query for runs with excluded factor changes.

**Solution**: Integrated diff telemetry into existing metadata.json and metrics.json outputs with appropriate data split (full audit trail in metadata, lightweight queryable fields in metrics).

**Files Changed**:
- `TRAINING/utils/diff_telemetry.py` - Added `_count_excluded_factors_changed()` method, improved `_format_excluded_factors_summary()` with consistent counting logic
- `TRAINING/utils/reproducibility_tracker.py` - Added diff telemetry integration into metadata.json
- `TRAINING/utils/metrics.py` - Added lightweight diff telemetry fields to metrics.json

**Key Changes**:

#### Metadata Integration (Full Audit Trail)

Added `diff_telemetry` section to `metadata.json` with:
- `fingerprint_schema_version`: Schema version for compatibility checking
- `comparison_group_key`: Stable key for grouping comparable runs
- `comparison_group`: Full structured comparison group
- `fingerprints`: All fingerprint fields (config, data, feature, target)
- `fingerprint_sources`: Source descriptions (e.g., `fold_assignment_hash: "hash over row_id→fold_id mapping"`)
- `comparability`: Comparable flag, reason, and prev_run_id
- `excluded_factors`: Full diff payload with:
  - `changed`: Boolean flag
  - `summary`: Human-readable summary (e.g., "learning_rate: 0.01→0.05, max_depth: 5→7 (+2 more)")
  - `changes`: Full `{"prev":..., "curr":...}` map for all hyperparameters, train_seed, and versions

#### Metrics Integration (Lightweight Queryable Fields)

Added lightweight `diff_telemetry` section to `metrics.json` with:
- `comparable`: 0/1 flag
- `excluded_factors_changed`: 0/1 flag
- `excluded_factors_changed_count`: Integer count of changed keys (number of keys whose value differs)
- `excluded_factors_summary`: Human-readable summary (for quick log-style viewing)

**Count Rule**: `excluded_factors_changed_count` counts the number of keys whose value differs (one key = one change). This matches the summary formatter counting logic:
- Each hyperparameter key = 1 change
- `train_seed` = 1 change (if present)
- Each version key = 1 change

#### Backwards Compatibility

- `diff_telemetry` is optional in metadata (older runs without it are handled gracefully)
- Schema mismatches produce valid `diff_telemetry` structure with `comparability.comparable = false` and clear reason
- First run / no previous run returns stable shape with empty excluded factors

#### Edge Cases Handled

- **First run**: `prev_run_id = None`, `excluded_factors.changes = {}` (empty but present)
- **Schema mismatch**: Valid structure with `comparability.reason = "Different fingerprint schema versions: ..."`
- **Not comparable**: Still returns valid structure with excluded_factors present

#### Metrics Cardinality Control

- Removed `comparison_group_key` from metrics (can be gated behind config flag if needed)
- Only lightweight fields in metrics (no full payloads)
- Full payloads only in metadata.json (for audit trail)

## Integration Flow

1. `DiffTelemetry.finalize_run()` returns diff telemetry data
2. `ReproducibilityTracker._save_to_cohort()` captures return value and stores in `additional_data['diff_telemetry']`
3. Metadata integration: Full diff telemetry added to `metadata.json` before save
4. Metrics integration: Lightweight diff telemetry added to `metrics.json` via `MetricsWriter.write_cohort_metrics()`

## Benefits

- **Fast queries**: Lightweight fields in metrics for aggregation/dashboards
- **Full auditability**: Complete diff payload in metadata for reviewers
- **Consistent contract**: All stages (TARGET_RANKING, FEATURE_SELECTION, TRAINING) share same telemetry contract
- **Backwards compatible**: Older runs without diff_telemetry handled gracefully
- **Stable shapes**: All edge cases return consistent structure

## Testing

Minimal test cases verified:
- `metadata.json` contains full `diff_telemetry` with `fingerprint_sources`
- `metrics.json` contains only lightweight subset
- Schema mismatch produces correct `comparability.reason`
- `split_seed` change blocks comparability (in comparison_group)
- `train_seed` change does not block comparability but shows up in excluded factor changes
- Count logic matches summary formatter (number of keys that changed)

## Migration Notes

- No migration needed - system is backwards compatible
- Older runs without `diff_telemetry` in metadata are handled gracefully
- New runs automatically include diff telemetry in both metadata and metrics

## Additional Improvements (Later Same Day)

### Diff Telemetry Digest Enhancements

**Problem**: Digest computation used truncated hash and fallback string coercion, which could hide normalization bugs.

**Solution**: Implemented fail-fast assertion with full SHA256 hash for maximum integrity and early bug detection.

**Files Changed**:

**Key Changes**:

#### Fail-Fast Type Safety
- **Removed `default=str` fallback** - Now raises `RuntimeError` if non-JSON-primitive types are detected
- **Strict serialization** - `json.dumps()` called without fallback, ensuring normalization bugs are caught immediately
- **Clear error messages** - Logs indicate normalization failures with full context

#### Full SHA256 Hash
- **64 hex characters** - Changed from truncated 32 chars to full SHA256 hash (256 bits of entropy)
- **Maximum collision resistance** - Full hash eliminates any truncation concerns
- **Consistent across all outputs** - Same full hash stored in both `metadata.json` and `metrics.json`

#### Documentation Updates
- Updated algorithm description to reflect fail-fast behavior
- Updated verification example to show strict serialization
- Updated JSON examples to show full 64-character hash format

**Benefits**:
- **Early bug detection** - Normalization bugs fail immediately rather than being silently hidden
- **Maximum integrity** - Full SHA256 provides strongest possible collision resistance
- **Review-proof** - No fallback coercion means reviewers can trust the digest represents exact data
- **Clear failure modes** - Errors clearly indicate normalization issues upstream

**Breaking Changes**: None - this is a hardening change that makes the system more strict, but existing valid data continues to work.

