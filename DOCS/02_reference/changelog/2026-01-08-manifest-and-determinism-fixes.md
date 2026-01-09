# Manifest and Determinism Fixes (2026-01-08)

## Summary

Fixed manifest.json schema consistency and deterministic fingerprint computation to ensure proper determinism tracking and manifest completeness.

---

## Manifest.json Schema Fixes

**Problem**: Manifest.json was missing required fields (`run_metadata`, `target_index`) when they weren't populated, causing schema inconsistency.

**Solution**: 
- **UPDATED**: `create_manifest()` - Always includes `run_metadata` and `target_index` fields (empty dicts if not populated)
  - Ensures consistent schema across all runs
  - Fields are populated when data is available
- **NEW**: `update_manifest_with_run_hash()` - Updates manifest after run_hash is computed
  - Called automatically after `save_run_hash()` completes
  - Refreshes `run_hash`, `run_changes`, `target_index`, and `targets` list
  - Ensures manifest is complete at end of run

**Files Changed**:
- `TRAINING/orchestration/utils/manifest.py`:
  - `create_manifest()` - Always includes `run_metadata` and `target_index` fields
  - `update_manifest_with_run_hash()` - New function to update manifest with run_hash
- `TRAINING/orchestration/intelligent_trainer.py`:
  - Calls `update_manifest_with_run_hash()` after run_hash is saved

**Result**: Manifest.json now has consistent schema with all required fields present, even if empty.

---

## Deterministic Fingerprint Fix

**Problem**: `git.dirty` field (working directory state) was included in deterministic fingerprint, causing fingerprints to change between runs even with identical settings.

**Solution**:
- **UPDATED**: `compute_deterministic_config_fingerprint()` - Now excludes `git.dirty` from deterministic fingerprint
  - Excludes: `run.run_id`, `run.timestamp`, `git.dirty`
  - Keeps: `run.seed_global`, `run.mode`, `git.commit` (code version), all other deterministic fields

**Files Changed**:
- `TRAINING/orchestration/utils/manifest.py`:
  - `compute_deterministic_config_fingerprint()` - Excludes `git.dirty` field

**Result**: Deterministic fingerprints are now truly stable across runs with identical settings, even if working directory has uncommitted changes.

---

## Related Changes

- Config loading fixes for `multi_model` and `multi_model_feature_selection` (see config cleanup changelog)
- Universe signature fixes for batch handling (see audit report)

---

**Impact**: 
- ✅ Manifest.json schema is now consistent and complete
- ✅ Deterministic fingerprints are stable across runs with identical settings
- ✅ Run hash computation uses correct deterministic fingerprint
- ✅ All determinism tracking features working correctly in strict mode
