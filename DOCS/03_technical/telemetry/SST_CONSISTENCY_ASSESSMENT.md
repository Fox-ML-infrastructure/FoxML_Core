# SST Consistency & Production Readiness Assessment

## Status: ✅ Core Architecture Fixed, ⚠️ Production Hardening Needed

This document assesses the current implementation against the identified concerns and provides a prioritized action plan.

---

## ✅ Immediate Concerns (Already Addressed)

### 1. ✅ Mutation Safety
**Status:** FIXED
- `_build_resolved_context()` uses `dict(resolved_metadata)` shallow copy
- No mutations found in `finalize_run()` or `normalize_snapshot()`
- Only read operations (`.get()`) on metadata dict

### 2. ⚠️ Stage-Scoping of `full_metadata`
**Status:** NEEDS VERIFICATION
- Both call sites use same `full_metadata` built at line 1077
- **Risk:** If call sites are in different stages, wrong stage metadata could be used
- **Fix Needed:** Add assertion: `assert resolved_metadata["stage"] == stage`

### 3. ⚠️ Output Timing / Completeness
**Status:** NEEDS VERIFICATION
- `full_metadata` is built before `finalize_run()` is called
- **Risk:** If required fields (date ranges, N_effective, fold hashes) are computed later, incomplete SST
- **Fix Needed:** Verify `full_metadata` is built AFTER all required fields are finalized, or add hard asserts

### 4. ✅ Digest Correctness
**Status:** FIXED
- Strict JSON-primitive-only validation (raises `RuntimeError` on non-primitive types)
- No `default=str` fallback
- Full SHA256 hash (64 hex characters)

### 5. ✅ Fallback Path Strictness
**Status:** FIXED
- `run_id` guard: ✅ Verifies `file_run_id == current_run_id`
- `stage` guard: ✅ Verifies `file_stage == current_stage`
- Clear mismatch logging with specific reasons

---

## ⚠️ Production Concerns (Priority Assessment)

### 🔴 CRITICAL (Fix Before Production)

#### 1. Atomic Writes + Crash Consistency
**Status:** NOT IMPLEMENTED
**Risk:** Process crash mid-write → corrupted cohorts, half-updated metadata
**Current:** Direct writes to `metadata.json`, `metrics.json`, `snapshot_index.json`
**Fix:**
```python
# Write to temp file, then atomic rename
temp_file = metadata_file.with_suffix('.tmp')
with open(temp_file, 'w') as f:
    json.dump(full_metadata, f, indent=2)
    f.flush()
    os.fsync(f.fileno())
os.replace(temp_file, metadata_file)  # Atomic on POSIX
```

**Priority:** P0 - Can cause data corruption

#### 2. Concurrency / Race Conditions
**Status:** NOT IMPLEMENTED
**Risk:** Two runs finalizing same cohort → corrupted `snapshot_index.json`, lost updates
**Current:** No file locking around index updates
**Fix:**
```python
import fcntl
with open(index_file, 'r+') as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
    # Read, update, write
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

**Priority:** P0 - Can cause data corruption in parallel runs

#### 3. "Prev Comparable Run" Selection Correctness
**Status:** ⚠️ POTENTIALLY UNSAFE
**Risk:** Timestamp-based ordering can be wrong (clock skew, resumed runs)
**Current:** `candidates.sort(key=lambda x: x[0], reverse=True)` uses `snap.timestamp`
**Fix:**
- Prefer internal monotonic sequence number (`snapshot_seq`)
- Or use `run_start_ns` captured once at run start
- Or use file modification time as tiebreaker

**Priority:** P1 - Can cause wrong comparisons

---

### 🟡 HIGH (Fix Soon)

#### 4. Idempotency + Reruns
**Status:** PARTIALLY HANDLED
**Risk:** Retry with same `run_id` → compare against self, duplicate index entries
**Current:** `run_id` is unique per run, but no deduplication in index
**Fix:**
- Ensure `run_id` includes attempt number or timestamp
- Dedupe index by `(run_id, stage)` when updating
- Or use stable snapshot ID for deduplication

**Priority:** P1 - Can cause confusion in retries

#### 5. Stage-Scoping Assertion
**Status:** NOT IMPLEMENTED
**Risk:** Cross-stage metadata contamination
**Fix:**
```python
# In finalize_run(), before using resolved_metadata:
if resolved_metadata:
    assert resolved_metadata.get("stage") == stage, \
        f"Stage mismatch: resolved_metadata stage={resolved_metadata.get('stage')}, current={stage}"
    assert resolved_metadata.get("run_id") == run_data.get('run_id'), \
        f"Run ID mismatch: resolved_metadata run_id={resolved_metadata.get('run_id')}, current={run_data.get('run_id')}"
```

**Priority:** P1 - Prevents cross-stage bugs

#### 6. Required Fields Assertion
**Status:** PARTIALLY HANDLED
**Risk:** Incomplete SST if `full_metadata` built before all fields finalized
**Current:** `_validate_stage_schema()` checks required fields, but only after context is built
**Fix:**
- Add assertion in `finalize_run()` that required fields are present in `resolved_metadata`
- Or document that `full_metadata` must be built AFTER all required fields are finalized

**Priority:** P1 - Prevents incomplete snapshots

---

### 🟢 MEDIUM (Nice to Have)

#### 7. High-Cardinality Drift and Storage Bloat
**Status:** ACCEPTABLE FOR NOW
**Risk:** `excluded_factors.changes` can be huge (full hyperparam dicts)
**Current:** Full payload stored in `metadata.json` (SST)
**Fix Options:**
- Cap stored changes (top N + digest)
- Compress metadata artifacts (gzip)
- Store full changes in separate file, reference in metadata

**Priority:** P2 - Monitor storage growth

#### 8. Schema Evolution Beyond Fingerprints
**Status:** PARTIALLY HANDLED
**Current:** `fingerprint_schema_version` exists
**Missing:** `snapshot_schema_version`, `diff_telemetry_schema_version`
**Fix:**
- Add top-level `artifact_schema_version` to all artifacts
- Or separate versions per artifact type

**Priority:** P2 - Future-proofing

#### 9. Canonicalization Pitfalls
**Status:** MOSTLY HANDLED
**Current:** Float normalization (`repr()`), list ordering, NaN/inf handling
**Potential Issues:**
- Timezone normalization of dates (not checked)
- String normalization (e.g., "CROSS_SECTIONAL" vs "cross_sectional") - handled by uppercase normalization
- NaN in int fields (not checked)

**Priority:** P2 - Edge cases

#### 10. Consumer Ergonomics
**Status:** PARTIALLY HANDLED
**Current:** `prev_run_id` in diff telemetry
**Missing:** `prev_snapshot_path`, `LATEST.json` per cohort/stage
**Fix:**
- Add `prev_snapshot_path` to diff telemetry
- Write `LATEST.json` symlink or file pointing to latest snapshot

**Priority:** P3 - UX improvement

---

### 🔵 LOW (Future)

#### 11. Secret / Sensitive Leakage
**Status:** NOT ADDRESSED
**Risk:** Usernames, hostnames, repo paths, dataset locations in metadata
**Fix:** Explicit redaction layer (allowlist keys)

**Priority:** P3 - Security hardening

#### 12. Cross-Stage Coupling via Shared Context
**Status:** HANDLED
**Current:** `ResolvedRunContext` is stage-agnostic, but `_build_comparison_group_from_context()` filters stage-irrelevant fields
**Status:** ✅ Safe - stage-specific filtering prevents contamination

---

## ✅ Implementation Status: ALL CRITICAL & HIGH-PRIORITY FIXES COMPLETED + ROBUSTNESS HARDENING

### Phase 1: Critical Fixes (Before Production) - ✅ COMPLETED
1. ✅ **COMPLETED** - Add atomic writes (`os.replace()` pattern)
   - All JSON writes now use `_write_atomic_json()` helper
   - Implemented in: `reproducibility_tracker.py`, `diff_telemetry.py`, `metrics.py`
   - Files: `metadata.json`, `metrics.json`, `snapshot.json`, `diff_prev.json`, `diff_baseline.json`, `snapshot_index.json`

2. ✅ **COMPLETED** - Add file locking for index updates (`fcntl.flock`)
   - Exclusive lock (`LOCK_EX`) prevents race conditions
   - Re-reads index after acquiring lock (handles concurrent updates)
   - Deduplicates by `(run_id, phase)` for idempotency
   - Lock automatically released on completion or error

3. ✅ **COMPLETED** - Fix "prev run" selection (use file mtime)
   - Uses `st_mtime` (file modification time) instead of timestamp
   - Handles clock skew, resumed runs, and timestamp inconsistencies
   - Falls back to timestamp if snapshot file not found

### Phase 2: High-Priority Hardening - ✅ COMPLETED
4. ✅ **COMPLETED** - Add stage/run_id assertions in `finalize_run()`
   - Validates `resolved_metadata["stage"] == stage`
   - Validates `resolved_metadata["run_id"] == current_run_id`
   - Fail-fast with clear error messages

5. ✅ **COMPLETED** - Add required fields assertion (validates before snapshot computation)
   - New method: `_get_required_fields_for_stage()` returns stage-specific required fields
   - Validates in `finalize_run()` BEFORE snapshot computation
   - Catches incomplete SST early (missing or null required fields)
   - Stage-specific requirements:
     - TARGET_RANKING: 11 required fields
     - FEATURE_SELECTION: 11 required fields
     - TRAINING: 12 required fields (includes model_family)

6. ✅ **COMPLETED** - Ensure idempotency (dedupe index entries by run_id+stage, atomic writes)

### Phase 3: Robustness Hardening (Production Multi-Run Safety) - ✅ COMPLETED
7. ✅ **COMPLETED** - Replace mtime-based prev run selection with monotonic sequence (`snapshot_seq`)
   - Added `snapshot_seq` field to `NormalizedSnapshot`
   - Assigned at save time (max existing + 1)
   - Used for "prev run" selection instead of mtime/timestamp
   - Handles file copies, post-processing, filesystem quirks, coarse timestamp resolution
   - Backwards compatible: old snapshots fall back to timestamp ordering

8. ✅ **COMPLETED** - Fix dedupe keys to include stage (`run_id:stage` format)
   - Snapshot index now uses `"run_id:stage"` as key (e.g., `"run123:TARGET_RANKING"`)
   - Prevents cross-stage overwrites when same run_id produces multiple stages
   - Backwards compatible: handles old format (`run_id` key) during load

9. ✅ **COMPLETED** - Add full durability (fsync tempfile + directory)
   - `fsync(tempfile)` before rename
   - `os.replace()` atomic rename
   - `fsync(directory)` after rename
   - Power-loss safe for audit-ready systems

10. ✅ **COMPLETED** - Document flock portability requirements
    - Created `PRODUCTION_READINESS.md` with filesystem compatibility notes
    - Documents NFS/shared storage limitations
    - Provides recommendations for network filesystem scenarios
   - Index updates dedupe by `(run_id, phase)` - keeps latest entry
   - Snapshot index dedupe by `run_id` - overwrites existing entries
   - Atomic writes prevent partial updates on crash
   - File locking prevents concurrent write corruption

### Phase 3: Medium-Priority Improvements
7. Monitor storage growth, add compression if needed
8. Add schema versioning for all artifacts
9. Add consumer ergonomics (`LATEST.json`, `prev_snapshot_path`)

### Phase 4: Future Enhancements
10. Add redaction layer for sensitive data
11. Enhanced canonicalization (timezone, NaN in ints)

---

## Implementation Notes

### Atomic Writes Pattern
```python
def write_atomic_json(file_path: Path, data: Dict[str, Any]) -> None:
    """Write JSON file atomically using temp file + rename."""
    temp_file = file_path.with_suffix('.tmp')
    try:
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, file_path)  # Atomic on POSIX
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()  # Cleanup on failure
        raise
```

### File Locking Pattern
```python
import fcntl

def update_index_with_lock(index_file: Path, update_fn: Callable) -> None:
    """Update index file with exclusive lock."""
    index_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file if it doesn't exist
    if not index_file.exists():
        with open(index_file, 'w') as f:
            json.dump({}, f)
    
    with open(index_file, 'r+') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        try:
            data = json.load(f)
            update_fn(data)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

---

## Conclusion

**Current Status:** Core architecture is correct (SST consistency fixed), but production hardening is needed.

**Immediate Next Steps:**
1. Add atomic writes (P0)
2. Add file locking (P0)
3. Add stage/run_id assertions (P1)
4. Fix "prev run" selection (P1)

**Timeline:** Critical fixes should be done before production use. High-priority hardening should be done within 1-2 sprints.

