# File Locking for JSON Writes

**Date**: 2026-01-08  
**Type**: Enhancement, Bug Fix  
**Impact**: High - Prevents race conditions and data corruption from concurrent writes

## Problem

Multiple processes/threads could write to the same JSON files (`metadata.json`, `snapshot.json`, `metrics.json`, diff files) concurrently, causing:
- Race conditions where one process overwrites another's write
- Timestamp serialization errors if data structures are modified between sanitization and write
- Data corruption or incomplete writes
- "Object of type Timestamp is not JSON serializable" errors from concurrent modifications

## Solution

Added file locking around all JSON writes using `fcntl.flock` with `LOCK_EX` for exclusive access. Created `_write_atomic_json_with_lock()` helper function that:
- Sanitizes data (converts Timestamps to ISO strings) before writing
- Acquires exclusive file lock before write
- Uses timeout protection (30s default) to prevent deadlocks
- Automatically releases lock on completion or error

## Implementation

### 1. Added Locked JSON Write Helper

**Files**: `reproducibility_tracker.py`, `diff_telemetry.py`

Created `_write_atomic_json_with_lock()` function that wraps `_write_atomic_json()` with file locking:

```python
def _write_atomic_json_with_lock(
    file_path: Path,
    data: Dict[str, Any],
    lock_timeout: float = 30.0
) -> None:
    """Write JSON file atomically with file locking to prevent race conditions."""
    # Sanitize data (convert Timestamps to ISO strings)
    sanitized_data = _sanitize_for_json(data)
    
    # Create lock file (same directory, .lock extension)
    lock_file = file_path.with_suffix('.lock')
    
    # Acquire exclusive lock with timeout
    with open(lock_file, 'w') as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        # ... (retry logic with timeout) ...
        
        # Lock acquired - perform write
        _write_atomic_json(file_path, sanitized_data)
```

### 2. Updated All Critical JSON Writes

**In `reproducibility_tracker.py`:**
- Line 2166: `metadata.json` write (full_metadata)
- Line 2378: `metrics.json` write
- Line 4435: `metadata.json` fallback write (minimal_metadata)

**In `diff_telemetry.py`:**
- Line 2785: `snapshot.json` write (cohort directory)
- Line 2879: `snapshot.json` write (target-first structure)
- Line 4707: `metric_deltas.json` write
- Line 4722: `diff_prev.json` write
- Line 4737: `diff_baseline.json` write

## Files Changed

- `TRAINING/orchestration/utils/reproducibility_tracker.py`
  - Added `_write_atomic_json_with_lock()` helper function
  - Updated 3 `_write_atomic_json` calls to use locked version

- `TRAINING/orchestration/utils/diff_telemetry.py`
  - Added `_write_atomic_json_with_lock()` helper function
  - Updated 5 `_write_atomic_json` calls to use locked version

- `TRAINING/training_strategies/reproducibility/io.py`
  - Added `_write_atomic_json_with_lock()` helper function
  - Updated `save_training_snapshot()` to use locked version for training snapshots

- `TRAINING/stability/feature_importance/io.py`
  - Added `_write_atomic_json_with_lock()` helper function
  - Updated `save_fs_snapshot()` to use locked version for feature selection snapshots

## Benefits

- **Prevents race conditions**: Only one process can write to a file at a time
- **Prevents data corruption**: No partial writes or overwrites
- **Prevents Timestamp errors**: Data is sanitized once under lock (TARGET_RANKING/FEATURE_SELECTION) or uses default=str (TRAINING), preventing modification between sanitization and write
- **Consistent with existing patterns**: Uses same locking approach as index files (index.parquet, snapshot sequence assignment)
- **Timeout protection**: Prevents deadlocks with configurable timeout (30s default)
- **Applied across all stages**: TARGET_RANKING, FEATURE_SELECTION, and TRAINING stages all use file locking for JSON writes

## Testing

- Verify files write successfully with locking
- Verify concurrent writes are serialized (no corruption)
- Verify timeout works if lock is held too long
- Verify lock files are cleaned up properly

## Backward Compatibility

- Lock files use `.lock` extension (hidden files, won't interfere with existing code)
- If locking fails, falls back to error (same as current behavior)
- No changes to file formats or data structures
- All existing JSON files remain compatible
