# Production Readiness: File Locking & Filesystem Requirements

## File Locking (`fcntl.flock`)

### Current Implementation

The system uses `fcntl.flock()` with `LOCK_EX` (exclusive lock) for index updates to prevent race conditions when multiple runs update the index simultaneously.

### Filesystem Compatibility

**✅ Reliable on:**
- Local Linux filesystems (ext4, xfs, btrfs, zfs)
- Local filesystems on macOS/BSD (with advisory locks)
- Local filesystems on Windows (with proper locking support)

**⚠️ May be unreliable on:**
- Network filesystems (NFS, CIFS/SMB, GlusterFS)
- Some distributed filesystems
- Filesystems with coarse-grained locking

### Behavior

- **Advisory locks**: All writers must cooperate (use `flock`)
- **Blocking**: `LOCK_EX` blocks until the lock is available
- **Automatic release**: Lock is released when file is closed or process exits

### Recommendations

1. **For local filesystems**: Current implementation is sufficient
2. **For NFS/shared storage**: 
   - Consider append-only journal pattern + periodic compaction
   - Or use a database-backed index (PostgreSQL, SQLite)
   - Or ensure all writers are on the same machine (NFS client)

### Future Improvements

If NFS support is required:
- Implement append-only journal per cohort
- Periodic compaction to index (still under lock)
- Or migrate to database-backed index

---

## Atomic Writes & Durability

### Current Implementation

All JSON writes use `_write_atomic_json()` which:
1. Writes to temp file (`.tmp` extension)
2. `fsync(tempfile)` - ensures data is on disk
3. `os.replace()` - atomic rename
4. `fsync(directory)` - ensures directory entry is on disk

### Power-Loss Safety

This pattern provides **power-loss safety** for audit-ready systems:
- If process dies before `os.replace()`: original file intact
- If process dies after `os.replace()` but before directory sync: file exists but may not be visible (rare)
- If power loss after directory sync: file is fully durable

### Filesystem Requirements

- **POSIX**: `os.replace()` is atomic
- **Windows**: Best-effort atomic (usually works)
- **Network filesystems**: May have reduced guarantees (NFS can have delayed writes)

---

## Monotonic Sequence Numbers

### Implementation

- `snapshot_seq` field added to `NormalizedSnapshot`
- Assigned at save time (max existing + 1)
- Used for "prev run" selection instead of mtime/timestamp

### Benefits

- **Correct ordering**: Not affected by file copies, post-processing, filesystem quirks
- **Coarse timestamp resolution**: Handles filesystems with 1s timestamp resolution
- **Clock skew**: Immune to clock adjustments

### Backwards Compatibility

- Old snapshots without `snapshot_seq` fall back to timestamp ordering
- Logs warning when fallback is used

---

## Deduplication Keys

### Index Deduplication

**Global index (`index.parquet`):**
- Dedupe by: `(run_id, phase)` where `phase` = stage
- Preserves multiple stages per run_id

**Snapshot index (`snapshot_index.json`):**
- Key format: `"run_id:stage"` (e.g., `"run123:TARGET_RANKING"`)
- Prevents cross-stage overwrites
- Backwards compatible with old format (`run_id` key)

### Why This Matters

A single `run_id` can produce multiple stages:
- `TARGET_RANKING` → `FEATURE_SELECTION` → `TRAINING`

Deduping by just `run_id` would cause later stages to overwrite earlier stages.

---

## Summary

✅ **Production-ready for:**
- Local filesystems (Linux, macOS, Windows)
- Single-machine deployments
- Audit-ready durability requirements

⚠️ **Consider alternatives for:**
- Network filesystems (NFS, CIFS)
- Multi-machine concurrent writes to shared storage
- Very high concurrency (100+ simultaneous writers)

🔧 **Future improvements:**
- Database-backed index for NFS scenarios
- Append-only journal pattern for high concurrency
- Distributed locking (Redis, etcd) for multi-machine scenarios

