# Auto-Fixer Backup Fix

**Date:** 2025-12-10  
**Issue:** Auto-fixer was not creating backups when no leaks were detected

## Problem

When auto-fix mode was triggered (e.g., 100% training accuracy detected), but `detect_leaking_features()` returned an empty list (no leaks found), the code in `model_evaluation.py` would skip calling `apply_fixes()`. Since `apply_fixes()` is responsible for creating backups, no backup was created.

**Expected behavior:** Backups should be created whenever auto-fix mode is triggered, regardless of whether leaks are detected (to preserve state history for debugging).

## Root Cause

In `TRAINING/ranking/predictability/model_evaluation.py` (line 2576), the code had:

```python
if detections:
    # Call apply_fixes() which creates backups
    updates, autofix_info = fixer.apply_fixes(...)
else:
    # No backup created here!
    logger.info("🔍 Auto-fix detected no leaks (may need manual review)")
```

The `apply_fixes()` method in `leakage_auto_fixer.py` already creates backups even when no leaks are detected (lines 558-572), but it was never called when `detections` was empty.

## Fix

Modified `model_evaluation.py` to explicitly call `_backup_configs()` when no leaks are detected:

```python
else:
    logger.info("🔍 Auto-fix detected no leaks (may need manual review)")
    # Still create backup even when no leaks detected (to preserve state history)
    # This ensures we have a backup whenever auto-fix mode is triggered
    try:
        backup_files = fixer._backup_configs(
            target_name=target_name,
            max_backups_per_target=None  # Use instance config
        )
        if backup_files:
            logger.info(f"📦 Backup created (no leaks detected): {len(backup_files)} backup file(s)")
    except Exception as backup_error:
        logger.warning(f"Failed to create backup when no leaks detected: {backup_error}")
```

## Verification

- ✅ Backups are now created whenever auto-fix mode is triggered
- ✅ Backups are created even when no leaks are detected
- ✅ Backup location: `CONFIG/backups/{target_name}/{timestamp}/`
- ✅ Each backup includes manifest.json with git commit, timestamp, and file paths

## Related Code

- `TRAINING/ranking/predictability/model_evaluation.py` (lines 2576-2607)
- `TRAINING/common/leakage_auto_fixer.py` (lines 558-584, 697-796)

## Notes

- `run_auto_fix_loop()` in `leakage_auto_fixer.py` has a similar pattern (line 978), but this is correct behavior - it's an iterative loop that exits when no leaks are detected, which is success. No backup needed in that context.
- The fix ensures state history is preserved for debugging when auto-fix mode triggers but finds no leaks (which might indicate a false positive trigger or overfitting).
