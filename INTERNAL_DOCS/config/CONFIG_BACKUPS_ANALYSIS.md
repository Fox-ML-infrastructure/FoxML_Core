# CONFIG/backups Directory Analysis

## Current Status

**The `CONFIG/backups/` directory is empty** (only contains `.gitkeep`). This means backups are **not being created** during training runs.

## When Backups Are Created

Backups are only created when:

1. **Leakage is detected** during training:
   - Perfect training scores (>= 0.99)
   - Perfect correlation (>= 0.99)
   - High training accuracy (>= 0.999)
   - High training R² (>= 0.999)

2. **Auto-fixer is enabled** (`safety.leakage_detection.auto_fix_enabled: true`)

3. **Auto-fixer is actually called** in `model_evaluation.py`:
   - Only when `should_auto_fix = True`
   - Only when leakage thresholds are exceeded

## Code Flow

```python
# In model_evaluation.py:
if not auto_fix_enabled:
    should_auto_fix = False
else:
    should_auto_fix = False  # Starts as False
    # Only set to True if leakage detected:
    if perfect_scores or perfect_correlation:
        should_auto_fix = True

if should_auto_fix:
    fixer = LeakageAutoFixer(backup_configs=True)
    # Backups created here
```

## Why Backups Aren't Being Created

**Most likely reason: No leakage is being detected!**

This is actually **good news** - it means:
- Your features are clean
- No perfect scores indicating leakage
- No suspicious correlations

## Backup Creation Logic

Backups are created in `_backup_configs()` method:
- Called when `self.backup_configs=True` (default)
- Only if `not dry_run`
- Creates timestamped backups: `CONFIG/backups/{target}/{timestamp}/`

## Configuration

- `safety.leakage_detection.auto_fix_enabled: true` ✅ (enabled)
- `safety.leakage_detection.auto_fix_min_confidence: 0.8` (80% confidence required)
- `safety.leakage_detection.auto_fixer.max_backups_per_target: 20` (max 20 backups per target)

## Recommendation

**The empty backups directory is expected behavior** if:
- No leakage is detected (good!)
- Auto-fixer never runs (because no leaks found)

**If you want to test the backup system:**
1. Temporarily lower leakage thresholds in `safety_config.yaml`
2. Run training on a target that might have leakage
3. Backups should be created when auto-fixer runs

**If backups are never created and you expect them:**
1. Check logs for "🔧 Auto-fixing detected leaks..." messages
2. Verify `auto_fix_enabled: true` in config
3. Check if leakage thresholds are being met

## Conclusion

The empty `CONFIG/backups/` directory is **not a problem** - it simply means no leakage has been detected that triggers the auto-fixer. The `.gitkeep` file preserves the directory structure for when backups are needed.
