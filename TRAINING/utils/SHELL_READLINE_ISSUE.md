# Shell/Readline Library Issue

**Date**: 2025-12-13  
**Status**: Environment issue (not code)

## Problem

Logs show:
```
sh: symbol lookup error: sh: undefined symbol: rl_print_keybinding
```

This is a **libreadline / environment mismatch** issue, not related to ML logic.

## Root Cause

Common causes:
- `LD_LIBRARY_PATH` poisoning (conda/venv tooling setting it incorrectly)
- PATH shadowing (wrong `sh` binary in PATH)
- Cursor AppImage's `LD_LIBRARY_PATH` interfering with system libraries

## Diagnosis

Run these commands to diagnose:

```bash
which sh
# Should be: /usr/bin/sh or /bin/sh

echo $LD_LIBRARY_PATH
# If set, that's suspect (especially if it points to conda/venv libs)
```

## Fix Options

### Option 1: Clean Shell (Recommended)

Run training from a clean shell with `LD_LIBRARY_PATH` unset:

```bash
unset LD_LIBRARY_PATH
# Then run your training command
```

### Option 2: Fix PATH

If `which sh` shows a non-standard path, fix PATH:

```bash
export PATH="/usr/bin:/bin:$PATH"
```

### Option 3: Fix LD_LIBRARY_PATH

If `LD_LIBRARY_PATH` is set incorrectly, unset or fix it:

```bash
unset LD_LIBRARY_PATH
# Or if you need it, ensure it doesn't shadow system libs
```

## Impact

This is a **cosmetic issue** - it doesn't affect ML logic or training results. However, it can:
- Cause random side effects
- Make runs feel "haunted"
- Potentially interfere with subprocess calls

## Prevention

For automated runs, ensure clean environment:

```bash
#!/bin/bash
# Clean environment for training
unset LD_LIBRARY_PATH
export PATH="/usr/bin:/bin:$PATH"
# Run training
python -m TRAINING.orchestration.intelligent_trainer ...
```
