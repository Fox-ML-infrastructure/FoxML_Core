# Readline Library Fix Analysis

## Problem Summary

**Root Cause Identified:**
- Conda's `libreadline.so.8` in `trader_env` is **missing** the `rl_print_keybinding` symbol
- System's `/usr/lib/libreadline.so.8` **has** the symbol
- `sqlite3` in the conda environment links to Conda's broken libreadline via RPATH `$ORIGIN/../lib`
- When any subprocess calls `sqlite3` (or other tools that might link to Conda's readline), it fails with:
  ```
  sh: symbol lookup error: sh: undefined symbol: rl_print_keybinding
  ```

## Verification Results

```bash
# Conda's libreadline: MISSING symbol
nm -D $CONDA_PREFIX/lib/libreadline.so.8 | grep rl_print_keybinding
# Result: Symbol NOT found

# System's libreadline: HAS symbol  
nm -D /usr/lib/libreadline.so.8 | grep rl_print_keybinding
# Result: 0000000000027150 T rl_print_keybinding

# Binaries linking to Conda's readline
find $CONDA_PREFIX/bin -type f -executable | while read bin; do
  ldd "$bin" 2>/dev/null | grep -q "libreadline.*trader_env" && echo "$bin"
done
# Result: /home/Jennifer/miniconda3/envs/trader_env/bin/sqlite3
```

## Solutions (Choose One)

### Solution 1: Replace Conda's libreadline with System Version (Recommended)

**Quick fix:** Symlink Conda's libreadline to system's version:

```bash
conda activate trader_env
cd $CONDA_PREFIX/lib

# Backup Conda's version
mv libreadline.so.8 libreadline.so.8.conda-backup
mv libreadline.so libreadline.so.conda-backup

# Create symlinks to system version
ln -s /usr/lib/libreadline.so.8 libreadline.so.8
ln -s /usr/lib/libreadline.so.8 libreadline.so
```

**Pros:** Immediate fix, no code changes needed
**Cons:** May break if Conda packages expect specific readline version

### Solution 2: Update Conda's readline Package

```bash
conda activate trader_env
conda update -c conda-forge readline
# OR
conda install -c conda-forge readline=8.2
```

**Pros:** Proper fix using Conda's package manager
**Cons:** May not be available in all Conda channels

### Solution 3: Use System sqlite3 Instead

```bash
conda activate trader_env
# Remove conda's sqlite3 from PATH priority
# Or create wrapper that uses system sqlite3
```

**Pros:** Avoids the broken binary entirely
**Cons:** May miss Conda-specific sqlite3 features

### Solution 4: Ensure All Subprocess Calls Use Safe Wrapper (Already Partially Implemented)

The codebase already has `TRAINING/common/subprocess_utils.py` with `safe_subprocess_run()` that:
- Sets `TERM=dumb` to disable readline
- Sets `SHELL=/usr/bin/bash`
- Filters `LD_LIBRARY_PATH` to remove AppImage mounts

**Action Required:** Audit codebase to ensure ALL subprocess calls use `safe_subprocess_run()` instead of `subprocess.run()`.

**Check for direct subprocess calls:**
```bash
grep -r "subprocess\.run\|subprocess\.Popen\|subprocess\.call" --include="*.py" | grep -v "subprocess_utils.py" | grep -v "safe_subprocess"
```

## Recommended Immediate Action

1. **Apply Solution 1** (symlink fix) for immediate relief
2. **Audit subprocess usage** to ensure `safe_subprocess_run()` is used everywhere
3. **Long-term:** Update Conda's readline package (Solution 2) when available

## Testing

After applying fix, verify:

```bash
conda activate trader_env
python -c "
import subprocess
result = subprocess.run(['sqlite3', '--version'], capture_output=True, text=True)
print('sqlite3 version:', result.stdout.strip() if result.returncode == 0 else 'FAILED')
"
```

## Related Files

- `TRAINING/common/subprocess_utils.py` - Safe subprocess wrapper (already exists)
- `TRAINING/common/isolation_runner.py` - Sets readline workarounds at import time
- `TRAINING/model_fun/xgboost_trainer.py` - Uses `safe_subprocess_run()` for nvidia-smi
