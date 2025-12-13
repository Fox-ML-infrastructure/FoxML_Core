# Quick Check: Undefined Names

## One Command to Rule Them All

```bash
ruff check TRAINING --select F821
```

That's it. This catches all "name is not defined" errors like:
- Missing imports (`pl`, `np`, `pd`, `logger`)
- Undefined variables
- Typos in variable names

## Usage

```bash
# From repo root - check TRAINING directory
ruff check TRAINING --select F821

# Check entire repo
ruff check . --select F821

# Auto-fix what can be fixed (use with caution)
ruff check TRAINING --select F821 --fix
```

## Helper Script

```bash
# Use the wrapper script
python TRAINING/tools/check_undefined_names.py
python TRAINING/tools/check_undefined_names.py --all   # entire repo
python TRAINING/tools/check_undefined_names.py --fix   # auto-fix
```

## What You'll See

```
F821 Undefined name `pl`
  --> TRAINING/training_strategies/data_preparation.py:261:12
   |
260 |         df_use = df
261 |     df_pl = pl.from_pandas(df_use)
   |            ^^
```

## Note

Some F821 errors may be false positives (variables defined conditionally, runtime globals, etc.), but most are real issues that need fixing.
