# Checking for Undefined Names (Missing Imports)

Quick way to catch all "name is not defined" / missing-import style bugs.

## Quick Command

```bash
# From repo root
ruff check TRAINING --select F821
```

This will show all undefined names (like `pl`, `np`, `pd`, `logger`, etc.) that are used but not imported/defined.

## Using the Helper Script

```bash
# Check TRAINING directory only
python TRAINING/tools/check_undefined_names.py

# Check entire repo
python TRAINING/tools/check_undefined_names.py --all

# Auto-fix what can be fixed (ruff --fix)
python TRAINING/tools/check_undefined_names.py --fix
```

## What It Catches

- Missing imports: `pl`, `np`, `pd`, `logger`, etc.
- Undefined variables used before assignment
- Typos in variable names

## Example Output

```
F821 Undefined name `pl`
  --> TRAINING/training_strategies/data_preparation.py:261:12
   |
260 |         df_use = df
261 |     df_pl = pl.from_pandas(df_use)
   |            ^^
```

## Integration

Add to your workflow:

```bash
# In Makefile or CI
lint-undefined:
	ruff check TRAINING --select F821
```

Or run before commits:

```bash
ruff check TRAINING --select F821 && git commit ...
```
