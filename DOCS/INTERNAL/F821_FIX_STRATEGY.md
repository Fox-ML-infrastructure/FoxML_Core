# F821 (Undefined Name) Fix Strategy

**Current Status:** 2,301 F821 errors across TRAINING directory  
**Baseline:** This is normal for an organically-grown codebase. We're systematically fixing them.

## Analysis Results (2025-12-10)

### Top Files by Error Count

Using `ruff check TRAINING --select F821 --no-cache --output-format=json`:

```
  93 TRAINING/training_strategies/training.py          ← START HERE
  51 TRAINING/training_strategies/main.py
  17 TRAINING/ranking/predictability/leakage_detection.py
  10 TRAINING/data_processing/data_loader.py          ← Already partially fixed
  10 TRAINING/utils/core_utils.py
   4 TRAINING/training_strategies/strategies.py
   3 TRAINING/EXPERIMENTS/phase1_feature_engineering/run_phase1.py
   3 TRAINING/data_processing/data_utils.py
   3 TRAINING/utils/target_utils.py
   2 TRAINING/ranking/target_ranker.py
   1 TRAINING/common/determinism.py
   1 TRAINING/orchestration/intelligent_trainer.py
   1 TRAINING/orchestration/target_routing.py
   ... (and more)
```

**Total: ~200 errors in core production paths** (out of 2,301 total)

### Analysis Commands

```bash
# Top files with most undefined names (JSON format - cleanest)
ruff check TRAINING --select F821 --no-cache --output-format=json 2>&1 | \
  python3 -c "import sys, json; data = json.load(sys.stdin); files = {}; \
  [files.setdefault(e['filename'], []).append(e) for e in data if e.get('code') == 'F821']; \
  [(print(f\"{len(errors):4d} {f}\")) for f, errors in sorted(files.items(), key=lambda x: -len(x[1]))[:20]]"

# Alternative: concise format (has ANSI codes but works)
ruff check TRAINING --select F821 --no-cache --output-format=concise 2>&1 | \
  grep "F821" | cut -d: -f1 | sort | uniq -c | sort -rn | head -20
```

### Check core production paths

```bash
# Just the stuff that actually runs
ruff check TRAINING/orchestration TRAINING/training_strategies TRAINING/utils \
  --select F821 --no-cache
```

### Track progress

```bash
# Count total errors (watch it trend down)
ruff check TRAINING --select F821 --no-cache 2>&1 | wc -l
```

## Fix Strategy: Layers

### 1. Core Production Paths First

**Priority order (by error count):**

1. **`TRAINING/training_strategies/training.py`** (93 errors) - Core training pipeline
2. **`TRAINING/training_strategies/main.py`** (51 errors) - Main entry point
3. **`TRAINING/ranking/predictability/leakage_detection.py`** (17 errors) - Leakage detection
4. **`TRAINING/data_processing/data_loader.py`** (10 errors) - Data loading (partially fixed)
5. **`TRAINING/utils/core_utils.py`** (10 errors) - Core utilities
6. **`TRAINING/training_strategies/strategies.py`** (4 errors) - Training strategies
7. **`TRAINING/orchestration/`** (2 errors) - Orchestration layer
8. **`TRAINING/utils/`** (remaining) - Other utilities

**Goal:** Get these core files to 0 errors first (~200 errors total in production paths)

### 2. Fix by Pattern, Not One-Off

Common patterns you'll see:

#### Missing imports
- `pl.` → add `import polars as pl`
- `np.` → add `import numpy as np`
- `pd.` → add `import pandas as pd`
- `logger.info(...)` → add:
  ```python
  import logging
  logger = logging.getLogger(__name__)
  ```

#### Copy-paste leftovers
- Variables used but never defined in that scope (`results`, `df`, `X`, `y`, etc.)
- Usually means code was moved/copied and variable definitions didn't come along

#### Environment-specific false positives
- Some imports exist in `trader_env` but not in current environment
- These are **not real errors** - they work at runtime
- Can be ignored or documented

### 3. Exclude Legacy/Experimental Code

Once core is clean, exclude non-production code:

Create `ruff.toml`:
```toml
[tool.ruff]
exclude = [
  "legacy/",
  "SCRIPTS/",
  "EXPERIMENTS/",  # if experimental
  "archive/",
]
```

### 4. Track Progress

After each fix pass:
```bash
ruff check TRAINING/orchestration TRAINING/training_strategies TRAINING/utils \
  --select F821 --no-cache 2>&1 | wc -l
```

Goal: Get core paths to 0, then decide if rest is worth cleaning.

## False Positives: Environment Differences

**Important:** Some F821 errors are false positives because:
- Imports exist in `trader_env` (the correct runtime environment)
- But not in the current environment where ruff is running
- These are **not real errors** - code works at runtime

**How to handle:**
1. Run ruff from `trader_env` when possible
2. Document known false positives
3. Focus on real errors first (missing imports that would fail at runtime)

## Quick Fix Workflow

1. **Pick a file with cluster of issues:**
   ```bash
   ruff check TRAINING/training_strategies/data_preparation.py --select F821
   ```

2. **Fix all of that pattern at once:**
   - Open file
   - Add missing imports at top
   - Fix variable scope issues
   - Save

3. **Verify:**
   ```bash
   ruff check TRAINING/training_strategies/data_preparation.py --select F821
   ```

4. **Move on to next file**

## Notes

- This is a **grown-up infra move**, not a red flag
- 2.3k errors on an organically-grown codebase is normal
- Nobody has run a real linter on this codebase before
- Fixing this systematically improves reliability across the whole stack
