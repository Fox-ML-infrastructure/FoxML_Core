# Demo Run with Baseline Management

This folder supports safe refactoring by establishing a reproducible demo run with baseline comparison.

## Quick Start

```bash
# Run demo (uses CONFIG/experiments/demo.yaml)
python demo/run_demo.py

# Save current output as golden baseline
python demo/run_demo.py --save-baseline

# Run and compare against baseline
python demo/run_demo.py --check-baseline

# Show baseline info
python demo/run_demo.py --show-baseline
```

## Workflow for Refactoring

1. **Before refactoring**: Create baseline
   ```bash
   python demo/run_demo.py --save-baseline
   ```

2. **After each change**: Check against baseline
   ```bash
   python demo/run_demo.py --check-baseline
   ```

3. **If behavior change is intentional**: Update baseline
   ```bash
   python demo/run_demo.py --save-baseline
   ```

## What Gets Compared

- `manifest.json`: Config fingerprints, versions, inputs
- `globals/run_context.json`: Resolved view, scope, symbols
- Directory structure (file list)

Fields skipped during comparison:
- Timestamps (`created_at`, `timestamp`)
- Runtime (`elapsed_seconds`)
- Git commit (may change between runs)

## Baseline Location

Baseline files are stored in `demo/baseline/`:
- `manifest.json`
- `globals/run_context.json`
- `structure.json` (list of all output files)
- `metadata.json` (when baseline was created)
