# Config Analysis Tools

## validate_config_paths.py

Validates that all config paths have been migrated from hardcoded paths to the centralized config loader API.

### Purpose

This tool helps ensure the migration from hardcoded `Path("CONFIG/...")` patterns to the config loader API is complete. It scans the TRAINING directory for remaining hardcoded paths and validates that all config files are accessible via the loader.

### Usage

```bash
# Run validation
python CONFIG/tools/validate_config_paths.py
```

### What It Checks

1. **Hardcoded paths**: Scans for `Path("CONFIG/...")` patterns in Python files
2. **Config loader access**: Validates that config files are accessible via the loader API
3. **Symlink validity**: Checks that all symlinks are valid and point to existing files

### Output

The script reports:
- Files with hardcoded paths (if any)
- Config loader API access status
- Symlink validity

### Example Output

```
✅ No hardcoded config paths found!
✅ All config files accessible via config loader
✅ All 4 symlinks are valid
```

### Integration

This tool can be run as part of CI/CD to ensure new code uses the config loader API instead of hardcoded paths.

---

## find_repeated_defaults.py

Scans all YAML config files in `CONFIG/` and identifies repeated settings that are good candidates for centralization into `CONFIG/defaults.yaml`.

### Purpose

This tool helps maintain **Single Source of Truth (SST)** by finding configuration values that are duplicated across multiple files. These can then be moved to `defaults.yaml` and automatically injected via the config loader.

### Usage

```bash
# Basic scan (default: min 3 occurrences, 60% coverage)
python CONFIG/tools/find_repeated_defaults.py

# Stricter thresholds (must appear in 5+ files and 70%+ coverage)
python CONFIG/tools/find_repeated_defaults.py --min-occurrences 5 --min-coverage 0.7

# Group results by model family (tree models, neural networks, etc.)
python CONFIG/tools/find_repeated_defaults.py --group-by-family

# Scan only specific directories
python CONFIG/tools/find_repeated_defaults.py --include 'model_config/*.yaml' 'training_config/*.yaml'
```

### Parameters

- `--root CONFIG` - Root directory to scan (default: CONFIG)
- `--min-occurrences N` - Minimum number of configs that must share a value (default: 3)
- `--min-coverage FLOAT` - Minimum fraction (0.0-1.0) of configs that share the value (default: 0.6)
- `--include PATTERN` - Glob pattern(s) to restrict which files to scan
- `--group-by-family` - Group candidates by model family

### Interpreting Results

The script outputs candidates like:

```
hyperparameters.dropout
  value       : 0.2
  used in     : 7 / 8 configs (87.5%)
  files:
      - model_config/cnn1d.yaml
      - model_config/lstm.yaml
      ...
```

This means:
- **7 out of 8 configs** that use `hyperparameters.dropout` set it to `0.2`
- **87.5% coverage** - high confidence this should be a default
- The files listed show where this value appears

### Action Items

When you see a good candidate:

1. **Add to defaults.yaml** - Place it in the appropriate section:
   - `neural_networks.dropout: 0.2` (for neural network defaults)
   - `randomness.random_state: 42` (for global randomness)
   - etc.

2. **Remove from individual configs** (optional, gradual):
   - Delete the explicit value from model configs
   - The config loader will automatically inject the default

3. **Keep overrides** - If a config uses a different value (e.g., `dropout: 0.3`), keep it as an explicit override

### What Gets Skipped

The script automatically skips:
- Non-scalar values (lists, dicts)
- Identifiers (names, paths, symbols, tickers)
- Content-specific values (descriptions, notes)
- Files: `defaults.yaml`, `logging_config.yaml`, registries, etc.

### Example Workflow

```bash
# 1. Run the scan
python CONFIG/tools/find_repeated_defaults.py --min-occurrences 5

# 2. Review candidates (e.g., random_state appears in 15/16 files)

# 3. Add to defaults.yaml:
#    randomness:
#      random_state: 42

# 4. Remove from individual configs (gradually, as you test)

# 5. Re-run to verify reduction in duplicates
```

### Tips

- Start with **high coverage** (≥80%) candidates - these are safest to centralize
- Use `--group-by-family` to see if defaults should be per-family (tree vs neural)
- Lower `--min-occurrences` to find more candidates (but be more selective)
- Keep explicit values in configs that legitimately need different defaults
