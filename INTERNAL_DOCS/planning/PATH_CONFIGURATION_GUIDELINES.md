# Path Configuration Guidelines

**Status:** Design Guidelines  
**Last Updated:** 2025-12-07

---

## Principle: Configurable User Paths, Hardcoded Internal Paths

Not all paths should be configurable. Follow these guidelines:

---

## ✅ **SHOULD Be Configurable** (Add to `system_config.yaml`)

### User-Facing Paths
- **Data directories**: Where users store their datasets
  - `data_dir`, `data_labeled`, `data_raw`
- **Output directories**: Where results/models are saved
  - `output_dir`, `results_dir`, `models_dir`
- **Config file locations**: User might want custom config locations
  - `config_dir`, `excluded_features`, `feature_registry`
- **Backup/temp directories**: User might want specific locations
  - `config_backup_dir`, `temp_dir`, `joblib_temp`, `cache_dir`
- **Log directories**: Where logs are written
  - `log_dir`, `log_file`

### Environment-Specific Paths
- Paths that differ between dev/staging/prod
- Paths that might be on different filesystems
- Paths that users deploy to custom locations

### Examples Already Configurable:
```yaml
system:
  paths:
    data_dir: "data/data_labeled/interval=5m"
    output_dir: null
    config_dir: "CONFIG"
    excluded_features: null  # Uses config_dir/excluded_features.yaml
    config_backup_dir: null  # Uses config_dir/backups/
```

---

## ❌ **SHOULD NOT Be Configurable** (Keep Hardcoded/Relative)

### Internal Module Paths
- Paths relative to codebase structure
- Python import paths
- Internal module directories

**Examples:**
```python
# ✅ GOOD - Relative to code structure
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _REPO_ROOT / "CONFIG"

# ❌ BAD - Don't make these configurable
# TRAINING/common/leakage_auto_fixer.py
# TRAINING/utils/leakage_filtering.py
# These are implementation details
```

### Relative Paths Within Codebase
- Paths that are part of the code structure
- Paths that should always be relative to the repo

**Examples:**
```python
# ✅ GOOD - Relative paths
script_file = Path(__file__).resolve()
repo_root = script_file.parents[2]
config_path = repo_root / "CONFIG" / "excluded_features.yaml"

# ❌ BAD - Don't make these configurable
# The relationship between TRAINING/utils/ and CONFIG/ is fixed
```

### Default Fallback Paths
- Paths used as fallbacks when config is unavailable
- Paths that define the "standard" structure

**Examples:**
```python
# ✅ GOOD - Default fallback
if config_path is None:
    config_path = _REPO_ROOT / "CONFIG" / "excluded_features.yaml"
```

---

## 🎯 **Decision Framework**

Ask these questions:

1. **Will users need to change this path?**
   - ✅ Yes → Make it configurable
   - ❌ No → Keep it hardcoded/relative

2. **Does this path vary by environment?**
   - ✅ Yes → Make it configurable
   - ❌ No → Keep it hardcoded/relative

3. **Is this an implementation detail?**
   - ✅ Yes → Keep it hardcoded/relative
   - ❌ No → Consider making it configurable

4. **Is this part of the codebase structure?**
   - ✅ Yes → Keep it hardcoded/relative
   - ❌ No → Consider making it configurable

---

## 📋 **Current Status**

### Already Configurable ✅
- Data directories (`data_dir`)
- Output directories (`output_dir`)
- Config file paths (`config_dir`, `excluded_features`, `feature_registry`, `feature_target_schema`)
- Backup directory (`config_backup_dir`)
- Temp directories (`temp_dir`, `joblib_temp`)
- Cache directory (`model_cache_dir`)

### Should Remain Hardcoded ✅
- Internal module paths (`TRAINING/`, `CONFIG/` relative to repo)
- Path resolution logic (`Path(__file__).resolve().parents[N]`)
- Default fallback paths (used when config unavailable)

---

## 🔧 **Implementation Pattern**

When making a path configurable:

1. **Add to `system_config.yaml`:**
```yaml
system:
  paths:
    my_custom_path: null  # null = use default, or set custom path
```

2. **Update code with fallback:**
```python
# Try config first
if _CONFIG_AVAILABLE:
    try:
        system_cfg = get_system_config()
        custom_path = system_cfg.get('system', {}).get('paths', {}).get('my_custom_path')
        if custom_path:
            path = Path(custom_path)
            if not path.is_absolute():
                path = _REPO_ROOT / custom_path
        else:
            # Use default
            path = _REPO_ROOT / "default" / "path"
    except Exception:
        # Fallback to default
        path = _REPO_ROOT / "default" / "path"
else:
    # Fallback to default
    path = _REPO_ROOT / "default" / "path"
```

3. **Document the default:**
- Add comment explaining what the default is
- Document in config file what `null` means

---

## ⚠️ **Anti-Patterns to Avoid**

### Don't Make Everything Configurable
```yaml
# ❌ BAD - Over-configuration
system:
  paths:
    training_module: "TRAINING"  # This is part of code structure!
    common_module: "TRAINING/common"  # Implementation detail!
    utils_module: "TRAINING/utils"  # Should never change!
```

### Don't Remove Fallbacks
```python
# ❌ BAD - No fallback
path = Path(config_path)  # What if config is missing?

# ✅ GOOD - Always have fallback
path = Path(config_path) if config_path else _REPO_ROOT / "default"
```

### Don't Make Relative Paths Absolute-Only
```python
# ❌ BAD - Only supports absolute
path = Path(config_path)  # Must be absolute

# ✅ GOOD - Supports both
path = Path(config_path)
if not path.is_absolute():
    path = _REPO_ROOT / config_path
```

---

## 📊 **Summary**

| Path Type | Configurable? | Example |
|-----------|--------------|---------|
| Data directories | ✅ Yes | `data_dir`, `data_labeled` |
| Output directories | ✅ Yes | `output_dir`, `results_dir` |
| Config file locations | ✅ Yes | `config_dir`, `excluded_features` |
| Backup/temp directories | ✅ Yes | `config_backup_dir`, `temp_dir` |
| Internal module paths | ❌ No | `TRAINING/`, `CONFIG/` (relative) |
| Code structure paths | ❌ No | `__file__`-based paths |
| Default fallbacks | ❌ No | Hardcoded defaults |

---

## 🎯 **Recommendation**

**Current approach is good.** We've made the right paths configurable:
- ✅ User-facing paths (data, output, config files)
- ✅ Environment-specific paths (backups, temp)
- ✅ Kept internal paths hardcoded (module structure)

**Don't over-configure.** The codebase structure (`TRAINING/`, `CONFIG/`) should remain fixed relative to the repo root. Only make paths configurable if users actually need to customize them.

---

**Status:** Guidelines established  
**Next Review:** When new path configuration needs arise

