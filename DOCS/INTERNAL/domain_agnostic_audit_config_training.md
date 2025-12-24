# Domain-Agnostic Audit: Config & Training Modules

**Date:** 2025-01-27  
**Scope:** CONFIG/ and TRAINING/ directories only  
**Purpose:** First-pass audit to identify finance/trading-specific terminology that should be generalized  
**Status:** Audit only - no changes made

---

## Executive Summary

The codebase has already removed hard coupling (APIs, trading modules). Remaining issues are primarily **surface-level terminology** in configs and training code. The architecture is domain-agnostic; what remains is naming and documentation polish.

**Key Finding:** The term `symbol` is pervasive throughout config and training code. This is the primary finance-specific term that needs generalization to `entity` or `series_id`.

**Risk Level:** Low - These are naming/documentation issues, not architectural problems.

---

## 1. Code-Level Findings: Finance/Trading Terminology

### 1.1 High Priority: Core Terminology

#### `symbol` / `symbols` (CRITICAL - Pervasive)

**Location:** Throughout CONFIG/ and TRAINING/

**Impact:** This is the most common finance-specific term. It appears in:
- Config schemas (`config_schemas.py`)
- Config builder (`config_builder.py`)
- All experiment configs
- All feature selection configs
- All target ranking configs
- Training strategies
- Data loading utilities
- Cross-sectional processing

**Examples:**
- `CONFIG/config_schemas.py:65` - `symbols: List[str]` in `ExperimentConfig`
- `CONFIG/config_builder.py:92-93` - Validation requiring `data.symbols`
- `TRAINING/utils/core_utils.py:94` - `SYMBOL_COL = "symbol"`
- `TRAINING/training_strategies/main.py:177` - `--symbols` CLI argument
- `TRAINING/utils/cross_sectional_data.py:38` - Function signatures with `symbols: List[str]`

**Recommendation:**
- Rename `symbol` → `entity` or `series_id` in:
  - Config schemas
  - Function parameters
  - Variable names
  - Column names (where `symbol` is a data column)
- Keep `symbol` as an **alias** for backward compatibility during transition
- Update all YAML configs to use `entities` or `series_ids` (with `symbols` as deprecated alias)

**Files to Update (Priority Order):**
1. `CONFIG/config_schemas.py` - Core schema definitions
2. `CONFIG/config_builder.py` - Config loading/validation
3. `TRAINING/utils/core_utils.py` - Core constants
4. `TRAINING/training_strategies/main.py` - CLI interface
5. `TRAINING/utils/cross_sectional_data.py` - Data loading
6. All YAML configs in `CONFIG/`

**Estimated Impact:** ~50-70 files need updates (mostly variable renames, some schema changes)

---

#### `ticker` / `tickers` (MEDIUM - Single File)

**Location:** `TRAINING/common/tickers.py`

**Impact:** Entire file is finance-specific:
- `normalize_symbol()` function with broker-compatible format
- `CONFIRMED_BLACKLIST` for non-tradable symbols
- Comments mentioning "broker-compatible format"

**Examples:**
- `TRAINING/common/tickers.py:40` - `"""Normalize symbol to broker-compatible format."""`
- `TRAINING/common/tickers.py:39` - Function name `normalize_symbol()`

**Recommendation:**
- Rename file: `tickers.py` → `entity_normalization.py` or `series_utils.py`
- Rename functions: `normalize_symbol()` → `normalize_entity()` or `normalize_series_id()`
- Generalize blacklist concept (if needed) or remove if finance-specific
- Update comments to remove "broker-compatible" language

**Files to Update:**
1. `TRAINING/common/tickers.py` - Rename and generalize
2. Any imports of `tickers.py` (search for `from common.tickers import`)

---

### 1.2 Medium Priority: Comments & Documentation

#### Trading/Market References in Comments

**Location:** Various config files and training code

**Examples:**
- `CONFIG/model_config/gmm_regime.yaml:5` - `"market regime classification"`
- `CONFIG/model_config/change_point.yaml:5` - `"market regime transitions"`
- `CONFIG/model_config/reward_based.yaml:2` - `"trading decisions"`
- `CONFIG/training_config/first_batch_specs.yaml:124` - `"trading returns"`
- `CONFIG/feature_target_schema.yaml:62` - `"Core market data (OHLCV)"`

**Recommendation:**
- Update comments to be domain-agnostic:
  - "market regime" → "regime" or "state regime"
  - "trading decisions" → "decision making" or "action selection"
  - "trading returns" → "returns" or "outcomes"
  - "market data" → "time series data" or "OHLCV data" (OHLCV is generic enough)

**Files to Update:**
- `CONFIG/model_config/*.yaml` - Model descriptions
- `CONFIG/feature_target_schema.yaml` - Schema descriptions
- `CONFIG/training_config/first_batch_specs.yaml` - Spec descriptions

---

#### "Live" Terminology

**Location:** `TRAINING/live/` directory, `CONFIG/training_config/sequential_config.yaml`

**Impact:** "Live" is used for real-time inference, which is generic enough, but some comments may imply trading context.

**Examples:**
- `TRAINING/live/__init__.py:18` - `"""Live trading and real-time data processing utilities"""`
- `CONFIG/training_config/sequential_config.yaml:35-36` - `live:` section with `ttl_seconds`

**Recommendation:**
- Update `TRAINING/live/__init__.py` docstring: `"Live trading"` → `"Real-time inference and data processing utilities"`
- Keep `live:` config key (it's generic enough for "live inference")
- Review other comments in `live/` directory for trading-specific language

**Files to Update:**
1. `TRAINING/live/__init__.py` - Docstring
2. Review other files in `TRAINING/live/` for trading-specific comments

---

#### "Session" Terminology

**Location:** Configs and training code

**Impact:** "Session" can mean:
- Training session (generic) ✅
- Trading session (finance-specific) ❌

**Examples:**
- `CONFIG/feature_target_schema.yaml:17` - `- session` in schema
- `TRAINING/training_strategies/main.py:205` - `--session-id` CLI argument (this is generic - training session)
- `TRAINING/common/threads.py:610` - Comment about "session/context" (TensorFlow session - generic)

**Recommendation:**
- `--session-id` is fine (training session is generic)
- `CONFIG/feature_target_schema.yaml:17` - Check if `session` column is finance-specific (trading session hours) or generic (any session identifier)
  - If finance-specific → document as optional/domain-specific
  - If generic → keep as-is

**Files to Review:**
1. `CONFIG/feature_target_schema.yaml` - Determine if `session` is domain-specific

---

### 1.3 Low Priority: Examples & Test Data

#### Example Symbols in Configs

**Location:** Example configs, test scripts

**Examples:**
- `CONFIG/experiments/fwd_ret_60m_test.yaml:10` - `symbols: [AAPL, MSFT]`
- `CONFIG/README.md:39` - `symbols: [AAPL, MSFT]`
- `TRAINING/test_gpu_models.sh:82` - Default symbols: `AAPL TSLA MSFT`

**Recommendation:**
- Keep finance examples in example configs (they're just examples)
- Consider adding a comment: `# Example: Using finance symbols (AAPL, MSFT). Replace with your domain's entity IDs.`
- Test scripts can keep finance symbols (they're just test data)

**Priority:** Very Low - Examples are fine to keep finance-specific

---

## 2. Config Surface Area

### 2.1 YAML Config Keys

**Current State:** Most config keys are already generic:
- `model_families` ✅
- `targets` ✅
- `features` ✅
- `data_dir` ✅
- `interval` / `bar_interval` ✅ (generic time intervals)

**Finance-Specific Keys Found:**
- `symbols` → Should become `entities` or `series_ids` (with `symbols` as deprecated alias)
- `symbol_holdout_test_size` → `entity_holdout_test_size` or `series_holdout_test_size`
- `min_symbols` → `min_entities` or `min_series`
- `max_samples_per_symbol` → `max_samples_per_entity` or `max_samples_per_series`
- `max_rows_per_symbol` → `max_rows_per_entity` or `max_rows_per_series`
- `parallel_symbols` → `parallel_entities` or `parallel_series`
- `symbol_threshold` → `entity_threshold` or `series_threshold`
- `symbol_specific` → `entity_specific` or `series_specific`

**Recommendation:**
- Create migration plan:
  1. Add new keys (`entities`, `max_samples_per_entity`, etc.)
  2. Support both old and new keys (with deprecation warnings)
  3. Update all internal code to use new keys
  4. Remove old keys in next major version

**Files to Update:**
- All YAML files in `CONFIG/` that use `symbol*` keys
- Config loaders to support both old/new keys

---

### 2.2 CLI Arguments

**Current State:**
- `--symbols` → Should become `--entities` or `--series-ids` (with `--symbols` as deprecated alias)
- `--max-symbols` → `--max-entities` or `--max-series`
- `--max-samples-per-symbol` → `--max-samples-per-entity` or `--max-samples-per-series`
- `--max-rows-per-symbol` → `--max-rows-per-entity` or `--max-rows-per-series`

**Recommendation:**
- Add new CLI arguments with generic names
- Keep old arguments as deprecated aliases (with warnings)
- Update help text to use generic language

**Files to Update:**
- `TRAINING/training_strategies/main.py` - CLI argument definitions
- Any other scripts with `--symbols` arguments

---

## 3. Logging & Error Messages

### 3.1 Log Messages

**Current State:** Most log messages are already generic (talk about data, features, models, etc.)

**Finance-Specific Logs Found:**
- Logs mentioning "symbol" (should say "entity" or "series")
- Some logs may mention "trading" or "market" in comments (need to scan)

**Recommendation:**
- Scan log messages for finance-specific language
- Update to generic terminology:
  - "Processing symbol X" → "Processing entity X" or "Processing series X"
  - "Symbol data" → "Entity data" or "Series data"

**Files to Review:**
- All files in `TRAINING/` that log messages with "symbol"
- Use grep: `rg "logger.*symbol" -i TRAINING/`

---

## 4. Data Structure & Column Names

### 4.1 Column Names

**Current State:**
- `SYMBOL_COL = "symbol"` in `TRAINING/utils/core_utils.py:94`
- Data frames use `symbol` as a column name for entity identification

**Recommendation:**
- Rename constant: `SYMBOL_COL` → `ENTITY_COL` or `SERIES_ID_COL`
- Update default column name: `"symbol"` → `"entity"` or `"series_id"`
- Support both column names during transition (with deprecation)

**Files to Update:**
1. `TRAINING/utils/core_utils.py:94` - Constant definition
2. All code that references `SYMBOL_COL` or uses `"symbol"` as column name

---

## 5. Function & Variable Names

### 5.1 Function Names

**Finance-Specific Function Names Found:**
- `load_mtf_data(data_dir, symbols, ...)` → `load_mtf_data(data_dir, entities, ...)`
- `process_symbol_universe()` → `process_entity_universe()` or `process_series_universe()`
- `normalize_symbol()` → `normalize_entity()` or `normalize_series_id()`

**Recommendation:**
- Rename functions to use generic terminology
- Keep old functions as deprecated wrappers during transition

**Files to Update:**
- `TRAINING/utils/cross_sectional_data.py` - `load_mtf_data()` signature
- `TRAINING/common/tickers.py` - `normalize_symbol()`, `process_symbol_universe()`
- All call sites of these functions

---

## 6. Summary: Action Items

### High Priority (Core Functionality)

1. **Rename `symbol` → `entity` or `series_id`** (throughout codebase)
   - Config schemas
   - Function parameters
   - Variable names
   - Column names
   - YAML config keys
   - CLI arguments

2. **Rename `tickers.py` → `entity_normalization.py`**
   - Update file name
   - Update function names
   - Generalize comments

3. **Update core constants**
   - `SYMBOL_COL` → `ENTITY_COL` or `SERIES_ID_COL`

### Medium Priority (Documentation & Comments)

4. **Update model config descriptions**
   - Remove "market" / "trading" language
   - Use generic terminology

5. **Update `live/` directory docstrings**
   - Remove "trading" references

6. **Review `session` usage**
   - Determine if finance-specific or generic

### Low Priority (Examples & Tests)

7. **Add comments to example configs**
   - Note that finance symbols are just examples
   - Suggest replacing with domain-specific entity IDs

---

## 7. Migration Strategy

### Phase 1: Add Generic Names (Backward Compatible)
- Add new config keys (`entities`, `max_samples_per_entity`, etc.)
- Add new CLI arguments (`--entities`, etc.)
- Support both old and new names (old names log deprecation warnings)

### Phase 2: Update Internal Code
- Update all internal code to use new generic names
- Update function signatures
- Update variable names
- Update column names

### Phase 3: Update Documentation
- Update all YAML configs to use new keys
- Update README and docs
- Update example configs

### Phase 4: Remove Old Names (Breaking Change)
- Remove deprecated aliases
- Update version number (major version bump)

---

## 8. Terminology Decision: `entity` vs `series_id`

**Recommendation:** Use `entity` (shorter, clearer, more generic)

**Rationale:**
- `entity` is domain-agnostic (works for stocks, sensors, customers, etc.)
- `series_id` implies time series specifically (but we also support cross-sectional)
- `entity` is shorter and easier to type
- Common in ML/data science (e.g., "entity embeddings", "entity resolution")

**Alternative:** `series_id` if you want to emphasize time series nature

**Decision Needed:** Choose one and use consistently throughout.

---

## 9. Risk Assessment

**Overall Risk:** Low

**Reasons:**
1. Architecture is already domain-agnostic ✅
2. Only surface-level terminology needs changes ✅
3. Can be done incrementally with backward compatibility ✅
4. No breaking changes to core algorithms ✅

**Potential Issues:**
1. Large number of files to update (~50-70 files)
2. Need to ensure all references are updated (easy to miss some)
3. Need to maintain backward compatibility during transition
4. Test coverage needed to ensure nothing breaks

**Mitigation:**
- Use IDE refactoring tools for systematic renames
- Add deprecation warnings for old names
- Comprehensive testing after changes
- Incremental rollout (one module at a time)

---

## 10. Next Steps

1. **Decision:** Choose `entity` vs `series_id` terminology
2. **Plan:** Create detailed migration plan with file list
3. **Implement:** Start with core config schemas, then propagate outward
4. **Test:** Comprehensive testing after each phase
5. **Document:** Update all documentation with new terminology

---

## Appendix: File Count Summary

**Files with `symbol` references:**
- CONFIG/: ~15 files
- TRAINING/: ~35-40 files
- Total: ~50-55 files

**Files with `ticker` references:**
- TRAINING/: 1 file (`common/tickers.py`)

**Files with trading/market comments:**
- CONFIG/: ~5-10 files (mostly in comments)

**Total files needing updates:** ~60-70 files (mostly renames, some logic changes)

---

**End of Audit**
