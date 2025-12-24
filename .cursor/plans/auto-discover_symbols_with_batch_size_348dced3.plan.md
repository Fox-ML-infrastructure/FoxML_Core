---
name: Auto-Discover Symbols with Batch Size
overview: Add auto-discovery of symbols from data_dir when symbols list is empty, with a configurable batch_size to limit how many symbols are selected. The selected symbols flow through the entire pipeline (target ranking, feature selection, routing, training).
todos:
  - id: "1"
    content: Add symbol_batch_size to DataConfig schema and update validation
    status: pending
  - id: "2"
    content: Create symbol_discovery.py utility with discover_symbols_from_data_dir() and select_symbol_batch()
    status: pending
  - id: "3"
    content: Update config_builder.py to handle empty symbols and load symbol_batch_size
    status: pending
    dependencies:
      - "1"
  - id: "4"
    content: Integrate auto-discovery in intelligent_trainer.py __init__() method
    status: pending
    dependencies:
      - "2"
      - "3"
  - id: "5"
    content: Update e2e_full_targets_test.yaml with example auto-discovery config
    status: pending
    dependencies:
      - "1"
  - id: "6"
    content: Update _template.yaml with documentation for auto-discovery feature
    status: pending
    dependencies:
      - "1"
  - id: "7"
    content: Verify symbols flow correctly through target ranking, feature selection, routing, and training
    status: pending
    dependencies:
      - "4"
---

# Aut

o-Discover Symbols with Batch Size Configuration

## Overview

Enable automatic symbol discovery from `data_dir` when `symbols` field is empty, with a configurable `symbol_batch_size` to limit selection. Selected symbols flow through the entire pipeline: target ranking → feature selection → routing decisions → training plan → model training.

## Changes Required

### 1. Config Schema Updates

**File**: `CONFIG/config_schemas.py`

- Add `symbol_batch_size: Optional[int] = None `to `DataConfig` dataclass
- Update validation to allow empty `symbols` list when auto-discovery is enabled
- Add validation: `symbol_batch_size` must be >= 1 if provided

**File**: `CONFIG/config_builder.py`

- Modify `load_experiment_config()` to:
- Allow `symbols: []` or missing `symbols` field (remove strict validation)
- Load `symbol_batch_size` from `data_data.get('symbol_batch_size')`
- Pass `symbol_batch_size` to `DataConfig` and `ExperimentConfig`

### 2. Symbol Discovery Utility

**File**: `TRAINING/orchestration/utils/symbol_discovery.py` (NEW)

- Create `discover_symbols_from_data_dir(data_dir: Path, interval: Optional[str] = None) -> List[str]`
- Scan `data_dir/interval={interval}/symbol=*/` directories
- Extract symbol names from directory names (e.g., `symbol=AAPL` → `AAPL`)
- Handle both new structure (`interval=5m/symbol=SYMBOL/`) and legacy (`symbol=SYMBOL/`)
- Validate that parquet files exist for each symbol
- Return sorted list of valid symbols
- Create `select_symbol_batch(symbols: List[str], batch_size: Optional[int] = None, random_seed: Optional[int] = None) -> List[str]`
- If `batch_size` is None or >= len(symbols), return all symbols
- Otherwise, randomly sample `batch_size` symbols (with seed for reproducibility)
- Log selection

### 3. Intelligent Trainer Integration

**File**: `TRAINING/orchestration/intelligent_trainer.py`

- Modify `__init__()` method (around line 190-199):
- After loading `experiment_config.symbols`, check if empty/missing
- If empty/missing:
    - Call `discover_symbols_from_data_dir(self.data_dir, interval)`
    - Apply `symbol_batch_size` limit using `select_symbol_batch()`
    - Log discovered and selected symbols
- Store final symbol list in `self.symbols`
- Ensure `symbol_batch_size` is loaded from config and passed through

### 4. Config File Updates

**File**: `CONFIG/experiments/e2e_full_targets_test.yaml`

- Add example configuration:
  ```yaml
      data:
        data_dir: data/data_labeled_v3/interval=5m
        symbols: []  # Empty = auto-discover all symbols
        symbol_batch_size: 15  # Select 15 symbols from discovered set
        interval: 5m
  ```


**File**: `CONFIG/experiments/_template.yaml`

- Add documentation for auto-discovery:
  ```yaml
      data:
        # Option 1: Explicit symbols
        symbols: [AAPL, MSFT, GOOGL]
        
        # Option 2: Auto-discover with batch limit
        symbols: []  # Empty = auto-discover all symbols from data_dir
        symbol_batch_size: 15  # Select 15 symbols (random sample)
  ```




### 5. Pipeline Flow Verification

**Files to verify symbols flow correctly**:

- `TRAINING/orchestration/intelligent_trainer.py` - `rank_targets_auto()` uses `self.symbols`
- `TRAINING/ranking/utils/cross_sectional_data.py` - `prepare_cross_sectional_data_for_ranking()` receives symbols
- `TRAINING/ranking/feature_selector.py` - Feature selection per symbol
- `TRAINING/orchestration/training_router.py` - Routing decisions use symbols
- `TRAINING/orchestration/training_plan_generator.py` - Training plan uses symbols

**Action**: Verify all these components use `self.symbols` from `IntelligentTrainer` and don't re-discover symbols independently.

## Implementation Details

### Symbol Discovery Logic

```python
def discover_symbols_from_data_dir(data_dir: Path, interval: Optional[str] = None) -> List[str]:
    """
    Discover all available symbols from data directory structure.
    
    Supports:
    - data_dir/interval=5m/symbol=AAPL/AAPL.parquet (new structure)
    - data_dir/symbol=AAPL/AAPL.parquet (legacy structure)
    """
    symbols = []
    data_path = Path(data_dir)
    
    # Try new structure first: interval={interval}/symbol={symbol}/
    if interval:
        interval_dir = data_path / f"interval={interval}"
        if interval_dir.exists():
            for symbol_dir in interval_dir.glob("symbol=*"):
                symbol = symbol_dir.name.split("=")[1]
                parquet_file = symbol_dir / f"{symbol}.parquet"
                if parquet_file.exists():
                    symbols.append(symbol)
    
    # Fallback to legacy: symbol={symbol}/
    if not symbols:
        for symbol_dir in data_path.glob("symbol=*"):
            symbol = symbol_dir.name.split("=")[1]
            parquet_file = symbol_dir / f"{symbol}.parquet"
            if parquet_file.exists():
                symbols.append(symbol)
    
    return sorted(symbols)
```



### Batch Selection Logic

```python
def select_symbol_batch(symbols: List[str], batch_size: Optional[int] = None, random_seed: Optional[int] = None) -> List[str]:
    """
    Select a batch of symbols from the full list.
    
    If batch_size is None or >= len(symbols), returns all symbols.
    Otherwise, randomly samples batch_size symbols.
    """
    if not symbols:
        return []
    
    if batch_size is None or batch_size >= len(symbols):
        return symbols
    
    import random
    if random_seed is not None:
        random.seed(random_seed)
    
    selected = random.sample(symbols, batch_size)
    logger.info(f"Selected {len(selected)}/{len(symbols)} symbols: {selected}")
    return sorted(selected)
```



## Testing Checklist

- [ ] Empty `symbols: []` with `symbol_batch_size: 15` discovers and selects 15 symbols
- [ ] Empty `symbols: []` without `symbol_batch_size` uses all discovered symbols
- [ ] Explicit `symbols: [AAPL, MSFT]` ignores auto-discovery (backward compatible)
- [ ] Selected symbols flow through target ranking (cross-sectional + individual)
- [ ] Selected symbols flow through feature selection
- [ ] Selected symbols flow through routing decisions
- [ ] Selected symbols flow through training plan generation
- [ ] Selected symbols flow through model training
- [ ] Random seed ensures reproducibility when `symbol_batch_size` is set

## Files to Modify

1. `CONFIG/config_schemas.py` - Add `symbol_batch_size` to `DataConfig`
2. `CONFIG/config_builder.py` - Handle empty symbols + load batch size
3. `TRAINING/orchestration/utils/symbol_discovery.py` - NEW: Discovery utility
4. `TRAINING/orchestration/intelligent_trainer.py` - Integrate auto-discovery
5. `CONFIG/experiments/e2e_full_targets_test.yaml` - Add example config
6. `CONFIG/experiments/_template.yaml` - Document feature

## Backward Compatibility

- Explicit `symbols: [AAPL, MSFT, ...]` continues to work (no change)
- If `symbols` is provided, auto-discovery is skipped