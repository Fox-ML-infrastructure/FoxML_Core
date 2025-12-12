# Training Routing & Plan System - Setup Complete

**Status: System is fully set up and ready to use.**

## ✅ What's Implemented

### 1. Master Training Plan Structure
- ✅ `master_training_plan.json` is the canonical source of truth
- ✅ `training_plan.json` maintained as convenience mirror for backward compatibility
- ✅ Full metadata: `run_id`, `git_commit`, `config_hash`, `routing_plan_path`, `metrics_snapshot`
- ✅ Derived views generated automatically:
  - `by_target/<target>.json`
  - `by_symbol/<symbol>.json`
  - `by_type/<type>.json`
  - `by_route/<route>.json`

### 2. Training Plan Consumer
- ✅ Loads `master_training_plan.json` (falls back to `training_plan.json` for compatibility)
- ✅ Helper functions:
  - `get_cs_jobs()` - Get all cross-sectional jobs
  - `get_symbol_jobs()` - Get all symbol-specific jobs
  - `get_jobs_for_target()` - Get all jobs for a target
  - `get_jobs_for_symbol()` - Get all jobs for a symbol
  - `get_model_families_for_job()` - Get model families for a specific job
- ✅ Validation function: `validate_training_plan()`
- ✅ Summary function: `get_training_plan_summary()`

### 3. Integration with Intelligent Trainer
- ✅ Automatic training plan loading and filtering
- ✅ **Cross-sectional target filtering** - Fully implemented
- ✅ **Symbol filtering per target** - Computed and stored (ready for symbol-specific execution)
- ✅ **Model family filtering** - Per-target families from training plan
- ✅ Logging: Shows filtering results and family usage

### 4. Training Execution
- ✅ `train_models_for_interval_comprehensive()` accepts `target_families` parameter
- ✅ Per-target model family filtering supported
- ✅ Falls back to global families if per-target not specified

### 5. Utilities
- ✅ `training_plan_utils.py` with helper functions:
  - `print_training_plan_summary()` - Human-readable summary
  - `compare_training_plans()` - Compare two plans

## 📁 File Structure

```
METRICS/
├── routing_candidates.parquet (or .csv)
├── routing_candidates.json
│
├── routing_plan/
│   ├── routing_plan.json
│   ├── routing_plan.yaml
│   └── routing_plan.md
│
└── training_plan/
    ├── master_training_plan.json  ⭐ Single source of truth
    ├── training_plan.json          (convenience mirror)
    ├── training_plan.yaml
    ├── training_plan.md
    │
    ├── by_target/
    │   └── <target>.json
    ├── by_symbol/
    │   └── <symbol>.json
    ├── by_type/
    │   ├── cross_sectional.json
    │   └── symbol_specific.json
    └── by_route/
        ├── cross_sectional.json
        ├── symbol_specific.json
        ├── both.json
        └── experimental_only.json
```

## 🔄 Runtime Flow

1. **Feature Selection** completes
2. **Metrics Aggregation** → `METRICS/routing_candidates.parquet`
3. **Routing Plan Generation** → `METRICS/routing_plan/`
4. **Training Plan Generation** → `METRICS/training_plan/` (with master plan + derived views)
5. **Training Plan Consumption**:
   - Loads `master_training_plan.json`
   - Filters targets for CS training ✅
   - Filters symbols per target (computed, ready for execution) ⚠️
   - Extracts model families per target ✅
6. **Training Execution**:
   - Uses filtered targets
   - Uses per-target model families from plan ✅
   - Logs all filtering decisions

## 📊 Current Capabilities

### ✅ Fully Functional
- Cross-sectional training filtering
- Model family filtering (per-target from training plan)
- Master plan structure with derived views
- Backward compatibility (falls back to `training_plan.json`)

### ⚠️ Ready for Future Enhancement
- Symbol-specific execution filtering (computed, stored, ready to wire into execution)
- Per-job model families (infrastructure exists, can be extended)

## 🧪 Testing

To verify the system works:

```python
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer
from pathlib import Path

trainer = IntelligentTrainer(
    data_dir=Path("data"),
    symbols=["AAPL", "MSFT", "GOOGL"],
    output_dir=Path("results")
)

# Run with automatic routing and training plan
results = trainer.train_with_intelligence(
    auto_targets=True,
    top_n_targets=5,
    auto_features=True,
    top_m_features=100
)

# Check logs for:
# - "Training routing plan generated"
# - "Training plan generated"
# - "Training plan filter applied: X → Y targets"
# - "Using model families from training plan"
```

## 📝 Usage Examples

### Print Training Plan Summary
```python
from TRAINING.orchestration.training_plan_utils import print_training_plan_summary
from pathlib import Path

print_training_plan_summary(Path("results/METRICS/training_plan"))
```

### Compare Two Plans
```python
from TRAINING.orchestration.training_plan_utils import compare_training_plans
from pathlib import Path

diff = compare_training_plans(
    Path("results/run1/METRICS/training_plan"),
    Path("results/run2/METRICS/training_plan")
)
print(f"Added: {len(diff['added_jobs'])}")
print(f"Removed: {len(diff['removed_jobs'])}")
print(f"Changed: {len(diff['changed_jobs'])}")
```

### Manual Plan Loading
```python
from TRAINING.orchestration.training_plan_consumer import (
    load_training_plan,
    get_cs_jobs,
    get_model_families_for_job
)
from pathlib import Path

plan = load_training_plan(Path("results/METRICS/training_plan"))
cs_jobs = get_cs_jobs(plan)
families = get_model_families_for_job(plan, "y_will_swing_low_10m_0.20", None, "cross_sectional")
```

## 🎯 Next Steps (Optional Enhancements)

1. **Symbol-Specific Execution**: Wire `filtered_symbols_by_target` into symbol-specific training loops
2. **Per-Job Families**: Extend to support different families per `(target, symbol)` job
3. **Priority-Based Scheduling**: Use job priorities to schedule training order
4. **Progress Tracking**: Track which jobs completed successfully

## 📚 Documentation

**For architecture and implementation details**, see the internal documentation.
- `README.md` - User-facing guide

## ✨ Summary

The training routing & plan system is **fully set up and operational**. All core features are implemented:
- ✅ Master plan structure
- ✅ Derived views
- ✅ CS filtering
- ✅ Model family filtering
- ✅ Symbol filtering (computed, ready for execution)

The system is backward compatible and ready for production use!
