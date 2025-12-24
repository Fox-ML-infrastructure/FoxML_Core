# Errors Fixed & Training Plan Added

## Errors Found & Fixed

### 1. Parquet Dependency Issue ✅ FIXED

**Problem:** `to_parquet()` requires `pyarrow` which may not be installed.

**Fix:** Added fallback to CSV if parquet fails:
```python
try:
    candidates_df.to_parquet(output_path, index=False)
except Exception as e:
    logger.warning(f"Failed to save as parquet ({e}), falling back to CSV")
    csv_path = output_path.with_suffix(".csv")
    candidates_df.to_csv(csv_path, index=False)
```

**Location:** `TRAINING/orchestration/metrics_aggregator.py`

### 2. Missing Training Plan Artifact ✅ ADDED

**Problem:** Routing plan existed but no actionable training plan was generated.

**Solution:** Created `TrainingPlanGenerator` that converts routing decisions into training jobs.

**New File:** `TRAINING/orchestration/training_plan_generator.py`

**Features:**
- Converts routing decisions → actionable training jobs
- Includes job priorities, model families, reasons
- Generates JSON/YAML/Markdown outputs
- Automatically integrated into routing pipeline

## New Training Plan Artifact

The system now generates a **training plan** with:

1. **Job Specifications**
   - Job ID, target, symbol
   - Training type (CS vs symbol-specific)
   - Model families to train
   - Priority (for scheduling)
   - Reason for job creation

2. **Summary Statistics**
   - Jobs by route
   - Jobs by type
   - Jobs by priority
   - Total counts

3. **Output Formats**
   - `training_plan.json` - Machine-readable
   - `training_plan.yaml` - YAML format
   - `training_plan.md` - Human-readable report

## Usage

### Automatic (Integrated)

Training plan is automatically generated after routing plan:

```python
# In intelligent_trainer.py, after feature selection:
routing_plan = generate_routing_plan_after_feature_selection(
    output_dir=self.output_dir,
    targets=targets,
    symbols=self.symbols,
    generate_training_plan=True,  # Default: True
    model_families=["lightgbm", "xgboost"]  # Optional
)
# Training plan saved to METRICS/training_plan/
```

### Manual

```python
from TRAINING.orchestration.training_plan_generator import (
    generate_training_plan_from_routing
)

plan = generate_training_plan_from_routing(
    routing_plan_path=Path("METRICS/routing_plan/routing_plan.json"),
    output_dir=Path("METRICS/training_plan"),
    model_families=["lightgbm", "xgboost", "random_forest"]
)
```

### Programmatic

```python
from TRAINING.orchestration.training_plan_generator import (
    TrainingPlanGenerator,
    load_routing_plan
)

routing_plan = load_routing_plan(Path("METRICS/routing_plan"))
generator = TrainingPlanGenerator(
    routing_plan=routing_plan,
    model_families=["lightgbm", "xgboost"]
)
training_plan = generator.generate_training_plan(
    output_dir=Path("METRICS/training_plan"),
    include_blocked=False
)

# Access jobs
for job_data in training_plan["jobs"]:
    print(f"Job: {job_data['job_id']}")
    print(f"  Target: {job_data['target']}")
    print(f"  Symbol: {job_data['symbol']}")
    print(f"  Route: {job_data['route']}")
    print(f"  Priority: {job_data['priority']}")
```

## Training Plan Structure

```json
{
  "metadata": {
    "generated_at": "2025-12-11T18:45:00Z",
    "routing_plan_path": "routing_candidates.parquet",
    "total_jobs": 42,
    "model_families": ["lightgbm", "xgboost"]
  },
  "jobs": [
    {
      "job_id": "cs_y_will_swing_low_10m_0.20",
      "target": "y_will_swing_low_10m_0.20",
      "symbol": null,
      "route": "ROUTE_CROSS_SECTIONAL",
      "training_type": "cross_sectional",
      "model_families": ["lightgbm", "xgboost"],
      "priority": 2,
      "reason": "CS training enabled",
      "metadata": {...}
    },
    {
      "job_id": "sym_y_will_swing_low_10m_0.20_AAPL",
      "target": "y_will_swing_low_10m_0.20",
      "symbol": "AAPL",
      "route": "ROUTE_BOTH",
      "training_type": "symbol_specific",
      "model_families": ["lightgbm", "xgboost"],
      "priority": 3,
      "reason": "Both CS and local strong → ROUTE_BOTH",
      "metadata": {
        "needs_cs_ensemble": true
      }
    }
  ],
  "summary": {
    "by_route": {...},
    "by_type": {...},
    "by_priority": {...},
    "total_cs_jobs": 5,
    "total_symbol_jobs": 37,
    "total_blocked": 0
  }
}
```

## Integration Points

1. **Automatic Generation**: After routing plan creation in `routing_integration.py`
2. **Standalone**: Can be generated independently from routing plan file
3. **Programmatic**: Full API for custom training job generation

## Next Steps

The training plan can now be consumed by:
- Training schedulers
- Job queues (Celery, Dask, etc.)
- Parallel training executors
- CI/CD pipelines

Each job is self-contained with all information needed to execute training.
