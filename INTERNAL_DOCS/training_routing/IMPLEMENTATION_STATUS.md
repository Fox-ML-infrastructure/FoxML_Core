# Implementation Status

**What's implemented vs. what's planned for the training routing & plan system.**

## ✅ Fully Implemented

### 1. Metrics Aggregation
- ✅ Collects metrics from feature selection outputs
- ✅ Aggregates cross-sectional (pooled) metrics
- ✅ Aggregates symbol-specific metrics
- ✅ Loads stability metrics from snapshots
- ✅ **Parquet → CSV fallback** (prevents crashes when `pyarrow` not installed)
- ✅ Saves routing candidates to `METRICS/routing_candidates.parquet` (or `.csv`)

**Files:**
- `TRAINING/orchestration/metrics_aggregator.py`

### 2. Routing Plan Generation
- ✅ Priority-ordered routing decision logic
- ✅ Stability classification from metrics
- ✅ Cross-sectional eligibility evaluation
- ✅ Symbol-specific eligibility evaluation
- ✅ Route combination logic (CS + local states → route)
- ✅ Hard blocks (leakage, insufficient data)
- ✅ Experimental lane support
- ✅ Both-strong behavior configuration
- ✅ Saves routing plan as JSON/YAML/Markdown

**Files:**
- `TRAINING/orchestration/training_router.py`
- `TRAINING/orchestration/routing_integration.py`

### 3. Training Plan Generation
- ✅ Converts routing decisions to actionable job specs
- ✅ Job specification includes:
  - `job_id`, `target`, `symbol`, `route`, `training_type`
  - `model_families`, `priority`, `reason`, `metadata`
- ✅ Summary statistics (by route, type, priority)
- ✅ Saves training plan as JSON/YAML/Markdown
- ✅ Automatic generation after routing plan creation

**Files:**
- `TRAINING/orchestration/training_plan_generator.py`

### 4. Training Plan Consumption (Cross-Sectional)
- ✅ Loads training plan from disk
- ✅ Filters targets for cross-sectional training
- ✅ Integrated into `intelligent_trainer.py`
- ✅ Logs filtering results: `"Training plan filter applied: 10 → 7 targets"`
- ✅ Backward compatible (if plan missing, trains all targets)

**Files:**
- `TRAINING/orchestration/training_plan_consumer.py`
- `TRAINING/orchestration/intelligent_trainer.py` (integration)

### 5. Automatic Integration
- ✅ Routing plan automatically generated after feature selection
- ✅ Training plan automatically generated after routing plan
- ✅ Training phase automatically consumes training plan
- ✅ Non-blocking (fails gracefully if metrics unavailable)

**Files:**
- `TRAINING/orchestration/routing_integration.py`
- `TRAINING/orchestration/intelligent_trainer.py`

### 6. Documentation
- ✅ User-facing guide (`README.md`)
- ✅ Architecture documentation (`ARCHITECTURE.md`)
- ✅ Implementation details (`ROUTING_SYSTEM_SUMMARY.md`)
- ✅ Known issues and fixes (`ERRORS_FIXED.md`)
- ✅ Integration summary (`INTEGRATION_SUMMARY.md`)

## ⚠️ Partially Implemented / Planned

### 1. Symbol-Specific Training Execution Filtering

**Current Status:**
- ✅ Symbol-specific jobs are present in the **training plan**
- ✅ Routing decisions include `ROUTE_SYMBOL_SPECIFIC`, `ROUTE_BOTH`, `ROUTE_EXPERIMENTAL_ONLY`
- ✅ `training_plan_consumer.py` has `filter_symbols_by_training_plan()` function
- ⚠️ **Execution phase filtering based on symbol-specific jobs is not fully wired**

**What's Missing:**
- Symbol-specific training loops need to check training plan before executing
- Per-`(target, symbol)` filtering needs to be integrated into training execution
- Currently, symbol-specific jobs exist in plan but aren't enforced at execution time

**TODO:**
- Extend `intelligent_trainer.py` to filter per `(target, symbol)` for symbol-specific training
- Align symbol-specific training loops with `training_plan` entries
- Ensure only approved symbol-specific jobs are executed

**Files to Update:**
- `TRAINING/orchestration/intelligent_trainer.py`
- Symbol-specific training execution code (wherever it lives)

### 2. Model-Family-Level Filtering

**Current Status:**
- ✅ Training plan includes `model_families` list per job
- ⚠️ **Training loop still trains all model families** for a given job

**What's Missing:**
- Training loop should respect `model_families` from training plan
- Only specified families should be trained per job

**TODO:**
- Use `model_families` list in training loop to restrict which families run per job
- Potentially assign different priorities/resources by family
- Ensure training respects per-job family specifications

**Files to Update:**
- Training execution code (wherever model families are selected)

### 3. Master Training Plan Structure

**Current Status:**
- ✅ Training plan saved as `training_plan.json`
- ⚠️ **Not yet formalized as "master" plan with derived views**

**What's Missing:**
- Rename/formalize `training_plan.json` as `master_training_plan.json` (canonical)
- Generate derived views:
  - `by_target/<target>.json` - All jobs for a target
  - `by_symbol/<symbol>.json` - All jobs for a symbol
  - `by_type/<type>.json` - All jobs of a type
  - `by_route/<route>.json` - All jobs with a route

**TODO:**
- Update `training_plan_generator.py` to:
  - Save master plan as `master_training_plan.json`
  - Keep `training_plan.json` as convenience mirror
  - Generate derived view artifacts
- Update documentation to clarify master plan is single source of truth
- Update `training_plan_consumer.py` to only read master plan

**Files to Update:**
- `TRAINING/orchestration/training_plan_generator.py`
- `TRAINING/orchestration/training_plan_consumer.py`
- Documentation

### 4. Advanced Routing Logic

**Current Status:**
- ✅ Basic routing logic implemented (CS vs symbol, both, experimental, blocked)
- ✅ Stability classification
- ✅ Hard blocks (leakage, insufficient data)
- ⚠️ **Some advanced features may not be fully encoded in config**

**What May Be Missing:**
- Explicit stability state rules (`STRONG` / `WEAK_BUT_OK` / `EXPERIMENTAL` / `DISALLOWED`) may be implemented but not clearly documented
- Feature leakage status and safe-feature enforcement may need clearer integration
- Experimental lane limits (`max_fraction_symbols_per_target`) may need enforcement logic

**TODO (if not already done):**
- Verify all stability-state rules are fully implemented
- Ensure feature-safety rules are clearly encoded in config and enforced
- Verify experimental lane limits are enforced (not just checked)
- Document all routing rules explicitly

**Files to Review:**
- `TRAINING/orchestration/training_router.py`
- `CONFIG/training_config/routing_config.yaml`

## 🔮 Future Enhancements

### 1. Per-Target/Symbol/Type Views
- Generate `by_target/`, `by_symbol/`, `by_type/` views automatically
- Useful for dashboards, debugging, and external schedulers

### 2. Training Plan Validation
- Validate training plan against routing plan
- Check for consistency (e.g., all CS jobs have corresponding routing decisions)
- Warn if plan is stale (older than routing plan)

### 3. Training Plan Diffing
- Compare training plans between runs
- Show what changed (new jobs, removed jobs, priority changes)
- Useful for understanding routing changes

### 4. Priority-Based Scheduling
- Use job priorities to schedule training order
- Higher priority jobs train first
- Resource allocation based on priority

### 5. Model Family Assignment Logic
- Smarter assignment of model families per job
- Based on job priority, route, or metadata
- Different families for experimental vs. production jobs

### 6. Training Plan Metrics
- Track which jobs completed successfully
- Compare planned vs. executed jobs
- Identify jobs that were planned but not executed (and why)

## Summary

**Core System:** ✅ Fully functional
- Metrics aggregation ✅
- Routing decisions ✅
- Training plan generation ✅
- CS training filtering ✅

**Execution Integration:** ⚠️ Partially complete
- Symbol-specific filtering: planned, not fully wired
- Model-family filtering: planned, not implemented
- Master plan structure: planned, not formalized

**Advanced Features:** 🔮 Future work
- Derived views, validation, diffing, priority scheduling, etc.

## Migration Path

1. **Short-term (Current):**
   - Use system as-is for CS training filtering ✅
   - Symbol-specific jobs exist in plan but aren't enforced (acceptable for now)

2. **Medium-term (Next):**
   - Wire symbol-specific filtering into execution
   - Implement model-family filtering
   - Formalize master plan structure

3. **Long-term (Future):**
   - Add derived views, validation, diffing
   - Priority-based scheduling
   - Advanced metrics and tracking
