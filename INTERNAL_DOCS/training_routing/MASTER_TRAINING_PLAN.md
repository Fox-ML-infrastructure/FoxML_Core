# Master Training Plan Structure

**Design and structure of the master training plan and its derived artifacts.**

## Design Principle

**One master training plan = single source of truth, consumed by the training phase.**

**Many derived artifacts = human-readable and machine-friendly "views" of pieces of that master plan.**

## Master Plan File

### Canonical Location

**`METRICS/training_plan/master_training_plan.json`**

This is the **only file the training phase should read**. All other files are derived views for humans, dashboards, and external schedulers.

### Convenience Mirrors

For backward compatibility and tooling that expects `training_plan.json`:

- `METRICS/training_plan/training_plan.json` → Exact copy or symlink to master plan
- `METRICS/training_plan/training_plan.yaml` → YAML representation
- `METRICS/training_plan/training_plan.md` → Human-readable Markdown report

### Schema

```jsonc
{
  "metadata": {
    "generated_at": "2025-12-11T18:45:00Z",
    "run_id": "20251211_184500",
    "git_commit": "abc1234",
    "config_hash": "routing_cfg_v3",
    "routing_plan_path": "METRICS/routing_plan/routing_plan.json",
    "metrics_snapshot": "METRICS/routing_candidates.parquet",
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
      "priority": 90,
      "reason": "CS strong and stable",
      "metadata": {
        "cs_state": "STRONG",
        "sample_size": 123456,
        "score": 0.62
      }
    },
    {
      "job_id": "sym_AAPL_y_will_swing_low_10m_0.20",
      "target": "y_will_swing_low_10m_0.20",
      "symbol": "AAPL",
      "route": "ROUTE_SYMBOL_SPECIFIC",
      "training_type": "symbol_specific",
      "model_families": ["lightgbm"],
      "priority": 80,
      "reason": "Local strong, CS weak for this symbol",
      "metadata": {
        "local_state": "STRONG",
        "sample_size": 8000,
        "score": 0.66
      }
    }
  ],
  "summary": {
    "by_route": {
      "ROUTE_CROSS_SECTIONAL": 5,
      "ROUTE_SYMBOL_SPECIFIC": 30,
      "ROUTE_BOTH": 7
    },
    "by_type": {
      "cross_sectional": 5,
      "symbol_specific": 37
    },
    "by_priority": {
      90: 5,
      80: 20,
      70: 17
    },
    "total_cs_jobs": 5,
    "total_symbol_jobs": 37,
    "total_blocked": 0
  }
}
```

### Field Descriptions

**Metadata:**
- `generated_at`: ISO 8601 timestamp when plan was generated
- `run_id`: Unique run identifier (timestamp-based)
- `git_commit`: Git commit hash when plan was generated
- `config_hash`: Hash of routing config used (for reproducibility)
- `routing_plan_path`: Path to routing plan that generated this training plan
- `metrics_snapshot`: Path to routing candidates snapshot used
- `total_jobs`: Total number of jobs in plan
- `model_families`: Default model families (can be overridden per job)

**Job Fields:**
- `job_id`: Unique identifier (e.g., `cs_<target>`, `sym_<target>_<symbol>`)
- `target`: Target name
- `symbol`: Symbol name (`null` for cross-sectional jobs)
- `route`: Routing decision (`ROUTE_CROSS_SECTIONAL`, `ROUTE_SYMBOL_SPECIFIC`, `ROUTE_BOTH`, `ROUTE_EXPERIMENTAL_ONLY`, `ROUTE_BLOCKED`)
- `training_type`: `"cross_sectional"` or `"symbol_specific"`
- `model_families`: List of model families to train (can override default)
- `priority`: Job priority (higher = more important)
- `reason`: Human-readable explanation for why this job exists
- `metadata`: Additional context (sample sizes, states, scores, etc.)

**Summary:**
- `by_route`: Count of jobs by route
- `by_type`: Count of jobs by training type
- `by_priority`: Count of jobs by priority bucket
- `total_cs_jobs`: Total cross-sectional jobs
- `total_symbol_jobs`: Total symbol-specific jobs
- `total_blocked`: Total blocked jobs (if included)

## Derived Artifacts (Views)

These are **generated views** over the master plan. They are for humans, dashboards, and external schedulers. The training phase should **never** read these files.

### By Target

**Path Pattern:** `METRICS/training_plan/by_target/<target>.json`

**Contents:**
```jsonc
{
  "target": "y_will_swing_low_10m_0.20",
  "jobs": [
    // All CS + symbol-specific jobs for this target
  ]
}
```

**Use Cases:**
- Researcher opens one file to see "what will happen" for a particular target
- Easier diffing between runs for a single target
- Target-level debugging

### By Symbol

**Path Pattern:** `METRICS/training_plan/by_symbol/<symbol>.json`

**Contents:**
```jsonc
{
  "symbol": "AAPL",
  "jobs": [
    // All jobs touching AAPL (multi-target)
  ]
}
```

**Use Cases:**
- "What are we doing to AAPL this run?"
- Symbol-level debugging/perf dashboards
- Symbol-specific analysis

### By Training Type

**Path Pattern:** `METRICS/training_plan/by_type/<type>.json`

**Examples:**
- `by_type/cross_sectional.json`
- `by_type/symbol_specific.json`

**Contents:**
```jsonc
{
  "training_type": "cross_sectional",
  "jobs": [
    // All jobs of this type
  ]
}
```

**Use Cases:**
- "How many CS jobs do we have?"
- Type-level scheduling
- Resource allocation by type

### By Route

**Path Pattern:** `METRICS/training_plan/by_route/<route>.json`

**Examples:**
- `by_route/ROUTE_CROSS_SECTIONAL.json`
- `by_route/ROUTE_BOTH.json`
- `by_route/ROUTE_EXPERIMENTAL_ONLY.json`

**Contents:**
```jsonc
{
  "route": "ROUTE_EXPERIMENTAL_ONLY",
  "jobs": [
    // All jobs with this route
  ]
}
```

**Use Cases:**
- "How many experimental jobs do we actually have?"
- Route-level analysis
- Experimental lane monitoring

### Markdown Summary

**Path:** `METRICS/training_plan/MASTER_TRAINING_PLAN_SUMMARY.md`

**Contents:**
- Counts by training_type, route, priority bucket
- Top N jobs by priority (grouped by target)
- Links to per-target/per-symbol views
- Overall statistics

**Use Cases:**
- Human-readable overview
- Quick sanity checks
- Reports and presentations

## How Training Phase Should Consume

### Consumer API

In `training_plan_consumer.py`:

```python
def load_master_training_plan(path: Path) -> MasterPlan:
    """Load master plan from disk. Only reads master_training_plan.json."""
    ...

def get_cs_jobs(plan: MasterPlan) -> list[Job]:
    """Get all cross-sectional jobs."""
    ...

def get_symbol_jobs(plan: MasterPlan) -> list[Job]:
    """Get all symbol-specific jobs."""
    ...

def get_jobs_for_target(plan: MasterPlan, target: str) -> list[Job]:
    """Get all jobs for a target."""
    ...

def get_jobs_for_symbol(plan: MasterPlan, symbol: str) -> list[Job]:
    """Get all jobs for a symbol."""
    ...
```

### Usage in Training Phase

```python
# Load master plan (only file training phase reads)
plan = load_master_training_plan(
    Path("METRICS/training_plan/master_training_plan.json")
)

# Filter targets for CS training
cs_jobs = get_cs_jobs(plan)
cs_targets = sorted({job.target for job in cs_jobs})

# Filter symbols per target for symbol-specific training (future)
symbol_jobs = get_symbol_jobs(plan)
symbols_by_target = {}
for job in symbol_jobs:
    if job.target not in symbols_by_target:
        symbols_by_target[job.target] = []
    symbols_by_target[job.target].append(job.symbol)
```

**Important:** The training phase should **only read `master_training_plan.json`**. All other files are for humans and tooling.

## Implementation Status

**Current:**
- ✅ Training plan saved as `training_plan.json`
- ✅ Includes all job specs and summary
- ⚠️ Not yet formalized as "master" plan

**Planned:**
- Rename/formalize as `master_training_plan.json`
- Generate derived views (by_target, by_symbol, by_type, by_route)
- Update consumer to only read master plan
- Update documentation

See `IMPLEMENTATION_STATUS.md` for details.

## Directory Structure

```
METRICS/training_plan/
├── master_training_plan.json  # ⭐ Single source of truth
├── training_plan.json         # Convenience mirror
├── training_plan.yaml         # YAML format
├── training_plan.md           # Human-readable report
├── MASTER_TRAINING_PLAN_SUMMARY.md  # Summary report
│
├── by_target/                 # Derived views (future)
│   └── <target>.json
├── by_symbol/                 # Derived views (future)
│   └── <symbol>.json
├── by_type/                   # Derived views (future)
│   ├── cross_sectional.json
│   └── symbol_specific.json
└── by_route/                  # Derived views (future)
    ├── ROUTE_CROSS_SECTIONAL.json
    ├── ROUTE_SYMBOL_SPECIFIC.json
    ├── ROUTE_BOTH.json
    └── ROUTE_EXPERIMENTAL_ONLY.json
```

## Benefits

1. **Single Source of Truth**: Training phase only reads one file, reducing complexity
2. **Human-Friendly Views**: Derived artifacts make it easy to explore the plan
3. **Tooling Integration**: External schedulers can read specific views without parsing the full plan
4. **Debugging**: Per-target/per-symbol views make debugging easier
5. **Reproducibility**: Master plan includes all metadata needed to reproduce decisions

## See Also

- `ARCHITECTURE.md` - Complete system architecture
- `IMPLEMENTATION_STATUS.md` - What's implemented vs. planned
- `README.md` - User-facing guide
