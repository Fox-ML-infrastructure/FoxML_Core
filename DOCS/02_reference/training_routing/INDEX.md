# Training Routing & Plan System - Documentation Index

**Quick navigation guide for the training routing and plan system documentation.**

## Quick Start

**New to the system?** Start here:
1. `README.md` - User-facing guide with quick start
2. `SUMMARY.md` - Quick reference

## Architecture & Design

**Want to understand how it works?** See the internal documentation for architecture details.

## Implementation Status

**What's done vs. what's planned?** See the internal documentation for implementation status.

## Integration

**How does it integrate with training?** See the internal documentation for integration details.

## Reference

**Need specific details?**
1. `CONFIG/training_config/routing_config.yaml` - Configuration schema
2. See the internal documentation for implementation details

## Document Map

### User-Facing (Operational)

```
DOCS/02_reference/training_routing/
├── INDEX.md                    # This file - navigation guide
├── README.md                    # Main user guide (start here)
├── QUICK_START.md               # Quick start guide
├── END_TO_END_FLOW.md           # End-to-end flow guide
├── ONE_COMMAND_TRAINING.md      # One-command examples
├── TWO_STAGE_TRAINING.md        # 2-stage training guide
├── SUMMARY.md                   # Quick reference
├── READY_TO_USE.md              # Status confirmation
└── SETUP_COMPLETE.md            # Setup confirmation
```


## By Use Case

### "I want to use the routing system"
→ `README.md` - Quick start and usage guide

### "I want to understand the architecture"
→ See the internal documentation for architecture details
→ `SUMMARY.md` - Quick overview

### "I want to know what's implemented"
→ `SUMMARY.md` - Quick overview
→ See the internal documentation for detailed status

### "I want to integrate it into my code"
→ See the internal documentation for integration details

### "I want to configure routing decisions"
→ `README.md` - See "Configuration" section
→ `CONFIG/training_config/routing_config.yaml` - Config file

### "I want to understand the training plan structure"
→ See the internal documentation for master plan details

### "I'm debugging an issue"
→ See the internal documentation for known issues and fixes

## Key Concepts

**Routing Plan** - Decisions about where to train (CS, symbol-specific, both, experimental, blocked)
- Location: `METRICS/routing_plan/`

**Training Plan** - Actionable job specifications derived from routing decisions
- Location: `globals/training_plan/` (primary), `METRICS/training_plan/` (legacy fallback)
- Master file: `master_training_plan.json` (single source of truth)

**Metrics Aggregation** - Collecting metrics from feature selection, stability, leakage detection
- Location: `METRICS/routing_candidates.parquet` (or `.csv`)

**Training Plan Consumption** - Filtering training based on the plan
- Implementation: `TRAINING/orchestration/training_plan_consumer.py`

## Implementation Files

**Core Implementation:**
- `TRAINING/orchestration/metrics_aggregator.py` - Metrics collection
- `TRAINING/orchestration/training_router.py` - Routing decisions
- `TRAINING/orchestration/training_plan_generator.py` - Job spec generation
- `TRAINING/orchestration/training_plan_consumer.py` - Plan consumption
- `TRAINING/orchestration/routing_integration.py` - Integration hooks
- `TRAINING/orchestration/intelligent_trainer.py` - Training orchestrator

**Configuration:**
- `CONFIG/training_config/routing_config.yaml` - Routing policy

## Status Summary

**✅ Fully Implemented:**
- Metrics aggregation (Parquet → CSV fallback)
- Routing plan generation
- Training plan generation
- CS training filtering
- Automatic integration

**⚠️ Planned / Future:**
- Symbol-specific execution filtering
- Model-family-level filtering
- Master plan derived views
- Advanced routing logic enhancements

See the internal documentation for details.
