# Training Routing & Plan System - Internal Documentation

**Planning, architecture, and implementation details for the training routing system.**

## Overview

This directory contains internal documentation for the training routing and plan system:
- Architecture and design decisions
- Implementation status and planning
- Technical details and integration guides
- Bug fixes and hardening notes

**For operational/user-facing docs**, see: `DOCS/02_reference/training_routing/`

## Contents

### Architecture & Design

- `ARCHITECTURE.md` - Complete end-to-end architecture
- `MASTER_TRAINING_PLAN.md` - Master plan structure and design principles
- `ROUTING_SYSTEM_SUMMARY.md` - Implementation details

### Implementation Status

- `IMPLEMENTATION_STATUS.md` - Detailed breakdown of implemented vs. TODO features
- `INTEGRATION_SUMMARY.md` - Integration with training phase (technical details)
- `PHASE3_INTEGRATION.md` - Phase 3 (sequential models) integration details

### Internal Tracking

- `ERRORS_FIXED.md` - Known issues and fixes
- `HARDENING_SUMMARY.md` - Error handling and defensive programming measures
- `KNOWN_ISSUES.md` - Current issues and edge cases

### Legacy/Reference

- `SYSTEM_SUMMARY.md` - System summary (for reference)
- `README_ROUTING.md` - Legacy routing README (for reference)

## User-Facing Documentation

For operational guides and quick starts, see:
- `DOCS/02_reference/training_routing/README.md` - Main user guide
- `DOCS/02_reference/training_routing/QUICK_START.md` - Quick start guide
- `DOCS/02_reference/training_routing/END_TO_END_FLOW.md` - End-to-end flow
- `DOCS/02_reference/training_routing/ONE_COMMAND_TRAINING.md` - Usage examples
- `DOCS/02_reference/training_routing/TWO_STAGE_TRAINING.md` - 2-stage training guide
- `DOCS/02_reference/training_routing/SUMMARY.md` - Quick reference

## Key Concepts

**Routing Plan** - Decisions about where to train (CS, symbol-specific, both, experimental, blocked)
- Location: `METRICS/routing_plan/`

**Training Plan** - Actionable job specifications derived from routing decisions
- Location: `METRICS/training_plan/`
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
