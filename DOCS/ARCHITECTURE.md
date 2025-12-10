# Documentation Architecture

This document defines the 4-tier documentation hierarchy for FoxML Core.

## Tier A: Executive / High-Level

**Purpose**: First impression, orientation, business context

**Location**: `docs/00_executive/`

**Contents**:
- README.md (root) - Project overview, licensing, contact
- QUICKSTART.md - Get running in 5 minutes
- ARCHITECTURE_OVERVIEW.md - System architecture at a glance
- GETTING_STARTED.md - Onboarding guide

**Audience**: New users, stakeholders, decision makers

**Style**: Clear, concise, minimal jargon

---

## Tier B: Tutorials / Walkthroughs

**Purpose**: Step-by-step guides for common tasks

**Location**: `docs/01_tutorials/`

**Contents**:
- setup/
  - INSTALLATION.md
  - ENVIRONMENT_SETUP.md
  - GPU_SETUP.md
- pipelines/
  - FIRST_PIPELINE_RUN.md
  - DATA_PROCESSING_WALKTHROUGH.md
  - FEATURE_ENGINEERING_TUTORIAL.md
- training/
  - MODEL_TRAINING_GUIDE.md
  - WALKFORWARD_VALIDATION.md
  - FEATURE_SELECTION_TUTORIAL.md
- configuration/
  - CONFIG_BASICS.md
  - CONFIG_EXAMPLES.md
  - ADVANCED_CONFIG.md

**Audience**: Users following a specific workflow

**Style**: Sequential, example-driven, copy-paste ready

---

## Tier C: Core Reference Docs

**Purpose**: Complete technical reference for daily use

**Location**: `docs/02_reference/`

**Contents**:
- api/
  - MODULE_REFERENCE.md
  - CLI_REFERENCE.md
  - CONFIG_SCHEMA.md
- data/
  - DATA_FORMAT_SPEC.md
  - COLUMN_REFERENCE.md
  - DATA_SANITY_RULES.md
- models/
  - MODEL_CATALOG.md
  - MODEL_CONFIG_REFERENCE.md
  - TRAINING_PARAMETERS.md
- systems/
  - PIPELINE_REFERENCE.md
- configuration/
  - CONFIG_LOADER_API.md
  - CONFIG_OVERLAYS.md
  - ENVIRONMENT_VARIABLES.md

**Audience**: Developers, operators, integrators

**Style**: Complete, precise, searchable

---

## Tier D: Deep Technical Appendices

**Purpose**: Research notes, design rationale, advanced topics

**Location**: `docs/03_technical/`

**Contents**:
- research/
  - LEAKAGE_ANALYSIS.md
  - FEATURE_IMPORTANCE_METHODOLOGY.md
  - TARGET_DISCOVERY.md
  - VALIDATION_METHODOLOGY.md
- design/
  - ARCHITECTURE_DEEP_DIVE.md
  - (moved to internal/)
- benchmarks/
  - PERFORMANCE_METRICS.md
  - MODEL_COMPARISONS.md
  - DATASET_SIZING.md
- fixes/
  - KNOWN_ISSUES.md
  - BUG_FIXES.md
  - MIGRATION_NOTES.md
- roadmaps/
  - ALPHA_ENHANCEMENT_ROADMAP.md
  - FUTURE_WORK.md

**Audience**: Contributors, researchers, advanced users

**Style**: Detailed, technical, may include equations

---

## Navigation Structure

### Root README.md
- Links to QUICKSTART
- Links to ARCHITECTURE_OVERVIEW
- Links to GETTING_STARTED
- Links to full documentation index

### Documentation Index (`docs/INDEX.md`)
- Complete table of contents
- Search-friendly organization
- Links to all tiers

### Cross-Linking Rules
- Every doc links back to relevant tier index
- Every doc links to related docs in "See Also"
- Code references link to source files
- Config examples link to schema

---

## Maintenance Policy

### When to Update Docs
- New features: Add to appropriate tier
- API changes: Update reference docs immediately
- Breaking changes: Update migration notes + tutorials
- Bug fixes: Document in technical/fixes

### Deprecation Process
1. Mark deprecated in reference doc
2. Add migration path
3. Update tutorials to use new approach
4. Remove after 2 major versions

### New Module Documentation
- Must include: Purpose, API, Examples, See Also
- Place in appropriate tier based on audience
- Link from relevant index

### Style Guide
- **Tense**: Present tense
- **Voice**: Active voice
- **Tone**: Precise, minimal, engineer-to-engineer
- **Formatting**: Consistent headers, code blocks, bullets
- **Code**: Always include working examples

---

## Migration Status

This architecture is being implemented. Existing docs are being categorized and moved into this structure.

See `docs/MIGRATION_PLAN.md` for current status.

