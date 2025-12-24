# Internal Documentation Index

**Last Updated**: 2025-12-08  
**Purpose**: Comprehensive index of all internal planning, research, analysis, and fixes documentation.

---

## Quick Navigation

- [Planning Documents](#planning-documents) - Future work, design, implementation plans
- [Research Documents](#research-documents) - Research notes, methodology, findings
- [Analysis Documents](#analysis-documents) - System analysis, architecture reviews
- [Fixes & Bug Reports](#fixes--bug-reports) - Bug fixes, issue resolutions, post-mortems

---

## Planning Documents

### Distributed Computing
- [Distributed Training Architecture](planning/DISTRIBUTED_TRAINING_ARCHITECTURE.md) - **NEW**: Architecture for distributing model training across compute nodes (replaces sequential execution)

### Core Learning Systems

#### 0. Adaptive Intelligence Architecture (Summary)
**File**: `planning/ADAPTIVE_INTELLIGENCE_ARCHITECTURE.md`  
**Status**: Design Phase  
**Size**: 691 lines

**Description**: Comprehensive architectural overview synthesizing all adaptive intelligence components. This is the **entry point** for understanding the complete system architecture.

**Key Sections**:
- System Overview (high-level architecture diagrams)
- Component Architecture (CILS, ARPO, CLEARFRAME, iDiffODE, Orchestrator)
- Data Flow Architecture (training, portfolio, conditional, continuous-time flows)
- Integration Architecture (how all components work together)
- Implementation Architecture (module structure, data flow between modules)
- Key Algorithms & Methods (formulas, priority scoring, partial correlation, etc.)
- Implementation Phases (16-week roadmap)
- Expected Outcomes (performance, intelligence, operational benefits)
- Research Questions & Future Work
- References & Glossary

**Related**:
- Synthesizes: `CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md`, `ADAPTIVE_REALTIME_PORTFOLIO_OPTIMIZATION.md`, `INTEGRATED_LEARNING_FEEDBACK_LOOP.md`
- **Start here** to understand the complete adaptive intelligence framework

---

#### 1. Continuous Integrated Learning System (CILS)
**File**: `planning/CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md`  
**Status**: Design Phase  
**Size**: 42K, 1,222 lines

**Description**: Adaptive meta-layer that learns from training runs, leakage patterns, and feature performance to continuously improve decision-making thresholds, feature selection strategies, and leakage detection sensitivity.

**Key Sections**:
- Current state analysis (existing intelligence pipeline)
- System architecture (Learning Engine, Pattern Recognition, Adaptive Thresholds)
- Learning mechanisms (supervised learning, pattern recognition, strategy optimization)
- Implementation roadmap (5 phases, 10 weeks)
- **iDiffODE Framework** (Section 11.4): Neural ODE-Diffusion framework for time-series intelligence

**Related**:
- Links to: `INTEGRATED_LEARNING_FEEDBACK_LOOP.md`, `ADAPTIVE_REALTIME_PORTFOLIO_OPTIMIZATION.md`
- References: Intelligence Layer, Leakage Analysis, Target Ranking

---

#### 2. Adaptive Real-Time Portfolio Optimization (ARPO)
**File**: `planning/ADAPTIVE_REALTIME_PORTFOLIO_OPTIMIZATION.md`  
**Status**: Research & Design Phase  
**Size**: 42K, 1,218 lines

**Description**: Framework for adaptive real-time portfolio optimization using iDiffODE architecture. Learns continuously from profit/loss metrics, market conditions, and trading outcomes to optimize portfolio allocation, position sizing, and risk management in real time.

**Key Sections**:
- Continuous portfolio dynamics (Neural ODE for portfolio weights)
- Real-time P&L learning (position sizing adaptation)
- Diffusion-based regime refinement
- Invertible portfolio mapping
- Implementation roadmap (4 phases, 10 weeks)

**Related**:
- Links to: `INTEGRATED_LEARNING_FEEDBACK_LOOP.md`, `CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md`
- Uses: iDiffODE framework (same as CILS)

---

#### 3. Integrated Learning Feedback Loop
**File**: `planning/INTEGRATED_LEARNING_FEEDBACK_LOOP.md`  
**Status**: Design Phase  
**Size**: 36K, 991 lines

**Description**: Links CILS ↔ ARPO ↔ Training Pipeline. Enables automated training scheduling based on learned insights from both systems. Propagates config updates and manages feedback loops.

**Key Sections**:
- Unified learning database schema
- Retraining signal generation (from CILS and ARPO)
- Priority scoring algorithm (combines training-side and live trading insights)
- Automated scheduling system (resource-aware)
- Config update propagation
- Feedback loops (CILS ↔ Training, ARPO ↔ Training, cross-system)

**Key Features**:
- Automated training scheduling based on learned data
- Priority scoring: 25% CILS, 35% ARPO, 25% ROI, 15% urgency
- Resource-aware scheduling (GPU memory, CPU, budget constraints)
- Config updates automatically propagated to training pipeline

**Related**:
- Links to: `CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md`, `ADAPTIVE_REALTIME_PORTFOLIO_OPTIMIZATION.md`
- Integrates: Training Pipeline, CILS, ARPO

---

### Feature Engineering & Registry

#### 4. Feature Registry Design
**File**: `planning/FEATURE_REGISTRY_DESIGN.md`  
**Status**: Design Document

**Description**: Design for feature registry system that tracks feature metadata, temporal rules, and leakage prevention.

**Related**: `FEATURE_REGISTRY_PHASE1_COMPLETE.md`, `FEATURE_REGISTRY_PHASE2_COMPLETE.md`, `FEATURE_REGISTRY_PHASE3_COMPLETE.md`, `FEATURE_REGISTRY_PHASE4_COMPLETE.md`, `FEATURE_REGISTRY_COMPLETE.md`

---

#### 5. Feature Registry Phase Completion Docs
**Files**: 
- `planning/FEATURE_REGISTRY_PHASE1_COMPLETE.md`
- `planning/FEATURE_REGISTRY_PHASE2_COMPLETE.md`
- `planning/FEATURE_REGISTRY_PHASE3_COMPLETE.md`
- `planning/FEATURE_REGISTRY_PHASE4_COMPLETE.md`
- `planning/FEATURE_REGISTRY_COMPLETE.md`

**Description**: Phase-by-phase completion documentation for feature registry implementation.

---

### Target Ranking & Selection

#### 6. Target Ranking Selection Integration
**File**: `planning/TARGET_RANKING_SELECTION_INTEGRATION.md`  
**Status**: Implementation Plan

**Description**: Plan for integrating target ranking and feature selection into training pipeline.

**Related**: `MAIN_TRAINING_SCRIPT_OUTLINE.md`, `IMPLEMENTATION_STATUS.md`

---

### Configuration & Architecture

#### 7. Config Refactor Plan
**File**: `planning/CONFIG_REFACTOR_PLAN.md`  
**Status**: Implementation Plan

**Description**: Plan for modular config system refactor (experiment-level, module-level, system-level configs).

**Related**: Current modular config system in `CONFIG/`

---

#### 8. Path Configuration Guidelines
**File**: `planning/PATH_CONFIGURATION_GUIDELINES.md`  
**Status**: Guidelines

**Description**: Guidelines for path configuration and folder structure.

---

#### 9. NVLink Ready Architecture
**File**: `planning/NVLINK_READY_ARCHITECTURE.md`  
**Status**: Design Document

**Description**: Architecture considerations for NVLink compatibility and multi-GPU training.

---

### Leakage & Safety

#### 10. Adaptive Leakage Controller
**File**: `planning/ADAPTIVE_LEAKAGE_CONTROLLER.md`  
**Status**: Design Document

**Description**: Adaptive leakage detection and prevention system.

**Related**: `CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md` (Section on leakage learning)

---

### Performance & Optimization

#### 11. Performance Optimization Plan
**File**: `planning/PERFORMANCE_OPTIMIZATION_PLAN.md`  
**Status**: Implementation Plan

**Description**: Plan for performance optimizations across the system.

---

#### 12. Pressure Test Implementation Roadmap
**File**: `planning/PRESSURE_TEST_IMPLEMENTATION_ROADMAP.md`  
**Status**: Implementation Plan

**Description**: Roadmap for pressure testing system components.

**Related**: `PRESSURE_TEST_UPGRADES.md`

---

#### 13. Pressure Test Upgrades
**File**: `planning/PRESSURE_TEST_UPGRADES.md`  
**Status**: Implementation Plan

**Description**: Upgrades and improvements for pressure testing.

---

### Trading & Deployment

#### 14. Enhanced Rebalancing Trading Plan
**File**: `planning/ENHANCED_REBALANCING_TRADING_PLAN.md`  
**Status**: Design Document

**Description**: Enhanced rebalancing strategies for live trading.

---

#### 15. SystemD Deployment Plan
**File**: `planning/SYSTEMD_DEPLOYMENT_PLAN.md`  
**Status**: Implementation Plan

**Description**: Plan for deploying system as systemd service.

---

### Implementation Status

#### 16. Implementation Status
**File**: `planning/IMPLEMENTATION_STATUS.md`  
**Status**: Status Document

**Description**: Current implementation status of various features and components.

---

#### 17. Main Training Script Outline
**File**: `planning/MAIN_TRAINING_SCRIPT_OUTLINE.md`  
**Status**: Design Document

**Description**: Outline for main training script architecture.

---

#### 18. Import Audit and Structure
**File**: `planning/IMPORT_AUDIT_AND_STRUCTURE.md`  
**Status**: Audit Document

**Description**: Audit of imports and code structure.

---

#### 19. Alpha Enhancement Roadmap
**File**: `planning/ALPHA_ENHANCEMENT_ROADMAP.md`  
**Status**: Roadmap

**Description**: Roadmap for alpha generation enhancements.

---

## Research Documents

### Feature Selection & Models

#### 20. Complete Feature Selection Models
**File**: `research/COMPLETE_FEATURE_SELECTION_MODELS.md`  
**Description**: Complete list of feature selection models and methodologies.

---

#### 21. Additional Feature Selection Models
**File**: `research/ADDITIONAL_FEATURE_SELECTION_MODELS.md`  
**Description**: Additional feature selection models beyond core set.

---

#### 22. Additional Models Quickstart
**File**: `research/ADDITIONAL_MODELS_QUICKSTART.md`  
**Description**: Quickstart guide for additional models.

---

#### 23. All Models Enabled
**File**: `research/ALL_MODELS_ENABLED.md`  
**Description**: Documentation for enabling all available models.

---

#### 24. Model Enabling Recommendations
**File**: `research/MODEL_ENABLING_RECOMMENDATIONS.md`  
**Description**: Recommendations for which models to enable and when.

---

### Feature Importance & Analysis

#### 25. Feature Importance Fix
**File**: `research/FEATURE_IMPORTANCE_FIX.md`  
**Description**: Fixes and improvements to feature importance calculation.

---

#### 26. Importance R2 Weighting
**File**: `research/IMPORTANCE_R2_WEIGHTING.md`  
**Description**: Methodology for weighting feature importance by R² scores.

---

#### 27. Importance Score Interpretation
**File**: `research/IMPORTANCE_SCORE_INTERPRETATION.md`  
**Description**: Guide to interpreting feature importance scores.

---

### Target Discovery & Analysis

#### 28. Target Discovery Update
**File**: `research/TARGET_DISCOVERY_UPDATE.md`  
**Description**: Updates to target discovery methodology.

---

#### 29. Target Recommendations
**File**: `research/TARGET_RECOMMENDATIONS.md`  
**Description**: Recommendations for target selection and prioritization.

---

#### 30. Target Model Pipeline Analysis
**File**: `research/TARGET_MODEL_PIPELINE_ANALYSIS.md`  
**Description**: Analysis of target-to-model pipeline.

---

#### 31. Target to Feature Workflow
**File**: `research/TARGET_TO_FEATURE_WORKFLOW.md`  
**Description**: Workflow for mapping targets to features.

---

### GPU & Performance

#### 32. GPU Setup Multi Model
**File**: `research/GPU_SETUP_MULTI_MODEL.md`  
**Description**: GPU setup for multi-model training.

---

### Dataset & Sizing

#### 33. Dataset Sizing Strategy
**File**: `research/DATASET_SIZING_STRATEGY.md`  
**Description**: Strategy for dataset sizing and sampling.

---

## Analysis Documents

### Trading Analysis

#### 34. Intraday Trading Analysis
**File**: `analysis/INTRADAY_TRADING_ANALYSIS.md`  
**Description**: Analysis of intraday trading patterns and strategies.

---

### Architecture Analysis

#### 35. Optimization Architecture
**File**: `analysis/OPTIMIZATION_ARCHITECTURE.md`  
**Description**: Analysis of optimization architecture and design.

---

#### 36. Optimization Engine Analysis
**File**: `analysis/OPTIMIZATION_ENGINE_ANALYSIS.md`  
**Description**: Deep dive into optimization engine architecture.

---

### Integration Analysis

#### 37. Yahoo Finance Integration
**File**: `analysis/YAHOO_FINANCE_INTEGRATION.md`  
**Description**: Analysis of Yahoo Finance data integration.

---

## Fixes & Bug Reports

### Leakage Fixes

#### 38. Final Leakage Summary
**File**: `fixes/FINAL_LEAKAGE_SUMMARY.md`  
**Description**: Final summary of leakage detection and fixes.

---

#### 39. Leakage Fixed Next Steps
**File**: `fixes/LEAKAGE_FIXED_NEXT_STEPS.md`  
**Description**: Next steps after leakage fixes.

---

#### 40. Leakage (General)
**File**: `fixes/leakage.md`  
**Description**: General leakage detection and prevention documentation.

---

#### 41. Deeper Leak Fix
**File**: `fixes/DEEPER_LEAK_FIX.md`  
**Description**: Deep fixes for leakage issues.

---

#### 42. Target Is Leaked
**File**: `fixes/TARGET_IS_LEAKED.md`  
**Description**: Analysis of cases where target itself is leaked.

---

#### 43. Target Leakage Clarification
**File**: `fixes/TARGET_LEAKAGE_CLARIFICATION.md`  
**Description**: Clarification of target leakage concepts.

---

#### 44. Forward Return 20D Leakage Analysis
**File**: `fixes/FWD_RET_20D_LEAKAGE_ANALYSIS.md`  
**Description**: Specific analysis of 20-day forward return leakage.

---

#### 45. Round 3 Temporal Overlap Fix
**File**: `fixes/ROUND3_TEMPORAL_OVERLAP_FIX.md`  
**Description**: Fix for temporal overlap issues (round 3).

---

#### 46. Validation Leak Audit
**File**: `fixes/VALIDATION_LEAK_AUDIT.md`  
**Description**: Audit of validation methodology for leakage.

---

### Code Quality & Performance

#### 47. Code Review Bugs
**File**: `fixes/CODE_REVIEW_BUGS.md`  
**Description**: Bugs found during code review.

---

#### 48. Avoid Long Runs
**File**: `fixes/AVOID_LONG_RUNS.md`  
**Description**: Strategies to avoid long-running processes.

---

#### 49. Quick Start Clean Baseline
**File**: `fixes/QUICK_START_CLEAN_BASELINE.md`  
**Description**: Guide for quick start with clean baseline.

---

## Other Internal Documents

### Configuration

#### 50. Config Locations Audit
**File**: `CONFIG_LOCATIONS_AUDIT.md`  
**Description**: Audit of configuration file locations.

---

### Legal

#### 51. Legal Assessment
**File**: `LEGAL_ASSESSMENT.md`  
**Description**: Legal assessment and compliance documentation.

---

### Templates

#### 52. Changelog Template
**File**: `CHANGELOG_TEMPLATE.md`  
**Description**: Template for changelog entries.

---

### Logs

#### 53. Journal Log
**File**: `journallog.md`  
**Description**: Journal log of development activities.

---

## Document Relationships

### Learning Systems Trilogy

The three core learning system documents form an integrated suite:

1. **CILS** (Training-side learning)
   - Learns from training runs
   - Optimizes thresholds, strategies, feature patterns
   - Feeds insights to Training Pipeline

2. **ARPO** (Portfolio-side learning)
   - Learns from live trading P&L
   - Optimizes position sizing, allocation, risk limits
   - Feeds insights to Training Pipeline

3. **Integrated Feedback Loop** (Orchestration)
   - Links CILS ↔ ARPO ↔ Training Pipeline
   - Automated scheduling based on learned data
   - Config update propagation
   - Priority scoring and resource management

**All three use iDiffODE framework** (Neural ODE-Diffusion-Invertible ResNet) for continuous-time modeling.

---

### Feature Registry Suite

- `FEATURE_REGISTRY_DESIGN.md` - Original design
- `FEATURE_REGISTRY_PHASE1_COMPLETE.md` through `FEATURE_REGISTRY_PHASE4_COMPLETE.md` - Phase completion docs
- `FEATURE_REGISTRY_COMPLETE.md` - Final completion summary

---

### Leakage Documentation

Leakage-related documents span multiple categories:
- **Planning**: `ADAPTIVE_LEAKAGE_CONTROLLER.md`, `CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md` (leakage learning)
- **Fixes**: Multiple leakage fix documents in `fixes/`
- **Research**: See `DOCS/03_technical/research/LEAKAGE_ANALYSIS.md` (not in internal)

---

## Key Concepts Index

### iDiffODE Framework
- **Definition**: Neural ODE-Diffusion-Invertible ResNet for continuous-time modeling
- **Used in**: CILS (Section 11.4), ARPO (entire document)
- **Components**: Transformer Encoder, Neural ODE, Diffusion Model, Invertible ResNet
- **Applications**: Time-series intelligence, portfolio optimization

### CILS (Continuous Integrated Learning System)
- **Purpose**: Learn from training runs to improve thresholds, strategies, feature patterns
- **Key Features**: Adaptive thresholds, pattern recognition, strategy optimization
- **Status**: Design Phase
- **Timeline**: 5 phases, 10 weeks

### ARPO (Adaptive Real-Time Portfolio Optimization)
- **Purpose**: Learn from live trading P&L to optimize portfolio allocation
- **Key Features**: Real-time P&L learning, regime adaptation, adaptive risk management
- **Status**: Research & Design Phase
- **Timeline**: 4 phases, 10 weeks

### Integrated Feedback Loop
- **Purpose**: Orchestrate learning between CILS, ARPO, and Training Pipeline
- **Key Features**: Automated scheduling, priority scoring, config propagation
- **Status**: Design Phase
- **Timeline**: 5 phases, 10 weeks

### Priority Scoring
- **Formula**: 25% CILS + 35% ARPO + 25% ROI + 15% urgency
- **Rationale**: ARPO weighted higher (live money), ROI for efficiency, urgency for timeliness
- **Implementation**: `compute_training_priority()` in `INTEGRATED_LEARNING_FEEDBACK_LOOP.md`

---

## Status Summary

### Active Development
- **CILS**: Design Phase
- **ARPO**: Research & Design Phase
- **Integrated Feedback Loop**: Design Phase

### Completed
- **Feature Registry**: All phases complete
- **Target Ranking Integration**: Implemented
- **Config Refactor**: Implemented (modular config system)

### Maintenance
- **Leakage Fixes**: Ongoing
- **Performance Optimization**: Ongoing
- **Code Quality**: Ongoing

---

## Quick Reference

### Most Important Documents
1. **`ADAPTIVE_INTELLIGENCE_ARCHITECTURE.md`** - **START HERE**: Complete architecture overview
2. `CONTINUOUS_INTEGRATED_LEARNING_SYSTEM.md` - Training-side learning (CILS)
3. `ADAPTIVE_REALTIME_PORTFOLIO_OPTIMIZATION.md` - Portfolio-side learning (ARPO)
4. `INTEGRATED_LEARNING_FEEDBACK_LOOP.md` - System integration & orchestration
5. `IMPLEMENTATION_STATUS.md` - Current status
6. `FEATURE_REGISTRY_COMPLETE.md` - Feature registry completion

### For New Features
- Start with planning documents in `planning/`
- Check `IMPLEMENTATION_STATUS.md` for current state
- Review related research in `research/`

### For Bug Fixes
- Check `fixes/` directory for similar issues
- Review `FINAL_LEAKAGE_SUMMARY.md` for leakage context
- Check `CODE_REVIEW_BUGS.md` for known issues

### For Architecture Decisions
- Review `NVLINK_READY_ARCHITECTURE.md` for GPU considerations
- Check `OPTIMIZATION_ARCHITECTURE.md` for optimization design
- Review `PATH_CONFIGURATION_GUIDELINES.md` for structure

---

## Document Statistics

- **Total Internal Documents**: 54
- **Planning Documents**: 20
- **Research Documents**: 14
- **Analysis Documents**: 4
- **Fixes Documents**: 12
- **Other Documents**: 4

---

**Last Updated**: 2025-12-08  
**Maintainer**: Internal Documentation System

