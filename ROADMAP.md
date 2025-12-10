# Fox ML Infrastructure — Strategic Roadmap

FoxML Core is the core research and training engine of Fox ML Infrastructure.

Direction and priorities for ongoing development. Timelines are aspirational and subject to change.

---

## Development Philosophy

FoxML Core is maintained with an **enterprise reliability mindset**:
* Critical issues are fixed immediately.
* Documentation and legal compliance are treated as first-class deliverables.
* Stability always precedes new features.
* GPU acceleration, orchestration, and model tooling will remain backwards-compatible.
* **UI integration approach** — UI is intentionally decoupled from the core. FoxML Core provides Python programmatic interfaces and integration points so teams can plug into their existing dashboards, monitoring tools, or UX layers. *As always, priorities are subject to change based upon client demand.*

---

# Current Status — Winter 2025

---

## What Works Today

**You can use these capabilities right now:**

* ✅ **Full TRAINING Pipeline** — Run complete ranking + feature selection + leakage detection workflows
* ✅ **GPU-Accelerated Model Training** — Train LGBM/XGBoost + sequential models (CNN1D, LSTM, Transformer) with GPU acceleration
* ✅ **Centralized YAML Configuration** — Use structured configs for experiments, model families, and system parameters
* ✅ **Complete Documentation & Legal** — Full 4-tier docs hierarchy + enterprise legal package for commercial evaluation

**This is production-grade ML infrastructure, not a prototype.**

---

## Core System Status

* **TRAINING Pipeline — Phase 1** ✅ — Functioning properly. The intelligent training framework (target ranking, feature selection, automated leakage detection) is operational. **End-to-end testing currently underway** to validate full pipeline from target ranking → feature selection → model training.
* **Code Refactoring** ✅ (2025-12-09) — Large monolithic files split into modular components (4.5k → 82 lines, 3.4k → 56 lines, 2.5k → 66 lines) while maintaining 100% backward compatibility. Largest file now: 2,542 lines (cohesive subsystem, not monolithic).
* **Model Family Status Tracking** ✅ — Comprehensive debugging system added to identify which families succeed/fail in multi-model feature selection. Status persisted to JSON for analysis.
* **GPU Families** ✅ — Confirmed working. All GPU model families operational and producing artifacts. Expect some noise and warnings during training (harmless, do not affect functionality).
* **Sequential Models** ✅ — All sequential models (CNN1D, LSTM, Transformer, and LSTM-based variants) working and producing outputs. 3D preprocessing issues resolved.
* **Target Ranking & Selection** ✅ — Integrated and operational as part of Phase 1 pipeline.
* **Documentation Overhaul** ✅ — 55+ new files created, 50+ rewritten, standardized across all tiers.
* **Legal Compliance** ✅ — Enhanced with IP assignment agreement, regulatory disclaimers, and explicit "No Financial Advice" sections. Compliance assessment: 95% complete (after IP assignment signing).

---

# Phase 0 — Stability & Documentation ✅

**Deliverables:**
* ✅ Full 4-tier documentation hierarchy
* ✅ Enterprise-grade legal docs
* ✅ Navigation and cross-linking
* ✅ Internal docs organized
* ✅ Consistent formatting, naming, and structure
* ✅ Stable release for evaluators and enterprise inquiries

**Outcome:** Established FoxML Core as an evaluable, commercial-grade product.

---

# Phase 1 — Intelligent Training Framework ✅

**Status**: Functioning properly. **End-to-end testing underway** (2025-12-10).

**Completed:**
* ✅ TRAINING pipeline restored and validated
* ✅ GPU acceleration functional across major model families
* ✅ XGBoost source-build stability fixes
* ✅ Readline and child-process dependency issues resolved
* ✅ Sequential models 3D preprocessing fix
* ✅ Scaffolded base trainers for 2D and 3D models
* ✅ Intelligent training pipeline — target ranking, feature selection, and automated leakage detection integrated into unified workflow
* ✅ Target ranking and selection modules operational
* ✅ VAE serialization fixed — all models appear to be working correctly
* ✅ Structured logging configuration system implemented
* ✅ **Large file refactoring** (2025-12-09) — Split monolithic files into modular components
* ✅ **Model family status tracking** — Added debugging for multi-model feature selection
* ✅ **Interval detection robustness** — Fixed timestamp gap filtering
* ✅ **Import fixes** — Resolved all missing imports in refactored modules

**Current Testing:**
* 🔄 **End-to-end testing underway** — Full pipeline validation from target ranking → feature selection → model training
* Testing with multiple symbols and model families
* Validating data flow through Phase 3 (model training)
* Verifying model family status tracking output

**Planned Investigation:**
* Feature engineering review and validation (temporal alignment, lag structure, leakage validation)

**Planned Enhancements:**
* Intelligent training orchestration improvements
* Smarter model/ensemble selection
* **Feature Engineering Revamp** — Thorough review and validation of:
  * Proper temporal alignment and lag structure
  * Leakage prevention and validation
  * Statistical significance and predictive power
  * Cross-sectional vs. time-series feature design
  * Integration with the feature registry and schema system
* Automated hyperparameter search
* Robust cross-sectional + time-series workflows
* Refactor trainers to use scaffolded base classes for centralized dimension-specific logic

**Outcome:** TRAINING pipeline operational with intelligent framework. Phase 2 work now underway.

---

# Phase 2 — Centralized Configuration & UX Modernization 🔄

**Status:** Underway and mostly complete

**Completed:**
* ✅ YAML-based config schema (single source of truth) — 9 training config files created
* ✅ Config loader with nested access and family-specific overrides
* ✅ Integration into all model trainers (preprocessing, callbacks, optimizers, safety guards)
* ✅ Pipeline, threading, memory, GPU, and system configs integrated
* ✅ Backward compatibility maintained with hardcoded defaults

**In Progress:**
* Validation layer + example templates
* Unified logging + consistent output formatting
* Optional LLM-friendly structured logs
* Naming and terminology cleanup across modules

**Outcome:** Faster onboarding, easier enterprise deployment, more predictable behavior.

---

# Phase 3 — Memory & Data Efficiency

**Automated Memory Batching:**
* Fix current unstable behavior
* Add monitoring & adaptive batching
* Ensure clean memory reuse across model families

**Polars Cross-Sectional Optimization:**
* Streaming build operations
* Large-universe symbol handling
* Memory-efficient aggregation patterns

**Outcome:** Enables large-scale training without OOM failures.

---

# Phase 4 — Multi-GPU & NVLink Exploration (Future)

**NVLink Compatibility Research:**
* Explore NVLink support for multi-GPU training workflows
* Evaluate performance benefits for large model families (LSTM, Transformer, large MLPs)
* Test multi-GPU data parallelism patterns
* Benchmark NVLink vs PCIe bandwidth for model parameter synchronization
* Investigate framework support (TensorFlow, PyTorch, XGBoost multi-GPU)

**Multi-GPU Training Architecture:**
* Design multi-GPU training patterns for cross-sectional + sequential models
* Evaluate model parallelism vs data parallelism trade-offs
* Test gradient aggregation strategies across GPUs
* Memory-efficient multi-GPU batch distribution

**Outcome:** Foundation for scaling to multi-GPU systems when needed, with validated performance characteristics.

---

# Phase 5 — Trading Modules (Alpaca & IBKR)

**Alpaca (Fixes Required):**
* Resolve current module breakage
* Add paper-trading test suite
* Improve stability and retry logic
* Update documentation & examples

**IBKR (Untested):**
* Complete C++ component verification
* Live and paper trading tests
* Performance benchmarking
* Risk management validation

**Outcome:** Production-ready trading connectivity.

---

# Phase 6 — Web Presence & Payments

**Website + Stripe Checkout:**
* Public Fox ML Infrastructure homepage
* Pricing tiers + purchase flow
* "Request Access / Contact Sales" onboarding
* Hosted docs + system overview

**Outcome:** Enterprise-ready commercial experience.

---

# Phase 7 — Production Hardening

**Tasks:**
* Full test coverage
* Monitoring & observability
* Deployment guides
* Disaster recovery patterns
* Performance tuning

**Outcome:** Ready for institutional deployment.

---

# Phase 8 — High-Performance Rewrite Track (Long-Term)

**Low-Level Rewrites (C/C++/Rust):**
* Rewrite performance-critical paths
* HPC alignment + low-latency architecture (planned/WIP - single-node optimized currently)
* GPU-first and multi-GPU scaling

**Advanced Features:**
* Model ensemble architecture
* Multi-asset support
* Advanced execution and risk engines
* Real-time analytics pipeline

**ROCm Support (Future):**
* AMD GPU backend
* TensorFlow/XGBoost/LightGBM support via ROCm
* Parity with CUDA workflows

**Outcome:** FoxML Core becomes an HPC-aligned, cross-platform ML stack.

---

# Development Priorities

## Near-Term Focus

**Active development priorities — commercially leverageable work:**

1. **Configuration system validation & logging revamp** — Complete validation layer, unified logging, naming cleanup (mostly complete, validation in progress)
2. **Memory batching + Polars optimizations** — Fix unstable batching behavior, enable large-scale training without OOM
3. **Alpaca module fixes** — Resolve breakage, add paper-trading test suite, improve stability
4. **IBKR validation** — Complete C++ component verification, live/paper trading tests
5. **Website + Stripe checkout** — Public homepage, pricing tiers, purchase flow, hosted docs

**Guideline:** These priorities build on the completed Phase 0–2 foundation and represent the next commercially-leverageable deltas. Focus remains on validation, polish, and strategic sequencing rather than new major subsystems.

## Longer-Term / R&D

**Research and future-track work — not blocking near-term goals:**

6. **GPU + ranking regression testing** — Ongoing validation and edge-case coverage
7. **Production readiness** — Full test coverage, monitoring, deployment guides, disaster recovery
8. **Exploratory modules** — Experimental features and research tracks
9. **NVLink & multi-GPU exploration** — Research phase, performance benchmarking, architecture design
10. **High-performance rewrite track** — Low-level rewrites (C/C++/Rust), HPC alignment, ROCm support

**Guideline:** These items represent longer-horizon research and infrastructure work. They are valuable but not prerequisites for near-term commercial readiness.

---

# Vision

Fox ML Infrastructure is evolving into:
* A full-scale enterprise ML & trading infrastructure stack
* Multi-strategy, multi-model, GPU-accelerated
* Well-documented, configurable, and production-grade
* A foundation for future HPC and cross-platform ML systems

This roadmap reflects the maturation of FoxML Core from a high-powered solo project into a **commercially viable, enterprise-class ML platform.**

**Development Approach:** Phases 0–2 (documentation, intelligent training, configuration) are complete or substantially complete. Development is ahead of schedule on infrastructure and documentation. Current focus is validation, polish, and strategic sequencing of next-phase work. Priorities emphasize commercially-leverageable improvements over exploratory research.

---

# Known Issues & Environment Notes

## TensorFlow GPU Warnings

Some TensorFlow warnings (version compatibility, plugin registration) may appear during training. These are typically environment-specific and do not affect functionality:
* **Version compatibility warnings** (`use_unbounded_threadpool`) — Harmless version mismatch messages when graphs are created with newer TensorFlow than runtime.
* **Plugin registration warnings** (cuFFT, cuDNN, cuBLAS) — Expected when TensorFlow initializes CUDA plugins multiple times in isolated processes.
* **Status:** TensorFlow GPU support is functional. Warnings are being investigated and may be computer-specific. Waiting on feedback if others experience similar issues.

## Module Status

* **Alpaca Trading Module:** Broken due to minor errors — needs fixes
* **IBKR Trading Module:** Untested — requires comprehensive testing
* **Focus:** Training and ranking are stable and under ongoing validation; current active work centers on configuration, memory/polars, and trading connectivity.
