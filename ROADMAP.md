# Fox ML Infrastructure — Strategic Roadmap

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

## Core System Status

* **TRAINING Pipeline** ✅ — Testing and verified. XGBoost issues resolved, orchestration stable, GPU training operational. (See Phase 1 for intelligent training framework work.)
* **GPU Families** ✅ — Confirmed working. All GPU model families operational and producing artifacts. Expect some noise and warnings during training (harmless, do not affect functionality).
* **Sequential Models** ✅ — All sequential models (CNN1D, LSTM, Transformer, and LSTM-based variants) working and producing outputs. 3D preprocessing issues resolved. (See Phase 1 for planned intelligent training orchestration refactors.)
* **Target Ranking & Selection** — Testing in progress, validating correctness before integration into TRAINING pipeline.
* **Documentation Overhaul** ✅ — 55 new files created, 50+ rewritten, standardized across all tiers.

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

# Phase 1 — Intelligent Training Framework (In Progress)

**Completed:**
* ✅ TRAINING pipeline restored and validated
* ✅ GPU acceleration functional across major model families
* ✅ XGBoost source-build stability fixes
* ✅ Readline and child-process dependency issues resolved
* ✅ Sequential models 3D preprocessing fix
* ✅ Scaffolded base trainers for 2D and 3D models

**Active Work:**
* Testing target-ranking and selection modules
* Reducing TensorFlow output noise and warnings
* ✅ VAE serialization fixed — all models appear to be working correctly

**Planned Refactors:**
* Intelligent training orchestration
* Smarter model/ensemble selection
* Improved feature engineering integration
* Automated hyperparameter search
* Robust cross-sectional + time-series workflows
* Refactor trainers to use scaffolded base classes for centralized dimension-specific logic

**Outcome:** TRAINING evolves from "functional" → "adaptive intelligence layer."

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

# Phase 4 — Trading Modules (Alpaca & IBKR)

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

# Phase 5 — Web Presence & Payments

**Website + Stripe Checkout:**
* Public Fox ML Infrastructure homepage
* Pricing tiers + purchase flow
* "Request Access / Contact Sales" onboarding
* Hosted docs + system overview

**Outcome:** Enterprise-ready commercial experience.

---

# Phase 6 — Production Hardening

**Tasks:**
* Full test coverage
* Monitoring & observability
* Deployment guides
* Disaster recovery patterns
* Performance tuning

**Outcome:** Ready for institutional deployment.

---

# Phase 7 — High-Performance Rewrite Track (Long-Term)

**Low-Level Rewrites (C/C++/Rust):**
* Rewrite performance-critical paths
* HPC alignment + low-latency architecture
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

# Development Priorities (Live List)

1. **Testing GPU models + ranking system**
2. **Configuration system validation & logging revamp** (mostly complete, validation in progress)
3. **Memory batching + Polars optimizations**
4. **Alpaca module fixes**
5. **IBKR validation**
6. **Website + Stripe checkout**
7. **Production readiness**
8. **Exploratory modules**
9. **High-performance rewrite track**

---

# Vision

Fox ML Infrastructure is evolving into:
* A full-scale enterprise ML & trading infrastructure stack
* Multi-strategy, multi-model, GPU-accelerated
* Well-documented, configurable, and production-grade
* A foundation for future HPC and cross-platform ML systems

This roadmap reflects the maturation of FoxML Core from a high-powered solo project into a **commercially viable, enterprise-class ML platform.**

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
* **Focus:** Training and ranking portions are the primary development priority
