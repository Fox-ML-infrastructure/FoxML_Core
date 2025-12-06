# Fox ML Infrastructure — Roadmap

Direction and priorities for ongoing development. Timelines are aspirational and subject to change.

## Development Approach

As issues are discovered, they will be fixed immediately as time allows. Getting documentation and legal compliance sorted was a significant undertaking, and now the focus can shift back to core functionality. Response times will vary based on availability, but broken functionality will be addressed as quickly as possible.

**Current Development Focus:**
- **TRAINING pipeline** - ✅ **BACK TO BEING FUNCTIONAL** - Pipeline is now working correctly after fixing XGBoost installation issues and readline errors
- **Orchestration** - ✅ **WORKING FINE** - Training pipeline orchestration is functioning correctly
- **TensorFlow** - TensorFlow is working correctly with GPU support. Some warnings (version compatibility, plugin registration) are just noise - investigating further to clean up output. **Note:** These warnings may be computer-specific; waiting on feedback if others experience similar issues.
- **GPU models** - ✅ **WORKING** - GPU models are functional and producing artifacts. Working on reducing noise and warning messages in output.
- **GPU models** - Testing GPU models more later today
- **Target ranking and selection scripts** - Testing these scripts today to confirm they're working
- **Deeper refactors** - Planned for more intelligent training capabilities
- **Integration** - Target ranking functionality will eventually be rolled into the TRAINING pipeline for easier use

## Phase 0 — Winter 2025 (Right Now)

### Stability & Documentation Hardening

**Status: ✅ COMPLETE**

Documentation restructuring and creation completed:
- ✅ Created complete 4-tier documentation hierarchy (55 new files)
- ✅ Rewrote 50+ existing documentation files to enterprise standards
- ✅ Organized internal documentation into `docs/internal/`
- ✅ Created comprehensive navigation structure (`docs/INDEX.md`)
- ✅ Standardized formatting and naming conventions
- ✅ Added cross-links and "See Also" sections throughout
- ✅ Configured enterprise `.gitignore` (excludes datasets, internal files)
- ✅ Updated all legal documentation to 2025-2026
- ✅ Added enterprise-grade commercial license clauses

**Documentation Structure:**
- **Tier A (Executive)**: 4 files - Quick Start, Architecture Overview, Getting Started
- **Tier B (Tutorials)**: 14 files - Setup, Pipelines, Training, Trading, Configuration
- **Tier C (Reference)**: 15 files - API, Data, Models, Systems, Configuration
- **Tier D (Technical)**: 26 files - Research, Design, Benchmarks, Fixes, Roadmaps, Implementation, Testing, Operations

Goal: Deliver a stable, understandable release for early evaluators and commercial inquiries. **✅ ACHIEVED**

## Phase 1 — Next Development Cycle

This phase addresses the highest-impact improvements to onboarding, reliability, and developer experience.

### TRAINING Pipeline Refactoring

**Current Status: ✅ FUNCTIONAL**
- ✅ TRAINING pipeline is back to being functional
- ✅ Orchestration working fine - training pipeline orchestration is functioning correctly
- ✅ Fixed XGBoost installation issues (was pointing to deleted /tmp directory)
- ✅ Fixed readline symbol lookup errors in child processes
- ✅ All model families can now be imported and trained correctly
- ⚠️ TensorFlow warnings - TensorFlow is working correctly with GPU support. Some warnings (version compatibility, plugin registration) are just noise - investigating further to clean up output. **Note:** These warnings may be computer-specific; waiting on feedback if others experience similar issues.
- ✅ **GPU models working** - GPU models are functional and producing artifacts. Working on reducing noise and warning messages in output.
- 🔄 Testing target ranking and selection scripts today

**Planned Deeper Refactors:**
- More intelligent training capabilities
- Enhanced model selection and ensemble strategies
- Improved feature engineering integration
- Smarter hyperparameter optimization
- Better handling of cross-sectional and time-series data
- Advanced training workflows and automation

Impact: Transforms TRAINING from a functional system into an intelligent, adaptive training framework.

### Centralized Configuration System
- Move scattered config values into structured YAML.
- Introduce a documented configuration schema.
- Add validation + example templates to simplify onboarding.

Impact: Makes the system easier to deploy, audit, and reason about.

### Logging, Output & Developer UX Modernization
- Standardize log formatting across all modules.
- Improve readability and consistency of pipeline output.
- Add optional LLM-friendly output modes for automated parsing.
- Clean up naming conventions and internal terminology.

Impact: A smoother experience for both open-source users and enterprise clients.

### Automated Memory Batching & Control
- ✅ **Auto memory cleanup working** - Memory cleanup between model families is functioning correctly
- ⚠️ **Auto memory management and batching still problematic** - The automated memory batching system has "wonky" behavior
- Fix and improve automated memory batching system
- Implement reliable memory control mechanisms
- Address current "wonky" behavior in memory management module
- Ensure stable memory usage during large-scale training runs
- Add proper memory monitoring and automatic batching adjustments

Impact: Prevents OOM errors and enables training on larger datasets reliably.

### Polars Cross-Sectional DataFrame Optimization
- Optimize Polars cross-sectional dataframe building to avoid huge RAM spikes
- Create cleaner, more manageable memory usage patterns
- Implement streaming/chunked processing for large cross-sectional datasets
- Reduce memory footprint during dataframe concatenation and transformation
- Add memory-efficient cross-sectional aggregation methods

Impact: Enables processing of larger symbol universes without memory exhaustion, improves stability during training.

### Testing & Validation

**Current Status:**
- **TensorFlow Module**: ✅ **FUNCTIONAL** - Previously had system-level library issue (executable stack permissions) that was resolved by user. TensorFlow GPU support is now working for the main TRAINING module. All TensorFlow families (MLP, VAE, GAN, MetaLearning, MultiTask, etc.) can use GPU.
- **Alpaca Trading Module**: Broken due to minor errors - needs fixes
- **IBKR Trading Module**: Untested - requires comprehensive testing
- **Focus**: Training and ranking portions are the primary development priority

**Alpaca Trading Module**
- Fix minor errors causing module breakage
- Comprehensive testing suite for paper trading system
- Integration testing with Alpaca API
- Performance validation and optimization
- Bug fixes and stability improvements
- Documentation updates based on testing findings

**IBKR Trading Module**
- Complete C++ component testing and validation
- IBKR API integration testing
- Live trading system testing and validation
- Performance benchmarking and optimization
- Safety and risk management validation
- Bug fixes and stability improvements

## Phase 2 — Web Presence & Payment Flow

Stripe is already fully functional through email-based invoicing.
This phase turns it into a seamless, professional customer journey.

### Public Website + Integrated Payment Flow
- Launch official Fox ML Infrastructure website.
- Convert existing Stripe setup into a self-serve checkout flow.
- Add clear pricing tiers and a "Request Access / Contact Sales" path.
- Centralize docs, onboarding, system overview, and commercial materials.

Impact: Makes Fox ML Infrastructure feel like a complete, production-ready product.

## Phase 3 — Production Readiness

- Complete test coverage for all trading modules
- Performance optimization across all components
- Enhanced monitoring and observability
- Production deployment guides
- Disaster recovery procedures

## Phase 4 — Exploratory & Architectural Extensions

Focused on expanding capabilities while keeping the core stable.

### Exploratory Modules & Internal Enhancements
- Explore toon-like or related module implementations.
- Review user feedback and prioritize expansions accordingly.
- Refine internal architecture based on evaluator insights.

Impact: Controlled innovation without destabilizing the core system.

## Phase 5 — High-Performance Rewrite Track

Long-term improvements targeting performance at the systems level.

### Lower-Level Rewrites (C/C++/Rust)
- Reimplement performance-critical paths in lower-level languages.
- Boost throughput, memory efficiency, and integration with HPC tooling.
- Build the foundation for future high-throughput variants of Fox-v1.

Impact: Positions Fox ML Infrastructure as a high-performance, HPC-aligned ML stack.

### Advanced Features
- Enhanced model ensemble strategies
- Advanced risk management features
- Multi-asset class support
- Real-time performance analytics
- Advanced execution algorithms

### ROCm Support (Future)
- **Status**: Planned for post-architecture-solidification phase
- Add AMD GPU support via ROCm once major architecture is solidified
- Enable TensorFlow/XGBoost/LightGBM GPU acceleration on AMD hardware
- Requires stable CUDA abstraction layer and framework compatibility
- Will follow same patterns as CUDA support but with ROCm backend

Impact: Expands GPU acceleration support to AMD hardware, increasing accessibility and deployment options.

## Development Priorities

Current focus areas (in rough order, subject to change):

- ✅ **Completed**: Documentation hardening (55 new files, 50+ rewritten)
- ✅ **Completed**: TRAINING pipeline fixes - back to being functional
- **Current Focus**: Testing GPU models and target ranking/selection scripts today
- **Future**: Integration - target ranking rolled into TRAINING pipeline for easier use
- **Next**: Config system + logging/output overhaul
- **Next**: Fix automated memory batching and control
- **Next**: Optimize Polars cross-sectional dataframe building
- **Pending**: Alpaca module fixes (broken due to minor errors)
- **Pending**: IBKR module testing (untested, needs validation)
- **Future**: Website + integrated Stripe checkout
- **Future**: Production readiness and deployment guides
- **Future**: Exploratory modules and enhancements
- **Long-term**: Lower-level performance rewrites and advanced features

*Note: As a solo developer, timelines are flexible and priorities may shift based on user feedback and project needs.*

## Long-term Vision

- Enterprise-grade trading infrastructure
- Scalable multi-strategy framework
- Comprehensive risk management
- Production-ready deployment
- Extensive documentation and support
