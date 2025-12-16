# Implementation Documentation

Detailed implementation guides and execution order documentation.

## Contents

### Core Systems
- **[Feature Selection Guide](FEATURE_SELECTION_GUIDE.md)** - Feature selection implementation
- **[Feature Selection Lookback Cap Integration](FEATURE_SELECTION_LOOKBACK_CAP_INTEGRATION.md)** - Lookback cap enforcement in feature selection
- **[Unified Lookback Cap Structure](UNIFIED_LOOKBACK_CAP_STRUCTURE.md)** - Standard structure for all lookback cap enforcement phases
- **[SST Enforcement Design](../../../TRAINING/utils/SST_ENFORCEMENT_DESIGN.md)** - **NEW**: EnforcedFeatureSet contract, type boundary wiring, boundary assertions (provably split-brain free)
- **[Reproducibility Tracking](REPRODUCIBILITY_TRACKING.md)** - Reproducibility system
- **[Active Sanitization](ACTIVE_SANITIZATION.md)** - Proactive feature quarantine
- **[Parallel Execution](PARALLEL_EXECUTION.md)** - Parallel execution infrastructure

### Execution Order
- **[Feature Filtering Execution Order](FEATURE_FILTERING_EXECUTION_ORDER.md)** - Feature filtering hierarchy
- **[Feature Selection Execution Order](FEATURE_SELECTION_EXECUTION_ORDER.md)** - Feature selection hierarchy
- **[Feature Pruning Execution Order](FEATURE_PRUNING_EXECUTION_ORDER.md)** - Feature pruning hierarchy
- **[Data Loading and Preprocessing Execution Order](DATA_LOADING_PREPROCESSING_EXECUTION_ORDER.md)** - Data pipeline hierarchy

### Reproducibility
- **[Reproducibility API](REPRODUCIBILITY_API.md)** - API reference
- **[Reproducibility Structure](REPRODUCIBILITY_STRUCTURE.md)** - Directory structure guide
- **[Cohort-Aware Reproducibility](COHORT_AWARE_REPRODUCIBILITY.md)** - Cohort-aware system
- **[Telemetry System](../02_reference/changelog/2025-12-14-telemetry-system.md)** - **NEW**: Sidecar-based telemetry with view isolation, hierarchical rollups

### Decision-Making (EXPERIMENTAL)
- **[Decision Engine](decisioning/DECISION_ENGINE.md)** - ⚠️ EXPERIMENTAL: Automated decision-making
- **[Bayesian Policy](decisioning/BAYESIAN_POLICY.md)** - ⚠️ EXPERIMENTAL: Adaptive config tuning
- **[Verification Checklist](decisioning/VERIFICATION_CHECKLIST.md)** - Decision verification

### Other
- **[Training Optimization Guide](TRAINING_OPTIMIZATION_GUIDE.md)** - Training optimization
- **[Adding Proprietary Models](ADDING_PROPRIETARY_MODELS.md)** - Custom model integration
- **[Performance Optimization](PERFORMANCE_OPTIMIZATION.md)** - Performance work
- See directory for complete list

## Related Documentation

- [Research Documentation](../research/README.md) - Research findings
- [Design Documentation](../design/README.md) - Design rationale

