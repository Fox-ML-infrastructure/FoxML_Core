# Architecture Deep Dive

Detailed system architecture and design decisions.

## System Overview

FoxML Core is a multi-layered ML research infrastructure with:

1. **Data Processing Layer**: Raw data → Features → Targets
2. **Model Training Layer**: 17+ model types with centralized configs
3. **Model Output Layer**: Trained models, predictions, and performance metrics
4. **Performance Layer**: Optimized pipelines for high-throughput processing

## Data Flow

```
Raw Market Data
    ↓
Normalization (RTH alignment, grid correction)
    ↓
Feature Engineering (200+ technical features)
    ↓
Target Generation (barrier, excess returns, HFT)
    ↓
Labeled Dataset
    ↓
Model Training (17+ models, walk-forward validation)
    ↓
Trained Models
    ↓
Model Artifacts & Performance Reports
```

## Component Architecture

### Data Processing

- **Normalization**: Session alignment, grid correction
- **Feature Builders**: Simple, Comprehensive, Streaming
- **Target Builders**: Barrier, Excess Returns, HFT Forward Returns

### Model Training

- **Training Strategies**: Single-task, Multi-task
- **Validation**: Walk-forward validation
- **Configuration**: Centralized YAML configs with variants

### Model Output & Artifacts

**Trained Models**:
- Serialized model artifacts
- Performance metrics and validation reports
- Feature importance and model diagnostics

**Integration Points**:
- Models can be integrated with external trading systems
- Standard formats for model serialization
- Performance tracking and monitoring interfaces

## Design Principles

1. **Configuration-Driven**: All runtime parameters from config
2. **Safety-First**: Multiple layers of guards and checks
3. **Performance-Critical**: C++ for hot paths, Python for orchestration
4. **Modular**: Clear separation of concerns
5. **Extensible**: Easy to add new models, features, strategies

## See Also

- [Architecture Overview](../../00_executive/ARCHITECTURE_OVERVIEW.md) - High-level overview
- [Module Reference](../../02_reference/api/MODULE_REFERENCE.md) - API documentation

