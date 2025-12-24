# Training Pipeline Flow Diagram

## Current Flow (With Redundancies)

```mermaid
flowchart TD
    A[Feature Generation] --> B[Feature Filtering]
    B --> C[SAFE_CANDIDATES<br/>fingerprint: SAFE_CANDIDATES]
    
    C --> D1[create_resolved_config<br/>PRE-GATEKEEPER<br/>lookback: pre-enforcement]
    D1 --> D2[compute_feature_lookback_max<br/>Recompute 1]
    
    C --> E[apply_lookback_cap FS_PRE]
    E --> F[Feature Selection]
    F --> G[apply_lookback_cap FS_POST]
    
    G --> H[POST_GATEKEEPER<br/>fingerprint: POST_GATEKEEPER]
    H --> I[apply_lookback_cap GATEKEEPER]
    I --> I1[compute_budget<br/>Recompute 2]
    
    I --> J[Feature Pruning]
    J --> K[POST_PRUNE<br/>fingerprint: POST_PRUNE]
    
    K --> L1[create_resolved_config<br/>POST-PRUNE<br/>Recompute 3]
    L1 --> L2[compute_feature_lookback_max<br/>Recompute 4]
    
    K --> M1[compute_budget<br/>POST-PRUNE POLICY CHECK<br/>Recompute 5]
    
    K --> N[PurgedTimeSeriesSplit CV]
    N --> O{Check CV<br/>Compatibility?}
    O -->|No| P[CatBoost CV<br/>May return NaN]
    O -->|Yes| Q[Fallback Policy]
    
    P --> R[Model Training]
    Q --> R
    
    style D1 fill:#ffcccc
    style D2 fill:#ffcccc
    style I1 fill:#ffcccc
    style L1 fill:#ffcccc
    style L2 fill:#ffcccc
    style M1 fill:#ffcccc
    style O fill:#ccffcc
    style Q fill:#ccffcc
```

**Red boxes**: Redundant recomputation  
**Green boxes**: Missing (should exist)

## Proposed Flow (With FeatureSet Artifact)

```mermaid
flowchart TD
    A[Feature Generation] --> B[Feature Filtering]
    B --> C[SAFE_CANDIDATES<br/>FeatureSetArtifact]
    
    C --> D[apply_lookback_cap FS_PRE<br/>Updates artifact]
    D --> E[Feature Selection]
    E --> F[apply_lookback_cap FS_POST<br/>Updates artifact]
    
    F --> G[POST_GATEKEEPER<br/>FeatureSetArtifact]
    G --> H[apply_lookback_cap GATEKEEPER<br/>Updates artifact]
    
    H --> I[Feature Pruning]
    I --> J[POST_PRUNE<br/>FeatureSetArtifact<br/>Final artifact]
    
    J --> K[PurgedTimeSeriesSplit CV<br/>Uses artifact.budget]
    K --> L{Pre-CV<br/>Compatibility Check}
    L -->|Degenerate| M[Fallback Policy<br/>reduce_folds/skip_cv]
    L -->|OK| N[CatBoost CV]
    
    M --> O[Model Training]
    N --> O
    
    style C fill:#ccffcc
    style G fill:#ccffcc
    style J fill:#ccffcc
    style L fill:#ccffcc
    style M fill:#ccffcc
```

**Green boxes**: FeatureSet artifacts (single source of truth per stage)

## FeatureSet Artifact Structure

```mermaid
classDiagram
    class FeatureSetArtifact {
        +List[str] features
        +str fingerprint_set
        +str fingerprint_ordered
        +Dict[str, float] canonical_lookback_map
        +LeakageBudget budget
        +ResolvedConfig resolved_config
        +Dict[str, str] removal_reasons
        +str stage
        +to_enforced_set() EnforcedFeatureSet
        +validate_invariants() bool
    }
    
    class EnforcedFeatureSet {
        +List[str] features
        +str fingerprint_set
        +str fingerprint_ordered
        +Optional[float] cap_minutes
        +float actual_max_minutes
        +Dict[str, float] canonical_map
        +Dict[str, float] quarantined
        +List[str] unknown
        +str stage
        +LeakageBudget budget
    }
    
    class LeakageBudget {
        +float interval_minutes
        +float horizon_minutes
        +float max_feature_lookback_minutes
        +Optional[float] cap_max_lookback_minutes
        +Optional[float] allowed_max_lookback_minutes
        +required_gap_minutes() float
    }
    
    class ResolvedConfig {
        +float purge_minutes
        +float embargo_minutes
        +float feature_lookback_max_minutes
        +int features_safe
        +int features_dropped_nan
        +int features_final
    }
    
    FeatureSetArtifact --> EnforcedFeatureSet : converts to
    FeatureSetArtifact --> LeakageBudget : contains
    FeatureSetArtifact --> ResolvedConfig : contains
```

## Budget Computation Call Sites (Current)

```mermaid
graph LR
    A[evaluate_target_predictability] --> B[create_resolved_config PRE-GATEKEEPER]
    B --> C[compute_feature_lookback_max]
    
    A --> D[_enforce_final_safety_gate]
    D --> E[apply_lookback_cap]
    E --> F[compute_budget]
    
    A --> G[train_and_evaluate_models]
    G --> H[create_resolved_config POST-PRUNE]
    H --> I[compute_feature_lookback_max]
    
    G --> J[compute_budget POST-PRUNE POLICY]
    
    style C fill:#ffcccc
    style F fill:#ffcccc
    style I fill:#ffcccc
    style J fill:#ffcccc
```

**Red boxes**: Redundant calls (should be cached/reused)

