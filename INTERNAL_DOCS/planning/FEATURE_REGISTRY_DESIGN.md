# Feature Registry & Automated Leakage Prevention Design

**Status**: Design Phase  
**Created**: 2025-12-07  
**Goal**: Implement systematic feature registry with time-based rules to make leakage structurally impossible

---

## Problem Statement

Current system uses **reactive pattern matching** (`CONFIG/excluded_features.yaml`) which:
- ✅ Works but requires manual maintenance
- ❌ Easy to miss new leaky features
- ❌ No structural guarantees
- ❌ Manual pruning is error-prone

**Proposed Solution**: **Proactive structural rules** that make leakage impossible without lying to the system.

---

## Design Principles

1. **Structural Impossibility**: Features cannot be used if they violate time rules
2. **Metadata-Driven**: Every feature has explicit temporal metadata
3. **Automated Validation**: System validates rules at load time
4. **Manual Sign-Off**: Human review for edge cases, not manual pruning
5. **Backward Compatible**: Works with existing features via auto-inference

---

## Architecture

### 1. Feature Registry (`CONFIG/feature_registry.yaml`)

```yaml
features:
  # Lagged returns (safe)
  ret_1:
    source: price
    lag_bars: 1
    allowed_horizons: [1, 3, 5, 15, 30, 60]  # Can predict up to 60 bars ahead
    description: "1-bar lagged return"
  
  ret_5:
    source: price
    lag_bars: 5
    allowed_horizons: [5, 15, 30, 60]
    description: "5-bar lagged return"
  
  # Technical indicators (safe)
  rsi_10:
    source: derived
    lag_bars: 10
    allowed_horizons: [1, 3, 5, 15, 30, 60]
    description: "RSI with 10-bar lookback"
  
  sma_200:
    source: derived
    lag_bars: 200
    allowed_horizons: [1, 3, 5, 15, 30, 60]
    description: "200-bar simple moving average"
  
  # Future-looking features (REJECTED at load time)
  ret_future_1:
    source: price
    lag_bars: -1  # Negative = looks into future
    allowed_horizons: []  # Empty = rejected
    description: "REJECTED: Future return"
  
  # Time-to-hit features (REJECTED)
  tth_5m:
    source: derived
    lag_bars: 0  # Computed at time of hit (requires future)
    allowed_horizons: []  # Empty = rejected
    description: "REJECTED: Time-to-hit requires future path"
  
  # Barrier features (REJECTED)
  barrier_5m_0.8:
    source: derived
    lag_bars: 0
    allowed_horizons: []
    description: "REJECTED: Barrier features encode barrier logic"

# Feature families (groups of related features)
feature_families:
  lagged_returns:
    pattern: "^ret_\\d+$"
    default_lag_bars: null  # Must be specified per feature
    default_allowed_horizons: [1, 3, 5, 15, 30, 60]
  
  technical_indicators:
    pattern: "^(rsi|sma|ema|macd|bb)_"
    default_lag_bars: null
    default_allowed_horizons: [1, 3, 5, 15, 30, 60]
  
  rejected_families:
    pattern: "^(tth|mfe|mdd|barrier|y_|p_|future_)"
    default_lag_bars: 0
    default_allowed_horizons: []  # Always rejected

# Validation rules
validation:
  hard_rules:
    - "lag_bars >= 0"  # Cannot look into future
    - "lag_bars >= horizon_bars for price/derived features"  # Must lag by at least horizon
    - "allowed_horizons must be non-empty for usable features"
  
  warnings:
    - "lag_bars < horizon_bars * 0.5"  # Warning if lag is too small
    - "source == 'unknown'"  # Warning if source not specified
```

### 2. FeatureRegistry Class

```python
class FeatureRegistry:
    """Manages feature metadata and enforces temporal rules."""
    
    def __init__(self, config_path: Path = None):
        """Load feature registry from YAML."""
        self.config = self._load_config(config_path)
        self.features = self.config.get('features', {})
        self.families = self.config.get('feature_families', {})
        self._validate_registry()
    
    def _validate_registry(self):
        """Validate all features against hard rules."""
        for name, metadata in self.features.items():
            self._validate_feature(name, metadata)
    
    def _validate_feature(self, name: str, metadata: Dict[str, Any]):
        """Validate a single feature against hard rules."""
        lag_bars = metadata.get('lag_bars', 0)
        allowed_horizons = metadata.get('allowed_horizons', [])
        source = metadata.get('source', 'unknown')
        
        # Hard rule: Cannot look into future
        if lag_bars < 0:
            raise ValueError(f"Feature {name}: lag_bars={lag_bars} < 0 (looks into future)")
        
        # Hard rule: For price/derived features, lag must be >= horizon
        if source in ['price', 'derived']:
            for horizon in allowed_horizons:
                if lag_bars < horizon:
                    raise ValueError(
                        f"Feature {name}: lag_bars={lag_bars} < horizon={horizon} "
                        f"(would leak future information)"
                    )
        
        # Hard rule: Usable features must have allowed horizons
        if not allowed_horizons and metadata.get('rejected', False) is False:
            logger.warning(f"Feature {name}: No allowed_horizons (will be rejected)")
    
    def is_allowed(self, feature_name: str, target_horizon: int) -> bool:
        """Check if feature is allowed for a target horizon."""
        # Check explicit feature metadata
        if feature_name in self.features:
            metadata = self.features[feature_name]
            allowed_horizons = metadata.get('allowed_horizons', [])
            if target_horizon in allowed_horizons:
                return True
            return False
        
        # Check feature families
        for family_name, family_config in self.families.items():
            pattern = family_config.get('pattern')
            if pattern and re.match(pattern, feature_name):
                # Rejected families
                if family_name.startswith('rejected_'):
                    return False
                # Allowed families
                default_horizons = family_config.get('default_allowed_horizons', [])
                if target_horizon in default_horizons:
                    return True
        
        # Unknown feature: reject by default (safe)
        logger.warning(f"Unknown feature {feature_name}: rejecting (safe default)")
        return False
    
    def get_allowed_features(self, all_features: List[str], target_horizon: int) -> List[str]:
        """Get list of allowed features for a target horizon."""
        return [f for f in all_features if self.is_allowed(f, target_horizon)]
    
    def auto_infer_metadata(self, feature_name: str) -> Dict[str, Any]:
        """Auto-infer metadata for unknown features (backward compatibility)."""
        # Try to infer from name patterns
        if re.match(r"^ret_\d+$", feature_name):
            lag = int(feature_name.split('_')[1])
            return {
                'source': 'price',
                'lag_bars': lag,
                'allowed_horizons': [lag, lag*3, lag*5] if lag > 0 else []
            }
        # ... more inference rules
        
        # Default: reject (safe)
        return {
            'source': 'unknown',
            'lag_bars': 0,
            'allowed_horizons': []
        }
```

### 3. Integration Points

#### A. Feature Engineering Level

```python
# In DATA_PROCESSING/features/
class FeatureBuilder:
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
    
    def register_feature(self, name: str, metadata: Dict[str, Any]):
        """Register a new feature with metadata."""
        # Validate at registration time
        self.registry._validate_feature(name, metadata)
        # Add to registry
        self.registry.features[name] = metadata
```

#### B. Feature Selection Level

```python
# In TRAINING/ranking/feature_selector.py
def select_features_for_target(
    target_column: str,
    symbols: List[str],
    data_dir: Path,
    registry: FeatureRegistry = None,  # NEW
    ...
):
    # Extract target horizon
    target_horizon = extract_horizon_from_target(target_column)
    
    # Get allowed features from registry
    if registry:
        all_features = get_all_features(data)
        allowed_features = registry.get_allowed_features(all_features, target_horizon)
        # Only select from allowed features
        # ... rest of selection logic
```

#### C. Training Level

```python
# In TRAINING/train_with_strategies.py
def prepare_training_data_cross_sectional(
    mtf_data: Dict[str, pd.DataFrame],
    target: str,
    feature_names: List[str] = None,
    registry: FeatureRegistry = None,  # NEW
    ...
):
    # Extract target horizon
    target_horizon = extract_horizon_from_target(target)
    
    # Filter features using registry
    if registry:
        if feature_names:
            # Filter provided features
            feature_names = registry.get_allowed_features(feature_names, target_horizon)
        else:
            # Auto-discover and filter
            all_features = discover_features(mtf_data)
            feature_names = registry.get_allowed_features(all_features, target_horizon)
    
    # ... rest of data preparation
```

### 4. Automated Leakage Sentinels

```python
class LeakageSentinel:
    """Automated tests to detect leakage."""
    
    def shifted_target_test(self, model, X, y, horizon: int):
        """Test with target shifted by +N bars."""
        y_shifted = np.roll(y, horizon)  # Shift target forward
        score_shifted = model.score(X, y_shifted)
        
        if score_shifted > 0.5:  # Suspiciously good
            logger.warning(
                f"🚨 LEAKAGE ALERT: Model performs well on shifted target "
                f"(score={score_shifted:.3f}). Features may look into future."
            )
            return False
        return True
    
    def symbol_holdout_test(self, model, X_train, y_train, X_test, y_test):
        """Test on never-seen symbols."""
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        if train_score > 0.9 and test_score < 0.3:
            logger.warning(
                f"🚨 LEAKAGE ALERT: Large train/test gap "
                f"(train={train_score:.3f}, test={test_score:.3f}). "
                f"Possible symbol-specific leakage."
            )
            return False
        return True
    
    def randomized_time_test(self, model, X, y):
        """Test with shuffled time (features/targets paired)."""
        # Shuffle time index but keep feature-target pairs
        indices = np.arange(len(X))
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        score_shuffled = model.score(X_shuffled, y_shuffled)
        
        if score_shuffled > 0.5:  # Should be random
            logger.warning(
                f"🚨 LEAKAGE ALERT: Model performs well on time-shuffled data "
                f"(score={score_shuffled:.3f}). Features may encode future info."
            )
            return False
        return True
```

### 5. Feature Importance Diff Leak Detector

```python
def detect_leakage_via_importance_diff(
    model_full: Any,
    model_safe: Any,
    feature_names: List[str],
    threshold: float = 0.1
) -> List[str]:
    """Compare feature importances to detect leaks."""
    importance_full = get_feature_importance(model_full, feature_names)
    importance_safe = get_feature_importance(model_safe, feature_names)
    
    # Features with high importance in full but low in safe
    diff = importance_full - importance_safe
    suspicious = [
        name for name, d in zip(feature_names, diff)
        if d > threshold
    ]
    
    if suspicious:
        logger.warning(
            f"🚨 SUSPECTED LEAKS: {len(suspicious)} features have high importance "
            f"in full model but low in safe model: {suspicious[:10]}"
        )
    
    return suspicious
```

---

## Migration Path

### Phase 1: Registry Infrastructure (Week 1)
- [ ] Create `FeatureRegistry` class
- [ ] Create `CONFIG/feature_registry.yaml` template
- [ ] Auto-inference for existing features
- [ ] Integration with `leakage_filtering.py`

### Phase 2: Validation & Enforcement (Week 2)
- [ ] Hard rule validation at load time
- [ ] Integration with feature selection
- [ ] Integration with training pipeline
- [ ] Backward compatibility layer

### Phase 3: Automated Sentinels (Week 3)
- [ ] Implement leakage sentinels
- [ ] Integration with intelligent trainer
- [ ] Logging and reporting
- [ ] Optional diagnostic mode

### Phase 4: Feature Importance Diff (Week 4)
- [ ] Implement importance diff detector
- [ ] Integration with feature selection
- [ ] Reporting and flagging
- [ ] Documentation

---

## Benefits

1. **Structural Safety**: Leakage becomes impossible without lying to config
2. **Automated Detection**: Sentinels catch issues automatically
3. **Manual Sign-Off**: Human review for edge cases, not manual pruning
4. **Backward Compatible**: Works with existing features via auto-inference
5. **Documentation**: Feature metadata serves as documentation

---

## Risks & Mitigations

### Risk: Breaking Existing Workflows
**Mitigation**: Backward compatibility layer with auto-inference

### Risk: Over-Engineering
**Mitigation**: Start simple, add complexity only if needed

### Risk: False Positives
**Mitigation**: Sentinels are warnings, not blockers (configurable)

---

## Next Steps

1. Review this design
2. Create `FeatureRegistry` class (MVP)
3. Create `CONFIG/feature_registry.yaml` template
4. Integrate with existing `leakage_filtering.py`
5. Test with existing features
6. Add sentinels incrementally

---

## Related Documentation

- [Leakage Analysis](../../03_technical/research/LEAKAGE_ANALYSIS.md)
- [Feature Importance Methodology](../../03_technical/research/FEATURE_IMPORTANCE_METHODOLOGY.md)
- [Validation Methodology](../../03_technical/research/VALIDATION_METHODOLOGY.md)

