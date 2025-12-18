# Feature Engineering Tutorial

Create and customize technical features for machine learning models.

## Overview

Feature engineering transforms raw market data into predictive features. FoxML Core provides three feature builders with increasing complexity.

## Feature Builders

### SimpleFeatureComputer

Basic technical indicators (50+ features):

```python
from DATA_PROCESSING.features import SimpleFeatureComputer

computer = SimpleFeatureComputer()
features = computer.compute(df)
```

**Includes:**
- Price returns (1m, 5m, 15m, 30m, 60m)
- Volatility (rolling std)
- Momentum indicators
- Volume ratios

### ComprehensiveFeatureBuilder

Extended feature set (200+ features):

```python
from DATA_PROCESSING.features import ComprehensiveFeatureBuilder

builder = ComprehensiveFeatureBuilder(config_path="config/features.yaml")
# Note: build_features() processes files in batch, not single DataFrames
features = builder.build_features(input_paths, output_dir, universe_config)
```

**Adds:**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Microstructure features (spread, order flow)
- Cross-asset features
- Regime indicators

### Streaming Features

> **Note**: `StreamingFeatureBuilder` is not available as a class. For streaming processing, use functions from `DATA_PROCESSING.features.streaming_builder` module.

**Use for:**
- Live trading systems
- Real-time inference
- Low-latency applications

## Custom Features

### Adding Custom Features

```python
from DATA_PROCESSING.features import SimpleFeatureComputer
import pandas as pd

class CustomFeatureComputer(SimpleFeatureComputer):
    def compute(self, df):
        features = super().compute(df)
        
        # Add custom feature
        features['custom_ratio'] = df['close'] / df['volume'].rolling(20).mean()
        
        return features

computer = CustomFeatureComputer()
features = computer.compute(df)
```

## Feature Selection

After building features, select the most important:

```python
from TRAINING.training_strategies.strategies.single_task import SingleTaskStrategy
# Backward compatibility: from TRAINING.strategies.single_task import ... still works

# Train model to get feature importance
strategy = SingleTaskStrategy(config)
strategy.train(X, y, feature_names)

# Get feature importance from trained model
importances = strategy.get_feature_importance()

# Select top 50 features
top_50 = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:50]
selected_features = [f[0] for f in top_50]
```

> **Note**: `scripts.feature_selection` module does not exist. Use the strategy's `get_feature_importance()` method instead.

## Best Practices

1. **Start Simple**: Use SimpleFeatureComputer first
2. **Validate**: Check for NaN values and data quality
3. **Select**: Use feature selection to reduce dimensionality
4. **Monitor**: Track feature importance over time

## Next Steps

- [Feature Selection Tutorial](../training/FEATURE_SELECTION_TUTORIAL.md) - Select best features
- [Column Reference](../../02_reference/data/COLUMN_REFERENCE.md) - Feature documentation

