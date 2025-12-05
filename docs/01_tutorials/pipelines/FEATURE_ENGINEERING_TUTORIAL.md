# Feature Engineering Tutorial

Create and customize technical features for machine learning models.

## Overview

Feature engineering transforms raw market data into predictive features. Fox-v1-infra provides three feature builders with increasing complexity.

## Feature Builders

### SimpleFeatureBuilder

Basic technical indicators (50+ features):

```python
from DATA_PROCESSING.features import SimpleFeatureBuilder

builder = SimpleFeatureBuilder()
features = builder.build(df)
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

builder = ComprehensiveFeatureBuilder()
features = builder.build(df)
```

**Adds:**
- Technical indicators (RSI, MACD, Bollinger Bands)
- Microstructure features (spread, order flow)
- Cross-asset features
- Regime indicators

### StreamingFeatureBuilder

Real-time feature computation:

```python
from DATA_PROCESSING.features import StreamingFeatureBuilder

builder = StreamingFeatureBuilder()
features = builder.build(df)
```

**Use for:**
- Live trading systems
- Real-time inference
- Low-latency applications

## Custom Features

### Adding Custom Features

```python
from DATA_PROCESSING.features import SimpleFeatureBuilder
import pandas as pd

class CustomFeatureBuilder(SimpleFeatureBuilder):
    def build(self, df):
        features = super().build(df)
        
        # Add custom feature
        features['custom_ratio'] = df['close'] / df['volume'].rolling(20).mean()
        
        return features

builder = CustomFeatureBuilder()
features = builder.build(df)
```

## Feature Selection

After building features, select the most important:

```python
from TRAINING.strategies.single_task import SingleTaskStrategy
from scripts.feature_selection import select_top_features

# Train model to get feature importance
strategy = SingleTaskStrategy(config)
strategy.train(X, y, feature_names)

# Select top 50 features
selected_features = select_top_features(strategy, n_features=50)
```

## Best Practices

1. **Start Simple**: Use SimpleFeatureBuilder first
2. **Validate**: Check for NaN values and data quality
3. **Select**: Use feature selection to reduce dimensionality
4. **Monitor**: Track feature importance over time

## Next Steps

- [Feature Selection Tutorial](../training/FEATURE_SELECTION_TUTORIAL.md) - Select best features
- [Column Reference](../../02_reference/data/COLUMN_REFERENCE.md) - Feature documentation

