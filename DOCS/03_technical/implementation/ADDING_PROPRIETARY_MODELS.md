# Adding Proprietary Models to the Training Factory

This guide explains how to add proprietary or custom models to the TRAINING pipeline using the `BaseModelTrainer` class.

> **Note:** This example uses the standard AGPL header. If you are implementing a truly proprietary model outside of the AGPL'd FoxML Core codebase (e.g., in a private extension layer), you should use your own license header consistent with your organization's policies.

## TL;DR

To add a custom/proprietary model:

1. Implement a trainer subclassing `BaseModelTrainer`.
2. Register it in `TRAINING/model_fun/__init__.py` and the runner maps.
3. Point your config's `model_family` to the new family name.

## Overview

The TRAINING pipeline uses a factory pattern where all model trainers inherit from `BaseModelTrainer`. This provides:
- Standardized preprocessing (imputation, column masking, data validation)
- Thread-safe training with automatic OMP/MKL thread management
- Consistent model saving/loading with preprocessors
- Integration with the isolation runner for process-level isolation
- Automatic memory management and cleanup

## BaseModelTrainer Features

The `BaseModelTrainer` abstract base class provides:

### Preprocessing
- **Automatic imputation**: Median imputation for missing values
- **Column masking**: Removes all-NaN columns during training
- **Data validation**: Guards against invalid features/targets
- **Type conversion**: Automatic float32 conversion for efficiency

### Threading Management
- **Smart threading**: `fit_with_threads()` and `predict_with_threads()` methods
- **BLAS optimization**: Automatic BLAS thread calculation per model family
- **OMP/MKL isolation**: Prevents threading conflicts between model families

### Model Persistence
- **Save/load**: Includes model, config, preprocessors, and metadata
- **Atomic writes**: Safe model saving with temporary files
- **Feature tracking**: Preserves feature names and column masks

## Step-by-Step Guide

### 1. Create Your Trainer Class

Create a new file in `TRAINING/model_fun/` (e.g., `my_proprietary_trainer.py`):

```python
"""
PSEUDOCODE EXAMPLE – DO NOT USE AS-IS

This is a template showing the structure. Replace all commented sections
with your actual implementation.

Copyright (c) 2025-2026 Fox ML Infrastructure LLC

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Note: If implementing outside the AGPL'd codebase, use your own license header.
"""

import numpy as np
import logging
from typing import Dict, Any, List
from .base_trainer import BaseModelTrainer

logger = logging.getLogger(__name__)

class MyProprietaryTrainer(BaseModelTrainer):
    """
    Custom trainer for proprietary model.
    
    Inherits all preprocessing, threading, and persistence from BaseModelTrainer.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        # Initialize your proprietary model here
        # self.proprietary_model = YourModelClass()
    
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              feature_names: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Train the proprietary model.
        
        Args:
            X_tr: Training features (numpy array)
            y_tr: Training targets (numpy array)
            feature_names: Optional feature names
            **kwargs: Additional training parameters
        
        Returns:
            Dictionary with trained model and metadata
        """
        # Validate data (inherited from BaseModelTrainer)
        self.validate_data(X_tr, y_tr)
        
        # Store feature names
        if feature_names:
            self.feature_names = feature_names
        
        # Preprocess data (handles imputation, column masking, validation)
        X_processed, y_processed = self.preprocess_data(X_tr, y_tr)
        
        # Initialize your proprietary model with config
        proprietary_config = self.config.get('proprietary_params', {})
        # self.proprietary_model = YourModelClass(**proprietary_config)
        
        # Train using fit_with_threads for proper threading
        # self.model = self.fit_with_threads(
        #     self.proprietary_model, 
        #     X_processed, 
        #     y_processed,
        #     phase="fit"  # or "meta", "linear_solve" for BLAS-heavy ops
        # )
        
        # For models that don't follow sklearn interface:
        # Train directly but use thread_guard for safety
        # from TRAINING.common.threads import thread_guard
        # with thread_guard(self.family_name, self._threads()):
        #     self.model.fit(X_processed, y_processed)
        
        # Mark as trained
        self.is_trained = True
        
        # Post-fit sanity check (inherited)
        self.post_fit_sanity(X_processed, self.family_name)
        
        return {
            'model': self.model,
            'feature_names': self.feature_names,
            'config': self.config
        }
    
    def predict(self, X_tr: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model.
        
        Args:
            X_tr: Features to predict on
        
        Returns:
            Predictions (numpy array)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Preprocess data (reuses fitted imputer and colmask)
        X_processed, _ = self.preprocess_data(X_tr, y=None)
        
        # Predict using predict_with_threads for proper threading
        predictions = self.predict_with_threads(self.model, X_processed)
        
        # For models that don't follow sklearn interface:
        # from TRAINING.common.threads import thread_guard
        # with thread_guard(self.family_name, self._threads()):
        #     predictions = self.model.predict(X_processed)
        
        # Return predictions from your proprietary model
        return predictions  # <-- implement this with your model
```

### 2. Register in model_fun/__init__.py

Add your trainer to the imports in `TRAINING/model_fun/__init__.py`:

```python
# ---- CPU-only families (safe to import everywhere) ----
from .lightgbm_trainer import LightGBMTrainer
from .quantile_lightgbm_trainer import QuantileLightGBMTrainer
from .xgboost_trainer import XGBoostTrainer
from .my_proprietary_trainer import MyProprietaryTrainer  # Add this
# ... other imports ...

__all__ = [
    'LightGBMTrainer',
    'QuantileLightGBMTrainer',
    'XGBoostTrainer',
    'MyProprietaryTrainer',  # Add this
    # ... other trainers ...
]
```

### 3. Register in Isolation Runner

Add your trainer to `TRAINING/common/isolation_runner.py` in the `TRAINER_MODULE_MAP`:

```python
TRAINER_MODULE_MAP = {
    "LightGBM": ("model_fun.lightgbm_trainer", "LightGBMTrainer"),
    "XGBoost": ("model_fun.xgboost_trainer", "XGBoostTrainer"),
    "MyProprietary": ("model_fun.my_proprietary_trainer", "MyProprietaryTrainer"),  # Add this
    # ... other mappings ...
}
```

**Important**: The key in `TRAINER_MODULE_MAP` should match the family name (class name without "Trainer" suffix). This string must also match the `model_family` / `family_name` you pass in your config when calling the TRAINING pipeline.

### 4. Register in In-Process Runner (Optional)

If you want in-process training (not just isolated), add to `TRAINING/training_strategies/family_runners.py` in the `MODMAP` (split from original `train_with_strategies.py`):

```python
MODMAP = {
    "LightGBM": ("model_fun.lightgbm_trainer", "LightGBMTrainer"),
    "XGBoost": ("model_fun.xgboost_trainer", "XGBoostTrainer"),
    "MyProprietary": ("model_fun.my_proprietary_trainer", "MyProprietaryTrainer"),  # Add this
    # ... other mappings ...
}
```

### 5. Configure Threading Policy (Optional)

If your model needs special threading behavior, add it to `TRAINING/common/family_config.py`:

```yaml
MyProprietary:
  thread_policy: omp_heavy  # or "mkl_heavy", "balanced", "blas_only"
  needs_gpu: false  # or true if GPU required
  backends: []  # or ["tf"], ["torch"], etc.
```

> **Note:** If `backends` is non-empty, the runner will ensure the corresponding framework is available before selecting this family. This prevents runtime errors when GPU frameworks aren't installed.

## Key Methods to Override

### Required Methods

- **`train()`**: Must implement model training logic
- **`predict()`**: Must implement prediction logic

### Optional Methods

- **`predict_proba()`**: For classification models with probability output
- **`get_feature_importance()`**: For models with feature importance
- **`__init__()`**: Override to add custom initialization (call `super().__init__()`)

## Best Practices

### 1. Use Inherited Preprocessing

Always use `self.preprocess_data()` instead of manual preprocessing:

```python
# ✅ Good
X_processed, y_processed = self.preprocess_data(X_tr, y_tr)

# ❌ Bad
X_processed = X_tr.copy()  # Missing imputation, validation, etc.
```

### 2. Use Threading Helpers

Use `fit_with_threads()` and `predict_with_threads()` for sklearn-compatible models:

```python
# ✅ Good (for sklearn-compatible models)
self.model = self.fit_with_threads(estimator, X, y, phase="fit")

# ✅ Good (for custom models)
from TRAINING.common.threads import thread_guard
with thread_guard(self.family_name, self._threads()):
    self.model.fit(X, y)
```

### 3. Handle GPU Models

If your model uses GPU, ensure it's registered correctly:

```python
# In family_config.py
MyProprietary:
  needs_gpu: true
  backends: ["tf"]  # or ["torch"], etc.
```

### 4. Thread Safety

For models that spawn threads internally, use the threading helpers:

```python
from TRAINING.common.threads import set_estimator_threads

# Set threads before training
set_estimator_threads(self.model, self._threads(), self.family_name)
self.model.fit(X, y)
```

### 5. Memory Management

The isolation runner handles memory cleanup automatically. For in-process training, ensure your model releases memory:

```python
# After training, if needed
del self.model
import gc
gc.collect()
```

## Example: Complete Trainer

Here's a complete example for a hypothetical proprietary model:

```python
import numpy as np
import logging
from typing import Dict, Any, List
from .base_trainer import BaseModelTrainer
from TRAINING.common.threads import thread_guard

logger = logging.getLogger(__name__)

class CustomRidgeTrainer(BaseModelTrainer):
    """Custom Ridge regression trainer using BaseModelTrainer."""
    
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, 
              feature_names: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Train Ridge model with proper preprocessing."""
        self.validate_data(X_tr, y_tr)
        
        if feature_names:
            self.feature_names = feature_names
        
        # Preprocess (imputation, column masking, validation)
        X_processed, y_processed = self.preprocess_data(X_tr, y_tr)
        
        # Get alpha from config
        alpha = self.config.get('alpha', 1.0)
        
        # Create and train model
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=alpha, random_state=42)
        
        # Use fit_with_threads for proper threading
        self.model = self.fit_with_threads(model, X_processed, y_processed, phase="linear_solve")
        
        self.is_trained = True
        self.post_fit_sanity(X_processed, self.family_name)
        
        return {'model': self.model, 'feature_names': self.feature_names}
    
    def predict(self, X_tr: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X_processed, _ = self.preprocess_data(X_tr, y=None)
        return self.predict_with_threads(self.model, X_processed)
```

## Testing Your Trainer

1. **Unit Test**: Test your trainer in isolation
2. **Integration Test**: Test with the full pipeline
3. **Threading Test**: Verify threading works correctly
4. **Memory Test**: Ensure no memory leaks

## Troubleshooting

### Import Errors

If your trainer isn't found:
- Check `model_fun/__init__.py` has the import
- Check `TRAINER_MODULE_MAP` has the correct mapping
- Verify the module path is correct

### Threading Issues

If you see threading conflicts:
- Use `fit_with_threads()` or `thread_guard()`
- Check `family_config.py` has correct thread policy
- Verify OMP/MKL thread counts are set correctly

### Preprocessing Issues

If predictions don't match training:
- Ensure you call `preprocess_data()` in both `train()` and `predict()`
- Don't manually fit imputers - use inherited preprocessing
- Check that `colmask` is applied consistently

### GPU Issues

If GPU models don't work:
- Verify `needs_gpu: true` in `family_config.py`
- Check `backends` list includes your framework
- Ensure CUDA_VISIBLE_DEVICES is set correctly

## See Also

- [BaseModelTrainer Source](../../../TRAINING/model_fun/base_trainer.py) - Full implementation
- [Model Training Guide](../../01_tutorials/training/MODEL_TRAINING_GUIDE.md) - Training workflow
- [Model Catalog](../../02_reference/models/MODEL_CATALOG.md) - All available models
- [Model Config Reference](../../02_reference/models/MODEL_CONFIG_REFERENCE.md) - Model configurations
- [Training Parameters](../../02_reference/models/TRAINING_PARAMETERS.md) - Training settings

