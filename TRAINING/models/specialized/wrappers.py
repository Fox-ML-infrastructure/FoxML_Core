# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


"""Model wrapper classes for specialized models."""

class TFSeriesRegressor:
    """Wrapper for TensorFlow models to ensure consistent preprocessing during prediction."""

    def __init__(self, keras_model, imputer, scaler, n_feat):
        self.model = keras_model
        self.imputer = imputer
        self.scaler = scaler
        self.n_feat = n_feat
        self.handles_preprocessing = True
    
    def _prep(self, X):
        """Apply imputation, scaling and reshaping for TF models."""
        Xc = self.imputer.transform(X)
        Xs = self.scaler.transform(Xc)
        return Xs.reshape(Xs.shape[0], self.n_feat, 1)
    
    def predict(self, X):
        """Predict with proper preprocessing."""
        Xp = self._prep(X)
        return self.model.predict(Xp, verbose=0).ravel()



class GMMRegimeRegressor:
    """GMM-based regime detection with regime-specific models."""
    def __init__(self, gmm, regressors, scaler, imputer, n_regimes):
        self.gmm = gmm
        self.regressors = regressors
        self.scaler = scaler          # fit on ENHANCED features
        self.imputer = imputer
        self.n_regimes = n_regimes
        self.handles_preprocessing = True

    def _enhance(self, X_clean):
        """Build enhanced features with regime information."""
        import numpy as np
        regime_post = self.gmm.predict_proba(X_clean)        # (N, K)
        regime_labels = regime_post.argmax(1)                # (N,)
        regime_features = np.column_stack([
            regime_labels.reshape(-1, 1),                    # (N, 1)
            regime_post,                                     # (N, K)
            np.mean(X_clean, axis=1, keepdims=True),         # (N, 1)
            np.std(X_clean, axis=1, keepdims=True),         # (N, 1)
        ])
        return np.column_stack([X_clean, regime_features]), regime_labels

    def predict(self, X):
        """Predict using GMM regime detection with proper feature pipeline."""
        import numpy as np
        X_clean = self.imputer.transform(X)
        X_enh, regime_labels = self._enhance(X_clean)
        X_scaled = self.scaler.transform(X_enh)

        preds = np.zeros(len(X_scaled), dtype=float)
        for r, reg in enumerate(self.regressors):
            m = (regime_labels == r)
            if m.any():
                preds[m] = reg.predict(X_scaled[m])
        return preds



class OnlineChangeHeuristic:
    def __init__(self, window_size=20, variance_threshold=1.5):
        self.window_size = window_size
        self.variance_threshold = variance_threshold
        self.window = []
        self.var_prev = 0.0
        self.change_points = []
        self.mean = 0.0
        self.precision = 1.0
        
    def update(self, x, idx):
        """Update with new observation and detect change points deterministically."""
        import numpy as np
        self.window.append(x)
        if len(self.window) > self.window_size:
            self.window.pop(0)
        
        if len(self.window) == self.window_size:
            v_now = np.var(self.window)
            if self.var_prev > 0 and v_now > self.var_prev * self.variance_threshold:
                self.change_points.append(idx)
                self.var_prev = v_now
            else:
                self.var_prev = v_now
        else:
            self.var_prev = np.var(self.window) if len(self.window) > 1 else 0.0
        
        # Update running statistics
        self.precision += 1
        self.mean = (self.mean * (self.precision - 1) + x) / self.precision
        
        return self.mean, self.precision, len(self.change_points)
    
    def predict(self, X):
        """Predict using BOCPD state."""
        import numpy as np
        predictions = []
        for i, x in enumerate(X):
            mean, precision, n_changes = self.update(x, i)
            # Use current state for prediction
            predictions.append(mean)
        return np.array(predictions)

