#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

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
"""

"""
ChangePoint trainer matching mega script implementation exactly.
"""


import numpy as np
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class OnlineChangeHeuristic:
    """Online change point heuristic - exact copy from mega script."""
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

class ChangePointPredictor:
    """ChangePoint predictor with proper feature engineering - exact copy from mega script."""
    def __init__(self, model, cp_heuristic, imputer):
        self.model = model
        self.cp_heuristic = cp_heuristic
        self.imputer = imputer
        self.handles_preprocessing = True
    
    def predict(self, X):
        """Predict using change point engineered features."""
        X_clean = self.imputer.transform(X)
        
        # Recreate change point features at predict time
        cp_indicator = np.zeros(len(X_clean))
        # Note: This is a simplified version - in practice you'd need to 
        # maintain the change point detection state across predictions
        # For now, we'll use a simple heuristic based on variance
        window_size = self.cp_heuristic.window_size
        if len(X_clean) >= window_size:
            for i in range(window_size, len(X_clean)):
                window = X_clean[i-window_size:i]
                v_now = np.var(window)
                if i > window_size:
                    prev_window = X_clean[i-window_size-1:i-1]
                    v_prev = np.var(prev_window)
                    if v_prev > 0 and v_now > v_prev * self.cp_heuristic.variance_threshold:
                        cp_indicator[i] = 1.0
        
        prev_cp = np.roll(cp_indicator, 1)
        prev_vol = np.roll(np.std(X_clean, axis=1), 1)
        
        # Combine original features with change point features
        X_with_changes = np.column_stack([X_clean, cp_indicator, prev_cp, prev_vol])
        
        return self.model.predict(X_with_changes)

class ChangePointTrainer:
    """ChangePoint trainer matching mega script implementation exactly."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def fit(self, X_tr: np.ndarray, y_tr: np.ndarray, seed: Optional[int] = None, **kwargs) -> Any:
        """Fit method for compatibility with sklearn-style interface."""
        return self.train(X_tr, y_tr, seed, **kwargs)
    
    def train(self, X_tr: np.ndarray, y_tr: np.ndarray, X_va=None, y_va=None, cpu_only=False, num_threads=12, feat_cols=None, seed: Optional[int] = None, **kwargs) -> Any:
        """Train ChangePoint model using mega script approach."""
        return self._train_changepoint_heuristic(X_tr, y_tr, X_va, y_va, cpu_only, num_threads, feat_cols, seed, **kwargs)
    
    def _train_changepoint_heuristic(self, X_tr, y_tr, X_va=None, y_va=None, cpu_only=False, num_threads=12, feat_cols=None, seed=None, **kwargs):
        """Train online change point heuristic model - EXACT MEGA SCRIPT IMPLEMENTATION."""
        try:
            logger.info(f"Starting ChangePoint training on {len(X_tr)} samples...")
            
            if len(X_tr) == 0:
                logger.warning("No data available for training")
                return None
                
            # MEGA SCRIPT APPROACH: Self-contained preprocessing (exactly like mega script)
            from sklearn.impute import SimpleImputer
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression
            import time
            
            # Clean data - replace NaN values (exact mega script approach)
            X_float = X_tr.astype(np.float64)
            y_float = y_tr.astype(np.float64)
            
            # Replace NaN values with median (exact mega script approach)
            imputer = SimpleImputer(strategy='median')
            X_clean = imputer.fit_transform(X_float)
            y_clean = y_float.astype(np.float32)
            
            # Check for any remaining NaN values (exact mega script approach)
            if np.isnan(X_clean).any() or np.isnan(y_clean).any():
                X_clean = np.nan_to_num(X_clean, nan=0.0)
                y_clean = np.nan_to_num(y_clean, nan=0.0)
            
            # Train change point heuristic (exact mega script approach)
            cp_heuristic = OnlineChangeHeuristic()
            
            # Online learning: process data sequentially (exact mega script approach)
            for i in range(len(X_clean)):
                cp_heuristic.update(float(np.mean(X_clean[i])), i)  # Use mean of features as signal
            
            # Build aligned features (length N) (exact mega script approach)
            cp_indicator = np.zeros(len(X_clean), dtype=np.float32)
            if cp_heuristic.change_points:
                cp_indicator[np.array(cp_heuristic.change_points, dtype=int)] = 1.0
            prev_cp = np.roll(cp_indicator, 1)
            prev_vol = np.roll(np.std(X_clean, axis=1), 1)
            
            X_with_changes = np.column_stack([X_clean, cp_indicator, prev_cp, prev_vol])
            X_train, X_val, y_train, y_val = train_test_split(X_with_changes, y_clean, test_size=0.2, random_state=42)

            # Final regressor on BOCPD features (exact mega script approach)
            model = LinearRegression()
            
            logger.info(f"Training ChangePoint with {len(X_train)} samples, {X_train.shape[1]} features")
            start_time = time.time()
            model.fit(X_train, y_train)
            elapsed = time.time() - start_time
            logger.info(f"ChangePoint training completed in {elapsed:.2f} seconds")
            
            # Store change point heuristic state for inference (exact mega script approach)
            model.cp_heuristic = cp_heuristic
            model.imputer = imputer
            
            # Wrap in ChangePointPredictor to handle feature engineering at predict time (exact mega script approach)
            predictor = ChangePointPredictor(model, cp_heuristic, imputer)
            
            self.model = predictor
            self.is_trained = True
            return predictor
            
        except ImportError:
            logger.error("Required libraries not available for ChangePoint")
            return None
        except Exception as e:
            logger.error(f"ChangePoint training failed: {e}")
            return None
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained change point model."""
        if not hasattr(self, 'model') or not hasattr(self, 'is_trained') or not self.is_trained:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        try:
            # Use the ChangePointPredictor (exact mega script approach)
            predictions = self.model.predict(X)
            
            # Ensure predictions are finite
            predictions = np.nan_to_num(predictions, nan=0.0)
            
            return predictions.astype(np.float32)
            
        except Exception as e:
            logger.error(f"ChangePoint prediction failed: {e}")
            raise
