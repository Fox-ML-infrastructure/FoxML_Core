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
Unified Training Interface - Mega Script + Modular Integration
Combines all mega script functionality with modular system maintainability.
"""


import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import all components
from .core.determinism import set_global_determinism, ensure_deterministic_environment
from .core.environment import setup_training_environment, setup_gpu_environment
from .memory.memory_manager import MemoryManager
from .preprocessing.mega_script_pipeline import MegaScriptPreprocessor
from .processing.cross_sectional import CrossSectionalProcessor
from .processing.polars_optimizer import PolarsOptimizer

logger = logging.getLogger(__name__)

class UnifiedTrainingInterface:
    """Unified training interface combining mega script power with modular maintainability."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize all components
        self._setup_environment()
        self._initialize_components()
        
    def _setup_environment(self):
        """Set up the training environment."""
        # Set global determinism
        set_global_determinism(self.config.get('seed', 42))
        ensure_deterministic_environment()
        
        # Setup training environment
        setup_training_environment(self.config)
        setup_gpu_environment()
        
        logger.info("✅ Unified training environment configured")
    
    def _initialize_components(self):
        """Initialize all training components."""
        # Memory management
        self.memory_manager = MemoryManager(self.config)
        
        # Preprocessing pipeline
        self.preprocessor = MegaScriptPreprocessor(self.config)
        
        # Cross-sectional processing
        self.cross_sectional = CrossSectionalProcessor(self.config)
        
        # Polars optimization
        self.polars_optimizer = PolarsOptimizer(self.config)
        
        logger.info("✅ All training components initialized")
    
    def train_model(self, trainer, X: np.ndarray, y: np.ndarray, 
                   timestamps: Optional[np.ndarray] = None,
                   symbols: Optional[np.ndarray] = None,
                   **kwargs) -> Any:
        """Train a model with full mega script functionality."""
        
        logger.info(f"🚀 Starting unified training with {len(X)} samples")
        
        # Memory monitoring
        self.memory_manager.log_memory_usage("training start")
        
        try:
            # 1. Apply mega script preprocessing
            X_processed, y_processed = self.preprocessor.preprocess(X, y, timestamps, symbols)
            
            if len(X_processed) == 0:
                logger.warning("No data remaining after preprocessing")
                return None
            
            # 2. Apply cross-sectional normalization (if timestamps available)
            if timestamps is not None:
                X_processed, y_processed = self.cross_sectional.normalize_cross_sectional(
                    X_processed, y_processed, timestamps
                )
            
            # 3. Train the model
            logger.info("🎯 Training model with unified interface...")
            start_time = time.time()
            
            model = trainer.train(X_processed, y_processed, **kwargs)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Model training completed in {elapsed:.2f} seconds")
            
            # 4. Memory cleanup
            self.memory_manager.cleanup()
            
            return model
            
        except Exception as e:
            logger.error(f"Unified training failed: {e}")
            self.memory_manager.cleanup()
            return None
    
    def train_with_validation(self, trainer, X: np.ndarray, y: np.ndarray,
                             timestamps: Optional[np.ndarray] = None,
                             symbols: Optional[np.ndarray] = None,
                             test_size: float = 0.2,
                             **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Train model with cross-sectional validation."""
        
        logger.info(f"🚀 Starting unified training with validation: {len(X)} samples")
        
        try:
            # 1. Apply preprocessing
            X_processed, y_processed = self.preprocessor.preprocess(X, y, timestamps, symbols)
            
            if len(X_processed) == 0:
                logger.warning("No data remaining after preprocessing")
                return None, {}
            
            # 2. Create time-aware splits
            if timestamps is not None:
                X_train, X_val, y_train, y_val = self.cross_sectional.create_time_aware_splits(
                    X_processed, y_processed, timestamps, test_size
                )
            else:
                # Fallback to random split
                from sklearn.model_selection import train_test_split
                X_train, X_val, y_train, y_val = train_test_split(
                    X_processed, y_processed, test_size=test_size, random_state=42
                )
            
            # 3. Train model
            model = self.train_model(trainer, X_train, y_train, **kwargs)
            
            if model is None:
                return None, {}
            
            # 4. Evaluate model
            y_pred = trainer.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            mae = np.mean(np.abs(y_val - y_pred))
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse),
                'r2': 1 - (mse / np.var(y_val))
            }
            
            logger.info(f"✅ Validation metrics: MSE={mse:.6f}, MAE={mae:.6f}")
            
            return model, metrics
            
        except Exception as e:
            logger.error(f"Unified training with validation failed: {e}")
            return None, {}
    
    def train_with_cross_validation(self, trainer, X: np.ndarray, y: np.ndarray,
                                  timestamps: Optional[np.ndarray] = None,
                                  symbols: Optional[np.ndarray] = None,
                                  n_splits: int = 5,
                                  **kwargs) -> Tuple[Any, Dict[str, float]]:
        """Train model with cross-sectional cross-validation."""
        
        logger.info(f"🚀 Starting unified training with CV: {len(X)} samples, {n_splits} splits")
        
        try:
            # 1. Apply preprocessing
            X_processed, y_processed = self.preprocessor.preprocess(X, y, timestamps, symbols)
            
            if len(X_processed) == 0:
                logger.warning("No data remaining after preprocessing")
                return None, {}
            
            # 2. Apply cross-sectional validation
            if timestamps is not None:
                cv_results = self.cross_sectional.apply_cross_sectional_validation(
                    X_processed, y_processed, timestamps, trainer, n_splits
                )
            else:
                # CRITICAL: Use PurgedTimeSeriesSplit instead of standard K-Fold
                # Standard K-Fold shuffles data randomly, causing temporal leakage
                from scripts.utils.purged_time_series_split import PurgedTimeSeriesSplit
                from sklearn.model_selection import cross_val_score
                
                # Use conservative default purge (60m target = 17 bars)
                # If target_column is available in kwargs, we could extract horizon
                # For now, use safe default
                purge_overlap = 17  # 60m / 5m + 5 buffer
                purged_cv = PurgedTimeSeriesSplit(n_splits=n_splits, purge_overlap=purge_overlap)
                
                cv_scores = cross_val_score(trainer, X_processed, y_processed, cv=purged_cv)
                cv_results = {
                    'mean_score': np.mean(cv_scores),
                    'std_score': np.std(cv_scores),
                    'scores': cv_scores.tolist()
                }
            
            # 3. Train final model on all data
            final_model = self.train_model(trainer, X_processed, y_processed, **kwargs)
            
            return final_model, cv_results
            
        except Exception as e:
            logger.error(f"Unified training with CV failed: {e}")
            return None, {}
    
    def get_training_stats(self, X: np.ndarray, y: np.ndarray,
                          timestamps: Optional[np.ndarray] = None,
                          symbols: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        
        stats = {
            'data_shape': X.shape,
            'target_shape': y.shape,
            'memory_usage': self.memory_manager.get_memory_usage(),
            'preprocessing_stats': self.preprocessor.get_preprocessing_stats(X, y, X, y)
        }
        
        if timestamps is not None:
            stats['cross_sectional_stats'] = self.cross_sectional.get_cross_sectional_stats(
                X, y, timestamps
            )
        
        return stats
    
    def optimize_for_large_dataset(self, df: pd.DataFrame, 
                                  processing_func: callable) -> pd.DataFrame:
        """Optimize processing for large datasets."""
        
        logger.info(f"🔧 Optimizing large dataset processing: {df.shape}")
        
        # Use Polars optimization
        optimized_df = self.polars_optimizer.optimize_dataframe(df)
        
        # Process with chunking if needed
        if len(df) > self.config.get('large_dataset_threshold', 1000000):
            result_df = self.polars_optimizer.process_large_dataset(
                optimized_df, processing_func
            )
        else:
            result_df = processing_func(optimized_df)
        
        return result_df
