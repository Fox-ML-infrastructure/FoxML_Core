# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


"""Extended training functions and helpers for specialized models."""

# Import shared constants
from TRAINING.models.specialized.constants import TF_DEVICE

# Import base trainers for any dependencies
from TRAINING.models.specialized.trainers import *

# Add CONFIG directory to path for centralized config loading
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Try to import config loader
_CONFIG_AVAILABLE = False
try:
    from config_loader import get_cfg
    _CONFIG_AVAILABLE = True
except ImportError:
    logger.debug("Config loader not available; using hardcoded defaults")

def _get_learning_rate(default: float = 0.001) -> float:
    """Get learning_rate from config, with fallback to default."""
    if _CONFIG_AVAILABLE:
        try:
            return float(get_cfg("optimizer.learning_rate", default=default, config_name="optimizer_config"))
        except Exception:
            pass
    return default  # Final fallback

def _train_lgb_cpu(train_data, val_data=None, is_ranker=False, num_threads=12):
    """Train LightGBM on CPU only."""
    import lightgbm as lgb
    
    # Load hyperparameters from config
    learning_rate = 0.03  # Default fallback
    seed = 42  # Default fallback
    if _CONFIG_AVAILABLE:
        try:
            learning_rate = float(get_cfg("models.lightgbm.learning_rate", default=0.03, config_name="model_config"))
        except Exception:
            pass
    
    try:
        from TRAINING.common.determinism import BASE_SEED
        if BASE_SEED is not None:
            seed = BASE_SEED
    except Exception:
        pass
    
    base = dict(
        boosting_type='gbdt',
        max_bin=255,                 # fine on CPU; 255 is fast/compact
        learning_rate=learning_rate,
        feature_fraction=0.8,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l1=1.0, lambda_l2=10.0,
        verbose=-1, seed=seed,
        feature_fraction_seed=seed, bagging_seed=seed,
        num_threads=num_threads, force_row_wise=True
    )
    
    # Add deterministic parameter only for LightGBM 4.x+
    if int(lgb.__version__.split('.')[0]) >= 4:
        base['deterministic'] = True

    if is_ranker:
        base.update(objective='lambdarank', metric='ndcg', eval_at=[3,5,10],
                    num_leaves=255, min_data_in_leaf=2000, max_depth=-1, force_row_wise=True, lambdarank_truncation_level=10)
    else:
        base.update(objective='regression', metric='rmse',
                    num_leaves=255, min_data_in_leaf=2000, max_depth=-1, force_row_wise=True)

    # Guard against empty validation set
    if val_data is not None:
        try:
            val_data.construct()
            if val_data.num_data() == 0:
                logger.warning("Empty validation set, skipping early stopping")
                val_data = None
        except Exception:
            logger.warning("Could not inspect validation set, proceeding without pre-check")
    
    model = lgb.train(
        base, train_data,
        valid_sets=[val_data] if val_data else None,
        num_boost_round=20000 if is_ranker else 50000,  # Balanced for quality vs speed
        callbacks=([lgb.early_stopping(200 if is_ranker else 1000), lgb.log_evaluation(0)]
                   if val_data else [lgb.log_evaluation(0)])
    )
    
    # Log early stopping summary
    if hasattr(model, 'best_iteration') and model.best_iteration:
        logger.info(f"Early stopping at iteration {model.best_iteration}")
    
    return model

def _train_lgb_with_fallback(train_data, val_data=None, is_ranker=False, num_threads=12):
    """Train LightGBM with GPU fallback to CPU."""
    import lightgbm as lgb
    
    base = dict(
        boosting_type='gbdt', max_bin=511, learning_rate=_get_learning_rate(0.03),
        feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1,
        lambda_l1=1.0, lambda_l2=10.0, verbose=-1, seed=42,
        feature_fraction_seed=42, bagging_seed=42,
        num_threads=num_threads, force_row_wise=True
    )
    
    # Add deterministic parameter only for LightGBM 4.x+
    if int(lgb.__version__.split('.')[0]) >= 4:
        base['deterministic'] = True
    
    if is_ranker:
        base.update(objective='lambdarank', metric='ndcg', eval_at=[3,5,10], 
                   num_leaves=255, min_data_in_leaf=2000, max_depth=-1, lambdarank_truncation_level=10)
    else:
        base.update(objective='regression', metric='rmse', 
                   num_leaves=255, min_data_in_leaf=2000, max_depth=-1)

    # LightGBM uses 'device_type' across 3.x and 4.x
    gpu_key = 'device_type'
    params = {**base, gpu_key: 'gpu'}
    if is_ranker:
        # Memory-friendlier path on GPU too; mirrors CPU behavior.
        params['force_row_wise'] = True
    try:
        return lgb.train(params, train_data, valid_sets=[val_data] if val_data else None,
                         num_boost_round=50000 if not is_ranker else 20000,  # Balanced for quality vs speed
                         callbacks=[lgb.early_stopping(500 if not is_ranker else 200), lgb.log_evaluation(0)] if val_data else [lgb.log_evaluation(0)])
    except Exception as e:
        logger.warning(f"LGBM GPU unavailable, falling back to CPU: {e}")
        params.pop(gpu_key, None)
        return lgb.train(params, train_data, valid_sets=[val_data] if val_data else None,
                         num_boost_round=50000 if not is_ranker else 20000,  # Balanced for quality vs speed
                         callbacks=[lgb.early_stopping(500 if not is_ranker else 200), lgb.log_evaluation(0)] if val_data else [lgb.log_evaluation(0)])

def train_lightgbm(X_tr, y_tr, X_va=None, y_va=None, cpu_only=False, num_threads=12, feat_cols=None):
    """Train LightGBM regression model with validation set."""
    try:
        import lightgbm as lgb
        
        # Create datasets with real feature names if available
        feature_names = feat_cols if feat_cols is not None else [str(i) for i in range(X_tr.shape[1])]
        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names, params={"data_random_seed": 42})
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data, feature_name=feature_names, params={"data_random_seed": 42}) if X_va is not None and y_va is not None else None
        
        if cpu_only:
            return _train_lgb_cpu(train_data, val_data, is_ranker=False, num_threads=num_threads)
        else:
            return _train_lgb_with_fallback(train_data, val_data, is_ranker=False, num_threads=num_threads)
        
    except ImportError:
        logger.error("LightGBM not available")
        return None

def _xgb_params_cpu(base):
    """Get XGBoost parameters for CPU-only training."""
    return {**base, 'tree_method':'hist'}

def _xgb_params_with_fallback(base):
    """Get XGBoost parameters with GPU fallback to CPU."""
    # Start with CPU params as default
    p = {**base, 'tree_method':'hist'}
    
    # Skip GPU probe on non-POSIX systems to reduce log noise
    if os.name != 'posix':
        return p
    
    # Try GPU if not cpu_only (with timeout to avoid hanging)
    try:
        import xgboost as xgb
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("GPU probe timed out")
        
        # Set 5-second timeout for GPU probe (POSIX only)
        try:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)
        except (AttributeError, OSError):
            # Windows doesn't support SIGALRM, fall back to CPU
            return p
        
        try:
            # More robust GPU test - try actual training on tiny dataset
            test_X = np.random.rand(10, 5).astype(np.float32)
            test_y = np.random.rand(10).astype(np.float32)
            test_dmat = xgb.DMatrix(test_X, label=test_y)
            
            # Try GPU training on test data with aggressive memory settings for 8GB+ GPU
            gpu_params = {
                **base, 
                'tree_method': 'hist',
                'device': 'cuda',
                'max_bin': 512,           # Aggressive for 8GB+ GPU (was 128)
                'max_leaves': 512,        # Aggressive for 8GB+ GPU (was 128)
                'grow_policy': 'lossguide',
                'subsample': 0.9,         # More data for better quality
                'colsample_bytree': 0.9,   # More features for better quality
                'min_child_weight': 16,   # Less restrictive for better quality
                # 'gpu_id': 0,              # Removed: conflicts with device='cuda'
                'predictor': 'gpu_predictor',  # Force GPU prediction
                'seed': 42
            }
            xgb.train(gpu_params, test_dmat, num_boost_round=1, verbose_eval=False)
            
            # If we get here, GPU works
            signal.alarm(0)  # Cancel timeout
            logger.warning("⚠️  XGBoost GPU training enabled. For strict reproducibility, consider using --cpu-only")
            return gpu_params
        finally:
            signal.alarm(0)  # Always cancel timeout
            
    except Exception as e:
        error_msg = str(e)
        if "cudaErrorMemoryAllocation" in error_msg or "bad_alloc" in error_msg:
            logger.warning(f"💥 XGBoost GPU OOM ({error_msg}), falling back to CPU")
        else:
            logger.warning(f"XGBoost GPU unavailable ({error_msg}), using CPU 'hist'")
        return p

def train_xgboost(X_tr, y_tr, X_va=None, y_va=None, cpu_only=False, num_threads=12, feat_cols=None):
    """Train XGBoost regression model with validation set."""
    try:
        import xgboost as xgb
        # Create datasets with real feature names if available
        feature_names = feat_cols if feat_cols is not None else [str(i) for i in range(X_tr.shape[1])]
        train_data = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
        
        # Base parameters - optimized for 8GB+ GPU with memory management
        base_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'min_child_weight': 8,
            'subsample': 0.9,         # More data for better quality
            'colsample_bytree': 0.9,  # More features for better quality
            'reg_alpha': 1.0,
            'reg_lambda': 12.0,
            'eta': 0.03,
            'seed': 42,
            'seed_per_iteration': True,
            'nthread': num_threads,
            # 'gpu_id': 0,              # Removed: conflicts with device='cuda'
            'predictor': 'gpu_predictor'  # Force GPU prediction
        }
        
        if cpu_only:
            params = _xgb_params_cpu(base_params)
        else:
            params = _xgb_params_with_fallback(base_params)
        
        # Clear GPU memory before training to reduce fragmentation
        try:
            import gc
            gc.collect()
            if hasattr(xgb, 'clear_cache'):
                xgb.clear_cache()
        except:
            pass
        
        # Train model with GPU OOM fallback
        try:
            if X_va is not None and y_va is not None and len(X_va) > 0:
                val_data = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
                model = xgb.train(
                    params,
                    train_data,
                    num_boost_round=50000,
                    evals=[(val_data, 'validation')],
                    early_stopping_rounds=500,
                    verbose_eval=False
                )
                # Log best iteration for debugging
                if hasattr(model, 'best_iteration'):
                    logger.info(f"XGBoost best iteration: {model.best_iteration}")
            else:
                model = xgb.train(
                    params,
                    train_data,
                    num_boost_round=50000,
                    verbose_eval=False
                )
        except Exception as train_error:
            error_msg = str(train_error)
            if "cudaErrorMemoryAllocation" in error_msg or "bad_alloc" in error_msg:
                logger.warning(f"💥 XGBoost GPU OOM during training ({error_msg}), falling back to CPU")
                # Fallback to CPU parameters
                cpu_params = {**params, 'tree_method': 'hist', 'device': 'cpu'}
                if X_va is not None and y_va is not None and len(X_va) > 0:
                    val_data = xgb.DMatrix(X_va, label=y_va, feature_names=feature_names)
                    model = xgb.train(
                        cpu_params,
                        train_data,
                        num_boost_round=50000,
                        evals=[(val_data, 'validation')],
                        early_stopping_rounds=500,
                        verbose_eval=False
                    )
                else:
                    model = xgb.train(
                        cpu_params,
                        train_data,
                        num_boost_round=50000,
                        verbose_eval=False
                    )
            else:
                raise train_error
        
        # Store feature names for prediction consistency
        if model is not None:
            model.feature_names = feature_names
        
        return model
        
    except ImportError:
        logger.error("XGBoost not available")
        return None

def train_mlp(X_tr, y_tr, X_va=None, y_va=None):
    """Train MLP model with GPU acceleration."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Impute and scale features
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(imputer.fit_transform(X_tr)).astype('float32')
        X_vas = scaler.transform(imputer.transform(X_va)).astype('float32') if X_va is not None else None
        
        n_features = X_trs.shape[1]
        
        logger.info(f"🧠 MLP training on {TF_DEVICE}")
        
        # Create MLP with GPU acceleration and seeded initializers
        with tf.device(TF_DEVICE):
            inputs = layers.Input(shape=(n_features,))
            # Use seeded initializers for reproducibility
            k0 = tf.keras.initializers.GlorotUniform(seed=42)
            k1 = tf.keras.initializers.GlorotUniform(seed=43)
            k2 = tf.keras.initializers.GlorotUniform(seed=44)
            x = layers.Dense(512, activation='relu', kernel_initializer=k0)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3, seed=42)(x)
            x = layers.Dense(256, activation='relu', kernel_initializer=k1)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3, seed=43)(x)
            x = layers.Dense(128, activation='relu', kernel_initializer=k2)(x)
            x = layers.Dropout(0.2, seed=44)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.Dropout(0.1, seed=45)(x)
            outputs = layers.Dense(1, dtype="float32")(x)
            
            model = Model(inputs, outputs)
            model.compile(
                optimizer=Adam(learning_rate=_get_learning_rate()),
                loss='mse',
                metrics=['mae']
            )
        
        # Clear GPU memory before training
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Training with GPU fallback
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        try:
            if X_vas is not None:
                history = model.fit(
                    X_trs, y_tr,
                    validation_data=(X_vas, y_va),
                    epochs=100,
                    batch_size=256,  # Reduced for memory efficiency
                    callbacks=callbacks,
                    verbose=0
                )
            else:
                history = model.fit(
                    X_trs, y_tr,
                    epochs=100,
                    batch_size=256,  # Reduced for memory efficiency
                    callbacks=callbacks,
                    verbose=0
                )
        except Exception as e:
            if "Dst tensor is not initialized" in str(e) or "GPU" in str(e):
                logger.warning(f"GPU training failed: {e}, falling back to CPU")
                # Clear GPU memory and retry on CPU
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Recreate model on CPU
                with tf.device('/CPU:0'):
                    inputs = layers.Input(shape=(n_features,))
                    k0 = tf.keras.initializers.GlorotUniform(seed=42)
                    k1 = tf.keras.initializers.GlorotUniform(seed=43)
                    k2 = tf.keras.initializers.GlorotUniform(seed=44)
                    x = layers.Dense(512, activation='relu', kernel_initializer=k0)(inputs)
                    x = layers.BatchNormalization()(x)
                    x = layers.Dropout(0.3, seed=42)(x)
                    x = layers.Dense(256, activation='relu', kernel_initializer=k1)(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Dropout(0.3, seed=43)(x)
                    x = layers.Dense(128, activation='relu', kernel_initializer=k2)(x)
                    x = layers.Dropout(0.2, seed=44)(x)
                    x = layers.Dense(64, activation='relu')(x)
                    x = layers.Dropout(0.1, seed=45)(x)
                    outputs = layers.Dense(1, dtype="float32")(x)
                    
                    model = Model(inputs, outputs)
                    model.compile(
                        optimizer=Adam(learning_rate=_get_learning_rate()),
                        loss='mse',
                        metrics=['mae']
                    )
                
                # Retry training on CPU
                if X_vas is not None:
                    history = model.fit(
                        X_trs, y_tr,
                        validation_data=(X_vas, y_va),
                        epochs=100,
                        batch_size=256,  # Reduced for memory efficiency
                        callbacks=callbacks,
                        verbose=0
                    )
                else:
                    history = model.fit(
                        X_trs, y_tr,
                        epochs=100,
                        batch_size=256,  # Reduced for memory efficiency
                        callbacks=callbacks,
                        verbose=0
                    )
            else:
                raise e
        
        # Attach scaler and imputer for inference
        model.scaler = scaler
        model.imputer = imputer
        
        return model
        
    except ImportError:
        logger.error("MLP not available")
        return None

def train_cnn1d_temporal(seq, device):
    """Train true temporal CNN1D with causal convolutions over time."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, callbacks, optimizers
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        Xtr, Xva = seq["X_tr"], seq["X_va"]            # (N,L,F)
        ytr, yva = seq["y_tr"][:, :1], seq["y_va"][:, :1]  # single-task here
        
        # Preprocess sequences: impute and scale
        N, L, F = Xtr.shape
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr2 = sc.fit_transform(imp.fit_transform(Xtr.reshape(-1, F))).reshape(N, L, F)
        if Xva is not None:
            Xva2 = sc.transform(imp.transform(Xva.reshape(-1, F))).reshape(Xva.shape[0], L, F)
        else:
            Xva2 = None

        with tf.device(device):
            inp = layers.Input(shape=Xtr2.shape[1:])    # (L,F)
            x = layers.Conv1D(128, 7, padding="causal", activation="relu")(inp)
            x = layers.BatchNormalization()(x)
            x = layers.Conv1D(128, 5, padding="causal", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
            x = layers.BatchNormalization()(x)
            x = layers.GlobalAveragePooling1D()(x)     # only uses history → causal
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation="relu")(x)
            out = layers.Dense(1, dtype="float32")(x)
            model = Model(inp, out)
            model.compile(optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

        cb = [callbacks.EarlyStopping(patience=20, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)]
        model.fit(Xtr2, ytr, validation_data=(Xva2, yva), epochs=1000, batch_size=1024, verbose=0, callbacks=cb)
        
        # Store preprocessors for inference
        model.imputer = imp
        model.scaler = sc
        return model
        
    except ImportError:
        logger.error("TensorFlow not available for CNN1D")
        return None
    except Exception as e:
        logger.error(f"Error training CNN1D: {e}")
        return None

def train_tabcnn(X_tr, y_tr, X_va=None, y_va=None):
    """Train TabCNN model with TFSeriesRegressor wrapper.
    
    NOTE: This is tabular CNN - it convolves over features, not time.
    This learns feature interactions in a 1D convolution manner.
    For true temporal modeling, use CNN1D (temporal by default).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Impute and scale features
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(imputer.fit_transform(X_tr)).astype('float32')
        X_vas = scaler.transform(imputer.transform(X_va)).astype('float32') if X_va is not None else None
        
        n_feat = X_trs.shape[1]
        
        logger.info(f"🧠 TabCNN training on {TF_DEVICE}")
        
        # Clear GPU memory before training
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Create improved model with better architecture and seeded initializers
        try:
            with tf.device(TF_DEVICE):
                # Use proper Keras Input pattern instead of Sequential
                from tensorflow.keras import layers, Model, initializers
                
                inputs = layers.Input(shape=(n_feat, 1))
                # Use seeded initializers for reproducibility
                k0 = initializers.HeNormal(seed=42)
                k1 = initializers.HeNormal(seed=43)
                k2 = initializers.HeNormal(seed=44)
                
                x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer=k0)(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.2, seed=42)(x)
                
                x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=k1)(x)
                x = layers.BatchNormalization()(x)
                x = layers.MaxPooling1D(pool_size=2)(x)
                x = layers.Dropout(0.3, seed=43)(x)
                
                x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=k2)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3, seed=44)(x)
                
                x = layers.Flatten()(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.4, seed=45)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.3, seed=46)(x)
                x = layers.Dense(64, activation='relu')(x)
                x = layers.Dropout(0.2, seed=47)(x)
                outputs = layers.Dense(1, dtype="float32")(x)
                
                model = Model(inputs, outputs)
        except Exception as e:
            if "Dst tensor is not initialized" in str(e) or "GPU" in str(e):
                logger.warning(f"TabCNN GPU model creation failed: {e}, falling back to CPU")
                # Clear GPU memory and retry on CPU
                tf.keras.backend.clear_session()
                gc.collect()
                
                # Recreate model on CPU
                with tf.device('/CPU:0'):
                    from tensorflow.keras import layers, Model, initializers
                    
                    inputs = layers.Input(shape=(n_feat, 1))
                    k0 = initializers.HeNormal(seed=42)
                    k1 = initializers.HeNormal(seed=43)
                    k2 = initializers.HeNormal(seed=44)
                    
                    x = layers.Conv1D(filters=128, kernel_size=5, activation='relu', kernel_initializer=k0)(inputs)
                    x = layers.BatchNormalization()(x)
                    x = layers.Dropout(0.2, seed=42)(x)
                    
                    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', kernel_initializer=k1)(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.MaxPooling1D(pool_size=2)(x)
                    x = layers.Dropout(0.3, seed=43)(x)
                    
                    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', kernel_initializer=k2)(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Dropout(0.3, seed=44)(x)
                    
                    x = layers.Flatten()(x)
                    x = layers.Dense(256, activation='relu')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Dropout(0.4, seed=45)(x)
                    x = layers.Dense(128, activation='relu')(x)
                    x = layers.BatchNormalization()(x)
                    x = layers.Dropout(0.3, seed=46)(x)
                    x = layers.Dense(64, activation='relu')(x)
                    x = layers.Dropout(0.2, seed=47)(x)
                    outputs = layers.Dense(1, dtype="float32")(x)
                    
                    model = Model(inputs, outputs)
            else:
                raise e
        
        # Better optimizer with learning rate scheduling
        model.compile(
            optimizer=Adam(learning_rate=_get_learning_rate(), beta_1=0.9, beta_2=0.999),
            loss='mse',
            metrics=['mae']
        )
        
        # Reshape for training
        X_tr3 = X_trs.reshape(X_trs.shape[0], n_feat, 1)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_vas is not None else 'loss', 
                         patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_vas is not None else 'loss', 
                            factor=0.5, patience=10, min_lr=1e-6)
        ]
        
        # Train with validation if available
        if X_vas is not None:
            X_va3 = X_vas.reshape(X_vas.shape[0], n_feat, 1)
            model.fit(
                X_tr3, y_tr, 
                validation_data=(X_va3, y_va), 
                epochs=1000,  # More epochs for 10M rows
                batch_size=256,  # Reduced from 1024 for memory efficiency
                callbacks=callbacks,
                verbose=0
            )
        else:
            model.fit(
                X_tr3, y_tr, 
                epochs=1000,
                batch_size=256,  # Reduced for memory efficiency
                callbacks=callbacks,
                verbose=0
            )
        
        # Return wrapped model
        return TFSeriesRegressor(model, imputer, scaler, n_feat)
        
    except ImportError:
        logger.error("TensorFlow not available for TabCNN")
        return None

def train_lstm_temporal(seq, device):
    """Train true temporal LSTM with unidirectional processing over time."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, callbacks, optimizers
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        Xtr, Xva = seq["X_tr"], seq["X_va"]
        ytr, yva = seq["y_tr"][:, :1], seq["y_va"][:, :1]
        
        # Preprocess sequences: impute and scale
        N, L, F = Xtr.shape
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr2 = sc.fit_transform(imp.fit_transform(Xtr.reshape(-1, F))).reshape(N, L, F)
        if Xva is not None:
            Xva2 = sc.transform(imp.transform(Xva.reshape(-1, F))).reshape(Xva.shape[0], L, F)
        else:
            Xva2 = None

        with tf.device(device):
            inp = layers.Input(shape=Xtr2.shape[1:])          # (L,F)
            x = layers.LSTM(128, return_sequences=True)(inp) # uni, not bidirectional
            x = layers.LayerNormalization()(x)
            x = layers.LSTM(64)(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation="relu")(x)
            out = layers.Dense(1, dtype="float32")(x)
            model = Model(inp, out)
            model.compile(optimizers.Adam(1e-3, clipnorm=1.0), loss="mse", metrics=["mae"])

        cb = [callbacks.EarlyStopping(patience=20, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)]
        model.fit(Xtr2, ytr, validation_data=(Xva2, yva), epochs=1000, batch_size=1024, verbose=0, callbacks=cb)
        
        # Store preprocessors for inference
        model.imputer = imp
        model.scaler = sc
        return model
        
    except ImportError:
        logger.error("TensorFlow not available for LSTM")
        return None
    except Exception as e:
        logger.error(f"Error training LSTM: {e}")
        return None

def train_tablstm(X_tr, y_tr, X_va=None, y_va=None):
    """Train TabLSTM model with TFSeriesRegressor wrapper.
    
    NOTE: This is tabular LSTM - it processes features as sequences.
    This learns feature interactions in a sequential manner.
    For true temporal modeling, use LSTM (temporal by default).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Impute and scale features
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        X_trs = scaler.fit_transform(imputer.fit_transform(X_tr)).astype('float32')
        X_vas = scaler.transform(imputer.transform(X_va)).astype('float32') if X_va is not None else None
        
        n_feat = X_trs.shape[1]
        
        logger.info(f"🧠 TabLSTM training on {TF_DEVICE}")
        
        # Clear GPU memory before training
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Enable mixed precision for memory efficiency
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Create improved model with better architecture and seeded initializers
        with tf.device(TF_DEVICE):
            # Use proper Keras Input pattern instead of Sequential
            from tensorflow.keras import layers, Model, initializers
            
            inputs = layers.Input(shape=(n_feat, 1))
            # Use seeded initializers for reproducibility
            k0 = initializers.GlorotUniform(seed=42)
            k1 = initializers.GlorotUniform(seed=43)
            k2 = initializers.GlorotUniform(seed=44)
            
            # REDUCED LSTM sizes for memory efficiency
            x = layers.LSTM(64, return_sequences=True, kernel_initializer=k0)(inputs)  # Reduced from 256
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3, seed=42)(x)
            
            x = layers.LSTM(32, return_sequences=True, kernel_initializer=k1)(x)  # Reduced from 128
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3, seed=43)(x)
            
            x = layers.LSTM(16, return_sequences=False, kernel_initializer=k2)(x)  # Reduced from 64
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4, seed=44)(x)
            
            # Dense layers with better regularization - REDUCED for memory efficiency
            x = layers.Dense(32, activation='relu')(x)  # Reduced from 128
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3, seed=45)(x)
            x = layers.Dense(16, activation='relu')(x)  # Reduced from 64
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2, seed=46)(x)
            outputs = layers.Dense(1, dtype="float32")(x)
            
            model = Model(inputs, outputs)
            
            # Better optimizer with learning rate scheduling
            model.compile(
                optimizer=Adam(learning_rate=_get_learning_rate(), beta_1=0.9, beta_2=0.999),
                loss='mse',
                metrics=['mae']
            )
        
        # Reshape for training
        X_tr3 = X_trs.reshape(X_trs.shape[0], n_feat, 1)
        
        # Callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_vas is not None else 'loss', 
                         patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_vas is not None else 'loss', 
                            factor=0.5, patience=12, min_lr=1e-6)
        ]
        
        # Train with validation if available
        if X_vas is not None:
            X_va3 = X_vas.reshape(X_vas.shape[0], n_feat, 1)
            model.fit(
                X_tr3, y_tr, 
                validation_data=(X_va3, y_va), 
                epochs=1000,  # More epochs for 10M rows
                batch_size=256,  # Reduced from 1024 for memory efficiency
                callbacks=callbacks,
                verbose=0
            )
        else:
            model.fit(
                X_tr3, y_tr, 
                epochs=1000,
                batch_size=256,  # Reduced for memory efficiency
                callbacks=callbacks,
                verbose=0
            )
        
        # Return wrapped model
        return TFSeriesRegressor(model, imputer, scaler, n_feat)
        
    except ImportError:
        logger.error("TensorFlow not available for LSTM")
        return None

def train_transformer_temporal(seq, device, d_model=96, n_heads=8, n_blocks=3, ff_mult=4):
    """Train true temporal Transformer with causal attention over time."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, callbacks, optimizers
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        Xtr, Xva = seq["X_tr"], seq["X_va"]
        ytr, yva = seq["y_tr"][:, :1], seq["y_va"][:, :1]
        L, F = Xtr.shape[1], Xtr.shape[2]
        
        # Preprocess sequences: impute and scale
        N = Xtr.shape[0]
        imp = SimpleImputer(strategy="median")
        sc = StandardScaler()
        Xtr2 = sc.fit_transform(imp.fit_transform(Xtr.reshape(-1, F))).reshape(N, L, F)
        if Xva is not None:
            Xva2 = sc.transform(imp.transform(Xva.reshape(-1, F))).reshape(Xva.shape[0], L, F)
        else:
            Xva2 = None

        class PositionalEncoding(layers.Layer):
            def call(self, x):
                # simple learned PE
                pe = self.add_weight("pe", shape=(1, L, d_model), initializer="zeros", trainable=True)
                return x + pe

        def encoder_block(x):
            # self-attn over time with causal mask
            attn = layers.MultiHeadAttention(n_heads, key_dim=d_model//n_heads)
            y = attn(x, x, use_causal_mask=True)
            x = layers.LayerNormalization()(x + y)
            y = layers.Dense(ff_mult*d_model, activation="relu")(x)
            y = layers.Dense(d_model)(y)
            x = layers.LayerNormalization()(x + y)
            return x

        with tf.device(device):
            inp = layers.Input(shape=(L, F))
            x = layers.Dense(d_model)(inp)           # per-time linear projection of features
            x = PositionalEncoding()(x)
            for _ in range(n_blocks):
                x = encoder_block(x)
            x = layers.Lambda(lambda t: t[:, -1, :])(x)   # take representation at last (current) time
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(64, activation="relu")(x)
            out = layers.Dense(1, dtype="float32")(x)
            model = Model(inp, out)
            model.compile(optimizers.Adam(1e-3), loss="mse", metrics=["mae"])

        cb = [callbacks.EarlyStopping(patience=20, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)]
        model.fit(Xtr2, ytr, validation_data=(Xva2, yva), epochs=1000, batch_size=1024, verbose=0, callbacks=cb)
        
        # Store preprocessors for inference
        model.imputer = imp
        model.scaler = sc
        return model
        
    except ImportError:
        logger.error("TensorFlow not available for Transformer")
        return None
    except Exception as e:
        logger.error(f"Error training Transformer: {e}")
        return None

def train_tabtransformer(X, y, config, X_va=None, y_va=None):
    """Train TabTransformer model for tabular data.
    
    NOTE: This is tabular Transformer - it does attention over features.
    This learns feature interactions through attention mechanisms.
    For true temporal modeling, use Transformer (temporal by default).
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Add preprocessing
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X = scaler.fit_transform(imputer.fit_transform(X))
        
        # Real Transformer implementation for tabular data
        n_features = X.shape[1]
        
        logger.info(f"🧠 TabTransformer training on {TF_DEVICE}")
        
        # Clear GPU memory before training
        tf.keras.backend.clear_session()
        import gc
        gc.collect()
        
        # Enable mixed precision for memory efficiency
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Input layer
        with tf.device(TF_DEVICE):
            from tensorflow.keras import initializers
            
            inputs = layers.Input(shape=(n_features,))
            
            # Use seeded initializers for reproducibility
            k0 = initializers.GlorotUniform(seed=42)
            k1 = initializers.GlorotUniform(seed=43)
            
            # Feature embedding (convert features to embeddings)
            # Project each feature to d_model dimensions - REDUCED for memory efficiency
            d_model = 32  # Reduced from 64
            x = layers.Dense(d_model * n_features, activation='relu', kernel_initializer=k0)(inputs)
            x = layers.Dropout(0.1, seed=42)(x)
            
            # Reshape to (n_features, d_model) for proper attention over features
            x = layers.Reshape((n_features, d_model))(x)
            
            # Multi-head attention over features - REDUCED for memory efficiency
            attention = layers.MultiHeadAttention(
                num_heads=4,  # Reduced from 8
                key_dim=8,    # Reduced from 16
                dropout=0.1
            )
            x = attention(x, x)
            x = layers.Dropout(0.1)(x)
            
            # Feed forward - REDUCED for memory efficiency
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(32, activation='relu')(x)  # Reduced from 64
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(16, activation='relu')(x)  # Reduced from 32
            x = layers.Dropout(0.1)(x)
            outputs = layers.Dense(1, dtype="float32")(x)
            
            model = Model(inputs, outputs)
            model.compile(
                optimizer=Adam(learning_rate=_get_learning_rate()),
                loss='mse',
                metrics=['mae']
            )
        
        # Use provided validation data if available, otherwise do random split
        if X_va is None or y_va is None:
            from sklearn.model_selection import train_test_split
            # Load test_size and seed from config
            test_size = 0.2  # Default fallback
            seed = 42  # Default fallback
            if _CONFIG_AVAILABLE:
                try:
                    test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
                except Exception:
                    pass
            
            try:
                from TRAINING.common.determinism import BASE_SEED
                if BASE_SEED is not None:
                    seed = BASE_SEED
            except Exception:
                pass
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, seed=seed)
        else:
            X_train, y_train = X, y
            # Preprocess validation data with same imputer and scaler
            X_val = scaler.transform(imputer.transform(X_va))
            y_val = y_va
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        ]
        
        # Train model with reduced batch size for memory efficiency
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=1000,
            batch_size=256,  # Reduced from 1024 for memory efficiency
            callbacks=callbacks,
            verbose=0
        )
        
        # Attach preprocessors for inference
        model.scaler = scaler
        model.imputer = imputer
        
        return model
        
    except ImportError:
        logger.error("TensorFlow not available for Transformer")
        return None
    except Exception as e:
        logger.error(f"Error training Transformer: {e}")
        return None

def train_reward_based(X, y, config):
    """Train Reward-Based model."""
    try:
        # Gate on dataset size - GradientBoostingRegressor is slow on large datasets
        if len(X) > 50_000_000:  # 50M rows threshold (increased from 10M)
            logger.warning(f"RewardBased skipped on large dataset ({len(X):,} rows). Consider using HistGradientBoostingRegressor for better performance.")
            return None
            
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.impute import SimpleImputer
        
        # Self-contained preprocessing (trees don't need scaling)
        X_float = X.astype(np.float64, copy=False)
        y_float = y.astype(np.float64, copy=False)
        imputer = SimpleImputer(strategy='median')
        X_clean = imputer.fit_transform(X_float)
        y_clean = np.nan_to_num(y_float, nan=0.0).astype(np.float32)
        
        # Load hyperparameters from config
        n_estimators = 300  # Default fallback
        learning_rate = 0.03  # Default fallback
        seed = 42  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                n_estimators = int(get_cfg("models.gradient_boosting.n_estimators", default=300, config_name="model_config"))
                learning_rate = float(get_cfg("models.gradient_boosting.learning_rate", default=0.03, config_name="model_config"))
            except Exception:
                pass
        
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                seed = BASE_SEED
        except Exception:
            pass
        
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,        # Increased for 10M rows
            learning_rate=learning_rate,      # Reduced for stability at scale
            max_depth=10,           # Increased for 10M rows capacity
            min_samples_split=200,  # Increased regularization for scale
            min_samples_leaf=100,   # Increased regularization for scale
            subsample=0.7,          # More subsampling for 10M rows
            seed=seed
        )
        
        # Train on all provided rows (outer split is already time-aware)
        model.fit(X_clean, y_clean)

        # Attach preprocessor for inference consistency
        model.imputer = imputer
        return model
        
    except ImportError:
        logger.error("Required libraries not available for Reward-Based")
        return None

def train_quantile_lightgbm(X, y, config, X_va=None, y_va=None):
    """Train Quantile LightGBM model."""
    try:
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        
        # Use provided validation data if available, otherwise split
        if X_va is not None and y_va is not None:
            X_train, y_train = X, y
            X_val, y_val = X_va, y_va
        else:
            # Load test_size and seed from config
            test_size = 0.2  # Default fallback
            seed = 42  # Default fallback
            if _CONFIG_AVAILABLE:
                try:
                    test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
                except Exception:
                    pass
            
            try:
                from TRAINING.common.determinism import BASE_SEED
                if BASE_SEED is not None:
                    seed = BASE_SEED
            except Exception:
                pass
            
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, seed=seed)
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=[str(i) for i in range(X_train.shape[1])])
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=[str(i) for i in range(X_val.shape[1])])
        
        # Parameters for quantile regression
        alpha = config.get('quantile_alpha', 0.5)
        params = {
            'objective': 'quantile',
            'metric': 'quantile',
            'alpha': alpha,
            'boosting_type': 'gbdt',
            'num_leaves': 255,  # Increased for better capacity
            'learning_rate': 0.03,  # Reduced for stability
            'feature_fraction': 0.8,
            'bagging_fraction': 0.9,
            'bagging_freq': 1,
            'lambda_l1': 1.0,  # Added regularization
            'lambda_l2': 10.0,
            'min_data_in_leaf': 1500,  # Added regularization
            'verbose': -1,
            'seed': 42,
            'deterministic': True  # Added for reproducibility
        }
        
        # Train model with more rounds and better early stopping for 10M rows
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=50000,  # Balanced for quality vs speed
            callbacks=[lgb.early_stopping(2000), lgb.log_evaluation(0)]  # More patience for 10M rows
        )
        
        return model
        
    except ImportError:
        logger.error("LightGBM not available for Quantile LightGBM")
        return None

def train_ngboost(X, y, config, X_va=None, y_va=None):
    """Train NGBoost model with bulletproof adapter."""
    try:
        # Gate on dataset size - NGBoost is slow on large datasets
        if len(X) > 100_000_000:  # 100M rows threshold for NGBoost (increased from 15M)
            logger.warning(f"NGBoost skipped on large dataset ({len(X):,} rows). Consider using faster alternatives for large datasets.")
            return None
            
        from ml.ngboost_adapter import fit_ngboost_safe
        import numpy as np
        
        # Use the bulletproof adapter
        model = fit_ngboost_safe(
            X_tr=X,
            y_tr=y,
            X_va=X_va,
            y_va=y_va,
        n_estimators=1500,  # This is model-specific, could be configurable but is typically set per model
        learning_rate=_get_learning_rate(0.05),
            early_stopping_rounds=200
        )
        
        return model
        
    except ImportError:
        logger.error("NGBoost not available")
        return None
    except Exception as e:
        logger.error(f"NGBoost training failed: {e}")
        import traceback
        logger.error(f"NGBoost traceback: {traceback.format_exc()}")
        return None

def train_gmm_regime(X, y, config):
    """Train GMM-based regime detection model.
    
    This implements a Gaussian Mixture Model for regime detection
    and regime-specific regression models for financial time series.
    """
    try:
        from sklearn.mixture import GaussianMixture
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        # Clean data - replace NaN values
        X_float = X.astype(np.float64)
        y_float = y.astype(np.float64)
        
        # Replace NaN values with median
        imputer = SimpleImputer(strategy='median')
        X_clean = imputer.fit_transform(X_float)
        y_clean = y_float.astype(np.float32)
        
        # Check for any remaining NaN values
        if np.isnan(X_clean).any() or np.isnan(y_clean).any():
            X_clean = np.nan_to_num(X_clean, nan=0.0)
            y_clean = np.nan_to_num(y_clean, nan=0.0)
        
        # Real HMM: Use Gaussian Mixture for regime detection
        n_regimes = 3  # Bull, Bear, Sideways
        
        # Split data first, then compute regimes per split
        test_size = 0.2  # Default fallback
        seed = 42  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
            except Exception:
                pass
        
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                seed = BASE_SEED
        except Exception:
            pass
        
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=test_size, seed=seed)
        
        # Fit GMM on training data only
        gmm = GaussianMixture(n_components=n_regimes, seed=seed)
        gmm.fit(X_train)
        
        # Build enhanced features on TRAIN only
        def enhance(gmm, Xc):
            post = gmm.predict_proba(Xc)
            labels = post.argmax(1)
            feats = np.column_stack([
                labels.reshape(-1, 1),
                post,
                np.mean(Xc, axis=1, keepdims=True),
                np.std(Xc, axis=1, keepdims=True),
            ])
            return np.column_stack([Xc, feats]), labels

        Xtr_enh, train_labels = enhance(gmm, X_train)
        scaler = StandardScaler()
        Xtr_scaled = scaler.fit_transform(Xtr_enh)
        
        # Train regime-specific models on enhanced, scaled features
        regressors = []
        for r in range(n_regimes):
            sel = (train_labels == r)
            reg = LinearRegression()
            if sel.any():
                reg.fit(Xtr_scaled[sel], y_train[sel])
            else:
                reg.fit(Xtr_scaled, y_train)
            regressors.append(reg)

        model = GMMRegimeRegressor(gmm, regressors, scaler, imputer, n_regimes)
        return model
        
    except ImportError:
        logger.error("Required libraries not available for GMM Regime")
        return None

# Online Change Point Heuristic class (moved outside function for pickle compatibility)


