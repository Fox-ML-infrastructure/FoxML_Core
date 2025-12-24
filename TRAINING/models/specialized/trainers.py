# MIT License - see LICENSE file

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


"""Training functions for specialized models."""

# Import dependencies
from TRAINING.models.specialized.wrappers import OnlineChangeHeuristic
from TRAINING.models.specialized.predictors import ChangePointPredictor, GANPredictor

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

def train_changepoint_heuristic(X, y, config):
    """Train online change point heuristic model.
    
    This implements a heuristic change point detection algorithm
    for identifying regime changes in financial time series.
    """
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
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
        
        # Train change point heuristic
        cp_heuristic = OnlineChangeHeuristic()
        
        # Online learning: process data sequentially
        for i in range(len(X_clean)):
            cp_heuristic.update(float(np.mean(X_clean[i])), i)  # Use mean of features as signal
        
        # Build aligned features (length N)
        cp_indicator = np.zeros(len(X_clean), dtype=np.float32)
        if cp_heuristic.change_points:
            cp_indicator[np.array(cp_heuristic.change_points, dtype=int)] = 1.0
        prev_cp = np.roll(cp_indicator, 1)
        prev_vol = np.roll(np.std(X_clean, axis=1), 1)
        
        X_with_changes = np.column_stack([X_clean, cp_indicator, prev_cp, prev_vol])
        
        # Load test_size and random_state from config
        test_size = 0.2  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
            except Exception:
                pass
        
        random_state = 42  # Default fallback
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                random_state = BASE_SEED
        except Exception:
            pass
        
        X_train, X_val, y_train, y_val = train_test_split(X_with_changes, y_clean, test_size=test_size, random_state=random_state)

        # Final regressor on BOCPD features
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Store change point heuristic state for inference
        model.cp_heuristic = cp_heuristic
        model.imputer = imputer
        
        # Wrap in ChangePointPredictor to handle feature engineering at predict time
        return ChangePointPredictor(model, cp_heuristic, imputer)
        
    except ImportError:
        logger.error("Required libraries not available for BOCPD")
        return None

def train_ftrl_proximal(X, y, config):
    """Train FTRL-Proximal model."""
    try:
        from sklearn.linear_model import SGDRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean).astype(np.float32)
        
        # Load hyperparameters from config
        alpha = 1e-5  # Default fallback
        random_state = 42  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                alpha = float(get_cfg("models.sgd.alpha", default=1e-5, config_name="model_config"))
            except Exception:
                pass
        
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                random_state = BASE_SEED
        except Exception:
            pass
        
        model = SGDRegressor(
            loss='squared_error',  # Fixed: was 'squared_loss'
            penalty='elasticnet',
            l1_ratio=0.15,
            alpha=alpha,
            learning_rate='adaptive',
            eta0=0.01,
            random_state=random_state,
            max_iter=1000
        )
        
        # Split data
        test_size = 0.2  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
            except Exception:
                pass
        
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_clean, test_size=test_size, random_state=random_state)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Store scaler with model
        model.scaler = scaler
        model.imputer = imputer
        
        return model
        
    except ImportError:
        logger.error("Required libraries not available for FTRL-Proximal")
        return None

def train_vae(X, y, config):
    """Train Variational Autoencoder model with Keras-native adapter."""
    try:
        from ml.vae_adapter import train_vae_safe
        import numpy as np
        
        # Use the Keras-native adapter
        model = train_vae_safe(
            X=X,
            y=y,
            config=config,
            X_va=config.get("X_val"),
            y_va=config.get("y_val"),
            device=TF_DEVICE
        )
        
        return model
        
    except ImportError:
        logger.error("TensorFlow not available for VAE")
        return None
    except Exception as e:
        logger.error(f"Error training VAE: {e}")
        import traceback
        logger.error(f"VAE traceback: {traceback.format_exc()}")
        return None

def train_gan(X, y, config):
    """Train Generative Adversarial Network model."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        import numpy as np
        import hashlib  # move up

        n_features = X.shape[1]
        latent_dim = 32
        logger.info(f"🧠 GAN training on {TF_DEVICE}")

        with tf.device(TF_DEVICE):
            def build_generator():
                inputs = layers.Input(shape=(latent_dim,))
                x = layers.Dense(128, activation='relu')(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.2)(x)
                x = layers.Dense(512, activation='relu')(x)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.2)(x)
                outputs = layers.Dense(n_features, activation='tanh')(x)
                return Model(inputs, outputs)

            def build_discriminator():
                inputs = layers.Input(shape=(n_features,))
                x = layers.Dense(512, activation='relu')(inputs)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(256, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                x = layers.Dense(128, activation='relu')(x)
                x = layers.Dropout(0.3)(x)
                outputs = layers.Dense(1, activation='sigmoid')(x)
                return Model(inputs, outputs)

            generator = build_generator()
            discriminator = build_discriminator()

            z = layers.Input(shape=(latent_dim,))
            validity = discriminator(generator(z))
            gan = Model(z, validity)

            # Load learning_rate from config
            learning_rate = 0.0002  # Default fallback
            if _CONFIG_AVAILABLE:
                try:
                    learning_rate = float(get_cfg("models.gan.learning_rate", default=0.0002, config_name="model_config"))
                except Exception:
                    pass
            
            discriminator.compile(optimizer=Adam(learning_rate=learning_rate, beta_1=0.5),
                                  loss='binary_crossentropy', metrics=['accuracy'])
            # Load learning_rate from config
            gan_lr = 0.0002  # Default fallback
            if _CONFIG_AVAILABLE:
                try:
                    gan_lr = float(get_cfg("models.gan.learning_rate", default=0.0002, config_name="model_config"))
                except Exception:
                    pass
            gan.compile(optimizer=Adam(learning_rate=gan_lr, beta_1=0.5),
                        loss='binary_crossentropy')

        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import MinMaxScaler
        imputer = SimpleImputer(strategy='median')
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(imputer.fit_transform(X))

        batch_size = 64
        epochs = 1000

        # Use seeded RNG for reproducible batch selection
        rng_batch = np.random.RandomState(42)
        
        for epoch in range(epochs):
            # Train discriminator
            discriminator.trainable = True
            idx = rng_batch.randint(0, X_scaled.shape[0], batch_size)
            real_data = X_scaled[idx]

            # Deterministic noise (hash of real samples)
            noise = np.zeros((batch_size, latent_dim))
            for i, row in enumerate(real_data):
                row_hash = hashlib.md5(row.tobytes()).hexdigest()
                seed = int(row_hash[:8], 16) % (2**32)
                rng = np.random.RandomState(seed)
                noise[i] = rng.normal(0, 1, latent_dim)

            fake_data = generator.predict(noise, verbose=0)
            d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
            _ = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train generator (freeze discriminator)
            discriminator.trainable = False
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
            discriminator.trainable = True

            if epoch % 100 == 0:
                logger.info(f"Epoch {epoch}, G loss: {g_loss:.4f}")

        # Regressor using the same deterministic-noise scheme used in predict()
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()

        synth_noise = np.zeros((len(X_scaled), latent_dim))
        for i, row in enumerate(X_scaled):
            row_hash = hashlib.md5(row.tobytes()).hexdigest()
            seed = int(row_hash[:8], 16) % (2**32)
            rng = np.random.RandomState(seed)
            synth_noise[i] = rng.normal(0, 1, latent_dim)

        synthetic_features = generator.predict(synth_noise, verbose=0)
        combined_features = np.concatenate([X_scaled, synthetic_features], axis=1)
        regressor.fit(combined_features, y)

        model = GANPredictor(generator, imputer, scaler, regressor)
        return model

    except ImportError:
        logger.error("TensorFlow not available for GAN")
        return None
    except Exception as e:
        logger.error(f"Error training GAN: {e}")
        return None

def train_ensemble(X, y, config):
    """Train Ensemble model."""
    try:
        from sklearn.ensemble import VotingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        
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
        
        # Create ensemble of different models
        # Load hyperparameters from config
        dt_random_state = 42  # Default fallback
        rf_n_estimators = 50  # Default fallback
        rf_random_state = 42  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                rf_n_estimators = int(get_cfg("models.random_forest.n_estimators", default=50, config_name="model_config"))
            except Exception:
                pass
        
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                dt_random_state = BASE_SEED
                rf_random_state = BASE_SEED
        except Exception:
            pass
        
        models = [
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor(random_state=dt_random_state)),
            ('rf', RandomForestRegressor(n_estimators=rf_n_estimators, random_state=rf_random_state))
        ]
        
        model = VotingRegressor(models)
        
        # Split data
        test_size = 0.2  # Default fallback
        random_state = 42  # Default fallback
        if _CONFIG_AVAILABLE:
            try:
                test_size = float(get_cfg("preprocessing.validation.test_size", default=0.2, config_name="preprocessing_config"))
            except Exception:
                pass
        
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                random_state = BASE_SEED
        except Exception:
            pass
        
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=test_size, random_state=random_state)
        
        # Train model
        model.fit(X_train, y_train)
        
        return model
        
    except ImportError:
        logger.error("Required libraries not available for Ensemble")
        return None

def train_meta_learning(X, y, config):
    """Train Meta-Learning model with GPU acceleration.
    
    Note: This is not true MAML but rather multi-task pretraining.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean).astype(np.float32)
        
        # Real Meta-Learning implementation using Model-Agnostic Meta-Learning (MAML)
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Meta-learning neural network
        n_features = X_scaled.shape[1]
        
        logger.info(f"🧠 MetaLearning training on {TF_DEVICE}")
        
        # Meta-learner architecture
        with tf.device(TF_DEVICE):
            inputs = layers.Input(shape=(n_features,))
            x = layers.Dense(256, activation='relu')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            x = layers.Dense(64, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.2)(x)
            outputs = layers.Dense(1, dtype="float32")(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=_get_learning_rate()),
            loss='mse',
            metrics=['mae']
        )
        
        # Meta-learning training (simplified MAML)
        # Create multiple tasks by splitting data
        from sklearn.model_selection import KFold
        random_state = 42  # Default fallback
        try:
            from TRAINING.common.determinism import BASE_SEED
            if BASE_SEED is not None:
                random_state = BASE_SEED
        except Exception:
            pass
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # Train on multiple tasks
        for train_idx, val_idx in kf.split(X_scaled):
            X_task = X_scaled[train_idx]
            y_task = y_clean[train_idx]
            
            # Quick adaptation training
            model.fit(
                X_task, y_task,
                epochs=50,
                batch_size=256,  # Reduced for memory efficiency
                verbose=0
            )
        
        # Final training on full dataset
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        ]
        
        model.fit(
            X_scaled, y_clean,
            epochs=1000,
            batch_size=1024,
            callbacks=callbacks,
            verbose=0
        )
        
        # Store scaler with model
        model.scaler = scaler
        model.imputer = imputer
        
        return model
        
    except ImportError:
        logger.error("Required libraries not available for Meta-Learning")
        return None

def train_multitask_temporal(seq, device, loss_weights=None):
    """Train true temporal multi-task model with multiple heads for different horizons."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model, callbacks, optimizers
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        Xtr, Xva = seq["X_tr"], seq["X_va"]
        ytr, yva = seq["y_tr"], seq["y_va"]     # shape (N, n_tasks)
        task_names = seq["task_names"]
        n_tasks = ytr.shape[1]
        if loss_weights is None:
            loss_weights = {t:1.0 for t in task_names}
        
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
            inp = layers.Input(shape=Xtr2.shape[1:])
            x = layers.Conv1D(128, 5, padding="causal", activation="relu")(inp)
            x = layers.Conv1D(64, 3, padding="causal", activation="relu")(x)
            x = layers.GlobalAveragePooling1D()(x)
            x = layers.Dense(128, activation="relu")(x)
            x = layers.Dropout(0.2)(x)

            outs = {t: layers.Dense(1, name=t)(x) for t in task_names}
            model = Model(inp, list(outs.values()))
            model.compile(
                optimizer=optimizers.Adam(1e-3),
                loss={t:"mse" for t in task_names},
                loss_weights=loss_weights
            )

        ytr_dict = {t: ytr[:, i] for i, t in enumerate(task_names)}
        yva_dict = {t: yva[:, i] for i, t in enumerate(task_names)}

        cb = [callbacks.EarlyStopping(patience=20, restore_best_weights=True),
              callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-6)]
        model.fit(Xtr2, ytr_dict, validation_data=(Xva2, yva_dict),
                  epochs=1000, batch_size=1024, verbose=0, callbacks=cb)
        
        # Store preprocessors for inference
        model.imputer = imp
        model.scaler = sc
        return model
        
    except ImportError:
        logger.error("TensorFlow not available for MultiTask")
        return None
    except Exception as e:
        logger.error(f"Error training MultiTask: {e}")
        return None

def train_multi_task(X, y, config):
    """Train Multi-Task model with GPU acceleration."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean).astype(np.float32)
        
        # Real Multi-Task learning implementation
        import tensorflow as tf
        from tensorflow.keras import layers, Model
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Multi-task neural network
        n_features = X_scaled.shape[1]
        
        logger.info(f"🧠 MultiTask training on {TF_DEVICE}")
        
        # Detect if we have multiple targets (y should be 2D for multi-task)
        is_multi_target = len(y_clean.shape) > 1 and y_clean.shape[1] > 1
        
        # Shared layers
        with tf.device(TF_DEVICE):
            inputs = layers.Input(shape=(n_features,))
            shared = layers.Dense(256, activation='relu')(inputs)
            shared = layers.BatchNormalization()(shared)
            shared = layers.Dropout(0.3)(shared)
            shared = layers.Dense(128, activation='relu')(shared)
            shared = layers.BatchNormalization()(shared)
            shared = layers.Dropout(0.3)(shared)
            shared = layers.Dense(64, activation='relu')(shared)
            shared = layers.BatchNormalization()(shared)
            shared = layers.Dropout(0.2)(shared)
            
            if is_multi_target:
                # Multiple task heads
                n_tasks = y_clean.shape[1]
                task_outputs = []
                task_names = []
                for i in range(n_tasks):
                    task_name = f'task_{i+1}'
                    task_names.append(task_name)
                    task_head = layers.Dense(32, activation='relu')(shared)
                    task_head = layers.Dropout(0.1)(task_head)
                    task_head = layers.Dense(1, name=task_name)(task_head)
                    task_outputs.append(task_head)
                
                model = Model(inputs=inputs, outputs=task_outputs)
                
                # Create loss and metrics dictionaries
                loss_dict = {name: 'mse' for name in task_names}
                metrics_dict = {name: 'mae' for name in task_names}
                
                model.compile(
                    optimizer=Adam(learning_rate=_get_learning_rate()),
                    loss=loss_dict,
                    metrics=metrics_dict
                )
            else:
                # Single task head (backward compatibility)
                task1_output = layers.Dense(32, activation='relu')(shared)
                task1_output = layers.Dropout(0.1)(task1_output)
                task1_output = layers.Dense(1, name='task1')(task1_output)
                
                model = Model(inputs=inputs, outputs=task1_output)
                model.compile(
                    optimizer=Adam(learning_rate=_get_learning_rate()),
                    loss={'task1': 'mse'},
                    metrics={'task1': 'mae'}
                )
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=20, restore_best_weights=True),
            ReduceLROnPlateau(patience=10, factor=0.5, min_lr=1e-6)
        ]
        
        # Train model
        if is_multi_target:
            # Multi-target training
            y_dict = {f'task_{i+1}': y_clean[:, i] for i in range(y_clean.shape[1])}
            model.fit(
                X_scaled, y_dict,
                epochs=1000,
                batch_size=256,  # Reduced for memory efficiency
                callbacks=callbacks,
                verbose=0
            )
        else:
            # Single target training (backward compatibility)
            model.fit(
                X_scaled, {'task1': y_clean},
                epochs=1000,
                batch_size=256,  # Reduced for memory efficiency
                callbacks=callbacks,
                verbose=0
            )
        
        # Store scaler with model
        model.scaler = scaler
        model.imputer = imputer
        
        return model
        
    except ImportError:
        logger.error("Required libraries not available for Multi-Task")
        return None

def train_lightgbm_ranker(X, y, groups, X_val=None, y_val=None, groups_val=None, cpu_only=False, num_threads=12, rank_labels="dense", feat_cols=None):
    """Train LightGBM with ranking objective for cross-sectional training - FIXED VERSION."""
    try:
        # Use the fixed LightGBM ranking implementation
        from ml.lightgbm_ranking_fix import train_lightgbm_ranker_safe
        logger.info("🔧 Using fixed LightGBM ranking implementation")
        return train_lightgbm_ranker_safe(X, y, groups, X_val, y_val, groups_val, cpu_only, num_threads, rank_labels, feat_cols)
    except Exception as e:
        logger.error(f"LightGBM ranker training failed: {e}")
        return None

def train_xgboost_ranker(X, y, groups, X_val=None, y_val=None, groups_val=None, cpu_only=False, num_threads=12, rank_labels="dense", feat_cols=None):
    """Train XGBoost with ranking objective for cross-sectional training."""
    try:
        import xgboost as xgb
        
        # Convert continuous targets to ranks for ranking objective
        y_ranks, rank_method = _convert_to_ranks(y, groups, rank_labels)
        
        # For XGBoost ranking, we need to convert continuous values to integer relevance scores (0-31)
        # Scale the ranks to 0-31 range for NDCG compatibility
        if rank_method != 'raw':
            # Convert ranks to integer relevance scores (0-31)
            y_ranks_scaled = np.clip(np.round(y_ranks * 31.0 / np.max(y_ranks)), 0, 31).astype(np.int32)
        else:
            # For raw values, scale to 0-31 range
            y_min, y_max = np.min(y_ranks), np.max(y_ranks)
            if y_max > y_min:
                y_ranks_scaled = np.clip(np.round((y_ranks - y_min) * 31.0 / (y_max - y_min)), 0, 31).astype(np.int32)
            else:
                y_ranks_scaled = np.full_like(y_ranks, 15, dtype=np.int32)  # Default to middle relevance
        
        feature_names = feat_cols if feat_cols is not None else [str(i) for i in range(X.shape[1])]
        dtrain = xgb.DMatrix(X, label=y_ranks_scaled, feature_names=feature_names)
        dtrain.set_group(groups)
        
        base_params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@10",
            "ndcg_exp_gain": False,  # Disable exponential NDCG to allow continuous relevance scores
            "max_depth": 0,          # Use grow_policy instead
            "min_child_weight": 16,  # Less restrictive for better quality
            "subsample": 0.9,        # More data for better quality
            "colsample_bytree": 0.9, # More features for better quality
            "eta": 0.05,            # Slightly higher learning rate
            "lambda": 1.5,           # L2 regularization
            "seed": 42,
            "seed_per_iteration": True,
            "nthread": num_threads
        }
        
        if cpu_only:
            params = _xgb_params_cpu(base_params)
        else:
            params = _xgb_params_with_fallback(base_params)
        
        # Add validation set if provided and not empty
        if X_val is not None and y_val is not None and groups_val is not None and len(X_val) > 0:
            # Convert validation targets to ranks and scale to 0-31 range
            y_val_ranks, _ = _convert_to_ranks(y_val, groups_val, rank_labels)
            if rank_method != 'raw':
                y_val_ranks_scaled = np.clip(np.round(y_val_ranks * 31.0 / np.max(y_val_ranks)), 0, 31).astype(np.int32)
            else:
                y_val_min, y_val_max = np.min(y_val_ranks), np.max(y_val_ranks)
                if y_val_max > y_val_min:
                    y_val_ranks_scaled = np.clip(np.round((y_val_ranks - y_val_min) * 31.0 / (y_val_max - y_val_min)), 0, 31).astype(np.int32)
                else:
                    y_val_ranks_scaled = np.full_like(y_val_ranks, 15, dtype=np.int32)
            
            dval = xgb.DMatrix(X_val, label=y_val_ranks_scaled, feature_names=feature_names)
            dval.set_group(groups_val)
            evals = [(dtrain, 'train'), (dval, 'val')]
        else:
            evals = [(dtrain, 'train')]
        
        # Clear GPU memory before training to reduce fragmentation
        try:
            import gc
            gc.collect()
            if hasattr(xgb, 'clear_cache'):
                xgb.clear_cache()
        except:
            pass
        
        # Train the model with GPU OOM fallback
        try:
            model = xgb.train(params, dtrain, num_boost_round=50000,  # Balanced for quality vs speed
                              evals=evals, early_stopping_rounds=1000)
        except Exception as train_error:
            error_msg = str(train_error)
            if "cudaErrorMemoryAllocation" in error_msg or "bad_alloc" in error_msg:
                logger.warning(f"💥 XGBoost GPU OOM during training ({error_msg}), falling back to CPU")
                # Fallback to CPU parameters
                cpu_params = {**params, 'tree_method': 'hist', 'device': 'cpu'}
                model = xgb.train(cpu_params, dtrain, num_boost_round=50000,
                                  evals=evals, early_stopping_rounds=1000)
            else:
                raise train_error
        
        # Store rank method in model for metadata
        if model is not None:
            model.rank_method = rank_method
        return model
    except Exception as e:
        logger.error(f"XGBoost ranker training failed: {e}")
        return None

def safe_predict(model, X_val, meta):
    """Safe prediction with proper preprocessing and model type handling."""
    try:
        import pandas as pd
        import numpy as np
        
        # 1) dataframe & column order (always reindex to ensure correct order)
        if not hasattr(X_val, "reindex"):
            cols = meta.get("features") if meta else None
            if not cols:
                cols = range(np.shape(X_val)[1])
            X_val = pd.DataFrame(X_val, columns=cols)
        if 'features' in meta and meta.get('features'):
            X_val = X_val.reindex(columns=meta['features'], fill_value=0.0)

        # 2) apply any saved preprocessors *first* (unless the model handles it)
        if not getattr(model, "handles_preprocessing", False):
            imputer = getattr(model, "imputer", None)
            scaler  = getattr(model, "scaler",  None)
            if imputer is not None:
                X_val = imputer.transform(X_val)
            if scaler is not None:
                X_val = scaler.transform(X_val)

        # 3) boosters
        try:
            import xgboost as xgb
            if isinstance(model, xgb.Booster):
                # Use the same feature names that were used during training
                feature_names = getattr(model, 'feature_names', None)
                if feature_names is None:
                    feature_names = [str(i) for i in range(X_val.shape[1])]
                dm = xgb.DMatrix(np.asarray(X_val), feature_names=feature_names)
                if hasattr(model, "best_iteration") and model.best_iteration is not None:
                    return model.predict(dm, iteration_range=(0, model.best_iteration + 1))
                if hasattr(model, "best_ntree_limit") and model.best_ntree_limit:
                    return model.predict(dm, ntree_limit=model.best_ntree_limit)
                return model.predict(dm)
        except Exception as e:
            # Log the specific error for debugging
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"XGBoost prediction failed: {e}")
            pass
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster) or getattr(model, "__class__", None).__name__ == "Booster":
                return model.predict(np.asarray(X_val), num_iteration=getattr(model, "best_iteration", None))
        except Exception:
            pass

        # 4) keras
        try:
            import tensorflow as _tf
            if isinstance(model, _tf.keras.Model):
                return np.asarray(model.predict(np.asarray(X_val), verbose=0)).ravel()
        except Exception:
            pass

        # 5) sklearn fallback (but not for XGBoost/LightGBM)
        try:
            import xgboost as xgb
            import lightgbm as lgb
            if isinstance(model, (xgb.Booster, lgb.Booster)):
                raise ValueError("XGBoost/LightGBM model should have been handled above")
        except ImportError:
            pass
        
        X_np = np.asarray(X_val, dtype=np.float32)
        return model.predict(X_np)
        
    except Exception as e:
        family = (meta or {}).get('family', 'unknown')
        logger.warning(f"Prediction failed for {family}: {e}")
        y_pred = np.zeros(len(X_val))
        
        # Check if this is a silent failure (all zeros predicted)
        if np.std(y_pred) < 1e-12 and np.allclose(y_pred, 0):
            logger.error(f"❌ Silent prediction failure for {meta.get('family','unknown')} - all zeros predicted")
            raise RuntimeError(f"Prediction failed for {meta.get('family','unknown')}: {e}")
        
        return y_pred



