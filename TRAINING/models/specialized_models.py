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

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

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
        X_train, X_val, y_train, y_val = train_test_split(X_with_changes, y_clean, test_size=0.2, random_state=42)

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
        
        model = SGDRegressor(
            loss='squared_error',  # Fixed: was 'squared_loss'
            penalty='elasticnet',
            l1_ratio=0.15,
            alpha=1e-5,
            learning_rate='adaptive',
            eta0=0.01,
            random_state=42,
            max_iter=1000
        )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_clean, test_size=0.2, random_state=42)
        
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

            discriminator.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
                                  loss='binary_crossentropy', metrics=['accuracy'])
            gan.compile(optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
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
        models = [
            ('lr', LinearRegression()),
            ('dt', DecisionTreeRegressor(random_state=42)),
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42))
        ]
        
        model = VotingRegressor(models)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        
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
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Meta-learning training (simplified MAML)
        # Create multiple tasks by splitting data
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
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
                    optimizer=Adam(learning_rate=0.001),
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
                    optimizer=Adam(learning_rate=0.001),
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


def cs_metrics_by_time(y_true: np.ndarray, y_pred: np.ndarray, ts: np.ndarray) -> Dict[str, float]:
    """Calculate cross-sectional metrics per timestamp (true CS evaluation)."""
    try:
        from scipy.stats import spearmanr, pearsonr
        scipy_available = True
    except Exception:
        scipy_available = False
        
    ts = np.asarray(ts)
    ic_list, ric_list = [], []
    grp_sizes, grp_hits = [], []
    
    total_timestamps = len(np.unique(ts))
    skipped_timestamps = 0
    
    # Single pass through unique timestamps
    for t in np.unique(ts):
        m = (ts == t)
        if m.sum() <= 2:
            skipped_timestamps += 1
            continue
        y_t, pred_t = y_true[m], y_pred[m]
        
        # skip degenerate groups
        if np.std(y_t) < 1e-12 or np.std(pred_t) < 1e-12:
            skipped_timestamps += 1
            continue
        
        # Compute correlations
        if scipy_available:
            ic = pearsonr(y_t, pred_t)[0]
            ric = spearmanr(y_t, pred_t)[0]
        else:
            # Simple numpy fallback
            def _corr(a, b):
                if a.size < 2: return np.nan
                return float(np.corrcoef(a, b)[0,1])
            ic = _corr(y_t, pred_t)
            # Rank-IC fallback
            ric = _corr(y_t.argsort().argsort(), pred_t.argsort().argsort())
        
        if not np.isnan(ic): ic_list.append(ic)
        if not np.isnan(ric): ric_list.append(ric)
        
        # Hit rate per timestamp: majority vote on direction
        hit_rate_t = float(np.mean(np.sign(y_t) == np.sign(pred_t)))
        grp_sizes.append(m.sum())
        grp_hits.append(hit_rate_t)
    
    # Weight hit rate by group size
    hit_rate = float(np.average(grp_hits, weights=grp_sizes)) if grp_sizes else 0.0
    
    # Log fraction of skipped timestamps
    if total_timestamps > 0:
        skipped_fraction = skipped_timestamps / total_timestamps
        if skipped_fraction > 0.1:  # Log if more than 10% skipped
            logger.warning(f"⚠️  Skipped {skipped_timestamps}/{total_timestamps} timestamps ({skipped_fraction:.1%}) due to degenerate groups")
        else:
            logger.info(f"📊 Skipped {skipped_timestamps}/{total_timestamps} timestamps ({skipped_fraction:.1%}) due to degenerate groups")
    
    # Calculate IC_IR (Information Ratio)
    ic_arr = np.asarray(ic_list)
    ic_ir = float(ic_arr.mean() / (ic_arr.std(ddof=1) + 1e-12)) if ic_list else 0.0
    
    return {
        "mean_IC": float(np.mean(ic_list)) if ic_list else 0.0,
        "mean_RankIC": float(np.mean(ric_list)) if ric_list else 0.0,
        "IC_IR": ic_ir,
        "n_times": int(len(ic_list)),
        "hit_rate": hit_rate,
        "skipped_timestamps": skipped_timestamps,
        "total_timestamps": total_timestamps
    }

def train_model(family: str, X: np.ndarray, y: np.ndarray, config: Dict[str, Any], symbols: np.ndarray = None, cpu_only: bool = False, rank_objective: str = "on", num_threads: int = 12, feat_cols: List[str] = None, seq_lookback: int = 64, mtf_data: Dict[str, pd.DataFrame] = None, target: str = None):
    """Train a model from the specified family."""
    
    # Sequence-only families that require temporal data
    SEQ_ONLY_FAMILIES = {"CNN1D", "LSTM", "Transformer", "MultiTask"}
    
    # Check if this is a sequence-only family in cross-sectional mode
    if family in SEQ_ONLY_FAMILIES and mtf_data is None:
        logger.info(f"⏭️  Skipping {family} (requires sequence inputs for cross-sectional mode)")
        return None
    
    logger.info(f"🎯 Training {family} model (cross-sectional)...")
    
    # Memory monitoring for 10M rows
    try:
        import psutil
        process = psutil.Process()
        memory_gb = process.memory_info().rss / 1024**3
        logger.info(f"💾 Memory at training start: {memory_gb:.1f} GB")
        
        # Warn if memory is getting high
        if memory_gb > 100:  # 100GB threshold
            logger.warning(f"⚠️  High memory usage: {memory_gb:.1f} GB")
    except ImportError:
        pass
    
    # Check family capabilities
    if family not in FAMILY_CAPS:
        logger.warning(f"Model family {family} not in capabilities map. Skipping.")
        return None
    
    caps = FAMILY_CAPS[family]
    
    # No timeout - let models train as long as needed for quality
    
    # Check TensorFlow dependency
    if caps["needs_tf"] and not tf_available():
        logger.warning(f"TensorFlow missing → skipping {family}")
        return None
    
    # Check NGBoost dependency
    if family == "NGBoost" and not ngboost_available():
        logger.warning(f"NGBoost missing → skipping {family}")
        return None
    
    # Skip feature emitters for now (they need different architecture)
    if caps.get("feature_emitter", False):
        logger.warning(f"Skipping {family} (feature emitter - needs different architecture)")
        return None
    
    # Apply preprocessing pipeline for families that need it
    if (not caps["nan_ok"]) and (not caps.get("preprocess_in_family", False)):
        # Apply family pipeline for NaN-sensitive models
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        
        # Clean data
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
        
        # Assert no NaNs
        assert_no_nan(pd.DataFrame(X_scaled), list(range(X_scaled.shape[1])), f"{family} features")
        assert_no_nan(pd.DataFrame(y_clean.reshape(-1, 1)), [0], f"{family} targets")
        
        # Use cleaned data
        X = X_scaled
        y = y_clean
    
    try:
        # Route to appropriate training function
        if family == 'LightGBM':
            # Use ranker when groups provided for cross-sectional training and rank_objective is on
            if "groups" in config and config["groups"] is not None and rank_objective == "on":
                return train_lightgbm_ranker(X, y, config["groups"], 
                                            config.get("X_val"), config.get("y_val"), config.get("groups_val"), cpu_only, num_threads, config.get("rank_labels", "dense"), feat_cols)
            elif rank_objective == "on":
                logger.error("❌ LightGBM ranker requested but no groups provided. Groups are required for cross-sectional ranking.")
                return None
            else:
                return train_lightgbm(X, y, config.get("X_val"), config.get("y_val"), cpu_only, num_threads, feat_cols)
        elif family == 'XGBoost':
            # Use ranker when groups provided for cross-sectional training and rank_objective is on
            if "groups" in config and config["groups"] is not None and rank_objective == "on":
                return train_xgboost_ranker(X, y, config["groups"],
                                          config.get("X_val"), config.get("y_val"), config.get("groups_val"), cpu_only, num_threads, config.get("rank_labels", "dense"), feat_cols)
            elif rank_objective == "on":
                logger.error("❌ XGBoost ranker requested but no groups provided. Groups are required for cross-sectional ranking.")
                return None
            else:
                return train_xgboost(X, y, config.get("X_val"), config.get("y_val"), cpu_only, num_threads, feat_cols)
        elif family == 'MLP':
            return train_mlp(X, y, config.get("X_val"), config.get("y_val"))
        elif family == 'CNN1D':
            # CNN1D is now temporal by default - requires sequence data
            if mtf_data is not None and target is not None:
                seq_data = prepare_sequence_cs(mtf_data, feat_cols, [target], lookback=seq_lookback, min_cs=config.get("min_cs", 10))
                return train_cnn1d_temporal(seq_data, TF_DEVICE)
            else:
                logger.warning("CNN1D requires sequence data (mtf_data and target). Use TabCNN for tabular modeling.")
                return None
        elif family == 'LSTM':
            # LSTM is now temporal by default - requires sequence data
            if mtf_data is not None and target is not None:
                seq_data = prepare_sequence_cs(mtf_data, feat_cols, [target], lookback=seq_lookback, min_cs=config.get("min_cs", 10))
                return train_lstm_temporal(seq_data, TF_DEVICE)
            else:
                logger.warning("LSTM requires sequence data (mtf_data and target). Use TabLSTM for tabular modeling.")
                return None
        elif family == 'Transformer':
            # Transformer is now temporal by default - requires sequence data
            if mtf_data is not None and target is not None:
                seq_data = prepare_sequence_cs(mtf_data, feat_cols, [target], lookback=seq_lookback, min_cs=config.get("min_cs", 10))
                return train_transformer_temporal(seq_data, TF_DEVICE)
            else:
                logger.warning("Transformer requires sequence data (mtf_data and target). Use TabTransformer for tabular modeling.")
                return None
        elif family == 'TabCNN':
            return train_tabcnn(X, y, config.get("X_val"), config.get("y_val"))
        elif family == 'TabLSTM':
            return train_tablstm(X, y, config.get("X_val"), config.get("y_val"))
        elif family == 'TabTransformer':
            return train_tabtransformer(X, y, config, config.get("X_val"), config.get("y_val"))
        elif family == 'RewardBased':
            return train_reward_based(X, y, config)
        elif family == 'QuantileLightGBM':
            return train_quantile_lightgbm(X, y, config, config.get("X_val"), config.get("y_val"))
        elif family == 'NGBoost':
            return train_ngboost(X, y, config, config.get("X_val"), config.get("y_val"))
        elif family == 'GMMRegime':
            if len(X) < 1_000:
                logger.warning(f"GMMRegime requires at least 1000 samples for stable regimes (got {len(X)}). Skipping.")
                return None
            return train_gmm_regime(X, y, config)
        elif family == 'ChangePoint':
            if len(X) < 1_000:
                logger.warning(f"ChangePoint requires at least 1000 samples for stable detection (got {len(X)}). Skipping.")
                return None
            return train_changepoint_heuristic(X, y, config)
        elif family == 'FTRLProximal':
            return train_ftrl_proximal(X, y, config)
        elif family == 'VAE':
            return train_vae(X, y, config)
        elif family == 'GAN':
            return train_gan(X, y, config)
        elif family == 'Ensemble':
            return train_ensemble(X, y, config)
        elif family == 'MetaLearning':
            return train_meta_learning(X, y, config)
        elif family == 'MultiTask':
            # MultiTask is now temporal by default - requires sequence data
            if mtf_data is not None and target is not None:
                # Use temporal sequence model with multiple horizons
                # For MultiTask, we need to get all available targets
                all_targets = [target]  # Start with current target
                # Add other horizons if available in ALL symbols
                for horizon in [5, 10, 15, 30, 60]:  # Common horizons
                    other_target = f"fwd_ret_{horizon}m"
                    # Only add if present in ALL symbols
                    if all(other_target in df.columns for df in mtf_data.values()):
                        all_targets.append(other_target)
                seq_data = prepare_sequence_cs(mtf_data, feat_cols, all_targets, lookback=seq_lookback, min_cs=config.get("min_cs", 10))
                return train_multitask_temporal(seq_data, TF_DEVICE)
            else:
                logger.warning(
                    "MultiTask requires sequence data (mtf_data and target) for "
                    "temporal multi-horizon prediction. Skipping."
                )
                return None
        else:
            logger.warning(f"Model family {family} not implemented yet. Skipping.")
            return None
    except MemoryError as e:
        logger.error(f"❌ Out of memory training {family}: {e}")
        logger.error("💡 Try reducing batch size or MAX_ROWS_PER_BATCH")
        # Force cleanup
        import gc
        gc.collect()
        return None
    except Exception as e:
        logger.error(f"❌ Error training {family}: {e}")
        return None

def save_model(model, family: str, target: str, output_dir: str, batch_id: int = None, metadata: Dict = None):
    """Save trained model with special handling for TF models."""
    if model is None:
        return {}
    
    # Create output directory
    model_dir = Path(output_dir) / family / target
    model_dir.mkdir(parents=True, exist_ok=True)
    
    tag = f"_b{batch_id}" if batch_id is not None else ""
    
    # Handle TF models wrapped in TFSeriesRegressor (CNN1D/LSTM) OR plain Keras Models (Transformer/VAE/GAN/MetaLearning/MultiTask)
    try:
        import tensorflow as _tf
        is_keras = isinstance(model, _tf.keras.Model)
    except Exception:
        is_keras = False

    if family in {"CNN1D", "LSTM"} and hasattr(model, "model"):
        keras_path = model_dir / f"{family.lower()}_mtf{tag}.keras"
        scaler_path = model_dir / f"{family.lower()}_mtf{tag}_scaler.joblib"
        imputer_path = model_dir / f"{family.lower()}_mtf{tag}_imputer.joblib"
        meta_path = model_dir / f"{family.lower()}_mtf{tag}.meta.joblib"
        try:
            model.model.save(keras_path)
            joblib.dump(model.scaler, scaler_path, compress=3)
            joblib.dump(model.imputer, imputer_path, compress=3)
            # Save basic meta info
            basic_meta = {
                "n_feat": model.n_feat, 
                "keras_path": str(keras_path), 
                "scaler_path": str(scaler_path),
                "imputer_path": str(imputer_path),
                "family": family, 
                "target": target,
                "features": (metadata.get("features", []) if metadata else [])
            }
            joblib.dump(basic_meta, meta_path, compress=3)
            
            # ALWAYS also write the full JSON metadata if provided
            saved = {"model": str(keras_path), "scaler": str(scaler_path), "imputer": str(imputer_path), "meta": str(meta_path)}
            if metadata:
                meta_file = model_dir / f"meta{tag}.json"
                import json
                meta_file.write_text(json.dumps({**metadata, "keras_path": str(keras_path)}, indent=2), encoding='utf-8')
                saved["metadata"] = str(meta_file)
            logger.info(f"Saved TF model: {keras_path} (+ scaler + imputer + meta)")
            return saved
        except Exception as e:
            logger.error(f"Failed to save TF model {family}: {e}")
            return {}

    if is_keras:
        keras_path = model_dir / f"{family.lower()}_mtf{tag}.keras"
        try:
            model.save(keras_path)
            
            # save preprocessors if present
            saved = {"model": str(keras_path)}
            if hasattr(model, "scaler"):
                scaler_path = model_dir / f"{family.lower()}_mtf{tag}_scaler.joblib"
                joblib.dump(model.scaler, scaler_path, compress=3)
                saved["scaler"] = str(scaler_path)
            if hasattr(model, "imputer"):
                imputer_path = model_dir / f"{family.lower()}_mtf{tag}_imputer.joblib"
                joblib.dump(model.imputer, imputer_path, compress=3)
                saved["imputer"] = str(imputer_path)

            # include features in meta so safe_predict can reindex after reload
            meta_path = model_dir / f"{family.lower()}_mtf{tag}.meta.joblib"
            # Ensure features are always present - force-fill from metadata or use empty list
            features = (metadata or {}).get("features") or []
            if not features:
                logger.warning(f"No features list provided for {family}; "
                             "meta will include an empty list.")
            joblib.dump({"family": family, "target": target, "features": features}, meta_path, compress=3)
            saved["meta"] = str(meta_path)
            logger.info(f"Saved Keras model: {keras_path}")
            return saved
        except Exception as e:
            logger.error(f"Failed to save Keras model {family}: {e}")
            # fall back to joblib
    
    # XGBoost: prefer native saver (more portable)
    if family == "XGBoost":
        booster_json = model_dir / f"{family.lower()}_mtf{tag}.json"
        try:
            model.save_model(str(booster_json))
            logger.info(f"Saved XGBoost booster: {booster_json}")
            model_path = str(booster_json)
        except Exception:
            # fall back to joblib below
            model_file = model_dir / f"{family.lower()}_mtf{tag}.joblib"
            joblib.dump(model, model_file, compress=3)
            logger.info(f"Saved model: {model_file}")
            model_path = str(model_file)
    # LightGBM: prefer native saver (more portable)
    elif family == "LightGBM":
        booster_txt = model_dir / f"{family.lower()}_mtf{tag}.txt"
        try:
            import lightgbm as lgb
            if isinstance(model, lgb.Booster):
                model.save_model(str(booster_txt))
                logger.info(f"Saved LightGBM booster: {booster_txt}")
                model_path = str(booster_txt)
            else:
                raise TypeError("Not a LightGBM Booster")
        except Exception:
            # fall back to joblib
            model_file = model_dir / f"{family.lower()}_mtf{tag}.joblib"
            joblib.dump(model, model_file, compress=3)
            logger.info(f"Saved model: {model_file}")
            model_path = str(model_file)
    else:
        # Standard sklearn models
        model_file = model_dir / f"{family.lower()}_mtf{tag}.joblib"
        joblib.dump(model, model_file, compress=3)
        logger.info(f"Saved model: {model_file}")
        model_path = str(model_file)
    
    # Save metadata if provided (for all model types)
    if metadata:
        meta_file = model_dir / f"meta{tag}.json"
        import json
        # Ensure features are always present
        if metadata:
            if not metadata.get("features"):
                logger.warning("Metadata missing 'features'; writing empty list.")
                metadata = {**metadata, "features": []}
            meta_file.write_text(json.dumps(metadata, indent=2))
        logger.info(f"Saved metadata: {meta_file}")
        return {"model": model_path, "metadata": str(meta_file)}
    
    return {"model": model_path}

def _predict_temporal_model(model, Xseq):
    """Apply imputer/scaler (if present) to sequences and predict."""
    import numpy as np
    X2 = Xseq
    if hasattr(model, "imputer") and hasattr(model, "scaler"):
        N, L, F = Xseq.shape
        X2 = model.scaler.transform(model.imputer.transform(Xseq.reshape(-1, F))).reshape(N, L, F)
    # keras models may output a list (multitask); make it flat if needed
    y = model.predict(X2, verbose=0)
    if isinstance(y, (list, tuple)):
        # return first head by default; caller can index specifically
        y = y[0]
    return np.asarray(y).ravel()

def train_with_strategy(args.strategy, mtf_data: Dict[str, pd.DataFrame], target: str, families: List[str], common_features: List[str], output_dir: str, min_cs: int, max_samples_per_symbol: int = 10000, batch_id: int = None, cs_normalize: str = "per_ts_split", args=None, all_targets: set = None):
    """Train all model families for a specific interval/target with cross-sectional evaluation."""
    logger.info(f"\n🎯 Training models for target: {target} (CROSS-SECTIONAL)")
    
    # Prepare TRUE cross-sectional training data
    # Get time column from first dataframe
    if not mtf_data:
        logger.error("No data provided for training")
        return {"error": "No data provided"}
    first_df = next(iter(mtf_data.values()))
    time_col = resolve_time_col(first_df)
    
    X, y, symbols, groups, ts_index, feat_cols = prepare_training_data_cross_sectional(
        mtf_data, target, common_features, min_cs, max_samples_per_symbol, all_targets
    )
    
    if X is None:
        logger.error(f"Skipping {target} - no training data")
        return {"status": "skipped", "error": "No training data"}
    
    # Create time-aware train/validation split
    tr_idx, va_idx, train_ts, val_ts = create_time_aware_split(ts_index, train_ratio=0.8)
    
    # Split data
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_va, y_va = X[va_idx], y[va_idx]
    symbols_tr, symbols_va = symbols[tr_idx], symbols[va_idx]
    ts_tr, ts_va = ts_index[tr_idx], ts_index[va_idx]
    
    # Apply data capping after split to avoid biasing validation
    X_tr, y_tr, symbols_tr, ts_tr = cap_split(X_tr, y_tr, symbols_tr, ts_tr, args.max_rows_train, mode="random")
    X_va, y_va, symbols_va, ts_va = cap_split(X_va, y_va, symbols_va, ts_va, args.max_rows_val, mode="tail")
    
    # Re-enforce min_cs after capping
    X_tr, y_tr, symbols_tr, ts_tr = _drop_small_cs(X_tr, y_tr, symbols_tr, ts_tr, args.min_cs)
    X_va, y_va, symbols_va, ts_va = _drop_small_cs(X_va, y_va, symbols_va, ts_va, args.min_cs)
    
    # Apply per-split preprocessing (leak-free)
    logger.info("🔧 Applying per-split CS preprocessing...")
    
    # Ensure all arrays are numpy to avoid index alignment issues
    ts_tr = np.asarray(ts_tr)
    ts_va = np.asarray(ts_va)
    symbols_tr = np.asarray(symbols_tr)
    symbols_va = np.asarray(symbols_va)
    y_tr = np.asarray(y_tr)
    y_va = np.asarray(y_va)
    
    # Reconstruct dataframes for preprocessing using actual feat_cols (position-wise assignment)
    # Only keep necessary columns to save memory at scale
    df_tr = pd.DataFrame(X_tr, columns=feat_cols)
    df_tr['ts'] = ts_tr  # Only ts needed for CS transforms
    
    df_va = pd.DataFrame(X_va, columns=feat_cols)
    df_va['ts'] = ts_va  # Only ts needed for CS transforms
    
    # Apply CS transforms if requested
    TIME_COL_NAME = 'ts'  # canonical internal name
    if cs_normalize == "per_ts_split":
        # Use CLI parameters for CS transforms
        cs_block = args.cs_block if args else 32
        cs_winsor_p = args.cs_winsor_p if args else 0.01
        cs_ddof = args.cs_ddof if args else 1
        df_tr = _apply_cs_transforms_per_split(df_tr, feat_cols, TIME_COL_NAME, p=cs_winsor_p, feat_block=cs_block, ddof=cs_ddof)
        df_va = _apply_cs_transforms_per_split(df_va, feat_cols, TIME_COL_NAME, p=cs_winsor_p, feat_block=cs_block, ddof=cs_ddof)
    else:
        logger.info("🔧 Skipping CS normalization (--cs-normalize=none)")
    
    # Extract processed features
    X_tr = df_tr[feat_cols].values.astype(np.float32)
    X_va = df_va[feat_cols].values.astype(np.float32)
    
    # Free memory after extracting arrays
    del df_tr, df_va
    import gc
    gc.collect()
    
    # Additional aggressive memory cleanup
    try:
        import psutil
        process = psutil.Process()
        logger.info(f"💾 Memory after preprocessing: {process.memory_info().rss / 1024**3:.1f} GB")
    except ImportError:
        pass
    
    # Ensure contiguity before building groups (stable sort preserves intra-timestamp order)
    order_tr = np.argsort(ts_tr, kind="mergesort")
    X_tr, y_tr, ts_tr, symbols_tr = X_tr[order_tr], y_tr[order_tr], ts_tr[order_tr], symbols_tr[order_tr]
    
    order_va = np.argsort(ts_va, kind="mergesort")
    X_va, y_va, ts_va, symbols_va = X_va[order_va], y_va[order_va], ts_va[order_va], symbols_va[order_va]
    
    # Build group arrays for train/val (cast to plain Python int for compatibility)
    groups_tr = [int(g) for g in groups_from_ts(ts_tr)]
    groups_va = [int(g) for g in groups_from_ts(ts_va)]
    
    # Timestamps are guaranteed contiguous due to mergesort above
    
    # Assert group integrity for both splits
    assert sum(groups_tr) == len(X_tr), f"Train group/row mismatch: {sum(groups_tr)} != {len(X_tr)}"
    assert min(groups_tr) >= args.min_cs, f"Train min CS {min(groups_tr)} < min_cs={args.min_cs}"
    assert sum(groups_va) == len(X_va), f"Val group/row mismatch: {sum(groups_va)} != {len(X_va)}"
    assert min(groups_va) >= args.min_cs, f"Val min CS {min(groups_va)} < min_cs={args.min_cs}"
    
    # Log CS coverage for both splits
    log_cs_coverage(ts_tr, "train")
    log_cs_coverage(ts_va, "val")
    
    logger.info(f"📊 Time-aware split: train={len(X_tr)} rows ({len(train_ts)} timestamps), val={len(X_va)} rows ({len(val_ts)} timestamps)")
    
    results = {}
    successful_models = 0
    
    for i, family in enumerate(families, 1):
        try:
            logger.info(f"🚀 Training {family} on {target} (cross-sectional) with {len(feat_cols)} features... [{i}/{len(families)}]")
            logger.info(f"📅 Validation start timestamp: {_safe_val_start_ts(val_ts)}")

            # Train model with cross-sectional data and groups
            config = {"groups": groups_tr, "X_val": X_va, "y_val": y_va, "groups_val": groups_va, "rank_labels": args.rank_labels, "min_cs": args.min_cs}
            if family == 'QuantileLightGBM' and args is not None:
                config["quantile_alpha"] = args.quantile_alpha
            model = train_model(
                family,
                X_tr, y_tr,
                config,
                symbols_tr,
                cpu_only=args.cpu_only if args else False,
                rank_objective=args.rank_objective if args else "on",
                num_threads=args.threads if args else 12,
                feat_cols=feat_cols,
                seq_lookback=args.seq_lookback if args else 64,
                mtf_data=mtf_data,
                target=target
            )

            if model is not None:
                # Validate on the held-out period
                temporal_fams = {"CNN1D","LSTM","Transformer","MultiTask"}
                
                if family in temporal_fams:
                    # rebuild the same sequence split the trainer used
                    seq_targets = [target] if family != "MultiTask" else (
                        [target] + [t for t in [f"fwd_ret_{h}m" for h in (5,10,15,30,60)] 
                                    if all(t in df.columns for df in mtf_data.values()) and t != target]
                    )
                    seq_data = prepare_sequence_cs(
                        mtf_data, feat_cols, seq_targets, 
                        lookback=args.seq_lookback if args else 64, 
                        min_cs=args.min_cs,
                        val_start_ts=_safe_val_start_ts(val_ts)  # align with tabular split
                    )
                    X_va_seq = seq_data["X_va"]
                    y_va_seq = seq_data["y_va"]
                    ts_va_seq = seq_data["ts_va"]

                    # pick the right target head
                    if family == "MultiTask":
                        idx = seq_data["task_names"].index(target)
                        y_true = y_va_seq[:, idx]
                        # MultiTask returns list of heads; select current one
                        y_pred_all = model.predict(
                            model.scaler.transform(model.imputer.transform(
                                X_va_seq.reshape(-1, X_va_seq.shape[2])
                            )).reshape(X_va_seq.shape[0], X_va_seq.shape[1], X_va_seq.shape[2]),
                            verbose=0
                        )
                        y_pred = np.asarray(y_pred_all[idx]).ravel()
                    else:
                        y_true = y_va_seq[:, 0]
                        y_pred = _predict_temporal_model(model, X_va_seq)

                    metrics = cs_metrics_by_time(y_true, y_pred, ts_va_seq)
                else:
                    # existing tabular path
                    meta = {'family': family, 'features': feat_cols}
                    y_pred_va = safe_predict(model, X_va, meta)
                    metrics = cs_metrics_by_time(y_va, y_pred_va, ts_va)
                logger.info(
                    f"📊 {family} CS metrics (val): "
                    f"IC={metrics['mean_IC']:.4f}, "
                    f"RankIC={metrics['mean_RankIC']:.4f}, "
                    f"Hit Rate={metrics['hit_rate']:.4f}"
                )

                # Save model + metadata
                # Extract useful training metadata
                extra = {}
                try:
                    import lightgbm as _lgb
                    if isinstance(model, _lgb.Booster):
                        extra.update(best_iteration=model.best_iteration or None)
                except Exception:
                    pass
                try:
                    import xgboost as _xgb
                    if isinstance(model, _xgb.Booster):
                        extra.update(best_ntree_limit=getattr(model, 'best_ntree_limit', None))
                except Exception:
                    pass

                # Save feature importance if available
                feature_importance = None
                try:
                    if hasattr(model, 'feature_importance'):
                        # LightGBM
                        feature_importance = dict(zip(feat_cols, model.feature_importance(importance_type="gain")))
                    elif hasattr(model, 'get_score'):
                        # XGBoost - map f0, f1, ... back to feature names
                        fmap = {f"f{i}": name for i, name in enumerate(feat_cols)}
                        raw_importance = model.get_score(importance_type="gain")
                        feature_importance = {fmap.get(k, k): v for k, v in raw_importance.items()}
                except Exception:
                    pass
                
                meta_out = {
                    "family": family,
                    "target": target,
                    "min_cs": min_cs,
                    "features": tuple(feat_cols),
                    "feature_names": list(feat_cols),  # String list for other languages/tools
                    "n_features": len(feat_cols),
                    "package_versions": _get_package_versions(),
                    "cli_args": {
                        "min_cs": min_cs,
                        "max_samples_per_symbol": max_samples_per_symbol,
                        "cs_normalize": cs_normalize,
                        "cs_block": args.cs_block if args else 32,
                        "cs_winsor_p": args.cs_winsor_p if args else 0.01,
                        "cs_ddof": args.cs_ddof if args else 1,
                        "batch_id": batch_id,
                        "families": families
                    },
                    "n_rows_train": int(len(X_tr)),
                    "n_rows_val": int(len(X_va)),
                    "train_timestamps": len(train_ts),
                    "val_timestamps": len(val_ts),
                    "time_col": time_col,
                    "val_start_ts": _safe_val_start_ts(val_ts),
                    "metrics": metrics,
                    "best": extra,
                    "params_used": getattr(model, 'attributes', lambda: {})() if hasattr(model, 'attributes') else None,
                    "learner_params": getattr(model, 'params', None),
                    "cs_norm": {"mode": cs_normalize, "p": args.cs_winsor_p, "ddof": args.cs_ddof, "method": CS_WINSOR},
                    "rank_method": getattr(model, 'rank_method', 'unknown'),
                    "feature_importance": feature_importance,
                }
                saved_paths = save_model(model, family, target, output_dir, batch_id, meta_out)
                results[family] = {
                    "status": "success", 
                    "paths": saved_paths, 
                    "metrics": metrics,
                    "val_start_ts": _safe_val_start_ts(val_ts)
                }
                successful_models += 1
                logger.info(f"✅ {family} on {target} completed (CS training)")
                
                # Cleanup model from memory after saving
                del model
                import gc
                gc.collect()
                
                # Clear TF session to free GPU memory
                try:
                    import tensorflow as _tf
                    _tf.keras.backend.clear_session()
                except Exception:
                    pass
            else:
                logger.error(f"❌ Failed to train {family} on {target}: Model returned None")
                results[family] = {"status": "failed", "error": "Model returned None"}

        except Exception as e:
            logger.error(f"❌ Error training {family} on {target}: {e}")
            results[family] = {"status": "failed", "error": str(e)}
    
    logger.info(f"📊 {target} training complete: {successful_models}/{len(families)} models successful")
    return results

def normalize_symbols(args):
    """Normalize symbol list from CLI args or file."""
    import re
    
    if args.symbols_file:
        with open(args.symbols_file) as f:
            syms = [re.sub(r":.*$", "", s.strip()) for s in f if s.strip()]
    elif args.symbols:
        s = re.split(r"[,\s]+", " ".join(args.symbols).strip())
        syms = [re.sub(r":.*$", "", x) for x in s if x]
    else:
        syms = []  # fall back to auto-discovery if you support it
    return sorted(set(syms))

def setup_tf(cpu_only: bool = False):
    """Initialize TensorFlow with proper GPU/CPU configuration."""
    global tf, TF_DEVICE
    
    # Prevent re-initialization
    if tf is not None:
        return True
    
    try:
        # Clear any existing TensorFlow sessions
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
        except:
            pass
        # Set CPU-only mode if requested
        if cpu_only:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        
        import tensorflow as tf
        
        # Make GPUs visible & growth-friendly
        gpus = tf.config.list_physical_devices('GPU')
        
        # Optional: enable mixed precision on GPU for speed (controlled by env var)
        try:
            from tensorflow.keras import mixed_precision
            enable_mp = os.getenv("ENABLE_MIXED_PRECISION", "0") == "1"
            if gpus and not cpu_only and enable_mp:
                mixed_precision.set_global_policy("mixed_float16")
                logger.info("✅ Mixed precision enabled (float16 compute).")
            else:
                logger.info("ℹ️ Mixed precision skipped (CPU-only mode or ENABLE_MIXED_PRECISION=0)")
        except Exception as e:
            logger.warning(f"Mixed precision not enabled: {e}")
            
        if gpus and not cpu_only:
            try:
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
                logger.info("✅ TF sees GPU(s): %s", tf.config.list_logical_devices('GPU'))
            except RuntimeError as e:
                if "Physical devices cannot be modified after being initialized" in str(e):
                    logger.warning("⚠️  TensorFlow already initialized, using existing GPU configuration")
                else:
                    logger.warning(f"⚠️  GPU setup failed: {e}")
        else:
            logger.info("💻 Using CPU for TensorFlow models")
        # Mild perf niceties for CPU kernels
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("KMP_BLOCKTIME", "0")
        
        # Enable soft device placement for better GPU/CPU fallback
        tf.config.set_soft_device_placement(True)
        logger.info("✅ Soft device placement enabled")
        
        # Enable XLA separately (don't let XLA failures disable TF)
        # Default to XLA=0 for better determinism, enable with ENABLE_XLA=1
        if os.getenv("ENABLE_XLA", "0") == "1" and not cpu_only:
            try:
                tf.config.optimizer.set_jit(True)  # enable XLA
                logger.info("✅ XLA enabled.")
            except Exception as e:
                logger.warning(f"XLA not enabled: {e}")
        else:
            logger.info("XLA disabled via ENABLE_XLA=0 or CPU-only mode")

        # Use dynamic device detection
        TF_DEVICE = pick_tf_device()
        
        # Set global determinism after TF is imported
        set_global_determinism(42)
        
        logger.info(f"🚀 TensorFlow initialized on {TF_DEVICE}")
        return True

    except Exception as e:
        logger.warning(f"❌ TensorFlow import/setup failed: {e}")
        tf = None
        TF_DEVICE = '/CPU:0'
        return False



def train_with_strategy(strategy: str, X: np.ndarray, y_dict: Dict[str, np.ndarray], 
                       feature_names: List[str], config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Train models using specified strategy"""
    
    if not STRATEGY_SUPPORT:
        logger.warning("Strategy support not available, falling back to single-task")
        return train_models_for_interval(X, y_dict, feature_names)
    
    logger.info(f"Training with strategy: {strategy}")
    
    # Create strategy manager
    if strategy == 'single_task':
        strategy_manager = SingleTaskStrategy(config or {})
    elif strategy == 'multi_task':
        strategy_manager = MultiTaskStrategy(config or {})
    elif strategy == 'cascade':
        strategy_manager = CascadeStrategy(config or {})
    else:
        logger.warning(f"Unknown strategy: {strategy}, using single_task")
        strategy_manager = SingleTaskStrategy(config or {})
    
    # Train models
    results = strategy_manager.train(X, y_dict, feature_names)
    
    # Test predictions
    test_predictions = strategy_manager.predict(X[:100])
    
    return {
        'strategy_manager': strategy_manager,
        'results': results,
        'test_predictions': test_predictions,
        'success': True
    }

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Cross-Sectional MTF Model Training")
    parser.add_argument("--data-dir", default="5m_with_barrier_targets_full",
                        help="Directory containing MTF parquet files")
    parser.add_argument("--output-dir", default="ml/zoo/mtf_models",
                        help="Output directory for trained models")
    parser.add_argument("--intervals", nargs="+", default=["5m"],
                        help="Cadence keys to train on (default: 5m).")
    parser.add_argument("--exec-cadence", type=str, default="5m",
                        help="Live execution cadence (e.g., 5m). Used to derive horizons.")
    parser.add_argument("--horizons-min", nargs="+", type=int, default=[5, 10, 15],
                        help="Forward-return horizons in minutes to train for the exec cadence.")
    parser.add_argument("--families", nargs="+", default=["LightGBM", "XGBoost"],
                        help="Model families to train. Neural networks (CNN1D, LSTM, Transformer, MultiTask) are temporal by default. Use TabCNN, TabLSTM, TabTransformer for tabular versions.")
    parser.add_argument("--symbols", nargs="+",
                        help="Specific symbols to train on (default: all available)")
    parser.add_argument("--symbols-file", type=str,
                        help="Path to file with one symbol per line")
    parser.add_argument("--max-symbols", type=int,
                        help="Maximum number of symbols to process")
    parser.add_argument("--max-samples-per-symbol", type=int, default=10000,
                        help="(Ignored in cross-sectional mode) Maximum samples per symbol")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Number of symbols to process per batch")
    parser.add_argument("--batch-id", type=int, default=0,
                        help="Batch ID for this training run")
    parser.add_argument("--session-id", type=str, default=None,
                        help="Session ID for this training run")
    parser.add_argument("--feature-list", type=str, help="Path to JSON file of global feature list")
    parser.add_argument("--save-features", action="store_true", help="Save global feature list to features_all.json")
    parser.add_argument("--min-cs", type=int, default=10, help="Minimum cross-sectional size per timestamp (default: 10)")
    parser.add_argument("--cs-normalize", choices=["none", "per_ts_split"], default="per_ts_split", 
                        help="Cross-sectional normalization mode (default: per_ts_split)")
    parser.add_argument("--cs-block", type=int, default=32,
                        help="Block size for CS transforms (default: 32)")
    parser.add_argument("--cs-winsor-p", type=float, default=0.01,
                        help="Winsorization percentile (default: 0.01)")
    parser.add_argument("--cs-ddof", type=int, default=1,
                        help="Degrees of freedom for standard deviation (default: 1)")
    parser.add_argument("--include-experimental", action="store_true", 
                        help="Include experimental/placeholder model families")
    parser.add_argument("--quantile-alpha", type=float, default=0.5,
                        help="Alpha parameter for QuantileLightGBM (default: 0.5)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="Force CPU for all learners (LightGBM/XGBoost)")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() - 1),
                        help=f"Number of threads for training (default: {max(1, os.cpu_count() - 1)})")
    parser.add_argument("--max-rows-train", type=int, default=3000000,
                        help="Maximum rows for training (default: 3000000)")
    parser.add_argument("--max-rows-val", type=int, default=600000,
                        help="Maximum rows for validation (default: 600000)")
    parser.add_argument("--validate-targets", action="store_true",
                        help="Run preflight validation checks on targets before training")
    parser.add_argument("--max-rows-per-symbol", type=int,
                        help="Maximum rows per symbol to prevent OOM (default: no limit)")
    parser.add_argument("--rank-objective", choices=["on", "off"], default="on",
                        help="Enable ranking objectives for LGB/XGB (default: on)")
    parser.add_argument("--rank-labels", choices=["dense", "raw"], default="dense",
                        help="Ranking label method: 'dense' for dense ranks (default), 'raw' for continuous values")
    parser.add_argument("--strict-exit", action="store_true",
                        help="Exit with error code if any model fails (default: only exit on complete failure)")
    parser.add_argument("--seq-lookback", type=int, default=64,
                        help="Lookback window for temporal sequence models (default: 64)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without executing")
    parser.add_argument("--targets", nargs="+",

    # Strategy arguments
    parser.add_argument("--strategy", choices=['single_task', 'multi_task', 'cascade', 'auto'], 
                       default='auto', help="Training strategy (auto = single_task)")
    parser.add_argument("--strategy-config", type=str, help="Path to strategy configuration file")

                        help="Specific targets to train on (default: auto-discover all targets)")
    
    args = parser.parse_args()
    
    # Initialize TensorFlow with proper CPU/GPU configuration
    setup_tf(cpu_only=args.cpu_only)
    
    # Threading environment variables were set at import time
    # LGB/XGB will use the num_threads parameter passed to them
    
    # MIN_CS is now passed as parameter to functions (no global needed)
    
    # Generate session ID if not provided
    if args.session_id is None:
        from datetime import datetime
        args.session_id = f"mtf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Get available symbols using normalize_symbols
    symbols = normalize_symbols(args)
    if not symbols:
        # Try new directory structure first: interval={exec_cadence}/symbol={symbol}/{symbol}.parquet
        mtf_files = glob.glob(str(Path(args.data_dir) / f"interval={args.exec_cadence}" / "symbol=*" / "*.parquet"))
        if mtf_files:
            symbols = [Path(f).parent.name.replace("symbol=", "") for f in mtf_files]
        else:
            # Fallback to old format: *_mtf.parquet
            mtf_files = glob.glob(str(Path(args.data_dir) / "*_mtf.parquet"))
            symbols = [Path(f).stem.replace("_mtf", "") for f in mtf_files]
    
    # Batch processing already applied above
    
    # Update output directory with session ID
    output_dir = Path(args.output_dir) / args.session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter experimental families if not requested
    if not args.include_experimental:
        original_families = args.families.copy()
        args.families = [f for f in args.families if not FAMILY_CAPS.get(f, {}).get("experimental", False)]
        if len(args.families) < len(original_families):
            filtered = set(original_families) - set(args.families)
            logger.info(f"🔧 Filtered out experimental families: {filtered}")
    
    # Assert CLI contract: only single interval supported
    if len(args.intervals) > 1:
        raise ValueError(f"Multiple intervals not supported yet. Got: {args.intervals}. Use single interval matching exec_cadence: {args.exec_cadence}")
    
    # Assert CLI contract: interval must match exec_cadence
    if args.intervals[0] != args.exec_cadence:
        raise ValueError(f"Interval {args.intervals[0]} must match exec_cadence {args.exec_cadence}. Use --exec-cadence {args.intervals[0]} to fix.")
    
    logger.info(f"🎯 CROSS-SECTIONAL MTF TRAINING")
    logger.info(f"Session ID: {args.session_id}")
    logger.info(f"Intervals: {args.intervals}")
    logger.info(f"Families: {args.families}")
    logger.info(f"Symbols: {len(symbols)} symbols")
    logger.info(f"Max samples per symbol: {args.max_samples_per_symbol}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Horizons: {args.horizons_min}")
    
    if args.dry_run:
        print(f"✅ dry-run: intervals={args.intervals} exec-cadence={args.exec_cadence} "
              f"horizons={args.horizons_min} families={args.families}")
        syms = symbols
        if args.max_symbols: 
            syms = syms[:args.max_symbols]
        mtf = load_mtf_data(args.data_dir, syms, args.exec_cadence, args.max_rows_per_symbol)
        feats = get_common_feature_columns(mtf)
        print(f"symbols={len(mtf)} features={len(feats)} (showing first 10): {feats[:10]}")
        tgts = [f"fwd_ret_{h}m" for h in args.horizons_min]
        missing = {t:[s for s,df in mtf.items() if t not in df.columns] for t in tgts}
        for t, miss in missing.items():
            print(f"{t}: present in {len(mtf)-len(miss)}/{len(mtf)} symbols")
        sys.exit(0)
    
    # Apply batch processing FIRST
    if args.batch_size > 0:
        start_idx = args.batch_id * args.batch_size
        end_idx = start_idx + args.batch_size
        symbols = symbols[start_idx:end_idx]
        logger.info(f"Batch {args.batch_id}: Processing symbols {start_idx}-{end_idx-1} ({len(symbols)} symbols)")
    
    if args.max_symbols:
        symbols = symbols[:args.max_symbols]
        logger.info(f"Limited to {len(symbols)} symbols")
    
    if not symbols:
        logger.error("No symbols selected after batching/max-symbols filtering")
        return
    
    # Filter families based on available libraries
    original_families = args.families.copy()
    if not _LGB_OK:
        args.families = [f for f in args.families if f not in ["LightGBM", "QuantileLightGBM"]]
        logger.info("🔧 LightGBM missing → filtering LightGBM families")
    if not _XGB_OK:
        args.families = [f for f in args.families if f != "XGBoost"]
        logger.info("🔧 XGBoost missing → filtering XGBoost")
    
    if not args.families:
        logger.error("No model families available after filtering")
        return
    
    logger.info(f"📊 Available families: {args.families}")
    
    # Load MTF data AFTER batching
    logger.info("📊 Loading MTF data...")
    # Note: Currently only supports single interval per run
    # TODO: Support multiple intervals by moving load_mtf_data inside interval loop
    mtf_data = load_mtf_data(args.data_dir, symbols, interval=args.exec_cadence, max_rows_per_symbol=args.max_rows_per_symbol)
    
    if not mtf_data:
        logger.error("No MTF data loaded")
        return
    
    # Validate data quality and target availability
    logger.info("🔍 Validating data quality...")
    total_symbols = len(mtf_data)
    logger.info(f"📊 Loaded data for {total_symbols} symbols")
    
    # Check symbol data quality
    for symbol, df in list(mtf_data.items()):
        if df.empty:
            logger.warning(f"⚠️  Symbol {symbol} has empty data, removing")
            del mtf_data[symbol]
        elif len(df) < args.min_cs:
            logger.warning(f"⚠️  Symbol {symbol} has only {len(df)} rows (need {args.min_cs}), removing")
            del mtf_data[symbol]
    
    if not mtf_data:
        logger.error("❌ No valid symbols after quality filtering")
        return
    
    logger.info(f"✅ {len(mtf_data)} symbols passed quality checks")
    
    # Early family validation (before expensive data loading)
    temporal_families = {'CNN1D', 'LSTM', 'Transformer', 'MultiTask'}
    tabular_families = {'TabCNN', 'TabLSTM', 'TabTransformer'}
    requested_temporal = set(args.families) & temporal_families
    requested_tabular = set(args.families) & tabular_families
    
    if requested_temporal:
        logger.info(f"📊 Temporal models requested: {requested_temporal}")
        logger.info("    These will train on time-series sequences with causal structure")
    
    if requested_tabular:
        logger.info(f"📊 Tabular models requested: {requested_tabular}")
        logger.info("    These will train on flat feature interactions")
    
    if requested_temporal and requested_tabular:
        logger.warning(
            "⚠️  Both temporal and tabular variants requested. "
            "This is valid but may be redundant for similar architectures."
        )
    
    # Get common features (use global list if provided)
    if args.feature_list and Path(args.feature_list).exists():
        logger.info(f"Loading global feature list from {args.feature_list}")
        common_features = load_global_feature_list(args.feature_list)
    else:
        common_features = get_common_feature_columns(mtf_data)
        if args.save_features:
            save_global_feature_list(common_features)
    
    if not common_features:
        logger.error("No common features found")
        return
    
    # Train models for each interval and target combination
    all_results = []
    total_models = 0
    successful_models = 0
    
    for interval in args.intervals:
        # NOTE: Currently only supports single interval per run
        # Data is loaded with exec_cadence, so interval must match exec_cadence
        if interval != args.exec_cadence:
            logger.warning(f"Skipping {interval} - data loaded with exec_cadence={args.exec_cadence}")
            continue
            
        # Use specific targets if provided, otherwise auto-discover
        if args.targets:
            tgt_list = args.targets
            # For specified targets, we need to discover all_targets from the data
            sample_symbol = list(mtf_data.keys())[0]
            all_targets = set(col for col in mtf_data[sample_symbol].columns 
                            if (col.startswith('fwd_ret_') or 
                                col.startswith('will_peak') or 
                                col.startswith('will_valley') or
                                col.startswith('y_will_') or
                                col.startswith('y_first_touch') or
                                col.startswith('p_up') or
                                col.startswith('p_down') or
                                col.startswith('mfe') or
                                col.startswith('mdd')))
            logger.info(f"🎯 Using specified targets: {tgt_list}")
        else:
            tgt_list, all_targets = targets_for_interval(interval, args.exec_cadence, args.horizons_min, mtf_data)
        
        logger.info(f"🎯 Available targets for {interval}: {tgt_list}")
        
        # Debug: Show what targets are actually available in the data
        sample_symbol = list(mtf_data.keys())[0]
        available_targets = [col for col in mtf_data[sample_symbol].columns if col.startswith('fwd_ret_')]
        logger.info(f"🎯 Targets available in data: {available_targets}")
        
        # FORWARD RETURN VALIDATION - Bulletproof training verification
        if args.validate_targets:
            logger.info("🔍 Running forward return validation checks...")
            try:
                from fwdret_validation import (
                    discover_fwdret_targets, preflight_fwdret, smoke_all_fwdret,
                    begin_interval_run, end_interval_run, save_fold_artifact, write_oof
                )
                
                # Combine all symbols into single dataframe for validation
                logger.info("📊 Combining data for validation...")
                combined_data = []
                for sym, df in list(mtf_data.items())[:min(10, len(mtf_data))]:  # Sample for validation
                    df_copy = df.copy()
                    df_copy['symbol'] = sym
                    combined_data.append(df_copy)
                
                if combined_data:
                    validation_df = pd.concat(combined_data, ignore_index=True)
                    validation_df = validation_df.sort_values(['time', 'symbol']).reset_index(drop=True)
                    
                    # Get feature columns
                    fcols = [c for c in validation_df.columns if c.startswith('f_') or 
                            (not c.startswith(('time', 'symbol', 'close', 'high', 'low', 'open', 'volume', 'fwd_ret_', 'y_', 'p_', 'mfe_', 'mdd_')))]
                    
                    # Discover forward return targets
                    fwdret_targets = discover_fwdret_targets(validation_df, bar_seconds=300)  # 5min bars
                    logger.info(f"🎯 Discovered {len(fwdret_targets)} forward return targets")
                    
                    # Run preflight checks on forward return targets
                    preflight_df = preflight_fwdret(validation_df, fcols, bar_seconds=300, min_rows=1000)
                    fwdret_passed = preflight_df['preflight_pass'].sum()
                    logger.info(f"📋 Forward return preflight: {fwdret_passed}/{len(preflight_df)} targets passed")
                    
                    # Run smoke tests on forward return targets
                    if fwdret_passed > 0:
                        smoke_df = smoke_all_fwdret(validation_df, fcols, bar_seconds=300, max_intervals=8)
                        smoke_passed = smoke_df['ok'].sum() if 'ok' in smoke_df.columns else 0
                        logger.info(f"🔥 Forward return smoke tests: {smoke_passed}/{len(smoke_df)} targets passed")
                    
                    # Filter targets to only include validated forward return targets
                    validated_fwdret = preflight_df[preflight_df['preflight_pass']]['target'].tolist()
                    tgt_list = [t for t in tgt_list if t in validated_fwdret or not t.startswith('fwd_ret_')]
                    logger.info(f"🎯 Filtered to {len(tgt_list)} validated targets (including {len(validated_fwdret)} forward returns)")
                
            except Exception as e:
                logger.warning(f"⚠️ Forward return validation failed: {e}. Proceeding without validation.")
                import traceback
                logger.warning(f"Validation traceback: {traceback.format_exc()}")
        
        for i, target in enumerate(tgt_list):
            logger.info(f"🎯 Processing target {i+1}/{len(tgt_list)}: {target}")
            if target in SKIP_TARGETS:
                logger.info(f"Skipping {interval} -> {target} (classification target)")
                continue
            
            # RUNTIME INSTRUMENTATION - Track forward return training
            if args.validate_targets and target.startswith('fwd_ret_'):
                try:
                    from fwdret_validation import begin_interval_run, end_interval_run
                    # Extract horizon from target name (e.g., fwd_ret_5m -> 5m)
                    horizon_str = target.replace('fwd_ret_', '') + 'm'
                    begin_interval_run(horizon_str)
                    logger.info(f"🚀 Started training instrumentation for {horizon_str}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to start instrumentation for {target}: {e}")
            
            # Validate target presence and filter data to only include symbols with this target
            available_symbols = [sym for sym, df in mtf_data.items() if target in df.columns]
            missing_symbols = [sym for sym, df in mtf_data.items() if target not in df.columns]
            
            logger.info(f"🎯 {target}: available in {len(available_symbols)}/{len(mtf_data)} symbols")
            
            if len(available_symbols) < args.min_cs:
                logger.error(f"❌ Skipping target '{target}' - only {len(available_symbols)} symbols available (need {args.min_cs})")
                continue
                
            if missing_symbols:
                logger.warning(
                    f"Target '{target}' missing in {len(missing_symbols)} symbols "
                    f"(first 5: {missing_symbols[:5]}). Training on {len(available_symbols)} symbols."
                )
            else:
                logger.info(f"✅ Target '{target}' found in all {len(mtf_data)} symbols")
            
            # Filter mtf_data to only include symbols with this target
            filtered_mtf_data = {sym: df for sym, df in mtf_data.items() if sym in available_symbols}
            
            total_models += len(args.families)
            
            result = train_models_for_interval(
                mtf_data=filtered_mtf_data,  # Use filtered data
                target=target,
                families=args.families,
                common_features=common_features,
                output_dir=str(output_dir),
                min_cs=args.min_cs,
                args=args,
                max_samples_per_symbol=args.max_samples_per_symbol,
                batch_id=args.batch_id,
                cs_normalize=args.cs_normalize,
                all_targets=all_targets
            )
            all_results.append({f"{interval}:{target}": result})
            
            # END RUNTIME INSTRUMENTATION - Mark forward return training complete
            if args.validate_targets and target.startswith('fwd_ret_'):
                try:
                    from fwdret_validation import end_interval_run
                    horizon_str = target.replace('fwd_ret_', '') + 'm'
                    summary = {
                        "target": target,
                        "horizon": horizon_str,
                        "models_trained": len([r for r in result.values() if isinstance(r, dict) and r.get("status") == "success"]) if isinstance(result, dict) else 0,
                        "timestamp": time.time()
                    }
                    end_interval_run(horizon_str, summary)
                    logger.info(f"✅ Completed training instrumentation for {horizon_str}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to end instrumentation for {target}: {e}")
            
            # RUNTIME INTEGRITY CHECKS
            if args.validate_targets and isinstance(result, dict):
                try:
                    from validation_checks import validate_oof_integrity
                    
                    # Check each successful model for integrity
                    for model_name, model_result in result.items():
                        if isinstance(model_result, dict) and model_result.get("status") == "success":
                            # Get OOF predictions if available
                            oof_pred = model_result.get("oof_predictions")
                            if oof_pred is not None:
                                # Determine target type
                                target_type = "multiclass" if target.startswith(('y_will_', 'y_first_touch')) else "regression"
                                
                                # Validate OOF integrity
                                integrity_checks = validate_oof_integrity(oof_pred, target_type)
                                
                                # Log integrity results
                                if integrity_checks.get("prob_sum_valid", True) and integrity_checks.get("finite_values", True):
                                    logger.info(f"✅ {model_name} OOF integrity: PASS")
                                else:
                                    logger.warning(f"⚠️ {model_name} OOF integrity: FAIL - {integrity_checks}")
                                
                                # Save integrity results
                                model_result["oof_integrity"] = integrity_checks
                
                except Exception as e:
                    logger.warning(f"⚠️ Runtime integrity check failed: {e}")
            
            # Count successful models for this target
            if isinstance(result, dict):
                target_success = sum(1 for r in result.values() if isinstance(r, dict) and r.get("status") == "success")
                target_failed = sum(1 for r in result.values() if isinstance(r, dict) and r.get("status") == "failed")
                target_skipped = sum(1 for r in result.values() if isinstance(r, dict) and r.get("status") == "skipped")
                successful_models += target_success
                logger.info(f"✅ Completed target {i+1}/{len(tgt_list)}: {target} - {target_success} successful, {target_failed} failed, {target_skipped} skipped")
            else:
                logger.info(f"❌ Failed target {i+1}/{len(tgt_list)}: {target} - result was {type(result)}")
    
    # Final memory cleanup and summary
    logger.info(f"\n🎯 Target Processing Summary:")
    logger.info(f"   Total targets processed: {len(tgt_list)}")
    logger.info(f"   Successful targets: {len([r for r in all_results if isinstance(r, dict)])}")
    logger.info(f"   Total models trained: {total_models}")
    logger.info(f"   Successful models: {successful_models}")
    
    try:
        import psutil
        import gc
        process = psutil.Process()
        logger.info(f"\n🧹 Final memory cleanup...")
        for _ in range(3):
            gc.collect()
        logger.info(f"Final memory usage: {process.memory_info().rss / 1024**3:.1f} GB")
    except ImportError:
        pass
    
    # Generate summary
    logger.info(f"\n📊 TRAINING SUMMARY")
    logger.info(f"Total models: {total_models}")
    logger.info(f"✅ Successful models: {successful_models}")
    logger.info(f"❌ Failed models: {total_models - successful_models}")
    
    # POST-RUN VERIFICATION - Bulletproof forward return training verification
    if args.validate_targets:
        try:
            from fwdret_validation import verify_fwdret_training
            logger.info("🔍 Running post-run verification...")
            
            # Get model families from args
            model_families = args.families if hasattr(args, 'families') else ["LightGBM", "XGBoost", "MLP"]
            
            # Run verification
            verification_df = verify_fwdret_training(
                pd.DataFrame(),  # We don't need the full dataframe for verification
                bar_seconds=300,  # 5min bars
                model_families=model_families,
                min_folds=3
            )
            
            # Log results
            total_intervals = len(verification_df)
            passed_intervals = verification_df['PASS'].sum()
            logger.info(f"✅ Forward return verification: {passed_intervals}/{total_intervals} intervals PASSED")
            
            if not verification_df['PASS'].all():
                failed_intervals = verification_df[~verification_df['PASS']]
                logger.warning(f"⚠️ Failed intervals: {failed_intervals[['horizon', 'missing_models']].to_dict('records')}")
            
        except Exception as e:
            logger.warning(f"⚠️ Post-run verification failed: {e}")
    
    logger.info(f"\n📈 RESULTS BY INTERVAL:")
    # Aggregate results by interval (handles multiple targets per interval)
    by_interval = {}
    for item in all_results:
        (k, v), = item.items()  # "5m:fwd_ret_5m": {...}
        interval = k.split(":")[0]
        ok = sum(1 for r in v.values() if isinstance(r, dict) and r.get("status") == "success")
        total = sum(1 for r in v.values() if isinstance(r, dict))
        a, b = by_interval.get(interval, (0, 0))
        by_interval[interval] = (a + ok, b + total)
    
    for interval, (ok, tot) in by_interval.items():
        logger.info(f"  {interval}: {ok}/{tot} models")
    
    # Write run-level summary manifest
    try:
        import json
        from datetime import datetime
        
        # Collect top-line metrics from all_results
        top_metrics = {}
        for item in all_results:                       # item: {"5m:fwd_ret_5m": {family -> {...}}}
            (k, fam_map), = item.items()
            for fam, info in fam_map.items():
                if isinstance(info, dict) and info.get("status") == "success":
                    m = info.get("metrics", {})
                    top_metrics[f"{k}:{fam}"] = {
                        "mean_IC": m.get("mean_IC", 0.0),
                        "mean_RankIC": m.get("mean_RankIC", 0.0),
                        "IC_IR": m.get("IC_IR", 0.0),
                        "hit_rate": m.get("hit_rate", 0.0),
                        "val_start_ts": info.get("val_start_ts", "unknown"),
                    }
        
        summary = {
            "session_id": args.session_id,
            "timestamp": datetime.now().isoformat(),
            "intervals": list(by_interval.keys()),
            "results_by_interval": by_interval,
            "total_models": sum(tot for _, tot in by_interval.values()),
            "successful_models": sum(ok for ok, _ in by_interval.values()),
            "top_metrics": top_metrics,
            "output_dir": str(output_dir),
            "data_dir": args.data_dir,
            "families": args.families,
            "horizons_min": args.horizons_min,
            "threads": args.threads,
            "cpu_only": args.cpu_only,
            "package_versions": _get_package_versions()
        }
        
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        logger.info(f"📋 Run summary saved: {summary_path}")
        
    except Exception as e:
        logger.warning(f"Could not save run summary: {e}")
    
    logger.info(f"\n🎉 Cross-sectional training complete!")
    
    # Return appropriate exit code for orchestration scripts
    total_models = sum(tot for _, tot in by_interval.values())
    successful_models = sum(ok for ok, _ in by_interval.values())
    
    if total_models == 0:
        logger.error("❌ No models trained - no data available")
        sys.exit(1)
    elif successful_models == 0:
        logger.error("❌ All models failed")
        sys.exit(1)
    elif successful_models < total_models:
        if args.strict_exit:
            logger.warning(f"⚠️ {successful_models}/{total_models} models succeeded (strict-exit enabled)")
            sys.exit(1)
        else:
            logger.warning(f"⚠️ {successful_models}/{total_models} models succeeded (partial success allowed)")
            sys.exit(0)
    else:
        logger.info(f"✅ All {successful_models} models succeeded")
        sys.exit(0)

if __name__ == "__main__":
    main()


class GANPredictor:
    """GAN predictor with proper scaling."""
    def __init__(self, generator, imputer, scaler, regressor):
        self.generator = generator
        self.imputer = imputer
        self.scaler = scaler
        self.regressor = regressor
        self.handles_preprocessing = True
    
    def predict(self, X):
        """Predict using generator-augmented features."""
        X_scaled = self.scaler.transform(self.imputer.transform(X))
        
        # Generate deterministic synthetic features using hash-based noise
        # This ensures the same X always produces the same synthetic features
        import hashlib
        noise = np.zeros((len(X_scaled), 32))
        for i, row in enumerate(X_scaled):
            # Create deterministic noise based on input features
            row_hash = hashlib.md5(row.tobytes()).hexdigest()
            seed = int(row_hash[:8], 16) % (2**32)
            rng = np.random.RandomState(seed)
            noise[i] = rng.normal(0, 1, 32)
        
        synthetic_features = self.generator.predict(noise, verbose=0)
        
        # Combine original and synthetic features
        combined_features = np.concatenate([X_scaled, synthetic_features], axis=1)
        
        return self.regressor.predict(combined_features)



class ChangePointPredictor:
    """ChangePoint predictor with proper feature engineering."""
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

def load_mtf_data(data_dir: str, symbols: List[str], interval: str = "5m", max_rows_per_symbol: int = None) -> Dict[str, pd.DataFrame]:
    """Load MTF data for specified symbols and interval.
    
    Args:
        data_dir: Directory containing parquet files
        symbols: List of symbols to load
        interval: Data interval (e.g., "5m")
        max_rows_per_symbol: Optional limit to prevent OOM on large datasets
    """
    if not USE_POLARS:
        return _load_mtf_data_pandas(data_dir, symbols, interval, max_rows_per_symbol)
    
    mtf_data = {}
    
    for symbol in symbols:
        # Try new directory structure first: interval={interval}/symbol={symbol}/{symbol}.parquet
        new_path = Path(data_dir) / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.parquet"
        legacy_path = Path(data_dir) / f"{symbol}_mtf.parquet"
        
        file_path = new_path if new_path.exists() else legacy_path
        if not file_path.exists():
            logger.warning(f"File not found for {symbol} at {new_path} or {legacy_path}")
            continue
            
        try:
            # Lazy scan - won't materialize until collect()
            lf = pl.scan_parquet(str(file_path))
            # Detect/standardize time column
            tcol = _resolve_time_col_polars(lf.collect_schema().names())
            # Use tolerant cast instead of strptime (handles both string and datetime columns)
            lf = lf.with_columns(pl.col(tcol).cast(pl.Datetime, strict=False).alias(tcol))\
                   .drop_nulls([tcol])
            if max_rows_per_symbol:
                lf = lf.tail(max_rows_per_symbol)  # Keep most recent
            df = lf.collect(streaming=True)
            # Hand back pandas for compatibility with the rest of your code
            mtf_data[symbol] = df.to_pandas(use_pyarrow_extension_array=False)
            logger.info(f"Loaded {symbol}: {len(mtf_data[symbol]):,} rows, {len(mtf_data[symbol].columns)} cols")
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    return mtf_data

def _load_mtf_data_pandas(data_dir: str, symbols: List[str], interval: str = "5m", max_rows_per_symbol: int = None) -> Dict[str, pd.DataFrame]:
    """Original pandas-based MTF loading."""
    mtf_data = {}
    
    for symbol in symbols:
        # Try new directory structure first: interval={interval}/symbol={symbol}/{symbol}.parquet
        new_path = Path(data_dir) / f"interval={interval}" / f"symbol={symbol}" / f"{symbol}.parquet"
        legacy_path = Path(data_dir) / f"{symbol}_mtf.parquet"
        
        file_path = new_path if new_path.exists() else legacy_path
        if not file_path.exists():
            logger.warning(f"File not found for {symbol} at {new_path} or {legacy_path}")
            continue
            
        try:
            df = pd.read_parquet(file_path)
            # Normalize timestamp dtype immediately
            tcol = resolve_time_col(df)
            df[tcol] = pd.to_datetime(df[tcol], errors="coerce")
            if df[tcol].isna().any():
                df = df.dropna(subset=[tcol])
            
            # Apply row limit if specified (keep most recent data)
            if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                df = df.tail(max_rows_per_symbol)
                logger.info(f"📊 Limited {symbol} to {len(df):,} rows (most recent)")
            
            mtf_data[symbol] = df
            logger.info(f"Loaded {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
    
    return mtf_data

def _resolve_time_col_polars(cols):
    """Resolve time column name for Polars."""
    for c in ("ts","timestamp","time","datetime","ts_pred"):
        if c in cols:
            return c
    raise KeyError(f"No time column in {cols}")

def get_common_feature_columns(mtf_data: Dict[str, pd.DataFrame]) -> List[str]:
    """Get common feature columns across all symbols."""
    if not USE_POLARS:
        return _get_common_feature_columns_pandas(mtf_data)
    
    if not mtf_data:
        return []

    PROBLEMATIC = {'fractal_high','fractal_low','cmf_ema','chaikin_oscillator',
                   'ad_line_ema','williams_accumulation','ad_line','order_flow_imbalance'}
    # Detect time col from first item
    first = next(iter(mtf_data.values()))
    time_col = resolve_time_col(first)
    leak_cols = {'ts','timestamp','time','date','symbol','interval','source','ticker','id','index', time_col}

    # Intersect names across symbols using Polars schemas
    name_sets = []
    for df in mtf_data.values():
        name_sets.append(set(df.columns))
    common = set.intersection(*name_sets)

    # Exclude target columns (forward returns)
    target_cols = {col for col in first.columns if col.startswith('fwd_ret_') or col.startswith('will_peak') or col.startswith('will_valley')}
    
    # Keep numeric/bool, drop leaks/problematic/targets
    keep = [c for c in common
            if c not in leak_cols and c not in PROBLEMATIC and c not in target_cols
            and (pd.api.types.is_numeric_dtype(first[c]) or pd.api.types.is_bool_dtype(first[c]))]
    keep.sort()

    # Compute global NaN rates with pandas (NaN + None) - Polars null_count() only counts None
    counts = {c: 0 for c in keep}
    total = 0
    for df in mtf_data.values():
        s = df[keep]
        for c in keep:
            counts[c] += int(s[c].isna().sum())
        total += len(s)
    good = [c for c in keep if (counts[c] / max(1, total)) <= 0.30]
    logger.info(f"Found {len(good)} common features across {len(mtf_data)} symbols (filtered {len(keep)-len(good)} high-NaN)")
    return good

def _get_common_feature_columns_pandas(mtf_data: Dict[str, pd.DataFrame]) -> List[str]:
    """Original pandas-based common feature discovery."""
    # Define problematic indicators that have high NaN rates due to lookback/lookahead requirements
    PROBLEMATIC_INDICATORS = {
        'fractal_high', 'fractal_low',  # Require lookahead (shift(-1), shift(-2))
        'cmf_ema', 'chaikin_oscillator', 'ad_line_ema',  # Require long warmup periods
        'williams_accumulation', 'ad_line', 'order_flow_imbalance'  # Volume-based with warmup
    }
    
    # Get feature columns for each symbol
    # Detect time column to exclude it from features
    if not mtf_data:
        return []
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    
    # Comprehensive leak protection - exclude common time/symbol leakage columns
    leak_cols = {'ts', 'timestamp', 'time', 'date', 'symbol', 'interval', 'source', 'ticker', 'id', 'index', 'Unnamed: 0'}
    leak_cols.add(time_col)  # Add the detected time column
    
    all_feature_cols = []
    for symbol, df in mtf_data.items():
        feature_cols = [col for col in df.columns 
                       if not col.startswith('fwd_ret') and 
                       not col.startswith('will_peak') and 
                       not col.startswith('will_valley') and
                       col not in leak_cols and  # Comprehensive leak protection
                       col not in PROBLEMATIC_INDICATORS and  # Filter out problematic indicators
                       (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]))]
        all_feature_cols.append(set(feature_cols))
    
    # Find intersection of all feature sets
    if not all_feature_cols:
        return []
    
    common_features = all_feature_cols[0]
    for feature_set in all_feature_cols[1:]:
        common_features = common_features.intersection(feature_set)
    
    common_features = sorted(list(common_features))
    
    # Filter out features with >30% NaN across the union (streaming approach)
    if common_features:
        # Compute NaN rates streaming per symbol to avoid huge concat
        counts = pd.Series(0, index=common_features, dtype="int64")
        nrows = 0
        for df in mtf_data.values():
            s = df[common_features]
            counts += s.isnull().sum()
            nrows += len(s)
        nan_rates = counts / max(1, nrows)
        good_features = [col for col in common_features if nan_rates[col] <= 0.3]
        
        if len(good_features) < len(common_features):
            logger.info(f"Filtered out {len(common_features) - len(good_features)} features with >30% NaN")
            common_features = good_features
    
    logger.info(f"Found {len(common_features)} common features across {len(mtf_data)} symbols")
    
    return common_features

def load_global_feature_list(feature_list_path: str) -> List[str]:
    """Load global feature list from JSON file."""
    import json
    with open(feature_list_path, 'r') as f:
        return json.load(f)

def save_global_feature_list(features: List[str], output_path: str = "features_all.json"):
    """Save global feature list to JSON file."""
    import json
    with open(output_path, 'w') as f:
        json.dump(features, f, indent=2)
    logger.info(f"Saved global feature list to {output_path}")

def targets_for_interval(interval: str, exec_cadence: str, horizons_min: List[int], mtf_data: Dict[str, pd.DataFrame] = None) -> tuple[List[str], set]:
    """
    If interval matches exec cadence, discover ALL available targets in the data.
    Otherwise fall back to INTERVAL_TO_TARGET (single target).
    """
    if interval == exec_cadence:
        if mtf_data:
            # Discover all available targets in the data
            sample_symbol = list(mtf_data.keys())[0]
            all_targets = [col for col in mtf_data[sample_symbol].columns 
                          if (col.startswith('fwd_ret_') or 
                              col.startswith('will_peak') or 
                              col.startswith('will_valley') or
                              col.startswith('y_will_') or
                              col.startswith('y_first_touch') or
                              col.startswith('p_up') or
                              col.startswith('p_down') or
                              col.startswith('mfe') or
                              col.startswith('mdd'))]
            
            # Count how many symbols have each target
            target_counts = {}
            for target in all_targets:
                count = sum(1 for df in mtf_data.values() if target in df.columns)
                target_counts[target] = count
            
            # Sort by count (most common first) then alphabetically
            sorted_targets = sorted(target_counts.items(), key=lambda x: (-x[1], x[0]))
            common_targets = [target for target, count in sorted_targets]
            
            logger.info(f"🎯 Discovered {len(common_targets)} targets in data:")
            for target, count in sorted_targets[:10]:  # Show top 10
                logger.info(f"  {target}: {count}/{len(mtf_data)} symbols")
            if len(sorted_targets) > 10:
                logger.info(f"  ... and {len(sorted_targets) - 10} more targets")
            
            return common_targets, set(all_targets)
        else:
            # Fallback to horizons_min if no data provided
            fallback_targets = [f"fwd_ret_{h}m" for h in horizons_min]
            return fallback_targets, set(fallback_targets)
    if interval in INTERVAL_TO_TARGET:
        single_target = [INTERVAL_TO_TARGET[interval]]
        return single_target, set(single_target)
    raise KeyError(f"No target mapping for interval '{interval}'. "
                   f"Either run with --exec-cadence={interval} or populate INTERVAL_TO_TARGET.")

def cs_transform_live(df_snapshot: pd.DataFrame, feature_cols: List[str], p=0.01, ddof=1, method="quantile"):
    """
    Apply cross-sectional transforms to live data snapshot.
    This replicates the per-timestamp winsorize + z-score from training.
    Uses the same parameters (p, ddof, method) as saved in model metadata for parity.
    """
    available_cols = [c for c in feature_cols if c in df_snapshot.columns]
    if not available_cols:
        return df_snapshot
    
    s = df_snapshot[available_cols]
    
    if method.lower() == "ksigma":
        # k-sigma winsorization (matches Polars k-sigma path)
        import math
        try:
            from scipy.stats import norm
            k = float(norm.ppf(1 - p))
        except Exception:
            k = 2.33 if abs(p - 0.01) < 1e-6 else 2.0
        
        mu = s.mean()
        sd = s.std(ddof=ddof)
        lo = mu - k * sd
        hi = mu + k * sd
        s = s.clip(lo, hi, axis=1)
        df_snapshot[available_cols] = (s - mu) / (sd + 1e-8)
    else:
        # quantile winsorization (default, matches pandas path)
        lo = s.quantile(p)
        hi = s.quantile(1-p)
        s = s.clip(lo, hi, axis=1)
        df_snapshot[available_cols] = (s - s.mean()) / (s.std(ddof=ddof) + 1e-8)
    
    return df_snapshot

def prepare_sequence_cs(mtf_data, feature_cols, target_cols, lookback=64, min_cs=10,
                        val_start_ts=None, train_ratio=0.8):
    """
    Build temporal sequences for neural network training.
    Creates (batch, lookback, features) sequences with causal structure.
    """
    # 1) detect time col
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    SYM = SYMBOL_COL

    # 2) concat (keep only time, sym, feats, targets)
    dfs = []
    for s, df in mtf_data.items():
        have = [c for c in feature_cols if c in df.columns]
        t_have = [t for t in target_cols if t in df.columns]
        if not t_have: 
            continue
        d = df[[time_col, *have, *t_have]].copy()
        d[SYM] = s
        d = d.sort_values(time_col, kind="mergesort").dropna(subset=t_have)
        dfs.append(d)
    if not dfs:
        raise ValueError("no data for requested targets")

    big = pd.concat(dfs, ignore_index=True)

    # 3) keep timestamps with enough cross-section (so CS eval still makes sense)
    counts = big.groupby(time_col)[SYM].nunique()
    keep_ts = counts[counts >= min_cs].index
    big = big[big[time_col].isin(keep_ts)].copy()

    # 4) build rolling sequences per-symbol (causal: last step is "t", predicts targets at "t")
    Xseq, ymat, ts_last = [], [], []
    for s, g in big.groupby(SYM, sort=False):
        g = g.sort_values(time_col, kind="mergesort")
        F = g[feature_cols].to_numpy(dtype=np.float32)
        T = g[target_cols].to_numpy(dtype=np.float32)
        ts = g[time_col].to_numpy()
        if len(g) < lookback: 
            continue
        for i in range(lookback, len(g)):  # window [i-lookback, i)
            Xseq.append(F[i-lookback:i])
            ymat.append(T[i])              # predict forward returns anchored at time ts[i]
            ts_last.append(ts[i])
    Xseq = np.stack(Xseq)           # (N, L, F)
    ymat = np.stack(ymat)           # (N, n_tasks)
    ts_last = np.asarray(ts_last)   # (N,)

    # 5) time-aware split (no leakage)
    if val_start_ts is not None:
        try:
            cut = np.datetime64(val_start_ts)
        except Exception:
            cut = val_start_ts  # numeric timestamps still ok
        tr_idx = ts_last < cut
        va_idx = ~tr_idx
        train_ts = set(np.unique(ts_last[tr_idx]))
        val_ts   = set(np.unique(ts_last[va_idx]))
    else:
        tr_idx, va_idx, train_ts, val_ts = create_time_aware_split(ts_last, train_ratio=train_ratio)

    pack = {
        "X_tr": Xseq[tr_idx], "X_va": Xseq[va_idx],
        "y_tr": ymat[tr_idx], "y_va": ymat[va_idx],
        "ts_tr": ts_last[tr_idx], "ts_va": ts_last[va_idx],
        "feature_cols": list(feature_cols),
        "task_names": list(target_cols),
        "lookback": lookback, "time_col": time_col,
        "train_ts": train_ts, "val_ts": val_ts
    }
    return pack

def _pick_one(colnames, base):
    """
    Pick exactly one column matching the base name (exact preferred, else shortest suffix).
    
    ✅ ENHANCED: Better error handling for ambiguous targets to prevent tolist() crashes.
    """
    # Exact match first (highest priority)
    if base in colnames:
        return base
    
    # Look for prefix matches
    candidates = [c for c in colnames if c.startswith(base + "_")]
    
    if not candidates:
        return None
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Multiple candidates - this causes the tolist() crash!
    logger.error(f"❌ AMBIGUOUS TARGET '{base}': {len(candidates)} matches found")
    logger.error(f"   Candidates: {candidates}")
    logger.error(f"   This will cause tolist() crash when df['{base}'] returns 2-D DataFrame")
    logger.error(f"   Solution: Use exact column name or fix duplicate columns")
    
    # Fail fast instead of silently picking first
    raise ValueError(f"Ambiguous target '{base}': {len(candidates)} matches found: {candidates}. Use exact column name.")

def strip_targets(cols, all_targets=None):
    """
    Remove ALL target-like columns from feature list.
    
    Args:
        cols: List of column names
        all_targets: Set of all discovered target columns (if None, uses heuristics)
    
    Returns:
        List of feature columns only (no targets, no symbol/timestamp)
    """
    if all_targets is None:
        # Fallback to heuristics if all_targets not provided
        EXCLUDE_PREFIXES = ("fwd_ret_", "will_peak", "will_valley", "mdd_", "mfe_", "y_will_")
        return [c for c in cols if not any(c.startswith(p) for p in EXCLUDE_PREFIXES) and c not in ("symbol", "timestamp")]
    else:
        # Use explicit target list for precise filtering
        return [c for c in cols if c not in all_targets and c not in ("symbol", "timestamp")]

def collapse_identical_duplicate_columns(df):
    """
    Collapse identical duplicate columns, raise error if conflicting.
    
    Args:
        df: DataFrame with potentially duplicate columns
        
    Returns:
        DataFrame with unique columns, duplicates removed
    """
    if len(df.columns) == len(set(df.columns)):
        return df  # No duplicates
    
    # Group columns by name
    from collections import defaultdict
    col_groups = defaultdict(list)
    for i, col in enumerate(df.columns):
        col_groups[col].append(i)
    
    # Check for conflicts and remove exact duplicates
    new_cols = []
    for col_name, indices in col_groups.items():
        if len(indices) == 1:
            new_cols.append(col_name)
        else:
            # Multiple columns with same name - check if identical
            first_idx = indices[0]
            is_identical = all(df.iloc[:, idx].equals(df.iloc[:, first_idx]) for idx in indices[1:])
            
            if is_identical:
                # Keep one copy
                new_cols.append(col_name)
            else:
                # Conflicting data - this is the root cause of our crashes
                logger.error(f"❌ CONFLICTING DUPLICATE COLUMNS: '{col_name}' has {len(indices)} copies with different data")
                logger.error(f"   This will cause tolist() crashes when df['{col_name}'] returns 2-D DataFrame")
                raise ValueError(f"Conflicting duplicate columns found: '{col_name}' has {len(indices)} copies with different data")
    
    return df[new_cols]

def prepare_training_data_cross_sectional(
    mtf_data: Dict[str, pd.DataFrame],
    target: str,
    common_features: List[str],
    min_cs: int,
    max_samples_per_symbol: int = None,  # ignore per-symbol sampling for CS
    all_targets: set = None  # Set of all discovered target columns for precise filtering
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.Series, List[str]]:
    """
    Prepare TRUE cross-sectional training data with timestamp-based grouping.
    Clean implementation using long-format schema to prevent column collisions.
    
    Args:
        mtf_data: Dictionary of symbol -> DataFrame
        target: Target column name
        common_features: List of common feature columns
        max_samples_per_symbol: Ignored for CS training (we need all data for grouping)
    
    Returns:
        X: Feature matrix
        y: Target vector
        symbols: Symbol array
        groups: Group sizes for ranking
        ts_index: Timestamp index
        feat_cols: Actual feature columns used (may differ from common_features)
    """
    if not USE_POLARS:
        return _prepare_training_data_cross_sectional_pandas(mtf_data, target, common_features, min_cs, max_samples_per_symbol)

    logger.info("🎯 Building TRUE cross-sectional training data (polars, clean)…")
    if not mtf_data:
        return None, None, None, None, None, None

    # Check if target exists in any of the datasets
    target_found = False
    for sym, pdf in mtf_data.items():
        if target in pdf.columns:
            target_found = True
            break
    
    if not target_found:
        logger.error(f"Target '{target}' not found in any dataset. Available targets: {[col for col in next(iter(mtf_data.values())).columns if col.startswith('fwd_ret_')]}")
        return None, None, None, None, None, None

    # Detect time col
    time_col = resolve_time_col(next(iter(mtf_data.values())))
    SYM = SYMBOL_COL

    # Build CS table in long format (one row per symbol-timestamp)
    lfs = []
    processed_symbols = 0
    for sym, pdf in mtf_data.items():
        cols = pdf.columns.tolist()
        
        # Pick exactly one target-like column (handles suffixes: _x/_y/_SYM/etc)
        tgt_in = _pick_one(cols, target)
        if tgt_in is None:
            # no target in this symbol → skip
            continue
        
        # ✅ NEW SAFE PATTERN: Strip ALL targets from features
        have_feats = [c for c in strip_targets(common_features, all_targets) if c in cols]
        if not have_feats:
            continue
        
        # Build selection: features + exactly one target + metadata
        sel = [time_col] + have_feats + [tgt_in]
        logger.info(f"Symbol {sym}: Selecting columns: {len(sel)} total")
        logger.info(f"Symbol {sym}: Target column '{tgt_in}' in selection: {tgt_in in sel}")
        df_subset = pdf[sel].copy()
        
        # Rename target column to canonical name
        if tgt_in != target:
            df_subset = df_subset.rename(columns={tgt_in: target})
        
        # ✅ NEW SAFE PATTERN: Collapse duplicate columns safely
        try:
            df_subset = collapse_identical_duplicate_columns(df_subset)
        except ValueError as e:
            logger.error(f"Symbol {sym}: {e}")
            continue
        
        logger.info(f"Symbol {sym}: After selection - shape: {df_subset.shape}, target column: {target in df_subset.columns}")
        
        # ✅ NEW SAFE PATTERN: Final contract validation
        if target in df_subset.columns:
            try:
                from target_resolver import safe_target_extraction
                target_series, actual_col = safe_target_extraction(df_subset, target)
                logger.info(f"Symbol {sym}: Target values sample: {target_series.head(3).tolist()}")
                
                # ✅ Contract validation: X = features only, y = exactly one target
                feature_cols = [c for c in df_subset.columns if c not in (target, time_col, "symbol")]
                assert target not in feature_cols, f"Target '{target}' leaked into features!"
                assert len(feature_cols) > 0, "No features found after target filtering"
                
            except Exception as e:
                logger.error(f"Symbol {sym}: Target extraction failed for '{target}': {e}")
                continue
        
        try:
            # ✅ Make sure target is numeric BEFORE Polars (see: all-NULL issue)
            df_subset[target] = pd.to_numeric(df_subset[target], errors="coerce")
            nnz = int(df_subset[target].notna().sum())
            if nnz == 0:
                logger.warning(f"Symbol {sym}: target '{target}' is all-NaN in pandas (before Polars); skipping")
                continue

            # Convert to Polars
            lf = pl.from_pandas(df_subset)
            logger.info(f"Symbol {sym}: After pandas->polars conversion: {lf.shape}")
            
            # Debug: check target column immediately after conversion
            try:
                target_after_conv = lf.select(pl.col(target).count()).item()
                target_nulls_after_conv = lf.select(pl.col(target).null_count()).item()
                logger.info(f"Symbol {sym}: After pandas->polars - target count: {target_after_conv}, nulls: {target_nulls_after_conv}")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not check target after conversion: {e}")

            # --- Timestamp handling (do NOT overwrite ts with year etc.) ---
            if lf[time_col].dtype in (pl.Int64, pl.UInt64, pl.Int32, pl.UInt32):
                # infer unit by span and convert with from_epoch
                span = max(abs(int(lf[time_col].min() or 0)), abs(int(lf[time_col].max() or 0)))
                def _unit_from_span(v: int) -> str:
                    if v >= 1_000_000_000_000_000_000: return "ns"
                    if v >= 1_000_000_000_000_000:     return "us"
                    if v >= 1_000_000_000_000:         return "ms"
                    return "s"
                unit = _unit_from_span(span)
                lf = lf.with_columns(pl.from_epoch(pl.col(time_col).cast(pl.Int64), time_unit=unit).alias(time_col))
            elif lf[time_col].dtype == pl.Datetime:
                lf = lf.with_columns(pl.col(time_col).dt.cast_time_unit("us").alias(time_col))
            else:
                lf = lf.with_columns(pl.col(time_col).str.strptime(pl.Datetime, strict=False).alias(time_col))

            # Debug: check timestamp range before filtering
            try:
                ts_epoch = lf.select(pl.col(time_col).dt.epoch("us").alias("__t_us"))
                ts_min = ts_epoch.select(pl.col("__t_us").min()).item()
                ts_max = ts_epoch.select(pl.col("__t_us").max()).item()
                logger.info(f"Symbol {sym}: Timestamp range: {ts_min} to {ts_max} (μs)")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not check timestamp range: {e}")
                ts_min = 0
                ts_max = 0
            
            # Clamp timestamps to [1970, 2100] (μs) and add symbol
            MIN_US = 0
            MAX_US = 4102444800000000  # 2100-01-01 UTC in μs
            # But if timestamps are in nanoseconds, adjust the range
            if ts_min > 1_000_000_000_000_000_000:  # If timestamps are in nanoseconds
                MIN_US = MIN_US * 1000  # Convert to nanoseconds
                MAX_US = MAX_US * 1000  # Convert to nanoseconds
                logger.info(f"Symbol {sym}: Adjusted filter range for nanoseconds: {MIN_US} to {MAX_US}")
            else:
                logger.info(f"Symbol {sym}: Using microsecond filter range: {MIN_US} to {MAX_US}")
            
            lf = (
                lf.with_columns(pl.col(time_col).dt.epoch("us").alias("__t_us"))
                  .filter(pl.col("__t_us").is_between(MIN_US, MAX_US))
                  .with_columns(pl.from_epoch(pl.col("__t_us"), time_unit="us").alias(time_col))
                  .drop(["__t_us"])
                  .with_columns(pl.lit(sym).alias("symbol"))
            )
            
            # Debug: check target column after timestamp processing
            try:
                target_after_ts = lf.select(pl.col(target).count()).item()
                target_nulls_after_ts = lf.select(pl.col(target).null_count()).item()
                logger.info(f"Symbol {sym}: After timestamp processing - target count: {target_after_ts}, nulls: {target_nulls_after_ts}")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not check target after timestamp processing: {e}")

            # ✅ Robust target filter: cast to float and keep *finite* values
            # Debug: check target before filtering
            try:
                target_before = lf.select(pl.col(target).count()).item()
                target_nulls = lf.select(pl.col(target).null_count()).item()
                logger.info(f"Symbol {sym}: Before finite filter - total: {target_before}, nulls: {target_nulls}")
            except Exception as e:
                logger.warning(f"Symbol {sym}: Could not get target debug info: {e}")
            
            t = pl.col(target).cast(pl.Float64, strict=False).alias(target)
            finite_mask = t.is_not_null() & t.is_finite()
            lf = lf.with_columns(t).filter(finite_mask)

            if lf.height == 0:
                logger.warning(f"Symbol {sym}: all rows dropped after finite target filter; schema={lf.schema}")
                # Debug: check what happened to the target values
                try:
                    # ✅ FIX: Safe target extraction to prevent tolist() crash on 2-D DataFrames
                    from target_resolver import safe_target_extraction
                    target_series, actual_col = safe_target_extraction(df_subset, target)
                    logger.info("Sample target values (raw pandas): %s", target_series.astype("string").head(5).tolist())
                    # Check if the issue is with the cast
                    import numpy as np
                    test_cast = target_series.astype('float64')
                    logger.info(f"Pandas cast to float64: {test_cast.head(5).tolist()}")
                    logger.info(f"Pandas finite check: {np.isfinite(test_cast).sum()} / {len(test_cast)}")
                except Exception as e:
                    logger.warning(f"Debug failed: {e}")
                continue

            lfs.append(lf)
            processed_symbols += 1
        except Exception as e:
            logger.warning(f"Symbol {sym}: Failed to process datetime column: {e}")
            continue

    logger.info(f"Processed {processed_symbols} symbols with target {target}")
    if not lfs:
        logger.error("No data with target present across symbols")
        return None, None, None, None, None, None

    # Vertical concat = one target column, not one per symbol
    big = pl.concat(lfs, how="vertical")
    logger.info(f"After concat: {big.shape}")

    # Filter timestamps with enough cross-section
    cs_counts = big.group_by(time_col).agg(pl.col("symbol").n_unique().alias("_n"))
    big = (big.join(cs_counts, on=time_col, how="left")
              .filter(pl.col("_n") >= min_cs)
              .drop("_n")
              .sort(time_col))
    logger.info(f"After CS filter: {big.shape}")

    # Sort by time (finite target filtering already done per-symbol)
    big = big.sort(time_col)
    logger.info(f"After sort: {big.shape}")

    # Timestamps are already sanitized per-symbol, no need to re-sanitize
    logger.info(f"Final big shape: {big.shape}")

    # Decide time cut using epoch microseconds (no Python datetime conversion)
    ts_us = big.select(pl.col(time_col).dt.epoch("us").alias("t")).unique().sort("t")["t"]
    logger.info(f"Found {len(ts_us)} unique timestamps")
    if len(ts_us) < 2:
        logger.error(f"Not enough distinct timestamps: {len(ts_us)}")
        return None, None, None, None, None, None
    
    cut_idx = int(0.8 * len(ts_us))
    t_cut_us = ts_us[cut_idx]
    
    def _collect(ldf):
        # Check if it's a LazyFrame or DataFrame
        if hasattr(ldf, 'collect'):
            # LazyFrame
            return (ldf.with_columns([pl.col(c).cast(pl.Float32) for c in common_features + [target]])
                        .collect(streaming=True)
                        .to_pandas(use_pyarrow_extension_array=False))
        else:
            # DataFrame
            return (ldf.with_columns([pl.col(c).cast(pl.Float32) for c in common_features + [target]])
                        .to_pandas(use_pyarrow_extension_array=False))
    
    pdf_tr = _collect(big.filter(pl.col(time_col).dt.epoch("us") < t_cut_us))
    pdf_va = _collect(big.filter(pl.col(time_col).dt.epoch("us") >= t_cut_us))
    
    # stitch back if your caller still expects a single arrays pack
    pdf = pd.concat([pdf_tr, pdf_va], ignore_index=True)
    
    if pdf.empty:
        logger.error("No timestamps with sufficient cross-section after filtering")
        return None, None, None, None, None, None
    
    # Sanity checks to add (cheap, helpful)
    logger.info(
        "Final target health: not-null ratio=%.3f, min=%s, max=%s",
        float(pd.notna(pdf[target]).mean()),
        pd.Series(pdf[target]).min(skipna=True),
        pd.Series(pdf[target]).max(skipna=True),
    )

    # Final min_cs check (cheap)
    counts = pdf.groupby(time_col)["symbol"].nunique()
    keep_ts = counts[counts >= min_cs].index
    pdf = pdf[pdf[time_col].isin(keep_ts)]

    # Features - strip targets and make unique
    common_features = strip_targets(list(dict.fromkeys(common_features)))
    feat_cols = [c for c in common_features if c in pdf.columns and pd.api.types.is_numeric_dtype(pdf[c])]
    leak_cols = {'ts','timestamp','time','date','symbol','interval','source','ticker','id','index', time_col}
    feat_cols = [c for c in feat_cols if c not in leak_cols]
    if not feat_cols:
        logger.error("No numeric common features present after leak protection")
        return None, None, None, None, None, None

    # Arrays for learners
    import numpy as np
    X = pdf[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = pdf[target].to_numpy(dtype=np.float32, copy=False)
    symbols = pdf["symbol"].astype('U32').to_numpy()
    ts_index = pdf[time_col]  # Pandas Series is fine for your split code

    # Group sizes (RLE on sorted timestamps)
    sizes = (pdf[time_col].ne(pdf[time_col].shift()).cumsum()
             .groupby(pdf[time_col].ne(pdf[time_col].shift()).cumsum()).size().tolist())

    logger.info(f"✅ CS data: rows={len(ts_index)}, features={len(feat_cols)}, times={len(sizes)}")
    return X, y, symbols, sizes, ts_index, feat_cols

def _prepare_training_data_cross_sectional_pandas(
    mtf_data: Dict[str, pd.DataFrame],
    target: str,
    common_features: List[str],
    min_cs: int,
    max_samples_per_symbol: int = None  # ignore per-symbol sampling for CS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], pd.Series, List[str]]:
    """Original pandas-based CS dataset assembly."""
    logger.info("🎯 Building TRUE cross-sectional training data...")

    # 1) Detect time column from first dataframe
    if not mtf_data:
        logger.error("No data provided")
        return None, None, None, None, None, None
    first_df = next(iter(mtf_data.values()))
    time_col = resolve_time_col(first_df)
    logger.info(f"Using time column: {time_col}")

    # 2) concat all symbols
    dfs = []
    for sym, df in mtf_data.items():
        if target not in df.columns:
            logger.warning(f"Target {target} not found in {sym}, skipping")
            continue
        
        # Verify time column exists in this symbol
        if time_col not in df.columns:
            logger.warning(f"Time column {time_col} not found in {sym}, skipping")
            continue
            
        d = df[[time_col, target] + [c for c in common_features if c in df.columns]].copy()
        d[SYMBOL_COL] = sym
        d = d.dropna(subset=[target])        # targets must be finite
        dfs.append(d)
    
    if not dfs:
        logger.error("No data with target present across symbols")
        return None, None, None, None, None, None

    df = pd.concat(dfs, ignore_index=True)
    
    # Convert symbol column to category for memory efficiency
    df[SYMBOL_COL] = df[SYMBOL_COL].astype('category')
    
    logger.info(f"Concatenated data shape: {df.shape}")
    logger.info(f"Concatenated columns: {len(df.columns)}")
    
    # Remove duplicate columns if any
    if df.columns.duplicated().any():
        logger.warning(f"Removing duplicate columns: {df.columns[df.columns.duplicated()].tolist()}")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # CRITICAL: Apply row capping to prevent OOM
    MAX_ROWS_PER_BATCH = int(os.environ.get("MAX_ROWS_PER_BATCH", "8000000"))  # 8M rows max
    if len(df) > MAX_ROWS_PER_BATCH:
        logger.warning(f"⚠️  Dataset too large ({len(df)} rows), capping to {MAX_ROWS_PER_BATCH} rows")
        # Sample uniformly by timestamp to preserve cross-sectional structure
        unique_ts = df[time_col].unique()
        avg_per_ts = max(1, len(df) // max(1, len(unique_ts)))
        n_ts_needed = max(1, MAX_ROWS_PER_BATCH // avg_per_ts)
        if n_ts_needed < len(unique_ts):
            rng = np.random.default_rng(42)  # Deterministic sampling
            selected_ts = rng.choice(unique_ts, size=n_ts_needed, replace=False)
            df = df[df[time_col].isin(selected_ts)].copy()
            logger.info(f"📊 After capping: {len(df)} rows, {len(df[time_col].unique())} timestamps")
        logger.info(f"📊 Final dataset shape: {df.shape}")
    # enforce numeric features
    feat_cols = [c for c in common_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    # Final leak protection - exclude common time/symbol columns
    leak_cols = {'ts', 'timestamp', 'time', 'date', 'symbol', 'interval', 'source', 'ticker', 'id', 'index'}
    leak_cols.add(time_col)  # Add the detected time column
    feat_cols = [c for c in feat_cols if c not in leak_cols]
    
    if not feat_cols:
        logger.error("No numeric common features present after leak protection")
        return None, None, None, None, None, None

    # 2) keep timestamps with enough symbols (CS)
    # Ensure we get a Series, not DataFrame
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]
    counts = df.groupby(time_col)[SYMBOL_COL].nunique()
    keep_ts = counts[counts >= min_cs].index
    df = df[time_series.isin(keep_ts)]
    if df.empty:
        logger.error("No timestamps with sufficient cross-section")
        return None, None, None, None, None, None

    # 3) sort by time - NO preprocessing yet (avoid leakage)
    df = df.sort_values(time_col, kind="mergesort")
    # Update time_series after filtering and sorting
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]

    # 4) remove rows with NaN targets / infinities
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    
    # Convert targets to float32 for memory efficiency
    df[target] = df[target].astype(np.float32)
    
    # Refresh time_series after any row filtering
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]
    
    # 5) Final min_cs check after all filtering operations
    counts = df.groupby(time_col)[SYMBOL_COL].nunique()
    keep_ts = counts[counts >= min_cs].index
    df = df[df[time_col].isin(keep_ts)]
    if df.empty:
        logger.error("No timestamps with sufficient cross-section after all filtering")
        return None, None, None, None, None, None

    # Refresh time_series after final filtering to ensure alignment
    time_series = df[time_col] if isinstance(df[time_col], pd.Series) else df[time_col].iloc[:, 0]

    # 6) build groups array efficiently (rows per timestamp)
    logger.info("🔧 Building group sizes...")
    grp_sizes = df.groupby(time_col).size().tolist()
    ts_index = time_series

    # 6) Assert group integrity
    n = len(df)
    assert sum(grp_sizes) == n, f"group size mismatch {sum(grp_sizes)} != {n}"
    assert min(grp_sizes) >= min_cs, f"min CS {min(grp_sizes)} < min_cs={min_cs}"

    # 7) assemble arrays with memory optimization
    logger.info("🔧 Assembling feature matrix (memory-safe)...")
    
    # Memory monitoring
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        logger.info(f"💾 RSS before assemble: {rss:.1f} GiB")
    except Exception:
        pass
    
    # compact encodings first
    df[SYMBOL_COL] = df[SYMBOL_COL].astype('category')
    
    # cheap views for ids & time
    symbols = df[SYMBOL_COL].cat.codes.to_numpy(dtype=np.int32, copy=False)
    # Safe datetime handling - convert nanoseconds to datetime
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        # Handle nanoseconds (divide by 1000 to get microseconds)
        if df[time_col].dtype == 'int64' and df[time_col].max() > 1e15:  # Likely nanoseconds
            df[time_col] = pd.to_datetime(df[time_col] / 1000, unit='us', errors="coerce")
        else:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    if getattr(df[time_col].dt, "tz", None) is not None:
        df[time_col] = df[time_col].dt.tz_convert(None)
    
    # Filter out invalid timestamps (year > 2100 or < 1900)
    valid_years = (df[time_col].dt.year >= 1900) & (df[time_col].dt.year <= 2100)
    df = df[valid_years].copy()
    
    ts_index = df[time_col].astype('int64', copy=False).to_numpy(copy=False)
    
    # one extraction, no temporary float64 frames
    X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y = df[target].to_numpy(dtype=np.float32, copy=False)
    
    # spill to file if huge (keeps peak RAM flat)
    if int(os.getenv("USE_MEMMAP", "1")) and X.nbytes > 8_000_000_000:  # ~8 GB
        mm_dir = Path(os.getenv("MEMMAP_DIR", "/tmp/mmap"))
        mm_dir.mkdir(parents=True, exist_ok=True)
        mm_path = mm_dir / f"X_{X.shape[0]}x{X.shape[1]}.f32.mmap"
        X_mm = np.memmap(mm_path, dtype=np.float32, mode="w+", shape=X.shape)
        X_mm[:] = X  # one pass copy
        X = X_mm
    
    # convert back to original format for compatibility
    ts = df[time_col].to_numpy()
    
    # free the big frame promptly
    del df
    import gc; gc.collect()
    
    # Memory monitoring after assembly
    try:
        import psutil, os
        rss = psutil.Process(os.getpid()).memory_info().rss / 1024**3
        logger.info(f"💾 RSS after assemble: {rss:.1f} GiB")
    except Exception:
        pass
    
    # Additional memory cleanup for 10M rows (optional)
    try:
        import psutil
        process = psutil.Process()
        logger.info(f"Memory before cleanup: {process.memory_info().rss / 1024**3:.1f} GB")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        logger.info(f"Memory after cleanup: {process.memory_info().rss / 1024**3:.1f} GB")
    except ImportError:
        # psutil not available, just do basic cleanup
        for _ in range(3):
            gc.collect()
        logger.info("🧹 Basic memory cleanup completed (psutil not available)")

    # Memory management is handled by symbol batching (--batch-size)
    # No need for additional row capping that would throw away data
    logger.info(f"📊 Dataset loaded: {len(ts_index)} rows, {len(pd.unique(ts_index))} timestamps")
    
    # Note: max_samples_per_symbol is ignored in cross-sectional mode
    # (we use all available data per timestamp for CS training)

    # 9) Log group size distribution
    g = np.array(grp_sizes)
    logger.info(f"✅ CS data: rows={len(ts_index)}, features={len(feat_cols)}, times={len(grp_sizes)}")
    logger.info(f"📊 CS group sizes mean={g.mean():.1f}, p5={np.percentile(g,5):.0f}, p95={np.percentile(g,95):.0f}")
    

    return X, y, symbols, grp_sizes, ts_index, feat_cols


def _train_lgb_cpu(train_data, val_data=None, is_ranker=False, num_threads=12):
    """Train LightGBM on CPU only."""
    import lightgbm as lgb
    
    base = dict(
        boosting_type='gbdt',
        max_bin=255,                 # fine on CPU; 255 is fast/compact
        learning_rate=0.03,
        feature_fraction=0.8,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l1=1.0, lambda_l2=10.0,
        verbose=-1, seed=42,
        feature_fraction_seed=42, bagging_seed=42,
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
        boosting_type='gbdt', max_bin=511, learning_rate=0.03,
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
                optimizer=Adam(learning_rate=0.001),
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
                        optimizer=Adam(learning_rate=0.001),
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
            optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
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
                optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
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
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
        
        # Use provided validation data if available, otherwise do random split
        if X_va is None or y_va is None:
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
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
        
        model = GradientBoostingRegressor(
            n_estimators=300,        # Increased for 10M rows
            learning_rate=0.03,      # Reduced for stability at scale
            max_depth=10,           # Increased for 10M rows capacity
            min_samples_split=200,  # Increased regularization for scale
            min_samples_leaf=100,   # Increased regularization for scale
            subsample=0.7,          # More subsampling for 10M rows
            random_state=42
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
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
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
            'random_state': 42,
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
            n_estimators=1500,
            learning_rate=0.05,
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
        X_train, X_val, y_train, y_val = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
        
        # Fit GMM on training data only
        gmm = GaussianMixture(n_components=n_regimes, random_state=42)
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


