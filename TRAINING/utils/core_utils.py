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

"""Core utility functions extracted from original 5K line file."""

import numpy as np
import pandas as pd
import logging
import os
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

def assert_no_nan(df, cols, name):
    """Assert no NaN values in specified columns."""

    bad = df[cols].isna().sum().sum()
    if bad:
        raise ValueError(f"{name}: still has {bad} NaNs after preprocessing")

# All 20 model families
ALL_FAMILIES = [
    'LightGBM',
    'XGBoost', 
    'MLP',
    'CNN1D',
    'LSTM',
    'Transformer',
    'TabCNN',
    'TabLSTM',
    'TabTransformer',
    'RewardBased',
    'QuantileLightGBM',
    'NGBoost',
    'GMMRegime',
    'ChangePoint',
    'FTRLProximal',
    'VAE',
    'GAN',
    'Ensemble',
    'MetaLearning',
    'MultiTask'
]

# Family capabilities map
FAMILY_CAPS = {
    "LightGBM": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "XGBoost": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "MLP": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "CNN1D": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "LSTM": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "Transformer": {"nan_ok": False, "needs_tf": True, "experimental": False, "preprocess_in_family": True},
    "TabCNN": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True},
    "TabLSTM": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True},
    "TabTransformer": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True},
    "RewardBased": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "QuantileLightGBM": {"nan_ok": True, "needs_tf": False, "experimental": False},
    "NGBoost": {"nan_ok": False, "needs_tf": False, "experimental": True, "preprocess_in_family": True},
    "GMMRegime": {"nan_ok": False, "needs_tf": False, "experimental": True, "feature_emitter": False},
    "ChangePoint": {"nan_ok": False, "needs_tf": False, "experimental": True, "feature_emitter": False},
    "FTRLProximal": {"nan_ok": False, "needs_tf": False, "experimental": False, "preprocess_in_family": True},
    "VAE": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True},
    "GAN": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True},
    "Ensemble": {"nan_ok": False, "needs_tf": False, "experimental": False, "preprocess_in_family": True},
    "MetaLearning": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True},
    "MultiTask": {"nan_ok": False, "needs_tf": True, "experimental": True, "preprocess_in_family": True}
}

# Cadence keys for cross-sectional training
CADENCES = ["1m", "5m", "15m", "30m", "1h", "1d"]

# Map cadences to target columns - HFT "few-bars-ahead" scheme
# This will be dynamically generated based on --horizons-min parameter
INTERVAL_TO_TARGET = {}

# Skip classification targets in this regression trainer
SKIP_TARGETS = {"will_peak_k6", "will_valley_k6"}

# Cross-sectional training constants
SYMBOL_COL = "symbol"
# MIN_CS will be set from CLI args



def tf_available():
    """Check if TensorFlow is available."""
    try:
        import tensorflow as tf  # noqa
        return True
    except Exception:
        return False



def ngboost_available():
    """Check if NGBoost is available."""
    try:
        import ngboost  # noqa
        return True
    except Exception:
        return False



def create_time_aware_split(ts_index: pd.Series, train_ratio: float = 0.8):
    """Create time-aware train/validation split to avoid leakage."""
    logger.info("🔧 Creating time-aware split...")
    # Defensive cast to ensure we have a Series
    ts_index = pd.Series(ts_index)
    unique_ts = ts_index.drop_duplicates().sort_values().to_numpy()

    if len(unique_ts) < 2:
        raise ValueError("Need at least 2 distinct timestamps for a time-aware split")

    cut = int(train_ratio * len(unique_ts))
    if cut == 0 or cut == len(unique_ts):
        raise ValueError(
            f"Time split produced an empty set (train_ratio={train_ratio}, "
            f"n_times={len(unique_ts)}, cut={cut}). Try a ratio like 0.7–0.9."
        )

    train_ts, val_ts = set(unique_ts[:cut]), set(unique_ts[cut:])
    tr_idx = ts_index.isin(train_ts).to_numpy()
    va_idx = ~tr_idx

    logger.info(
        f"📊 Time-aware split: train={tr_idx.sum()} rows ({len(train_ts)} timestamps), "
        f"val={va_idx.sum()} rows ({len(val_ts)} timestamps)"
    )
    return tr_idx, va_idx, train_ts, val_ts



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



