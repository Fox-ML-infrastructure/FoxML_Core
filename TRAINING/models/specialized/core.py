# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""Specialized model classes extracted from original 5K line file."""

import numpy as np
import pandas as pd
import time
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


"""Core training functions and main entry point for specialized models."""

# Import shared constants first
from TRAINING.models.specialized.constants import (
    FAMILY_CAPS, assert_no_nan, tf_available, ngboost_available,
    USE_POLARS, TF_DEVICE, tf, STRATEGY_SUPPORT
)

# Import all dependencies
from TRAINING.models.specialized.wrappers import *
from TRAINING.models.specialized.predictors import *
from TRAINING.models.specialized.trainers import *
from TRAINING.models.specialized.trainers_extended import *
from TRAINING.models.specialized.metrics import *
from TRAINING.models.specialized.data_utils import *

# Additional imports that may be needed
import os
import sys
from pathlib import Path

# Import helper functions
try:
    from TRAINING.common.utils.core_utils import pick_tf_device, set_global_determinism
except ImportError:
    def pick_tf_device():
        return '/CPU:0'
    def set_global_determinism(seed):
        pass

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

def train_with_strategy(strategy: str, mtf_data: Dict[str, pd.DataFrame], target: str, families: List[str], common_features: List[str], output_dir: str, min_cs: int, max_samples_per_symbol: int = 10000, batch_id: int = None, cs_normalize: str = "per_ts_split", args=None, all_targets: set = None):
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
                        help="Specific targets to train on (default: auto-discover all targets)")

    # Strategy arguments
    parser.add_argument("--strategy", choices=['single_task', 'multi_task', 'cascade', 'auto'],
                        default='auto', help="Training strategy (auto = single_task)")
    parser.add_argument("--strategy-config", type=str, help="Path to strategy configuration file")
    
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
                mifolds=3
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

