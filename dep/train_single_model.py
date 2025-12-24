#!/usr/bin/env python3

# MIT License - see LICENSE file

"""
Train Single Model
Train one model on one symbol using centralized configs.
"""


import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import polars as pl
import joblib
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from CONFIG.config_loader import load_model_config
from TRAINING.model_fun import (
    LightGBMTrainer, XGBoostTrainer, EnsembleTrainer,
    MultiTaskTrainer, MLPTrainer, TransformerTrainer,
    LSTMTrainer, CNN1DTrainer
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODEL_MAP = {
    "lightgbm": LightGBMTrainer,
    "xgboost": XGBoostTrainer,
    "ensemble": EnsembleTrainer,
    "multi_task": MultiTaskTrainer,
    "mlp": MLPTrainer,
    "transformer": TransformerTrainer,
    "lstm": LSTMTrainer,
    "cnn1d": CNN1DTrainer,
}

def train_model(symbol: str, model_name: str, variant: str = None, target: str = "y_will_peak"):
    """Train single model on symbol"""
    
    logger.info(f"{'='*80}")
    logger.info(f"Training {model_name} on {symbol}")
    logger.info(f"{'='*80}")
    
    # Paths
    data_file = PROJECT_ROOT / f"DATA_PROCESSING/data/labeled/{symbol}_labeled.parquet"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    model_file = PROJECT_ROOT / f"models/{symbol}_{model_name}_{timestamp}.pkl"
    metrics_file = PROJECT_ROOT / f"models/{symbol}_{model_name}_{timestamp}_metrics.json"
    model_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Load data
    logger.info(f"[1/5] Loading data from {data_file}")
    if not data_file.exists():
        logger.error(f"❌ File not found: {data_file}")
        logger.error(f"   Run: python SCRIPTS/process_single_symbol.py {symbol}")
        return False
    
    try:
        df = pl.read_parquet(data_file).to_pandas()
        logger.info(f"  ✅ Loaded {len(df):,} rows")
    except Exception as e:
        logger.error(f"❌ Failed to load data: {e}")
        return False
    
    # 2. Prepare features and target
    logger.info(f"[2/5] Preparing features and target")
    try:
        # Separate features and target
        exclude_cols = ['ts', 'open', 'high', 'low', 'close', 'volume', 'datetime', 'symbol', 'interval']
        feature_cols = [c for c in df.columns if not c.startswith('y_') and c not in exclude_cols]
        
        X = df[feature_cols]
        y = df[target]
        
        # Train/validation split (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        logger.info(f"  ✅ Features: {len(feature_cols)}")
        logger.info(f"  ✅ Train: {len(X_train):,} rows")
        logger.info(f"  ✅ Validation: {len(X_val):,} rows")
    except Exception as e:
        logger.error(f"❌ Failed to prepare data: {e}")
        return False
    
    # 3. Load config
    logger.info(f"[3/5] Loading config (variant={variant or 'default'})")
    try:
        config = load_model_config(model_name, variant=variant)
        logger.info(f"  ✅ Loaded config from CONFIG/model_config/{model_name}.yaml")
        logger.info(f"     Key params: n_estimators={config.get('n_estimators', 'N/A')}, "
                   f"learning_rate={config.get('learning_rate', 'N/A')}")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return False
    
    # 4. Train model
    logger.info(f"[4/5] Training {model_name}")
    try:
        TrainerClass = MODEL_MAP[model_name]
        trainer = TrainerClass(config)
        
        trainer.train(
            X_tr=X_train,
            y_tr=y_train,
            X_va=X_val,
            y_va=y_val,
            feature_names=feature_cols
        )
        
        logger.info(f"  ✅ Training complete")
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 5. Evaluate and save
    logger.info(f"[5/5] Evaluating and saving")
    try:
        # Predictions
        y_pred_train = trainer.predict(X_train)
        y_pred_val = trainer.predict(X_val)
        
        # Metrics
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        val_mse = mean_squared_error(y_val, y_pred_val)
        train_r2 = r2_score(y_train, y_pred_train)
        val_r2 = r2_score(y_val, y_pred_val)
        
        # Try classification metrics if binary
        if len(y.unique()) == 2:
            y_pred_train_class = (y_pred_train > 0.5).astype(int)
            y_pred_val_class = (y_pred_val > 0.5).astype(int)
            train_acc = accuracy_score(y_train, y_pred_train_class)
            val_acc = accuracy_score(y_val, y_pred_val_class)
        else:
            train_acc = None
            val_acc = None
        
        metrics = {
            "symbol": symbol,
            "model": model_name,
            "variant": variant or "default",
            "target": target,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "train_mse": float(train_mse),
            "val_mse": float(val_mse),
            "train_r2": float(train_r2),
            "val_r2": float(val_r2),
            "train_accuracy": float(train_acc) if train_acc is not None else None,
            "val_accuracy": float(val_acc) if val_acc is not None else None,
            "timestamp": timestamp,
        }
        
        # Save model
        joblib.dump(trainer.model, model_file)
        logger.info(f"  ✅ Saved model to {model_file}")
        
        # Save config
        config_file = model_file.with_suffix('.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"  ✅ Saved config to {config_file}")
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"  ✅ Saved metrics to {metrics_file}")
        
        # Display metrics
        logger.info(f"\n📊 Metrics:")
        logger.info(f"  Train MSE:  {train_mse:.6f}")
        logger.info(f"  Val MSE:    {val_mse:.6f}")
        logger.info(f"  Train R²:   {train_r2:.4f}")
        logger.info(f"  Val R²:     {val_r2:.4f}")
        if train_acc is not None:
            logger.info(f"  Train Acc:  {train_acc:.4f}")
            logger.info(f"  Val Acc:    {val_acc:.4f}")
        
        # Feature importance (if available)
        try:
            importance = trainer.get_feature_importance()
            logger.info(f"\n📊 Top 10 Features:")
            for i, (feat, imp) in enumerate(importance.head(10).items(), 1):
                logger.info(f"  {i:2d}. {feat:30s} {imp:.4f}")
        except:
            pass
        
    except Exception as e:
        logger.error(f"❌ Evaluation/save failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    logger.info(f"\n✅ Successfully trained {model_name} on {symbol}!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Train single model on symbol")
    parser.add_argument("symbol", help="Symbol to train on (e.g., AAPL)")
    parser.add_argument("model", choices=list(MODEL_MAP.keys()), 
                       help="Model to train")
    parser.add_argument("--variant", choices=["conservative", "balanced", "aggressive"],
                       help="Config variant")
    parser.add_argument("--target", default="y_will_peak",
                       help="Target column (default: y_will_peak)")
    args = parser.parse_args()
    
    success = train_model(args.symbol, args.model, args.variant, args.target)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

