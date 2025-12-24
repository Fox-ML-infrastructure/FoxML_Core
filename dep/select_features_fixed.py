# MIT License - see LICENSE file

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import json
from collections import defaultdict

# Add project root to path for imports
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from DATA_PROCESSING.utils.logging_setup import CentralLoggingManager
from DATA_PROCESSING.utils.io_helpers import safe_load_dataframe
from CONFIG.config_loader import load_model_config

# Setup logging
CentralLoggingManager()
logger = logging.getLogger(__name__)

def train_and_get_importance(
    symbol: str,
    data_path: Path,
    target_column: str,
    lgbm_config: Dict[str, Any]
) -> Optional[pd.Series]:
    """Trains LightGBM for a single symbol and returns feature importance."""
    try:
        df = safe_load_dataframe(data_path)
        
        # Ensure target column exists
        if target_column not in df.columns:
            logger.warning(f"Skipping {symbol}: Target column '{target_column}' not found")
            return None

        # Drop rows with NaN in target
        df = df.dropna(subset=[target_column])
        if df.empty:
            logger.warning(f"Skipping {symbol}: No valid data after dropping NaN in target")
            return None

        # Prepare data - remove target columns and metadata
        exclude_cols = [col for col in df.columns if col.startswith(('y_', 'fwd_ret_', 'barrier_', 'zigzag_', 'p_')) or col in ['ts', 'datetime', 'symbol', target_column]]
        X = df.drop(columns=exclude_cols, errors='ignore')
        y = df[target_column]

        feature_names = X.columns.tolist()
        if not feature_names:
            logger.warning(f"Skipping {symbol}: No features found after filtering")
            return None

        # Create LightGBM Dataset
        lgb_data = lgb.Dataset(X, label=y, feature_name=feature_names, free_raw_data=False)

        # Train LightGBM model
        logger.info(f"Training LightGBM for {symbol} with {len(feature_names)} features...")
        model = lgb.train(
            params=lgbm_config,
            train_set=lgb_data,
            num_boost_round=lgbm_config.get("n_estimators", 100),
            callbacks=[lgb.log_evaluation(period=0)]  # Suppress verbose output
        )

        # Get feature importance
        importance = pd.Series(model.feature_importance(importance_type='gain'), index=feature_names)
        logger.info(f"Finished training for {symbol}. Top 5 features: \n{importance.nlargest(5)}")
        return importance

    except Exception as e:
        logger.error(f"Error processing {symbol}: {e}", exc_info=True)
        return None

def aggregate_importance(
    all_importances: List[pd.Series],
    method: str = "mean",
    top_n: Optional[int] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Aggregates feature importance from multiple symbols."""
    if not all_importances:
        return pd.DataFrame(), []

    # Combine all importances into a single DataFrame
    combined_df = pd.concat(all_importances, axis=1, sort=False).fillna(0)

    # Calculate aggregated importance
    if method == "mean":
        aggregated_scores = combined_df.mean(axis=1)
    elif method == "median":
        aggregated_scores = combined_df.median(axis=1)
    elif method == "frequency":
        # Count how many times each feature appeared with non-zero importance
        feature_frequency = (combined_df > 0).sum(axis=1)
        aggregated_scores = feature_frequency
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        "feature": aggregated_scores.index,
        "score": aggregated_scores.values,
        "frequency": (combined_df > 0).sum(axis=1),
        "frequency_pct": ((combined_df > 0).sum(axis=1) / combined_df.shape[1]) * 100
    }).sort_values(by="score", ascending=False).reset_index(drop=True)

    # Select top N features
    if top_n:
        summary_df = summary_df.head(top_n)
    
    selected_features = summary_df["feature"].tolist()
    return summary_df, selected_features

def save_results(results: Tuple[pd.DataFrame, List[str]], output_dir: Path):
    """Saves the feature selection results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_df, selected_features = results

    # Save ranked list of features
    with open(output_dir / "selected_features.txt", "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    logger.info(f"Saved ranked features to {output_dir}/selected_features.txt")

    # Save detailed importance summary
    summary_df.to_csv(output_dir / "feature_importance_summary.csv", index=False)
    logger.info(f"Saved detailed importance summary to {output_dir}/feature_importance_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Perform feature selection using LightGBM importance.")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols to process (e.g., AAPL,MSFT). If not provided, all symbols in data_labeled will be used.")
    parser.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data" / "data_labeled" / "interval=5m", help="Directory containing labeled data parquet files.")
    parser.add_argument("--output-dir", type=Path, default=_REPO_ROOT / "DATA_PROCESSING" / "data" / "features", help="Output directory for selected features.")
    parser.add_argument("--target-column", type=str, default="y_will_peak", help="Target column to use for training LightGBM.")
    parser.add_argument("--top-n", type=int, default=60, help="Number of top features to select.")
    parser.add_argument("--method", type=str, default="mean", choices=["mean", "median", "frequency"], help="Aggregation method for feature importance.")
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 2), help="Number of parallel processes for feature importance calculation.")
    
    args = parser.parse_args()

    logger.info(f"\n{'='*80}")
    logger.info(f"🚀 Starting Feature Selection Pipeline")
    logger.info(f"{'='*80}")
    logger.info(f"Configuration:")
    logger.info(f"  Data Directory: {args.data_dir}")
    logger.info(f"  Output Directory: {args.output_dir}")
    logger.info(f"  Target Column: {args.target_column}")
    logger.info(f"  Top N Features: {args.top_n}")
    logger.info(f"  Aggregation Method: {args.method}")
    logger.info(f"  Parallel Workers: {args.num_workers}")
    logger.info(f"{'-'*80}")

    # Load LightGBM config
    lgbm_config = load_model_config("lightgbm", variant="balanced")
    # Ensure it's a regressor for feature importance
    lgbm_config["objective"] = "regression_l1"  # MAE objective for robustness
    lgbm_config["metric"] = "mae"
    lgbm_config["n_estimators"] = 500  # Moderate number of estimators
    lgbm_config["learning_rate"] = 0.05
    lgbm_config["num_leaves"] = 31
    lgbm_config["max_depth"] = -1
    lgbm_config["min_child_samples"] = 20
    lgbm_config["subsample"] = 0.8
    lgbm_config["colsample_bytree"] = 0.8
    lgbm_config["reg_alpha"] = 0.1
    lgbm_config["reg_lambda"] = 0.1
    lgbm_config["n_jobs"] = 1  # Each worker will run a single-threaded LGBM
    lgbm_config["verbose"] = -1  # Suppress verbose output

    # Find labeled data (Hive partitioned: symbol=XYZ/XYZ.parquet)
    if not args.data_dir.exists():
        logger.error(f"❌ Labeled data directory not found: {args.data_dir}")
        logger.error("   Expected structure: data/data_labeled/interval=5m/symbol=XXX/XXX.parquet")
        return 1
    
    # Discover symbols from directory structure
    symbol_dirs = [d for d in args.data_dir.glob("symbol=*") if d.is_dir()]
    if not symbol_dirs:
        logger.error(f"❌ No symbol directories found in {args.data_dir}")
        return 1
    
    # Extract symbol names and find parquet files
    labeled_files = []
    for symbol_dir in symbol_dirs:
        symbol_name = symbol_dir.name.replace("symbol=", "")
        parquet_file = symbol_dir / f"{symbol_name}.parquet"
        if parquet_file.exists():
            labeled_files.append((symbol_name, parquet_file))
    
    if not labeled_files:
        logger.error(f"❌ No labeled parquet files found in {args.data_dir}")
        return 1
    
    logger.info(f"📁 Found {len(labeled_files)} symbols in {args.data_dir}")
    
    # Filter by symbols if provided
    if args.symbols:
        requested_symbols = [s.upper() for s in args.symbols.split(',')]
        labeled_files = [(sym, path) for sym, path in labeled_files if sym.upper() in requested_symbols]
        if not labeled_files:
            logger.error(f"❌ None of the requested symbols {requested_symbols} found")
            return 1

    logger.info(f"📊 Processing {len(labeled_files)} symbols")

    all_importances = []
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        future_to_symbol = {
            executor.submit(train_and_get_importance, symbol, path, args.target_column, lgbm_config): symbol
            for symbol, path in labeled_files
        }
        
        for i, future in enumerate(as_completed(future_to_symbol)):
            symbol = future_to_symbol[future]
            try:
                importance = future.result()
                if importance is not None:
                    all_importances.append(importance)
                logger.info(f"[{i+1}/{len(labeled_files)}] Processed {symbol}")
            except Exception as exc:
                logger.error(f"{symbol} generated an exception: {exc}")

    if not all_importances:
        logger.error("No feature importances collected. Exiting.")
        return 1
    
    # Aggregate and select features
    summary_df, selected_features = aggregate_importance(all_importances, args.method, args.top_n)
    
    if summary_df.empty:
        logger.error("No features selected after aggregation. Exiting.")
        return 1
    
    # Save results
    save_results((summary_df, selected_features), args.output_dir)
    
    # Save metadata
    metadata = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "method": args.method,
        "top_n": args.top_n,
        "target_column": args.target_column,
        "symbols_processed": len(all_importances),
        "output_dir": str(args.output_dir)
    }
    with open(args.output_dir / "feature_selection_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Saved metadata to {args.output_dir}/feature_selection_metadata.json")
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Feature selection complete!")
    logger.info(f"{'='*80}")
    logger.info(f"\nOutput files:")
    logger.info(f"  📄 {args.output_dir}/selected_features.txt          (ranked list)")
    logger.info(f"  📊 {args.output_dir}/feature_importance_summary.csv (detailed scores)")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review the feature rankings")
    logger.info(f"  2. Use with existing TRAINING pipeline:")
    logger.info(f"     # In your training script, load the features:")
    logger.info(f"     with open('{args.output_dir}/selected_features.txt') as f:")
    logger.info(f"         selected_features = [line.strip() for line in f]")
    logger.info(f"     X_selected = X[selected_features]")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

