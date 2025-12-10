"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

 # ---- PATH BOOTSTRAP: ensure project root on sys.path in parent AND children ----
import os, sys
from pathlib import Path

# CRITICAL: Set LD_LIBRARY_PATH for conda CUDA libraries BEFORE any imports
# This must happen before TensorFlow tries to load CUDA libraries
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib = os.path.join(conda_prefix, "lib")
    conda_targets_lib = os.path.join(conda_prefix, "targets", "x86_64-linux", "lib")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    new_paths = []
    if conda_lib not in current_ld_path:
        new_paths.append(conda_lib)
    if conda_targets_lib not in current_ld_path:
        new_paths.append(conda_targets_lib)
    if new_paths:
        updated_ld_path = ":".join(new_paths + [current_ld_path] if current_ld_path else new_paths)
        os.environ["LD_LIBRARY_PATH"] = updated_ld_path

# Show TensorFlow warnings so user knows if GPU isn't working
# os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # Removed - show warnings
# os.environ.setdefault("TF_LOGGING_VERBOSITY", "ERROR")  # Removed - show warnings

# project root: TRAINING/training_strategies/*.py -> parents[2] = repo root
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure Python can import `common`, `model_fun`, etc.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Propagate to spawned processes (spawned interpreter reads PYTHONPATH at startup)
os.environ.setdefault("PYTHONPATH", str(_PROJECT_ROOT))

# Additional safety: ensure the path is in sys.path for child processes
def _ensure_project_path():
    """Ensure project path is available for child processes."""
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

# Call it immediately
_ensure_project_path()

# Set global numeric guards for stability
# Add TRAINING to path for local imports
_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Also add current directory for relative imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

# Add CONFIG directory to path for centralized config loading
_CONFIG_DIR = _PROJECT_ROOT / "CONFIG"
if str(_CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(_CONFIG_DIR))

# Import config loader
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
import atexit
# Set persistent temp folder for joblib memmapping
# Load from config if available, otherwise use default
if _CONFIG_AVAILABLE:
    joblib_temp = get_cfg("system.paths.joblib_temp", config_name="system_config")
    if joblib_temp:
        _JOBLIB_TMP = Path(joblib_temp)
    else:
        _JOBLIB_TMP = Path.home() / "trainer_tmp" / "joblib"
else:
    _JOBLIB_TMP = Path.home() / "trainer_tmp" / "joblib"
_JOBLIB_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(_JOBLIB_TMP))

# Force clean loky worker shutdown at exit to prevent semlock/file leaks
try:
    from joblib.externals.loky import get_reusable_executor
    @atexit.register
    def _loky_shutdown():
        try:
            get_reusable_executor().shutdown(wait=True, kill_workers=True)
        except Exception:
            pass
except Exception:
    pass

"""
Enhanced Training Script with Multiple Strategies - Full Original Functionality

Replicates ALL functionality from train_mtf_cross_sectional_gpu.py but with:
- Modular architecture
- 3 training strategies (single-task, multi-task, cascade)
- All 20 model families from original script
- GPU acceleration
- Memory management
- Batch processing
- Cross-sectional training
- Target discovery
- Data validation
"""

# ANTI-DEADLOCK: Process-level safety (before importing TF/XGB/sklearn)
import time as _t
# Make thread pools predictable (also avoids weird deadlocks)


# Import the isolation runner (moved to TRAINING/common/isolation_runner.py)
# Add TRAINING to path for local imports
_TRAINING_ROOT = Path(__file__).resolve().parent
if str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

# Also add current directory for relative imports
if '.' not in sys.path:
    sys.path.insert(0, '.')

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications
TF_FAMS = {"MLP", "VAE", "GAN", "MetaLearning", "MultiTask"}
TORCH_FAMS = {"CNN1D", "LSTM", "Transformer", "TabCNN", "TabLSTM", "TabTransformer"}
CPU_FAMS = {"LightGBM", "QuantileLightGBM", "RewardBased", "NGBoost", "GMMRegime", "ChangePoint", "FTRLProximal", "Ensemble"}


"""Strategy functions for training."""

# Standard library imports
import logging
from typing import Dict, List, Any, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Import USE_POLARS and polars if available
import os
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"
if USE_POLARS:
    try:
        import polars as pl
    except ImportError:
        USE_POLARS = False

# Setup logger
logger = logging.getLogger(__name__)

# Import dependencies (these functions are defined in strategies.py, not data_preparation.py)
# Remove circular import - functions are defined below

def load_mtf_data(data_dir: str, symbols: List[str], max_rows_per_symbol: int = None) -> Dict[str, pd.DataFrame]:
    """Load MTF data for specified symbols with polars optimization (matches original script behavior)"""
    import time
    data_start = time.time()
    
    logger.info(f"Loading MTF data from {data_dir}")
    print(f"🔄 Loading MTF data from {data_dir}")  # Also print to stdout
    if max_rows_per_symbol:
        logger.info(f"📊 Limiting to {max_rows_per_symbol} most recent rows per symbol")
        print(f"📊 Limiting to {max_rows_per_symbol} most recent rows per symbol")
    else:
        logger.info("📊 Loading ALL data")
        print("📊 Loading ALL data")
    
    mtf_data = {}
    data_path = Path(data_dir)
    
    for symbol in symbols:
        # Try different possible file locations (matching original script)
        possible_paths = [
            data_path / f"symbol={symbol}" / f"{symbol}.parquet",  # New structure
            data_path / f"{symbol}.parquet",  # Direct file
            data_path / f"{symbol}_mtf.parquet",  # Legacy format
        ]
        
        symbol_file = None
        for path in possible_paths:
            if path.exists():
                symbol_file = path
                break
        
        if symbol_file and symbol_file.exists():
            try:
                if USE_POLARS:
                    # Use polars for memory-efficient loading (matching original)
                    lf = pl.scan_parquet(str(symbol_file))
                    
                    # Apply row limit if specified (most recent rows)
                    if max_rows_per_symbol:
                        lf = lf.tail(max_rows_per_symbol)
                    
                    df_pl = lf.collect(streaming=True)
                    df = df_pl.to_pandas(use_pyarrow_extension_array=False)
                    logger.info(f"Loaded {symbol} (polars): {df.shape}")
                else:
                    df = pd.read_parquet(symbol_file)
                    
                    # Apply row limit if specified (most recent rows)
                    if max_rows_per_symbol and len(df) > max_rows_per_symbol:
                        df = df.tail(max_rows_per_symbol)
                        logger.info(f"Limited {symbol} to {max_rows_per_symbol} most recent rows")
                    
                    logger.info(f"Loaded {symbol} (pandas): {df.shape}")
                
                mtf_data[symbol] = df
            except Exception as e:
                logger.error(f"Error loading {symbol}: {e}")
        else:
            logger.warning(f"File not found for {symbol}. Tried: {possible_paths}")
    
    data_elapsed = time.time() - data_start
    logger.info(f"✅ Data loading completed in {data_elapsed:.2f}s")
    print(f"✅ Data loading completed in {data_elapsed:.2f}s")
    
    return mtf_data

def discover_targets(mtf_data: Dict[str, pd.DataFrame], 
                   target_patterns: List[str] = None) -> List[str]:
    """Discover available targets in the data"""
    
    if target_patterns:
        return target_patterns
    
    # Auto-discover targets from first symbol
    if not mtf_data:
        return []
    
    sample_symbol = list(mtf_data.keys())[0]
    sample_df = mtf_data[sample_symbol]
    
    # Common target patterns
    target_columns = []
    for col in sample_df.columns:
        if any(col.startswith(prefix) for prefix in 
              ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_']):
            target_columns.append(col)
    
    logger.info(f"Discovered {len(target_columns)} targets: {target_columns[:10]}...")
    return target_columns

def prepare_training_data(mtf_data: Dict[str, pd.DataFrame], 
                         targets: List[str],
                         feature_names: List[str] = None) -> Dict[str, Any]:
    """Prepare training data for strategy training"""
    
    logger.info("Preparing training data...")
    
    # Optional schema harmonization: align per-symbol frames to a shared schema
    # Controls:
    #   CS_ALIGN_COLUMNS=0 to disable entirely
    #   CS_ALIGN_MODE=union|intersect (default union)
    import os
    align_cols = os.environ.get("CS_ALIGN_COLUMNS", "1") not in ("0", "false", "False")
    if align_cols and mtf_data:
        mode = os.environ.get("CS_ALIGN_MODE", "union").lower()
        first_df = next(iter(mtf_data.values()))
        if mode == "intersect":
            shared = None
            for _sym, _df in mtf_data.items():
                cols = list(_df.columns)
                shared = set(cols) if shared is None else (shared & set(cols))
            ordered = [c for c in first_df.columns if c in (shared or set())]
            for sym, df in mtf_data.items():
                if list(df.columns) != ordered:
                    mtf_data[sym] = df.loc[:, ordered]
            logger.info(f"🔧 Harmonized schema (intersect) with {len(ordered)} columns")
        else:
            # union mode: include all columns seen across symbols; fill missing as NaN
            union = []
            seen = set()
            # Start with first df order for determinism
            for c in first_df.columns:
                union.append(c); seen.add(c)
            for _sym, _df in mtf_data.items():
                for c in _df.columns:
                    if c not in seen:
                        union.append(c); seen.add(c)
            for sym, df in mtf_data.items():
                if list(df.columns) != union:
                    mtf_data[sym] = df.reindex(columns=union)
            logger.info(f"🔧 Harmonized schema (union) with {len(union)} columns")
    
    # Combine all symbol data
    all_data = []
    for symbol, df in mtf_data.items():
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        all_data.append(df_copy)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Extract features and targets
    if feature_names is None:
        # Auto-discover features (exclude targets and metadata)
        feature_names = [col for col in combined_df.columns 
                        if not any(col.startswith(prefix) for prefix in 
                                 ['fwd_ret_', 'will_peak', 'will_valley', 'mdd_', 'mfe_', 'y_will_'])
                        and col not in ['symbol', 'timestamp']]
    
    # Extract feature matrix - handle non-numeric columns
    feature_df = combined_df[feature_names].copy()
    
    # Convert to numeric, coercing errors to NaN
    for col in feature_df.columns:
        feature_df.loc[:, col] = pd.to_numeric(feature_df[col], errors='coerce')
    
    X = feature_df.values.astype(np.float32)
    
    # Extract targets
    y_dict = {}
    for target in targets:
        try:
            target_series, actual_col = safe_target_extraction(combined_df, target)
            y_dict[target] = target_series.values
            logger.info(f"Extracted target {target} from column {actual_col}")
        except Exception as e:
            logger.error(f"Error extracting target {target}: {e}")
    
    # Clean data
    valid_mask = ~np.isnan(X).any(axis=1)
    for target_name, y in y_dict.items():
        valid_mask = valid_mask & ~np.isnan(y)
    
    X_clean = X[valid_mask]
    y_clean = {name: y[valid_mask] for name, y in y_dict.items()}
    
    logger.info(f"Cleaned data: {len(X_clean)} samples, {X_clean.shape[1]} features, {len(y_clean)} targets")
    
    return {
        'X': X_clean,
        'y_dict': y_clean,
        'feature_names': feature_names,
        'target_names': list(y_clean.keys())
    }

def create_strategy_config(strategy: str, targets: List[str], 
                          model_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create configuration for training strategy"""
    
    base_config = {
        'strategy': strategy,
        'targets': targets,
        'models': model_config or {}
    }
    
    if strategy == 'multi_task':
        base_config.update({
            'shared_dim': 128,
            'head_dims': {},
            'loss_weights': {},
            'batch_size': 32,
            'learning_rate': 0.001,
            'n_epochs': 100
        })
    elif strategy == 'cascade':
        base_config.update({
            'gate_threshold': 0.5,
            'calibration_method': 'isotonic',
            'gating_rules': {
                'will_peak_5m': {'action': 'reduce', 'factor': 0.5},
                'will_valley_5m': {'action': 'boost', 'factor': 1.2}
            }
        })
    
    return base_config

def train_with_strategy(strategy: str, training_data: Dict[str, Any], 
                       config: Dict[str, Any]) -> Dict[str, Any]:
    """Train models using specified strategy"""
    
    logger.info(f"Training with strategy: {strategy}")
    
    # Create strategy manager
    if strategy == 'single_task':
        strategy_manager = SingleTaskStrategy(config)
    elif strategy == 'multi_task':
        strategy_manager = MultiTaskStrategy(config)
    elif strategy == 'cascade':
        strategy_manager = CascadeStrategy(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Train models
    results = strategy_manager.train(
        training_data['X'],
        training_data['y_dict'],
        training_data['feature_names']
    )
    
    # Test predictions
    test_predictions = strategy_manager.predict(training_data['X'][:100])
    
    return {
        'strategy_manager': strategy_manager,
        'results': results,
        'test_predictions': test_predictions,
        'success': True
    }

def compare_strategies(training_data: Dict[str, Any], 
                      strategies: List[str] = None) -> Dict[str, Any]:
    """Compare different training strategies"""
    
    if strategies is None:
        strategies = ['single_task', 'multi_task', 'cascade']
    
    logger.info(f"Comparing strategies: {strategies}")
    
    comparison_results = {}
    
    for strategy in strategies:
        logger.info(f"Testing strategy: {strategy}")
        
        try:
            # Create configuration
            config = create_strategy_config(strategy, training_data['target_names'])
            
            # Train with strategy
            result = train_with_strategy(strategy, training_data, config)
            comparison_results[strategy] = result
            
            logger.info(f"✅ {strategy} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {strategy} failed: {e}")
            comparison_results[strategy] = {
                'success': False,
                'error': str(e)
            }
    
    return comparison_results

