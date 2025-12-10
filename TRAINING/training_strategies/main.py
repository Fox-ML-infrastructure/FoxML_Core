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


"""Main entry point for training strategies."""

# Import all dependencies
from TRAINING.training_strategies.training import train_models_for_interval_comprehensive, train_model_comprehensive
from TRAINING.training_strategies.strategies import load_mtf_data, discover_targets, prepare_training_data, create_strategy_config, train_with_strategy, compare_strategies
from TRAINING.training_strategies.utils import setup_logging

# Standard library imports
import argparse
from datetime import datetime

# Third-party imports
import pandas as pd

def main():
    """Main training function with comprehensive approach (replicates original script functionality)"""
    
    
    parser = argparse.ArgumentParser(description='Enhanced Training with Multiple Strategies - Full Original Functionality')
    # Core arguments
    parser.add_argument('--data-dir', required=True, help='Data directory')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to train on')
    parser.add_argument('--targets', nargs='+', help='Specific targets to train on (default: auto-discover all targets)')
    parser.add_argument('--families', nargs='+', default=ALL_FAMILIES, help='Model families to train')
    parser.add_argument('--strategy', choices=['single_task', 'multi_task', 'cascade', 'all'], 
                       default='single_task', help='Training strategy')
    parser.add_argument('--seq-backend', choices=['torch', 'tf'], default='torch', 
                       help='Backend for sequential models (default: torch)')
    parser.add_argument('--output-dir', default='modular_output', help='Output directory')
    parser.add_argument('--log-level', default='INFO', help='Log level')
    
    # Data size and sampling controls
    parser.add_argument('--max-symbols', type=int, help='Maximum number of symbols to process')
    parser.add_argument('--max-samples-per-symbol', type=int, default=10000, help='Maximum samples per symbol')
    parser.add_argument('--max-rows-per-symbol', type=int, help='Maximum rows per symbol to prevent OOM (default: no limit)')
    parser.add_argument('--max-rows-train', type=int, default=3000000, help='Maximum rows for training (default: 3000000)')
    parser.add_argument('--max-rows-val', type=int, default=600000, help='Maximum rows for validation (default: 600000)')
    
    # Cross-sectional parameters
    parser.add_argument('--min-cs', type=int, default=10, help='Minimum cross-sectional size per timestamp (default: 10)')
    parser.add_argument('--cs-normalize', choices=['none', 'per_ts_split'], default='per_ts_split', 
                       help='Cross-sectional normalization mode (default: per_ts_split)')
    parser.add_argument('--cs-block', type=int, default=32, help='Block size for CS transforms (default: 32)')
    parser.add_argument('--cs-winsor-p', type=float, default=0.01, help='Winsorization percentile (default: 0.01)')
    parser.add_argument('--cs-ddof', type=int, default=1, help='Degrees of freedom for standard deviation (default: 1)')
    
    # Batch processing
    parser.add_argument('--batch-size', type=int, default=50, help='Number of symbols to process per batch')
    parser.add_argument('--batch-id', type=int, default=0, help='Batch ID for this training run')
    parser.add_argument('--session-id', type=str, default=None, help='Session ID for this training run')
    
    # Model configuration
    parser.add_argument('--experimental', action='store_true', help='Include experimental models')
    parser.add_argument('--include-experimental', action='store_true', help='Include experimental/placeholder model families')
    parser.add_argument('--quantile-alpha', type=float, default=0.5, help='Alpha parameter for QuantileLightGBM (default: 0.5)')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU for all learners (LightGBM/XGBoost)')
    parser.add_argument('--threads', type=int, default=max(1, os.cpu_count() - 1), 
                       help=f'Number of threads for training (default: {max(1, os.cpu_count() - 1)})')
    
    # Model type selection arguments
    parser.add_argument('--model-types', choices=['cross-sectional', 'sequential', 'both'], 
                       default='both', help='Which model types to train (default: both)')
    parser.add_argument('--train-order', choices=['cross-first', 'sequential-first', 'mixed'], 
                       default='cross-first', help='Training order for model types (default: cross-first)')
    
    # Ranking and objectives
    parser.add_argument('--rank-objective', choices=['on', 'off'], default='on', 
                       help='Enable ranking objectives for LGB/XGB (default: on)')
    parser.add_argument('--rank-labels', choices=['dense', 'raw'], default='dense', 
                       help='Ranking label method: dense for dense ranks (default), raw for continuous values')
    
    # Sequence models
    # Load default seq_lookback from config if available
    default_lookback = 64
    if _CONFIG_AVAILABLE:
        try:
            default_lookback = get_cfg("pipeline.sequential.default_lookback", default=64)
        except Exception:
            pass
    parser.add_argument('--seq-lookback', type=int, default=default_lookback, 
                       help=f'Lookback window for temporal sequence models (default: {default_lookback})')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs for sequence models (default: 50, use 1000 for production)')
    
    # Feature management
    parser.add_argument('--feature-list', type=str, help='Path to JSON file of global feature list')
    parser.add_argument('--save-features', action='store_true', help='Save global feature list to features_all.json')
    
    # Validation and debugging
    parser.add_argument('--validate-targets', action='store_true', 
                       help='Run preflight validation checks on targets before training')
    parser.add_argument('--strict-exit', action='store_true', 
                       help='Exit with error code if any model fails (default: only exit on complete failure)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without executing')
    
    # Memory optimization
    parser.add_argument('--use-polars', action='store_true', help='Use polars for memory optimization (default: enabled)')
    parser.add_argument('--no-polars', action='store_true', help='Disable polars, use pandas only')
    
    # Strategy configuration
    parser.add_argument('--strategy-config', type=str, help='Path to strategy configuration file')
    
    args = parser.parse_args()
    
    # Set global backend for sequential models
    global SEQ_BACKEND
    SEQ_BACKEND = args.seq_backend
    logger.info(f"Sequential backend: {SEQ_BACKEND}")
    
    # Handle polars settings
    global USE_POLARS
    if args.no_polars:
        USE_POLARS = False
        logger.info("Polars disabled by user")
    elif args.use_polars:
        USE_POLARS = True
        logger.info("Polars enabled by user")
    
    # Setup logging
    listener = setup_logging(args.log_level)
    
    # Optional: add live stack dumps for any future "quiet" periods
    try:
        import faulthandler, signal
        faulthandler.register(signal.SIGUSR2)  # run: kill -USR2 <pid> to dump all stacks
    except Exception:
        pass
    
    # Set global thread knobs from CLI
    global THREADS, MKL_THREADS_DEFAULT, CPU_ONLY
    THREADS = args.threads              # e.g., 16 on 11700K
    CPU_ONLY = args.cpu_only
    MKL_THREADS_DEFAULT = 1             # default; we'll override per-family
    
    # Apply environment guard with actual CLI values
    _env_guard(THREADS, mkl_threads=MKL_THREADS_DEFAULT)
    
    logger.info("🚀 Starting enhanced training with multiple strategies - Full Original Functionality")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Families: {args.families}")
    logger.info(f"Min cross-sectional size: {args.min_cs}")
    
    # Apply max_symbols limit if specified
    if args.max_symbols:
        args.symbols = args.symbols[:args.max_symbols]
        logger.info(f"Limited to {args.max_symbols} symbols: {args.symbols}")
    
    # Filter families based on experimental flag
    if not args.experimental:
        families = [f for f in args.families if not FAMILY_CAPS.get(f, {}).get('experimental', False)]
        logger.info(f"Filtered to non-experimental families: {families}")
    else:
        families = args.families
    
    # Filter by model type
    if args.model_types == 'cross-sectional':
        families = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        logger.info(f"🎯 Training only cross-sectional models: {len(families)} models")
    elif args.model_types == 'sequential':
        families = [f for f in families if f in SEQUENTIAL_MODELS]
        logger.info(f"🎯 Training only sequential models: {len(families)} models")
    else:  # both
        logger.info(f"🎯 Training both model types: {len(families)} models")
    
    # Sort models by training order
    if args.train_order == 'cross-first':
        # Train cross-sectional models first, then sequential
        cross_models = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        seq_models = [f for f in families if f in SEQUENTIAL_MODELS]
        families = cross_models + seq_models
        logger.info(f"📊 Training order: {len(cross_models)} cross-sectional → {len(seq_models)} sequential")
    elif args.train_order == 'sequential-first':
        # Train sequential models first, then cross-sectional
        cross_models = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        seq_models = [f for f in families if f in SEQUENTIAL_MODELS]
        families = seq_models + cross_models
        logger.info(f"📊 Training order: {len(seq_models)} sequential → {len(cross_models)} cross-sectional")
    else:  # mixed
        logger.info(f"📊 Training order: mixed (as specified)")
    
    # Create output directory with session ID (same as original)
    # Using top-level import: datetime
    session_id = f"mtf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    try:
        # Load data (with optional row limiting like original script)
        logger.info(f"📂 Loading data from {args.data_dir}")
        logger.info(f"📊 Symbols: {args.symbols}")
        logger.info(f"🔢 Max rows per symbol: {args.max_rows_per_symbol}")
        
        mtf_data = load_mtf_data(args.data_dir, args.symbols, args.max_rows_per_symbol)
        if not mtf_data:
            logger.error("No data loaded")
            return
        
        logger.info(f"✅ Loaded data for {len(mtf_data)} symbols")
        for symbol, df in mtf_data.items():
            logger.info(f"  📈 {symbol}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Discover targets
        logger.info(f"🎯 Discovering targets...")
        targets = discover_targets(mtf_data, args.targets)
        if not targets:
            logger.error("No targets found")
            return
        
        # Validate targets if requested
        if args.validate_targets:
            missing, empty = [], []
            for t in targets:
                exists_any = any(t in df.columns for df in mtf_data.values())
                if not exists_any:
                    missing.append(t); continue
                # consider empty if all-NaN across every symbol that has it
                has_any_non_nan = any((t in df.columns) and (~pd.isna(df[t])).any() for df in mtf_data.values())
                if not has_any_non_nan:
                    empty.append(t)
            if missing or empty:
                logger.error(f"Missing targets: {missing} | Empty targets: {empty}")
                if args.strict_exit: 
                    sys.exit(2)
        
        logger.info(f"✅ Found {len(targets)} targets: {targets[:5]}...")
        logger.info(f"🤖 Training {len(families)} model families: {families[:5]}...")
        logger.info(f"📋 Strategy: {args.strategy}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Memory cleanup
        aggressive_cleanup()
        
        # Train with strategy/strategies
        if args.strategy == 'all':
            # Compare all strategies using comprehensive approach
            comparison_results = {}
            for strategy in ['single_task', 'multi_task', 'cascade']:
                logger.info(f"Testing strategy: {strategy}")
                try:
                    result = train_models_for_interval_comprehensive(
                        'cross_sectional', targets, mtf_data, families,
                        strategy, str(output_dir), args.min_cs, args.max_samples_per_symbol,
                        args.max_rows_train
                    )
                    comparison_results[strategy] = result
                    logger.info(f"✅ {strategy} completed successfully")
                except Exception as e:
                    logger.error(f"❌ {strategy} failed: {e}")
                    comparison_results[strategy] = {'success': False, 'error': str(e)}
            
            # Save comparison results
            joblib.dump(comparison_results, output_dir / 'strategy_comparison.pkl')
            logger.info(f"Comparison results saved to {output_dir / 'strategy_comparison.pkl'}")
            
        else:
            # Train with single strategy using comprehensive approach
            results = train_models_for_interval_comprehensive(
                'cross_sectional', targets, mtf_data, families,
                args.strategy, str(output_dir), args.min_cs, args.max_samples_per_symbol,
                args.max_rows_train
            )
            
            # Save results
            joblib.dump(results, output_dir / f'{args.strategy}_results.pkl')
            logger.info(f"Results saved to {output_dir / f'{args.strategy}_results.pkl'}")
            
            # Print summary
            total_models = sum(len(target_results) for target_results in results['models'].values())
            logger.info(f"✅ {args.strategy} training completed: {total_models} models trained")
        
        logger.info("🎉 Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
