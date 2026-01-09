# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

# Set up all paths using centralized utilities
from TRAINING.common.utils.path_setup import setup_all_paths
_PROJECT_ROOT, _TRAINING_ROOT, _CONFIG_DIR = setup_all_paths(_PROJECT_ROOT)

# Import config loader (CONFIG is already in sys.path from setup_all_paths)
try:
    from config_loader import get_pipeline_config, get_family_timeout, get_cfg, get_system_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    import logging
    # Only log at debug level to avoid misleading warnings
    logging.getLogger(__name__).debug("Config loader not available; using hardcoded defaults")

from TRAINING.common.safety import set_global_numeric_guards
set_global_numeric_guards()

# ---- JOBLIB/LOKY CLEANUP: prevent resource tracker warnings ----
from TRAINING.common.utils.process_cleanup import setup_loky_cleanup_from_config
setup_loky_cleanup_from_config()

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

# Validate registries at startup (fail-fast if keys are non-canonical)
try:
    from TRAINING.common.utils.registry_validation import validate_all_registries
    validate_all_registries()
except Exception as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Registry validation skipped: {e}")

# ANTI-DEADLOCK: Process-level safety (before importing TF/XGB/sklearn)
import time as _t
# Make thread pools predictable (also avoids weird deadlocks)


# Import the isolation runner (moved to TRAINING/common/isolation_runner.py)
# Paths are already set up above

from TRAINING.common.isolation_runner import child_isolated
from TRAINING.common.threads import temp_environ, child_env_for_family, plan_for_family, thread_guard, set_estimator_threads
from TRAINING.common.tf_runtime import ensure_tf_initialized
from TRAINING.common.tf_setup import tf_thread_setup

# Family classifications - import from centralized constants
from TRAINING.common.family_constants import TF_FAMS, TORCH_FAMS, CPU_FAMS, TORCH_SEQ_FAMILIES


"""Main entry point for training strategies."""

# Import all dependencies
from .training import train_models_for_interval_comprehensive, train_model_comprehensive
from TRAINING.training_strategies.strategy_functions import load_mtf_data, discover_targets, prepare_training_data, create_strategy_config, train_with_strategy, compare_strategies
from TRAINING.training_strategies.utils import (
    setup_logging, ALL_FAMILIES, THREADS, MKL_THREADS_DEFAULT,
    _env_guard, USE_POLARS, FAMILY_CAPS, CROSS_SECTIONAL_MODELS, SEQUENTIAL_MODELS
)

# Standard library imports
import argparse
from datetime import datetime
import logging
import joblib

# Third-party imports
import pandas as pd

# Setup logger (will be reconfigured by setup_logging later, but needed for early calls)
logger = logging.getLogger(__name__)

def main():
    """Main training function with comprehensive approach (replicates original script functionality)"""
    
    parser = argparse.ArgumentParser(description='Enhanced Training with Multiple Strategies - Full Original Functionality')
    # Core arguments
    parser.add_argument('--data-dir', required=True, help='Data directory')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to train on')
    parser.add_argument('--targets', nargs='+', help='Specific targets to train on (default: auto-discover all targets)')
    parser.add_argument('--families', nargs='+', default=ALL_FAMILIES, help='Model families to train (default: all families). If --model-types is specified, will be filtered to that type.')
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
            default_lookback = int(get_cfg("pipeline.sequential.default_lookback", default=64))
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
    
    # Training plan integration
    parser.add_argument('--training-plan-dir', type=str, help='Path to training plan directory (METRICS/training_plan). If provided, will filter targets and model families based on plan.')
    
    args = parser.parse_args()
    
    # Setup logging first (before any logger calls)
    listener = setup_logging(args.log_level)
    # Re-get logger after setup_logging configures it
    logger = logging.getLogger(__name__)
    
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
        # Sequential mode: train BOTH sequential AND cross-sectional models
        # This enables the 2-stage approach (CPU first, then GPU)
        if families == ALL_FAMILIES:
            # Use all models (both sequential and cross-sectional)
            families = ALL_FAMILIES.copy()
            logger.info(f"🎯 Sequential mode: training all models (sequential + cross-sectional): {len(families)} models")
        else:
            # Filter to only include sequential and cross-sectional from provided list
            seq_fams = [f for f in families if f in SEQUENTIAL_MODELS]
            cross_fams = [f for f in families if f in CROSS_SECTIONAL_MODELS]
            families = seq_fams + cross_fams
            logger.info(f"🎯 Sequential mode: training {len(seq_fams)} sequential + {len(cross_fams)} cross-sectional models")
    else:  # both
        logger.info(f"🎯 Training both model types: {len(families)} models")
    
    # Apply 2-stage ordering: CPU models first, then GPU models
    # This prevents thread pollution and ensures efficient resource usage
    # Stage 1: CPU-only models (tree-based, sklearn-based)
    # Stage 2: GPU models (TensorFlow, PyTorch)
    cpu_families = [f for f in families if f in CPU_FAMS]
    gpu_families = [f for f in families if f not in CPU_FAMS]
    
    # Within GPU families, order: TF families first, then Torch families
    # This helps with resource management (TF and Torch can have different GPU memory patterns)
    tf_families = [f for f in gpu_families if f in TF_FAMS]
    torch_families = [f for f in gpu_families if f in TORCH_FAMS]
    other_families = [f for f in gpu_families if f not in TF_FAMS and f not in TORCH_FAMS]
    
    # Final order: Stage 1 (CPU) → Stage 2 (GPU: TF → Torch → Others)
    families = cpu_families + tf_families + torch_families + other_families
    
    if cpu_families:
        logger.info(f"📊 Stage 1 (CPU): {len(cpu_families)} models - {cpu_families[:5]}{'...' if len(cpu_families) > 5 else ''}")
    if gpu_families:
        logger.info(f"📊 Stage 2 (GPU): {len(gpu_families)} models - {len(tf_families)} TF, {len(torch_families)} Torch, {len(other_families)} others")
    
    # Legacy train_order argument (for backward compatibility, but 2-stage takes precedence)
    if args.train_order == 'sequential-first' and args.model_types != 'sequential':
        # Only apply if not in sequential mode (which already uses 2-stage)
        cross_models = [f for f in families if f in CROSS_SECTIONAL_MODELS]
        seq_models = [f for f in families if f in SEQUENTIAL_MODELS]
        families = seq_models + cross_models
        logger.info(f"📊 Legacy order override: {len(seq_models)} sequential → {len(cross_models)} cross-sectional")
    
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
        
        # Auto-detect training plan if not provided and not disabled
        training_plan_dir = None
        if args.no_training_plan:
            logger.debug("Training plan auto-detection disabled")
        elif args.training_plan_dir:
            training_plan_dir = Path(args.training_plan_dir)
        else:
            # Auto-detect: check globals/ first (new location), then METRICS/ as fallback (legacy)
            # This handles the case where intelligent_trainer was run first
            from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
            
            potential_plan_dirs = [
                # New location: globals/training_plan (preferred)
                get_globals_dir(output_dir) / "training_plan",
                get_globals_dir(output_dir.parent) / "training_plan",
                # Legacy location: METRICS/training_plan (fallback)
                output_dir.parent / "METRICS" / "training_plan",  # Same level as output
                output_dir / "METRICS" / "training_plan",  # Inside output_dir
                Path("results") / "METRICS" / "training_plan",  # Common results location
                Path.cwd() / "results" / "METRICS" / "training_plan",  # Current dir results
            ]
            
            for plan_dir in potential_plan_dirs:
                try:
                    if plan_dir.exists() and (plan_dir / "master_training_plan.json").exists():
                        training_plan_dir = plan_dir
                        logger.info(f"📋 Auto-detected training plan: {training_plan_dir}")
                        break
                except Exception as e:
                    logger.debug(f"Error checking {plan_dir}: {e}")
                    continue
        
        # Apply training plan filter if available
        filtered_targets = targets
        target_families_map = None
        
        if training_plan_dir:
            try:
                from TRAINING.orchestration.training_plan_consumer import (
                    load_training_plan,
                    filter_targets_by_training_plan,
                    get_model_families_for_job
                )
                
                # training_plan_dir is already a Path from auto-detection or args
                # But if it came from args.training_plan_dir, it might be a string
                if not isinstance(training_plan_dir, Path):
                    try:
                        training_plan_dir = Path(training_plan_dir)
                    except Exception as e:
                        logger.warning(f"Invalid training_plan_dir: {e}, skipping plan")
                        training_plan_dir = None
                
                if training_plan_dir:
                    training_plan = load_training_plan(training_plan_dir)
                else:
                    training_plan = None
                
                if training_plan:
                    logger.info("📋 Loading training plan for filtering...")
                    
                    # Filter targets based on plan
                    filtered_targets = filter_targets_by_training_plan(
                        targets=targets,
                        training_plan=training_plan,
                        training_type="cross_sectional"
                    )
                    
                    if len(filtered_targets) < len(targets):
                        logger.info(f"📋 Training plan filter applied: {len(targets)} → {len(filtered_targets)} targets")
                    
                    # Get model families per target from plan
                    target_families_map = {}
                    for target in filtered_targets:
                        plan_families = get_model_families_for_job(
                            training_plan,
                            target=target,
                            symbol=None,
                            training_type="cross_sectional"
                        )
                        if plan_families:
                            # Filter to only include families that are in the requested list
                            filtered_plan_families = [f for f in plan_families if f in families]
                            if filtered_plan_families:
                                target_families_map[target] = filtered_plan_families
                    
                    if target_families_map:
                        # If all targets have same families, update global list
                        all_plan_families = set()
                        for target_fams in target_families_map.values():
                            all_plan_families.update(target_fams)
                        
                        # Use intersection if all targets have same families
                        if len(target_families_map) == len(filtered_targets):
                            common_families = set(target_families_map[filtered_targets[0]])
                            for target in filtered_targets[1:]:
                                if target in target_families_map:
                                    common_families &= set(target_families_map[target])
                            
                            if common_families:
                                families = sorted(common_families)
                                logger.info(f"📋 Using model families from training plan: {families}")
                                target_families_map = None  # Clear since we're using global list
                        else:
                            logger.info(f"📋 Using per-target model families from training plan")
                else:
                    logger.warning(f"Training plan not found at {training_plan_dir}, proceeding without filtering")
            except Exception as e:
                logger.warning(f"Failed to load training plan (non-critical): {e}")
        
        logger.info(f"🤖 Training {len(families)} model families: {families[:5]}...")
        logger.info(f"📋 Strategy: {args.strategy}")
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Memory cleanup
        try:
            import gc
            gc.collect()
        except Exception:
            pass
            logger.debug("Memory cleanup: using gc.collect() fallback")
        
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
                'cross_sectional', filtered_targets, mtf_data, families,
                args.strategy, str(output_dir), args.min_cs, args.max_samples_per_symbol,
                args.max_rows_train,
                target_families=target_families_map
            )
            
            # Save results
            joblib.dump(results, output_dir / f'{args.strategy}_results.pkl')
            logger.info(f"Results saved to {output_dir / f'{args.strategy}_results.pkl'}")
            
            # Print summary using consistent counting from results
            # CRITICAL: Use training_summary if available (single source of truth)
            if 'training_summary' in results:
                summary = results['training_summary']
                logger.info(f"✅ {args.strategy} training completed: {summary['total_saved']} successful, "
                          f"{summary['total_failed']} failed, {summary['total_skipped']} skipped "
                          f"(total attempted: {summary['total_attempted']})")
            else:
                # Fallback: count from results dict
                total_models = sum(len(target_results) for target_results in results['models'].values())
                logger.info(f"✅ {args.strategy} training completed: {total_models} models in results dict")
        
        logger.info("🎉 Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
