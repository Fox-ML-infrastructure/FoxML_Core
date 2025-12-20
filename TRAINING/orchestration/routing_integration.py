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

"""
Routing Integration Hooks

Integration functions to call routing system from existing pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import yaml

from CONFIG.config_loader import CONFIG_DIR
from TRAINING.orchestration.metrics_aggregator import MetricsAggregator
from TRAINING.orchestration.training_router import TrainingRouter

logger = logging.getLogger(__name__)


def generate_routing_plan_after_feature_selection(
    output_dir: Path,
    targets: List[str],
    symbols: List[str],
    routing_config_path: Optional[Path] = None,
    generate_training_plan: bool = True,
    model_families: Optional[List[str]] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate routing plan after feature selection completes.
    
    This is the main integration hook to call after feature selection
    has finished for all targets.
    
    Args:
        output_dir: Base output directory (should contain feature_selections/)
        targets: List of target names that were processed
        symbols: List of symbol names
        routing_config_path: Optional path to routing config (defaults to CONFIG/training_config/routing_config.yaml)
    
    Returns:
        Routing plan dict or None if generation failed
    """
    try:
        logger.info("="*80)
        logger.info("GENERATING TRAINING ROUTING PLAN")
        logger.info("="*80)
        
        # Load routing config
        if routing_config_path is None:
            routing_config_path = CONFIG_DIR / "training_config" / "routing_config.yaml"
        
        if not routing_config_path.exists():
            logger.warning(f"Routing config not found: {routing_config_path}, skipping routing plan generation")
            return None
        
        with open(routing_config_path) as f:
            routing_config = yaml.safe_load(f)
        
        # Get git commit
        git_commit = None
        try:
            from TRAINING.common.subprocess_utils import safe_subprocess_run
            result = safe_subprocess_run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True
            )
            git_commit = result.stdout.strip()
        except Exception:
            pass
        
        # Compute config hash
        config_hash = None
        try:
            import hashlib
            with open(routing_config_path, "rb") as f:
                content = f.read()
            config_hash = hashlib.sha256(content).hexdigest()[:8]
        except Exception:
            pass
        
        # Step 1: Aggregate metrics
        logger.info("Step 1: Aggregating metrics from feature selection outputs...")
        # MetricsAggregator now checks target-first structure first, then falls back to legacy
        # Pass the run directory so it can find targets/<target>/reproducibility/
        aggregator = MetricsAggregator(output_dir)
        
        candidates_df = aggregator.aggregate_routing_candidates(
            targets=targets,
            symbols=symbols,
            git_commit=git_commit
        )
        
        if len(candidates_df) == 0:
            logger.warning("No routing candidates found. Feature selection may not have produced metrics.")
            return None
        
        logger.info(f"✅ Found {len(candidates_df)} routing candidates")
        
        # Save routing candidates
        metrics_dir = output_dir / "METRICS"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = aggregator.save_routing_candidates(
            candidates_df,
            output_path=metrics_dir / "routing_candidates.parquet"
        )
        
        # Step 2: Generate routing plan
        logger.info("Step 2: Generating routing decisions...")
        router = TrainingRouter(routing_config)
        
        plan_output = metrics_dir / "routing_plan"
        plan = router.generate_routing_plan(
            routing_candidates=candidates_df,
            output_dir=plan_output,
            git_commit=git_commit,
            config_hash=config_hash
        )
        
        # Step 3: Log summary
        logger.info("Step 3: Routing plan summary:")
        
        total_symbols = 0
        route_counts = {}
        
        for target, target_data in plan["targets"].items():
            cs_info = target_data["cross_sectional"]
            symbols_data = target_data.get("symbols", {})
            total_symbols += len(symbols_data)
            
            logger.info(f"  [{target}]")
            logger.info(f"    CS: {cs_info['route']} ({cs_info['state']})")
            
            for symbol, sym_data in symbols_data.items():
                route = sym_data['route']
                route_counts[route] = route_counts.get(route, 0) + 1
                
                # Log decision for each symbol (if log_decision_reasons is enabled)
                if routing_config.get("routing", {}).get("log_decision_reasons", False):
                    logger.info(f"      {symbol}: {route} (CS={sym_data['cs_state']}, LOCAL={sym_data['local_state']})")
        
        logger.info("\nRoute distribution:")
        for route, count in sorted(route_counts.items(), key=lambda x: -x[1]):
            logger.info(f"  {route}: {count} symbols")
        
        logger.info(f"\n✅ Routing plan generated: {plan_output}")
        logger.info(f"   Total targets: {len(plan['targets'])}")
        logger.info(f"   Total symbol decisions: {total_symbols}")
        
        # Step 4: Generate training plan (if requested)
        if generate_training_plan:
            try:
                from TRAINING.orchestration.training_plan_generator import TrainingPlanGenerator
                logger.info("\nStep 4: Generating training plan...")
                
                # Validate plan before generating training plan
                if not isinstance(plan, dict):
                    logger.warning(f"Routing plan is not a dict, got {type(plan)}, skipping training plan generation")
                else:
                    training_plan_dir = metrics_dir / "training_plan"
                    
                    # Validate model_families
                    if model_families is not None and not isinstance(model_families, list):
                        logger.warning(f"model_families is not a list, got {type(model_families)}, using default")
                        model_families = None
                    
                    try:
                        generator = TrainingPlanGenerator(
                            routing_plan=plan,
                            model_families=model_families
                        )
                    except Exception as e:
                        logger.error(f"Failed to create TrainingPlanGenerator: {e}", exc_info=True)
                        raise
                    
                    try:
                        training_plan = generator.generate_training_plan(
                            output_dir=training_plan_dir,
                            include_blocked=False,
                            git_commit=git_commit,
                            config_hash=config_hash,
                            metrics_snapshot=str(metrics_path.name) if metrics_path else None
                        )
                        
                        # Validate training_plan structure before accessing
                        if isinstance(training_plan, dict):
                            metadata = training_plan.get("metadata", {})
                            summary = training_plan.get("summary", {})
                            
                            logger.info(f"✅ Training plan generated: {training_plan_dir}")
                            logger.info(f"   Master plan: master_training_plan.json (single source of truth)")
                            
                            total_jobs = metadata.get("total_jobs", 0)
                            logger.info(f"   Total jobs: {total_jobs}")
                            
                            cs_jobs = summary.get("total_cs_jobs", 0)
                            symbol_jobs = summary.get("total_symbol_jobs", 0)
                            logger.info(f"   CS jobs: {cs_jobs}")
                            logger.info(f"   Symbol jobs: {symbol_jobs}")
                        else:
                            logger.warning(f"Training plan is not a dict, got {type(training_plan)}")
                    except ValueError as e:
                        # Critical error - re-raise
                        logger.error(f"Failed to generate training plan (critical): {e}", exc_info=True)
                        raise
                    except Exception as e:
                        logger.warning(f"Failed to generate training plan (non-critical): {e}", exc_info=True)
            except ImportError as e:
                logger.warning(f"Failed to import TrainingPlanGenerator: {e}, skipping training plan generation")
            except Exception as e:
                logger.warning(f"Failed to generate training plan (non-critical): {e}", exc_info=True)
        
        logger.info("="*80)
        
        return plan
        
    except Exception as e:
        logger.error(f"Failed to generate routing plan: {e}", exc_info=True)
        return None


def load_routing_plan(routing_plan_dir: Path) -> Optional[Dict[str, Any]]:
    """
    Load routing plan from disk.
    
    Args:
        routing_plan_dir: Directory containing routing_plan.json
    
    Returns:
        Routing plan dict or None if not found
    """
    json_path = routing_plan_dir / "routing_plan.json"
    if not json_path.exists():
        return None
    
    try:
        import json
        with open(json_path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load routing plan from {json_path}: {e}")
        return None


def get_route_for_target_symbol(
    routing_plan: Dict[str, Any],
    target: str,
    symbol: str
) -> Optional[str]:
    """
    Get route decision for a (target, symbol) pair.
    
    Args:
        routing_plan: Routing plan dict
        target: Target name
        symbol: Symbol name
    
    Returns:
        Route state string (e.g., "ROUTE_CROSS_SECTIONAL") or None if not found
    """
    targets = routing_plan.get("targets", {})
    if target not in targets:
        return None
    
    target_data = targets[target]
    symbols = target_data.get("symbols", {})
    
    if symbol not in symbols:
        return None
    
    return symbols[symbol].get("route")


def should_train_cross_sectional(
    routing_plan: Dict[str, Any],
    target: str
) -> bool:
    """
    Check if cross-sectional training should be enabled for a target.
    
    Args:
        routing_plan: Routing plan dict
        target: Target name
    
    Returns:
        True if CS training should be enabled
    """
    targets = routing_plan.get("targets", {})
    if target not in targets:
        return False
    
    target_data = targets[target]
    cs_info = target_data.get("cross_sectional", {})
    return cs_info.get("route") == "ENABLED"


def should_train_symbol_specific(
    routing_plan: Dict[str, Any],
    target: str,
    symbol: str
) -> bool:
    """
    Check if symbol-specific training should be enabled for a (target, symbol) pair.
    
    Args:
        routing_plan: Routing plan dict
        target: Target name
        symbol: Symbol name
    
    Returns:
        True if symbol-specific training should be enabled
    """
    route = get_route_for_target_symbol(routing_plan, target, symbol)
    if route is None:
        return False
    
    # Symbol-specific training is enabled if route is SYMBOL_SPECIFIC, BOTH, or EXPERIMENTAL_ONLY
    return route in [
        "ROUTE_SYMBOL_SPECIFIC",
        "ROUTE_BOTH",
        "ROUTE_EXPERIMENTAL_ONLY"
    ]
