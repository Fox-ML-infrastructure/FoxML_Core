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
Generate Routing Plan

Main entry point for generating training routing plans from metrics.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional
import yaml

from CONFIG.config_loader import CONFIG_DIR
from TRAINING.orchestration.metrics_aggregator import MetricsAggregator
from TRAINING.orchestration.training_router import TrainingRouter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def compute_config_hash(config_path: Path) -> str:
    """Compute hash of config file."""
    try:
        with open(config_path, "rb") as f:
            content = f.read()
        import hashlib
        return hashlib.sha256(content).hexdigest()[:8]
    except Exception:
        return "unknown"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate training routing plan from metrics"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Base output directory (e.g., feature_selections/)"
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="List of target names"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="List of symbol names"
    )
    parser.add_argument(
        "--routing-config",
        type=Path,
        default=None,
        help="Path to routing config YAML (default: CONFIG/training_config/routing_config.yaml)"
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Output path for routing_candidates (default: METRICS/routing_candidates.parquet)"
    )
    parser.add_argument(
        "--plan-output",
        type=Path,
        default=None,
        help="Output directory for routing plan (default: METRICS/routing_plan/)"
    )
    
    args = parser.parse_args()
    
    # Load routing config
    if args.routing_config is None:
        routing_config_path = CONFIG_DIR / "training_config" / "routing_config.yaml"
    else:
        routing_config_path = args.routing_config
    
    if not routing_config_path.exists():
        logger.error(f"Routing config not found: {routing_config_path}")
        sys.exit(1)
    
    with open(routing_config_path) as f:
        routing_config = yaml.safe_load(f)
    
    config_hash = compute_config_hash(routing_config_path)
    git_commit = get_git_commit()
    
    logger.info(f"📊 Generating routing plan for {len(args.targets)} targets, {len(args.symbols)} symbols")
    logger.info(f"   Config: {routing_config_path} (hash: {config_hash})")
    logger.info(f"   Git commit: {git_commit}")
    
    # Step 1: Aggregate metrics
    logger.info("Step 1: Aggregating metrics...")
    aggregator = MetricsAggregator(args.output_dir)
    candidates_df = aggregator.aggregate_routing_candidates(
        targets=args.targets,
        symbols=args.symbols,
        git_commit=git_commit
    )
    
    if len(candidates_df) == 0:
        logger.error("No routing candidates found. Check that feature selection has run.")
        sys.exit(1)
    
    logger.info(f"✅ Found {len(candidates_df)} routing candidates")
    
    # Save routing candidates
    metrics_path = aggregator.save_routing_candidates(
        candidates_df,
        output_path=args.metrics_output
    )
    
    # Step 2: Generate routing plan
    logger.info("Step 2: Generating routing plan...")
    router = TrainingRouter(routing_config)
    
    if args.plan_output is None:
        plan_output = metrics_path.parent / "routing_plan"
    else:
        plan_output = args.plan_output
    
    plan = router.generate_routing_plan(
        routing_candidates=candidates_df,
        output_dir=plan_output,
        git_commit=git_commit,
        config_hash=config_hash
    )
    
    # Step 3: Print summary
    logger.info("Step 3: Routing plan summary:")
    
    total_symbols = 0
    route_counts = {}
    
    for target, target_data in plan["targets"].items():
        cs_info = target_data["cross_sectional"]
        symbols = target_data.get("symbols", {})
        total_symbols += len(symbols)
        
        logger.info(f"  {target}:")
        logger.info(f"    CS: {cs_info['route']} ({cs_info['state']})")
        
        for symbol, sym_data in symbols.items():
            route = sym_data['route']
            route_counts[route] = route_counts.get(route, 0) + 1
    
    logger.info("\nRoute distribution:")
    for route, count in sorted(route_counts.items()):
        logger.info(f"  {route}: {count} symbols")
    
    logger.info(f"\n✅ Routing plan generated: {plan_output}")
    logger.info(f"   Total targets: {len(plan['targets'])}")
    logger.info(f"   Total symbol decisions: {total_symbols}")


if __name__ == "__main__":
    main()
