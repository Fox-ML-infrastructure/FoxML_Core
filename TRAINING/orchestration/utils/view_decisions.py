"""
View Aggregated Decisions

Utility to view and aggregate all decision files from target-first structure.
Provides a unified view of routing, target prioritization, and feature prioritization decisions.

Includes RoutingDecisions class for schema v2 support with get_final_route() accessor.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# RoutingDecisions: Schema-aware wrapper with single authoritative accessor
# =============================================================================

class RoutingDecisions:
    """
    Wrapper for routing_decisions.json with single authoritative accessor.
    
    Supports both schema v1 (legacy) and v2 (with final_routes).
    
    Schema v1:
        {
            "routing_decisions": {
                "fwd_ret_1d": {"route": "CROSS_SECTIONAL", ...},
                ...
            }
        }
    
    Schema v2:
        {
            "schema_version": 2,
            "metadata": {
                "generated_at": "...",
                "universe_sig": "...",
                "evaluation_modes_considered": ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]
            },
            "final_routes": {
                "fwd_ret_1d": "CROSS_SECTIONAL",
                ...
            },
            "final_route_summary": {"CROSS_SECTIONAL": 5, "SYMBOL_SPECIFIC": 2},
            "routing_decisions": {...}  # Kept for backward compat
        }
    
    Usage:
        decisions = RoutingDecisions.load(path, strict=True)
        route = decisions.get_final_route("fwd_ret_1d")  # Always reads from final_routes
    """
    
    def __init__(self, data: Dict[str, Any]):
        """Initialize with raw data dict."""
        self._data = data
        self._schema_version = data.get("schema_version", 1)
        
        # Ensure final_routes exists (v1 shim)
        if "final_routes" not in data:
            data["final_routes"] = {}
            for target, target_data in data.get("routing_decisions", {}).items():
                if isinstance(target_data, dict):
                    data["final_routes"][target] = target_data.get("route", "CROSS_SECTIONAL")
                else:
                    data["final_routes"][target] = "CROSS_SECTIONAL"
        
        self._final_routes = data["final_routes"]
        self._routing_decisions = data.get("routing_decisions", {})
        self._metadata = data.get("metadata", {})
    
    @property
    def schema_version(self) -> int:
        """Get schema version."""
        return self._schema_version
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata dict."""
        return self._metadata
    
    @property
    def universe_sig(self) -> Optional[str]:
        """Get universe signature from metadata."""
        return self._metadata.get("universe_sig")
    
    @property
    def evaluation_modes_considered(self) -> List[str]:
        """Get list of evaluation modes that were considered."""
        return self._metadata.get("evaluation_modes_considered", [])
    
    def get_final_route(self, target: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get authoritative final route for target.
        
        ONLY reads from final_routes. Never reads routing_decisions[target].route directly.
        This ensures single source of truth and prevents dual-source drift.
        
        Args:
            target: Target name
            default: Default value if target not found
        
        Returns:
            Route string ("CROSS_SECTIONAL" or "SYMBOL_SPECIFIC") or default
        """
        return self._final_routes.get(target, default)
    
    def get_all_routes(self) -> Dict[str, str]:
        """Get all final routes as a dict."""
        return dict(self._final_routes)
    
    def get_target_details(self, target: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed routing information for a target (for backward compat).
        
        Returns data from routing_decisions[target], not final_routes.
        """
        return self._routing_decisions.get(target)
    
    def __iter__(self):
        """Iterate over target names."""
        return iter(self._final_routes)
    
    def __len__(self):
        """Get number of targets with routing decisions."""
        return len(self._final_routes)
    
    def __contains__(self, target: str) -> bool:
        """Check if target has a routing decision."""
        return target in self._final_routes
    
    @classmethod
    def load(cls, path: Path, strict: bool = False) -> "RoutingDecisions":
        """
        Load routing decisions from file.
        
        Args:
            path: Path to routing_decisions.json
            strict: If True, check for dual-source drift (v2 only)
        
        Returns:
            RoutingDecisions instance
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If strict mode and dual-source drift detected
        """
        with open(path) as f:
            data = json.load(f)
        
        schema_version = data.get("schema_version", 1)
        
        if schema_version == 2 and strict:
            # Check for dual-source drift
            final_routes = data.get("final_routes", {})
            routing_decisions = data.get("routing_decisions", {})
            
            for target, target_data in routing_decisions.items():
                if isinstance(target_data, dict):
                    targets_route = target_data.get("route")
                    final_route = final_routes.get(target)
                    if targets_route is not None and final_route is not None and targets_route != final_route:
                        raise ValueError(
                            f"Dual-source drift: routing_decisions[{target}].route={targets_route} "
                            f"disagrees with final_routes[{target}]={final_route}. "
                            f"This indicates a bug in routing schema generation."
                        )
        
        return cls(data)
    
    @classmethod
    def from_output_dir(cls, output_dir: Path, strict: bool = False) -> Optional["RoutingDecisions"]:
        """
        Load routing decisions from output directory.
        
        Tries globals/routing_decisions.json first.
        
        Args:
            output_dir: Base run output directory
            strict: If True, check for dual-source drift
        
        Returns:
            RoutingDecisions instance or None if not found
        """
        output_dir = Path(output_dir)
        globals_file = output_dir / "globals" / "routing_decisions.json"
        
        if globals_file.exists():
            try:
                return cls.load(globals_file, strict=strict)
            except Exception as e:
                logger.warning(f"Failed to load routing decisions: {e}")
                return None
        
        logger.debug(f"No routing decisions found at {globals_file}")
        return None


def load_all_routing_decisions(output_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load all routing decisions from target-first structure.
    
    Aggregates per-target routing decisions from targets/<target>/decision/routing_decision.json
    and falls back to globals/routing_decisions.json if per-target files don't exist.
    
    Args:
        output_dir: Base run output directory
    
    Returns:
        Dict mapping target_name -> routing decision dict
    """
    output_dir = Path(output_dir)
    routing_decisions = {}
    
    # Try globals first (global summary)
    globals_file = output_dir / "globals" / "routing_decisions.json"
    if globals_file.exists():
        try:
            with open(globals_file, 'r') as f:
                data = json.load(f)
            routing_decisions = data.get('routing_decisions', {})
            logger.info(f"Loaded {len(routing_decisions)} routing decisions from globals")
        except Exception as e:
            logger.debug(f"Failed to load from globals: {e}")
    
    # Also check per-target decisions (may have more detail)
    targets_dir = output_dir / "targets"
    if targets_dir.exists():
        for target_dir in targets_dir.iterdir():
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            target_decision_file = target_dir / "decision" / "routing_decision.json"
            if target_decision_file.exists():
                try:
                    with open(target_decision_file, 'r') as f:
                        target_data = json.load(f)
                    if target in target_data:
                        # Per-target decision may have more detail, merge it
                        routing_decisions[target] = {**routing_decisions.get(target, {}), **target_data[target]}
                except Exception as e:
                    logger.debug(f"Failed to load per-target routing decision for {target}: {e}")
    
    return routing_decisions


def load_all_target_prioritizations(output_dir: Path) -> Dict[str, Any]:
    """
    Load all target prioritizations from target-first structure.
    
    Args:
        output_dir: Base run output directory
    
    Returns:
        Dict with global ranking and per-target prioritizations
    """
    output_dir = Path(output_dir)
    result = {
        'global_ranking': None,
        'per_target': {}
    }
    
    # Load global ranking
    globals_file = output_dir / "globals" / "target_prioritization.yaml"
    if globals_file.exists():
        try:
            with open(globals_file, 'r') as f:
                result['global_ranking'] = yaml.safe_load(f)
        except Exception as e:
            logger.debug(f"Failed to load global target prioritization: {e}")
    
    # Load per-target prioritizations
    targets_dir = output_dir / "targets"
    if targets_dir.exists():
        for target_dir in targets_dir.iterdir():
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            target_file = target_dir / "decision" / "target_prioritization.yaml"
            if target_file.exists():
                try:
                    with open(target_file, 'r') as f:
                        result['per_target'][target] = yaml.safe_load(f)
                except Exception as e:
                    logger.debug(f"Failed to load per-target prioritization for {target}: {e}")
    
    return result


def load_all_feature_prioritizations(output_dir: Path) -> Dict[str, Any]:
    """
    Load all feature prioritizations from target-first structure.
    
    Args:
        output_dir: Base run output directory
    
    Returns:
        Dict mapping target_name -> feature prioritization data
    """
    output_dir = Path(output_dir)
    feature_prioritizations = {}
    
    targets_dir = output_dir / "targets"
    if targets_dir.exists():
        for target_dir in targets_dir.iterdir():
            if not target_dir.is_dir():
                continue
            target = target_dir.name
            feature_file = target_dir / "decision" / "feature_prioritization.yaml"
            if feature_file.exists():
                try:
                    with open(feature_file, 'r') as f:
                        feature_prioritizations[target] = yaml.safe_load(f)
                except Exception as e:
                    logger.debug(f"Failed to load feature prioritization for {target}: {e}")
    
    return feature_prioritizations


def view_aggregated_decisions(
    output_dir: Path,
    format: str = "summary"  # "summary", "detailed", "json"
) -> Dict[str, Any]:
    """
    View all aggregated decisions from a run.
    
    Args:
        output_dir: Base run output directory
        format: Output format ("summary", "detailed", or "json")
    
    Returns:
        Dict with all decision data
    """
    output_dir = Path(output_dir)
    
    result = {
        'run_id': output_dir.name,
        'routing_decisions': load_all_routing_decisions(output_dir),
        'target_prioritizations': load_all_target_prioritizations(output_dir),
        'feature_prioritizations': load_all_feature_prioritizations(output_dir)
    }
    
    if format == "summary":
        # Print human-readable summary
        print(f"\n{'='*80}")
        print(f"AGGREGATED DECISIONS FOR RUN: {output_dir.name}")
        print(f"{'='*80}\n")
        
        # Routing decisions summary
        routing = result['routing_decisions']
        if routing:
            print(f"ROUTING DECISIONS: {len(routing)} targets")
            route_counts = {}
            for target, decision in routing.items():
                route = decision.get('route', 'UNKNOWN')
                route_counts[route] = route_counts.get(route, 0) + 1
            for route, count in sorted(route_counts.items()):
                print(f"  {route}: {count} targets")
            print()
        
        # Target prioritization summary
        target_prior = result['target_prioritizations']
        if target_prior.get('global_ranking'):
            rankings = target_prior['global_ranking'].get('target_rankings', [])
            if rankings:
                print(f"TARGET PRIORITIZATION: Top 5 targets")
                for i, rank in enumerate(rankings[:5]):
                    print(f"  {rank.get('rank', i+1)}. {rank.get('target', 'unknown')}: "
                          f"score={rank.get('composite_score', 0):.3f}, "
                          f"recommendation={rank.get('recommendation', 'N/A')}")
                print()
        
        # Feature prioritization summary
        feature_prior = result['feature_prioritizations']
        if feature_prior:
            print(f"FEATURE PRIORITIZATIONS: {len(feature_prior)} targets")
            for target, data in list(feature_prior.items())[:5]:
                summary = data.get('summary', {})
                print(f"  {target}: {summary.get('selected_features', 0)} features selected "
                      f"out of {summary.get('total_features', 0)} total")
            if len(feature_prior) > 5:
                print(f"  ... and {len(feature_prior) - 5} more targets")
            print()
    
    elif format == "detailed":
        # Print detailed view
        view_aggregated_decisions(output_dir, format="summary")
        print(f"\nDETAILED VIEW:")
        print(json.dumps(result, indent=2, default=str))
    
    return result


def export_decisions_to_csv(output_dir: Path, output_file: Optional[Path] = None) -> Path:
    """
    Export all decisions to a CSV file for easy analysis.
    
    Args:
        output_dir: Base run output directory
        output_file: Optional output file path (defaults to globals/decisions_summary.csv)
    
    Returns:
        Path to exported CSV file
    """
    output_dir = Path(output_dir)
    if output_file is None:
        output_file = output_dir / "globals" / "decisions_summary.csv"
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect all decision data
    routing_decisions = load_all_routing_decisions(output_dir)
    target_prioritizations = load_all_target_prioritizations(output_dir)
    feature_prioritizations = load_all_feature_prioritizations(output_dir)
    
    # Build DataFrame
    rows = []
    for target in set(list(routing_decisions.keys()) + 
                     list(target_prioritizations.get('per_target', {}).keys()) +
                     list(feature_prioritizations.keys())):
        row = {'target': target}
        
        # Routing decision
        if target in routing_decisions:
            route_info = routing_decisions[target]
            row['route'] = route_info.get('route', 'UNKNOWN')
            row['cs_auc'] = route_info.get('cross_sectional', {}).get('auc_mean', None)
            row['symbol_auc_max'] = route_info.get('symbol_specific', {}).get('max_auc', None)
        
        # Target prioritization
        if target in target_prioritizations.get('per_target', {}):
            target_prior = target_prioritizations['per_target'][target]
            row['target_rank'] = target_prior.get('rank', None)
            row['target_composite_score'] = target_prior.get('composite_score', None)
            row['target_recommendation'] = target_prior.get('recommendation', None)
        
        # Global ranking
        global_ranking = target_prioritizations.get('global_ranking', {}).get('target_rankings', [])
        for rank_entry in global_ranking:
            if rank_entry.get('target') == target:
                row['global_rank'] = rank_entry.get('rank', None)
                row['global_composite_score'] = rank_entry.get('composite_score', None)
                break
        
        # Feature prioritization
        if target in feature_prioritizations:
            feat_prior = feature_prioritizations[target]
            summary = feat_prior.get('summary', {})
            row['n_features_selected'] = summary.get('selected_features', None)
            row['n_features_total'] = summary.get('total_features', None)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values('target')
    df.to_csv(output_file, index=False)
    
    logger.info(f"Exported decisions summary to {output_file}")
    return output_file

