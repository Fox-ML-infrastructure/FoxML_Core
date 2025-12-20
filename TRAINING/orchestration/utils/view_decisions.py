"""
View Aggregated Decisions

Utility to view and aggregate all decision files from target-first structure.
Provides a unified view of routing, target prioritization, and feature prioritization decisions.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


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

