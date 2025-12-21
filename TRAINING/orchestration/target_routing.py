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
Target Routing Helper

Maps target confidence metrics to operational decisions (production/candidate/experimental).
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def classify_target_from_confidence(
    conf: Dict[str, Any],
    routing_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Classify target based on confidence metrics into operational buckets.
    
    Args:
        conf: Target confidence dict from compute_target_confidence()
        routing_config: Optional routing rules from config (confidence.routing section)
    
    Returns:
        Dict with:
            - allowed_in_production: bool
            - bucket: "core" | "candidate" | "experimental"
            - note: str explanation
    """
    conf_level = conf.get("confidence", "LOW")
    reason = conf.get("low_confidence_reason")
    score_tier = conf.get("score_tier", "LOW")
    
    # Use config if provided, otherwise use defaults
    if routing_config is None:
        routing_config = {}
    
    # Check experimental rule (Boruta zero confirmed)
    experimental_rule = routing_config.get('experimental', {})
    if (conf_level == experimental_rule.get('confidence', 'LOW') and
        reason == experimental_rule.get('low_confidence_reason', 'boruta_zero_confirmed')):
        return {
            "allowed_in_production": False,
            "bucket": "experimental",
            "note": experimental_rule.get('note', 'Boruta used and found zero robust features; fragile signal.')
        }
    
    # Check core rule (HIGH confidence)
    core_rule = routing_config.get('core', {})
    if conf_level == core_rule.get('confidence', 'HIGH'):
        return {
            "allowed_in_production": True,
            "bucket": "core",
            "note": core_rule.get('note', 'Strong, robust signal with good agreement and Boruta support.')
        }
    
    # Check candidate rule (MEDIUM confidence with score_tier requirement)
    candidate_rule = routing_config.get('candidate', {})
    if conf_level == candidate_rule.get('confidence', 'MEDIUM'):
        score_tier_min = candidate_rule.get('score_tier_min', 'MEDIUM')
        score_tier_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        if score_tier_order.get(score_tier, 0) >= score_tier_order.get(score_tier_min, 0):
            return {
                "allowed_in_production": False,
                "bucket": "candidate",
                "note": candidate_rule.get('note', f'Some signal present (score_tier={score_tier}) but not fully robust yet.')
            }
    
    # Default fallback
    default_rule = routing_config.get('default', {})
    if conf_level == "MEDIUM":
        bucket = default_rule.get('bucket', 'candidate')
    else:
        bucket = default_rule.get('fallback_bucket', 'experimental')
    
    return {
        "allowed_in_production": False,
        "bucket": bucket,
        "note": default_rule.get('note', f'Signal strength: {score_tier}, robustness: {conf_level}. Needs validation.')
    }


def load_target_confidence(output_dir: Path, target_name: str) -> Optional[Dict[str, Any]]:
    """
    Load target confidence JSON for a specific target.
    
    Args:
        output_dir: Target output directory or base run directory
        target_name: Target column name
    
    Returns:
        Confidence dict or None if not found
    """
    # Try target-first structure first
    from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
    base_dir = output_dir
    # Walk up to find run root if needed
    # Only stop if we find a run directory (has targets/, globals/, or cache/)
    # Don't stop at RESULTS/ - continue to find actual run directory
    while base_dir.parent.exists():
        if (base_dir / "targets").exists() or (base_dir / "globals").exists() or (base_dir / "cache").exists():
            break
        base_dir = base_dir.parent
    
    target_name_clean = target_name.replace('/', '_').replace('\\', '_')
    target_repro_dir = get_target_reproducibility_dir(base_dir, target_name_clean)
    conf_path = target_repro_dir / "target_confidence.json"
    
    if conf_path.exists():
        try:
            with open(conf_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load confidence from target-first structure: {e}")
    
    # Fallback to legacy location
    legacy_path = output_dir / "target_confidence.json"
    if legacy_path.exists():
        try:
            with open(legacy_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load confidence from legacy location: {e}")
    
    return None


def collect_run_level_confidence_summary(
    feature_selections_dir: Path,
    output_dir: Path,
    routing_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Collect all target confidence files from a run and create run-level summary.
    
    Args:
        feature_selections_dir: Directory containing per-target feature selection outputs
        output_dir: Where to write the run-level summary
    
    Returns:
        List of all target confidence dicts
    """
    all_confidence = []
    
    # Find all target_confidence.json files
    for conf_file in feature_selections_dir.rglob("target_confidence.json"):
        try:
            with open(conf_file) as f:
                conf = json.load(f)
                all_confidence.append(conf)
        except Exception as e:
            logger.warning(f"Failed to load {conf_file}: {e}")
            continue
    
    if not all_confidence:
        logger.warning("No target confidence files found")
        return []
    
    # Save to target-first structure (globals/)
    from TRAINING.orchestration.utils.target_first_paths import get_globals_dir
    globals_dir = get_globals_dir(output_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    
    # Write run-level JSON (list of all targets) to globals/
    run_summary_path = globals_dir / "target_confidence_summary.json"
    with open(run_summary_path, "w") as f:
        json.dump(all_confidence, f, indent=2)
    logger.info(f"✅ Saved run-level confidence summary: {len(all_confidence)} targets to {run_summary_path}")
    
    # Write CSV summary for easy inspection to globals/ (target-first only)
    csv_path = globals_dir / "target_confidence_summary.csv"
    summary_rows = []
    for conf in all_confidence:
        routing = classify_target_from_confidence(conf, routing_config=routing_config)
        summary_rows.append({
            'target_name': conf.get('target_name', 'unknown'),
            'confidence': conf.get('confidence', 'LOW'),
            'score_tier': conf.get('score_tier', 'LOW'),
            'low_confidence_reason': conf.get('low_confidence_reason', ''),
            'mean_score': conf.get('mean_score', 0.0),
            'max_score': conf.get('max_score', 0.0),
            'mean_strong_score': conf.get('mean_strong_score', 0.0),
            'agreement_ratio': conf.get('agreement_ratio', 0.0),
            'model_coverage_ratio': conf.get('model_coverage_ratio', 0.0),
            'boruta_confirmed_count': conf.get('boruta_confirmed_count', 0),
            'boruta_tentative_count': conf.get('boruta_tentative_count', 0),
            'bucket': routing.get('bucket', 'experimental'),
            'allowed_in_production': routing.get('allowed_in_production', False),
            'note': routing.get('note', '')
        })
    
    df = pd.DataFrame(summary_rows)
    df = df.sort_values(['confidence', 'score_tier'], ascending=[False, False])
    df.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved confidence summary CSV: {csv_path}")
    
    return all_confidence


def save_target_routing_metadata(
    output_dir: Path,
    target_name: str,
    conf: Dict[str, Any],
    routing: Dict[str, Any],
    view: Optional[str] = None
) -> None:
    """
    Save routing decision metadata alongside confidence metrics.
    
    Structure (target-first):
    - Per-target: targets/<target>/decision/routing_decision.json (detailed record)
    - Global summary: globals/feature_selection_routing.json (lightweight, references per-target files)
    
    NOTE: This is separate from globals/routing_decisions.json (which is for target ranking routing only)
    
    Args:
        output_dir: Base run output directory or target-specific directory
        target_name: Target column name
        conf: Confidence metrics
        routing: Routing decision from classify_target_from_confidence()
        view: View type (CROSS_SECTIONAL, SYMBOL_SPECIFIC) - if None, defaults to CROSS_SECTIONAL
    """
    from TRAINING.orchestration.utils.target_first_paths import (
        get_target_decision_dir, get_globals_dir, ensure_target_structure
    )
    
    # Find base run directory
    base_dir = output_dir
    # Only stop if we find a run directory (has targets/, globals/, or cache/)
    # Don't stop at RESULTS/ - continue to find actual run directory
    while base_dir.parent.exists():
        if (base_dir / "targets").exists() or (base_dir / "globals").exists() or (base_dir / "cache").exists():
            break
        base_dir = base_dir.parent
    
    target_name_clean = target_name.replace('/', '_').replace('\\', '_')
    ensure_target_structure(base_dir, target_name_clean)
    decision_dir = get_target_decision_dir(base_dir, target_name_clean)
    decision_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize view (default to CROSS_SECTIONAL if not provided)
    view_normalized = (view or "CROSS_SECTIONAL").upper()
    if view_normalized not in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
        view_normalized = "CROSS_SECTIONAL"
    
    # Save per-target decision (detailed record with full confidence and routing info)
    routing_path = decision_dir / "routing_decision.json"
    routing_data = {
        target_name: {
            'target_name': target_name,
            'view': view_normalized,  # Add view information for completeness
            'confidence': conf,
            'routing': routing,
            # Reference to where selected features can be found
            'selected_features_path': f"targets/{target_name_clean}/reproducibility/selected_features.txt",
            'feature_selection_summary_path': f"targets/{target_name_clean}/reproducibility/feature_selection_summary.json"
        }
    }
    
    with open(routing_path, "w") as f:
        json.dump(routing_data, f, indent=2)
    
    logger.debug(f"Saved per-target routing decision for {target_name} to {routing_path}")
    
    # CRITICAL: Update lightweight summary in globals/feature_selection_routing.json
    # This is separate from globals/routing_decisions.json (which is for target ranking)
    globals_dir = get_globals_dir(base_dir)
    globals_dir.mkdir(parents=True, exist_ok=True)
    feature_routing_file = globals_dir / "feature_selection_routing.json"
    
    # Load existing feature selection routing (merge, don't overwrite)
    existing_routing = {}
    if feature_routing_file.exists():
        try:
            with open(feature_routing_file) as f:
                data = json.load(f)
                existing_routing = data.get('routing_decisions', {})
        except Exception as e:
            logger.warning(f"Failed to load existing feature selection routing: {e}")
    
    # Normalize view (default to CROSS_SECTIONAL if not provided)
    view_normalized = (view or "CROSS_SECTIONAL").upper()
    if view_normalized not in ["CROSS_SECTIONAL", "SYMBOL_SPECIFIC"]:
        view_normalized = "CROSS_SECTIONAL"
    
    # Create key with view: target_name:view (e.g., "fwd_ret_5d:CROSS_SECTIONAL")
    routing_key = f"{target_name}:{view_normalized}"
    
    # Update with this target's routing decision (lightweight - just key info)
    existing_routing[routing_key] = {
        'confidence': conf.get('confidence', 'LOW'),
        'score_tier': conf.get('score_tier', 'LOW'),
        'bucket': routing.get('bucket', 'experimental'),
        'allowed_in_production': routing.get('allowed_in_production', False),
        'view': view_normalized,  # Add view information
        # Reference to per-target file for full details
        'details_path': f"targets/{target_name_clean}/decision/routing_decision.json"
    }
    
    # Save updated feature selection routing summary
    routing_data_globals = {
        'routing_decisions': existing_routing,
        'summary': {
            'total_targets': len(existing_routing),
            'high_confidence': sum(1 for r in existing_routing.values() if r.get('confidence') == 'HIGH'),
            'medium_confidence': sum(1 for r in existing_routing.values() if r.get('confidence') == 'MEDIUM'),
            'low_confidence': sum(1 for r in existing_routing.values() if r.get('confidence') == 'LOW'),
            'production_allowed': sum(1 for r in existing_routing.values() if r.get('allowed_in_production', False))
        }
    }
    
    with open(feature_routing_file, "w") as f:
        json.dump(routing_data_globals, f, indent=2, default=str)
    
    logger.info(f"Updated feature selection routing summary in {feature_routing_file}")

