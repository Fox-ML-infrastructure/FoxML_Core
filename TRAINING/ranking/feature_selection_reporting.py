"""
Feature Selection Reporting

Saves feature selection results in the same structure as target ranking:
- CSV and YAML summary files
- REPRODUCIBILITY/FEATURE_SELECTION/ structure with CROSS_SECTIONAL/SYMBOL_SPECIFIC views
- Feature importances
- Stability snapshots
- Cross-sectional ranking results

File Naming Conventions (to distinguish from target ranking):
- Feature selection files:
  * feature_prioritization.yaml (ranks features for a target)
  * feature_selection_rankings.csv (feature rankings for a target)
  * selected_features.txt (list of selected features)
- Target ranking files (for reference, not saved here):
  * target_prioritization.yaml (ranks targets themselves)
  * target_predictability_rankings.csv (target rankings)
  * routing_decision.json (routing decisions for targets)
- Shared files (no conflicts):
  * target_confidence.json (feature selection writes, target ranking may read)

ROUTING NOTE: Feature selection artifacts are SCOPE-LEVEL SUMMARIES, not cohort artifacts.
They use OutputLayout.repro_dir() / feature_importance_dir() directly.

Do NOT use _save_to_cohort for these outputs. The cohort firewall is for
per-cohort metrics/run data under cohort=cs_* or cohort=sy_* directories.

If you later add per-model-family or per-fold outputs that need cohort scoping,
those SHOULD go through _save_to_cohort.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import yaml
import json

logger = logging.getLogger(__name__)


def save_feature_selection_rankings(
    summary_df: pd.DataFrame,
    selected_features: List[str],
    target_column: str,
    output_dir: Path,
    view: str,  # REQUIRED (no default)
    symbol: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    universe_sig: Optional[str] = None,  # Phase A: optional for backward compat
):
    """
    Save feature selection rankings. view parameter is REQUIRED.
    
    Structure:
    RESULTS/{run_id}/
      targets/{target}/
        reproducibility/
          {view}/
            feature_selection_rankings.csv
            selected_features.txt
            feature_importances/
            ...
    
    Args:
        summary_df: DataFrame with feature rankings and scores
        selected_features: List of selected feature names
        target_column: Target column name
        output_dir: Base output directory
        view: REQUIRED - "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Required if view="SYMBOL_SPECIFIC"
        metadata: Optional metadata dict
    
    Raises:
        ValueError: If view is None, invalid, or SYMBOL_SPECIFIC without symbol
    """
    if view not in ("CROSS_SECTIONAL", "SYMBOL_SPECIFIC"):
        raise ValueError(f"Invalid view: {view}. Must be 'CROSS_SECTIONAL' or 'SYMBOL_SPECIFIC'")
    if view == "SYMBOL_SPECIFIC" and symbol is None:
        raise ValueError("symbol required when view='SYMBOL_SPECIFIC'")
    
    # Clean target name for filesystem
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    
    # Phase A: Use OutputLayout if universe_sig provided (new canonical path)
    # Otherwise fall back to legacy path resolution with warning
    use_output_layout = bool(universe_sig)
    if not use_output_layout:
        logger.warning(
            f"universe_sig not provided for {target_column} feature selection rankings, "
            f"falling back to legacy path resolution. Pass universe_sig for canonical paths."
        )
    
    # Determine target-level directory (matching TARGET_RANKING structure)
    # output_dir is already at: REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/
    # Use it directly to avoid nested structures
    
    # Find base run directory for target-first structure
    # Walk up from output_dir to find the run directory (where "targets", "globals", or "cache" would be)
    # REMOVED: Legacy REPRODUCIBILITY/FEATURE_SELECTION path construction - only use target-first structure
    base_output_dir = None
    
    # Find run directory from output_dir
    if output_dir and output_dir.exists():
        temp_dir = Path(output_dir)
        for _ in range(10):  # Limit depth
            # Only stop if we find a run directory (has targets/, globals/, or cache/)
            # Don't stop at RESULTS/ - continue to find actual run directory
            if (temp_dir / "targets").exists() or (temp_dir / "globals").exists() or (temp_dir / "cache").exists():
                base_output_dir = temp_dir
                break
            if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                break
            temp_dir = temp_dir.parent
    
    # CRITICAL: Validate base_output_dir is not root and is absolute
    # If we walked all the way to root, fall back to original output_dir and try again
    if base_output_dir is None or base_output_dir == Path('/') or not base_output_dir.is_absolute() or str(base_output_dir) == '/':
        logger.warning(f"Path resolution failed - base_output_dir resolved to root or invalid: {base_output_dir}. Using original output_dir: {output_dir}")
        base_output_dir = Path(output_dir) if output_dir else None
        # Final attempt: Try to find run directory from output_dir
        if base_output_dir and base_output_dir.exists():
            temp_dir = base_output_dir
            for _ in range(10):
                if (temp_dir / "targets").exists() or (temp_dir / "globals").exists() or (temp_dir / "cache").exists():
                    base_output_dir = temp_dir
                    break
                if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                    break
                temp_dir = temp_dir.parent
    
    # Target-first structure: save to targets/<target>/decision/
    from TRAINING.orchestration.utils.target_first_paths import (
        get_target_decision_dir, ensure_target_structure
    )
    try:
        # Validate base_output_dir before using it
        if not base_output_dir.exists():
            logger.warning(f"base_output_dir does not exist: {base_output_dir}, using output_dir: {output_dir}")
            base_output_dir = output_dir
        
        ensure_target_structure(base_output_dir, target_name_clean)
        target_decision_dir = get_target_decision_dir(base_output_dir, target_name_clean)
        target_decision_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Target decision directory: {target_decision_dir}")
    except Exception as e:
        logger.warning(f"Failed to create target decision directory structure: {e}")
        logger.warning(f"  base_output_dir: {base_output_dir} (exists: {base_output_dir.exists()}, absolute: {base_output_dir.is_absolute()})")
        logger.warning(f"  output_dir: {output_dir} (exists: {output_dir.exists()})")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        target_decision_dir = None
    
    # Target-first structure only
    if target_decision_dir:
        target_decision_dir.mkdir(parents=True, exist_ok=True)
    else:
        logger.warning(f"Target decision directory not available for {target_name_clean}")
    
    # Handle empty results
    if summary_df is None or len(summary_df) == 0:
        logger.warning("No features to rank - all features were filtered or failed")
        # Create empty CSV file with headers
        empty_df = pd.DataFrame(columns=[
            'rank', 'feature', 'consensus_score', 'n_models_agree', 
            'consensus_pct', 'std_across_models', 'recommendation'
        ])
        # Save to target-first structure only
        try:
            from TRAINING.orchestration.utils.target_first_paths import get_target_reproducibility_dir
            # Validate base_output_dir before using it
            if base_output_dir == Path('/') or not base_output_dir.is_absolute() or str(base_output_dir) == '/':
                logger.warning(f"Invalid base_output_dir for CSV save: {base_output_dir}, using output_dir: {output_dir}")
                base_output_dir = output_dir
                # Try to find run directory from output_dir
                temp_dir = output_dir
                for _ in range(10):
                    if (temp_dir / "targets").exists() or (temp_dir / "globals").exists() or (temp_dir / "cache").exists():
                        base_output_dir = temp_dir
                        break
                    if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                        break
                    temp_dir = temp_dir.parent
            
            if not base_output_dir.exists():
                logger.warning(f"base_output_dir does not exist for CSV save: {base_output_dir}, using output_dir: {output_dir}")
                base_output_dir = output_dir
            
            target_repro_dir = get_target_reproducibility_dir(base_output_dir, target_name_clean)
            target_repro_dir.mkdir(parents=True, exist_ok=True)
            empty_csv_path = target_repro_dir / "feature_selection_rankings.csv"
            empty_df.to_csv(empty_csv_path, index=False)
            logger.info(f"Saved empty rankings file to {empty_csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save empty rankings file: {e}")
        return
    
    # Sort by consensus score
    summary_df_sorted = summary_df.sort_values('consensus_score', ascending=False).reset_index(drop=True)
    
    # Create DataFrame for rankings (similar to target ranking format)
    df = pd.DataFrame([{
        'rank': i + 1,
        'feature': row['feature'],
        'consensus_score': float(row.get('consensus_score', 0.0)),
        'consensus_score_base': float(row.get('consensus_score_base', 0.0)),
        'n_models_agree': int(row.get('n_models_agree', 0)),
        'consensus_pct': float(row.get('consensus_pct', 0.0)),
        'std_across_models': float(row.get('std_across_models', 0.0)),
        'cs_importance_score': float(row.get('cs_importance_score', 0.0)) if 'cs_importance_score' in row else None,
        'feature_category': row.get('feature_category', 'UNKNOWN'),
        'boruta_confirmed': bool(row.get('boruta_confirmed', False)) if 'boruta_confirmed' in row else None,
        'boruta_rejected': bool(row.get('boruta_rejected', False)) if 'boruta_rejected' in row else None,
        'boruta_tentative': bool(row.get('boruta_tentative', False)) if 'boruta_tentative' in row else None,
        'recommendation': _get_recommendation(row)
    } for i, (_, row) in enumerate(summary_df_sorted.iterrows())])
    
    # Save CSV to target-first structure (view/symbol-scoped)
    # NOTE: File naming convention to distinguish from target ranking:
    #   - Feature selection uses: "feature_selection_rankings.csv" (feature rankings for a target)
    #   - Target ranking uses: "target_predictability_rankings.csv" (target rankings)
    #   - These are distinct files with different purposes, no conflicts
    try:
        if use_output_layout and universe_sig:
            # Canonical path via OutputLayout (non-cohort write)
            from TRAINING.orchestration.utils.output_layout import OutputLayout
            layout = OutputLayout(
                output_root=base_output_dir,
                target=target_name_clean,
                view=view,
                universe_sig=universe_sig,
                symbol=symbol if view == "SYMBOL_SPECIFIC" else None,
            )
            repro_dir = layout.repro_dir()
            target_csv_path = repro_dir / "feature_selection_rankings.csv"
        else:
            # Legacy path resolution
            from TRAINING.orchestration.utils.target_first_paths import run_root, target_repro_file_path
            run_root_dir = run_root(base_output_dir)
            target_csv_path = target_repro_file_path(run_root_dir, target_name_clean, "feature_selection_rankings.csv", view=view, symbol=symbol)
        
        target_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_csv_path, index=False)
        logger.info(f"✅ Saved feature selection rankings CSV to {target_csv_path}")
    except Exception as e:
        logger.warning(f"Failed to write feature selection rankings CSV to target-first location: {e}")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
    
    # Save YAML with recommendations to DECISION (decision log)
    yaml_data = {
        'feature_rankings': [
            {
                'rank': i + 1,
                'feature': row['feature'],
                'consensus_score': float(row.get('consensus_score', 0.0)),
                'n_models_agree': int(row.get('n_models_agree', 0)),
                'consensus_pct': float(row.get('consensus_pct', 0.0)),
                'feature_category': row.get('feature_category', 'UNKNOWN'),
                'recommendation': _get_recommendation(row)
            }
            for i, (_, row) in enumerate(summary_df_sorted.iterrows())
        ],
        'summary': {
            'total_features': len(summary_df_sorted),
            'selected_features': len(selected_features),
            'target_column': target_column,
            'view': view,
            'symbol': symbol if symbol else None
        }
    }
    
    # Add metadata if provided
    if metadata:
        yaml_data['metadata'] = metadata
    
    # Save to target-first structure (primary location)
    # NOTE: File naming convention to distinguish from target ranking:
    #   - Feature selection uses: "feature_prioritization.yaml" (ranks features for a target)
    #   - Target ranking uses: "target_prioritization.yaml" (ranks targets themselves)
    #   - These are distinct files with different purposes, no conflicts
    if target_decision_dir:
        try:
            target_yaml_path = target_decision_dir / "feature_prioritization.yaml"
            with open(target_yaml_path, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            logger.info(f"✅ Saved feature prioritization YAML to {target_yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to save feature prioritization YAML to target-first location: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
    else:
        logger.warning(f"Target decision directory not available, skipping target-first save")
    
    # Save selected features list to target-first structure (view/symbol-scoped)
    try:
        if use_output_layout and universe_sig:
            # Canonical path via OutputLayout (non-cohort write)
            from TRAINING.orchestration.utils.output_layout import OutputLayout
            layout = OutputLayout(
                output_root=base_output_dir,
                target=target_name_clean,
                view=view,
                universe_sig=universe_sig,
                symbol=symbol if view == "SYMBOL_SPECIFIC" else None,
            )
            repro_dir = layout.repro_dir()
            target_selected_path = repro_dir / "selected_features.txt"
        else:
            # Legacy path resolution with validation
            from TRAINING.orchestration.utils.target_first_paths import run_root, target_repro_file_path
            if base_output_dir == Path('/') or not base_output_dir.is_absolute() or str(base_output_dir) == '/':
                logger.warning(f"Invalid base_output_dir for selected features: {base_output_dir}, using output_dir: {output_dir}")
                base_output_dir = output_dir
            if not base_output_dir.exists():
                logger.warning(f"base_output_dir does not exist for selected features: {base_output_dir}, using output_dir: {output_dir}")
                base_output_dir = output_dir
            run_root_dir = run_root(base_output_dir)
            target_selected_path = target_repro_file_path(run_root_dir, target_name_clean, "selected_features.txt", view=view, symbol=symbol)
        
        target_selected_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_selected_path, "w") as f:
            for feature in selected_features:
                f.write(f"{feature}\n")
        logger.debug(f"Saved selected features to {target_selected_path}")
    except Exception as e:
        logger.debug(f"Failed to write selected features: {e}")


def _get_recommendation(row: pd.Series) -> str:
    """Get recommendation based on consensus score (similar to target ranking)"""
    consensus_score = row.get('consensus_score', 0.0)
    boruta_confirmed = row.get('boruta_confirmed', False) if 'boruta_confirmed' in row else False
    feature_category = row.get('feature_category', 'UNKNOWN')
    
    if boruta_confirmed and consensus_score >= 0.5:
        return "PRIORITIZE - Strong signal + Boruta confirmed"
    elif consensus_score >= 0.5:
        return "ENABLE - Good consensus across models"
    elif consensus_score >= 0.3:
        return "TEST - Moderate consensus, worth exploring"
    elif feature_category == 'CORE':
        return "CONSIDER - Core feature (strong in both views)"
    else:
        return "DEPRIORITIZE - Weak consensus"


def save_dual_view_feature_selections(
    results_cs: Optional[Dict[str, Any]],
    results_sym: Optional[Dict[str, Dict[str, Any]]],
    target_column: str,
    output_dir: Path
):
    """
    Save dual-view feature selection results.
    
    NOTE: This function is deprecated. All data is now saved to the target-first
    structure by the reproducibility tracker. This function is kept as a no-op
    for backward compatibility but does not write any files.
    
    Args:
        results_cs: Cross-sectional results dict (if available)
        results_sym: Symbol-specific results dict {symbol: results} (if available)
        target_column: Target column name
        output_dir: Base output directory
    """
    # All data is already saved by reproducibility tracker to target-first structure
    # No legacy writes needed - target-first structure is the only source of truth
    logger.debug(f"save_dual_view_feature_selections called for {target_column} (no-op, data already in target-first structure)")
    pass


def aggregate_importances_cross_sectional(symbol_importances_by_symbol: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """
    Aggregate feature importances across symbols for CROSS_SECTIONAL view.
    
    Uses mean of normalized importances per model family (deterministic rule).
    
    Args:
        symbol_importances_by_symbol: Dict of {symbol: {model_family: {feature: importance}}}
    
    Returns:
        Aggregated dict of {model_family: {feature: mean_importance}}
    """
    if not symbol_importances_by_symbol:
        return {}
    
    # Collect all model families across all symbols
    all_model_families = set()
    for symbol_importances in symbol_importances_by_symbol.values():
        all_model_families.update(symbol_importances.keys())
    
    aggregated = {}
    for model_family in all_model_families:
        # Collect all features for this model family across all symbols
        all_features = set()
        for symbol_importances in symbol_importances_by_symbol.values():
            if model_family in symbol_importances:
                all_features.update(symbol_importances[model_family].keys())
        
        # Aggregate importances per feature (mean across symbols)
        feature_importances = {}
        for feature in all_features:
            importances = []
            for symbol_importances in symbol_importances_by_symbol.values():
                if model_family in symbol_importances and feature in symbol_importances[model_family]:
                    importances.append(symbol_importances[model_family][feature])
            
            if importances:
                # Use mean (deterministic rule)
                feature_importances[feature] = sum(importances) / len(importances)
        
        if feature_importances:
            aggregated[model_family] = feature_importances
    
    return aggregated


def save_feature_importances_for_reproducibility(
    all_feature_importances: Dict[str, Dict[str, float]],
    target_column: str,
    output_dir: Path,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    universe_sig: Optional[str] = None,  # Phase A: optional for backward compat
):
    """
    Save feature importances in the same structure as target ranking.
    
    Structure (same as target ranking):
    REPRODUCIBILITY/FEATURE_SELECTION/{view}/{target_column}/feature_importances/
      {symbol if SYMBOL_SPECIFIC}/
        {model_family}_importances.csv
    
    This matches the structure used by target ranking's _save_feature_importances.
    
    Args:
        all_feature_importances: Dict of {model_family: {feature: importance}}
        target_column: Target column name
        output_dir: Base output directory (REPRODUCIBILITY/FEATURE_SELECTION/CROSS_SECTIONAL/{target}/)
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name (for SYMBOL_SPECIFIC view)
    """
    import pandas as pd
    
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    
    # Phase A: Use OutputLayout if universe_sig provided (new canonical path)
    # Otherwise fall back to legacy path resolution with warning
    use_output_layout = bool(universe_sig)
    if not use_output_layout:
        logger.warning(
            f"universe_sig not provided for {target_column} feature importances, "
            f"falling back to legacy path resolution. Pass universe_sig for canonical paths."
        )
    
    # Find base run directory for target-first structure
    base_output_dir = output_dir
    for _ in range(10):
        # Only stop if we find a run directory (has targets/, globals/, or cache/)
        # Don't stop at RESULTS/ - continue to find actual run directory
        if (base_output_dir / "targets").exists() or (base_output_dir / "globals").exists() or (base_output_dir / "cache").exists():
            break
        if not base_output_dir.parent.exists():
            break
        base_output_dir = base_output_dir.parent
    
    # CRITICAL: Validate base_output_dir is not root and is absolute
    # If we walked all the way to root, fall back to original output_dir
    if base_output_dir == Path('/') or not base_output_dir.is_absolute() or str(base_output_dir) == '/':
        logger.warning(f"Path resolution failed for feature importances - base_output_dir resolved to root or invalid: {base_output_dir}. Using original output_dir: {output_dir}")
        base_output_dir = output_dir
        # Try to find run directory from output_dir instead
        temp_dir = output_dir
        for _ in range(10):
            if (temp_dir / "targets").exists() or (temp_dir / "globals").exists() or (temp_dir / "cache").exists():
                base_output_dir = temp_dir
                break
            if not temp_dir.parent.exists() or temp_dir.parent == temp_dir:
                break
            temp_dir = temp_dir.parent
    
    # Set up importances directory
    try:
        if use_output_layout and universe_sig:
            # Canonical path via OutputLayout.feature_importance_dir() (non-cohort write)
            from TRAINING.orchestration.utils.output_layout import OutputLayout
            layout = OutputLayout(
                output_root=base_output_dir,
                target=target_name_clean,
                view=view,
                universe_sig=universe_sig,
                symbol=symbol if view == "SYMBOL_SPECIFIC" else None,
            )
            importances_dir = layout.feature_importance_dir()
        else:
            # Legacy path resolution
            from TRAINING.orchestration.utils.target_first_paths import run_root, target_repro_dir, ensure_target_structure
            if not base_output_dir.exists():
                logger.warning(f"base_output_dir does not exist for feature importances: {base_output_dir}, using output_dir: {output_dir}")
                base_output_dir = output_dir
            
            run_root_dir = run_root(base_output_dir)
            ensure_target_structure(run_root_dir, target_name_clean)
            repro_dir = target_repro_dir(run_root_dir, target_name_clean, view=view, symbol=symbol)
            importances_dir = repro_dir / "feature_importances"
        
        importances_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.warning(f"Failed to set up feature importances directory: {e}")
        logger.warning(f"  base_output_dir: {base_output_dir} (exists: {base_output_dir.exists() if hasattr(base_output_dir, 'exists') else 'N/A'})")
        logger.warning(f"  output_dir: {output_dir} (exists: {output_dir.exists()})")
        import traceback
        logger.debug(f"Traceback: {traceback.format_exc()}")
        # Fallback: use output_dir directly (legacy path)
        importances_dir = output_dir / "feature_importances"
        importances_dir.mkdir(parents=True, exist_ok=True)
    
    # Save per-model-family importances as CSV (same format as target ranking)
    for model_family, importance_dict in all_feature_importances.items():
        if not importance_dict:
            continue
        
        # Create DataFrame sorted by importance (same as target ranking)
        df = pd.DataFrame([
            {'feature': feat, 'importance': imp}
            for feat, imp in sorted(importance_dict.items())
        ])
        df = df.sort_values('importance', ascending=False)
        
        # Normalize to percentages (same as target ranking)
        total = df['importance'].sum()
        if total > 0:
            df['importance_pct'] = (df['importance'] / total * 100).round(2)
            df['cumulative_pct'] = df['importance_pct'].cumsum().round(2)
        else:
            df['importance_pct'] = 0.0
            df['cumulative_pct'] = 0.0
        
        # Reorder columns (same as target ranking)
        df = df[['feature', 'importance', 'importance_pct', 'cumulative_pct']]
        
        # Save to CSV (same filename format as target ranking)
        csv_file = importances_dir / f"{model_family}_importances.csv"
        df.to_csv(csv_file, index=False)
        logger.debug(f"Saved {model_family} importances to {csv_file}")
    
    logger.info(f"Saved feature importances to {importances_dir}")
