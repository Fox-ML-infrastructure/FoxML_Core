"""
Feature Selection Reporting

Saves feature selection results in the same structure as target ranking:
- CSV and YAML summary files
- REPRODUCIBILITY/FEATURE_SELECTION/ structure with CROSS_SECTIONAL/SYMBOL_SPECIFIC views
- Feature importances
- Stability snapshots
- Cross-sectional ranking results
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
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save feature selection rankings in the same format as target ranking.
    
    Structure:
    RESULTS/{run_id}/
      feature_selections/
        {target_column}/
          feature_selection_rankings.csv
          feature_selection_rankings.yaml
          selected_features.txt
          feature_importance_multi_model.csv
          ...
      REPRODUCIBILITY/
        FEATURE_SELECTION/
          {view}/
            {target_column}/
              cohort={cohort_id}/
                metrics.json
                metadata.json
    
    Args:
        summary_df: DataFrame with feature rankings and scores
        selected_features: List of selected feature names
        target_column: Target column name
        output_dir: Base output directory
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name (for SYMBOL_SPECIFIC view)
        metadata: Optional metadata dict
    """
    # Determine base output directory (handle both old and new call patterns)
    if output_dir.name == "feature_selections":
        base_output_dir = output_dir.parent
    elif output_dir.parent.name == "feature_selections":
        base_output_dir = output_dir.parent.parent
    else:
        base_output_dir = output_dir.parent if output_dir.name == target_column else output_dir
    
    # Create directories
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    repro_dir = base_output_dir / "REPRODUCIBILITY" / "FEATURE_SELECTION" / target_name_clean
    decision_dir = base_output_dir / "DECISION" / "FEATURE_SELECTION" / target_name_clean
    repro_dir.mkdir(parents=True, exist_ok=True)
    decision_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle empty results
    if summary_df is None or len(summary_df) == 0:
        logger.warning("No features to rank - all features were filtered or failed")
        # Create empty CSV file with headers
        empty_df = pd.DataFrame(columns=[
            'rank', 'feature', 'consensus_score', 'n_models_agree', 
            'consensus_pct', 'std_across_models', 'recommendation'
        ])
        empty_df.to_csv(repro_dir / "feature_selection_rankings.csv", index=False)
        logger.info(f"Saved empty rankings file to {repro_dir / 'feature_selection_rankings.csv'}")
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
    
    # Save CSV to REPRODUCIBILITY (reproducibility artifact)
    csv_path = repro_dir / "feature_selection_rankings.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved feature selection rankings CSV to {csv_path}")
    
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
    
    yaml_path = decision_dir / "feature_prioritization.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
    
    logger.info(f"Saved feature prioritization YAML to {yaml_path}")
    
    # Save selected features list to REPRODUCIBILITY (reproducibility artifact)
    selected_features_path = repro_dir / "selected_features.txt"
    with open(selected_features_path, "w") as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    logger.info(f"Saved {len(selected_features)} selected features to {selected_features_path}")


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
    Save dual-view feature selection results (same structure as target ranking).
    
    Structure (same as target ranking):
    REPRODUCIBILITY/FEATURE_SELECTION/
      CROSS_SECTIONAL/
        {target_column}/
          cohort={cohort_id}/
            metrics.json
            metadata.json
      SYMBOL_SPECIFIC/
        {target_column}/
          symbol={symbol}/
            cohort={cohort_id}/
              metrics.json
              metadata.json
    
    Note: Individual view results are already saved by reproducibility tracker
    (with view/symbol metadata in RunContext), so this function just saves
    summary files for convenience (same pattern as target ranking).
    
    Args:
        results_cs: Cross-sectional results dict (if available)
        results_sym: Symbol-specific results dict {symbol: results} (if available)
        target_column: Target column name
        output_dir: Base output directory
    """
    import json
    from pathlib import Path
    
    # Determine REPRODUCIBILITY directory (same as target ranking)
    # output_dir might be: RESULTS/{run_id}/feature_selections/{target}/
    # We want: RESULTS/{run_id}/REPRODUCIBILITY/FEATURE_SELECTION/
    # Handle various output_dir structures
    if output_dir.name == target_column:
        # output_dir is feature_selections/{target}/
        repro_base = output_dir.parent.parent / "REPRODUCIBILITY"
    elif output_dir.name == 'feature_selections':
        # output_dir is feature_selections/
        repro_base = output_dir.parent / "REPRODUCIBILITY"
    elif (output_dir.parent / 'feature_selections').exists():
        # output_dir is in a subdirectory of feature_selections
        repro_base = output_dir.parent.parent / "REPRODUCIBILITY"
    else:
        # Fallback: use output_dir's parent
        repro_base = output_dir.parent / "REPRODUCIBILITY"
    
    repro_dir = repro_base / "FEATURE_SELECTION"
    repro_dir.mkdir(parents=True, exist_ok=True)
    
    # Save cross-sectional results if available
    if results_cs:
        cs_dir = repro_dir / "CROSS_SECTIONAL" / target_column
        cs_dir.mkdir(parents=True, exist_ok=True)
        
        # Results are already saved by reproducibility tracker, but we can save summary here
        summary_file = cs_dir / "feature_selection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results_cs, f, indent=2, default=str)
        logger.debug(f"Saved cross-sectional feature selection summary to {summary_file}")
    
    # Save symbol-specific results if available
    if results_sym:
        for symbol, sym_results in results_sym.items():
            sym_dir = repro_dir / "SYMBOL_SPECIFIC" / target_column / f"symbol={symbol}"
            sym_dir.mkdir(parents=True, exist_ok=True)
            
            # Results are already saved by reproducibility tracker, but we can save summary here
            summary_file = sym_dir / "feature_selection_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(sym_results, f, indent=2, default=str)
            logger.debug(f"Saved symbol-specific feature selection summary for {symbol} to {summary_file}")
    
    logger.info(f"Saved dual-view feature selection results to {repro_dir}")


def save_feature_importances_for_reproducibility(
    all_feature_importances: Dict[str, Dict[str, float]],
    target_column: str,
    output_dir: Path,
    view: str = "CROSS_SECTIONAL",
    symbol: Optional[str] = None
):
    """
    Save feature importances in the same structure as target ranking.
    
    Structure (same as target ranking):
    {output_dir}/feature_selections/feature_importances/
      {target_column}/
        {view}/
          {symbol if SYMBOL_SPECIFIC}/
            {model_family}_importances.csv
    
    This matches the structure used by target ranking's _save_feature_importances.
    
    Args:
        all_feature_importances: Dict of {model_family: {feature: importance}}
        target_column: Target column name
        output_dir: Base output directory
        view: "CROSS_SECTIONAL" or "SYMBOL_SPECIFIC"
        symbol: Symbol name (for SYMBOL_SPECIFIC view)
    """
    import pandas as pd
    
    # Determine base directory (same logic as target ranking)
    # output_dir might be: RESULTS/{run_id}/feature_selections/{target}/
    # We want: RESULTS/{run_id}/feature_selections/feature_importances/{target}/
    if output_dir.name == target_column:
        # output_dir is feature_selections/{target}/
        base_dir = output_dir.parent
    elif output_dir.name == 'feature_selections':
        # output_dir is feature_selections/
        base_dir = output_dir
    else:
        # Fallback: use output_dir directly
        base_dir = output_dir
    
    target_name_clean = target_column.replace('/', '_').replace('\\', '_')
    
    # Create directory structure (same as target ranking)
    if view == "SYMBOL_SPECIFIC" and symbol:
        importances_dir = base_dir / "feature_importances" / target_name_clean / view / symbol
    else:
        importances_dir = base_dir / "feature_importances" / target_name_clean / view
    
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
