# MIT License - see LICENSE file

"""
Aggregate Feature Importance by Concept Groups

Takes the output of select_features.py and aggregates importance scores
by feature families (RSI, momentum, volatility, etc.) to identify which
CONCEPTS are most predictive, not just which individual features.

Usage:
    python SCRIPTS/aggregate_feature_groups.py --input DATA_PROCESSING/data/features/feature_importance_summary.csv
"""


import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd
import yaml
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

def load_feature_groups(config_path: Path = None) -> Dict[str, List[str]]:
    """Load feature group definitions from YAML config."""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "feature_groups.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ Feature groups config not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        groups = yaml.safe_load(f)
    
    logger.info(f"📋 Loaded {len(groups)} feature groups from {config_path}")
    return groups

def aggregate_by_groups(
    importance_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    method: str = "sum"
) -> pd.DataFrame:
    """
    Aggregate feature importance by predefined groups.
    
    Args:
        importance_df: DataFrame with columns [feature, score, frequency, frequency_pct]
        feature_groups: Dict mapping group names to lists of feature names
        method: Aggregation method ('sum', 'mean', 'max')
    
    Returns:
        DataFrame with columns [group, score, num_features, avg_feature_score, coverage_pct]
    """
    results = []
    total_features = len(importance_df)
    
    # Track which features were assigned to groups
    assigned_features = set()
    
    for group_name, feature_list in feature_groups.items():
        # Find features in this group that appear in the importance data
        group_features = importance_df[importance_df['feature'].isin(feature_list)]
        
        if group_features.empty:
            continue
        
        # Track assigned features
        assigned_features.update(group_features['feature'].tolist())
        
        # Aggregate scores
        if method == "sum":
            group_score = group_features['score'].sum()
        elif method == "mean":
            group_score = group_features['score'].mean()
        elif method == "max":
            group_score = group_features['score'].max()
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        results.append({
            'group': group_name,
            'score': group_score,
            'num_features': len(group_features),
            'avg_feature_score': group_features['score'].mean(),
            'coverage_pct': (len(group_features) / total_features) * 100,
            'top_feature': group_features.nlargest(1, 'score')['feature'].iloc[0] if not group_features.empty else None,
            'top_feature_score': group_features['score'].max()
        })
    
    # Add ungrouped features
    unassigned_features = importance_df[~importance_df['feature'].isin(assigned_features)]
    if not unassigned_features.empty:
        logger.warning(f"⚠️  {len(unassigned_features)} features not assigned to any group")
        results.append({
            'group': '__UNGROUPED__',
            'score': unassigned_features['score'].sum(),
            'num_features': len(unassigned_features),
            'avg_feature_score': unassigned_features['score'].mean(),
            'coverage_pct': (len(unassigned_features) / total_features) * 100,
            'top_feature': unassigned_features.nlargest(1, 'score')['feature'].iloc[0],
            'top_feature_score': unassigned_features['score'].max()
        })
    
    # Create result DataFrame and sort by score
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('score', ascending=False).reset_index(drop=True)
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description="Aggregate feature importance by concept groups.")
    parser.add_argument("--input", type=Path, required=True, help="Path to feature_importance_summary.csv from select_features.py")
    parser.add_argument("--groups", type=Path, default=_REPO_ROOT / "CONFIG" / "feature_groups.yaml", help="Path to feature groups YAML config")
    parser.add_argument("--output", type=Path, help="Output path for aggregated results (default: same dir as input with '_grouped' suffix)")
    parser.add_argument("--method", type=str, default="sum", choices=["sum", "mean", "max"], help="Aggregation method")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top groups to display")
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output is None:
        args.output = args.input.parent / f"{args.input.stem}_grouped.csv"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🔍 Feature Importance Aggregation by Concept Groups")
    logger.info(f"{'='*80}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Groups Config: {args.groups}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Method: {args.method}")
    logger.info(f"{'-'*80}")
    
    # Load feature importance data
    if not args.input.exists():
        logger.error(f"❌ Input file not found: {args.input}")
        logger.error(f"   Run: python SCRIPTS/select_features.py first")
        return 1
    
    importance_df = pd.read_csv(args.input)
    logger.info(f"📊 Loaded {len(importance_df)} features from {args.input}")
    
    # Load feature groups
    feature_groups = load_feature_groups(args.groups)
    if not feature_groups:
        return 1
    
    # Aggregate by groups
    grouped_df = aggregate_by_groups(importance_df, feature_groups, args.method)
    
    # Save results
    grouped_df.to_csv(args.output, index=False)
    logger.info(f"💾 Saved aggregated results to {args.output}")
    
    # Display summary
    logger.info(f"\n{'='*80}")
    logger.info(f"📈 Top {args.top_n} Feature Concept Groups (by {args.method} score)")
    logger.info(f"{'='*80}\n")
    
    for idx, row in grouped_df.head(args.top_n).iterrows():
        logger.info(f"{idx+1:2d}. {row['group']:20s}  Score: {row['score']:>12,.0f}  "
                   f"Features: {row['num_features']:>3.0f}  "
                   f"Avg: {row['avg_feature_score']:>10,.0f}")
        logger.info(f"     └─ Top feature: {row['top_feature']} (score: {row['top_feature_score']:,.0f})")
        logger.info("")
    
    logger.info(f"{'-'*80}")
    logger.info(f"✅ Analysis complete!")
    logger.info(f"\n📊 Key Insights:")
    logger.info(f"  • Total groups analyzed: {len(grouped_df)}")
    logger.info(f"  • Groups with features: {len(grouped_df[grouped_df['group'] != '__UNGROUPED__'])}")
    logger.info(f"  • Top group: {grouped_df.iloc[0]['group']} (score: {grouped_df.iloc[0]['score']:,.0f})")
    logger.info(f"\n💡 Next Steps:")
    logger.info(f"  1. Review {args.output} for full results")
    logger.info(f"  2. Select top 1-2 features from each high-scoring group")
    logger.info(f"  3. Build a 'concept-balanced' feature set for training")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

