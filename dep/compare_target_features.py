# MIT License - see LICENSE file

"""
Compare Feature Importance Across Multiple Targets

Analyzes which feature CONCEPTS are important for which prediction tasks.
This helps understand:
  - Which features are universal (important for all targets)
  - Which features are task-specific (only important for certain targets)
  - Feature concept patterns across target types

Usage:
    # Compare all completed targets
    python SCRIPTS/compare_target_features.py
    
    # Compare specific targets
    python SCRIPTS/compare_target_features.py --targets peak_60m,valley_60m,swing_high_15m
"""


import argparse
import sys
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd
import numpy as np
import logging
import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]

def load_feature_groups(config_path: Path = None) -> Dict[str, List[str]]:
    """Load feature group definitions."""
    if config_path is None:
        config_path = _REPO_ROOT / "CONFIG" / "feature_groups.yaml"
    
    if not config_path.exists():
        logger.warning(f"Feature groups config not found: {config_path}")
        return {}
    
    with open(config_path) as f:
        groups = yaml.safe_load(f)
    
    return groups

def load_target_results(base_dir: Path, target_names: List[str] = None) -> Dict[str, pd.DataFrame]:
    """
    Load feature importance results for multiple targets.
    
    Returns:
        Dict mapping target_name -> importance DataFrame
    """
    results = {}
    
    # If no targets specified, find all subdirectories with results
    if target_names is None:
        target_dirs = [d for d in base_dir.iterdir() if d.is_dir() and (d / "feature_importance_summary.csv").exists()]
        target_names = [d.name for d in target_dirs]
        logger.info(f"Found {len(target_names)} targets with results")
    
    for target_name in target_names:
        result_file = base_dir / target_name / "feature_importance_summary.csv"
        
        if not result_file.exists():
            logger.warning(f"Results not found for {target_name}: {result_file}")
            continue
        
        try:
            df = pd.read_csv(result_file)
            results[target_name] = df
            logger.info(f"Loaded {len(df)} features for {target_name}")
        except Exception as e:
            logger.error(f"Failed to load {target_name}: {e}")
    
    return results

def map_features_to_groups(features: List[str], feature_groups: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Map each feature to its group.
    
    Returns:
        Dict mapping feature_name -> group_name
    """
    feature_to_group = {}
    
    for group_name, group_features in feature_groups.items():
        for feature in features:
            if feature in group_features:
                feature_to_group[feature] = group_name
    
    return feature_to_group

def analyze_cross_target_patterns(
    target_results: Dict[str, pd.DataFrame],
    feature_groups: Dict[str, List[str]],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Analyze feature group importance across multiple targets.
    
    For each target, aggregate importance by feature group,
    then compare patterns across targets.
    """
    # For each target, calculate group-level importance
    target_group_scores = {}
    
    for target_name, importance_df in target_results.items():
        # Map features to groups
        feature_to_group = map_features_to_groups(importance_df['feature'].tolist(), feature_groups)
        importance_df['group'] = importance_df['feature'].map(lambda f: feature_to_group.get(f, '__UNGROUPED__'))
        
        # Aggregate by group
        group_scores = importance_df.groupby('group')['score'].sum().sort_values(ascending=False)
        target_group_scores[target_name] = group_scores
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(target_group_scores).fillna(0)
    
    # Add summary statistics
    comparison_df['mean_score'] = comparison_df.mean(axis=1)
    comparison_df['std_score'] = comparison_df.std(axis=1)
    comparison_df['num_targets'] = (comparison_df > 0).sum(axis=1)
    comparison_df['universality'] = comparison_df['num_targets'] / len(target_results)
    
    # Sort by mean score
    comparison_df = comparison_df.sort_values('mean_score', ascending=False)
    
    return comparison_df

def identify_target_specific_features(
    target_results: Dict[str, pd.DataFrame],
    top_n: int = 10
) -> Dict[str, List[str]]:
    """
    Identify features that are uniquely important for each target.
    
    Returns:
        Dict mapping target_name -> list of target-specific features
    """
    # Collect all features that appear in top N for each target
    target_top_features = {}
    for target_name, importance_df in target_results.items():
        top_features = set(importance_df.head(top_n)['feature'].tolist())
        target_top_features[target_name] = top_features
    
    # Find unique features for each target
    target_specific = {}
    for target_name, top_features in target_top_features.items():
        # Features that appear in this target's top N but not in others' top N
        other_top_features = set()
        for other_target, other_features in target_top_features.items():
            if other_target != target_name:
                other_top_features.update(other_features)
        
        unique_features = top_features - other_top_features
        target_specific[target_name] = list(unique_features)
    
    return target_specific

def identify_universal_features(
    target_results: Dict[str, pd.DataFrame],
    min_targets: int = None,
    top_n: int = 20
) -> List[str]:
    """
    Identify features that are important across many targets.
    
    Args:
        min_targets: Minimum number of targets a feature must appear in (default: 70% of targets)
    
    Returns:
        List of universal features
    """
    if min_targets is None:
        min_targets = int(0.7 * len(target_results))
    
    # Count how many targets each feature appears in (top N)
    feature_counts = {}
    for target_name, importance_df in target_results.items():
        top_features = importance_df.head(top_n)['feature'].tolist()
        for feature in top_features:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Filter to universal features
    universal = [f for f, count in feature_counts.items() if count >= min_targets]
    
    return universal

def main():
    parser = argparse.ArgumentParser(description="Compare feature importance across multiple targets.")
    parser.add_argument("--base-dir", type=Path, default=_REPO_ROOT / "DATA_PROCESSING" / "data" / "features", help="Base directory containing target results")
    parser.add_argument("--targets", type=str, help="Comma-separated list of target names to compare (default: all)")
    parser.add_argument("--output", type=Path, help="Output CSV path (default: base_dir/cross_target_comparison.csv)")
    parser.add_argument("--top-n", type=int, default=20, help="Top N features to consider for each target")
    
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.base_dir / "cross_target_comparison.csv"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"📊 Cross-Target Feature Analysis")
    logger.info(f"{'='*80}")
    logger.info(f"Base Directory: {args.base_dir}")
    logger.info(f"Top N: {args.top_n}")
    logger.info(f"{'-'*80}")
    
    # Parse target names if provided
    target_names = args.targets.split(',') if args.targets else None
    
    # Load results
    target_results = load_target_results(args.base_dir, target_names)
    if not target_results:
        logger.error("❌ No target results found")
        return 1
    
    # Load feature groups
    feature_groups = load_feature_groups()
    if not feature_groups:
        logger.warning("⚠️  No feature groups found, skipping group analysis")
    
    # Analyze patterns
    if feature_groups:
        logger.info(f"\n{'='*80}")
        logger.info(f"Feature Group Importance by Target")
        logger.info(f"{'='*80}\n")
        
        comparison_df = analyze_cross_target_patterns(target_results, feature_groups, args.top_n)
        
        # Display top groups
        for idx, row in comparison_df.head(15).iterrows():
            logger.info(f"{idx:20s}  Mean: {row['mean_score']:>10,.0f}  "
                       f"Universality: {row['universality']:>5.0%}  "
                       f"Targets: {row['num_targets']:.0f}/{len(target_results)}")
        
        # Save full comparison
        comparison_df.to_csv(args.output)
        logger.info(f"\n💾 Saved full comparison to {args.output}")
    
    # Universal features
    logger.info(f"\n{'='*80}")
    logger.info(f"Universal Features (appear in ≥70% of targets)")
    logger.info(f"{'='*80}\n")
    
    universal = identify_universal_features(target_results, top_n=args.top_n)
    logger.info(f"Found {len(universal)} universal features:")
    for i, feature in enumerate(universal[:15], 1):
        logger.info(f"  {i:2d}. {feature}")
    if len(universal) > 15:
        logger.info(f"  ... and {len(universal) - 15} more")
    
    # Target-specific features
    logger.info(f"\n{'='*80}")
    logger.info(f"Target-Specific Features (unique to each target)")
    logger.info(f"{'='*80}\n")
    
    target_specific = identify_target_specific_features(target_results, top_n=args.top_n)
    for target_name, specific_features in sorted(target_specific.items()):
        if specific_features:
            logger.info(f"{target_name}:")
            for feature in specific_features[:5]:
                logger.info(f"  • {feature}")
            if len(specific_features) > 5:
                logger.info(f"  ... and {len(specific_features) - 5} more")
        else:
            logger.info(f"{target_name}: (no unique features)")
        logger.info("")
    
    logger.info(f"{'='*80}")
    logger.info(f"✅ Analysis complete!")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Key Insights:")
    logger.info(f"  • {len(universal)} features are important across most targets")
    logger.info(f"  • Each target has {np.mean([len(f) for f in target_specific.values()]):.0f} unique features on average")
    if feature_groups:
        top_group = comparison_df.index[0]
        logger.info(f"  • '{top_group}' is the most consistently important feature group")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

