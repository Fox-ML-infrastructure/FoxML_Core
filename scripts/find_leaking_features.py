#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

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
Find leaking features from feature importance results.

This script analyzes exported feature importance CSVs to identify:
1. Features matching known leakage patterns (from config)
2. Features with suspiciously high importance (>50% or >30% and 3x next)
3. Features that should be added to exclusion config

Usage:
    python scripts/find_leaking_features.py <output_dir> [--top-n 50] [--threshold 0.50]
    
Example:
    python scripts/find_leaking_features.py results/target_rankings_updated
    python scripts/find_leaking_features.py results/target_rankings_updated --top-n 30 --threshold 0.40
"""


import sys
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import yaml

# Load patterns from config
def load_exclusion_patterns():
    """Load exclusion patterns from CONFIG/excluded_features.yaml"""
    config_path = Path(__file__).resolve().parents[1] / "CONFIG" / "excluded_features.yaml"
    
    if not config_path.exists():
        print(f"WARNING: Config not found: {config_path}, using defaults")
        return {
            'regex': [],
            'prefix': [],
            'keyword': [],
            'exact': [],
            'barrier_keywords': [],
            'horizon_config': {},
            'target_classification': {}
        }
    
    with open(config_path) as f:
        config = yaml.safe_load(f) or {}
    
    always_exclude = config.get('always_exclude', {})
    barrier_rules = config.get('target_type_rules', {}).get('barrier', {})
    
    return {
        'regex': always_exclude.get('regex_patterns', []),
        'prefix': always_exclude.get('prefix_patterns', []),
        'keyword': always_exclude.get('keyword_patterns', []),
        'exact': always_exclude.get('exact_patterns', []),
        'barrier_keywords': barrier_rules.get('keyword_patterns', []),
        'horizon_config': config.get('horizon_extraction', {}),
        'target_classification': config.get('target_classification', {}),
        'barrier_horizon_overlap': barrier_rules.get('horizon_overlap', {})
    }


def extract_horizon(feature: str, horizon_config: dict) -> int:
    """Extract horizon from feature name (in minutes)."""
    import re
    
    patterns = horizon_config.get('patterns', [])
    for pattern_config in patterns:
        regex = pattern_config.get('regex')
        multiplier = pattern_config.get('multiplier', 1)
        
        if regex:
            match = re.search(regex, feature)
            if match:
                value = int(match.group(1))
                return value * multiplier
    
    return None


def matches_exclusion_pattern(feature: str, patterns: dict, target: str = None) -> tuple:
    """
    Check if feature matches any exclusion pattern.
    
    Now checks:
    1. Standard patterns (regex, prefix, keyword, exact)
    2. Horizon-based exclusions (target/4 rule)
    3. HIGH/LOW-based exclusions for peak/valley targets
    
    Returns:
        (is_match, pattern_type, pattern, description)
    """
    import re
    
    # Check regex patterns
    for pattern in patterns['regex']:
        try:
            if re.match(pattern, feature):
                return (True, 'regex', pattern, f'Matches regex: {pattern}')
        except re.error:
            continue
    
    # Check prefix patterns
    for pattern in patterns['prefix']:
        if feature.startswith(pattern):
            return (True, 'prefix', pattern, f'Starts with: {pattern}')
    
    # Check keyword patterns
    for pattern in patterns['keyword']:
        if pattern.lower() in feature.lower():
            return (True, 'keyword', pattern, f'Contains keyword: {pattern}')
    
    # Check exact patterns
    if feature in patterns['exact']:
        return (True, 'exact', feature, 'Exact match')
    
    # Check horizon-based exclusions (if target provided)
    if target:
        # Extract target horizon
        target_horizon = extract_horizon(target, patterns['horizon_config'])
        if target_horizon:
            # Extract feature horizon
            feat_horizon = extract_horizon(feature, patterns['horizon_config'])
            if feat_horizon:
                # Check matching horizon
                if feat_horizon == target_horizon:
                    return (True, 'horizon', f'{feat_horizon}m', 
                           f'Matches target horizon ({target_horizon}m)')
                
                # Check overlapping horizon (target/4 rule)
                horizon_overlap = patterns.get('barrier_horizon_overlap', {})
                if horizon_overlap.get('exclude_overlapping_horizon', True):
                    threshold = target_horizon / 4
                    if feat_horizon >= threshold:
                        return (True, 'horizon', f'{feat_horizon}m', 
                               f'Overlapping horizon ({feat_horizon}m >= {target_horizon}m/4)')
        
        # Check HIGH/LOW-based exclusions for peak/valley targets
        target_lower = target.lower()
        if 'peak' in target_lower:
            # Peak target: exclude HIGH-based features
            if any(kw in feature.lower() for kw in ['high', 'upper', 'max', 'top', 'ceiling']):
                # But allow legitimate exceptions
                if not any(allow in feature.lower() for allow in ['high_vol', 'high_freq', 'high_corr']):
                    return (True, 'high_feature', 'high/upper/max', 
                           'HIGH-based feature (excluded for peak targets)')
        
        if 'valley' in target_lower:
            # Valley target: exclude LOW-based features
            if any(kw in feature.lower() for kw in ['low', 'lower', 'min', 'bottom', 'floor']):
                # But allow legitimate exceptions
                if not any(allow in feature.lower() for allow in ['low_vol', 'low_freq', 'low_corr']):
                    return (True, 'low_feature', 'low/lower/min', 
                           'LOW-based feature (excluded for valley targets)')
    
    return (False, None, None, None)


def analyze_importances(output_dir: Path, top_n: int = 50, threshold: float = 0.50):
    """Analyze feature importance files and identify leaks."""
    importances_dir = output_dir / "feature_importances"
    
    if not importances_dir.exists():
        print(f"ERROR: Feature importances directory not found: {importances_dir}")
        return
    
    # Load exclusion patterns
    patterns = load_exclusion_patterns()
    
    # Find all importances files
    files = list(importances_dir.glob("**/*_importances.csv"))
    
    if not files:
        print(f"ERROR: No feature importance files found in {importances_dir}")
        return
    
    print("="*80)
    print("LEAKING FEATURES ANALYSIS")
    print("="*80)
    print(f"Found {len(files)} feature importance files")
    print(f"Analyzing top {top_n} features per model")
    print(f"High importance threshold: {threshold:.1%}")
    print()
    
    # Collect all suspicious features
    suspicious_features = defaultdict(lambda: {
        'pattern_match': False,
        'high_importance': False,
        'occurrences': [],
        'max_importance': 0.0,
        'pattern_type': None,
        'pattern': None,
        'description': None
    })
    
    for file in files:
        try:
            # Parse file path: .../feature_importances/{target}/{symbol}/{model}_importances.csv
            parts = file.parts
            if "feature_importances" not in parts:
                continue
            
            target_idx = parts.index("feature_importances")
            target = parts[target_idx + 1]
            symbol = parts[target_idx + 2]
            model = parts[target_idx + 3].replace("_importances.csv", "")
            
            df = pd.read_csv(file)
            top_features = df.head(top_n)
            
            for _, row in top_features.iterrows():
                feat = row['feature']
                imp_pct = row['importance_pct'] / 100.0  # Convert to 0-1 scale
                
                # Check if matches exclusion pattern (pass target for horizon/HIGH/LOW checks)
                is_match, pattern_type, pattern, description = matches_exclusion_pattern(feat, patterns, target=target)
                
                # Check for high importance
                is_high_importance = imp_pct >= threshold
                
                # Also check if top feature dominates
                is_dominant = False
                if len(top_features) > 1:
                    top_imp = top_features.iloc[0]['importance_pct'] / 100.0
                    second_imp = top_features.iloc[1]['importance_pct'] / 100.0
                    if top_imp >= 0.30 and top_imp > second_imp * 3:
                        is_dominant = (row.name == 0)  # Only flag the top feature
                
                if is_match or is_high_importance or is_dominant:
                    if feat not in suspicious_features:
                        suspicious_features[feat] = {
                            'pattern_match': is_match,
                            'high_importance': is_high_importance or is_dominant,
                            'occurrences': [],
                            'max_importance': 0.0,
                            'pattern_type': pattern_type,
                            'pattern': pattern,
                            'description': description
                        }
                    
                    suspicious_features[feat]['occurrences'].append({
                        'target': target,
                        'symbol': symbol,
                        'model': model,
                        'importance_pct': imp_pct * 100,
                        'rank': row.name + 1
                    })
                    
                    if imp_pct > suspicious_features[feat]['max_importance']:
                        suspicious_features[feat]['max_importance'] = imp_pct
        
        except Exception as e:
            print(f"WARNING: Failed to process {file}: {e}")
            continue
    
    # Print results
    if not suspicious_features:
        print("✓ No suspicious features found!")
        print()
        print("This could mean:")
        print("  1. All leaks are correctly filtered")
        print("  2. Leaks are below the top N features analyzed")
        print("  3. Leaks have low importance (unlikely if models show high scores)")
        return
    
    # Separate by type
    pattern_leaks = {k: v for k, v in suspicious_features.items() if v['pattern_match']}
    importance_leaks = {k: v for k, v in suspicious_features.items() if v['high_importance'] and not v['pattern_match']}
    
    print("="*80)
    print("LEAKS BY PATTERN MATCH (should be excluded but found in top features)")
    print("="*80)
    print()
    
    if pattern_leaks:
        for feat, info in sorted(pattern_leaks.items(), key=lambda x: len(x[1]['occurrences']), reverse=True):
            print(f"🚨 {feat}")
            print(f"   Pattern: {info['pattern']} ({info['pattern_type']})")
            print(f"   Description: {info['description']}")
            print(f"   Found in {len(info['occurrences'])} model/symbol combinations")
            print(f"   Max importance: {info['max_importance']:.1%}")
            
            # Show top occurrences
            top_occ = sorted(info['occurrences'], key=lambda x: x['importance_pct'], reverse=True)[:3]
            for occ in top_occ:
                print(f"     - {occ['model']:15s} {occ['symbol']:6s} rank={occ['rank']:2d} imp={occ['importance_pct']:5.2f}%")
            print()
    else:
        print("✓ No pattern-matched leaks found in top features")
        print()
    
    print("="*80)
    print("LEAKS BY HIGH IMPORTANCE (suspicious but not matching known patterns)")
    print("="*80)
    print()
    
    if importance_leaks:
        for feat, info in sorted(importance_leaks.items(), key=lambda x: x[1]['max_importance'], reverse=True):
            print(f"⚠️  {feat}")
            print(f"   Max importance: {info['max_importance']:.1%}")
            print(f"   Found in {len(info['occurrences'])} model/symbol combinations")
            
            # Show top occurrences
            top_occ = sorted(info['occurrences'], key=lambda x: x['importance_pct'], reverse=True)[:3]
            for occ in top_occ:
                print(f"     - {occ['model']:15s} {occ['symbol']:6s} rank={occ['rank']:2d} imp={occ['importance_pct']:5.2f}%")
            print()
    else:
        print("✓ No high-importance leaks found")
        print()
    
    # Generate recommendations
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()
    
    if importance_leaks:
        print("1. Features with high importance that should be investigated:")
        print()
        for feat in sorted(importance_leaks.keys(), key=lambda x: importance_leaks[x]['max_importance'], reverse=True)[:10]:
            max_imp = importance_leaks[feat]['max_importance']
            print(f"   - {feat} (max importance: {max_imp:.1%})")
        print()
        print("   Consider adding these to CONFIG/excluded_features.yaml if they encode:")
        print("   - Future information (e.g., centered moving averages)")
        print("   - Barrier logic (e.g., barrier levels, hit probabilities)")
        print("   - Target-related information (e.g., peak/valley indicators)")
        print()
    
    if pattern_leaks:
        print("2. Pattern-matched features found in top features:")
        print("   These match exclusion patterns but are still appearing in feature lists.")
        print("   This suggests:")
        print("   - The filtering might not be working correctly")
        print("   - The patterns need to be more specific")
        print("   - These features were added after filtering")
        print()
    
    print("3. To add new exclusion patterns, edit CONFIG/excluded_features.yaml:")
    print("   - Add regex patterns to: always_exclude -> regex_patterns")
    print("   - Add prefix patterns to: always_exclude -> prefix_patterns")
    print("   - Add keyword patterns to: always_exclude -> keyword_patterns")
    print("   - Add target-specific patterns to: target_type_rules -> barrier -> keyword_patterns")
    print()
    print("4. Current filtering rules:")
    print("   - Horizon overlap: Exclude features with horizon >= target_horizon/4")
    print("   - Peak targets: Exclude HIGH/upper/max/top/ceiling features")
    print("   - Valley targets: Exclude LOW/lower/min/bottom/floor features")
    print("   - Matching horizon: Exclude features with same horizon as target")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Find leaking features from feature importance results"
    )
    parser.add_argument("output_dir", type=Path, help="Output directory with feature_importances/")
    parser.add_argument("--top-n", type=int, default=50,
                       help="Number of top features to analyze per model (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.50,
                       help="High importance threshold (default: 0.50 = 50%%)")
    
    args = parser.parse_args()
    
    analyze_importances(args.output_dir, args.top_n, args.threshold)


if __name__ == "__main__":
    main()

