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
Analyze feature importance results to identify leaking features.

Usage:
    python scripts/analyze_feature_importances.py <output_dir>
    
Example:
    python scripts/analyze_feature_importances.py results/target_rankings_updated
"""


import sys
import pandas as pd
import glob
from pathlib import Path
from collections import defaultdict

# Leaking feature patterns
LEAKING_PATTERNS = {
    'mfe_': 'Max Favorable Excursion (requires future path)',
    'mdd_': 'Max Drawdown (requires future path)',
    'fwd_ret_': 'Forward Return (overlapping with target)',
    'tth_': 'Time-to-Hit (knows when barrier hits)',
    'p_': 'Probability/Prediction feature (target-related)',
    'y_': 'Target column (should not be feature)'
}


def analyze_importances(output_dir: Path):
    """Analyze feature importance files and identify leaks."""
    importances_dir = output_dir / "feature_importances"
    
    if not importances_dir.exists():
        print(f"ERROR: Feature importances directory not found: {importances_dir}")
        return
    
    # Find all importances files
    files = list(importances_dir.glob("**/*_importances.csv"))
    
    if not files:
        print(f"ERROR: No feature importance files found in {importances_dir}")
        return
    
    print("="*80)
    print("LEAK DETECTION ANALYSIS REPORT")
    print("="*80)
    print(f"Found {len(files)} feature importance files")
    print()
    
    # Collect all leaking features
    leaking_features = {}
    
    for file in files:
        parts = file.parts
        # Extract: .../feature_importances/{target}/{symbol}/{model}_importances.csv
        target_idx = parts.index("feature_importances")
        target = parts[target_idx + 1]
        symbol = parts[target_idx + 2]
        model = parts[target_idx + 3].replace("_importances.csv", "")
        
        try:
            df = pd.read_csv(file)
            top20 = df.head(20)
            
            for _, row in top20.iterrows():
                feat = row['feature']
                pct = row['importance_pct']
                
                for pattern, description in LEAKING_PATTERNS.items():
                    if feat.startswith(pattern):
                        key = f'{target}|{symbol}|{model}|{feat}'
                        if key not in leaking_features:
                            leaking_features[key] = {
                                'target': target,
                                'symbol': symbol,
                                'model': model,
                                'feature': feat,
                                'importance_pct': pct,
                                'pattern': pattern,
                                'description': description
                            }
                        break
        except Exception as e:
            print(f"WARNING: Failed to process {file}: {e}")
            continue
    
    # Group by feature name
    feature_summary = defaultdict(lambda: {
        'pattern': None,
        'description': None,
        'occurrences': []
    })
    
    for leak in leaking_features.values():
        feat = leak['feature']
        if feature_summary[feat]['pattern'] is None:
            feature_summary[feat]['pattern'] = leak['pattern']
            feature_summary[feat]['description'] = leak['description']
        
        feature_summary[feat]['occurrences'].append({
            'target': leak['target'],
            'symbol': leak['symbol'],
            'model': leak['model'],
            'pct': leak['importance_pct']
        })
    
    # Print summary
    print("LEAKING FEATURES SUMMARY:")
    print("="*80)
    print()
    
    if not feature_summary:
        print("✓ No leaking features detected in top 20!")
        return
    
    # Sort by number of occurrences
    sorted_features = sorted(
        feature_summary.items(),
        key=lambda x: len(x[1]['occurrences']),
        reverse=True
    )
    
    for feat, info in sorted_features:
        print(f"{feat}")
        print(f"  Pattern: {info['pattern']}")
        print(f"  Description: {info['description']}")
        print(f"  Found in {len(info['occurrences'])} model/symbol combinations:")
        
        # Show top occurrences by importance
        top_occurrences = sorted(
            info['occurrences'],
            key=lambda x: x['pct'],
            reverse=True
        )[:5]
        
        for occ in top_occurrences:
            print(f"    - {occ['model']:15s} {occ['symbol']:6s} {occ['pct']:5.2f}%")
        
        if len(info['occurrences']) > 5:
            print(f"    ... and {len(info['occurrences']) - 5} more")
        print()
    
    # Generate recommendations
    print("="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print()
    
    # Check which patterns are missing
    found_patterns = set(info['pattern'] for info in feature_summary.values())
    
    if 'mfe_' in found_patterns or 'mdd_' in found_patterns:
        print("1. MFE/MDD features detected - ensure 'mfe_' and 'mdd_' are in")
        print("   CONFIG/excluded_features.yaml -> always_exclude -> prefix_patterns")
        print()
    
    if 'fwd_ret_' in found_patterns:
        print("2. Forward return features detected - add 'fwd_ret_' to:")
        print("   CONFIG/excluded_features.yaml -> target_type_rules -> barrier -> prefix_patterns")
        print()
    
    # Generate config additions
    print("3. Suggested additions to CONFIG/excluded_features.yaml:")
    print()
    
    missing_prefixes = []
    if 'mfe_' in found_patterns:
        missing_prefixes.append('    - "mfe_"')
    if 'mdd_' in found_patterns:
        missing_prefixes.append('    - "mdd_"')
    if 'fwd_ret_' in found_patterns:
        missing_prefixes.append('    - "fwd_ret_"  # Add to barrier target_type_rules')
    
    if missing_prefixes:
        print("   always_exclude -> prefix_patterns:")
        for prefix in missing_prefixes:
            print(prefix)
    
    print()
    print("4. After updating config, re-run the ranking script to verify leaks are filtered")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_feature_importances.py <output_dir>")
        print("Example: python scripts/analyze_feature_importances.py results/target_rankings_updated")
        sys.exit(1)
    
    output_dir = Path(sys.argv[1])
    analyze_importances(output_dir)


if __name__ == "__main__":
    main()

