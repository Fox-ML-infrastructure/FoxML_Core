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
Compare Feature Sets

Compare features from different selection methods:
- Single-model (LightGBM only)
- Multi-model (consensus across families)
- Different targets
- Different configurations

Shows:
- Overlap/unique features
- Agreement analysis
- Performance comparison (if evaluation data provided)
"""


import argparse
import sys
from pathlib import Path
from typing import List, Set, Dict
import pandas as pd
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def load_feature_list(file_path: Path) -> List[str]:
    """Load feature list from file"""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if file_path.suffix == '.txt':
        with open(file_path) as f:
            return [line.strip() for line in f if line.strip()]
    elif file_path.suffix == '.csv':
        df = pd.read_parquet(file_path)
        return df['feature'].tolist() if 'feature' in df.columns else df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")


def compare_sets(set1: Set[str], set2: Set[str], name1: str, name2: str):
    """Compare two feature sets"""
    overlap = set1 & set2
    only_set1 = set1 - set2
    only_set2 = set2 - set1
    
    print(f"\n{'='*80}")
    print(f"Comparison: {name1} vs {name2}")
    print(f"{'='*80}")
    print(f"{name1}: {len(set1)} features")
    print(f"{name2}: {len(set2)} features")
    print(f"Overlap: {len(overlap)} features ({len(overlap)/len(set1)*100:.1f}% of {name1})")
    print(f"Only in {name1}: {len(only_set1)} features")
    print(f"Only in {name2}: {len(only_set2)} features")
    
    if overlap:
        print(f"\n✅ Common features (top 10):")
        for i, feature in enumerate(list(overlap)[:10], 1):
            print(f"  {i:2d}. {feature}")
    
    if only_set1:
        print(f"\n🔵 Only in {name1} (top 10):")
        for i, feature in enumerate(list(only_set1)[:10], 1):
            print(f"  {i:2d}. {feature}")
    
    if only_set2:
        print(f"\n🟢 Only in {name2} (top 10):")
        for i, feature in enumerate(list(only_set2)[:10], 1):
            print(f"  {i:2d}. {feature}")
    
    return {
        'overlap': overlap,
        'only_set1': only_set1,
        'only_set2': only_set2,
        'jaccard': len(overlap) / len(set1 | set2) if (set1 | set2) else 0.0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare feature sets from different selection methods"
    )
    parser.add_argument("--set1", type=Path, required=True,
                       help="First feature set (txt or csv)")
    parser.add_argument("--set2", type=Path, required=True,
                       help="Second feature set (txt or csv)")
    parser.add_argument("--name1", type=str, default="Set 1",
                       help="Name for first set")
    parser.add_argument("--name2", type=str, default="Set 2",
                       help="Name for second set")
    parser.add_argument("--output", type=Path,
                       help="Save comparison to CSV")
    
    args = parser.parse_args()
    
    # Load features
    print(f"\nLoading feature sets...")
    features1 = set(load_feature_list(args.set1))
    features2 = set(load_feature_list(args.set2))
    
    print(f"✅ Loaded {len(features1)} features from {args.set1.name}")
    print(f"✅ Loaded {len(features2)} features from {args.set2.name}")
    
    # Compare
    results = compare_sets(features1, features2, args.name1, args.name2)
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Jaccard similarity: {results['jaccard']:.3f}")
    print(f"Overlap rate: {len(results['overlap'])/max(len(features1), len(features2))*100:.1f}%")
    
    if results['jaccard'] > 0.7:
        print("✅ High agreement - methods are consistent")
    elif results['jaccard'] > 0.4:
        print("⚠️  Moderate agreement - some differences")
    else:
        print("❌ Low agreement - methods are quite different")
    
    # Save if requested
    if args.output:
        df = pd.DataFrame({
            'feature': sorted(features1 | features2),
            'in_set1': [f in features1 for f in sorted(features1 | features2)],
            'in_set2': [f in features2 for f in sorted(features1 | features2)],
            'in_both': [f in results['overlap'] for f in sorted(features1 | features2)]
        })
        df.to_csv(args.output, index=False)
        print(f"\n✅ Saved comparison to {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

