#!/usr/bin/env python3
"""
Quick test script for Phase 2: Feature Selection
Runs feature selection for a target without running target ranking.

Usage:
    python SCRIPTS/test_phase2_feature_selection.py \
        --target fwd_ret_60m \
        --symbols AAPL MSFT \
        --data-dir "data/data_labeled/interval=5m" \
        --top-n 30 \
        --output-dir "test_phase2_output"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from TRAINING.ranking.feature_selector import select_features_for_target

def main():
    parser = argparse.ArgumentParser(
        description="Test Phase 2: Feature Selection (standalone, no target ranking)"
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target column name (e.g., fwd_ret_60m, y_will_peak_60m_0.8)"
    )
    parser.add_argument(
        "--symbols",
        type=str,
        nargs="+",
        required=True,
        help="Symbols to process (e.g., AAPL MSFT)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing symbol data (e.g., data/data_labeled/interval=5m)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top features to select (default: 30)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results (default: test_phase2_output)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000,
        help="Maximum samples per symbol (default: 5000, use lower for faster testing)"
    )
    parser.add_argument(
        "--enable-families",
        type=str,
        default=None,
        help="Comma-separated model families to enable (e.g., lightgbm,xgboost). Default: all enabled in config"
    )
    
    args = parser.parse_args()
    
    # Set default output dir
    if args.output_dir is None:
        args.output_dir = Path("test_phase2_output")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and optionally filter model families
    from TRAINING.ranking.feature_selector import load_multi_model_config
    multi_model_config = load_multi_model_config()
    
    if args.enable_families:
        enabled_families = [f.strip() for f in args.enable_families.split(',')]
        # Disable all families first
        for family_name in multi_model_config.get('model_families', {}):
            if family_name in multi_model_config['model_families']:
                multi_model_config['model_families'][family_name]['enabled'] = False
        # Enable only specified families
        for family_name in enabled_families:
            if family_name in multi_model_config.get('model_families', {}):
                multi_model_config['model_families'][family_name]['enabled'] = True
                print(f"✅ Enabled model family: {family_name}")
    
    print(f"\n{'='*80}")
    print(f"🧪 Testing Phase 2: Feature Selection")
    print(f"{'='*80}")
    print(f"Target: {args.target}")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Data dir: {args.data_dir}")
    print(f"Top N: {args.top_n}")
    print(f"Max samples per symbol: {args.max_samples}")
    print(f"Output dir: {args.output_dir}")
    print(f"{'='*80}\n")
    
    try:
        # Run feature selection
        selected_features, importance_df = select_features_for_target(
            target_column=args.target,
            symbols=args.symbols,
            data_dir=args.data_dir,
            multi_model_config=multi_model_config,
            max_samples_per_symbol=args.max_samples,
            top_n=args.top_n,
            output_dir=args.output_dir
        )
        
        print(f"\n{'='*80}")
        print(f"✅ Feature Selection Complete!")
        print(f"{'='*80}")
        print(f"Selected {len(selected_features)} features:")
        for i, feat in enumerate(selected_features[:20], 1):  # Show top 20
            print(f"  {i:2d}. {feat}")
        if len(selected_features) > 20:
            print(f"  ... and {len(selected_features) - 20} more")
        
        print(f"\nImportance summary saved to: {args.output_dir}")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Feature selection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

