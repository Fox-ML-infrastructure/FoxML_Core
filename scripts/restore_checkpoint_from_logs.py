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
Restore Checkpoint from Log File

Parses ranking.md (or any log file) to extract completed target evaluations
and restores them into a checkpoint file so you can resume without re-running.

Usage:
    python scripts/restore_checkpoint_from_logs.py \
      --log-file scripts/ranking.md \
      --output-dir results/target_rankings \
      --script-name rank_target_predictability
"""


import argparse
import re
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add project root
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Don't import TargetPredictabilityScore - we'll just create dicts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_summary_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a summary line like: 'Summary: R²=0.219±0.207, importance=0.62, composite=0.555'"""
    # Match: Summary: R²=0.219±0.207, importance=0.62, composite=0.555
    # Also handle emoji prefixes like "📊 Summary:"
    pattern = r'(?:📊\s*)?Summary:\s*R²=([\d\.\-]+)±([\d\.]+),\s*importance=([\d\.]+),\s*composite=([\d\.]+)'
    match = re.search(pattern, line)
    if match:
        return {
            'mean_r2': float(match.group(1)),
            'std_r2': float(match.group(2)),
            'mean_importance': float(match.group(3)),
            'composite_score': float(match.group(4))
        }
    return None


def parse_scores_line(line: str) -> Optional[Dict[str, float]]:
    """Parse a scores line like: 'Scores: lightgbm=0.173, random_forest=0.121, ...'"""
    # Match: Scores: lightgbm=0.173, random_forest=0.121, ...
    # Also handle emoji prefixes like "✓ Scores:"
    pattern = r'(?:✓\s*)?Scores:\s*(.+)'
    match = re.search(pattern, line)
    if not match:
        return None
    
    scores_str = match.group(1)
    model_scores = {}
    
    # Parse each model=score pair (stop at 'importance=' if present)
    if 'importance=' in scores_str:
        scores_str = scores_str.split('importance=')[0].rstrip(',')
    
    # Parse each model=score pair
    for pair in scores_str.split(','):
        pair = pair.strip()
        if '=' in pair:
            parts = pair.split('=')
            if len(parts) == 2:
                model_name = parts[0].strip()
                try:
                    score = float(parts[1].strip())
                    model_scores[model_name] = score
                except ValueError:
                    continue
    
    return model_scores if model_scores else None


def parse_evaluating_line(line: str) -> Optional[Tuple[str, str]]:
    """Parse an evaluating line like: 'Evaluating: peak_60m_0.8 (y_will_peak_60m_0.8)'"""
    # Match: Evaluating: peak_60m_0.8 (y_will_peak_60m_0.8)
    pattern = r'Evaluating:\s*([^\s\(]+)\s*\(([^\)]+)\)'
    match = re.search(pattern, line)
    if match:
        return (match.group(1), match.group(2))
    return None


def extract_targets_from_log(log_file: Path) -> Dict[str, Dict]:
    """
    Extract completed target evaluations from log file.
    
    Returns:
        Dict mapping target_name -> TargetPredictabilityScore dict
    """
    logger.info(f"Parsing log file: {log_file}")
    
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        return {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    results = {}
    current_target = None
    current_target_col = None
    current_scores = {}
    current_summary = None
    
    for i, line in enumerate(lines):
        # Check for "Evaluating:" line
        eval_match = parse_evaluating_line(line)
        if eval_match:
            # Save previous target if we have data
            if current_target and current_summary:
                # Count models from scores
                n_models = len([s for s in current_scores.values() if not (isinstance(s, float) and (s != s or s == -999.0))])
                
                # Create result dict
                result_dict = {
                    'target_name': current_target,
                    'target_column': current_target_col or f'y_will_{current_target}',
                    'mean_r2': current_summary['mean_r2'],
                    'std_r2': current_summary['std_r2'],
                    'mean_importance': current_summary['mean_importance'],
                    'consistency': 1.0 - (current_summary['std_r2'] / (abs(current_summary['mean_r2']) + 1e-6)),
                    'n_models': n_models,
                    'model_scores': current_scores,
                    'composite_score': current_summary['composite_score'],
                    'leakage_flag': 'OK'  # Will be recalculated, but default to OK
                }
                
                # Detect leakage
                if current_summary['mean_r2'] > 0.70:
                    result_dict['leakage_flag'] = 'HIGH_R2'
                elif current_summary['composite_score'] > 0.5 and current_summary['mean_r2'] < 0.2:
                    result_dict['leakage_flag'] = 'INCONSISTENT'
                elif current_summary['mean_importance'] > 0.7 and current_summary['mean_r2'] < 0.1:
                    result_dict['leakage_flag'] = 'INCONSISTENT'
                
                results[current_target] = result_dict
                logger.info(f"Extracted: {current_target} (R²={current_summary['mean_r2']:.3f}, composite={current_summary['composite_score']:.3f})")
            
            # Start new target
            current_target, current_target_col = eval_match
            current_scores = {}
            current_summary = None
            continue
        
        # Check for "Scores:" line (per-symbol scores, we'll aggregate)
        scores_match = parse_scores_line(line)
        if scores_match and current_target:
            # Merge scores (keep latest or average - for now keep latest)
            current_scores.update(scores_match)
            continue
        
        # Check for "Summary:" line (final summary for target)
        summary_match = parse_summary_line(line)
        if summary_match and current_target:
            current_summary = summary_match
            continue
    
    # Don't forget the last target
    if current_target and current_summary:
        n_models = len([s for s in current_scores.values() if not (isinstance(s, float) and (s != s or s == -999.0))])
        result_dict = {
            'target_name': current_target,
            'target_column': current_target_col or f'y_will_{current_target}',
            'mean_r2': current_summary['mean_r2'],
            'std_r2': current_summary['std_r2'],
            'mean_importance': current_summary['mean_importance'],
            'consistency': 1.0 - (current_summary['std_r2'] / (abs(current_summary['mean_r2']) + 1e-6)),
            'n_models': n_models,
            'model_scores': current_scores,
            'composite_score': current_summary['composite_score'],
            'leakage_flag': 'OK'
        }
        
        # Detect leakage
        if current_summary['mean_r2'] > 0.70:
            result_dict['leakage_flag'] = 'HIGH_R2'
        elif current_summary['composite_score'] > 0.5 and current_summary['mean_r2'] < 0.2:
            result_dict['leakage_flag'] = 'INCONSISTENT'
        elif current_summary['mean_importance'] > 0.7 and current_summary['mean_r2'] < 0.1:
            result_dict['leakage_flag'] = 'INCONSISTENT'
        
        results[current_target] = result_dict
        logger.info(f"Extracted: {current_target} (R²={current_summary['mean_r2']:.3f}, composite={current_summary['composite_score']:.3f})")
    
    return results


def restore_checkpoint(
    log_file: Path,
    output_dir: Path,
    script_name: str = "rank_target_predictability"
):
    """Restore checkpoint from log file"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_file = output_dir / "checkpoint.json"
    
    # Extract results from log
    extracted_results = extract_targets_from_log(log_file)
    
    if not extracted_results:
        logger.error("No results extracted from log file!")
        return
    
    logger.info(f"\nExtracted {len(extracted_results)} completed targets from log")
    
    # Create checkpoint structure
    checkpoint_data = {
        'completed_items': extracted_results,
        'failed_items': [],
        'metadata': {
            'restored_from': str(log_file),
            'n_completed': len(extracted_results),
            'restore_timestamp': time.time()
        }
    }
    
    # Save checkpoint
    temp_file = checkpoint_file.with_suffix('.tmp')
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    
    temp_file.replace(checkpoint_file)
    
    logger.info(f"\nCheckpoint restored to: {checkpoint_file}")
    logger.info(f"Completed targets: {len(extracted_results)}")
    logger.info("\nYou can now resume with:")
    logger.info(f"  python scripts/{script_name}.py --resume --output-dir {output_dir}")
    
    # Also save as CSV for easy viewing
    try:
        import pandas as pd
        df_data = []
        for target_name, result in extracted_results.items():
            df_data.append({
                'target_name': result['target_name'],
                'target_column': result['target_column'],
                'composite_score': result['composite_score'],
                'mean_r2': result['mean_r2'],
                'std_r2': result['std_r2'],
                'mean_importance': result['mean_importance'],
                'n_models': result['n_models'],
                'leakage_flag': result['leakage_flag']
            })
        
        df = pd.DataFrame(df_data)
        df = df.sort_values('composite_score', ascending=False)
        csv_file = output_dir / "restored_rankings.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Also saved CSV summary to: {csv_file}")
    except ImportError:
        # Save as JSON instead if pandas not available
        summary_file = output_dir / "restored_rankings_summary.json"
        summary_data = [
            {
                'target_name': r['target_name'],
                'composite_score': r['composite_score'],
                'mean_r2': r['mean_r2'],
                'mean_importance': r['mean_importance'],
                'n_models': r['n_models'],
                'leakage_flag': r['leakage_flag']
            }
            for r in sorted(extracted_results.values(), key=lambda x: x['composite_score'], reverse=True)
        ]
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        logger.info(f"Also saved JSON summary to: {summary_file} (pandas not available for CSV)")


def main():
    parser = argparse.ArgumentParser(
        description="Restore checkpoint from log file"
    )
    parser.add_argument(
        '--log-file',
        type=Path,
        default=Path('scripts/ranking.md'),
        help='Log file to parse (default: scripts/ranking.md)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/target_rankings'),
        help='Output directory for checkpoint (default: results/target_rankings)'
    )
    parser.add_argument(
        '--script-name',
        type=str,
        default='rank_target_predictability',
        help='Script name for resume command (default: rank_target_predictability)'
    )
    
    args = parser.parse_args()
    
    restore_checkpoint(args.log_file, args.output_dir, args.script_name)


if __name__ == '__main__':
    main()

