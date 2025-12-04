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
Remove specific targets from checkpoint to force re-evaluation.

Useful when:
- Filtering logic has changed and you want to re-evaluate targets
- Targets showed suspicious scores and you want to re-run with better filtering
- You want to clear results for specific targets without clearing the entire checkpoint

Usage:
    # Remove specific targets
    python scripts/remove_targets_from_checkpoint.py \
        --checkpoint results/target_rankings/checkpoint.json \
        --targets fwd_ret_60m,fwd_ret_120m,fwd_ret_oc_same_day
    
    # Remove all targets matching a pattern
    python scripts/remove_targets_from_checkpoint.py \
        --checkpoint results/target_rankings/checkpoint.json \
        --pattern "fwd_ret_.*"
    
    # List all targets in checkpoint
    python scripts/remove_targets_from_checkpoint.py \
        --checkpoint results/target_rankings/checkpoint.json \
        --list
"""


import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Set

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.utils.checkpoint import CheckpointManager


def list_targets(checkpoint_file: Path):
    """List all targets in checkpoint"""
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda x: x if isinstance(x, str) else x[0]
    )
    
    completed = checkpoint.load_completed()
    
    print(f"\nFound {len(completed)} targets in checkpoint:")
    print("=" * 80)
    
    for i, target_name in enumerate(sorted(completed.keys()), 1):
        result = completed[target_name]
        mean_r2 = result.get('mean_r2', 'N/A')
        leakage_flag = result.get('leakage_flag', 'N/A')
        print(f"{i:3d}. {target_name:40s} | R²={mean_r2:8.3f} | Flag={leakage_flag}")
    
    print("=" * 80)


def remove_targets(
    checkpoint_file: Path,
    target_names: List[str] = None,
    pattern: str = None,
    dry_run: bool = False
):
    """
    Remove specific targets from checkpoint.
    
    Args:
        checkpoint_file: Path to checkpoint file
        target_names: List of target names to remove
        pattern: Regex pattern to match targets to remove
        dry_run: If True, only show what would be removed without actually removing
    """
    checkpoint = CheckpointManager(
        checkpoint_file=checkpoint_file,
        item_key_fn=lambda x: x if isinstance(x, str) else x[0]
    )
    
    completed = checkpoint.load_completed()
    
    # Determine which targets to remove
    targets_to_remove = set()
    
    if target_names:
        for target in target_names:
            if target in completed:
                targets_to_remove.add(target)
            else:
                print(f"Warning: Target '{target}' not found in checkpoint")
    
    if pattern:
        regex = re.compile(pattern)
        for target_name in completed.keys():
            if regex.match(target_name):
                targets_to_remove.add(target_name)
    
    if not targets_to_remove:
        print("No targets to remove!")
        return
    
    # Show what will be removed
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Removing {len(targets_to_remove)} targets from checkpoint:")
    print("=" * 80)
    for target in sorted(targets_to_remove):
        result = completed[target]
        mean_r2 = result.get('mean_r2', 'N/A')
        leakage_flag = result.get('leakage_flag', 'N/A')
        print(f"  {target:40s} | R²={mean_r2:8.3f} | Flag={leakage_flag}")
    print("=" * 80)
    
    if dry_run:
        print("\n[DRY RUN] No changes made. Remove --dry-run to actually remove these targets.")
        return
    
    # Remove targets
    for target in targets_to_remove:
        # Remove from completed items
        if target in checkpoint._completed_items:
            del checkpoint._completed_items[target]
        # Remove from failed items if present
        checkpoint._failed_items.discard(target)
    
    # Save checkpoint
    checkpoint.save()
    
    print(f"\n✅ Removed {len(targets_to_remove)} targets from checkpoint")
    print(f"   Remaining: {len(checkpoint._completed_items)} targets")
    print(f"   Checkpoint saved to: {checkpoint_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove specific targets from checkpoint to force re-evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to checkpoint JSON file"
    )
    parser.add_argument(
        "--targets",
        type=str,
        help="Comma-separated list of target names to remove"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Regex pattern to match targets to remove (e.g., 'fwd_ret_.*')"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all targets in checkpoint"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without actually removing"
    )
    
    args = parser.parse_args()
    
    if not args.checkpoint.exists():
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    if args.list:
        list_targets(args.checkpoint)
        return 0
    
    if not args.targets and not args.pattern:
        print("Error: Must specify --targets, --pattern, or --list")
        return 1
    
    target_names = None
    if args.targets:
        target_names = [t.strip() for t in args.targets.split(',')]
    
    remove_targets(
        checkpoint_file=args.checkpoint,
        target_names=target_names,
        pattern=args.pattern,
        dry_run=args.dry_run
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

