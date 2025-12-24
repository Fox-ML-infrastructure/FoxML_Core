#!/usr/bin/env python3
"""
One-time migration script to quarantine scope-violated directories.

This script moves:
- CROSS_SECTIONAL/**/cohort=sy_* to QUARANTINE/
- SYMBOL_SPECIFIC/**/cohort=cs_* to QUARANTINE/

This prevents trend/diff analyzers from ingesting bad history after the
Patch 0 + Patch 3 fixes are applied.

Usage:
    python scripts/quarantine_scope_violations.py RESULTS/runs/your_run_dir [--dry-run]
    
Options:
    --dry-run    Show what would be moved without actually moving
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple


def find_scope_violations(output_dir: Path) -> Tuple[List[Path], List[Path]]:
    """
    Find all scope violations in an output directory.
    
    Returns:
        Tuple of (sy_under_cs, cs_under_sy) lists of paths
    """
    sy_under_cs = []
    cs_under_sy = []
    
    if not output_dir.exists():
        return sy_under_cs, cs_under_sy
    
    # Find all cohort directories
    for cohort_dir in output_dir.rglob("cohort=*"):
        if not cohort_dir.is_dir():
            continue
        
        cohort_id = cohort_dir.name.replace("cohort=", "")
        parts = cohort_dir.parts
        
        # Check for sy_* under CROSS_SECTIONAL
        if "CROSS_SECTIONAL" in parts and cohort_id.startswith("sy_"):
            sy_under_cs.append(cohort_dir)
        
        # Check for cs_* under SYMBOL_SPECIFIC
        if "SYMBOL_SPECIFIC" in parts and cohort_id.startswith("cs_"):
            cs_under_sy.append(cohort_dir)
    
    return sy_under_cs, cs_under_sy


def quarantine_violations(
    output_dir: Path, 
    violations: List[Path], 
    violation_type: str,
    dry_run: bool = True
) -> int:
    """
    Move violations to quarantine directory.
    
    Args:
        output_dir: Base output directory
        violations: List of paths to quarantine
        violation_type: "sy_under_cs" or "cs_under_sy" for naming
        dry_run: If True, only print what would be done
        
    Returns:
        Number of directories moved
    """
    if not violations:
        return 0
    
    # Create quarantine directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quarantine_dir = output_dir / "QUARANTINE" / f"{violation_type}_{timestamp}"
    
    if not dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
    
    moved = 0
    for violation_path in violations:
        # Create relative path for organization
        try:
            rel_path = violation_path.relative_to(output_dir)
        except ValueError:
            rel_path = Path(violation_path.name)
        
        dest_path = quarantine_dir / rel_path
        
        if dry_run:
            print(f"  [DRY RUN] Would move: {violation_path}")
            print(f"            -> {dest_path}")
        else:
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(violation_path), str(dest_path))
                print(f"  Moved: {violation_path}")
                print(f"      -> {dest_path}")
                moved += 1
            except Exception as e:
                print(f"  ERROR moving {violation_path}: {e}")
    
    return moved


def main():
    parser = argparse.ArgumentParser(
        description="Quarantine scope-violated directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to scan (e.g., RESULTS/runs/your_run_dir)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Show what would be moved without actually moving (default: True)"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files (overrides --dry-run)"
    )
    
    args = parser.parse_args()
    dry_run = not args.execute
    
    if not args.output_dir.exists():
        print(f"ERROR: Directory does not exist: {args.output_dir}")
        return 1
    
    print(f"Scanning for scope violations in: {args.output_dir}")
    if dry_run:
        print("  (DRY RUN - use --execute to actually move files)")
    print()
    
    sy_under_cs, cs_under_sy = find_scope_violations(args.output_dir)
    
    print(f"Found {len(sy_under_cs)} sy_* cohorts under CROSS_SECTIONAL/")
    print(f"Found {len(cs_under_sy)} cs_* cohorts under SYMBOL_SPECIFIC/")
    print()
    
    if not sy_under_cs and not cs_under_sy:
        print("No scope violations found!")
        return 0
    
    # Quarantine sy_* under CROSS_SECTIONAL
    if sy_under_cs:
        print("Quarantining sy_* cohorts from CROSS_SECTIONAL/:")
        moved = quarantine_violations(
            args.output_dir, sy_under_cs, "sy_under_cs", dry_run
        )
        if not dry_run:
            print(f"  Moved {moved} directories")
        print()
    
    # Quarantine cs_* under SYMBOL_SPECIFIC
    if cs_under_sy:
        print("Quarantining cs_* cohorts from SYMBOL_SPECIFIC/:")
        moved = quarantine_violations(
            args.output_dir, cs_under_sy, "cs_under_sy", dry_run
        )
        if not dry_run:
            print(f"  Moved {moved} directories")
        print()
    
    if dry_run:
        print("To actually move files, run with --execute flag")
    
    return 0


if __name__ == "__main__":
    exit(main())



