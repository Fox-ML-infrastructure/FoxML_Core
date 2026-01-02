#!/usr/bin/env python3
"""
Golden Diff Checker

Compares current demo output against the baseline to detect unintended behavior changes.

Usage:
    python tools/check_golden_diff.py              # Compare against baseline
    python tools/check_golden_diff.py --verbose    # Show detailed diffs
    python tools/check_golden_diff.py --update     # Update baseline with current

Exit codes:
    0 = Match (or no baseline exists)
    1 = Differences found
    2 = Error
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = PROJECT_ROOT / "demo" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "RESULTS" / "demo_run"

# Fields to skip when comparing (these change between runs)
SKIP_FIELDS: Set[str] = {
    "created_at",
    "timestamp", 
    "elapsed_seconds",
    "git_commit",
    "run_timestamp",
    "start_time",
    "end_time",
    "wall_clock_time",
    "runtime_sec",
}

# Files to compare (relative to output directory)
COMPARE_FILES = [
    "manifest.json",
    "globals/run_context.json",
]


def load_json_safe(path: Path) -> Optional[Dict]:
    """Load JSON file, return None if not found or invalid."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def compare_values(
    baseline: Any, 
    current: Any, 
    path: str = "",
    skip_fields: Set[str] = SKIP_FIELDS
) -> List[str]:
    """
    Recursively compare two values, returning list of differences.
    
    Args:
        baseline: Expected value
        current: Actual value
        path: Current path for error messages
        skip_fields: Field names to skip
        
    Returns:
        List of difference descriptions
    """
    diffs = []
    
    # Handle None cases
    if baseline is None and current is None:
        return []
    if baseline is None:
        return [f"{path}: baseline is None, current is {type(current).__name__}"]
    if current is None:
        return [f"{path}: current is None, baseline is {type(baseline).__name__}"]
    
    # Type mismatch
    if type(baseline) != type(current):
        return [f"{path}: type mismatch ({type(baseline).__name__} vs {type(current).__name__})"]
    
    # Dict comparison
    if isinstance(baseline, dict):
        all_keys = set(baseline.keys()) | set(current.keys())
        
        for key in sorted(all_keys):
            if key in skip_fields:
                continue
                
            child_path = f"{path}.{key}" if path else key
            
            if key not in baseline:
                diffs.append(f"{child_path}: new field in current (value: {current[key]})")
            elif key not in current:
                diffs.append(f"{child_path}: missing in current (was: {baseline[key]})")
            else:
                diffs.extend(compare_values(baseline[key], current[key], child_path, skip_fields))
        
        return diffs
    
    # List comparison
    if isinstance(baseline, list):
        if len(baseline) != len(current):
            diffs.append(f"{path}: length mismatch ({len(baseline)} vs {len(current)})")
        
        for i, (b, c) in enumerate(zip(baseline, current)):
            child_path = f"{path}[{i}]"
            diffs.extend(compare_values(b, c, child_path, skip_fields))
        
        return diffs
    
    # Primitive comparison
    if baseline != current:
        return [f"{path}: {baseline!r} != {current!r}"]
    
    return []


def check_structure() -> List[str]:
    """Compare directory structures."""
    diffs = []
    
    baseline_structure = BASELINE_DIR / "structure.json"
    if not baseline_structure.exists():
        return []
    
    with open(baseline_structure) as f:
        baseline = json.load(f)
    
    baseline_files = set(baseline.get('files', []))
    
    current_files = set()
    if RESULTS_DIR.exists():
        for path in RESULTS_DIR.rglob("*"):
            if path.is_file():
                current_files.add(str(path.relative_to(RESULTS_DIR)))
    
    missing = baseline_files - current_files
    extra = current_files - baseline_files
    
    if missing:
        diffs.append(f"Missing files: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if extra:
        diffs.append(f"Extra files: {sorted(extra)[:5]}{'...' if len(extra) > 5 else ''}")
    
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Compare demo output against baseline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed diffs")
    parser.add_argument("--update", action="store_true", help="Update baseline with current output")
    args = parser.parse_args()
    
    # Update mode
    if args.update:
        print("Updating baseline...")
        sys.path.insert(0, str(PROJECT_ROOT))
        from demo.run_demo import save_baseline
        save_baseline()
        return 0
    
    # Check mode
    if not BASELINE_DIR.exists():
        print("No baseline exists. Run: python demo/run_demo.py --save-baseline")
        return 0  # Not an error, just no baseline yet
    
    if not RESULTS_DIR.exists():
        print("No demo output exists. Run: python demo/run_demo.py")
        return 2
    
    all_diffs = []
    
    # Compare key files
    for rel_path in COMPARE_FILES:
        baseline_file = BASELINE_DIR / rel_path
        current_file = RESULTS_DIR / rel_path
        
        baseline_data = load_json_safe(baseline_file)
        current_data = load_json_safe(current_file)
        
        if baseline_data is None and current_data is None:
            continue
        elif baseline_data is None:
            all_diffs.append(f"{rel_path}: not in baseline but exists in current")
        elif current_data is None:
            all_diffs.append(f"{rel_path}: in baseline but missing in current")
        else:
            file_diffs = compare_values(baseline_data, current_data, rel_path)
            all_diffs.extend(file_diffs)
    
    # Compare structure
    structure_diffs = check_structure()
    all_diffs.extend(structure_diffs)
    
    # Report
    if all_diffs:
        print(f"DIFFERENCES FOUND ({len(all_diffs)}):\n")
        for diff in all_diffs[:20]:  # Limit output
            print(f"  - {diff}")
        if len(all_diffs) > 20:
            print(f"  ... and {len(all_diffs) - 20} more")
        print("\nIf these changes are intentional, update baseline:")
        print("  python tools/check_golden_diff.py --update")
        return 1
    else:
        print("OK: Demo output matches baseline")
        return 0


if __name__ == "__main__":
    sys.exit(main())
