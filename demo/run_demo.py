#!/usr/bin/env python3
"""
Demo Run Script with Baseline Capture

Wraps tools/demo.py with additional baseline management for refactoring safety.

Usage:
    python demo/run_demo.py                    # Run demo
    python demo/run_demo.py --save-baseline    # Run and save as golden baseline
    python demo/run_demo.py --check-baseline   # Run and compare against baseline
    python demo/run_demo.py --show-baseline    # Show current baseline info
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Paths
DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
BASELINE_DIR = DEMO_DIR / "baseline"
RESULTS_DIR = PROJECT_ROOT / "RESULTS" / "demo_run"

# Files to capture for baseline
BASELINE_FILES = [
    "manifest.json",
    "globals/run_context.json",
]


def run_demo() -> int:
    """Run the demo pipeline using tools/demo.py."""
    sys.path.insert(0, str(PROJECT_ROOT))
    
    # Import and run the existing demo
    from tools.demo import run_demo as _run_demo, clean_demo_output
    
    clean_demo_output()
    return _run_demo()


def save_baseline():
    """Save current demo output as the golden baseline."""
    print(f"Saving baseline to {BASELINE_DIR}")
    
    # Ensure baseline dir exists
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy key files
    files_saved = []
    for rel_path in BASELINE_FILES:
        src = RESULTS_DIR / rel_path
        if src.exists():
            dst = BASELINE_DIR / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            files_saved.append(rel_path)
            print(f"  Saved: {rel_path}")
        else:
            print(f"  Skipped (not found): {rel_path}")
    
    # Also copy directory structure listing
    structure = list_directory_structure(RESULTS_DIR)
    structure_file = BASELINE_DIR / "structure.json"
    with open(structure_file, 'w') as f:
        json.dump(structure, f, indent=2)
    print(f"  Saved: structure.json ({len(structure['files'])} files)")
    
    # Record metadata
    metadata = {
        "created_at": datetime.now().isoformat(),
        "files_saved": files_saved,
        "total_files": len(structure['files']),
    }
    with open(BASELINE_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nBaseline saved with {len(files_saved)} key files")


def list_directory_structure(directory: Path) -> dict:
    """List directory structure for comparison."""
    files = []
    if directory.exists():
        for path in directory.rglob("*"):
            if path.is_file():
                rel = path.relative_to(directory)
                files.append(str(rel))
    return {
        "root": str(directory),
        "files": sorted(files)
    }


def check_baseline() -> int:
    """Compare current demo output against baseline."""
    print(f"Checking against baseline in {BASELINE_DIR}")
    
    if not BASELINE_DIR.exists():
        print("ERROR: No baseline exists. Run with --save-baseline first.")
        return 1
    
    differences = []
    
    # Compare structure
    baseline_structure_file = BASELINE_DIR / "structure.json"
    if baseline_structure_file.exists():
        with open(baseline_structure_file) as f:
            baseline_structure = json.load(f)
        
        current_structure = list_directory_structure(RESULTS_DIR)
        
        baseline_files = set(baseline_structure['files'])
        current_files = set(current_structure['files'])
        
        missing = baseline_files - current_files
        extra = current_files - baseline_files
        
        if missing:
            differences.append(f"Missing files: {missing}")
        if extra:
            differences.append(f"Extra files: {extra}")
    
    # Compare key files content
    for rel_path in BASELINE_FILES:
        baseline_file = BASELINE_DIR / rel_path
        current_file = RESULTS_DIR / rel_path
        
        if not baseline_file.exists():
            continue
        
        if not current_file.exists():
            differences.append(f"Missing: {rel_path}")
            continue
        
        with open(baseline_file) as f:
            baseline_data = json.load(f)
        with open(current_file) as f:
            current_data = json.load(f)
        
        # Compare key fields (not timestamps)
        skip_fields = {'created_at', 'timestamp', 'elapsed_seconds', 'git_commit'}
        
        diff = compare_dicts(baseline_data, current_data, skip_fields)
        if diff:
            differences.append(f"{rel_path}: {diff}")
    
    if differences:
        print("\nDIFFERENCES FOUND:")
        for d in differences:
            print(f"  - {d}")
        return 1
    else:
        print("\nOK: Demo output matches baseline")
        return 0


def compare_dicts(d1: dict, d2: dict, skip_fields: set, path: str = "") -> list:
    """Compare two dicts, returning list of differences."""
    diffs = []
    
    all_keys = set(d1.keys()) | set(d2.keys())
    
    for key in all_keys:
        if key in skip_fields:
            continue
        
        full_path = f"{path}.{key}" if path else key
        
        if key not in d1:
            diffs.append(f"{full_path}: missing in baseline")
        elif key not in d2:
            diffs.append(f"{full_path}: missing in current")
        elif type(d1[key]) != type(d2[key]):
            diffs.append(f"{full_path}: type mismatch")
        elif isinstance(d1[key], dict):
            diffs.extend(compare_dicts(d1[key], d2[key], skip_fields, full_path))
        elif d1[key] != d2[key]:
            diffs.append(f"{full_path}: {d1[key]} != {d2[key]}")
    
    return diffs


def show_baseline():
    """Show information about current baseline."""
    metadata_file = BASELINE_DIR / "metadata.json"
    
    if not metadata_file.exists():
        print("No baseline exists.")
        return
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    print(f"Baseline created: {metadata.get('created_at', 'unknown')}")
    print(f"Files saved: {metadata.get('files_saved', [])}")
    print(f"Total output files: {metadata.get('total_files', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="Run demo with baseline management")
    parser.add_argument("--save-baseline", action="store_true", 
                        help="Run demo and save as golden baseline")
    parser.add_argument("--check-baseline", action="store_true",
                        help="Run demo and compare against baseline")
    parser.add_argument("--show-baseline", action="store_true",
                        help="Show current baseline info")
    args = parser.parse_args()
    
    if args.show_baseline:
        show_baseline()
        return 0
    
    if args.save_baseline:
        exit_code = run_demo()
        if exit_code == 0:
            save_baseline()
        return exit_code
    
    if args.check_baseline:
        exit_code = run_demo()
        if exit_code == 0:
            return check_baseline()
        return exit_code
    
    # Default: just run demo
    return run_demo()


if __name__ == "__main__":
    sys.exit(main())
