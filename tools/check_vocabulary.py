#!/usr/bin/env python3
"""
SST Vocabulary Linter - Ensures deprecated terms don't creep back into the codebase.

Returns exit code 0 if no deprecated terms found, 1 otherwise.

Usage:
    python tools/check_vocabulary.py [--fix]
    
    --fix: Show suggested replacements (does not auto-fix)
"""
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Deprecated term -> canonical replacement
DEPRECATED_TERMS: Dict[str, str] = {
    # Sample size
    "N_effective": "n_effective",
    "N_effective_cs": "n_effective",
    # Target
    "target_name": "target",
    "item_name": "target",
    # View/mode
    "resolved_mode": "view",
    "requested_mode": "requested_view",
    "mode_reason": "view_reason",
    "route_type": "view",
    "mode_policy": "view_policy",
    # Folds
    "n_folds": "folds",
    "cv_folds": "folds",
    # Seed
    "random_state": "seed",
    # Universe
    "universe_id": "universe_sig",
    # Output directory
    "out_dir": "output_dir",
    # Metrics
    "mean_score": "auc",
    "cs_auc": "auc",
    "cs_logloss": "logloss",
    "cs_pr_auc": "pr_auc",
    # Hashes
    "cs_config_hash": "config_hash",
    "featureset_fingerprint": "featureset_hash",
    # Dates
    "date_range_start": "date_start",
    "date_range_end": "date_end",
}

# Files/patterns to exclude from checking
EXCLUDE_PATTERNS = [
    "check_vocabulary.py",  # This file itself
    "VOCABULARY.md",        # Documentation of deprecated terms
    "__pycache__",
    ".pyc",
    "test_",                # Test files may reference deprecated terms for backward compat testing
]


def should_exclude(filepath: str) -> bool:
    """Check if file should be excluded from vocabulary checking."""
    for pattern in EXCLUDE_PATTERNS:
        if pattern in filepath:
            return True
    return False


def find_deprecated_terms(directory: Path) -> List[Tuple[str, int, str, str, str]]:
    """
    Find all deprecated terms in Python files.
    
    Returns list of (filepath, line_number, line_content, deprecated_term, replacement)
    """
    violations = []
    
    # Build regex pattern for all deprecated terms
    pattern = "|".join(DEPRECATED_TERMS.keys())
    
    try:
        result = subprocess.run(
            ["rg", "-n", "--type", "py", pattern, str(directory)],
            capture_output=True,
            text=True,
            cwd=directory.parent
        )
        
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
                
            # Parse ripgrep output: filepath:line_number:content
            parts = line.split(":", 2)
            if len(parts) < 3:
                continue
                
            filepath, line_num, content = parts[0], parts[1], parts[2]
            
            if should_exclude(filepath):
                continue
            
            # Find which deprecated term was matched
            for deprecated, replacement in DEPRECATED_TERMS.items():
                if deprecated in content:
                    violations.append((filepath, int(line_num), content.strip(), deprecated, replacement))
                    
    except FileNotFoundError:
        print("ERROR: ripgrep (rg) not found. Please install it.")
        sys.exit(2)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)
        
    return violations


def main():
    parser = argparse.ArgumentParser(description="Check for deprecated vocabulary terms")
    parser.add_argument("--fix", action="store_true", help="Show suggested replacements")
    parser.add_argument("--directory", type=Path, default=Path("TRAINING"), 
                        help="Directory to check (default: TRAINING)")
    args = parser.parse_args()
    
    # Resolve directory relative to script location
    script_dir = Path(__file__).parent.parent
    check_dir = script_dir / args.directory
    
    if not check_dir.exists():
        print(f"ERROR: Directory {check_dir} not found")
        sys.exit(2)
    
    print(f"Checking vocabulary in {check_dir}...")
    violations = find_deprecated_terms(check_dir)
    
    if not violations:
        print("OK: No deprecated terms found")
        sys.exit(0)
    
    # Group by file for cleaner output
    by_file: Dict[str, List[Tuple[int, str, str, str]]] = {}
    for filepath, line_num, content, deprecated, replacement in violations:
        if filepath not in by_file:
            by_file[filepath] = []
        by_file[filepath].append((line_num, content, deprecated, replacement))
    
    print(f"\nFOUND {len(violations)} deprecated term(s) in {len(by_file)} file(s):\n")
    
    for filepath, file_violations in sorted(by_file.items()):
        print(f"  {filepath}:")
        for line_num, content, deprecated, replacement in file_violations:
            print(f"    L{line_num}: '{deprecated}' -> '{replacement}'")
            if args.fix:
                print(f"           {content[:80]}...")
        print()
    
    print("See TRAINING/VOCABULARY.md for canonical names.")
    sys.exit(1)


if __name__ == "__main__":
    main()
