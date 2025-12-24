#!/usr/bin/env python3
"""
Quick check for undefined names (missing imports) using Ruff.

Usage:
    python TRAINING/tools/check_undefined_names.py
    python TRAINING/tools/check_undefined_names.py --fix  # Auto-fix what can be fixed
    python TRAINING/tools/check_undefined_names.py --all   # Check entire repo, not just TRAINING
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = PROJECT_ROOT / "TRAINING"


def main():
    """Run ruff F821 check for undefined names."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Check for undefined names (missing imports) using Ruff F821"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix issues where possible (ruff --fix)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check entire repo, not just TRAINING directory"
    )
    args = parser.parse_args()

    target = "." if args.all else str(TRAINING_DIR)
    cmd = ["ruff", "check", target, "--select", "F821"]
    
    if args.fix:
        cmd.append("--fix")

    print(f"🔍 Checking for undefined names (F821) in {target}...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    
    if result.returncode == 0:
        print("✅ No undefined names found!")
        return 0
    else:
        print("\n❌ Found undefined names. Fix imports or define variables.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
