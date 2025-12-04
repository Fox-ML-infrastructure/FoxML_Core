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
Helper script to create UPDATE directory entries

Usage:
    python scripts/utils/create_update_entry.py [description]
    python scripts/utils/create_update_entry.py --interactive
    python scripts/utils/create_update_entry.py "Fixed bug in target ranking"
"""


import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
import subprocess

# Add repo root to path
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

UPDATE_DIR = _REPO_ROOT / "UPDATE"


def get_modified_files():
    """Get list of modified files from git"""
    try:
        # Get modified files
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        modified = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        
        # Get untracked files
        result = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False
        )
        untracked = [f.strip() for f in result.stdout.splitlines() if f.strip()]
        
        return modified, untracked
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [], []


def create_update_entry(description: str, interactive: bool = False):
    """Create a new UPDATE entry"""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H-%M-%S")
    
    entry_dir = UPDATE_DIR / date_str / time_str
    entry_dir.mkdir(parents=True, exist_ok=True)
    
    # Get description
    if interactive and not description:
        print("Enter a brief description of the changes:")
        description = input("> ").strip()
        if not description:
            print("❌ Description required")
            return 1
    
    if not description:
        description = "Code changes"
    
    # Create CHANGES.md
    changes_file = entry_dir / "CHANGES.md"
    changes_content = f"""# {description}

**Date:** {date_str}  
**Time:** {time_str}

## Summary

{description}

## Changes Made

<!-- Describe the changes in detail -->

## Files Modified

<!-- List modified files -->

## Impact

<!-- Describe the impact of these changes -->

## Testing

<!-- Describe testing performed or needed -->
"""
    changes_file.write_text(changes_content)
    
    # Get modified files
    modified, untracked = get_modified_files()
    
    # Create files_changed.txt
    files_file = entry_dir / "files_changed.txt"
    if modified or untracked:
        files_content = "# Modified Files\n\n"
        if modified:
            files_content += "\n".join(modified) + "\n"
        if untracked:
            files_content += "\n# New Files\n\n"
            files_content += "\n".join(untracked) + "\n"
    else:
        files_content = "# Files changed (manual list)\n# Add files manually\n"
    
    files_file.write_text(files_content)
    
    print(f"✅ Created UPDATE entry: {entry_dir}")
    print(f"📝 Edit CHANGES.md to add details: {changes_file}")
    print(f"📋 Review files_changed.txt: {files_file}")
    
    # Optionally open in editor
    if interactive:
        editor = os.environ.get("EDITOR", "nano")
        try:
            subprocess.run([editor, str(changes_file)], check=False)
        except FileNotFoundError:
            print(f"⚠️  Editor '{editor}' not found. Edit manually: {changes_file}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Create UPDATE directory entry for tracking changes"
    )
    parser.add_argument(
        "description",
        nargs="?",
        help="Brief description of changes"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode (prompt for description and open editor)"
    )
    
    args = parser.parse_args()
    
    return create_update_entry(args.description, args.interactive)


if __name__ == "__main__":
    sys.exit(main())

