#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""
Add copyright headers to files missing them.

Scans the repository for Python and shell script files and adds the standard
SPDX-License-Identifier header if missing.
"""

import os
import sys
from pathlib import Path
import re

# Standard header for Python files
PYTHON_HEADER = """# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""

# Standard header for shell scripts
SHELL_HEADER = """#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
# Copyright (c) 2025-2026 Fox ML Infrastructure LLC

"""

# Directories to exclude
EXCLUDE_DIRS = {
    '.git', '__pycache__', '.pytest_cache', '.mypy_cache', '.cursor',
    'RESULTS', 'dep', 'IBKR_trading', 'INTERNAL_DOCS', 'INTERNAL_EXPERIMENTAL',
    'tests', 'demo', 'SCRIPTS', 'archive', 'tools', 'node_modules', '.venv', 'venv'
}

# Files to exclude
EXCLUDE_FILES = {
    # '__init__.py',  # Include these - they need headers too
}

def has_copyright_header(file_path: Path) -> bool:
    """Check if file already has copyright header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = f.read(500)  # Read first 500 chars
            return (
                'SPDX-License-Identifier' in first_lines or
                'Copyright' in first_lines and 'Fox ML Infrastructure' in first_lines
            )
    except Exception:
        return False

def should_process_file(file_path: Path) -> bool:
    """Check if file should be processed."""
    # Check if in excluded directory
    parts = file_path.parts
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
    
    # Check if excluded file
    if file_path.name in EXCLUDE_FILES:
        return False
    
    # Only process .py and .sh files
    if file_path.suffix not in {'.py', '.sh'}:
        return False
    
    return True

def add_header_to_file(file_path: Path, dry_run: bool = False) -> bool:
    """Add copyright header to a file."""
    if not should_process_file(file_path):
        return False
    
    if has_copyright_header(file_path):
        return False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return False
    
    # Determine header based on file type
    if file_path.suffix == '.py':
        header = PYTHON_HEADER
        # Check if file starts with shebang
        if content.startswith('#!/'):
            # Find end of shebang line
            shebang_end = content.find('\n') + 1
            # Check if there's already content after shebang
            after_shebang = content[shebang_end:].lstrip()
            if after_shebang.startswith('#'):
                # Already has comments, insert after first blank line or after comments
                new_content = content[:shebang_end] + '\n' + PYTHON_HEADER + after_shebang
            else:
                new_content = content[:shebang_end] + '\n' + PYTHON_HEADER + content[shebang_end:]
        elif content.strip().startswith('"""') or content.strip().startswith("'''"):
            # File starts with docstring, insert header before it
            new_content = header + content
        else:
            new_content = header + content
    elif file_path.suffix == '.sh':
        # Shell scripts should have shebang, but check
        if content.startswith('#!/'):
            # Replace or add after shebang
            shebang_end = content.find('\n') + 1
            # Check if already has SPDX after shebang
            after_shebang = content[shebang_end:shebang_end+100]
            if 'SPDX-License-Identifier' in after_shebang:
                return False
            new_content = content[:shebang_end] + SHELL_HEADER.split('\n', 1)[1] + content[shebang_end:]
        else:
            new_content = SHELL_HEADER + content
    else:
        return False
    
    if not dry_run:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except Exception as e:
            print(f"Error writing {file_path}: {e}", file=sys.stderr)
            return False
    else:
        return True

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Add copyright headers to files missing them'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without making changes'
    )
    parser.add_argument(
        '--root',
        type=str,
        default='.',
        help='Root directory to scan (default: current directory)'
    )
    args = parser.parse_args()
    
    root = Path(args.root).resolve()
    if not root.exists():
        print(f"Error: {root} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Find all Python and shell script files
    files_to_process = []
    for ext in ['*.py', '*.sh']:
        for file_path in root.rglob(ext):
            if should_process_file(file_path) and not has_copyright_header(file_path):
                files_to_process.append(file_path)
    
    if not files_to_process:
        print("No files need copyright headers.")
        return
    
    print(f"Found {len(files_to_process)} files missing copyright headers:")
    for f in sorted(files_to_process):
        print(f"  {f}")
    
    if args.dry_run:
        print("\n[DRY RUN] No changes made. Run without --dry-run to apply changes.")
        return
    
    print(f"\nAdding headers to {len(files_to_process)} files...")
    success_count = 0
    for file_path in sorted(files_to_process):
        if add_header_to_file(file_path, dry_run=False):
            print(f"  ✓ {file_path}")
            success_count += 1
        else:
            print(f"  ✗ {file_path} (failed or skipped)")
    
    print(f"\nDone. Added headers to {success_count} files.")

if __name__ == '__main__':
    main()
