#!/bin/bash
# Helper script to create UPDATE directory entries
# Usage: ./SCRIPTS/utils/create_update_entry.sh [description]

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UPDATE_DIR="$REPO_ROOT/UPDATE"
DATE=$(date +%Y-%m-%d)
TIME=$(date +%H-%M-%S)
ENTRY_DIR="$UPDATE_DIR/$DATE/$TIME"

# Create directory
mkdir -p "$ENTRY_DIR"

# Get description from argument or prompt
if [ -n "$1" ]; then
    DESCRIPTION="$1"
else
    echo "Enter a brief description of the changes:"
    read -r DESCRIPTION
fi

# Create CHANGES.md
cat > "$ENTRY_DIR/CHANGES.md" << EOF
# $DESCRIPTION

**Date:** $DATE  
**Time:** $TIME

## Summary

$DESCRIPTION

## Changes Made

<!-- Describe the changes in detail -->

## Files Modified

<!-- List modified files -->

## Impact

<!-- Describe the impact of these changes -->

## Testing

<!-- Describe testing performed or needed -->
EOF

# Try to detect modified files (if in git repo)
if git -C "$REPO_ROOT" rev-parse --git-dir > /dev/null 2>&1; then
    # Get modified files from git
    MODIFIED_FILES=$(git -C "$REPO_ROOT" diff --name-only HEAD 2>/dev/null || echo "")
    UNTRACKED_FILES=$(git -C "$REPO_ROOT" ls-files --others --exclude-standard 2>/dev/null || echo "")
    
    if [ -n "$MODIFIED_FILES" ] || [ -n "$UNTRACKED_FILES" ]; then
        {
            echo "# Modified Files"
            echo ""
            if [ -n "$MODIFIED_FILES" ]; then
                echo "$MODIFIED_FILES"
            fi
            if [ -n "$UNTRACKED_FILES" ]; then
                echo ""
                echo "# New Files"
                echo "$UNTRACKED_FILES"
            fi
        } > "$ENTRY_DIR/files_changed.txt"
    else
        echo "# No modified files detected" > "$ENTRY_DIR/files_changed.txt"
    fi
else
    echo "# Files changed (manual list)" > "$ENTRY_DIR/files_changed.txt"
    echo "# Add files manually" >> "$ENTRY_DIR/files_changed.txt"
fi

echo "✅ Created UPDATE entry: $ENTRY_DIR"
echo "📝 Edit CHANGES.md to add details: $ENTRY_DIR/CHANGES.md"
echo "📋 Review files_changed.txt: $ENTRY_DIR/files_changed.txt"

