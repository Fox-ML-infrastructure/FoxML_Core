#!/usr/bin/env python3
"""
Find hardcoded config file paths that need updating after migration.

This script searches for direct references to old config paths and suggests
updates to use the new structure or config loaders.
"""

import re
from pathlib import Path
from typing import List, Tuple

# Patterns to search for
OLD_PATTERNS = [
    (r'CONFIG/excluded_features\.yaml', 'CONFIG/data/excluded_features.yaml'),
    (r'CONFIG/feature_registry\.yaml', 'CONFIG/data/feature_registry.yaml'),
    (r'CONFIG/feature_target_schema\.yaml', 'CONFIG/data/feature_target_schema.yaml'),
    (r'CONFIG/feature_groups\.yaml', 'CONFIG/data/feature_groups.yaml'),
    (r'CONFIG/logging_config\.yaml', 'CONFIG/core/logging.yaml'),
    (r'CONFIG/training_config/', 'CONFIG/pipeline/training/ or CONFIG/pipeline/'),
    (r'CONFIG/model_config/', 'CONFIG/models/'),
    (r'CONFIG/feature_selection/', 'CONFIG/ranking/features/'),
    (r'CONFIG/target_ranking/', 'CONFIG/ranking/targets/'),
    (r'training_config/intelligent_training_config\.yaml', 'pipeline/training/intelligent.yaml'),
    (r'training_config/safety_config\.yaml', 'pipeline/training/safety.yaml'),
    (r'training_config/system_config\.yaml', 'core/system.yaml'),
    (r'model_config/(\w+)\.yaml', r'models/\1.yaml'),
]

def find_matches(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find matches in a file."""
    matches = []
    try:
        content = file_path.read_text()
        for line_num, line in enumerate(content.split('\n'), 1):
            for pattern, replacement in OLD_PATTERNS:
                if re.search(pattern, line):
                    matches.append((line_num, line.strip(), replacement))
    except Exception as e:
        pass
    return matches

def main():
    """Find all hardcoded config paths."""
    repo_root = Path(__file__).resolve().parent.parent
    training_dir = repo_root / "TRAINING"
    config_dir = repo_root / "CONFIG"
    
    print("=" * 80)
    print("Finding Hardcoded Config Paths")
    print("=" * 80)
    print()
    
    files_with_matches = []
    
    # Search TRAINING directory
    for py_file in training_dir.rglob("*.py"):
        matches = find_matches(py_file)
        if matches:
            files_with_matches.append((py_file, matches))
    
    # Search CONFIG directory
    for py_file in config_dir.rglob("*.py"):
        matches = find_matches(py_file)
        if matches:
            files_with_matches.append((py_file, matches))
    
    if not files_with_matches:
        print("✅ No hardcoded config paths found!")
        return
    
    print(f"Found {len(files_with_matches)} files with hardcoded paths:\n")
    
    for file_path, matches in files_with_matches:
        rel_path = file_path.relative_to(repo_root)
        print(f"📄 {rel_path}")
        for line_num, line, replacement in matches:
            print(f"   Line {line_num}: {line[:80]}")
            print(f"   → Should use: {replacement}")
        print()

if __name__ == "__main__":
    main()

