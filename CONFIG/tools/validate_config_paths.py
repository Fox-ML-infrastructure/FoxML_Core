#!/usr/bin/env python3
"""
Validate Config Paths Migration

Scans TRAINING directory for remaining hardcoded config paths and verifies
that all config files are accessible via the centralized config loader API.

This script helps ensure the migration from hardcoded paths to config loader
API is complete.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set

# Patterns to search for hardcoded paths
HARDCODED_PATTERNS = [
    (r'Path\("CONFIG/', 'Hardcoded CONFIG path with Path()'),
    (r"Path\('CONFIG/", "Hardcoded CONFIG path with Path()"),
    (r'"CONFIG/experiments"', 'Hardcoded experiments path'),
    (r"'CONFIG/experiments'", "Hardcoded experiments path"),
    (r'"CONFIG/pipeline/', 'Hardcoded pipeline path'),
    (r"'CONFIG/pipeline/", "Hardcoded pipeline path"),
    (r'"CONFIG/training_config/', 'Hardcoded training_config path'),
    (r"'CONFIG/training_config/", "Hardcoded training_config path"),
    (r'"CONFIG/data/', 'Hardcoded data config path'),
    (r"'CONFIG/data/", "Hardcoded data config path"),
    (r'"CONFIG/ranking/', 'Hardcoded ranking path'),
    (r"'CONFIG/ranking/", "Hardcoded ranking path"),
    (r'"CONFIG/models/', 'Hardcoded models path'),
    (r"'CONFIG/models/", "Hardcoded models path"),
    (r'"CONFIG/core/', 'Hardcoded core path'),
    (r"'CONFIG/core/", "Hardcoded core path"),
]

# Patterns that are OK (in comments, docstrings, or fallback code)
ALLOWED_PATTERNS = [
    r'#.*CONFIG/',  # Comments
    r'""".*CONFIG/',  # Docstrings
    r"'''.*CONFIG/",  # Docstrings
    r'logger\.(debug|info|warning|error).*CONFIG/',  # Log messages
    r'f".*CONFIG/',  # F-strings in log messages (may contain examples)
    r"f'.*CONFIG/",  # F-strings in log messages
    r'fallback.*CONFIG/',  # Fallback code
    r'# Fallback',  # Fallback comments
    r'# DEPRECATED',  # Deprecated code comments
]

def find_hardcoded_paths(file_path: Path) -> List[Tuple[int, str, str]]:
    """Find hardcoded config paths in a file."""
    matches = []
    try:
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            # Skip if line matches allowed patterns
            is_allowed = any(re.search(pattern, line, re.IGNORECASE) for pattern in ALLOWED_PATTERNS)
            if is_allowed:
                continue
            
            # Check for hardcoded patterns
            for pattern, description in HARDCODED_PATTERNS:
                if re.search(pattern, line):
                    matches.append((line_num, line.strip(), description))
                    break  # Only report once per line
    except Exception as e:
        print(f"Warning: Could not read {file_path}: {e}", file=sys.stderr)
    
    return matches

def validate_config_loader_access() -> Dict[str, bool]:
    """Validate that config files are accessible via config loader."""
    results = {}
    
    try:
        # Add CONFIG to path
        repo_root = Path(__file__).resolve().parents[2]
        config_dir = repo_root / "CONFIG"
        if str(config_dir) not in sys.path:
            sys.path.insert(0, str(config_dir))
        
        from config_loader import (
            get_config_path,
            get_experiment_config_path,
            load_training_config,
            CONFIG_DIR
        )
        
        # Test common config paths
        test_configs = [
            "excluded_features",
            "feature_registry",
            "intelligent_training_config",
            "safety_config",
            "lightgbm",
        ]
        
        for config_name in test_configs:
            try:
                path = get_config_path(config_name)
                results[config_name] = path.exists()
            except Exception as e:
                results[config_name] = False
                print(f"Warning: Could not get path for {config_name}: {e}", file=sys.stderr)
        
        # Test experiment config path
        try:
            exp_path = get_experiment_config_path("e2e_full_targets_test")
            results["experiment_config"] = exp_path.exists()
        except Exception as e:
            results["experiment_config"] = False
            print(f"Warning: Could not get experiment config path: {e}", file=sys.stderr)
        
        # Test training config loading
        try:
            intel_config = load_training_config("intelligent_training_config")
            results["load_training_config"] = bool(intel_config)
        except Exception as e:
            results["load_training_config"] = False
            print(f"Warning: Could not load training config: {e}", file=sys.stderr)
        
    except ImportError as e:
        print(f"Error: Could not import config loader: {e}", file=sys.stderr)
        results["config_loader_available"] = False
    
    return results

def check_symlinks() -> List[Tuple[str, bool, str]]:
    """Check if symlinks are valid."""
    issues = []
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "CONFIG"
    
    # Check common symlinks
    symlinks_to_check = [
        "excluded_features.yaml",
        "feature_registry.yaml",
        "target_configs.yaml",
        "training_config/intelligent_training_config.yaml",
    ]
    
    for symlink_name in symlinks_to_check:
        symlink_path = config_dir / symlink_name
        if symlink_path.exists():
            if symlink_path.is_symlink():
                try:
                    target = symlink_path.resolve()
                    if not target.exists():
                        issues.append((str(symlink_path), False, f"Broken symlink: target {target} does not exist"))
                    else:
                        issues.append((str(symlink_path), True, f"Valid symlink -> {target}"))
                except Exception as e:
                    issues.append((str(symlink_path), False, f"Error resolving symlink: {e}"))
    
    return issues

def main():
    """Main validation function."""
    repo_root = Path(__file__).resolve().parents[2]
    training_dir = repo_root / "TRAINING"
    
    print("=" * 80)
    print("Config Paths Migration Validation")
    print("=" * 80)
    print()
    
    # 1. Scan for hardcoded paths
    print("1. Scanning for hardcoded config paths...")
    files_with_matches = []
    
    for py_file in training_dir.rglob("*.py"):
        matches = find_hardcoded_paths(py_file)
        if matches:
            files_with_matches.append((py_file, matches))
    
    if files_with_matches:
        print(f"   ⚠️  Found {len(files_with_matches)} files with hardcoded paths:\n")
        for file_path, matches in files_with_matches:
            rel_path = file_path.relative_to(repo_root)
            print(f"   📄 {rel_path}")
            for line_num, line, description in matches[:5]:  # Show first 5 matches
                print(f"      Line {line_num}: {description}")
                print(f"      {line[:80]}")
            if len(matches) > 5:
                print(f"      ... and {len(matches) - 5} more matches")
            print()
    else:
        print("   ✅ No hardcoded config paths found!")
    print()
    
    # 2. Validate config loader access
    print("2. Validating config loader API access...")
    loader_results = validate_config_loader_access()
    
    if loader_results:
        all_ok = all(loader_results.values())
        if all_ok:
            print("   ✅ All config files accessible via config loader")
        else:
            print("   ⚠️  Some config files not accessible:")
            for config_name, accessible in loader_results.items():
                status = "✅" if accessible else "❌"
                print(f"      {status} {config_name}")
    else:
        print("   ❌ Config loader not available")
    print()
    
    # 3. Check symlinks
    print("3. Checking symlinks...")
    symlink_issues = check_symlinks()
    
    if symlink_issues:
        valid_count = sum(1 for _, is_valid, _ in symlink_issues if is_valid)
        invalid_count = len(symlink_issues) - valid_count
        
        if invalid_count == 0:
            print(f"   ✅ All {len(symlink_issues)} symlinks are valid")
        else:
            print(f"   ⚠️  {invalid_count} broken symlink(s) found:")
            for path, is_valid, message in symlink_issues:
                if not is_valid:
                    print(f"      ❌ {path}: {message}")
    else:
        print("   ℹ️  No symlinks to check")
    print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_issues = len(files_with_matches)
    loader_issues = sum(1 for v in loader_results.values() if not v) if loader_results else 0
    symlink_issues_count = sum(1 for _, is_valid, _ in symlink_issues if not is_valid) if symlink_issues else 0
    
    if total_issues == 0 and loader_issues == 0 and symlink_issues_count == 0:
        print("✅ All validations passed! Migration appears complete.")
        return 0
    else:
        print(f"⚠️  Found issues:")
        print(f"   - {total_issues} file(s) with hardcoded paths")
        print(f"   - {loader_issues} config loader access issue(s)")
        print(f"   - {symlink_issues_count} broken symlink(s)")
        return 1

if __name__ == "__main__":
    sys.exit(main())

