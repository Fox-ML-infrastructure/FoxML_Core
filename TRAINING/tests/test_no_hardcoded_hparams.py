#!/usr/bin/env python3

"""
Copyright (c) 2025-2026 Fox ML Infrastructure LLC

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

SST (Single Source of Truth) Enforcement Test
==============================================

This test enforces that hyperparameters, thresholds, and behavioral knobs
are loaded from configuration files, not hardcoded in source code.

The goal: "Same config → same behavior → same results."

Allowed hardcoded patterns:
- Numerical epsilon constants (1e-9, np.finfo, etc.)
- Mathematical constants (math.pi, etc.)
- Design constants explicitly marked with "DESIGN CONSTANT" comment
- Debug-only flags in dev-only scripts

Must be config:
- Model hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- Data splits (test_size, cv_folds, shuffle)
- Randomness (random_state, seeds)
- Safety thresholds (leakage thresholds, AUC cutoffs, correlation cutoffs)
- Resource use (batch_size, max_rows, GPU flags)
- Routing/confidence thresholds (HIGH/MED/LOW cutoffs)
"""

import pathlib
import re
from typing import List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
TARGET_DIRS = ["TRAINING"]

# Strict markers that allow hardcoded values (must be explicitly reviewed)
ALLOWED_MARKERS = [
    "FALLBACK_DEFAULT_OK",  # Documented internal fallback defaults
    "DESIGN_CONSTANT_OK",    # Rare, non-config, mathematically intrinsic constants
]

# Allowlist for constants that are okay to hardcode (even without markers)
ALLOWED_PATTERNS = [
    # Numerical epsilon constants
    r"EPSILON\s*=\s*1e-9",
    r"1e-9",  # Small epsilon for numerical stability
    r"np\.finfo",
    r"math\.pi",
    r"math\.e",
    # Test/debug constants in test files
    r"test_.*\.py",
    r"smoke_test",
    r"debug",
    # Internal sanity checks (not behavioral knobs)
    r"if\s+.*\s*<=\s*0\s*:",
    r"if\s+.*\s*==\s*0\s*:",
    r"raise\s+ValueError",
    r"raise\s+AssertionError",
]

# Patterns that indicate configuration-y values (must be in config)
CONFIGY_PATTERNS = [
    # Model hyperparameters
    (r"n_estimators\s*=\s*(\d+)", "n_estimators"),
    (r"max_depth\s*=\s*(\d+)", "max_depth"),
    (r"learning_rate\s*=\s*([0-9.]+)", "learning_rate"),
    (r"alpha\s*=\s*([0-9.]+)", "alpha"),
    (r"num_leaves\s*=\s*(\d+)", "num_leaves"),
    (r"min_child_samples\s*=\s*(\d+)", "min_child_samples"),
    (r"subsample\s*=\s*([0-9.]+)", "subsample"),
    (r"colsample_bytree\s*=\s*([0-9.]+)", "colsample_bytree"),
    
    # Data splits
    (r"test_size\s*=\s*([0-9.]+)", "test_size"),
    (r"cv_folds?\s*=\s*(\d+)", "cv_folds"),
    (r"n_splits\s*=\s*(\d+)", "n_splits"),
    (r"train_size\s*=\s*([0-9.]+)", "train_size"),
    
    # Randomness
    (r"random_state\s*=\s*(\d+)", "random_state"),
    (r"seed\s*=\s*(\d+)", "seed"),
    (r"split_seed\s*=\s*(\d+)", "split_seed"),
    (r"leak_seed\s*=\s*(\d+)", "leak_seed"),
    (r"shuffle_seed\s*=\s*(\d+)", "shuffle_seed"),
    (r"ridge_seed\s*=\s*(\d+)", "ridge_seed"),
    (r"base_seed\s*=\s*(\d+)", "base_seed"),
    
    # Thresholds (suspicious high values)
    (r"0\.9\d{2,}", "high_threshold"),  # 0.99, 0.999, etc.
    (r"0\.95", "threshold_0.95"),
    (r">=\s*0\.9\d+", "threshold_comparison"),
    (r"correlation.*0\.9\d+", "correlation_threshold"),
    (r"accuracy.*0\.9\d+", "accuracy_threshold"),
    
    # Resource use
    (r"batch_size\s*=\s*(\d+)", "batch_size"),
    (r"max_rows\s*=\s*(\d+)", "max_rows"),
    (r"max_epochs\s*=\s*(\d+)", "max_epochs"),
    (r"n_jobs\s*=\s*(\d+)", "n_jobs"),
    
    # Top-K heuristics
    (r"top_k\s*=\s*(\d+)", "top_k"),
    (r"top_\d+\s*=", "top_n_literal"),
    (r"top-\d+", "top_n_dash"),
    (r"int\(len\(.*\)\s*\*\s*0\.1\)", "top_10_percent"),  # Top 10% pattern
]

# Files to exclude from checking (test files, debug scripts, etc.)
EXCLUDED_PATTERNS = [
    r"test_.*\.py$",
    r"_test\.py$",
    r"smoke_test.*\.py$",
    r"debug.*\.py$",
    r"experiment.*\.py$",
    r"EXPERIMENTS/",
]


def is_excluded_file(file_path: pathlib.Path) -> bool:
    """Check if file should be excluded from SST checking."""
    rel_path = str(file_path.relative_to(ROOT))
    return any(re.search(pattern, rel_path) for pattern in EXCLUDED_PATTERNS)


def is_allowed_pattern(line: str) -> bool:
    """Check if line matches an allowed hardcoded pattern."""
    return any(re.search(pattern, line, re.IGNORECASE) for pattern in ALLOWED_PATTERNS)


def has_allowed_marker(lines: List[str], line_idx: int, context: int = 3) -> bool:
    """Check if there's an allowed marker comment nearby."""
    start = max(0, line_idx - context)
    end = min(len(lines), line_idx + 1)
    context_lines = lines[start:end]
    line_upper = "\n".join(context_lines).upper()
    return any(marker in line_upper for marker in ALLOWED_MARKERS)


def check_file(file_path: pathlib.Path) -> List[Tuple[int, str, str]]:
    """
    Check a single file for hardcoded configuration values.
    
    Returns:
        List of (line_number, pattern_name, snippet) violations
    """
    violations = []
    
    try:
        text = file_path.read_text(encoding='utf-8')
        lines = text.splitlines()
    except Exception as e:
        # Skip files that can't be read
        return violations
    
    for pattern, pattern_name in CONFIGY_PATTERNS:
        for match in re.finditer(pattern, text, re.MULTILINE):
            line_num = text.count("\n", 0, match.start()) + 1
            line = lines[line_num - 1].strip()
            
            # Skip if excluded
            if is_excluded_file(file_path):
                continue
            
            # Skip if matches allowed pattern
            if is_allowed_pattern(line):
                continue
            
            # Skip if has explicit allowed marker (strict enforcement)
            if has_allowed_marker(lines, line_num - 1):
                continue
            
            violations.append((line_num, pattern_name, line))
    
    return violations


def test_no_hardcoded_hparams():
    """Main test: find all hardcoded hyperparameter-like values."""
    all_violations = []
    
    for target_dir in TARGET_DIRS:
        target_path = ROOT / target_dir
        if not target_path.exists():
            continue
        
        for py_file in target_path.rglob("*.py"):
            if is_excluded_file(py_file):
                continue
            
            violations = check_file(py_file)
            if violations:
                rel_path = py_file.relative_to(ROOT)
                for line_num, pattern_name, snippet in violations:
                    all_violations.append(f"{rel_path}:{line_num}: [{pattern_name}] {snippet}")
    
    if all_violations:
        violation_msg = "\n".join(all_violations[:50])  # Show first 50
        if len(all_violations) > 50:
            violation_msg += f"\n... and {len(all_violations) - 50} more violations"
        
        raise AssertionError(
            f"Found {len(all_violations)} hardcoded hyperparameter/configuration values.\n"
            f"These should be loaded from CONFIG/ YAML files via config_loader.\n\n"
            f"Violations:\n{violation_msg}\n\n"
            f"To fix:\n"
            f"1. Move values to appropriate CONFIG/*.yaml file\n"
            f"2. Load via get_cfg() or load_model_config()\n"
            f"3. If intentionally hardcoded, add '# FALLBACK_DEFAULT_OK' or '# DESIGN_CONSTANT_OK' marker\n"
            f"4. See DOCS/03_technical/internal/SST_DETERMINISM_GUARANTEES.md for guidelines"
        )


if __name__ == "__main__":
    test_no_hardcoded_hparams()
    print("✅ SST enforcement test passed: no hardcoded hyperparameters found")
