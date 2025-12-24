#!/usr/bin/env python3
"""
Find undefined names in Python files.

Scans Python files for names that are used but not defined/imported.
This catches missing imports like `pl` (polars) that should be imported.

Usage:
    python TRAINING/tools/find_undefined.py
    python TRAINING/tools/find_undefined.py --path TRAINING/training_strategies
"""

import ast
import builtins
import sys
from pathlib import Path
from typing import List, Tuple, Set

# Get project root (this script is in TRAINING/tools/)
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = TRAINING_ROOT.parent

# Built-in names that are always available
BUILTIN_NAMES = set(dir(builtins)) | {
    # Common runtime-provided names
    '__name__', '__file__', '__doc__', '__package__',
    '__builtins__', '__loader__', '__spec__',
    # Common module-level names
    'self', 'cls',  # Method/class context
}

# Common patterns that are false positives (provided at runtime)
RUNTIME_PROVIDED = {
    # Common in argparse scripts
    'args',
    # Common in test files
    'pytest', 'mock', 'unittest',
    # Common in config loading
    '_CONFIG_AVAILABLE', '_PROJECT_ROOT', '_TRAINING_ROOT',
    # Common in logging setup
    'logger',
    # Common in path setup
    'sys', 'os',  # Often set up before imports
}


def analyze_file(path: Path) -> List[Tuple[int, str, str]]:
    """
    Analyze a Python file for undefined names.
    
    Returns list of (lineno, name, context) tuples for undefined names.
    """
    try:
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        return [(e.lineno or 0, f"SYNTAX_ERROR: {e.msg}", "")]
    except Exception as e:
        return [(0, f"PARSE_ERROR: {e}", "")]

    defined: Set[str] = set()
    used: List[Tuple[int, str]] = []
    
    # Track function/class scope for better context
    current_scope = []

    class Visitor(ast.NodeVisitor):
        def visit_Import(self, node):
            for alias in node.names:
                # Handle "import polars as pl" -> defines "pl"
                if alias.asname:
                    defined.add(alias.asname)
                else:
                    # "import os" -> defines "os"
                    defined.add(alias.name.split(".")[0])
            self.generic_visit(node)

        def visit_ImportFrom(self, node):
            for alias in node.names:
                # "from polars import DataFrame" -> defines "DataFrame"
                # "from polars import DataFrame as DF" -> defines "DF"
                if alias.asname:
                    defined.add(alias.asname)
                else:
                    defined.add(alias.name)
            # "from polars import ..." -> also defines "polars" if used as module
            if node.module:
                defined.add(node.module.split(".")[0])
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            defined.add(node.name)
            # Function parameters are defined
            for arg in node.args.args:
                defined.add(arg.arg)
            # *args and **kwargs
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined.add(node.args.kwarg.arg)
            current_scope.append(f"function:{node.name}")
            self.generic_visit(node)
            current_scope.pop()

        def visit_ClassDef(self, node):
            defined.add(node.name)
            current_scope.append(f"class:{node.name}")
            self.generic_visit(node)
            current_scope.pop()

        def visit_Assign(self, node):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
                elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
                    # Handle tuple unpacking: a, b = ...
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
            self.generic_visit(node)

        def visit_With(self, node):
            # "with open(...) as f:" -> defines "f"
            for item in node.items:
                if item.optional_vars:
                    if isinstance(item.optional_vars, ast.Name):
                        defined.add(item.optional_vars.id)
            self.generic_visit(node)

        def visit_For(self, node):
            # "for x in ...:" -> defines "x"
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
            elif isinstance(node.target, ast.Tuple) or isinstance(node.target, ast.List):
                for elt in node.target.elts:
                    if isinstance(elt, ast.Name):
                        defined.add(elt.id)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            # "except Exception as e:" -> defines "e"
            if node.name:
                defined.add(node.name)
            self.generic_visit(node)

        def visit_Name(self, node):
            # Only track names that are being loaded (used), not stored (assigned)
            if isinstance(node.ctx, ast.Load):
                used.append((node.lineno, node.id))
            self.generic_visit(node)

    Visitor().visit(tree)

    # Find undefined names
    undefined = []
    for lineno, name in used:
        if (
            name not in defined
            and name not in BUILTIN_NAMES
            and name not in RUNTIME_PROVIDED
        ):
            undefined.append((lineno, name, ""))

    return undefined


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Find undefined names in Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan entire TRAINING directory
  python TRAINING/tools/find_undefined.py

  # Scan specific directory
  python TRAINING/tools/find_undefined.py --path TRAINING/training_strategies

  # Scan specific file
  python TRAINING/tools/find_undefined.py --path TRAINING/training_strategies/data_preparation.py
        """
    )
    parser.add_argument(
        '--path',
        type=Path,
        default=TRAINING_ROOT,
        help='Path to scan (default: TRAINING/)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show all files scanned (even if no issues)'
    )
    
    args = parser.parse_args()
    
    target_path = args.path.resolve()
    
    if target_path.is_file():
        files_to_scan = [target_path]
    elif target_path.is_dir():
        files_to_scan = list(target_path.rglob("*.py"))
    else:
        print(f"Error: {target_path} does not exist")
        return 1
    
    total_issues = 0
    files_with_issues = 0
    
    for py_file in sorted(files_to_scan):
        # Skip __pycache__ and .pyc files
        if '__pycache__' in str(py_file) or py_file.suffix != '.py':
            continue
        
        problems = analyze_file(py_file)
        
        if problems:
            files_with_issues += 1
            total_issues += len(problems)
            rel_path = py_file.relative_to(PROJECT_ROOT)
            print(f"\n{rel_path}:")
            for lineno, name, context in problems:
                print(f"  Line {lineno}: undefined name '{name}'")
        elif args.verbose:
            rel_path = py_file.relative_to(PROJECT_ROOT)
            print(f"{rel_path}: OK")
    
    if total_issues == 0:
        print(f"✅ No undefined names found in {len(files_to_scan)} file(s)")
        return 0
    else:
        print(f"\n❌ Found {total_issues} undefined name(s) in {files_with_issues} file(s)")
        print("\nTip: Install ruff for better analysis:")
        print("  pip install ruff")
        print("  ruff check TRAINING --select F821")
        return 1


if __name__ == "__main__":
    sys.exit(main())
