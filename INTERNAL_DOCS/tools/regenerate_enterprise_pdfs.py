#!/usr/bin/env python3
"""
Regenerate PDFs from LEGAL markdown files for enterprise bundle.

This script:
- Finds all .md files in LEGAL/ directory (including subdirectories)
- Includes root-level legal files (COMMERCIAL_LICENSE.md, DUAL_LICENSE.md, LICENSE)
- Converts Unicode characters to ASCII for LaTeX compatibility
- Generates PDFs using pandoc

Note: This script is kept in internal/ and should NOT be committed to the repo.
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

# Calculate paths
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent  # INTERNAL_DOCS/tools -> INTERNAL_DOCS -> repo root
LEGAL_DIR = REPO_ROOT / "LEGAL"
ENTERPRISE_BUNDLE_DIR = REPO_ROOT / "ENTERPRISE_BUNDLE"

# Files to exclude from PDF generation
EXCLUDE_FILES = {"README.md", "CHANGELOG_ENTERPRISE.md"}


def sanitize_unicode_for_latex(content: str) -> str:
    """Replace Unicode characters with ASCII equivalents for LaTeX compatibility."""
    replacements = {
        "✅": "[OK]",
        "❌": "[X]",
        "⚠️": "[WARNING]",
        "⚠": "[WARNING]",
        "📧": "[EMAIL]",
        "📋": "[CHECKLIST]",
        "📝": "[NOTE]",
        "┌": "+",
        "├": "+",
        "│": "|",
        "└": "+",
        "─": "-",
        "━": "-",
        "┃": "|",
        "┏": "+",
        "┗": "+",
        "┓": "+",
        "┛": "+",
        "┐": "+",
        "┘": "+",
        "┣": "+",
        "┫": "+",
        "┳": "+",
        "┻": "+",
        "╋": "+",
        "┴": "+",
        "┬": "+",
        "┼": "+",
        "┤": "|",
        "├": "+",
        "┐": "+",
        "┌": "+",
        "└": "+",
        "┘": "+",
        "⸻": "---",
        "→": "->",
        "←": "<-",
        "↑": "^",
        "↓": "v",
        "▼": "v",
        "▲": "^",
        "•": "*",
        "…": "...",
        "–": "-",
        "—": "--",
        "≤": "<=",
        "≥": ">=",
        "✓": "[CHECK]",
        "✔": "[CHECK]",
        "🚫": "[PROHIBITED]",
    }
    result = content
    for unicode_char, ascii_replacement in replacements.items():
        result = result.replace(unicode_char, ascii_replacement)
    return result


def find_markdown_files():
    """Find all markdown files to convert."""
    files = []
    
    # Find files in LEGAL/ directory
    if LEGAL_DIR.exists():
        for md_file in LEGAL_DIR.rglob("*.md"):
            if md_file.name not in EXCLUDE_FILES:
                files.append(md_file)
    
    # Add root-level legal files
    root_legal_files = [
        REPO_ROOT / "COMMERCIAL_LICENSE.md",
        REPO_ROOT / "DUAL_LICENSE.md",
        REPO_ROOT / "LICENSE",
    ]
    for legal_file in root_legal_files:
        if legal_file.exists():
            files.append(legal_file)
    
    return files


def convert_to_pdf(md_file: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Convert a markdown file to PDF."""
    # Read and sanitize content
    try:
        content = md_file.read_text(encoding="utf-8")
        sanitized_content = sanitize_unicode_for_latex(content)
    except Exception as e:
        print(f"ERROR: Failed to read {md_file}: {e}", file=sys.stderr)
        return False
    
    # Determine output filename
    if md_file.parent == LEGAL_DIR:
        # Top-level LEGAL file
        output_name = md_file.stem
    elif md_file.parent.parent == LEGAL_DIR:
        # Subdirectory file (e.g., LEGAL/consulting/...)
        subdir = md_file.parent.name
        output_name = f"{subdir}_{md_file.stem}"
    else:
        # Root-level file
        output_name = md_file.stem
    
    output_pdf = output_dir / f"{output_name}.pdf"
    
    # Write sanitized content to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as tmp:
        tmp.write(sanitized_content)
        tmp_path = tmp.name
    
    try:
        # Convert using pandoc
        cmd = [
            "pandoc",
            tmp_path,
            "-o",
            str(output_pdf),
            "--pdf-engine=pdflatex",
            "-V", "geometry:margin=1in",
        ]
        
        if verbose:
            print(f"Converting: {md_file.name} -> {output_pdf.name}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        
        if result.returncode != 0:
            print(f"ERROR: Failed to convert {md_file.name}: {result.stderr}", file=sys.stderr)
            return False
        
        return True
    
    finally:
        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)


def main():
    """Main entry point."""
    # Ensure output directory exists
    ENTERPRISE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check for pandoc
    if not shutil.which("pandoc"):
        print("ERROR: pandoc not found. Please install pandoc.", file=sys.stderr)
        sys.exit(1)
    
    # Find all markdown files
    md_files = find_markdown_files()
    
    if not md_files:
        print("WARNING: No markdown files found to convert.", file=sys.stderr)
        return
    
    print(f"Found {len(md_files)} markdown files to convert...")
    
    # Convert each file
    success_count = 0
    fail_count = 0
    
    for md_file in sorted(md_files):
        if convert_to_pdf(md_file, ENTERPRISE_BUNDLE_DIR, verbose=True):
            success_count += 1
        else:
            fail_count += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PDF Generation Summary:")
    print(f"  Success: {success_count}")
    print(f"  Failed:  {fail_count}")
    print(f"  Total:   {len(md_files)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
