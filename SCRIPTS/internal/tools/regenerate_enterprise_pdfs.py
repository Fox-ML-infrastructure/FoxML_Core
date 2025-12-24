#!/usr/bin/env python3
"""
Regenerate Enterprise Bundle PDFs from Markdown source files.

This script converts all legal and commercial documentation from Markdown to PDF
for enterprise procurement and legal review purposes.

Usage:
    python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py
    python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --clean
    python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --verbose
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Set

# Project root - find by looking for COMMERCIAL_LICENSE.md or LEGAL directory
SCRIPT_DIR = Path(__file__).resolve().parent
# Try going up from script location
for i in range(1, 6):
    candidate = SCRIPT_DIR.parents[i]
    if (candidate / "COMMERCIAL_LICENSE.md").exists() or (candidate / "LEGAL").exists():
        PROJECT_ROOT = candidate
        break
else:
    # Fallback: assume 4 levels up
    PROJECT_ROOT = SCRIPT_DIR.parents[3]

LEGAL_DIR = PROJECT_ROOT / "LEGAL"
ENTERPRISE_BUNDLE_DIR = PROJECT_ROOT / "ENTERPRISE_BUNDLE"
ROOT_DOCS = [
    PROJECT_ROOT / "COMMERCIAL_LICENSE.md",
    PROJECT_ROOT / "DUAL_LICENSE.md",
    PROJECT_ROOT / "LICENSE",
]

# Files to exclude from PDF generation
EXCLUDED_FILES: Set[str] = {
    "README.md",
    "CHANGELOG_ENTERPRISE.md",
    "BUNDLE_INDEX.md",
}


def find_markdown_files() -> List[Path]:
    """Find all Markdown files that should be converted to PDF."""
    files: List[Path] = []
    
    # Add root-level legal documents
    for doc in ROOT_DOCS:
        if doc.exists():
            files.append(doc)
    
    # Add all .md files from LEGAL directory (recursive)
    if LEGAL_DIR.exists():
        for md_file in LEGAL_DIR.rglob("*.md"):
            if md_file.name not in EXCLUDED_FILES:
                files.append(md_file)
    
    return sorted(files)


def sanitize_filename(path: Path) -> str:
    """Convert file path to PDF filename."""
    # Get relative path from project root
    if path.is_relative_to(PROJECT_ROOT):
        rel_path = path.relative_to(PROJECT_ROOT)
    else:
        rel_path = path.name
    
    # Convert to PDF filename (replace .md with .pdf, keep directory structure)
    pdf_name = str(rel_path).replace(".md", ".pdf")
    # Remove any directory separators for flat output
    pdf_name = pdf_name.replace("/", "_").replace("\\", "_")
    
    return pdf_name


def preprocess_markdown(content: str) -> str:
    """Preprocess markdown to handle Unicode characters for LaTeX."""
    import re
    
    # Replace common emojis with text equivalents
    replacements = {
        "⚠️": "[WARNING]",
        "⚠": "[WARNING]",
        "✅": "[OK]",
        "❌": "[ERROR]",
        "📧": "[EMAIL]",
        "📄": "[DOCUMENT]",
        "📋": "[CLIPBOARD]",
        "📚": "[BOOKS]",
        "💰": "[MONEY]",
        "📅": "[CALENDAR]",
    }
    
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining Unicode characters that might cause issues
    # Keep basic ASCII and common punctuation
    # This is a fallback - try to preserve most content
    content = re.sub(r'[^\x00-\x7F]+', '[UNICODE]', content)
    
    return content


def convert_to_pdf(md_file: Path, output_dir: Path, verbose: bool = False) -> bool:
    """Convert a single Markdown file to PDF using pandoc."""
    pdf_name = sanitize_filename(md_file)
    pdf_path = output_dir / pdf_name
    
    # Read and preprocess content
    try:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        content = preprocess_markdown(content)
        
        # Write to temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
    except Exception as e:
        if verbose:
            print(f"  ✗ Error reading {md_file.name}: {e}", file=sys.stderr)
        return False
    
    # Try multiple PDF engines in order of preference
    engines = ["pdflatex", "xelatex", "lualatex"]
    success = False
    
    for engine in engines:
        cmd = [
            "pandoc",
            tmp_path,
            "-o", str(pdf_path),
            f"--pdf-engine={engine}",
            "-V", "geometry:margin=1in",
            "-V", "documentclass=article",
            "-V", "fontsize=11pt",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # 60 second timeout per file
            )
            success = True
            break
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"  Timeout with {engine}, trying next...")
            continue
        except subprocess.CalledProcessError:
            # Try next engine
            continue
    
    if not success:
        # Clean up temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        if verbose:
            print(f"  ✗ Error converting {md_file.name}: All PDF engines failed", file=sys.stderr)
        else:
            print(f"  ✗ Error converting {md_file.name}", file=sys.stderr)
        return False
    
    # Clean up temp file
    try:
        os.unlink(tmp_path)
    except:
        pass
    
    if verbose:
        print(f"  ✓ Success: {pdf_name}")
    return True
    
    if verbose:
        print(f"Converting: {md_file.name} -> {pdf_name}")
        print(f"  Command: {' '.join(cmd)}")
    
    # FileNotFoundError check
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Clean up temp file
        if 'tmp_path' in locals():
            try:
                os.unlink(tmp_path)
            except:
                pass
        print("  ✗ Error: pandoc not found. Please install pandoc.", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate Enterprise Bundle PDFs from Markdown source files"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing PDFs before regenerating"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--engine",
        default="pdflatex",
        help="PDF engine to use (default: pdflatex)"
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ENTERPRISE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clean existing PDFs if requested
    if args.clean:
        if args.verbose:
            print("Cleaning existing PDFs...")
        for pdf_file in ENTERPRISE_BUNDLE_DIR.glob("*.pdf"):
            pdf_file.unlink()
            if args.verbose:
                print(f"  Removed: {pdf_file.name}")
    
    # Find all Markdown files
    md_files = find_markdown_files()
    
    if args.verbose:
        print(f"\nFound {len(md_files)} Markdown files to convert\n")
    
    # Convert each file
    success_count = 0
    error_count = 0
    
    for md_file in md_files:
        if convert_to_pdf(md_file, ENTERPRISE_BUNDLE_DIR, args.verbose):
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PDF Generation Complete")
    print(f"{'='*60}")
    print(f"Success: {success_count}")
    if error_count > 0:
        print(f"Errors: {error_count}", file=sys.stderr)
    print(f"Output directory: {ENTERPRISE_BUNDLE_DIR}")
    print(f"{'='*60}\n")
    
    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
