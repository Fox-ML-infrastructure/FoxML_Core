#!/usr/bin/env python3
"""
Regenerate enterprise PDF bundle from Markdown source files.

This script converts all legal and enterprise documentation from Markdown to PDF
for enterprise procurement and distribution.
"""

import subprocess
import sys
import tempfile
import re
from pathlib import Path
from typing import List, Tuple

# Repository root
REPO_ROOT = Path(__file__).parent.parent
LEGAL_DIR = REPO_ROOT / "LEGAL"
ENTERPRISE_BUNDLE_DIR = REPO_ROOT / "ENTERPRISE_BUNDLE"

# Mapping of PDF names to source markdown files
PDF_MAPPINGS: List[Tuple[str, Path]] = [
    # Root-level documents
    ("COMMERCIAL_LICENSE.pdf", REPO_ROOT / "COMMERCIAL_LICENSE.md"),
    ("DUAL_LICENSE.pdf", REPO_ROOT / "DUAL_LICENSE.md"),
    ("LICENSE.pdf", REPO_ROOT / "LICENSE"),
    
    # LEGAL directory documents (mapped to LEGAL_ prefix)
    ("LEGAL_ACCEPTABLE_USE_POLICY.pdf", LEGAL_DIR / "ACCEPTABLE_USE_POLICY.md"),
    ("LEGAL_BUSINESS_CONTINUITY_PLAN.pdf", LEGAL_DIR / "BUSINESS_CONTINUITY_PLAN.md"),
    ("LEGAL_CLA.pdf", LEGAL_DIR / "CLA.md"),
    ("LEGAL_CLIENT_ONBOARDING.pdf", LEGAL_DIR / "CLIENT_ONBOARDING.md"),
    ("LEGAL_COMMERCIAL_USE.pdf", LEGAL_DIR / "COMMERCIAL_USE.md"),
    ("LEGAL_COMPLIANCE_FAQ.pdf", LEGAL_DIR / "COMPLIANCE_FAQ.md"),
    ("LEGAL_COPYRIGHT_NOTICE.pdf", LEGAL_DIR / "COPYRIGHT_NOTICE.md"),
    ("LEGAL_CREDITS.pdf", LEGAL_DIR / "CREDITS.md"),
    ("LEGAL_DATA_PROCESSING_ADDENDUM.pdf", LEGAL_DIR / "DATA_PROCESSING_ADDENDUM.md"),
    ("LEGAL_DATA_RETENTION_DELETION_POLICY.pdf", LEGAL_DIR / "DATA_RETENTION_DELETION_POLICY.md"),
    ("LEGAL_DECISION_MATRIX.pdf", LEGAL_DIR / "DECISION_MATRIX.md"),
    ("LEGAL_ENTERPRISE_CHECKLIST.pdf", LEGAL_DIR / "ENTERPRISE_CHECKLIST.md"),
    ("LEGAL_ENTERPRISE_DELIVERY.pdf", LEGAL_DIR / "ENTERPRISE_DELIVERY.md"),
    ("LEGAL_EXPORT_COMPLIANCE.pdf", LEGAL_DIR / "EXPORT_COMPLIANCE.md"),
    ("LEGAL_FAQ.pdf", LEGAL_DIR / "FAQ.md"),
    ("LEGAL_INCIDENT_RESPONSE_PLAN.pdf", LEGAL_DIR / "INCIDENT_RESPONSE_PLAN.md"),
    ("LEGAL_INDEMNIFICATION.pdf", LEGAL_DIR / "INDEMNIFICATION.md"),
    ("LEGAL_INFOSEC_SELF_ASSESSMENT.pdf", LEGAL_DIR / "INFOSEC_SELF_ASSESSMENT.md"),
    ("LEGAL_IP_ASSIGNMENT_AGREEMENT.pdf", LEGAL_DIR / "IP_ASSIGNMENT_AGREEMENT.md"),
    ("LEGAL_IP_OWNERSHIP_CLARIFICATION.pdf", LEGAL_DIR / "IP_OWNERSHIP_CLARIFICATION.md"),
    ("LEGAL_LICENSE_ENFORCEMENT.pdf", LEGAL_DIR / "LICENSE_ENFORCEMENT.md"),
    ("LEGAL_LICENSING.pdf", LEGAL_DIR / "LICENSING.md"),
    ("LEGAL_PENETRATION_TESTING_STATEMENT.pdf", LEGAL_DIR / "PENETRATION_TESTING_STATEMENT.md"),
    ("LEGAL_PRIVACY_POLICY.pdf", LEGAL_DIR / "PRIVACY_POLICY.md"),
    ("LEGAL_PRODUCTION_USE_NOTIFICATION.pdf", LEGAL_DIR / "PRODUCTION_USE_NOTIFICATION.md"),
    ("LEGAL_REGULATORY_DISCLAIMERS.pdf", LEGAL_DIR / "REGULATORY_DISCLAIMERS.md"),
    ("LEGAL_RELEASE_NOTES_TAGGING_STANDARD.pdf", LEGAL_DIR / "RELEASE_NOTES_TAGGING_STANDARD.md"),
    ("LEGAL_RELEASE_POLICY.pdf", LEGAL_DIR / "RELEASE_POLICY.md"),
    ("LEGAL_RISK_ASSESSMENT_MATRIX.pdf", LEGAL_DIR / "RISK_ASSESSMENT_MATRIX.md"),
    ("LEGAL_SECURITY_CONTROLS_MATRIX.pdf", LEGAL_DIR / "SECURITY_CONTROLS_MATRIX.md"),
    ("LEGAL_SECURITY.pdf", LEGAL_DIR / "SECURITY.md"),
    ("LEGAL_SERVICE_LEVEL_AGREEMENT.pdf", LEGAL_DIR / "SERVICE_LEVEL_AGREEMENT.md"),
    ("LEGAL_SUBSCRIPTIONS.pdf", LEGAL_DIR / "SUBSCRIPTIONS.md"),
    ("LEGAL_SUPPORT_POLICY.pdf", LEGAL_DIR / "SUPPORT_POLICY.md"),
    ("LEGAL_SYSTEM_ARCHITECTURE_DIAGRAM.pdf", LEGAL_DIR / "SYSTEM_ARCHITECTURE_DIAGRAM.md"),
    ("LEGAL_TOS.pdf", LEGAL_DIR / "TOS.md"),
    ("LEGAL_TRADEMARK_POLICY.pdf", LEGAL_DIR / "TRADEMARK_POLICY.md"),
    ("LEGAL_WARRANTY_LIABILITY_ADDENDUM.pdf", LEGAL_DIR / "WARRANTY_LIABILITY_ADDENDUM.md"),
]


def check_pandoc() -> bool:
    """Check if pandoc is available."""
    try:
        subprocess.run(["pandoc", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def clean_unicode_for_latex(content: str) -> str:
    """Remove or replace Unicode characters that LaTeX can't handle."""
    # Common emoji replacements
    replacements = {
        "⚠️": "[WARNING]",
        "📧": "[EMAIL]",
        "✅": "[OK]",
        "❌": "[ERROR]",
        "💰": "[PAID]",
        "🔗": "[LINK]",
        "📝": "[NOTE]",
        "📊": "[STATS]",
        "📁": "[FOLDER]",
        "🗑️": "[DELETE]",
    }
    
    # Replace emojis
    for emoji, replacement in replacements.items():
        content = content.replace(emoji, replacement)
    
    # Remove any remaining emojis (Unicode ranges for common emojis)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002600-\U000026FF"  # miscellaneous symbols
        "\U00002700-\U000027BF"  # dingbats
        "]+",
        flags=re.UNICODE
    )
    content = emoji_pattern.sub("", content)
    
    # Replace box-drawing characters with ASCII equivalents
    # Extended box-drawing character set
    box_chars = {
        "┌": "+", "┐": "+", "└": "+", "┘": "+",
        "├": "+", "┤": "+", "┬": "+", "┴": "+",
        "│": "|", "─": "-", "═": "=",
        "┼": "+", "║": "|", "╔": "+", "╗": "+",
        "╚": "+", "╝": "+", "╠": "+", "╣": "+",
        "╦": "+", "╩": "+", "╬": "+",
    }
    for char, replacement in box_chars.items():
        content = content.replace(char, replacement)
    
    # Remove any remaining box-drawing characters (Unicode range 2500-257F)
    box_pattern = re.compile("[\u2500-\u257F]+", flags=re.UNICODE)
    content = box_pattern.sub("", content)
    
    # Replace other problematic Unicode characters with ASCII equivalents
    problematic_chars = {
        "⸻": "---",  # Triple hyphen
        "—": "--",   # Em dash
        "–": "-",    # En dash
        "…": "...",  # Ellipsis
        "•": "*",    # Bullet
        "→": "->",   # Right arrow
        "←": "<-",   # Left arrow
    }
    for char, replacement in problematic_chars.items():
        content = content.replace(char, replacement)
    
    return content


def convert_to_pdf(source: Path, output: Path, verbose: bool = False) -> bool:
    """Convert a markdown file to PDF using pandoc."""
    if not source.exists():
        if verbose:
            print(f"[WARNING] Source file not found: {source}")
        return False
    
    try:
        # Read and clean content
        content = source.read_text(encoding='utf-8')
        cleaned_content = clean_unicode_for_latex(content)
        
        # Write to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write(cleaned_content)
            tmp_path = tmp.name
        
        try:
            # Try xelatex first (better Unicode support), fall back to pdflatex
            engines = ["xelatex", "pdflatex"]
            last_error = None
            
            for engine in engines:
                cmd = [
                    "pandoc",
                    tmp_path,
                    "-o", str(output),
                    f"--pdf-engine={engine}",
                    "-V", "geometry:margin=1in",
                    "-V", "fontsize=11pt",
                    "--toc",
                ]
        
                if verbose:
                    print(f"Converting: {source.name} -> {output.name} (using {engine})")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    if verbose:
                        print(f"[OK] Generated: {output.name}")
                    return True
                else:
                    last_error = result.stderr
                    if verbose and engine == engines[0]:
                        print(f"[WARNING] {engine} failed, trying {engines[1]}...")
                    continue
            
            # If all engines failed
            if verbose:
                print(f"[ERROR] Error converting {source.name}: {last_error}")
            return False
        finally:
            # Clean up temp file
            Path(tmp_path).unlink()
    except Exception as e:
        if verbose:
            print(f"❌ Exception converting {source.name}: {e}")
        return False


def main():
    """Main function to regenerate all PDFs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate enterprise PDF bundle")
    parser.add_argument("--clean", action="store_true", help="Remove existing PDFs first")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    # Check pandoc
    if not check_pandoc():
        print("❌ Error: pandoc is not installed or not in PATH")
        print("Install pandoc: https://pandoc.org/installing.html")
        sys.exit(1)
    
    # Clean existing PDFs if requested
    if args.clean:
        pdf_files = list(ENTERPRISE_BUNDLE_DIR.glob("*.pdf"))
        for pdf in pdf_files:
            pdf.unlink()
            if args.verbose:
                print(f"🗑️  Removed: {pdf.name}")
    
    # Ensure output directory exists
    ENTERPRISE_BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Convert all files
    success_count = 0
    fail_count = 0
    
    for pdf_name, source_file in PDF_MAPPINGS:
        output_path = ENTERPRISE_BUNDLE_DIR / pdf_name
        
        if convert_to_pdf(source_file, output_path, args.verbose):
            success_count += 1
        else:
            fail_count += 1
            if args.verbose:
                print(f"⚠️  Skipped: {pdf_name} (source not found or conversion failed)")
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"   ✅ Successfully generated: {success_count} PDFs")
    if fail_count > 0:
        print(f"   ⚠️  Failed/Skipped: {fail_count} PDFs")
    print(f"   📁 Output directory: {ENTERPRISE_BUNDLE_DIR}")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()























