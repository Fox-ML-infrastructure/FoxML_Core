#!/usr/bin/env python3
"""
Automatically fix common broken link patterns in DOCS folder.
"""

import re
from pathlib import Path

DOCS_ROOT = Path(__file__).parent.parent / "DOCS"

# Common fixes: (pattern, replacement)
FIXES = [
    # Fix double DOCS paths
    (r'DOCS/DOCS/', ''),
    (r'../../DOCS/', '../'),
    (r'../../../DOCS/', '../../'),
    
    # Fix LEGACY paths
    (r'\.\./LEGACY/', 'LEGACY/'),
    (r'DOCS/LEGACY/', 'LEGACY/'),
    
    # Fix training paths
    (r'\]\(training/', '](01_tutorials/training/'),
    (r'\]\(\.\./training/', '](01_tutorials/training/'),
    
    # Fix SST_ENFORCEMENT_DESIGN paths
    (r'03_technical/implementation/training_utils/SST_ENFORCEMENT_DESIGN\.md',
     '03_technical/implementation/training_utils/INTERNAL/SST_ENFORCEMENT_DESIGN.md'),
    (r'TRAINING/utils/SST_ENFORCEMENT_DESIGN\.md',
     '03_technical/implementation/training_utils/INTERNAL/SST_ENFORCEMENT_DESIGN.md'),
    
    # Fix LEAKAGE_ANALYSIS paths
    (r'03_technical/research/LEAKAGE_ANALYSIS\.md',
     '03_technical/research/INTERNAL/LEAKAGE_ANALYSIS.md'),
    
    # Fix EXPERIMENTS paths in LEGACY
    (r'DOCS/LEGACY/EXPERIMENTS_',
     'LEGACY/EXPERIMENTS_'),
    (r'\.\./\.\./LEGACY/EXPERIMENTS_',
     'LEGACY/EXPERIMENTS_'),
    
    # Fix paths that should be relative from DOCS root
    (r'\]\(DOCS/([^)]+)\)', r'](\1)'),
    
    # Fix changelog paths that incorrectly include 03_technical
    (r'02_reference/changelog/03_technical/',
     '03_technical/'),
    
    # Fix paths in changelog that incorrectly include 02_reference
    (r'02_reference/changelog/02_reference/',
     '02_reference/'),
    
    # Fix telemetry paths
    (r'\]\(telemetry/',
     '](03_technical/telemetry/'),
    (r'\]\(\.\./telemetry/',
     '](03_technical/telemetry/'),
    
    # Fix DATA_PROCESSING paths (outside DOCS)
    (r'\]\(DATA_PROCESSING/README\.md\)',
     '](../DATA_PROCESSING/README.md)'),
    
    # Fix references to files in TRAINING/utils that should be in DOCS
    (r'TRAINING/utils/([^)]+\.md)',
     '03_technical/implementation/training_utils/INTERNAL/\1'),
    
    # Fix references that incorrectly go up too many levels
    (r'\]\(\.\./\.\./\.\./03_technical/',
     '](03_technical/'),
    (r'\]\(\.\./\.\./\.\./02_reference/',
     '](02_reference/'),
    (r'\]\(\.\./\.\./\.\./01_tutorials/',
     '](01_tutorials/'),
]

def fix_file(file_path: Path) -> bool:
    """Fix links in a single file. Returns True if changes were made."""
    try:
        content = file_path.read_text(encoding='utf-8')
        original = content
        
        for pattern, replacement in FIXES:
            content = re.sub(pattern, replacement, content)
        
        if content != original:
            file_path.write_text(content, encoding='utf-8')
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main entry point."""
    md_files = list(DOCS_ROOT.rglob("*.md"))
    fixed_count = 0
    
    for md_file in md_files:
        if fix_file(md_file):
            fixed_count += 1
            print(f"Fixed: {md_file.relative_to(DOCS_ROOT)}")
    
    print(f"\nFixed {fixed_count} files")

if __name__ == "__main__":
    main()

