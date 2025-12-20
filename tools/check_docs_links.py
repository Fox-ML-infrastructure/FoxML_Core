#!/usr/bin/env python3
"""
Check all internal documentation links in the DOCS folder.
Verifies that all markdown links point to existing files.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import unquote

# Base directory for documentation
DOCS_ROOT = Path(__file__).parent.parent / "DOCS"
REPO_ROOT = Path(__file__).parent.parent


def find_all_markdown_files(root: Path) -> List[Path]:
    """Find all markdown files in the DOCS directory."""
    md_files = []
    for path in root.rglob("*.md"):
        md_files.append(path)
    return sorted(md_files)


def extract_links(content: str, file_path: Path) -> List[Tuple[str, str, int]]:
    """
    Extract all markdown links from content.
    Returns list of (link_text, link_url, line_number) tuples.
    """
    links = []
    
    # Pattern for markdown links: [text](url) or [text](url "title")
    # Also handles reference-style links: [text][ref] and [ref]: url
    link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
    ref_pattern = r'^\[([^\]]+)\]:\s*(.+)$'
    
    lines = content.split('\n')
    for line_num, line in enumerate(lines, 1):
        # Regular markdown links
        for match in re.finditer(link_pattern, line):
            link_text = match.group(1)
            link_url = match.group(2)
            # Remove title if present: "url" or 'url'
            if link_url.startswith('"') and link_url.endswith('"'):
                link_url = link_url[1:-1]
            elif link_url.startswith("'") and link_url.endswith("'"):
                link_url = link_url[1:-1]
            # Split on space to get URL if title is present
            link_url = link_url.split()[0] if link_url.split() else link_url
            links.append((link_text, link_url, line_num))
        
        # Reference-style links
        ref_match = re.match(ref_pattern, line)
        if ref_match:
            link_text = ref_match.group(1)
            link_url = ref_match.group(2).strip()
            links.append((link_text, link_url, line_num))
    
    return links


def resolve_link(link_url: str, from_file: Path) -> Path:
    """
    Resolve a relative link URL to an absolute path.
    Handles:
    - Relative paths: ../file.md, ./file.md, file.md
    - Anchor links: file.md#section
    - Absolute paths from repo root: /DOCS/file.md
    """
    # Remove anchor if present
    if '#' in link_url:
        link_url = link_url.split('#')[0]
    
    # Skip external URLs
    if link_url.startswith('http://') or link_url.startswith('https://') or link_url.startswith('mailto:'):
        return None
    
    # Skip anchor-only links
    if link_url.startswith('#'):
        return None
    
    # Decode URL encoding
    link_url = unquote(link_url)
    
    # Handle absolute paths from repo root
    if link_url.startswith('/'):
        resolved = REPO_ROOT / link_url.lstrip('/')
    # Handle relative paths
    elif link_url.startswith('../'):
        resolved = (from_file.parent / link_url).resolve()
    elif link_url.startswith('./'):
        resolved = (from_file.parent / link_url).resolve()
    else:
        # Relative to current file's directory
        resolved = (from_file.parent / link_url).resolve()
    
    # Normalize the path
    try:
        resolved = resolved.resolve()
    except (OSError, ValueError):
        return None
    
    return resolved


def check_link(link_url: str, from_file: Path, all_doc_files: Set[Path]) -> Tuple[bool, str]:
    """
    Check if a link points to an existing file.
    Returns (is_valid, error_message).
    """
    resolved = resolve_link(link_url, from_file)
    
    if resolved is None:
        return (True, "")  # External link or anchor-only, skip
    
    # Check if file exists
    if resolved.exists() and resolved.is_file():
        return (True, "")
    
    # Check if it's a markdown file (case-insensitive)
    if resolved.suffix.lower() == '.md':
        # Try case-insensitive match
        parent = resolved.parent
        if parent.exists():
            for existing_file in parent.iterdir():
                if existing_file.name.lower() == resolved.name.lower() and existing_file.suffix.lower() == '.md':
                    return (False, f"Case mismatch: found {existing_file.name}, linked {resolved.name}")
    
    # Check if it's in our documentation set
    if resolved in all_doc_files:
        return (True, "")
    
    # Check if it's a directory (might be intentional for README.md)
    if resolved.is_dir():
        readme_path = resolved / "README.md"
        if readme_path.exists():
            return (True, "")
        return (False, f"Directory exists but no README.md found")
    
    return (False, f"File not found: {resolved}")


def check_all_links() -> Dict[str, List[Tuple[str, int, str]]]:
    """
    Check all links in all documentation files.
    Returns dict mapping file paths to list of (link_text, line_num, error) tuples.
    """
    all_md_files = find_all_markdown_files(DOCS_ROOT)
    all_doc_files = set(all_md_files)
    
    broken_links = {}
    
    for md_file in all_md_files:
        try:
            content = md_file.read_text(encoding='utf-8')
            links = extract_links(content, md_file)
            
            file_broken = []
            for link_text, link_url, line_num in links:
                is_valid, error = check_link(link_url, md_file, all_doc_files)
                if not is_valid:
                    file_broken.append((link_text, line_num, error))
            
            if file_broken:
                broken_links[str(md_file.relative_to(DOCS_ROOT))] = file_broken
        except Exception as e:
            broken_links[str(md_file.relative_to(DOCS_ROOT))] = [("FILE_ERROR", 0, str(e))]
    
    return broken_links


def main():
    """Main entry point."""
    print("Checking documentation links in DOCS folder...")
    print(f"DOCS root: {DOCS_ROOT}")
    print()
    
    broken_links = check_all_links()
    
    if not broken_links:
        print("✅ All links are valid!")
        return 0
    
    print(f"❌ Found broken links in {len(broken_links)} file(s):\n")
    
    total_broken = 0
    for file_path, errors in sorted(broken_links.items()):
        print(f"📄 {file_path}")
        for link_text, line_num, error in errors:
            total_broken += 1
            print(f"   Line {line_num}: [{link_text}] - {error}")
        print()
    
    print(f"\nTotal broken links: {total_broken}")
    return 1


if __name__ == "__main__":
    exit(main())

