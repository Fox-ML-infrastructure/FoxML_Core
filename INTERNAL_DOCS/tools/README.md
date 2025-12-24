# Internal Tools

Internal utility scripts for repository maintenance and operations.

## regenerate_enterprise_pdfs.py

Regenerates all Enterprise Bundle PDFs from Markdown source files.

### Usage

```bash
# From repository root
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py

# Clean and regenerate (removes old PDFs first)
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --clean

# Verbose output
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --verbose

# Specify LaTeX engine
python SCRIPTS/internal/tools/regenerate_enterprise_pdfs.py --engine pdflatex
```

### Requirements

- `pandoc` - Install with: `sudo apt-get install pandoc` (Linux) or `brew install pandoc` (macOS)
- LaTeX engine (`pdflatex`, `lualatex`, or `xelatex`) - Install with: `sudo apt-get install texlive-latex-base` (Linux)

### Features

- Automatically handles Unicode characters (emojis, box-drawing, arrows) by converting them to ASCII equivalents
- Preprocesses markdown files to avoid LaTeX Unicode errors
- Generates all 26 enterprise PDFs from source files in `LEGAL/` and root `COMMERCIAL_LICENSE.md`
- Provides detailed error reporting if any PDFs fail to generate

### Output

All PDFs are generated in `ENTERPRISE_BUNDLE/` directory.

