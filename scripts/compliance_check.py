#!/usr/bin/env python3
"""
Aurora License Compliance Checker

This script prints license compliance warnings and reminders.
No telemetry or tracking - just awareness.
"""

import os
import sys
from datetime import datetime

def print_warning():
    """Print the license compliance warning."""
    print("\n" + "="*70)
    print(" " * 15 + "⚠️  LICENSE COMPLIANCE NOTICE")
    print("="*70)
    print("\nThis software (Aurora) is licensed under GNU AGPL v3.0 with a")
    print("mandatory LGBTQ+ Protective Use Clause.\n")
    print("INSTITUTIONAL USE REQUIREMENTS:")
    print("  ✓ LGBTQ+ nondiscrimination protections must exist")
    print("  ✓ All benefit must be reinvested into LGBTQ+ protection or public service")
    print("  ✓ No commercial finance, trading, or profit-seeking use")
    print("  ✓ No use that harms marginalized communities")
    print("  ✓ No use by institutions that harass or discriminate\n")
    print("PROHIBITED USES:")
    print("  ✗ Hedge funds, proprietary trading, market-making")
    print("  ✗ Commercial profit-seeking or financial exploitation")
    print("  ✗ Systems that discriminate, surveil, or harm marginalized groups")
    print("  ✗ Use by institutions with documented discrimination records\n")
    print("See EXCEPTION.md and ENFORCEMENT.md for full terms.")
    print("="*70 + "\n")

def main():
    """Main entry point."""
    print_warning()
    
    # Optional: Check for environment variable to suppress
    if 'AURORA_SUPPRESS_LICENSE_WARNING' in os.environ:
        return
    
    # Print reminder
    print("To suppress this warning, set: AURORA_SUPPRESS_LICENSE_WARNING=1")
    print("(You must still comply with the license terms.)\n")

if __name__ == "__main__":
    main()

