#!/usr/bin/env python3
"""
License Notice

This script prints a brief license notice.
"""

import os
import sys

def print_notice():
    """Print the license notice."""
    print("\n" + "="*70)
    print(" " * 20 + "LICENSE NOTICE")
    print("="*70)
    print("\nThis software is licensed under GNU AGPL v3.0.")
    print("\nSee LICENSE file for full terms and conditions.")
    print("="*70 + "\n")

def main():
    """Main entry point."""
    print_notice()
    
    # Optional: Check for environment variable to suppress
    if 'AURORA_SUPPRESS_LICENSE_WARNING' in os.environ:
        return

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

