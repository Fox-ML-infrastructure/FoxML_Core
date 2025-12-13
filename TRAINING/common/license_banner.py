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
"""

"""
License and Commercial Use Banner

Prints a professional banner on startup directing users to licensing and pricing information.
This ensures compliance and helps capture value from automated systems that clone the repo.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Contact information (can be overridden via environment variable)
# Default contact email from README.md and LEGAL docs
_CONTACT_EMAIL = os.getenv("FOXML_CONTACT_EMAIL", "jenn.lewis5789@gmail.com")

# Project root (for finding legal docs)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def print_license_banner(suppress: bool = False):
    """
    Print a professional license and commercial use banner.
    
    This banner appears in logs and terminal output, ensuring users see licensing
    information even when using automated systems that clone the repo.
    
    Args:
        suppress: If True, skip printing the banner (useful for testing or if disabled via config)
    
    The banner includes:
    - License type (AGPL-3.0)
    - Commercial use requirements
    - Links to legal documentation
    - Contact information for enterprise licensing
    """
    if suppress:
        return
    
    # Check if banner should be suppressed via environment variable
    if os.getenv("FOXML_SUPPRESS_BANNER", "").lower() in ("true", "1", "yes"):
        return
    
    # Use logger if available, otherwise print directly
    log_func = logger.info if logger else print
    
    banner_lines = [
        "=" * 60,
        "FOXML CORE — ML CROSS-SECTIONAL INFRASTRUCTURE",
        "=" * 60,
        "",
        "LICENSING & COMMERCIAL USE:",
        "",
        "⚠️  BUSINESS USE REQUIRES A COMMERCIAL LICENSE",
        "",
        "This software is licensed under AGPL-3.0 for personal/academic use.",
        "Organizations using this software in production or revenue-generating",
        "environments require a commercial license.",
        "",
        "📅 30-DAY EVALUATION PERIOD:",
        "Commercial organizations may evaluate this software for up to 30 days",
        "from initial download for testing and evaluation purposes. Continued",
        "use after the evaluation period requires a commercial license.",
        "",
        "📄 License: See LICENSE file in repository root",
        "📋 Commercial Terms: LEGAL/SUBSCRIPTIONS.md",
        "📚 Legal Documentation: LEGAL/README.md",
        "📧 Enterprise Licensing: " + _CONTACT_EMAIL,
        "",
        "For complete licensing information, see:",
        "  - LEGAL/SUBSCRIPTIONS.md (pricing and commercial terms)",
        "  - LEGAL/LICENSING.md (license details)",
        "  - LEGAL/COMMERCIAL_USE.md (commercial use guidelines)",
        "  - COMMERCIAL_LICENSE.md (commercial license terms)",
        "",
        "=" * 60,
        ""
    ]
    
    for line in banner_lines:
        log_func(line)


def print_license_banner_once():
    """
    Print license banner only once per Python process.
    
    Uses a module-level flag to ensure the banner is printed only once,
    even if this function is called multiple times.
    """
    if not hasattr(print_license_banner_once, '_printed'):
        print_license_banner_once._printed = True
        print_license_banner()
    else:
        # Already printed, skip
        pass
