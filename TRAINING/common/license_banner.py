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
    - License type (source-available, free for non-commercial academic research)
    - Commercial use requirements (paid pilot required for orgs)
    - Links to legal documentation
    - Contact information for enterprise licensing
    """
    if suppress:
        return
    
    # Check if banner should be suppressed via environment variable
    # CRITICAL: Suppress in child processes (isolation runners) and non-interactive contexts
    if os.getenv("FOXML_SUPPRESS_BANNER", "").lower() in ("true", "1", "yes"):
        return
    
    # Suppress in child processes (identified by TRAINER_CHILD_FAMILY or isolation markers)
    if os.getenv("TRAINER_CHILD_FAMILY") or os.getenv("TRAINER_ISOLATION_CHILD"):
        return
    
    # Suppress if not a TTY (non-interactive, redirected output, etc.)
    if not sys.stdout.isatty() and not os.getenv("FOXML_FORCE_BANNER", "").lower() in ("true", "1", "yes"):
        return
    
    # Use logger if available and configured, otherwise print directly to stderr
    # Print to stderr so it appears even if stdout is redirected
    try:
        if logger and logger.handlers:
            log_func = logger.info
        else:
            log_func = lambda msg: print(msg, file=sys.stderr)
    except Exception:
        log_func = lambda msg: print(msg, file=sys.stderr)
    
    banner_lines = [
        "=" * 60,
        "FOXML CORE — ML CROSS-SECTIONAL INFRASTRUCTURE",
        "=" * 60,
        "",
        "LICENSING:",
        "",
        "✅ AGPL v3: Open research, experimentation, and projects that can comply",
        "   with AGPL copyleft (must publish modifications if deployed as service)",
        "❌ COMMERCIAL LICENSE: Proprietary deployments, production systems,",
        "   closed-source integrations, or any use where AGPL compliance is not possible",
        "",
        "FoxML Core is dual-licensed: AGPL v3 + Commercial",
        "",
        "Note: If you deploy AGPL-licensed software as a service (including internal",
        "services), you must publish your modifications under AGPL v3 unless you have",
        "a commercial license.",
        "",
        "💰 PRICING:",
        "Starts at $120,000/year per desk/team",
        "  • Core Desk (1–5 users): $120,000/year",
        "  • Team Desk (6–15 users): $180,000/year",
        "  • Division (16–40 users): $300,000/year",
        "  • Enterprise: $450,000–$900,000+/year",
        "",
        "📋 EVALUATION:",
        "1. 30-day $0 evaluation (strict limits: non-production, no client deliverables)",
        "2. Paid Pilot: $35,000 (4–6 weeks, credited 100% to Year 1 if converted)",
        "3. Annual License (starts at $120k/year)",
        "",
        "📧 For licensing inquiries: " + _CONTACT_EMAIL,
        "   Subject: FoxML Core Commercial Licensing",
        "",
        "📚 Documentation:",
        "  ⭐ LEGAL/QUICK_REFERENCE.md (one-page summary — start here)",
        "  📋 LEGAL/SUBSCRIPTIONS.md (detailed pricing and process)",
        "  📄 LICENSE (free use terms)",
        "  📄 COMMERCIAL_LICENSE.md (commercial license terms)",
        "",
        "Note: Licensing is scoped to business unit/desk, not parent company headcount.",
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
