#!/usr/bin/env python3
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
Main Training Script

Automatically ranks targets, selects features, and trains models.
This is the primary entry point for all training workflows.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import orchestrator
from TRAINING.orchestration.intelligent_trainer import IntelligentTrainer, main as orchestrator_main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Main entry point for intelligent training pipeline.
    
    This wrapper provides a cleaner interface and delegates to the orchestrator.
    """
    # Delegate to orchestrator's main function
    # This keeps all argument parsing and logic in one place
    return orchestrator_main()


if __name__ == "__main__":
    sys.exit(main())

