#!/usr/bin/env python3
# MIT License - see LICENSE file

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

