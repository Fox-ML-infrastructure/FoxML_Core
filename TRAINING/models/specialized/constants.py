# MIT License - see LICENSE file

"""Shared constants and global variables for specialized models."""

import os
import logging

logger = logging.getLogger(__name__)

# Import FAMILY_CAPS from core_utils
try:
    from TRAINING.common.utils.core_utils import FAMILY_CAPS, assert_no_nan, tf_available, ngboost_available
except ImportError:
    # Fallback if core_utils not available
    logger.warning("TRAINING.utils.core_utils not available, using fallback constants")
    FAMILY_CAPS = {}
    def assert_no_nan(df, cols, name):
        pass
    def tf_available():
        try:
            import tensorflow as tf
            return True
        except:
            return False
    def ngboost_available():
        try:
            import ngboost
            return True
        except:
            return False

# Polars usage flag
USE_POLARS = os.getenv("USE_POLARS", "1") == "1"
if USE_POLARS:
    try:
        import polars as pl
    except ImportError:
        logger.warning("Polars requested but not available, falling back to pandas")
        USE_POLARS = False
        pl = None
else:
    pl = None

# TensorFlow device (set by setup_tf)
TF_DEVICE = '/CPU:0'
tf = None

# Strategy support flag
STRATEGY_SUPPORT = False  # Can be enabled if strategy modules are available

__all__ = [
    'FAMILY_CAPS',
    'assert_no_nan',
    'tf_available',
    'ngboost_available',
    'USE_POLARS',
    'pl',
    'TF_DEVICE',
    'tf',
    'STRATEGY_SUPPORT',
]
