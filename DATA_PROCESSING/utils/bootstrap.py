#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial

import importlib
import exchange_calendars as xc
from functools import lru_cache

@lru_cache(maxsize=1)
def load_cal_guarded(name: str = "XNYS"):
    """Load calendar instance with robust error handling.
    Uses instance methods instead of class methods to avoid hot-reload issues.
    """
    try:
        importlib.reload(xc)
        cal = xc.get_calendar(name)
        
        # Test instance methods instead of class methods
        if not hasattr(cal, 'schedule') or not hasattr(cal, 'sessions_in_range'):
            raise RuntimeError(f"Calendar {name} missing required instance methods")
        
        # Test that schedule property works (not method)
        try:
            # Get a small date range to test
            test_start = "2024-01-01"
            test_end = "2024-01-02"
            sessions = cal.sessions_in_range(test_start, test_end)
            if len(sessions) > 0:
                sched = cal.schedule.loc[sessions[0]:sessions[-1]]
                if not hasattr(sched, 'open') or not hasattr(sched, 'close'):
                    raise RuntimeError("Schedule missing open/close columns")
        except Exception as e:
            raise RuntimeError(f"Calendar instance validation failed: {e}")
        
        return cal
        
    except Exception as e:
        raise RuntimeError(f"Failed to load calendar {name}: {e}")


