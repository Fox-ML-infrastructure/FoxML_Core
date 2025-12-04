#!/usr/bin/env python3

"""
Copyright (c) 2025 Fox ML Infrastructure

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


