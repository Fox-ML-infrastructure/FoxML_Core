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

"""
Data Processing Utilities

Common utilities used across data processing modules:
- memory_manager: MemoryManager for memory-efficient processing
- logging_setup: Centralized logging configuration
- schema_validator: Schema validation and expectations
- io_helpers: I/O utilities for Polars (io_safe_scan)
- bootstrap: Exchange calendar loading with guards
"""


from .memory_manager import MemoryManager, MemoryConfig
from .logging_setup import CentralLoggingManager
from .schema_validator import SchemaExpectations, validate_schema
from .bootstrap import load_cal_guarded

__all__ = [
    "MemoryManager",
    "MemoryConfig",
    "CentralLoggingManager",
    "SchemaExpectations",
    "validate_schema",
    "load_cal_guarded",
]

