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

# common/threading_utils.py
"""
DEPRECATED shim — routes to unified threading utilities in threads.py
Kept for backward compatibility with existing imports.
"""
import os
from contextlib import contextmanager

# Pull from the new, correct module
from .threads import (
    default_threads as _default_threads,
    thread_guard as _thread_guard,
    set_estimator_threads as _set_estimator_threads,
)

def default_threads():
    """Get default thread count."""
    return _default_threads()

def env_guard(omp: int, mkl: int):
    """Set environment variables for threading (legacy interface)."""
    os.environ["OMP_NUM_THREADS"] = str(omp)
    os.environ["OPENBLAS_NUM_THREADS"] = str(mkl)
    os.environ["MKL_NUM_THREADS"] = str(mkl)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_DYNAMIC", "false")

@contextmanager
def thread_guard(num_threads: int | None = None, blas_threads: int | None = None, openmp_threads: int | None = None):
    """
    Thread guard context manager (legacy interface).
    Routes to the correct implementation in threads.py.
    
    Args:
        num_threads: Default thread count
        blas_threads: BLAS thread count
        openmp_threads: OpenMP thread count
    """
    # Map old signature to new behavior (openmp = num_threads if provided)
    o = openmp_threads or num_threads or _default_threads()
    b = blas_threads or o
    # Delegate to the correct implementation
    with _thread_guard(omp=o, mkl=b):
        yield

def set_estimator_threads(est, n_jobs: int):
    """
    Set thread count on estimators (legacy interface).
    Routes to the correct implementation in threads.py.
    """
    return _set_estimator_threads(est, n_jobs)