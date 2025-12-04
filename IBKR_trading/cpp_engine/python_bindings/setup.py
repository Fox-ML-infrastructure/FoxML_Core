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

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11
import numpy

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "ibkr_trading_engine_py",
        [
            "trading_kernels.cpp",
        ],
        include_dirs=[
            # Path to pybind11 headers
            pybind11.get_include(),
            # Path to numpy headers
            numpy.get_include(),
            # Path to Eigen headers (adjust as needed)
            "/usr/include/eigen3",
            # Local include directory
            "../include",
        ],
        libraries=["m"],  # Math library
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="ibkr_trading_engine_py",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)
