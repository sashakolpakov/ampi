"""
Build the AMPI C++ extension (_ampi_ext).

If pybind11 is not installed the extension is silently skipped and the
package falls back to the numba implementations in ampi/_kernels.py.
"""

import sys
import platform
from setuptools import setup, Extension


def _ext_modules():
    try:
        import pybind11
    except ImportError:
        return []

    extra_compile = ["-O3", "-std=c++17", "-fvisibility=hidden"]
    extra_link    = []

    if platform.system() == "Darwin":
        import subprocess, shutil
        if shutil.which("gcc") and "Apple" not in subprocess.getoutput("gcc --version"):
            extra_compile += ["-march=native"]
        else:
            extra_compile += ["-mcpu=native"]   # Apple clang on arm64
    elif platform.system() == "Linux":
        extra_compile += ["-march=native"]
    elif platform.system() == "Windows":
        # MSVC flags
        extra_compile = ["/O2", "/std:c++17"]

    return [
        Extension(
            name="ampi._ampi_ext",
            sources=["ampi/_ext.cpp"],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=extra_compile,
            extra_link_args=extra_link,
        )
    ]


setup(ext_modules=_ext_modules())
