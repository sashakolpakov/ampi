"""
Build the AMPI C++ extension (_ampi_ext).

If pybind11 is not installed the extension is silently skipped and the
package falls back to the numba implementations in ampi/_kernels.py.

BLAS backend detection order (sets -DAMPI_USE_* and links accordingly):
  macOS   → Apple Accelerate  (always available, zero extra dependency)
  Linux   → OpenBLAS  (via pkg-config or common prefix search)
          → Intel MKL (via MKLROOT env var)
          → native tiled SIMD fallback (no external BLAS needed)
  Windows → Intel MKL (via MKLROOT) or native fallback
"""

import os
import platform
import shutil
import subprocess
from setuptools import setup, Extension


# ── BLAS detection ────────────────────────────────────────────────────────────

def _detect_blas():
    """Return (defines, include_dirs, link_args) for the best available BLAS."""
    sys = platform.system()

    # ── macOS: Accelerate is always present ───────────────────────────────────
    if sys == "Darwin":
        return (
            [("AMPI_USE_ACCELERATE", None)],
            [],
            ["-framework", "Accelerate"],
        )

    # ── Linux / Windows: try OpenBLAS via pkg-config ─────────────────────────
    try:
        cflags = subprocess.check_output(
            ["pkg-config", "--cflags-only-I", "openblas"],
            stderr=subprocess.DEVNULL,
        ).decode().split()
        libs = subprocess.check_output(
            ["pkg-config", "--libs", "openblas"],
            stderr=subprocess.DEVNULL,
        ).decode().split()
        inc = [f[2:] for f in cflags if f.startswith("-I")]
        print("AMPI: found OpenBLAS via pkg-config")
        return ([("AMPI_USE_OPENBLAS", None)], inc, libs)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # ── Linux: try common OpenBLAS install prefixes ───────────────────────────
    for prefix in ["/usr", "/usr/local", "/opt/homebrew", "/opt/local",
                   "/usr/local/opt/openblas"]:
        header = os.path.join(prefix, "include", "openblas", "cblas.h")
        if not os.path.exists(header):
            header = os.path.join(prefix, "include", "cblas.h")
        if os.path.exists(header):
            inc_dir = os.path.dirname(header)
            lib_dir = os.path.join(prefix, "lib")
            print(f"AMPI: found OpenBLAS at {prefix}")
            return (
                [("AMPI_USE_OPENBLAS", None)],
                [inc_dir],
                [f"-L{lib_dir}", "-lopenblas"],
            )

    # ── Intel MKL: check MKLROOT env var ─────────────────────────────────────
    mklroot = os.environ.get("MKLROOT", "")
    if mklroot:
        inc_dir  = os.path.join(mklroot, "include")
        lib_dir  = os.path.join(mklroot, "lib", "intel64")
        if os.path.exists(inc_dir):
            print(f"AMPI: found Intel MKL at {mklroot}")
            mkl_libs = [
                f"-L{lib_dir}",
                "-lmkl_rt",          # single dynamic library (works for both LP64/ILP64)
                "-lpthread", "-lm", "-ldl",
            ]
            return ([("AMPI_USE_MKL", None)], [inc_dir], mkl_libs)

    # ── Fallback: native tiled SIMD kernel (no external dependency) ───────────
    print("AMPI: no external BLAS found — using built-in tiled SIMD kernel")
    return ([], [], [])


# ── Extension build ───────────────────────────────────────────────────────────

def _ext_modules():
    try:
        import pybind11
    except ImportError:
        return []

    extra_compile = ["-O3", "-std=c++17", "-fvisibility=hidden"]
    extra_link    = []

    if platform.system() == "Darwin":
        if shutil.which("gcc") and "Apple" not in subprocess.getoutput("gcc --version"):
            extra_compile += ["-march=native"]
        else:
            extra_compile += ["-mcpu=native"]   # Apple clang on arm64
    elif platform.system() == "Linux":
        extra_compile += ["-march=native"]
    elif platform.system() == "Windows":
        extra_compile = ["/O2", "/std:c++17"]

    blas_defines, blas_inc, blas_link = _detect_blas()

    return [
        Extension(
            name="ampi._ampi_ext",
            sources=["ampi/_ext.cpp"],
            include_dirs=[pybind11.get_include(), "ampi"] + blas_inc,
            define_macros=blas_defines,
            language="c++",
            extra_compile_args=extra_compile,
            extra_link_args=extra_link + blas_link,
        )
    ]


setup(ext_modules=_ext_modules())
