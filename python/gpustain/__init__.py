"""gpustain — GPU-accelerated H&E stain normalization for Python.

A thin pybind11 wrapper around the C++ `stainkit` library. The Python
API is intentionally minimal: it only exposes the entry points a
typical notebook user needs.

Example
-------
>>> import gpustain
>>> img = gpustain.read_image("patches/patient_001.png")
>>> out = gpustain.run(img, target="default")
>>> gpustain.write_image(out.normalised_array(), "out.png")
"""

from __future__ import annotations

__all__ = [
    "is_cuda_available",
    "read_image",
    "write_image",
    "run",
    "Result",
    "BenchmarkRecord",
]

# `gpustain` is a pybind11 module compiled into a single shared
# library. The import below resolves to the binding declared in
# src/bindings/bindings.cpp.
from gpustain import (  # type: ignore  # noqa: E402
    BenchmarkRecord,
    Result,
    is_cuda_available,
    read_image,
    run,
    write_image,
)

__version__ = "0.1.0"
