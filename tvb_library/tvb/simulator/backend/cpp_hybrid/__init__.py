# -*- coding: utf-8 -*-
"""C++ hybrid simulator backend for TVB.

Runtime-tight SIMD-enabled C++ kernels with control flow in C++ (no static
codegen). Python keeps control of the simulation; data arrays stay in NumPy
float32.  The extension is built with **nanobind + scikit-build-core** (cp312
abi3 wheel, editable in-place auto-rebuild); the previously vendored tvbk
package has been removed.

Notebook-facing helpers
-----------------------
* :mod:`.notebook` — convenience for multithreaded SIMD parameter sweeps
  (``sweep(network_set, nstep, coupling_scale=[...], **params)`` with tqdm
  progress, pandas :meth:`SweepResult.to_dataframe`, and feature tables).
* :mod:`.features` — data-feature computation over monitor time series
  (vendored subset of the VBI feature taxonomy, Apache-2.0).
"""

from ._cpp_hybrid import Sim, enable_generic_dfuns  # noqa: F401
from .backend import CppHybridBackend, SweepResult  # noqa: F401

__all__ = [
    "Sim", "enable_generic_dfuns", "CppHybridBackend", "SweepResult",
    "notebook", "features",
]
