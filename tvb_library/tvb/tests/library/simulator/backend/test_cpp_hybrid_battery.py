# -*- coding: utf-8 -*-
"""Mirror of the nb_hybrid test battery, running against CppHybridBackend.

Patches NbHybridBackend with the C++ backend before importing the canonical
test module, so the full battery of parity/regression/sweep tests exercises
the C++ backend.  Tests that the C++ backend does not support yet are skipped
via explicit xfail markers recorded in SUPPORTED / UNSUPPORTED below.
"""

import tvb.simulator.backend.nb_hybrid as _nbh
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

# Patch, import (the canonical tests bind the patched backend as their
# module global, which is what makes THIS battery exercise the C++
# backend), then restore immediately: without the restore the patch
# leaks to every later import in the same pytest session (e.g. a full
# tvb_library test run), silently running the numba-reference tests
# against the C++ backend.
_nb_real_backend = _nbh.NbHybridBackend
_nbh.NbHybridBackend = CppHybridBackend

from tvb.tests.library.simulator.backend.test_nb_hybrid import *  # noqa: E402,F401,F403

_nbh.NbHybridBackend = _nb_real_backend
from tvb.tests.library.simulator.backend import test_nb_hybrid as _battery  # noqa: E402,F401
