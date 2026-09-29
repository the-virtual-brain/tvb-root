# -*- coding: utf-8 -*-
"""Mirror of the nb_hybrid test battery, running against CppHybridBackend.

Patches NbHybridBackend with the C++ backend before importing the canonical
test module, so the full battery of parity/regression/sweep tests exercises
the C++ backend.  Tests that the C++ backend does not support yet are skipped
via explicit xfail markers recorded in SUPPORTED / UNSUPPORTED below.
"""

import tvb.simulator.backend.nb_hybrid as _nbh
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

_nbh.NbHybridBackend = CppHybridBackend

from tvb.tests.library.simulator.backend.test_nb_hybrid import *  # noqa: E402,F401,F403
from tvb.tests.library.simulator.backend import test_nb_hybrid as _battery  # noqa: E402,F401
