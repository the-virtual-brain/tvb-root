# -*- coding: utf-8 -*-
"""Mirror of the nb_hybrid test battery, running against CppHybridBackend.

Loads the canonical nb_hybrid test module under a private name while
NbHybridBackend is substituted with the C++ backend, so the full battery of
parity/regression/sweep tests exercises the C++ backend.  Loading a copy
under a private name (instead of patching the shared module and importing
the canonical path) makes the substitution order-independent: whichever of
the two modules a pytest session collects first, the canonical
``test_nb_hybrid`` module is always imported afterwards with the untouched
Numba backend, and this battery always runs against C++.
"""

import importlib.util
import pathlib
import sys

import tvb.simulator.backend.nb_hybrid as _nbh
from tvb.simulator.backend.cpp_hybrid.backend import CppHybridBackend

_nb_real_backend = _nbh.NbHybridBackend
_nbh.NbHybridBackend = CppHybridBackend
try:
    _spec = importlib.util.spec_from_file_location(
        "tvb.tests.library.simulator.backend._cpp_battery_canonical",
        str(pathlib.Path(__file__).with_name("test_nb_hybrid.py")))
    _canonical = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _canonical
    _spec.loader.exec_module(_canonical)
finally:
    _nbh.NbHybridBackend = _nb_real_backend

# re-export the canonical tests from the isolated copy (import-* semantics)
for _name, _obj in list(vars(_canonical).items()):
    if not _name.startswith("_"):
        globals()[_name] = _obj
