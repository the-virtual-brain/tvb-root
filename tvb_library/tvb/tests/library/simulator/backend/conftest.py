# -*- coding: utf-8 -*-
#
# Collection guards for the backend test directory.
#
# The compiled cpp_hybrid extension targets the CPython 3.12 stable ABI
# (cp312-abi3; see tvb/simulator/backend/cpp_hybrid/README.md, "Support
# matrix & wheel policy"), so it cannot load on older interpreters.  On
# < 3.12 the whole cpp_hybrid test surface is out of scope: ignore it at
# collection instead of failing every module with a ModuleNotFoundError.
# (The nb_hybrid reference backend is pure numba and collects everywhere.)

import sys

if sys.version_info < (3, 12):
    collect_ignore_glob = ["test_cpp_hybrid_*.py"]
