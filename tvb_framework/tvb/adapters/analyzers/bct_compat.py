# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Compatibility patches for the bctpy library.
"""

import numpy as np
import bct.algorithms.modularity as bct_modularity


def _ls2ci(ls, zeroindexed=False):
    """
    Fixed copy of bct.algorithms.modularity.ls2ci.

    The bctpy 0.6.1 package on PyPI checks the input with np.size(ls), which fails with newer numpy
    when the modules have different sizes. This copy uses len(ls), as the fix on bctpy master does.
    See https://github.com/aestrivex/bctpy/issues/119
    """
    if ls is None or len(ls) == 0:
        return ()
    nr_indices = sum(map(len, ls))
    ci = np.zeros((nr_indices,), dtype=int)
    z = int(not zeroindexed)
    for i, module in enumerate(ls):
        for node in module:
            ci[node] = i + z
    return ci


def patch_bct_ls2ci():
    """
    Replace bct.algorithms.modularity.ls2ci, which fails with newer numpy, with the fixed copy.
    modularity_dir and modularity_und look up ls2ci in their module at call time, so they use the fixed copy.
    """
    bct_modularity.ls2ci = _ls2ci