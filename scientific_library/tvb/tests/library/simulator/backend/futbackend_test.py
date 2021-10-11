# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2022, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Tests for the Futhark backend.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np
from tvb.simulator.models.base import Model
from tvb.simulator.backend import fut

from .backendtestbase import (BaseTestCoupling, BaseTestDfun,
    BaseTestIntegrate, BaseTestSim)

class TestFtDfun(BaseTestDfun):

    def _test_dfun(self, model_: Model):
        class sim:
            model = model_
        template = r'''
<%include file="ft-dfuns.fut.mako"/>

entry kernel [n] (state:[n][2]f32) (cx:[n][2]f32): [n][2]f32  =
    let f x c = dfun (x[0],x[1]) (c[0],c[1]) ()
    let out = map2 f state cx
    in map (\ xi -> [xi.0, xi.1]) out
'''
        mod = fut.FutBackend().build_module(template, dict(sim=sim), print_source=True)
        cX = np.random.rand(128, 2) / 10.0
        state = np.random.rand(128, 2)
        drh, dVh = mod.from_futhark(mod.kernel(state, cX)).T
        dr, dV = sim.model.dfun(state.T, cX.T)
        np.testing.assert_allclose(drh, dr, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(dVh, dV, rtol=1e-4, atol=1e-5)

    def test_py_mpr_symmetric(self):
        "Test symmetric MPR model"
        self._test_dfun(self._prep_model())