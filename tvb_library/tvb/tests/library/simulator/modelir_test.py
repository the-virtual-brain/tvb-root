# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Currently tests for various backend templating.

TODO have just tests to cover various attributes of models have everything

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import unittest

from tvb.simulator.coupling import Sigmoidal, Linear
from tvb.simulator.models.infinite_theta import MontbrioPazoRoxin
from tvb.simulator.integrators import (EulerDeterministic, EulerStochastic,
    HeunDeterministic, HeunStochastic)


class TestModelIR(unittest.TestCase):
    
    def _test_model(self, model):
        for key in 'coupling_terms state_variable_dfuns parameter_names'.split():
            self.assertTrue(hasattr(model, key))

    def test_MontbrioPazoRoxin(self):
        self._test_model(MontbrioPazoRoxin())


class TestIntegratorIR(unittest.TestCase):

    def _test_integrator(self, integrator):
        for key in 'n_dx'.split():
            self.assertTrue(hasattr(integrator, key))

    def test_euler_deterministic(self): self._test_integrator(EulerDeterministic())
    def test_euler_stochast(self): self._test_integrator(EulerDeterministic())
    def test_heun_deterministic(self): self._test_integrator(HeunDeterministic())
    def test_heun_stochastic(self): self._test_integrator(HeunDeterministic())
