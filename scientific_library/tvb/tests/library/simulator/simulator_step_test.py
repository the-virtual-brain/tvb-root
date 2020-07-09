# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Test for tvb.simulator.simulator module

.. moduleauthor:: Lionel kusch <lionel.kusch@univ-amu.fr>

"""
import tvb.simulator.lab as lab
from tvb.tests.library.base_testcase import BaseTestCase
import numpy as np

class TestStep(BaseTestCase):
    def _sim(self):
        dt = 0.1
        model=lab.models.ReducedWongWang()
        connectivity = lab.connectivity.Connectivity().from_file()
        coupling = lab.coupling.Linear()
        integrator = lab.integrators.HeunDeterministic(dt=0.1)
        monitor=(lab.monitors.Raw(),)
        return model,connectivity,coupling, integrator,monitor,dt

    def test_no_modification(self):
        model,connectivity,coupling, integrator,monitor,dt = self._sim()
        for j in np.arange(0.0,10.0,0.1):
            simulator = lab.simulator.Simulator(model=model, connectivity=connectivity,
                                                coupling=coupling, integrator=integrator,
                                                monitors=monitor)
            simulator.configure()
            time = 0.0
            for i in simulator(j,test_step=True):
                time = i[0][0]
            assert time == j

    def test_modification(self):
        model,connectivity,coupling, integrator,monitor,dt = self._sim()
        for j in np.arange(0.0,10.0,0.1):
            simulator = lab.simulator.Simulator(model=model, connectivity=connectivity,
                                                coupling=coupling, integrator=integrator,
                                                monitors=monitor)
            simulator.configure()
            time = 0.0
            for i in simulator(j,test_step=False):
                time = i[0][0]
            assert time == j
