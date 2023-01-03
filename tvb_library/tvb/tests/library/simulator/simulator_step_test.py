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
Test for tvb.simulator.simulator module

.. moduleauthor:: Lionel kusch <lionel.kusch@univ-amu.fr>

"""
import tvb.simulator.lab as lab
from tvb.tests.library.base_testcase import BaseTestCase
import numpy as np
import pytest

class TestStep(BaseTestCase):
    def _sim(self):
        dt = 0.1
        model=lab.models.ReducedWongWang()
        connectivity = lab.connectivity.Connectivity().from_file()
        coupling = lab.coupling.Linear()
        integrator = lab.integrators.HeunDeterministic(dt=dt)
        monitor=(lab.monitors.Raw(),)
        return model,connectivity,coupling, integrator,monitor,dt

    def test_simulation_length(self):
        model,connectivity,coupling, integrator,monitor,dt = self._sim()
        for t in np.arange(0.1,10.0,0.1):
            simulator = lab.simulator.Simulator(model=model, connectivity=connectivity,
                                                coupling=coupling, integrator=integrator,
                                                monitors=monitor)
            simulator.configure()
            for (time,_), in simulator(simulation_length=t):
                pass
            assert time > t or np.isclose(time, t)


    def test_n_steps(self):
        model,connectivity,coupling, integrator,monitor,dt = self._sim()
        for n in range(1,10):
            simulator = lab.simulator.Simulator(model=model, connectivity=connectivity,
                                                coupling=coupling, integrator=integrator,
                                                monitors=monitor)
            simulator.configure()
            
            for i, ((time,_),) in enumerate(simulator(n_steps=n)):
                assert np.isclose(time, (i+1)*dt)
            assert i == n-1

    def test_n_steps_type(self):
        model,connectivity,coupling, integrator,monitor,dt = self._sim()
        simulator = lab.simulator.Simulator(model=model, connectivity=connectivity,
                                            coupling=coupling, integrator=integrator,
                                            monitors=monitor)
        simulator.configure()
        with pytest.raises(TypeError) as context:
            for _ in simulator(n_steps=1.1):
                pass
        assert('n_steps' in str(context.value))
