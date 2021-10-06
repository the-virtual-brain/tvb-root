# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
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
Tests full region mapping when both surface and sub-cortical regions are present.

.. moduleauthor:: Borana Dollomaja <borana.dollomaja@univ-amu.fr>
"""

import numpy

from tvb.simulator.lab import *
from tvb.tests.library.base_testcase import BaseTestCase


class TestRegionMapping(BaseTestCase):

    def setUp(self, monitors=(monitors.Raw(),)):
        """
        Initialize the structural information, coupling function, integrator, 
        monitors, surface and stimulation.
        """
        # Connectome
        con = connectivity.Connectivity.from_file("connectivity_192.zip")
        con.configure()

        # Surface and local connectivity kernel
        surf = cortex.Cortex.from_file(local_connectivity_file="local_connectivity_16384.mat")
        surf.region_mapping_data.connectivity = con
        surf.configure()

        # Model
        oscilator = models.Generic2dOscillator()

        # monitors[0].spatial_mask = RegionMapping.full_region_mapping(surf, con)

        self.sim = simulator.Simulator(
            conduction_speed=1.0,
            coupling= coupling.Difference(a=numpy.array([0.01])),
            surface=surf,
            integrator=integrators.Identity(dt=1.0),
            simulation_length=10.0,
            connectivity=con,
            model=oscilator,
            monitors=monitors
        )
        self.sim.configure()

    def test_spatial_average_monitor(self):
        self.setUp(monitors=(monitors.SpatialAverage(period=1.),))
        result = self.sim.run(simulation_length = 10)
        assert result is not None