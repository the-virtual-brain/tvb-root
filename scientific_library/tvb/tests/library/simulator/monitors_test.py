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
Test monitors for specific properties such as default periods, correct
handling of projection matrices, etc.

.. moduleauthor:: Paula Sanz Leon <sanzleon.paula@gmail.com>
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy
from tvb.datatypes.surfaces import CorticalSurface
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.datatypes import sensors
from tvb.simulator import monitors, models, coupling, integrators, noise, simulator
from tvb.datatypes import connectivity
from tvb.datatypes.cortex import Cortex
from tvb.datatypes.region_mapping import RegionMapping
from tvb.datatypes.sensors import SensorsInternal


class TestMonitors(BaseTestCase):
    """
    Define test cases for monitors:
        - initialise each class
        - check default parameters (period)
        -
    """

    default_period = 0.9765625  # 1024Hz

    def test_monitor_raw(self):
        monitors.Raw()

    def test_monitor_tavg(self):
        monitor = monitors.TemporalAverage()
        assert monitor.period == self.default_period

    def test_monitor_gavg(self):
        monitor = monitors.GlobalAverage()
        assert monitor.period == self.default_period

    def test_monitor_savg(self):
        monitor = monitors.SpatialAverage()
        assert monitor.period == self.default_period

    def test_monitor_subsample(self):
        monitor = monitors.SubSample()
        assert monitor.period == self.default_period

    def test_monitor_eeg(self):
        monitor = monitors.EEG()
        assert monitor.period == self.default_period

    def test_monitor_meg(self):
        monitor = monitors.MEG()
        assert monitor.period == self.default_period

    def test_monitor_stereoeeg(self):
        """
        This has to be verified.
        """
        monitor = monitors.iEEG()
        monitor.sensors = sensors.SensorsInternal.from_file()
        assert monitor.period == self.default_period

    def test_monitor_bold(self):
        """
        This has to be verified.
        """
        monitor = monitors.Bold()
        assert monitor.period == 2000.0


class TestMonitorsConfiguration(BaseTestCase):
    """
    Configure Monitors

    """

    def test_monitor_bold(self):
        """
        This has to be verified.
        """
        monitor = monitors.Bold()
        assert monitor.period == 2000.0


class TestProjectionMonitorsWithSubcorticalRegions(BaseTestCase):
    """
    Cortical surface with subcortical regions, sEEG, EEG & MEG, using a stochastic
    integration. This test verifies the shapes of the projection matrices, and
    indirectly covers region mapping, etc.

    """

    # hard code parameters to smoke test
    speed = 4.0
    period = 1e3 / 1024.0  # 1024 Hz
    coupling_a = numpy.array([0.014])
    n_regions = 192

    def test_surface_sim_with_projections(self):

        # Setup Simulator obj
        oscillator = models.Generic2dOscillator()
        white_matter = connectivity.Connectivity.from_file('connectivity_%d.zip' % (self.n_regions,))
        white_matter.speed = numpy.array([self.speed])
        white_matter_coupling = coupling.Difference(a=self.coupling_a)
        heunint = integrators.HeunStochastic(
            dt=2 ** -4,
            noise=noise.Additive(nsig=numpy.array([2 ** -10, ]))
        )
        mons = (
            monitors.EEG.from_file(period=self.period),
            monitors.MEG.from_file(period=self.period),
            # monitors.iEEG.from_file(period=self.period),
            # SEEG projection data is not part of tvb-data on Pypi, thus this can not work generic
        )
        local_coupling_strength = numpy.array([2 ** -10])
        region_mapping = RegionMapping.from_file('regionMapping_16k_%d.txt' % (self.n_regions,))
        region_mapping.surface = CorticalSurface.from_file()
        default_cortex = Cortex.from_file()
        default_cortex.region_mapping_data = region_mapping
        default_cortex.coupling_strength = local_coupling_strength

        sim = simulator.Simulator(model=oscillator, connectivity=white_matter, coupling=white_matter_coupling,
                                  integrator=heunint, monitors=mons, surface=default_cortex)
        sim.configure()

        # check configured simulation connectivity attribute
        conn = sim.connectivity
        assert conn.number_of_regions == self.n_regions
        assert conn.speed == self.speed

        # test monitor properties
        lc_n_node = sim.surface.local_connectivity.matrix.shape[0]
        for mon in sim.monitors:
            assert mon.period == self.period
            n_sens, g_n_node = mon.gain.shape
            assert g_n_node == sim.number_of_nodes
            assert n_sens == mon.sensors.number_of_sensors
            assert lc_n_node == g_n_node

        # check output shape
        ys = {}
        mons = 'eeg meg seeg'.split()
        for key in mons:
            ys[key] = []
        for data in sim(simulation_length=3.0):
            for key, dat in zip(mons, data):
                if dat:
                    _, y = dat
                    ys[key].append(y)
        for mon, key in zip(sim.monitors, mons):
            ys[key] = numpy.array(ys[key])
            assert ys[key].shape[2] == mon.gain.shape[0]


class TestProjectionMonitorsWithNoSubcortical(TestProjectionMonitorsWithSubcorticalRegions):
    """
    Idem. but with the 76 regions connectivity, where no sub-cortical regions are included
    """
    n_regions = 76


class TestAllAnalyticWithSubcortical(BaseTestCase):
    """Test correct gain matrix shape for all analytic with subcortical nodes."""

    def test_gain_size(self):
        sim = simulator.Simulator(
            connectivity=connectivity.Connectivity.from_file('connectivity_192.zip'),
            monitors=(monitors.iEEG(
                sensors=SensorsInternal.from_file(),
                region_mapping=RegionMapping.from_file('regionMapping_16k_192.txt')
            ),)
        ).configure()

        ieeg = sim.monitors[0]  # type: SensorsInternal
        n_sens, n_reg = ieeg.gain.shape
        assert ieeg.sensors.locations.shape[0] == n_sens
        assert sim.connectivity.number_of_regions == n_reg
