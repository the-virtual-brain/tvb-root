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
from tvb.datatypes.projections import ProjectionSurfaceEEG
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

    def test_monitor_rawvoi(self):
        monitors.RawVoi()

    def test_monitor_tavg(self):
        monitor = monitors.TemporalAverage()
        assert monitor.period == self.default_period

    def test_monitor_afferentcoupling(self):
        monitors.AfferentCoupling()

    def test_monitor_afferentcouplingtavg(self):
        monitors.AfferentCouplingTemporalAverage()

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
        default_cortex.region_mapping_data.connectivity = white_matter

        sim = simulator.Simulator(model=oscillator, connectivity=white_matter, coupling=white_matter_coupling,
                                  integrator=heunint, monitors=mons, surface=default_cortex)

        with numpy.errstate(all='ignore'):
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

    def _build_test_sim(self):
        sim = simulator.Simulator(
            connectivity=connectivity.Connectivity.from_file('connectivity_192.zip'),
            monitors=(monitors.iEEG(
                sensors=SensorsInternal.from_file(),
                region_mapping=RegionMapping.from_file('regionMapping_16k_192.txt')
            ),)
        ).configure()
        return sim

    def _build_test_sim_eeg(self):
        eeg_monitor = monitors.EEG()
        eeg_monitor.sensors = sensors.SensorsEEG.from_file()
        eeg_monitor.region_mapping = RegionMapping.from_file('regionMapping_16k_192.txt')
        eeg_monitor.projection = ProjectionSurfaceEEG.from_file()

        sim = simulator.Simulator(
                connectivity=connectivity.Connectivity.from_file('connectivity_192.zip'),
                monitors=[eeg_monitor],
                simulation_length=100
        ).configure()
        return sim

    def test_gain_size(self):
        sim = self._build_test_sim()
        ieeg = sim.monitors[0]  # type: SensorsInternal
        n_sens, n_reg = ieeg.gain.shape
        assert ieeg.sensors.locations.shape[0] == n_sens
        assert sim.connectivity.number_of_regions == n_reg

    def test_gain_config_idempotent(self):
        "Check that rerunning the config doesn't increase matrix size"
        sim = self._build_test_sim_eeg()
        eeg, = sim.monitors
        eeg: monitors.EEG
        initial_gain_shape = eeg.gain.shape
        eeg.config_for_sim(sim)
        reconfig_gain_shape = eeg.gain.shape
        assert initial_gain_shape == reconfig_gain_shape
        eeg.config_for_sim(sim)
        rereconfig_gain_shape = eeg.gain.shape
        assert reconfig_gain_shape == rereconfig_gain_shape

    def test_gain_order(self):
        conn = connectivity.Connectivity()
        conn.generate_surrogate_connectivity(4)
        conn.centres /= 100.0
        conn.cortical = numpy.array([1, 0, 1, 1], numpy.bool_)
        seeg_sensors = SensorsInternal(
            locations=conn.centres,
            labels=conn.region_labels)
        region_mapping = RegionMapping(
            array_data=numpy.r_[:conn.number_of_regions],
            connectivity=conn)
        seeg_monitor = monitors.iEEG(
            sensors=seeg_sensors,
            region_mapping=region_mapping,
            )
        sim = simulator.Simulator(
            connectivity=conn,
            monitors=[seeg_monitor],
        )
        sim.configure()
        # NB: each sensor above is on one node, in same order
        # they must create NaN values in gain matrix
        # we zero NaNs
        # order of zeros by tell us order of node-sensor pairs
        zero_mask = seeg_monitor.gain == 0.0
        row, col = numpy.where(zero_mask)
        assert (row == col).all()


class TestSVEEG(BaseTestCase):
    "Test use of multiple state variables in monitors."

    def _build_test_sim_eeg(self):
        import numpy as np

        conn = connectivity.Connectivity.from_file()
        conn.speed = np.r_[70.0]
        cfun = coupling.Linear(a=np.r_[0.2], b = np.r_[0.0])

        eeg = monitors.EEG.from_file()
        eeg.period = 1.0
        eeg.variables_of_interest = np.r_[0]
        sim = simulator.Simulator(
            model=models.ReducedSetHindmarshRose(),
            integrator=integrators.HeunDeterministic(dt=0.0122),
            connectivity=conn,
            coupling=cfun,
            monitors=(eeg,),
            simulation_length=10.0
        )
        return sim

    def test_config_run(self):
        sim = self._build_test_sim_eeg()
        sim.configure()
        (_, d),  = sim.run()
        assert d.shape[2] == 65
