# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
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

"""
.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy as np
from tvb.tests.library.base_testcase import BaseTestCase
from tvb.simulator.hybrid import NetworkSet, Subnetwork, Projection, Simulator, Recorder
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.datatypes.patterns import StimuliRegion
from tvb.datatypes.equations import DiscreteEquation, TemporalApplicableEquation
from tvb.simulator.monitors import TemporalAverage


class TestHybrid1(BaseTestCase):
    
    def setup(self, nvois=None, jrmon=None):
        conn = Connectivity.from_file()
        conn.configure()
        np.random.seed(42)

        # XXX not sure what user api is warranted here
        class AtlasPart:
            CORTEX = 0
            THALAMUS = 1

        ix = np.random.randint(
            low=0, high=2, size=conn.number_of_regions)

        scheme = HeunDeterministic(dt=0.1)

        jrkwargs = {}
        fhnkwargs = {}
        if nvois:
            jrkwargs['variables_of_interest'] = JansenRit.variables_of_interest.default[:nvois]
            fhnkwargs['variables_of_interest'] = ReducedSetFitzHughNagumo.variables_of_interest.default[:nvois]

        # Create subnetworks with just their size and behavior
        cortex = Subnetwork(
            name='cortex',
            model=JansenRit(**jrkwargs),
            scheme=scheme,
            nnodes=(ix == AtlasPart.CORTEX).sum(),
        ).configure()

        if jrmon:
            cortex.add_monitor(jrmon)

        thalamus = Subnetwork(
            name='thalamus',
            model=ReducedSetFitzHughNagumo(**fhnkwargs),
            scheme=scheme,
            nnodes=(ix == AtlasPart.THALAMUS).sum(),
        ).configure()

        # Create projections with explicit weights from global connectivity
        cortex_indices = np.where(ix == AtlasPart.CORTEX)[0]
        thalamus_indices = np.where(ix == AtlasPart.THALAMUS)[0]

        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=0, target_cvar=1,
                    weights=conn.weights[thalamus_indices][:, cortex_indices],
                ),
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=5, target_cvar=0,
                    weights=conn.weights[thalamus_indices][:, cortex_indices],
                ),
                Projection(
                    source=thalamus, target=cortex,
                    source_cvar=0, target_cvar=1,
                    scale=1e-2,
                    weights=conn.weights[cortex_indices][:, thalamus_indices],
                ),
            ]
        )

        return conn, ix, cortex, thalamus, AtlasPart, nets

    def test_subnetwork(self):
        conn, ix, c, t, a, nets = self.setup()
        self.assert_equal((6, (ix == a.CORTEX).sum(), 1), c.zero_states().shape)
        self.assert_equal((2, (ix == a.CORTEX).sum(), 1), c.zero_cvars().shape)
        self.assert_equal((4, (ix == a.THALAMUS).sum(), 3), t.zero_states().shape)

    # TODO test_projection, but networkset tests projection

    def test_networkset(self):
        conn, ix, cortex, thalamus, a, nets = self.setup()
        # Create a new network set with modified weights
        cortex_indices = np.where(ix == a.CORTEX)[0]
        thalamus_indices = np.where(ix == a.THALAMUS)[0]
        
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=0, target_cvar=1,
                    weights=conn.weights[thalamus_indices][:, cortex_indices],
                ),
                Projection(
                    source=cortex, target=thalamus,
                    source_cvar=5, target_cvar=0,
                    weights=0.5 * conn.weights[thalamus_indices][:, cortex_indices],
                ),
                Projection(
                    source=thalamus, target=cortex,
                    source_cvar=0, target_cvar=1,
                    scale=1e-2,
                    weights=conn.weights[cortex_indices][:, thalamus_indices],
                ),
            ]
        )

        x = nets.zero_states()
        x.cortex[:] += 1
        x.thalamus[0] += 1

        c0 = nets.zero_cvars()
        p: Projection = nets.projections[0]
        c0.thalamus[1] += p.weights @ x.cortex[0] * p.scale
        p: Projection = nets.projections[1]
        c0.thalamus[0] += p.weights @ x.cortex[5] * p.scale
        p: Projection = nets.projections[2]
        c0.cortex[1] += p.weights @ x.thalamus[0] @ np.ones((3, 1))/3 * p.scale

        c1 = nets.cfun(x)

        for c0_, c1_ in zip(c0, c1):
            np.testing.assert_allclose(c0_, c1_)

    def test_netset_step(self):
        conn, ix, cortex, thalamus, a, nets = self.setup()
        x = nets.zero_states()
        nx = nets.step(0, x)
        self.assert_equal(
            [(6, 37, 1), (4, 39, 3)], nx.shape
        )

    def test_sim(self):
        conn, ix, cortex, thalamus, a, nets = self.setup(nvois=2)
        tavg = TemporalAverage(period=1.0)
        sim = Simulator(
            nets=nets,
            simulation_length=10,
            monitors=[tavg],
        )
        sim.configure()
        (t,y), = sim.run()
        self.assert_equal(10, len(t))
        self.assert_equal((10, 2, 76, 1), y.shape)

    def test_sim_jrmon(self):
        jrmon = TemporalAverage(period=1.0)
        conn, ix, cortex, thalamus, a, nets = self.setup(jrmon=jrmon)
        sim = Simulator(nets=nets, simulation_length=10)
        sim.configure()
        xs = sim.run()
        self.assert_equal(0, len(xs))
        rec: Recorder = cortex.monitors[0]
        nn = cortex.nnodes
        self.assert_equal((10, 4, nn, 1), rec.shape)
        t, y = rec.to_arrays()
        self.assert_equal((10, ), t.shape)
        self.assert_equal((10, 4, nn, 1), y.shape)


class TestHybrid2(BaseTestCase):

    def test_stim(self):
        conn = Connectivity.from_file()
        nn = conn.weights.shape[0]
        conn.configure()
        class MyStim(StimuliRegion):
            def __call__(self, t):
                return np.random.randn(self.connectivity.weights.shape[0])
        stim = MyStim(connectivity=conn)
        I = stim(5)
        self.assert_equal((nn,), I.shape)
        model = JansenRit()
        model.configure()
        # TODO
