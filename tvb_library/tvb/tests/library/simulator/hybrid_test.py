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
from tvb.simulator.hybrid import NetworkSet, Subnetwork, Projection
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.models import JansenRit, ReducedSetFitzHughNagumo
from tvb.simulator.integrators import HeunDeterministic
from tvb.tests.library.base_testcase import BaseTestCase


class TestHybrid1(BaseTestCase):
    
    def setup(self):
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

        cortex = Subnetwork(
            conn=conn,
            mask=ix == AtlasPart.CORTEX,
            name='cortex',
            model=JansenRit(),
            scheme=scheme,
        ).configure()

        thalamus = Subnetwork(
            conn=conn,
            mask=ix == AtlasPart.THALAMUS,
            name='thalamus',
            model=ReducedSetFitzHughNagumo(),
            scheme=scheme,
        ).configure()

        return conn, ix, cortex, thalamus, AtlasPart

    def test_subnetwork(self):
        conn, ix, c, t, a = self.setup()
        self.assert_equal((6, (ix == a.CORTEX).sum(), 1), c.zero_states().shape)
        self.assert_equal((2, (ix == a.CORTEX).sum(), 1), c.zero_cvars().shape)
        self.assert_equal((4, (ix == a.THALAMUS).sum(), 3), t.zero_states().shape)

    # TODO test_projection, but networkset tests projection

    def test_networkset(self):
        conn, ix, cortex, thalamus, a = self.setup()
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                Projection(source=cortex, target=thalamus, source_cvar=0, target_cvar=1),
                Projection(source=cortex, target=thalamus, source_cvar=5, target_cvar=0),
                Projection(source=thalamus, target=cortex, source_cvar=0, target_cvar=1, scale=1e-2),
            ]
        )
        nets.projections[1]._weights *= 0.5

        x = nets.zero_states()
        x.cortex[:] += 1
        x.thalamus[0] += 1

        c0 = nets.zero_cvars()
        p: Projection = nets.projections[0]
        c0.thalamus[1] += p._weights @ x.cortex[0] * p.scale
        p: Projection = nets.projections[1]
        c0.thalamus[0] += p._weights @ x.cortex[5] * p.scale
        p: Projection = nets.projections[2]
        c0.cortex[1] += p._weights @ x.thalamus[0] @ np.ones((3, 1))/3 * p.scale

        c1 = nets.cfun(x)

        for c0_, c1_ in zip(c0, c1):
            np.testing.assert_allclose(c0_, c1_)

    def test_netset_step(self):
        conn, ix, cortex, thalamus, a = self.setup()
        nets = NetworkSet(
            subnets=[cortex, thalamus],
            projections=[
                Projection(source=cortex, target=thalamus, source_cvar=0, target_cvar=1),
                Projection(source=cortex, target=thalamus, source_cvar=5, target_cvar=0),
                Projection(source=thalamus, target=cortex, source_cvar=0, target_cvar=1, scale=1e-2),
            ]
        )
        x = nets.zero_states()
        heun = HeunDeterministic(dt=1e-3)
        nx = nets.step(heun.scheme, x)
        self.assert_equal(
            [(6, 37, 1), (4, 39, 3)], nx.shape
        )
        
