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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Test history in simulator.

.. moduleauthor:: Marmaduke Woodman <marmaduke.woodman@univ-amu.fr>

"""

import numpy
from tvb.basic.neotraits.api import List
from tvb.datatypes.connectivity import Connectivity
from tvb.simulator.coupling import SparseCoupling
from tvb.simulator.integrators import Identity
from tvb.simulator.models.base import Model
from tvb.simulator.monitors import Raw
from tvb.simulator.simulator import Simulator
from tvb.tests.library.base_testcase import BaseTestCase


class IdCoupling(SparseCoupling):
    """Implements an identity coupling function."""

    def pre(self, x_i, x_j):
        return x_j

    def post(self, gx):
        return gx


class Sum(Model):
    nvar = 1
    _nvar = 1
    state_variable_range = {'x': [0, 100]}
    variables_of_interest = List(of=str, default=('x',), choices=('x',))
    state_variables = ('x',)
    cvar = numpy.array([0])

    def dfun(self, X, coupling, local_coupling=0):
        return X + coupling + local_coupling


class TestsExactPropagation(BaseTestCase):

    def build_simulator(self, n=4):

        self.conn = numpy.zeros((n, n))  # , numpy.int32)
        for i in range(self.conn.shape[0] - 1):
            self.conn[i, i + 1] = 1

        self.dist = numpy.r_[:n * n].reshape((n, n))
        self.dist = numpy.array(self.dist, dtype=float)

        self.sim = Simulator(
            conduction_speed=1.0,
            coupling=IdCoupling(),
            surface=None,
            stimulus=None,
            integrator=Identity(dt=1.0),
            initial_conditions=numpy.ones((n * n, 1, n, 1)),
            simulation_length=10.0,
            connectivity=Connectivity(region_labels=numpy.array(['']), weights=self.conn, tract_lengths=self.dist,
                                      speed=numpy.array([1.0]), centres=numpy.array([0.0])),
            model=Sum(),
            monitors=(Raw(),),
        )

        self.sim.configure()

    def test_propagation(self):
        n = 4
        self.build_simulator(n=n)
        # x = numpy.zeros((n, ))
        xs = []
        for (t, raw), in self.sim(simulation_length=10):
            xs.append(raw.flat[:].copy())
        xs = numpy.array(xs)
        xs_ = numpy.array([[2., 2., 2., 1.],
                           [3., 3., 3., 1.],
                           [5., 4., 4., 1.],
                           [8., 5., 5., 1.],
                           [12., 6., 6., 1.],
                           [17., 7., 7., 1.],
                           [23., 8., 8., 1.],
                           [30., 10., 9., 1.],
                           [38., 13., 10., 1.],
                           [48., 17., 11., 1.]])
        assert numpy.allclose(xs, xs_)
