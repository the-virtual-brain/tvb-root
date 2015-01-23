
"""
Test history in simulator.

.. moduleauthor:: Marmaduke Woodman <mw@eml.cc>

"""

import numpy as np
try:
    from tvb.tests.library.base_testcase import BaseTestCase
except:
    BaseTestCase = object

from tvb.simulator import simulator as sim


class IdCoupling(sim.coupling_module.Coupling):
    "Implements an identity coupling function."
    def __call__(self, g_ij, x_i, x_j):
        return (g_ij * x_j).sum(axis=2).transpose((1, 0, 2))

class Sum(sim.models_module.Model):
    nvar = 1
    _nvar = 1
    state_variables_range = {'x': [0, 100]}
    variables_of_interest = sim.basic.Enumerate(default=['x'], options=['x'])
    cvar = np.array([0])
    def dfun(self, X, coupling, local_coupling=0):
        return X + coupling + local_coupling


class ExactPropagationTests(BaseTestCase):

    def build_simulator(self, n=4):
        self.conn = np.zeros((n, n), np.int32)
        for i in range(self.conn.shape[0] - 1):
            self.conn[i, i+1] = 1
        self.dist = np.r_[:n*n].reshape((n, n))
        self.sim = sim.Simulator(
            conduction_speed=1,
            coupling=IdCoupling(),
            surface=None,
            stimulus=None,
            integrator=sim.integrators_module.Identity(dt=1),
            initial_conditions=np.ones((n*n, 1, n, 1)),
            simulation_length=10,
            connectivity=sim.connectivity_dtype.Connectivity(
                weights=self.conn,
                tract_lengths=self.dist,
                speed=1
            ),
            model=Sum(),
            monitors=(sim.monitors_module.Raw(), ),
            )
        self.sim.configure()

    def test_propagation(self):
        n = 4
        self.build_simulator(n=n)
        x = np.zeros((n, ))
        xs = []
        for (t, raw), in self.sim():
            xs.append(raw.flat[:].copy())
        xs = np.array(xs)
        xs_ =np.array([[  2.,   2.,   2.,   1.],
                       [  3.,   3.,   3.,   1.],
                       [  5.,   4.,   4.,   1.],
                       [  8.,   5.,   5.,   1.],
                       [ 12.,   6.,   6.,   1.],
                       [ 17.,   7.,   7.,   1.],
                       [ 23.,   8.,   8.,   1.],
                       [ 30.,  10.,   9.,   1.],
                       [ 38.,  13.,  10.,   1.],
                       [ 48.,  17.,  11.,   1.]])
        self.assertTrue(np.allclose(xs, xs_))
