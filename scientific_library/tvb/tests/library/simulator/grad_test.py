# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
Gradient tests.

.. moduleauthor:: Marmaduke Woodman <mmwoodman@gmail.com>

"""

import pytest
from tvb.simulator.models.linear import Linear
from tvb.simulator.integrators import EulerDeterministic
from tvb.simulator.gradients import HasGradient, has_gradient
from autograd import grad
import autograd.numpy as np
from numpy.testing import assert_allclose


class TestGradients:

    def setup_method(self):
        self.dfun = Linear().make_dfun(numpy=np)
        self.scheme = EulerDeterministic(dt=0.1).make_scheme(self.dfun)
        def example(scl):
            X = np.ones((1, 5, 1))
            C = np.zeros((1, 5, 1)) + scl
            Xn = self.scheme(X, C)
            sse = np.sum(np.square(X - Xn))
            return sse
        self.scalar_func = example
        self.grad_func = grad(example)

    def test_scalar_func(self):
        assert self.scalar_func(0.4) == 4.608

    def test_grad(self):
        assert self.grad_func(0.4) == -0.96


class TestHasGradient:

    def setup_method(self):
        class TestClass(HasGradient):
            pass
        self.test_class = TestClass()

    def test_has_class(self):
        assert has_gradient(self.test_class)


class TestTimeSeriesGradient:

    def setup_method(self):
        self.dfun = Linear().make_dfun(numpy=np)
        self.scheme = EulerDeterministic(dt=0.1).make_scheme(self.dfun)
        self.init = np.random.randn(1, 5, 1)
        self.trace = [self.init]
        for i in range(10):
            self.trace.append(self.scheme(self.trace[-1], self.trace[-1]))
        self.trace = np.array(self.trace)
        def fn(init):
            trace = [init]
            for i in range(10):
                trace.append(self.scheme(trace[-1], trace[-1]))
            trace = np.array(trace)
            return np.sum(np.square(trace - self.trace))
        self.fn = fn
        self.grad_func = grad(self.fn)

    def test_grad(self):
        init = np.random.randn(1, 5, 1)
        err = np.sum(np.square(init - self.init))
        for i in range(5):
            init += -0.1 * self.grad_func(init)
            new_err = np.sum(np.square(init - self.init))
            assert new_err < err
            err = new_err


class TestGradDelays:
    "Test taking autodiffing through time delay ring buffer."

    class CatRingBuffer:
        "Concatenating ring buffer."

        def __init__(self, nn, nt):
            "setup data for delay buffer."
            self.nn = nn
            self.nt = nt
            self.state = np.r_[:self.nn].astype('d')
            self.trace = np.zeros((self.nt, self.nn))
            self.trpos = -1
            self.update(self.state)

        def update(self, new_state):
            "Non-in-place update for delay buffer 'trace'."
            self.state = new_state
            self.trpos = (self.trpos + 1) % self.nt
            self.trace = np.concatenate([
                self.trace[:self.trpos],
                self.state.reshape((1, -1)),
                self.trace[self.trpos + 1:]])

        def read(self, lag=None):
            "Read delayed data from buffer."
            # for the purposes of testing autodiff, the delays don't
            # matter, so we choose something simple for testing.
            lag = lag or self.nt - 1
            return self.trace[(self.trpos + self.nt - lag) % self.nt]

    def _loop_iter(self, k=0):
        "Loop body."
        lag2 = self.crb.read()
        self.crb.state = self.crb.state  * 0.5 + k * lag2.mean()
        self.crb.update(self.crb.state)

    def test_loop_k0(self):
        "Test loop for k=0 for known delay values."
        self.crb = self.CatRingBuffer(2, 3)
        for i in range(10):
            self._loop_iter(k=0)
            lag1 = self.crb.read(lag=1)
            lag0 = self.crb.state
            assert_allclose(lag1, 2.0 * lag0)
            assert self.crb.trpos == ((i + 1) % self.crb.nt)
            assert self.crb.trace.shape == (self.crb.nt, self.crb.nn)

    def _run_loop(self, k):
        "Run simulation loop for value of k."
        self.crb = self.CatRingBuffer(2, 3)
        trace = []
        for i in range(10):
            self._loop_iter(k=k)
            trace.append(self.crb.state)
        return np.array(trace)

    def _sse_loop(self, k, data):
        "Eval sum squared error of simulation vs data."
        sim = self._run_loop(k)
        sse = np.sum(np.square(sim - data))
        return sse

    def test_opt(self):
        "Attempt optimization by gradient descent."
        k = 0.15
        data = self._run_loop(k)
        # guess w/ initial sse
        k_hat = 0.1
        sse_i = self._sse_loop(k_hat, data)
        grad_sse = grad(lambda k_i: self._sse_loop(k_i, data))
        # do opt
        for i in range(5):
            g = grad_sse(k_hat)
            k_hat += -0.1 * g
            sse_ip1 = self._sse_loop(k_hat, data)
            assert sse_ip1 < sse_i
            sse_i = sse_ip1
