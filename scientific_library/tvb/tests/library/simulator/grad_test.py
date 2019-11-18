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
from tvb.simulator.history import CatRingBuffer, NoopBuffer
from autograd import grad
from autograd.extend import primitive, defvjp
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

    class Loop:

        def __init__(self, k, buf, lag=None):
            "Setup loop."
            self.k = k
            self.buf = buf
            self.lag = lag

        def step(self):
            "Loop body."
            lagged = self.buf.read(lag=self.lag)
            state = self.buf.state
            new_state = state * 0.5 + self.k * lagged.mean()
            self.buf.update(new_state)
            return new_state

    def setup_method(self):
        self.nn = 2
        self.init = np.r_[:self.nn].astype('d')

    def test_loop_k0(self):
        "Test loop for k=0 for known delay values."
        crb = CatRingBuffer(self.init, 3)
        loop = self.Loop(k=0, buf=crb)
        for i in range(10):
            lag0 = loop.step()
            lag1 = crb.read(lag=1)
            assert_allclose(lag1, 2.0 * lag0)
            assert crb.trpos == ((i + 1) % crb.nt)
            assert crb.trace.shape == ((crb.nt, ) + crb.state.shape)

    def _run_loop(self, k, Buf, lag=None):
        "Run simulation loop for value of k."
        loop = self.Loop(k=k, buf=Buf(self.init, 3), lag=lag)
        trace = []
        for i in range(10):
            trace.append(loop.step())
        return np.array(trace)

    def _sse_loop(self, k, data, Buf, lag=None):
        "Eval sum squared error of simulation vs data."
        sim = self._run_loop(k, Buf, lag=lag)
        sse = np.sum(np.square(sim - data))
        return sse

    def _run_opt(self, Buf, lag=None):
        "Attempt optimization by gradient descent."
        k = 0.15
        data = self._run_loop(k, Buf, lag=lag)
        # guess w/ initial sse
        k_hat = 0.1
        sse_i = self._sse_loop(k_hat, data, Buf, lag=lag)
        grad_sse = grad(lambda k_i: self._sse_loop(k_i, data, Buf, lag=lag))
        # do opt
        for i in range(5):
            g = grad_sse(k_hat)
            k_hat += -0.1 * g
            sse_ip1 = self._sse_loop(k_hat, data, Buf, lag=lag)
            assert sse_ip1 < sse_i
            sse_i = sse_ip1

    def test_delay(self):
        assert has_gradient(CatRingBuffer)
        self._run_opt(CatRingBuffer)

    def test_delay_matrix_lags(self):
        lag = np.array([[2, 1], [0, 2]])
        self._run_opt(CatRingBuffer, lag=lag)

    def test_no_delay(self):
        assert has_gradient(NoopBuffer)
        self._run_opt(NoopBuffer)


class TestMemberPrimitives:

    class Primitive:
        def __init__(self, param):
            self.param = param
            self.logsumexp1 = primitive(self.logsumexp)
            defvjp(self.logsumexp1, self.logsumexp1_vjp)
        
        def logsumexp(self, x):
            """Numerically stable log(sum(exp(x)))"""
            max_x = np.max(x) + self.param
            return max_x + np.log(np.sum(np.exp(x - max_x)))

        def logsumexp1_vjp(self, ans, x):
            x_shape = x.shape
            return lambda g: np.full(x_shape, g) * np.exp(x - np.full(x_shape, ans))

    def test_usage(self):
        # instantiate primitive
        primitive = self.Primitive(param=1.5)
        # use autograd version
        def f_ad(y):
            z = y**2
            lse = primitive.logsumexp(z)
            return np.sum(lse)
        # and version using custom primitive
        def f_prim(y):
            z = y**2
            lse = primitive.logsumexp1(z)
            return np.sum(lse)
        # check result is the same
        y = np.array([1.5, 6.7, 1e-10])
        assert (grad(f_ad)(y) == grad(f_prim)(y)).all()

