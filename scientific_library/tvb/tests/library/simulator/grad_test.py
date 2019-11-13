import pytest
from tvb.simulator.models import Linear
from tvb.simulator.integrators import EulerDeterministic
from autograd import grad
import autograd.numpy as anp


class TestGradients:

    def setup_method(self):
        self.dfun = Linear().make_dfun(numpy=anp)
        self.scheme = EulerDeterministic(dt=0.1).make_scheme(self.dfun)
        def example(scl):
            X = anp.ones((1, 5, 1))
            C = anp.zeros((1, 5, 1)) + scl
            Xn = self.scheme(X, C)
            sse = anp.sum(anp.square(X - Xn))
            return sse
        self.scalar_func = example
        self.grad_func = grad(example)

    def test_scalar_func(self):
        assert self.scalar_func(0.4) == 4.608

    def test_grad(self):
        assert self.grad_func(0.4) == -0.96
