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


class TestHasGradient:

    def setup_method(self):
        class TestClass(HasGradient):
            pass
        self.test_class = TestClass()

    def test_has_class(self):
        assert has_gradient(self.test_class)


class TestTimeSeriesGradient:

    def setup_method(self):
        self.dfun = Linear().make_dfun(numpy=anp)
        self.scheme = EulerDeterministic(dt=0.1).make_scheme(self.dfun)
        self.init = anp.random.randn(1, 5, 1)
        self.trace = [self.init]
        for i in range(10):
            self.trace.append(self.scheme(self.trace[-1], self.trace[-1]))
        self.trace = anp.array(self.trace)
        def fn(init):
            trace = [init]
            for i in range(10):
                trace.append(self.scheme(trace[-1], trace[-1]))
            trace = anp.array(trace)
            return anp.sum(anp.square(trace - self.trace))
        self.fn = fn
        self.grad_func = grad(self.fn)

    def test_grad(self):
        init = anp.random.randn(1, 5, 1)
        assert anp.sum(anp.square(init - self.init)) > 0.1
        for i in range(20):
            init += -0.1 * self.grad_func(init)
        assert anp.sum(anp.square(init - self.init)) < 0.1
