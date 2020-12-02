"A TensorFlow 2 backend"


from .base import BaseBackend
import numpy as np
import tensorflow as tf
import tqdm

# quick switch from compiled to not
jit = tf.function if 1 else lambda f: f

# TF modules play an infrastructure/instrumenting role in TF like traits in TVB
# Structuring code with modules should allow considerable flexiblity afterwards.

class MontbrioDfun(tf.Module):
    def __init__(self, tau=100.0, Delta=1.0, eta=-5.0, J=15.0, I=1.0, cr=0.01, cv=0.0):
        super().__init__()
        self.pi = tf.constant(np.pi, tf.float32)
        self.tau = tf.Variable(tau, dtype=tf.float32)
        self.Delta = tf.Variable(Delta, dtype=tf.float32)
        self.eta = tf.Variable(eta, dtype=tf.float32)
        self.J = tf.Variable(J, dtype=tf.float32)
        self.I = tf.Variable(I, dtype=tf.float32)
        self.cr = tf.Variable(cr, dtype=tf.float32)
        self.cv = tf.Variable(cv, dtype=tf.float32)
    @jit
    def __call__(self, x, c):
        o_tau = 1.0 / self.tau
        r, V = x[0], x[1]
        rc, Vc = c[0], c[1]
        dr = o_tau * (self.Delta / (self.pi * self.tau) + 2 * V * r)
        dV = o_tau * (V ** 2 - (self.pi ** 2) * (self.tau ** 2) * (r ** 2) + self.eta +
                      self.J * self.tau * r + self.I + self.cr * rc + self.cv * Vc)
        return tf.stack([dr, dV])


class RK4(tf.Module):
    def __init__(self, dt, dfun):
        super().__init__()
        self.dt = tf.constant(dt, tf.float32)
        self.dfun = dfun
    @jit
    def __call__(self, x, c):
        k1 = self.dfun(x, c)
        k2 = self.dfun(x + self.dt/2*k1, c)
        k3 = self.dfun(x + self.dt/2*k2, c)
        k4 = self.dfun(x + self.dt*k3, c)
        return x + self.dt/6*(k1 + 2*(k2 + k3) + k4)


class LinearCfun(tf.Module):
    def __init__(self, weights, a, b):
        super().__init__()
        self.weights = tf.Variable(weights, tf.float32)
        self.a = tf.Variable(a, tf.float32)
        self.b = tf.Variable(b, tf.float32)
    @jit
    def __call__(self, x):
        gx = tf.matmul(w, x, transpose_b=True)
        nsv, nb, nn, _ = gx.shape
        gx = tf.reshape(gx, (nsv, nb, 1, nn))
        assert x.shape == gx.shape
        return self.a * gx + self.b


class Stepper(tf.Module):
    def __init__(self, x0, dt, n_steps, dfun, cfun, scheme=RK4):
        super().__init__()
        self.n_steps = tf.constant(n_steps)
        self.scheme = scheme(dt, dfun)
        self.cfun = cfun
        self.tavg = tf.Variable(tf.zeros_like(x0))
    @jit
    def __call__(self, x):
        self.tavg.assign(tf.zeros_like(x))
        for i in range(self.n_steps):
            x = self.scheme(x, self.cfun(x))
            self.tavg.assign_add(x)
        return x, self.tavg/tf.cast(self.n_steps, tf.float32)

nn = 96  # network size
nb = 16  # batch size
w = tf.random.normal((nb, nn, nn))
x = tf.zeros(     (2, nb,  1, nn), tf.float32)

dfun = MontbrioDfun()
cfun = LinearCfun(w, 0.1, 0.0)
stepper = Stepper(x, 0.1, 10, dfun, cfun)

for i in tqdm.trange(int(600*1e3)):
    x, tavg = stepper(x)