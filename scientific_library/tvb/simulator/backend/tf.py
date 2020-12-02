"A TensorFlow 2 backend"


from .base import BaseBackend
import numpy as np
import tensorflow as tf
import tqdm

# mainly testing TF for parsweeps here, not yet developed
# into a proper backend

def make_odes(tau, Delta, eta, J, I, cr, cv):
    o_tau, pi = 1/tau, np.pi
    o_tau, pi, tau, Delta, eta, J, I, cr, cv = [tf.constant(_) for _ in (o_tau, pi, tau, Delta, eta, J, I, cr, cv)]
    @tf.function
    def odes(x, rc):
        r, V = x[0], x[1]
        Vc = tf.constant(0.0)
        dr = o_tau * (Delta / (pi * tau) + 2 * V * r)
        dV = o_tau * (V ** 2 - (pi ** 2) * (tau ** 2) * (r ** 2) + eta + J * tau * r + I + cr * rc + cv * Vc)
        return tf.stack([dr, dV])
    return odes

def make_rk4(dt, f, n_steps=10):
    dt = tf.constant(dt)
    sqrt_dt = tf.sqrt(dt)
    @tf.function
    def rk4_step(x, z, w, sigma):
        nsv, nt1, nt2, nn, nb = x.shape
        rc = tf.reduce_sum(w * tf.reshape(x[0], (nt1, nt2, 1, nn, nb)), axis=-2)
        k1 = f(x, rc)
        k2 = f(x + dt/2*k1, rc)
        k3 = f(x + dt/2*k2, rc)
        k4 = f(x + dt*k3, rc)
        nx = x + dt/6*(k1 + 2*(k2 + k3) + k4) + sqrt_dt*sigma*z
        return nx
    @tf.function
    def rk4(x, z, w, sigma):
        for i in range(n_steps):
            x = rk4_step(x, z[i], w, sigma)
        return x
    return rk4


odes = make_odes(tau=100.0, Delta=1.0, eta=-5.0, J=15.0, I=1.0, cr=0.01, cv=0.0)
rk4 = make_rk4(0.1, odes)

# dims & grid
nn = 96
nt1, nt2 = 2, 2
nb = 1

# doesn't hit linear scaling until 32 threads, but then CPU's using multiple cores
# for 16x16, using all 4 cores, around 4 hours for 10 min bold, single subjects

# state buffer
x = tf.zeros((2, nt1, nt2, nn, nb)) - tf.reshape(tf.constant([0.0, 2.0]), (2,1,1,1,1))
# generate random log normal weights then max normalized
w = tf.exp(tf.random.normal((nt1, nt2, nn, nn, nb)))
w = w / tf.reduce_max(tf.reduce_max(w, axis=-1, keepdims=True), axis=-1, keepdims=True)

# parameters to sweep, need to broadcast to grid above
cr = tf.convert_to_tensor(10**np.r_[-3:-1:1j*nt1], tf.float32)
cr = tf.reshape(cr, (nt1, 1, 1, 1))
sigma = tf.convert_to_tensor(10**np.r_[-3:-1:1j*nt2], tf.float32)
sigma = tf.reshape(sigma, (nt2, 1, 1))

for i in tqdm.trange(int(600*1e3)):
    z = tf.random.normal((10, 2, nt1, nt2, nn, nb))
    r, V = rk4(x, z, w, sigma)