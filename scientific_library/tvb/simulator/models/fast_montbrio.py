import tqdm
import numpy as np
import numba as nb
from numpy.random import SFC64

@nb.njit
def fmri(node_bold, x, dt):
    TAU_S = nb.float32(0.65)
    TAU_F = nb.float32(0.41)
    TAU_O = nb.float32(0.98)
    ALPHA = nb.float32(0.32)
    TE = nb.float32(0.04)
    V0 = nb.float32(4.0)
    E0 = nb.float32(0.4)
    EPSILON = nb.float32(0.5)
    NU_0 = nb.float32(40.3)
    R_0 = nb.float32(25.0)
    RECIP_TAU_S = nb.float32((1.0 / TAU_S))
    RECIP_TAU_F = nb.float32((1.0 / TAU_F))
    RECIP_TAU_O = nb.float32((1.0 / TAU_O))
    RECIP_ALPHA = nb.float32((1.0 / ALPHA))
    RECIP_E0 = nb.float32((1.0 / E0))
    k1 = nb.float32((4.3 * NU_0 * E0 * TE))
    k2 = nb.float32((EPSILON * R_0 * E0 * TE))
    k3 = nb.float32((1.0 - EPSILON))
    # end constants, start diff eqns
    s = node_bold[0]
    f = node_bold[1]
    v = node_bold[2]
    q = node_bold[3]
    ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0)
    df = s
    dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA))
    dq = RECIP_TAU_O * (f * (1.0 - pow(1.0 - E0, 1.0 / f))
         * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v))
    s += dt * ds
    f += dt * df
    v += dt * dv
    q += dt * dq
    node_bold[0] = s
    node_bold[1] = f
    node_bold[2] = v
    node_bold[3] = q
    return V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))


@nb.njit(boundscheck=False, fastmath=True)
def loop(r, V, wrV, w, d, tavg, bold_state, bold_out, dt, I, Delta, eta, tau, J, cr, cv, pi):
    o_tau = nb.float32(1 / tau)
    sqrt_dt = nb.float32(np.sqrt(dt))
    # assume dt=0.1ms, speed 10m/s, max delay 25.6ms ~ gamma band
    nh = nb.uint32(256)
    o_nh = nb.float32(1 / nh)
    nh_dt = nb.float32(nh * dt)
    assert r.shape[0] == V.shape[0] == nh
    for t0 in range(-1, r.shape[0] - 1):
        t = nh-1 if t0==0 else t0
        t1 = t0 + 1
        for i in range(r.shape[1]):
            rc = nb.float32(0)
            Vc = nb.float32(0)
            for j in range(r.shape[1]):
                dij = (t - d[i, j] + nh) & (nh-1)
                rc += w[i, j] * r[dij, j]
                Vc += w[i, j] * r[dij, j]
            dr = o_tau * (Delta / (pi * tau) + 2 * V[t, i] * r[t, i])
            dV = o_tau * (V[t, i] ** 2 - pi ** 2 * tau ** 2 * r[t, i] ** 2 + eta + J * tau * r[t, i] + I + cr * rc + cv * Vc)
            r[t1, i] = r[t, i] + dr * dt + sqrt_dt * 0.1 * wrV[0, t, i]
            V[t1, i] = V[t, i] + dV * dt + sqrt_dt * 0.01 * wrV[1, t, i]
            # t avg over 25.6 ms ~39 Hz sampling; TODO use hanning window
            tavg[0, i] += r[t1, i] * o_nh
            tavg[1, i] += V[t1, i] * o_nh
    # update bold (from r?)
    for i in range(r.shape[1]):
        bold_out[i] = fmri(bold_state[i], tavg[0, i], nh_dt)

nt, nn = 256, 96                                            # likely dimensions
dt = 0.1                                                    # fixed dt
w = (np.random.randn(nn, nn)**2).astype(np.float32)         # random weights
d = (np.random.rand(nn, nn)**2 * 256).astype(np.uint32)     # random delays
r, V = np.zeros((2, nt, nn), 'f')                           # buffers for r & V
V -= 2.0
I, Delta, eta, tau, J, cr, cv, pi, dt = [nb.float32(_) for _ in (0.0, 1.0, -5.0, 100.0, 15.0, 1.0/nn, 0.0, np.pi, dt)]
wrV = np.empty((2, nt, nn), 'f')                            # buffer for noise
tavg = np.zeros((2, nn), 'f')                               # buffer for temporal average
bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
bold_state[:,1:] = 1.0
bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed

# first call to jit the function
loop(r, V, wrV, w, d, tavg, bold_state, bold_out, dt, I, Delta, eta, tau, J, cr, cv, pi)

# 10 iters of 1 second each; ~10x over realtime
for i in tqdm.trange(10):
    # 40 iters of 25.6ms jit'ed inner loop = 1s
    for j in range(40):
        rng.standard_normal(size=(2, nt, nn), dtype='f', out=wrV)  # ~15% time here
        loop(r, V, wrV, w, d, tavg, bold_state, bold_out, dt, I, Delta, eta, tau, J, cr, cv, pi)