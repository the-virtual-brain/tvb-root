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

def make_loop(nh, nn, dt):
    # for now assume dt=0.1ms, speed 10m/s, max delay 25.6ms ~ gamma band
    assert nh == 256
    assert dt == 0.1
    nh, nn = [nb.uint32(_) for _ in (nh, nn)]
    dt, pi = [nb.float32(_) for _ in (dt, np.pi)]
    sqrt_dt = nb.float32(np.sqrt(dt))
    o_nh = nb.float32(1 / nh)
    nh_dt = nb.float32(nh * dt)
    @nb.njit(boundscheck=False, fastmath=True)
    def loop(r, V, wrV, w, d, tavg, bold_state, bold_out, dt, I, Delta, eta, tau, J, cr, cv):
        o_tau = nb.float32(1 / tau)
        assert r.shape[0] == V.shape[0] == nh  # shape asserts help numba optimizer
        assert r.shape[1] == V.shape[1] == nn
        for t0 in range(-1, nh - 1):
            t = nh-1 if t0==0 else t0
            t1 = t0 + 1
            for i in range(nn):
                rc = nb.float32(0)
                Vc = nb.float32(0)
                for j in range(nn):
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
    return loop

# TODO turn the following into a class w/ all vars as attributes
# inner loop setup dimensions, constants, buffers
nh, nn = 256, 96                                            # likely dimensions
dt = 0.1                                                    # fixed dt
w = (np.random.randn(nn, nn)**2).astype(np.float32)         # random weights
d = (np.random.rand(nn, nn)**2 * 256).astype(np.uint32)     # random delays (in units of dt)
r, V = np.zeros((2, nh, nn), 'f')                           # buffers for r & V
V -= 2.0
I, Delta, eta, tau, J, cr, cv = [nb.float32(_) for _ in (0.0, 1.0, -5.0, 100.0, 15.0, 1.0/nn, 0.0)]
wrV = np.empty((2, nh, nn), 'f')                            # buffer for noise
tavg = np.zeros((2, nn), 'f')                               # buffer for temporal average
bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
bold_state[:,1:] = 1.0
bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed

# first call to jit the function
loop = make_loop(nh, nn, dt)
loop(r, V, wrV, w, d, tavg, bold_state, bold_out, dt, I, Delta, eta, tau, J, cr, cv)

# outer loop setup
win_len = nh * dt
total_time = 60e3 # 1 min
total_wins = int(total_time / win_len)
bold_tr = 1800  # in ms
bold_skip = int(bold_tr / win_len)
tavg_trace = []  # TODO memmap
bold_trace = []

# start time stepping
for t in tqdm.trange(total_wins):
    rng.standard_normal(size=(2, nh, nn), dtype='f', out=wrV)  # ~15% time here
    loop(r, V, wrV, w, d, tavg, bold_state, bold_out, dt, I, Delta, eta, tau, J, cr, cv)
    tavg_trace.append(tavg.copy())
    if t % bold_skip == 0:
        bold_trace.append(bold_out)

tavg_trace = np.array(tavg_trace)
bold_trace = np.array(bold_trace)
print('tavg', tavg_trace.shape)
print('bold', bold_trace.shape)