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


def make_linear_cfun(scale=0.01):
    k = nb.float32(scale)
    @nb.njit(inline='always')
    def pre(xj, xi):
        return xj
    @nb.njit(inline='always')
    def post(gx):
        return k * gx
    return pre, post


def make_loop(nh, nn, dt, cfpre, cfpost):
    # w/ dt=0.1ms, speed 10m/s, max delay 25.6ms ~ gamma band
    # assert nh == 256
    # assert dt == 0.1
    nh, nn = [nb.uint32(_) for _ in (nh, nn)]
    dt, pi = [nb.float32(_) for _ in (dt, np.pi)]
    sqrt_dt = nb.float32(np.sqrt(dt))
    o_nh = nb.float32(1 / nh)
    nh_dt = nb.float32(nh * dt)
    @nb.njit(boundscheck=False, fastmath=True)
    def loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma):
        o_tau = nb.float32(1 / tau)
        assert r.shape[0] == V.shape[0] == nh  # shape asserts help numba optimizer
        assert r.shape[1] == V.shape[1] == nn
        for i in range(nn):
            tavg[0, i] = nb.float32(0.0)
            tavg[1, i] = nb.float32(0.0)
        for t0 in range(-1, nh - 1):
            t = nh-1 if t0<0 else t0
            t1 = t0 + 1
            for i in range(nn):
                rc = nb.float32(0) # using array here costs 50%+
                Vc = nb.float32(0)
                for j in range(nn):
                    dij = (t - d[i, j] + nh) & (nh-1)
                    rc += w[i, j] * cfpre(r[dij, j], r[t, i])
                    Vc += w[i, j] * cfpre(V[dij, j], V[t, i])
                rc = cfpost(rc)
                Vc = cfpost(Vc)
                dr = o_tau * (Delta / (pi * tau) + 2 * V[t, i] * r[t, i])
                dV = o_tau * (V[t, i] ** 2 - (pi ** 2) * (tau ** 2) * (r[t, i] ** 2) + eta + J * tau * r[t, i] + I + cr * rc + cv * Vc)
                r[t1, i] = r[t, i] + dr * dt + sqrt_dt * r_sigma * wrV[0, t, i]
                V[t1, i] = V[t, i] + dV * dt + sqrt_dt * V_sigma * wrV[1, t, i]
                # TODO use hanning window to lower leakage
                tavg[0, i] += r[t1, i] * o_nh
                tavg[1, i] += V[t1, i] * o_nh
                bold_out[i] = fmri(bold_state[i], tavg[0, i], dt)
    return loop


def default_icfun(t, rV):
    rV[0] = 0.0
    rV[1] = -2.0


def run_loop(weights, delays,
             total_time=60e3, bold_tr=1800, coupling_scaling=0.01,
             r_sigma=1e-3, V_sigma=1e-3,
             I=0.0, Delta=1.0, eta=-5.0, tau=100.0, J=15.0, cr=0.01, cv=0.0,
             dt=0.1, nh=256,
             icfun=default_icfun):
    assert weights.shape == delays.shape and weights.shape[0] == weights.shape[1]
    nn = weights.shape[0]
    w = weights.astype(np.float32)
    d = (delays / dt).astype(np.uint32)
    assert d.max() < nh
    # inner loop setup dimensions, constants, buffers
    r, V = rV = np.zeros((2, nh, nn), 'f')
    icfun(-np.r_[:nh]*dt, rV)
    I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma = [nb.float32(_) for _ in (I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)]
    wrV = np.empty((2, nh, nn), 'f')                            # buffer for noise
    tavg = np.zeros((2, nn), 'f')                               # buffer for temporal average
    bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
    bold_state[:,1:] = 1.0
    bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
    rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
    # first call to jit the function
    cfpre, cfpost = make_linear_cfun(coupling_scaling)
    loop = make_loop(nh, nn, dt, cfpre, cfpost)
    # outer loop setup
    win_len = nh * dt
    total_wins = int(total_time / win_len)
    bold_skip = int(bold_tr / win_len)
    tavg_trace = np.empty((total_wins, ) + tavg.shape, 'f')
    bold_trace = np.empty((total_wins//bold_skip + 1, ) + bold_out.shape, 'f')
    # start time stepping
    for t in tqdm.trange(total_wins):
        rng.standard_normal(size=(2, nh, nn), dtype='f', out=wrV)  # ~15% time here
        loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)
        tavg_trace[t] = tavg
        if t % bold_skip == 0:
            bold_trace[t//bold_skip] = bold_out
    return tavg_trace, bold_trace


if __name__ == '__main__':
    nn = 96
    w = np.random.randn(nn, nn)**2
    d = np.random.rand(nn, nn)**2 * 25
    run_loop(w, d)