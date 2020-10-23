import tqdm
import numpy as np
import numba as nb
from numpy.random import SFC64
import math


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


def make_gpu_loop(nh, nn, nt, dt):
    from numba import cuda
    from numba.cuda.random import xoroshiro128p_uniform_float32
    assert nh == 32
    nh, nn = [nb.uint32(_) for _ in (nh, nn)]
    dt, pi = [nb.float32(_) for _ in (dt, np.pi)]
    sqrt_dt = nb.float32(np.sqrt(dt))
    o_6 = nb.float32(1 / 6)
    pi_2 = nb.float32(np.pi * 2)
    o_nt = nb.float32(1.0 / nt)
    nl, nc = nb.int32(32), nb.int32(nn // 32)
    @cuda.jit(inline='always',device=True)
    def dr(r, V, o_tau, pi, tau, Delta):
        return o_tau * (Delta / (pi * tau) + 2 * V * r)
    @cuda.jit(inline='always',device=True)
    def dV(r, V, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc):
        return o_tau * (V ** 2 - (pi ** 2) * (tau ** 2) * (r ** 2) + eta + J * tau * r + I + cr * rc + cv * Vc)
    @cuda.jit(fastmath=True)
    def loop(r, V, w, d, tavg, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma, rng_states):
        it = cuda.threadIdx.x
        o_tau = nb.float32(1 / tau)
        assert r.shape[0] == V.shape[0] == nn  # shape asserts help numba optimizer
        assert r.shape[1] == V.shape[1] == nh
        # 48 KB shared memory
        # 96 nn x 16 nh x 4 f32 x 2 svar = 12 KB
        # 96^2*2 * 4f32 = 73 KB so we can't do everything in shared memory
        # coupling
        aff = cuda.shared.array((3, 32), nb.float32)
        # current state
        rt = cuda.shared.array((3, 32), nb.float32)
        Vt = cuda.shared.array((3, 32), nb.float32)
        nrt = cuda.shared.array((3, 32), nb.float32)
        nVt = cuda.shared.array((3, 32), nb.float32)
        # shared mem usage 640 per block, nvcc says
        # used 106 registers, 32 stack, 1920 bytes smem, 856 bytes cmem[0], 24 bytes cmem[2], 0 bytes lmem
        for i in range(nc):
            rt[i, it] = r[i * nl + it, 0]
            Vt[i, it] = V[i * nl + it, 0]
        for i in range(nc):
            tavg[0, i, it] = nb.float32(0.0)
            tavg[1, i, it] = nb.float32(0.0)
        for t in range(nt):
            for i in range(nc):
                aff[i, it] = nb.float32(0)
            for j in range(nn):
                for i in range(nc):
                    aff[i, it] += w[i,j,it] * cuda.shfl_sync(nb.int32(1), r[j,it], d[j,i,it])
            for i in range(nc):
                r_ = rt[i, it]  # we need history stride 1 in time, but state stride 1 in space..
                V_ = Vt[i, it]
                dr_0 = dr(r_, V_, o_tau, pi, tau, Delta)
                dV_0 = dV(r_, V_, o_tau, pi, tau, eta, J, I, cr, aff[i,it], cv, nb.float32(0))
                kh = nb.float32(0.5)
                dr_1 = dr(r_ + dt*kh*dr_0, V_ + dt*kh*dV_0, o_tau, pi, tau, Delta)
                dV_1 = dV(r_ + dt*kh*dr_0, V_ + dt*kh*dV_0, o_tau, pi, tau, eta, J, I, cr, aff[i,it], cv, nb.float32(0))
                dr_2 = dr(r_ + dt*kh*dr_1, V_ + dt*kh*dV_1, o_tau, pi, tau, Delta)
                dV_2 = dV(r_ + dt*kh*dr_1, V_ + dt*kh*dV_1, o_tau, pi, tau, eta, J, I, cr, aff[i,it], cv, nb.float32(0))
                kh = nb.float32(1.0)
                dr_3 = dr(r_ + dt*kh*dr_2, V_ + dt*kh*dV_2, o_tau, pi, tau, Delta)
                dV_3 = dV(r_ + dt*kh*dr_2, V_ + dt*kh*dV_2, o_tau, pi, tau, eta, J, I, cr, aff[i,it], cv, nb.float32(0))
                # drift
                nrt[i, it] = r_ + o_6 * dt * (dr_0 + 2 * (dr_1 + dr_2) + dr_3)
                nVt[i, it] = V_ + o_6 * dt * (dV_0 + 2 * (dV_1 + dV_2) + dV_3)
                # box-muller
                u1 = xoroshiro128p_uniform_float32(rng_states, t*nn*2 + i*nl*2 + it)
                u2 = xoroshiro128p_uniform_float32(rng_states, t*nn*2 + i*nl*2 + nl + it)
                z0 = math.sqrt(-nb.float32(2.0) * math.log(u1)) * math.cos(pi_2 * u2)
                z1 = math.sqrt(-nb.float32(2.0) * math.log(u1)) * math.sin(pi_2 * u2)
                nrt[i, it] += sqrt_dt * r_sigma * z0
                nVt[i, it] += sqrt_dt * V_sigma * z1
                nrt[i, it] *= nrt[i, it] >= 0
            for i in range(nc):
                rt[i, it] = nrt[i, it]
                Vt[i, it] = nVt[i, it]
            # global memory writes
            for i in range(nc):
                # shift history
                for l in range(nl):
                    r[i*nl+l, it] = cuda.shfl_up_sync(1, r[i*nl+l, it], 1) # [0, 1, 2] -> [0, 0, 1]
                    V[i*nl+l, it] = cuda.shfl_up_sync(1, V[i*nl+l, it], 1)
                # insert current state at 0
                r[i*nl+it, 0] = rt[i, it]
                V[i*nl+it, 0] = Vt[i, it]
                # update tavg
                tavg[0, i, it] += rt[i, it] * o_nt
                tavg[1, i, it] += Vt[i, it] * o_nt
            # TODO fmri
    return loop


def make_loop(nh, nto, nn, dt, cfpre, cfpost):
    nh, nn = [nb.uint32(_) for _ in (nh, nn)]
    dt, pi = [nb.float32(_) for _ in (dt, np.pi)]
    sqrt_dt = nb.float32(np.sqrt(dt))
    o_nh = nb.float32(1 / nh * nto)
    o_6 = nb.float32(1 / 6)
    @nb.njit(fastmath=True,boundscheck=False,inline='always')
    def dr(r, V, o_tau, pi, tau, Delta):
        return o_tau * (Delta / (pi * tau) + 2 * V * r)
    @nb.njit(fastmath=True,boundscheck=False,inline='always')
    def dV(r, V, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc):
        return o_tau * (V ** 2 - (pi ** 2) * (tau ** 2) * (r ** 2) + eta + J * tau * r + I + cr * rc + cv * Vc)
    @nb.njit(boundscheck=False, fastmath=True)
    def loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma):
        o_tau = nb.float32(1 / tau)
        assert r.shape[0] == V.shape[0] == nh  # shape asserts help numba optimizer
        assert r.shape[1] == V.shape[1] == nn
        for j in range(nto):
            for i in range(nn):
                tavg[j, 0, i] = nb.float32(0.0)
                tavg[j, 1, i] = nb.float32(0.0)
        for t0 in range(-1, nh - 1):
            t = nh-1 if t0<0 else t0
            t1 = t0 + 1
            t0_nto = t0 // (nh // nto)
            for i in range(nn):
                rc = nb.float32(0) # using array here costs 50%+
                Vc = nb.float32(0)
                for j in range(nn):
                    dij = (t - d[i, j] + nh) & (nh-1)
                    rc += w[i, j] * cfpre(r[dij, j], r[t, i])
                    Vc += w[i, j] * cfpre(V[dij, j], V[t, i])
                rc = cfpost(rc)
                Vc = cfpost(Vc)
                dr_0 = dr(r[t, i], V[t, i], o_tau, pi, tau, Delta)
                dV_0 = dV(r[t, i], V[t, i], o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
                kh = nb.float32(0.5)
                dr_1 = dr(r[t, i] + dt*kh*dr_0, V[t, i] + dt*kh*dV_0, o_tau, pi, tau, Delta)
                dV_1 = dV(r[t, i] + dt*kh*dr_0, V[t, i] + dt*kh*dV_0, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
                dr_2 = dr(r[t, i] + dt*kh*dr_1, V[t, i] + dt*kh*dV_1, o_tau, pi, tau, Delta)
                dV_2 = dV(r[t, i] + dt*kh*dr_1, V[t, i] + dt*kh*dV_1, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
                kh = nb.float32(1.0)
                dr_3 = dr(r[t, i] + dt*kh*dr_2, V[t, i] + dt*kh*dV_2, o_tau, pi, tau, Delta)
                dV_3 = dV(r[t, i] + dt*kh*dr_2, V[t, i] + dt*kh*dV_2, o_tau, pi, tau, eta, J, I, cr, rc, cv, Vc)
                r[t1, i] = r[t, i] + o_6*dt*(dr_0 + 2*(dr_1 + dr_2) + dr_3) + sqrt_dt * r_sigma * wrV[0, t, i]
                r[t1, i] *= r[t1, i] >= 0
                V[t1, i] = V[t, i] + o_6*dt*(dV_0 + 2*(dV_1 + dV_2) + dV_3) + sqrt_dt * V_sigma * wrV[1, t, i]
                tavg[t0_nto, 0, i] += r[t1, i] * o_nh
                tavg[t0_nto, 1, i] += V[t1, i] * o_nh
                bold_out[i] = fmri(bold_state[i], tavg[0, 0, i], dt)
    return loop


def default_icfun(t, rV):
    rV[0] = 0.0
    rV[1] = -2.0


def run_loop(weights, delays,
             total_time=60e3, bold_tr=1800, coupling_scaling=0.01,
             r_sigma=1e-3, V_sigma=1e-3,
             I=1.0, Delta=1.0, eta=-5.0, tau=100.0, J=15.0, cr=0.01, cv=0.0,
             dt=1.0,
             nh=256,  # history buf len, must be power of 2 & greater than delays.max()/dt
             nto=16,   # num parts of nh for tavg, e.g. nh=256, nto=4: tavg over 64 steps
             progress=False,
             icfun=default_icfun):
    assert weights.shape == delays.shape and weights.shape[0] == weights.shape[1]
    nn = weights.shape[0]
    w = weights.astype(np.float32)
    d = (delays / dt).astype(np.int32)
    assert d.max() < nh
    # inner loop setup dimensions, constants, buffers
    r, V = rV = np.zeros((2, nh, nn), 'f')
    icfun(-np.r_[:nh]*dt, rV)
    I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma = [nb.float32(_) for _ in (I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)]
    wrV = np.empty((2, nh, nn), 'f')                            # buffer for noise
    tavg = np.zeros((nto, 2, nn), 'f')                               # buffer for temporal average
    bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
    bold_state[:,1:] = 1.0
    bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
    rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
    # first call to jit the function
    cfpre, cfpost = make_linear_cfun(coupling_scaling)
    loop = make_loop(nh, nto, nn, dt, cfpre, cfpost)
    loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)
    # outer loop setup
    win_len = nh * dt
    total_wins = int(total_time / win_len)
    print(total_time, win_len, total_wins)
    bold_skip = int(bold_tr / win_len)
    tavg_trace = np.empty((total_wins, ) + tavg.shape, 'f')
    bold_trace = np.empty((total_wins//bold_skip + 1, ) + bold_out.shape, 'f')
    # start time stepping
    for t in (tqdm.trange if progress else range)(total_wins):
        rng.standard_normal(size=(2, nh, nn), dtype='f', out=wrV)  # ~15% time here
        loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)
        tavg_trace[t] = tavg
        if t % bold_skip == 0:
            bold_trace[t//bold_skip] = bold_out
    return tavg_trace.reshape((total_wins * nto, 2, nn)), bold_trace


def run_gpu_loop(weights, delays,
             total_time=60e3, coupling_scaling=0.01,
             r_sigma=1e-3, V_sigma=1e-3,
             I=1.0, Delta=1.0, eta=-5.0, tau=100.0, J=15.0, cr=0.01, cv=0.0,
             dt=1.0,
             nh=32,  # history buf len, must be power of 2 & greater than delays.max()/dt
             progress=False, nt=16,
             icfun=default_icfun):
    assert nh==32, 'only nh=32 supported'
    assert weights.shape == delays.shape and weights.shape[0] == weights.shape[1], 'weights delays square plz'
    nn = weights.shape[0]
    assert nn%32==0, 'only nn multiple of 32 supported'
    nc, nl = nn // 32, 32
    w = weights.astype(np.float32)
    d = (delays / dt).astype(np.uint32)
    # reshape for j,chunk,lane indexing
    w_, d_ = [_.T.copy().reshape((nn,nc,nl)) for _ in (w, d)]
    assert d.max() <= nh, 'max delay must be <= nh'
    # inner loop setup dimensions, constants, buffers
    r, V = rV = np.zeros((2, nn, nh), 'f')
    icfun(-np.r_[:nh]*dt, rV)
    I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma = [nb.float32(_) for _ in (I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)]
    tavg = np.zeros((2, nc, nl), 'f')                               # buffer for temporal average
    bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
    bold_state[:,1:] = 1.0
    bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
    from numba.cuda.random import create_xoroshiro128p_states
    rng_states = create_xoroshiro128p_states(nn*2*nt, seed=1)
    loop = make_gpu_loop(nh, nn, nt, dt)
    sloop = loop.specialize(r, V, w_, d_, tavg, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma, rng_states)
    print(sloop._func.ccinfos[0])
    # outer loop setup
    win_len = nt * dt
    total_wins = int(total_time / win_len)
    print(total_time, win_len, total_wins)
    tavg_trace = np.empty((total_wins, ) + tavg.shape, 'f')
    # manage gpu memory
    from numba.cuda import to_device, pinned_array, device_array_like
    g_r, g_V, g_w_, g_d_, g_rng_states = [to_device(_) for _ in (r, V, w_, d_, rng_states)]
    p_tavg = pinned_array(tavg.shape)
    g_tavg = device_array_like(p_tavg)
    # start time stepping
    for t in (tqdm.trange if progress else range)(total_wins):
        loop[1, 32](g_r, g_V, g_w_, g_d_, g_tavg, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma, g_rng_states)
        g_tavg.copy_to_host(p_tavg)
        tavg_trace[t] = p_tavg
    return tavg_trace.reshape((total_wins, 2, nn)), None


def grid_search(**params):
    import joblib, itertools
    n_jobs = params.pop('n_jobs', 1)
    verbose = params.pop('verbose', 1)
    keys = list(params.keys())
    vals = [params[key] for key in keys]
    # expand product of search dimensions into dict of run_loop kwargs
    args = [dict(list(zip(keys, _))) for _ in itertools.product(*vals)]
    # run in parallel and return
    return args, joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        joblib.delayed(run_loop)(**arg) for arg in args)


if __name__ == '__main__':
    nn = 96
    w = np.random.randn(nn, nn)**2
    d = np.random.rand(nn, nn)**2 * 2.5
    pars = dict(total_time=5e4, dt=1.0, I=1.0, tau=10.0, nh=32, progress=True)
    tavg0, _ = run_gpu_loop(w, d, nt=32, **pars)
    tavg1, _ = run_loop(w, d, nto=1, **pars)
    import pylab as pl
    pl.subplot(211); pl.plot(tavg0[:, 0, 0], 'k')
    pl.subplot(212); pl.plot(tavg1[:, 0, 0], 'k')
    pl.show()

    # args, mons = grid_search(n_jobs=2,
    #     weights=[w], delays=[d], total_time=[10e3],  # non-varying into single elem list
    #     cr=np.r_[:0.1:4j], cv=np.r_[:0.1:4j]         # varying as arrays/lists
    # )
    # for args, (tavg, bold) in zip(args, mons):
    #     print('cr/cv: ', args['cr'], '/', args['cv'], ', tavg std:', tavg[-100:].std())
