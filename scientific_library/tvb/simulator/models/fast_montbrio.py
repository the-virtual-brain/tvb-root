import tqdm
import numpy as np
import numba as nb
from numpy.random import SFC64

from tvb.simulator._numba.montbrio import make_gpu_loop, make_loop
from tvb.simulator._ispc.montbrio import run_ispc_montbrio


def make_linear_cfun(scale=0.01):
    k = nb.float32(scale)
    @nb.njit(inline='always')
    def pre(xj, xi):
        return xj
    @nb.njit(inline='always')
    def post(gx):
        return k * gx
    return pre, post


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
    #wrV = np.empty((2, nh, nn), 'f')                            # buffer for noise
    tavg = np.zeros((nto, 2, nn), 'f')                               # buffer for temporal average
    bold_state = np.zeros((nn, 4), 'f')                         # buffer for bold state
    bold_state[:,1:] = 1.0
    bold_out = np.zeros((nn,), 'f')                             # buffer for bold output
    rng = np.random.default_rng(SFC64(42))                      # create RNG w/ known seed
    # first call to jit the function
    cfpre, cfpost = make_linear_cfun(coupling_scaling)
    loop = make_loop(nh, nto, nn, dt, cfpre, cfpost)
    #loop(r, V, wrV, w, d, tavg, bold_state, bold_out, I, Delta, eta, tau, J, cr, cv, r_sigma, V_sigma)
    # outer loop setup
    win_len = nh * dt
    total_wins = int(total_time / win_len)
    print(total_time, win_len, total_wins)
    bold_skip = int(bold_tr / win_len)
    tavg_trace = np.empty((total_wins, ) + tavg.shape, 'f')
    bold_trace = np.empty((total_wins//bold_skip + 1, ) + bold_out.shape, 'f')
    # start time stepping
    for t in (tqdm.trange if progress else range)(total_wins):
        wrV = rng.standard_normal(size=(2, nh, nn), dtype='f') #, out=wrV)  # ~15% time here
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
    d = np.random.rand(nn, nn)**2 * 15
    ns = 60
    tavg0, _ = run_loop(w, d, nh=16, dt=1.0, I=1.0, cr=0.1, r_sigma=3e-3, V_sigma=1e-3, nto=1, tau=10.0, progress=True, total_time=ns*1e3)
    tavg1, _ = run_ispc_montbrio(w, d, total_time=ns*1e3)
    from numpy.testing import assert_allclose
    assert_allclose(tavg0.reshape((-1, 192)), tavg1)

    # args, mons = grid_search(n_jobs=2,
    #     weights=[w], delays=[d], total_time=[10e3],  # non-varying into single elem list
    #     cr=np.r_[:0.1:4j], cv=np.r_[:0.1:4j]         # varying as arrays/lists
    # )
    # for args, (tavg, bold) in zip(args, mons):
    #     print('cr/cv: ', args['cr'], '/', args['cv'], ', tavg std:', tavg[-100:].std())
