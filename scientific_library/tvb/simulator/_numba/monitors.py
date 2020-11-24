import numba as nb
import numba.cuda


def make_fmri(use_cuda=False):
    if use_cuda:
        jit = nb.cuda.jit(inline='always',device=True)
    else:
        jit = nb.njit(fastmath=True,boundscheck=False,inline='always')
    @jit
    def fmri_(it, node_bold, x, dt):
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
        s = node_bold[0, it]
        f = node_bold[1, it]
        v = node_bold[2, it]
        q = node_bold[3, it]
        ds = x - RECIP_TAU_S * s - RECIP_TAU_F * (f - 1.0)
        df = s
        dv = RECIP_TAU_O * (f - pow(v, RECIP_ALPHA))
        dq = RECIP_TAU_O * (f * (1.0 - pow(1.0 - E0, 1.0 / f))
             * RECIP_E0 - pow(v, RECIP_ALPHA) * (q / v))
        s += dt * ds
        f += dt * df
        v += dt * dv
        q += dt * dq
        node_bold[0, it] = s
        node_bold[1, it] = f
        node_bold[2, it] = v
        node_bold[3, it] = q
        return V0 * (k1 * (1.0 - q) + k2 * (1.0 - q / v) + k3 * (1.0 - v))
    # for plain version, fix it==0 compile time
    if not use_cuda:
        @nb.njit
        def fmri(node_bold, x, dt):
            return fmri_(nb.uint32(0), node_bold, x, dt)
    else:
        fmri = fmri_
    return fmri


fmri = make_fmri(use_cuda=False)
fmri_gpu = make_fmri(use_cuda=True)