import math

import numba as nb
import numpy as np

from .monitors import fmri


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
        # np.seterr(invalid='raise')
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
                r[t1, i] = r[t, i] + o_6*dt*(dr_0 + 2*(dr_1 + dr_2) + dr_3) + sqrt_dt * r_sigma * wrV[0, t1, i]
                r[t1, i] *= r[t1, i] > 0
                V[t1, i] = V[t, i] + o_6*dt*(dV_0 + 2*(dV_1 + dV_2) + dV_3) + sqrt_dt * V_sigma * wrV[1, t1, i]
                tavg[t0_nto, 0, i] += r[t1, i] * o_nh
                tavg[t0_nto, 1, i] += V[t1, i] * o_nh
                bold_out[i] = fmri(bold_state[i], tavg[0, 0, i], dt)
    return loop