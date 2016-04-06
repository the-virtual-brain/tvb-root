# experimental model definitions using numba, currently for
# spatially homogeneous parameters, i.e. par.shape == (1,) for
# all model parameters

import numpy
import numba
from . import models


@numba.guvectorize([(numba.float64[:],)*11],
                   '(n),(m)' + ',()'*8 + '->(n)',
                   nopython=True)
def numba_rww_dfun(S, c, a, b, d, g, ts, w, j, io, dx):
    if S[0] < 0.0:
        dx[0] = 0.0 - S[0]
    elif S[0] > 1.0:
        dx[0] = 1.0 - S[0]
    else:
        x = w[0]*j[0]*S[0] + io[0] + j[0]*c[0]
        h = (a[0]*x - b[0]) / (1 - numpy.exp(-d[0]*(a[0]*x - b[0])))
        dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h * g[0]

class NumbaRww(models.ReducedWongWang):

    def dfun(self, x, c, lc=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        return numba_rww_dfun(x_, c_, self.a, self.b, self.d, self.gamma,
                              self.tau_s, self.w, self.J_N, self.I_o)


@numba.guvectorize([(numba.float64[:],) * 18],
                   '(n),(m)' + ',()'*15 + '->(n)',
                   nopython=True)
def numba_epi_dfun(y, c_pop, x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, tau, ydot):
    c_pop1 = c_pop[0]
    c_pop2 = c_pop[1]
    # population 1
    if y[0] < 0.0:
        ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
    else:
        ydot[0] = slope[0] - y[3] + 0.6 * (y[2] - 4.0) ** 2
    ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + Kvf[0] * c_pop1 + ydot[0] * y[0])
    ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])
    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    ydot[2] = tt[0] * (r[0] * (4 * (y[0] - x0[0]) - y[2] + ydot[2] + Ks[0] * c_pop1))
    # population 2
    ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + 2 * y[5] - 0.3 * (y[2] - 3.5) + Kf[0] * c_pop2)
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = aa[0] * (y[3] + 0.25)
    ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau[0])
    # filter
    ydot[5] = tt[0] * (-0.01 * (y[5] - 0.1 * y[0]))


class NumbaEpileptor(models.Epileptor):
    def dfun(self, x, c, lc=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        return numba_epi_dfun(x_, c_,
                              self.x0, self.Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
                              self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.tau)