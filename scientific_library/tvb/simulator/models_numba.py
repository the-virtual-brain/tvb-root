# experimental model definitions using numba, currently for
# spatially homogeneous parameters, i.e. par.shape == (1,) for
# all model parameters

import numpy
import numba
from . import models
from ._numba.models import rww_dfun, hmje_dfun

class NumbaRww(models.ReducedWongWang):

    def dfun(self, x, c, lc=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        return rww_dfun(x_, c_, self.a, self.b, self.d, self.gamma,
                        self.tau_s, self.w, self.J_N, self.I_o)


class NumbaEpileptor(models.Epileptor):
    def dfun(self, x, c, lc=0.0):
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        return hmje_dfun(x_, c_,
                         self.x0, self.Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
                         self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.tau)