from tvb.simulator.models.wong_wang import ReducedWongWang,Final,List
from numba import guvectorize, float64
import numpy

@guvectorize([(float64[:],) * 12], '(n),(m)' + ',()' * 8 + '->(n),(n)', nopython=True)
def _numba_dfun(S, c, a, b, d, g, ts, w, j, io, dx, h):
    """Gufunc for reduced Wong-Wang model equations.(modification for saving the firing rate h)"""
    x = w[0] * j[0] * S[0] + io[0] + j[0] * c[0]
    h[0] = (a[0] * x - b[0]) / (1 - numpy.exp(-d[0] * (a[0] * x - b[0])))
    dx[0] = - (S[0] / ts[0]) + (1.0 - S[0]) * h[0] * g[0]


@guvectorize([(float64[:],) * 5], '(n),(m)' + ',()' * 2 + '->(n)', nopython=True)
def _numba_dfun_proxy(s, h, g, ts, dx):
    """Gufunc for reduced Wong-Wang model equations for proxy node."""
    dx[0] = - (s[0] / ts[0]) + (1.0 - s[0]) * h[0] * g[0]


class ReducedWongWangProxy(ReducedWongWang):
    """
    modify class in order to take in count proxy firing rate and to monitor the firing rate
    """
    state_variables = 'S H'.split()
    _nvar = 2
    state_variable_range = Final(
        label="State variable ranges [lo, hi]",
        default={"S": numpy.array([0.0, 1.0]),
                 "H":numpy.array([0.0,0.0])},
        doc="Population firing rate")
    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("S","H"),
        default=("S","H"),
        doc="""default state variables to be monitored""")
    _coupling_variable = None
    integration_variables = ['S']
    H_save = None

    def update_state_variables_before_integration(self,x,c, local_coupling=0.0, stimulus=0.0):
        return None

    def update_state_variables_after_integration(self,X):
        X[1,:] = self.H_save # only work for Euler integrator
        return X

    def dfun(self, x, c, local_coupling=0.0):
        # same has tvb implementation
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv, H = _numba_dfun(x_, c_, self.a, self.b, self.d, self.gamma,
                               self.tau_s, self.w, self.J_N, self.I_o)
        self.H_save = H
        return deriv