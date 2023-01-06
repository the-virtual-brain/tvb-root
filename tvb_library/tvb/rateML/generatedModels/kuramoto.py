
from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class KuramotoT(ModelNumbaDfun):

    omega = NArray(
        label = ":math:`omega`",
        default = numpy.array([60.0 * 2.0 * 3.1415927 / 1e3]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """"""
    )

    nsig = NArray(
        label = ":math:`nsig`",
        default = numpy.array([0.5]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """noiseparameter"""
    )

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"theta": numpy.array([0.0, 100])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"theta": numpy.array([1.0, 2.0])},
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('theta', ),
        default=('theta', ),
        doc="Variables to monitor"
    )

    state_variables = ['theta']

    _nvar = 1
    cvar = numpy.array([0,], dtype = numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_KuramotoT(vw_, c_,     # %if not (para.maxval):
self.omega,     # %endif
    # %if not (para.maxval):
self.nsig,     # %endif
local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64[:])], '(n),(m)' + ',()'*3 + '->(n)', nopython=True)
def _numba_dfun_KuramotoT(vw, coupling, omega, nsig, local_coupling, dx):
    "Gufunc for KuramotoT model equations."

    # long-range coupling
    c_pop0 = coupling[0]

    theta = vw[0]

    # derived variables
    c_pop0 = cpop0 * global_coupling

    dx[0] = omega + c_pop0
