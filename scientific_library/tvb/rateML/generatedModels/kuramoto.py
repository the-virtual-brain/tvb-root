from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class KuramotoT(ModelNumbaDfun):
        
    omega = NArray(
        label=":math:`omega`",
        default=numpy.array([60.0 * 2.0 * 3.1415927 / 1e3]),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([0.0])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"V": numpy.array([-2, 1])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('V', ),
        default=('V', ),
        doc="Variables to monitor"
    )

    state_variables = ['V']

    _nvar = 1
    cvar = numpy.array([0,], dtype = numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_KuramotoT(vw_, c_, self.omega, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64[:])], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)
def _numba_dfun_KuramotoT(vw, coupling, omega, local_coupling, dx):
    "Gufunc for KuramotoT model equations."

    # long-range coupling
    c_pop0 = coupling[0]

    V = vw[0]


    dx[0] = omega + c_pop1
    
