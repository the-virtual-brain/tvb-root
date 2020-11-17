from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class Kuramoto_pythonT(ModelNumbaDfun):
        
    omega = NArray(
        label=":math:`omega`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.01, hi=200.0, step=0.1),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"theta": numpy.array([0.0, numpy.pi * 2.0])},
        doc="""state variables"""
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
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_Kuramoto_pythonT(vw_, c_, self.omega, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64[:])], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)
def _numba_dfun_Kuramoto_pythonT(vw, coupling, omega, local_coupling, dx):
    "Gufunc for Kuramoto_pythonT model equations."

    # long-range coupling
    c_pop1 = coupling[0]
    c_pop2 = coupling[1]
    c_pop3 = coupling[2]
    c_pop4 = coupling[3]

    theta = vw[0]

    # derived variables
    coupl_strength = c_pop1 * sin(theta)

    dx[0] = omega + coupl_strength
    