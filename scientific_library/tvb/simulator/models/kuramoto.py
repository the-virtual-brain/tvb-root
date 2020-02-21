from .base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class Kuramoto(ModelNumbaDfun):

        
    omega = NArray(
        label=":math:`omega`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.01, hi=200.0, step=0.1),
        doc="""sets the base line frequency for the Kuramoto oscillator in [rad/ms]"""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"theta": numpy.array([0.0, pi * 2.0])},
        doc="""state variables"""
        )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("sin(theta)"),
        default=("theta", ),
        doc="The quantities of interest for monitoring for the generic 2D oscillator."
    )

    state_variables = ['theta']

    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):

        theta = state_variables[0,:]

        #[State_variables, nodes]

        omega = self.omega

        derivative = numpy.empty_like(state_variables)

        I = coupling[0] + sin(local_coupling * theta)




        ev('omega * I', out=derivative[0])

        return derivative

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_Kuramoto(vw_, c_, self.omega, local_coupling)

        return deriv.T[..., numpy.newaxis]

# @guvectorize([(float64[:],) * 5], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)
@guvectorize([(float64[:], float64[:], float64, float64, float64[:])], '(n),(m)' + ',()'*2 + '->(n)', nopython=True)

def _numba_dfun_Kuramoto(vw, coupling, omega, local_coupling, dx):
    "Gufunc for Kuramoto model equations."

    theta = vw[0]

    I = coupling[0] + sin(local_coupling * theta)




    dx[0] = omega * I
            