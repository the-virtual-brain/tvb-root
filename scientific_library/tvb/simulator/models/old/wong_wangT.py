from .base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class ReducedWongWangT(ModelNumbaDfun):

        
    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.270]),
        domain=Range(lo=0.0, hi=0.270, step=0.01),
        doc="""[n/C]. Input gain parameter, chosen to fit numerical solutions."""
    )    
        
    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.108]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[kHz]. Input shift parameter chosen to fit numerical solutions."""
    )    
        
    d = NArray(
        label=":math:`d`",
        default=numpy.array([154.]),
        domain=Range(lo=0.0, hi=200.0, step=0.01),
        doc="""[ms]. Parameter chosen to fit numerical solutions."""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([0.641]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Kinetic parameter"""
    )    
        
    tau_s = NArray(
        label=":math:`tau_s`",
        default=numpy.array([100.]),
        domain=Range(lo=50., hi=150., step=1.),
        doc="""Kinetic parameter. NMDA decay time constant."""
    )    
        
    w = NArray(
        label=":math:`w`",
        default=numpy.array([0.6]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Excitatory recurrence"""
    )    
        
    J_N = NArray(
        label=":math:`J_N`",
        default=numpy.array([0.2609]),
        domain=Range(lo=0.2609, hi=0.5, step=0.001),
        doc="""Excitatory recurrence"""
    )    
        
    I_o = NArray(
        label=":math:`I_o`",
        default=numpy.array([0.33]),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""[nA] Effective external input"""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"S": numpy.array([0.0, 1.0])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S": numpy.array([0.0, 1.0])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('S', ),
        default=('S', ),
        doc="default state variables to be monitored"
    )

    state_variables = ['S']

    _nvar = 1
    cvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):

        S = state_variables[0,:]

        #[State_variables, nodes]

        a = self.a
        b = self.b
        d = self.d
        gamma = self.gamma
        tau_s = self.tau_s
        w = self.w
        J_N = self.J_N
        I_o = self.I_o

        derivative = numpy.empty_like(state_variables)

        x = w * J_N * S + I_o + J_N * coupling[0] + J_N * local_coupling
        H = (a * x - b) / (1 - exp(-d * (a * x - b)))


        ev('- (S / tau_s) + (1 - S) * H * gamma', out=derivative[0])

        return derivative

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_ReducedWongWangT(vw_, c_, self.a, self.b, self.d, self.gamma, self.tau_s, self.w, self.J_N, self.I_o, local_coupling)

        return deriv.T[..., numpy.newaxis]

# @guvectorize([(float64[:],) * 12], '(n),(m)' + ',()'*9 + '->(n)', nopython=True)
@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*9 + '->(n)', nopython=True)

def _numba_dfun_ReducedWongWangT(vw, coupling, a, b, d, gamma, tau_s, w, J_N, I_o, local_coupling, dx):
    "Gufunc for ReducedWongWangT model equations."

    S = vw[0]

    x = w * J_N * S + I_o + J_N * coupling[0] + J_N * local_coupling
    H = (a * x - b) / (1 - exp(-d * (a * x - b)))


    dx[0] = - (S / tau_s) + (1 - S) * H * gamma
            