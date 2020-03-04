from .base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class Generic2dOscillatorT(ModelNumbaDfun):
        
    tau = NArray(
        label=":math:`tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=5.0, step=0.01),
        doc="""A time-scale hierarchy can be introduced for the state variables :math:`V` and :math:`W`. Default parameter is 1, which means no time-scale hierarchy."""
    )    
        
    I = NArray(
        label=":math:`I`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Baseline shift of the cubic nullcline"""
    )    
        
    a = NArray(
        label=":math:`a`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline"""
    )    
        
    b = NArray(
        label=":math:`b`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=15.0, step=0.01),
        doc="""Linear slope of the configurable nullcline"""
    )    
        
    c = NArray(
        label=":math:`c`",
        default=numpy.array([0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Parabolic term of the configurable nullcline"""
    )    
        
    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Temporal scale factor. Warning: do not use it unless you know what you are doing and know about time tides."""
    )    
        
    e = NArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the quadratic term of the cubic nullcline."""
    )    
        
    f = NArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the cubic term of the cubic nullcline."""
    )    
        
    g = NArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.5),
        doc="""Coefficient of the linear term of the cubic nullcline."""
    )    
        
    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the slow variable to the fast variable."""
    )    
        
    beta = NArray(
        label=":math:`beta`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the slow variable to itself"""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where excitatory input currents are negative. It scales both I and the long range coupling term.."""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]), 
				 "W": numpy.array([-6.0, 6.0])},
        doc="""state variables"""
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('V', 'W', 'V + W', 'V - W', ),
        default=('V', ),
        doc=""
    )

    state_variables = ['V', 'W']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_Generic2dOscillatorT(vw_, c_, self.tau, self.I, self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.alpha, self.beta, self.gamma, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*13 + '->(n)', nopython=True)
def _numba_dfun_Generic2dOscillatorT(vw, coupling, tau, I, a, b, c, d, e, f, g, alpha, beta, gamma, local_coupling, dx):
    "Gufunc for Generic2dOscillatorT model equations."

    V = vw[0]
    W = vw[1]


    dx[0] = d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma * coupling[0] + local_coupling)
    dx[1] = d * (a + b * V + c * V**2 - beta * W) / tau
            