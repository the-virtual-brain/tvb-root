from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class OscillatorT(ModelNumbaDfun):
        
    tau = NArray(
        label=":math:`tau`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    I = NArray(
        label=":math:`I`",
        default=numpy.array([0.1]),
        doc=""""""
    )    
        
    a = NArray(
        label=":math:`a`",
        default=numpy.array([0.5]),
        doc=""""""
    )    
        
    b = NArray(
        label=":math:`b`",
        default=numpy.array([0.4]),
        doc=""""""
    )    
        
    c = NArray(
        label=":math:`c`",
        default=numpy.array([-4.0]),
        doc=""""""
    )    
        
    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        doc=""""""
    )    
        
    e = NArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        doc=""""""
    )    
        
    f = NArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    g = NArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        doc=""""""
    )    
        
    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    beta = NArray(
        label=":math:`beta`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([1.0]),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([0.0, 0.0]), 
				 "W": numpy.array([0.0, 0.0])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]), 
				 "W": numpy.array([-6.0, 6.0])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('V', 'W', ),
        default=('V', 'W', ),

        doc="Variables to monitor"
    )

    state_variables = ['V', 'W']

    _nvar = 2
    cvar = numpy.array([0,1,], dtype = numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_OscillatorT(vw_, c_, self.tau, self.I, self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.alpha, self.beta, self.gamma, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*13 + '->(n)', nopython=True)
def _numba_dfun_OscillatorT(vw, coupling, tau, I, a, b, c, d, e, f, g, alpha, beta, gamma, local_coupling, dx):
    "Gufunc for OscillatorT model equations."

    # long-range coupling
    c_pop0 = coupling[0]
    c_pop1 = coupling[1]

    V = vw[0]
    W = vw[1]


    dx[0] = d * tau * (alpha * W - f * V**3 + e * V**2 + g * V + gamma * I + gamma * c_pop0 * V)
    dx[1] = d * (a + b * V + c * V**2 - beta * W) / tau
    
