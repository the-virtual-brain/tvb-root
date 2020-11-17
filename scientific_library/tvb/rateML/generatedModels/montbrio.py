from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class MontbrioT(ModelNumbaDfun):
        
    I = NArray(
        label=":math:`I`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc=""""""
    )    
        
    Delta = NArray(
        label=":math:`Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc=""""""
    )    
        
    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc=""""""
    )    
        
    s = NArray(
        label=":math:`s`",
        default=numpy.array([0.0]),
        domain=Range(lo=-15.0, hi=15.0, step=0.01),
        doc=""""""
    )    
        
    k = NArray(
        label=":math:`k`",
        default=numpy.array([0.0]),
        domain=Range(lo=-15.0, hi=15.0, step=0.01),
        doc=""""""
    )    
        
    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc=""""""
    )    
        
    eta = NArray(
        label=":math:`eta`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc=""""""
    )    
        
    Gamma = NArray(
        label=":math:`Gamma`",
        default=numpy.array([0.0]),
        domain=Range(lo=0., hi=10.0, step=0.1),
        doc=""""""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-2.0, hi=2.0, step=0.1),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]), 
				 "V": numpy.array([-2.0, 1.5])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"r": numpy.array([0.0, inf]), },
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('r', 'V', ),
        default=('V', 'V', ),
        doc="Variables to monitor"
    )

    state_variables = ['r', 'V']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_MontbrioT(vw_, c_, self.I, self.Delta, self.alpha, self.s, self.k, self.J, self.eta, self.Gamma, self.gamma, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*10 + '->(n)', nopython=True)
def _numba_dfun_MontbrioT(vw, coupling, I, Delta, alpha, s, k, J, eta, Gamma, gamma, local_coupling, dx):
    "Gufunc for MontbrioT model equations."

    # long-range coupling
    c_pop1 = coupling[0]
    c_pop2 = coupling[1]
    c_pop3 = coupling[2]
    c_pop4 = coupling[3]

    r = vw[0]
    V = vw[1]

    # derived variables
    Coupling_global = alpha * c_pop1
    Coupling_local = (1-alpha) * local_coupling * r
    Coupling_Term = Coupling_global + Coupling_local

    dx[0] = Delta / pi + 2 * V * r - k * r ** 2 + Gamma * r / pi
    dx[1] = V ** 2 - pi ** 2 * r ** 2 + eta + (k * s + J) * r - k * V * r + gamma * I + Coupling_Term
    