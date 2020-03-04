from .base import Model, ModelNumbaDfun
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
        doc="""???"""
    )    
        
    Delta = NArray(
        label=":math:`Delta`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline."""
    )    
        
    alpha = NArray(
        label=":math:`alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.0, hi=1.0, step=0.1),
        doc=""":math:`\alpha` ratio of effect between long-range and local connectivity."""
    )    
        
    s = NArray(
        label=":math:`s`",
        default=numpy.array([0.0]),
        domain=Range(lo=-15.0, hi=15.0, step=0.01),
        doc="""QIF membrane reversal potential."""
    )    
        
    k = NArray(
        label=":math:`k`",
        default=numpy.array([0.0]),
        domain=Range(lo=-15.0, hi=15.0, step=0.01),
        doc="""Switch for the terms specific to Coombes model."""
    )    
        
    J = NArray(
        label=":math:`J`",
        default=numpy.array([15.0]),
        domain=Range(lo=-25.0, hi=25.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the slow variable to the firing rate variable."""
    )    
        
    eta = NArray(
        label=":math:`eta`",
        default=numpy.array([-5.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the firing rate variable to itself"""
    )    
        
    Gamma = NArray(
        label=":math:`Gamma`",
        default=numpy.array([0.0]),
        domain=Range(lo=0., hi=10.0, step=0.1),
        doc="""Derived from eterogeneous currents and synaptic weights (see Montbrio p.12)."""
    )    
        
    gamma = NArray(
        label=":math:`gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-2.0, hi=2.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where excitatory input currents are negative. It scales both I and the long range coupling term."""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0., 2.0]), 
				 "V": numpy.array([-2.0, 1.5])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"r": numpy.array([0.0, inf])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('r', 'V', ),
        default=('r', 'V', ),
        doc="The quantities of interest for monitoring for the Infinite QIF 2D oscillator."
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

    r = vw[0]
    V = vw[1]

    Coupling_global = alpha * coupling[0]
    Coupling_local = (1-alpha) * local_coupling * r
    Coupling_Term = Coupling_global + Coupling_local

    dx[0] = Delta / pi + 2 * V * r - k * r**2 + Gamma * r / pi
    dx[1] = V**2 - pi**2 * r**2 + eta + (k * s + J) * r - k * V * r + gamma * I + Coupling_Term
            