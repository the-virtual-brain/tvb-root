from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class rwongwang(ModelNumbaDfun):
        
    w_plus = NArray(
        label=":math:`w_plus`",
        default=numpy.array([1.4f]),
        doc="""Excitatory population recurrence weight"""
    )    
        
    a_E = NArray(
        label=":math:`a_E`",
        default=numpy.array([310.0f]),
        doc="""[n/C]. Excitatory population input gain parameter, chosen to fit numerical solutions."""
    )    
        
    b_E = NArray(
        label=":math:`b_E`",
        default=numpy.array([125.0f]),
        doc="""[Hz]. Excitatory population input shift parameter chosen to fit numerical solutions."""
    )    
        
    d_E = NArray(
        label=":math:`d_E`",
        default=numpy.array([0.154f]),
        doc="""[s]. Excitatory population input scaling parameter chosen to fit numerical solutions."""
    )    
        
    a_I = NArray(
        label=":math:`a_I`",
        default=numpy.array([615.0f]),
        doc="""[n/C]. Inhibitory population input gain parameter, chosen to fit numerical solutions."""
    )    
        
    b_I = NArray(
        label=":math:`b_I`",
        default=numpy.array([177.0f]),
        doc="""[Hz]. Inhibitory population input shift parameter chosen to fit numerical solutions."""
    )    
        
    d_I = NArray(
        label=":math:`d_I`",
        default=numpy.array([0.087f]),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions."""
    )    
        
    gamma_E = NArray(
        label=":math:`gamma_E`",
        default=numpy.array([0.641f / 1000.0f]),
        doc="""Excitatory population kinetic parameter"""
    )    
        
    tau_E = NArray(
        label=":math:`tau_E`",
        default=numpy.array([100.0f]),
        doc="""[ms]. Excitatory population NMDA decay time constant."""
    )    
        
    tau_I = NArray(
        label=":math:`tau_I`",
        default=numpy.array([10.0f]),
        doc="""[ms]. Inhibitory population NMDA decay time constant."""
    )    
        
    I_0 = NArray(
        label=":math:`I_0`",
        default=numpy.array([0.382f]),
        doc="""[nA]. Effective external input"""
    )    
        
    w_E = NArray(
        label=":math:`w_E`",
        default=numpy.array([1.0f]),
        doc="""Excitatory population external input scaling weight"""
    )    
        
    w_I = NArray(
        label=":math:`w_I`",
        default=numpy.array([0.7f]),
        doc="""Inhibitory population external input scaling weight"""
    )    
        
    gamma_I = NArray(
        label=":math:`gamma_I`",
        default=numpy.array([1.0f / 1000.0f]),
        doc="""Inhibitory population kinetic parameter"""
    )    
        
    min_d_E = NArray(
        label=":math:`min_d_E`",
        default=numpy.array([-1.0f * d_E]),
        doc="""Only in CUDA"""
    )    
        
    min_d_I = NArray(
        label=":math:`min_d_I`",
        default=numpy.array([-1.0f * d_I]),
        doc="""Only in CUDA"""
    )    
        
    imintau_E = NArray(
        label=":math:`imintau_E`",
        default=numpy.array([-1.0f / tau_E]),
        doc="""Only in CUDA"""
    )    
        
    imintau_I = NArray(
        label=":math:`imintau_I`",
        default=numpy.array([-1.0f / tau_I]),
        doc="""Only in CUDA"""
    )    
        
    w_E__I_0 = NArray(
        label=":math:`w_E__I_0`",
        default=numpy.array([w_E * I_0]),
        doc="""Only in CUDA"""
    )    
        
    w_I__I_0 = NArray(
        label=":math:`w_I__I_0`",
        default=numpy.array([w_I * I_0]),
        doc="""Only in CUDA"""
    )    
        
    J_N = NArray(
        label=":math:`J_N`",
        default=numpy.array([0.15]),
        doc="""[nA] NMDA current"""
    )    
        
    J_I = NArray(
        label=":math:`J_I`",
        default=numpy.array([1.0]),
        doc="""[nA] Local inhibitory current"""
    )    
        
    G = NArray(
        label=":math:`G`",
        default=numpy.array([2.0]),
        doc="""Global coupling scaling"""
    )    
        
    lamda = NArray(
        label=":math:`lamda`",
        default=numpy.array([0.0]),
        doc="""Inhibitory global coupling scaling"""
    )    
        
    J_NMDA = NArray(
        label=":math:`J_NMDA`",
        default=numpy.array([0.15]),
        doc=""""""
    )    
        
    JI = NArray(
        label=":math:`JI`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    G_J_NMDA = NArray(
        label=":math:`G_J_NMDA`",
        default=numpy.array([G*J_NMDA]),
        doc=""""""
    )    
        
    w_plus__J_NMDA = NArray(
        label=":math:`w_plus__J_NMDA`",
        default=numpy.array([w_plus * J_NMDA]),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([]), 
				 "W": numpy.array([])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"V": numpy.array([0.0000001, 1])"W": numpy.array([0.0000001, 1])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('V ** 2', ),
        default=('V', 'W', ),
        doc="Variables to monitor"
    )

    state_variables = ['V', 'W']

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_rwongwang(vw_, c_, self.w_plus, self.a_E, self.b_E, self.d_E, self.a_I, self.b_I, self.d_I, self.gamma_E, self.tau_E, self.tau_I, self.I_0, self.w_E, self.w_I, self.gamma_I, self.min_d_E, self.min_d_I, self.imintau_E, self.imintau_I, self.w_E__I_0, self.w_I__I_0, self.J_N, self.J_I, self.G, self.lamda, self.J_NMDA, self.JI, self.G_J_NMDA, self.w_plus__J_NMDA, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*29 + '->(n)', nopython=True)
def _numba_dfun_rwongwang(vw, coupling, w_plus, a_E, b_E, d_E, a_I, b_I, d_I, gamma_E, tau_E, tau_I, I_0, w_E, w_I, gamma_I, min_d_E, min_d_I, imintau_E, imintau_I, w_E__I_0, w_I__I_0, J_N, J_I, G, lamda, J_NMDA, JI, G_J_NMDA, w_plus__J_NMDA, local_coupling, dx):
    "Gufunc for rwongwang model equations."

    V = vw[0]
    W = vw[1]

    # derived variables
    tmp_I_E = a_E * (w_E__I_0 + w_plus__J_NMDA * V + c_0 - JI*W) - b_E
    tmp_H_E = tmp_I_E/(1.0-exp(min_d_E * tmp_I_E))
    tmp_I_I = (a_I*((w_I__I_0+(J_NMDA * V))-W))-b_I
    tmp_H_I = tmp_I_I/(1.0-exp(min_d_I*tmp_I_I))



    dx[0] = (imintau_E* V)+(tmp_H_E*(1-V)*gamma_E)
    dx[1] = (imintau_I* W)+(tmp_H_I*gamma_I)
            