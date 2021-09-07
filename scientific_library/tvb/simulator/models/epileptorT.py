from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class EpileptorT(ModelNumbaDfun):
        
    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        doc=""""""
    )    
        
    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        doc=""""""
    )    
        
    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        doc=""""""
    )    
        
    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.00035]),
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        doc=""""""
    )    
        
    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        doc=""""""
    )    
        
    x0 = NArray(
        label=":math:`x0`",
        default=numpy.array([-1.6]),
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        doc=""""""
    )    
        
    Iext = NArray(
        label=":math:`Iext`",
        default=numpy.array([3.1]),
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        doc=""""""
    )    
        
    slope = NArray(
        label=":math:`slope`",
        default=numpy.array([0.]),
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        doc=""""""
    )    
        
    Iext2 = NArray(
        label=":math:`Iext2`",
        default=numpy.array([0.45]),
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        doc=""""""
    )    
        
    tau = NArray(
        label=":math:`tau`",
        default=numpy.array([10.0]),
        doc=""""""
    )    
        
    aa = NArray(
        label=":math:`aa`",
        default=numpy.array([6.0]),
        doc=""""""
    )    
        
    bb = NArray(
        label=":math:`bb`",
        default=numpy.array([2.0]),
        doc=""""""
    )    
        
    Kvf = NArray(
        label=":math:`Kvf`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc=""""""
    )    
        
    Kf = NArray(
        label=":math:`Kf`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc=""""""
    )    
        
    Ks = NArray(
        label=":math:`Ks`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc=""""""
    )    
        
    tt = NArray(
        label=":math:`tt`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=10.0, step=0.001),
        doc=""""""
    )    
        
    modification = NArray(
        label=":math:`modification`",
        default=numpy.array([0]),
        doc=""""""
    )    

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x1": numpy.array([0.0]), 
				 "y1": numpy.array([0.0]), 
				 "z": numpy.array([0.0]), 
				 "x2": numpy.array([0.0]), 
				 "y2": numpy.array([0.0]), 
				 "g": numpy.array([0.0])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"x1": numpy.array([-2.0, 1.0]), 
				 "y1": numpy.array([-20.0, 2.0]), 
				 "z": numpy.array([-2.0, 5.0]), 
				 "x2": numpy.array([-2.0, 0.0]), 
				 "y2": numpy.array([0.0, 2.0]), 
				 "g": numpy.array([-1.0, 1.0])},
    )
    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('x1 ** x2', 'x2', ),
        default=('x1', 'y1', 'z', 'x2', 'y2', 'g', ),
        doc="Variables to monitor"
    )

    state_variables = ['x1', 'y1', 'z', 'x2', 'y2', 'g']

    _nvar = 6
    cvar = numpy.array([0,1,2,3,4,5,], dtype = numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_EpileptorT(vw_, c_, self.a, self.b, self.c, self.d, self.r, self.s, self.x0, self.Iext, self.slope, self.Iext2, self.tau, self.aa, self.bb, self.Kvf, self.Kf, self.Ks, self.tt, self.modification, local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*19 + '->(n)', nopython=True)
def _numba_dfun_EpileptorT(vw, coupling, a, b, c, d, r, s, x0, Iext, slope, Iext2, tau, aa, bb, Kvf, Kf, Ks, tt, modification, local_coupling, dx):
    "Gufunc for EpileptorT model equations."

    # long-range coupling
    c_pop0 = coupling[0]
    c_pop1 = coupling[1]
    c_pop2 = coupling[2]
    c_pop3 = coupling[3]
    c_pop4 = coupling[4]
    c_pop5 = coupling[5]

    x1 = vw[0]
    y1 = vw[1]
    z = vw[2]
    x2 = vw[3]
    y2 = vw[4]
    g = vw[5]


    # Conditional variables
    if x1 < 0.0:
        ydot0 = -a * x1 ** 2 + b * x1
    else:
        ydot0 = slope - x2 + 0.6 * z-4 ** 2

    if z < 0.0:
        ydot2 = - 0.1 * z ** 7
    else:
        ydot2 = 0

    if modification:
        h = x0 + 3. / (1. + exp(-(x1 + 0.5) / 0.1))
    else:
        h = 4 * (x1 - x0) + ydot2

    if x2 < -0.25:
        ydot4 = 0.0
    else:
        ydot4 = aa * (x2 + 0.25)

    dx[0] = tt * (y1 - z + Iext + Kvf * c_pop1 + ydot0 )
    dx[1] = tt * (c - d * x1 ** 2 - y1)
    dx[2] = tt * (r * (h - z + Ks * c_pop1))
    dx[3] = tt * (-y2 + x2 - x2 ** 3 + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2)
    dx[4] = tt * (-y2 + ydot4) / tau
    dx[5] = tt * (-0.01 * (g - 0.1 * x1) )
    
