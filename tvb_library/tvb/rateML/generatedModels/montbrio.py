
from tvb.simulator.models.base import Model, ModelNumbaDfun
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class MontbrioT(ModelNumbaDfun):

    tau = NArray(
        label = ":math:`tau`",
        default = numpy.array([1.0]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """???"""
    )

    I = NArray(
        label = ":math:`I`",
        default = numpy.array([0.0]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """???"""
    )

    Delta = NArray(
        label = ":math:`Delta`",
        default = numpy.array([0.7]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """Vertical shift of the configurable nullcline."""
    )

    J = NArray(
        label = ":math:`J`",
        default = numpy.array([14.5]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """Constant parameter to scale the rate of feedback from the slow variable to the firing rate variable."""
    )

    eta = NArray(
        label = ":math:`eta`",
        default = numpy.array([-4.6]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """Constant parameter to scale the rate of feedback from the firing rate variable to itself"""
    )

    Gamma = NArray(
        label = ":math:`Gamma`",
        default = numpy.array([5.0]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """Derived from eterogeneous currents and synaptic weights (see Montbrio p.12)."""
    )

    cr = NArray(
        label = ":math:`cr`",
        default = numpy.array([1.0]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """It is the weight on Coupling through variable r."""
    )

    cv = NArray(
        label = ":math:`cv`",
        default = numpy.array([1.0]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """It is the weight on Coupling through variable V."""
    )

    nsig = NArray(
        label = ":math:`nsig`",
        default = numpy.array([0.5]),
        domain = Range(lo=0.0, hi=1000.0, step=0.1),
        doc = """noiseparameter"""
    )

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"r": numpy.array([0.0, -2.0]), 
				 "V": numpy.array([-2.0, 1.5])},
        doc="""state variables"""
    )

    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"r": numpy.array([0, INF]), 
				 "V": numpy.array([0.0, 0.0])},
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=('r', 'V', ),
        default=('r', 'V', ),
        doc="Variables to monitor"
    )

    state_variables = ['r', 'V']

    _nvar = 2
    cvar = numpy.array([0,1,], dtype = numpy.int32)

    def dfun(self, vw, c, local_coupling=0.0):
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_MontbrioT(vw_, c_,     # %if not (para.maxval):
self.tau,     # %endif
    # %if not (para.maxval):
self.I,     # %endif
    # %if not (para.maxval):
self.Delta,     # %endif
    # %if not (para.maxval):
self.J,     # %endif
    # %if not (para.maxval):
self.eta,     # %endif
    # %if not (para.maxval):
self.Gamma,     # %endif
    # %if not (para.maxval):
self.cr,     # %endif
    # %if not (para.maxval):
self.cv,     # %endif
    # %if not (para.maxval):
self.nsig,     # %endif
local_coupling)

        return deriv.T[..., numpy.newaxis]

@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], '(n),(m)' + ',()'*10 + '->(n)', nopython=True)
def _numba_dfun_MontbrioT(vw, coupling, tau, I, Delta, J, eta, Gamma, cr, cv, nsig, local_coupling, dx):
    "Gufunc for MontbrioT model equations."

    # long-range coupling
    c_pop0 = coupling[0]
    c_pop1 = coupling[1]

    r = vw[0]
    V = vw[1]


    dx[0] = 1/tau * (Delta / (pi * tau) + 2 * V * r)
    dx[1] = 1/tau * (V**2 - pi**2 * tau**2 * r**2 + eta + J * tau * r + I + cr * (cpop0 * global_coupling))
