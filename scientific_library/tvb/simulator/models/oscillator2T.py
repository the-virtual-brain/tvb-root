from tvb.simulator.models.base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range

class oscillator2(ModelNumbaDfun):

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=1.0, hi=5.0, step=0.01),
        doc="""A time-scale hierarchy can be introduced for the state
            variables :math:`V` and :math:`W`. Default parameter is 1, which means
            no time-scale hierarchy.""")

    I = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Baseline shift of the cubic nullcline""")

    a = NArray(
        label=":math:`a`",
        default=numpy.array([-2.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.01),
        doc="""Vertical shift of the configurable nullcline""")

    b = NArray(
        label=":math:`b`",
        default=numpy.array([-10.0]),
        domain=Range(lo=-20.0, hi=15.0, step=0.01),
        doc="""Linear slope of the configurable nullcline""")

    c = NArray(
        label=":math:`c`",
        default=numpy.array([0.0]),
        domain=Range(lo=-10.0, hi=10.0, step=0.01),
        doc="""Parabolic term of the configurable nullcline""")

    d = NArray(
        label=":math:`d`",
        default=numpy.array([0.02]),
        domain=Range(lo=0.0001, hi=1.0, step=0.0001),
        doc="""Temporal scale factor. Warning: do not use it unless
            you know what you are doing and know about time tides.""")

    e = NArray(
        label=":math:`e`",
        default=numpy.array([3.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the quadratic term of the cubic nullcline.""")

    f = NArray(
        label=":math:`f`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Coefficient of the cubic term of the cubic nullcline.""")

    g = NArray(
        label=":math:`g`",
        default=numpy.array([0.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.5),
        doc="""Coefficient of the linear term of the cubic nullcline.""")

    alpha = NArray(
        label=r":math:`\alpha`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
                slow variable to the fast variable.""")

    beta = NArray(
        label=r":math:`\beta`",
        default=numpy.array([1.0]),
        domain=Range(lo=-5.0, hi=5.0, step=0.0001),
        doc="""Constant parameter to scale the rate of feedback from the
                slow variable to itself""")

    # This parameter is basically a hack to avoid having a negative lower boundary in the global coupling strength.
    gamma = NArray(
        label=r":math:`\gamma`",
        default=numpy.array([1.0]),
        domain=Range(lo=-1.0, hi=1.0, step=0.1),
        doc="""Constant parameter to reproduce FHN dynamics where
                   excitatory input currents are negative.
                   It scales both I and the long range coupling term.""")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"V": numpy.array([-2.0, 4.0]),
                 "W": numpy.array([-6.0, 6.0])},
        doc="""The values for each state-variable should be set to encompass
                the expected dynamic range of that state-variable for the current
                parameters, it is used as a mechanism for bounding random initial
                conditions when the simulation isn't started from an explicit
                history, it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("V", "W", "V + W", "V - W"),
        default=("V",),
        doc="The quantities of interest for monitoring for the generic 2D oscillator.")

    state_variables = ('V', 'W')
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)


    def dfun(self, vw, c, local_coupling=0.0):
        r"""
        The two state variables :math:`V` and :math:`W` are typically considered
        to represent a function of the neuron's membrane potential, such as the
        firing rate or dendritic currents, and a recovery variable, respectively.
        If there is a time scale hierarchy, then typically :math:`V` is faster
        than :math:`W` corresponding to a value of :math:`\tau` greater than 1.

        The equations of the generic 2D population model read

        .. math::
                \dot{V} &= d \, \tau (-f V^3 + e V^2 + g V + \alpha W + \gamma I) \\
                \dot{W} &= \dfrac{d}{\tau}\,\,(c V^2 + b V - \beta W + a)

        where external currents :math:`I` provide the entry point for local,
        long-range connectivity and stimulation.

        """
        lc_0 = local_coupling * vw[0, :, 0]
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        deriv = _numba_dfun_g2d(vw_, c_, self.tau, self.I, self.a, self.b, self.c, self.d, self.e, self.f, self.g,
                                self.beta, self.alpha, self.gamma, lc_0)
        return deriv.T[..., numpy.newaxis]


@guvectorize([(float64[:],) * 16], '(n),(m)' + ',()' * 13 + '->(n)', nopython=True)
def _numba_dfun_g2d(vw, c_0, tau, I, a, b, c, d, e, f, g, beta, alpha, gamma, lc_0, dx):
    "Gufunc for reduced Wong-Wang model equations."
    V = vw[0]
    V2 = V * V
    W = vw[1]
    dx[0] = d[0] * tau[0] * (
            alpha[0] * W - f[0] * V2 * V + e[0] * V2 + g[0] * V + gamma[0] * I[0] + gamma[0] * c_0[0] + lc_0[0])
    dx[1] = d[0] * (a[0] + b[0] * V + c[0] * V2 - beta[0] * W) / tau[0]