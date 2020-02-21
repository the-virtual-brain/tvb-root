from .base import Model, ModelNumbaDfun
import numexpr
import numpy
from numpy import *
from numba import guvectorize, float64
from tvb.basic.neotraits.api import NArray, Final, List, Range


class Epileptor(ModelNumbaDfun):
    a = NArray(
        label=":math:`a`",
        default=numpy.array([1.0]),
        domain=Range(0, 0),
        doc="""Coefficient of the cubic term in the first state-variable."""
    )

    b = NArray(
        label=":math:`b`",
        default=numpy.array([3.0]),
        domain=Range(0, 0),
        doc="""Coefficient of the squared term in the first state-variable."""
    )

    c = NArray(
        label=":math:`c`",
        default=numpy.array([1.0]),
        domain=Range(0, 0),
        doc="""Additive coefficient for the second state-variable x_{2}, called :math:`y_{0}` in Jirsa paper."""
    )

    d = NArray(
        label=":math:`d`",
        default=numpy.array([5.0]),
        domain=Range(0, 0),
        doc="""Coefficient of the squared term in the second state-variable x_{2}."""
    )

    r = NArray(
        label=":math:`r`",
        default=numpy.array([0.00035]),
        domain=Range(lo=0.0, hi=0.001, step=0.00005),
        doc="""Temporal scaling in the slow state-variable, called :math:`1\\tau_{0}` in Jirsa paper (see class Epileptor)."""
    )

    s = NArray(
        label=":math:`s`",
        default=numpy.array([4.0]),
        domain=Range(0, 0),
        doc="""Linear coefficient in the third state variable"""
    )

    x0 = NArray(
        label=":math:`x0`",
        default=numpy.array([-1.6]),
        domain=Range(lo=-3.0, hi=-1.0, step=0.1),
        doc="""Epileptogenicity parameter."""
    )

    Iext = NArray(
        label=":math:`Iext`",
        default=numpy.array([3.1]),
        domain=Range(lo=1.5, hi=5.0, step=0.1),
        doc="""External input current to the first state-variable."""
    )

    slope = NArray(
        label=":math:`slope`",
        default=numpy.array([0.]),
        domain=Range(lo=-16.0, hi=6.0, step=0.1),
        doc="""Linear coefficient in the first state-variable."""
    )

    Iext2 = NArray(
        label=":math:`Iext2`",
        default=numpy.array([3.1]),
        domain=Range(lo=0.0, hi=1.0, step=0.05),
        doc="""External input current to the first state-variable."""
    )

    tau = NArray(
        label=":math:`tau`",
        default=numpy.array([10.0]),
        domain=Range(0, 0),
        doc="""Temporal scaling coefficient in fifth state variable."""
    )

    aa = NArray(
        label=":math:`aa`",
        default=numpy.array([6.0]),
        domain=Range(0, 0),
        doc="""Linear coefficient in fifth state variable."""
    )

    bb = NArray(
        label=":math:`bb`",
        default=numpy.array([2.0]),
        domain=Range(0, 0),
        doc="""Linear coefficient of lowpass excitatory coupling in fourth state variable."""
    )

    Kvf = NArray(
        label=":math:`Kvf`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="""Coupling scaling on a very fast time scale."""
    )

    Kf = NArray(
        label=":math:`Kf`",
        default=numpy.array([0.0]),
        domain=Range(lo=0.0, hi=4.0, step=0.5),
        doc="""Correspond to the coupling scaling on a fast time scale."""
    )

    Ks = NArray(
        label=":math:`Ks`",
        default=numpy.array([0.0]),
        domain=Range(lo=-4.0, hi=4.0, step=0.1),
        doc="""Permittivity coupling, that is from the fast time scale toward the slow time scale."""
    )

    tt = NArray(
        label=":math:`tt`",
        default=numpy.array([1.0]),
        domain=Range(lo=0.001, hi=1.0, step=0.001),
        doc="""Time scaling of the whole system to the system in real time."""
    )

    modification = NArray(
        label=":math:`modification`",
        default=numpy.array([False]),
        domain=Range(0, 0),
        doc="""When modification is True, then use nonlinear influence on z. The default value is False, i.e., linear influence."""
    )

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x1": numpy.array([-2., 1.]),
                 "y1": numpy.array([-20., 2.]),
                 "z": numpy.array([2.0, 5.0]),
                 "x2": numpy.array([-2., 0.]),
                 "y2": numpy.array([0., 2.]),
                 "g": numpy.array([-1, 1.])},
        doc="""state variables"""
    )

    variables_of_interest = List(
        of=str,
        label="Variables or quantities available to Monitors",
        choices=("'x1', 'y1', 'z', 'x2', 'y2', 'g', 'x2 - x1'"),
        default=("x1",),
        doc="The quantities of interest for monitoring for the generic 2D oscillator."
    )

    state_variables = ['x1', 'y1', 'z', 'x2', 'y2', 'g']

    _nvar = 6
    cvar = numpy.array([0], dtype=numpy.int32)

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0, ev=numexpr.evaluate):
        x1 = state_variables[0, :]
        y1 = state_variables[1, :]
        z = state_variables[2, :]
        x2 = state_variables[3, :]
        y2 = state_variables[4, :]
        g = state_variables[5, :]

        # [State_variables, nodes]

        a = self.a
        b = self.b
        c = self.c
        d = self.d
        r = self.r
        s = self.s
        x0 = self.x0
        Iext = self.Iext
        slope = self.slope
        Iext2 = self.Iext2
        tau = self.tau
        aa = self.aa
        bb = self.bb
        Kvf = self.Kvf
        Kf = self.Kf
        Ks = self.Ks
        tt = self.tt
        modification = self.modification

        derivative = numpy.empty_like(state_variables)

        c_pop1 = coupling[0]
        c_pop2 = coupling[1]
        Iext = Iext + local_coupling * x1

        if_x1 = (x1 < 0.0) * (- a * x1 ** 2 + b)
        else_x1 = (x1 >= 0.0) * (slope - x2 + 0.6 * (z - 4) ** 2)
        if_z = (y1 < 0.0) * (- 0.1 * (z ** 7))
        else_z = 0
        ifmod_h = (modification == True) * (x0 + 3. / (1. + exp(-(x1 + 0.5) / 0.1)))
        elsemod_h = (modification == False) * (4 * (x1 - x0) + if_z)
        if_y2 = 0
        else_y2 = (x2 >= -0.25) * (aa * (x2 + 0.25))

        ev('tt * (y1 - z + Iext + Kvf * c_pop1 + (if_x1 + else_x1) * x1)', out=derivative[0])
        ev('tt * (c - d * x1**2 - y1)', out=derivative[1])
        ev('tt * (r * ((ifmod_h + elsemod_h) - z + Ks * c_pop1))', out=derivative[2])
        ev('tt * (-y2 + x2 - x2**3 + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2)', out=derivative[3])
        ev('tt * (-y2 + else_y2) / tau', out=derivative[4])
        ev('tt * (-0.01 * (g - 0.1 * x1) )', out=derivative[5])

        print(('derivative', derivative[0], derivative[1], derivative[2], derivative[3], derivative[4], derivative[5]))

        return derivative

    def dfun(self, vw, c, local_coupling=0.0):
        # print((vw[0], vw[1], vw[2], vw[3], vw[4], vw[5]))
        vw_ = vw.reshape(vw.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T
        # Iext = self.Iext + local_coupling * x[0, :, 0]
        deriv = _numba_dfun_Epileptor(vw_, c_, self.a, self.b, self.c, self.d, self.r, self.s, self.x0, self.Iext,
                                      self.slope, self.Iext2, self.tau, self.aa, self.bb, self.Kvf, self.Kf, self.Ks,
                                      self.tt, self.modification, local_coupling)

        return deriv.T[..., numpy.newaxis]


#
# # @guvectorize([(float64[:],) * 22], '(n),(m)' + ',()'*19 + '->(n)', nopython=True)
# @guvectorize([(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:],\
#                float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:])],\
#              '(n),(m)' + ',()'*19 + '->(n)', nopython=True)
@guvectorize([(float64[:], float64[:], float64, float64, float64, float64, float64, float64, float64, float64, float64, \
               float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64[:])], \
             '(n),(m)' + ',()' * 19 + '->(n)', nopython=True)
def _numba_dfun_Epileptor(y, coupling, a, b, c, d, r, s, x0, Iext, slope, Iext2, tau, aa, bb, Kvf, Kf, Ks, tt,
                          modification, local_coupling, ydot):
    #     "Gufunc for Epileptor model equations."
    #
    # x1 = vw[0]
    # y1 = vw[1]
    # z = vw[2]
    # x2 = vw[3]
    # y2 = vw[4]
    # g = vw[5]

    c_pop1 = coupling[0]
    c_pop2 = coupling[1]
    Iext = Iext + local_coupling * y[0]

    # population 1
    if y[0] < 0.0:
        ydot[0] = - a * y[0] ** 2 + b * y[0]
    else:
        ydot[0] = slope - y[3] + 0.6 * (y[2] - 4.0) ** 2

    ydot[0] = tt * (y[1] - y[2] + Iext + Kvf * c_pop1 + (ydot[0]) * y[0])
    ydot[1] = tt * (c - d * y[0] ** 2 - y[1])

    # energy
    if y[2] < 0.0:
        ydot[2] = - 0.1 * y[2] ** 7
    else:
        ydot[2] = 0.0
    if modification:
        h = x0 + 3 / (1 + numpy.exp(-(y[0] + 0.5) / 0.1))
    else:
        h = 4 * (y[0] - x0) + ydot[2]
    ydot[2] = tt * (r * (h - y[2] + Ks * c_pop1))

    # population 2
    ydot[3] = tt * (-y[4] + y[3] - y[3] ** 3 + Iext2 + bb * y[5] - 0.3 * (y[2] - 3.5) + Kf * c_pop2)
    if y[3] < -0.25:
        ydot[4] = 0.0
    else:
        ydot[4] = aa * (y[3] + 0.25)
    ydot[4] = tt * ((-y[4] + ydot[4]) / tau)

    # filter
    ydot[5] = tt * (-0.01 * (y[5] - 0.1 * y[0]))

    # dx[0] = tt * (y1 - z + Iext + Kvf * c_pop1 + (if_x1 + else_x1) * x1)
    # dx[1] = tt * (c - d * x1**2 - y1)
    # dx[2] = tt * (r * ((ifmod_h + elsemod_h) - z + Ks * c_pop1))
    # dx[3] = tt * (-y2 + x2 - x2**3 + Iext2 + bb * g - 0.3 * (z - 3.5) + Kf * c_pop2)
    # dx[4] = tt * (-y2 + else_y2) / tau
    # dx[5] = tt * (-0.01 * (g - 0.1 * x1) )

    # def dfun(self, x, c, local_coupling=0.0):
    #     print((x[0], x[1], x[2], x[3], x[4], x[5]))
    #     x_ = x.reshape(x.shape[:-1]).T
    #     c_ = c.reshape(c.shape[:-1]).T
    #     Iext = self.Iext + local_coupling * x[0, :, 0]
    #     deriv = _numba_dfun(x_, c_,
    #                      self.x0, Iext, self.Iext2, self.a, self.b, self.slope, self.tt, self.Kvf,
    #                      self.c, self.d, self.r, self.Ks, self.Kf, self.aa, self.bb, self.tau, self.modification)
    #     return deriv.T[..., numpy.newaxis]

# @guvectorize([(float64[:],) * 20], '(n),(m)' + ',()'*17 + '->(n)', nopython=True)
# def _numba_dfun_Epileptor(y, coupling, a, b, c, d, r, s, x0, Iext, slope, Iext2, tau, aa, bb, Kvf, Kf, Ks, tt, modification, local_coupling, ydot):
#     "Gufunc for Hindmarsh-Rose-Jirsa Epileptor model equations."

# c_pop1 = coupling[0]
# c_pop2 = coupling[1]
#
# # population 1
# if y[0] < 0.0:
#     ydot[0] = - a[0] * y[0] ** 2 + b[0] * y[0]
# else:
#     ydot[0] = slope[0] - y[3] + 0.6 * (y[2] - 4.0) ** 2
# ydot[0] = tt[0] * (y[1] - y[2] + Iext[0] + Kvf[0] * c_pop1 + ydot[0] * y[0])
# ydot[1] = tt[0] * (c[0] - d[0] * y[0] ** 2 - y[1])
#
# # energy
# if y[2] < 0.0:
#     ydot[2] = - 0.1 * y[2] ** 7
# else:
#     ydot[2] = 0.0
# if modification[0]:
#     h =  x0[0] + 3/(1 + numpy.exp(-(y[0]+0.5)/0.1))
# else:
#     h = 4 * (y[0] - x0[0]) + ydot[2]
# ydot[2] = tt[0] * (r[0] * (h - y[2]  + Ks[0] * c_pop1))
#
# # population 2
# ydot[3] = tt[0] * (-y[4] + y[3] - y[3] ** 3 + Iext2[0] + bb[0] * y[5] - 0.3 * (y[2] - 3.5) + Kf[0] * c_pop2)
# if y[3] < -0.25:
#     ydot[4] = 0.0
# else:
#     ydot[4] = aa[0] * (y[3] + 0.25)
# ydot[4] = tt[0] * ((-y[4] + ydot[4]) / tau[0])
#
# # filter
# ydot[5] = tt[0] * (-0.01 * (y[5] - 0.1 * y[0]))