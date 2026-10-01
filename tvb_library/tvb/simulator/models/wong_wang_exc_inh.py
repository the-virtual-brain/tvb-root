# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
#
# This program is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE.  See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this
# program.  If not, see <http://www.gnu.org/licenses/>.
#
#
#   CITATION:
# When using The Virtual Brain for scientific publications, please cite it as explained here:
# https://www.thevirtualbrain.org/tvb/zwei/neuroscience-publications
#
#

"""
Models based on Wong-Wang's work.

First one follows [DPA_2014], with an excitatory and an inhibitory
population, mutually coupled.

Second adds regional heterogeneity (excitation-inhibition balance) as described in [Deco_2021].


.. moduleauthor:: Dionysios Perdikis <dionperd@gmail.com>
.. moduleauthor:: Ignacio Martín <natx.mc@gmail.com>
.. moduleauthor:: Jan Fousek <jan.fousek@univ-amu.fr>
"""

import numpy
from numba import guvectorize, float64, njit, prange
from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.base import ModelNumbaDfun


@guvectorize([(float64[:],) * 22], "(n),(m)" + ",()" * 19 + "->(n)", nopython=True)
def _numba_dfun(
    S, c, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, ji, g, l, io, ie, dx
):
    "Gufunc for reduced Wong-Wang model equations."

    cc = g[0] * jn[0] * c[0]

    jnSe = jn[0] * S[0]

    x = wp[0] * jnSe - ji[0] * S[1] + we[0] * io[0] + cc + ie[0]
    x = ae[0] * x - be[0]
    h = x / (1 - numpy.exp(-de[0] * x))
    dx[0] = -(S[0] / te[0]) + (1.0 - S[0]) * h * ge[0]

    x = jnSe - S[1] + wi[0] * io[0] + l[0] * cc
    x = ai[0] * x - bi[0]
    h = x / (1 - numpy.exp(-di[0] * x))
    dx[1] = -(S[1] / ti[0]) + h * gi[0]


@njit(fastmath=True, cache=True)
def _fast_ww_dfun(x, c, local_coupling,
                  ae, be, de, ge, te, wp, we, jn,
                  ai, bi, di, gi, ti, wi, ji,
                  g, l, io, ie, deriv):
    n_vars, n_nodes, n_modes = x.shape

    for m in range(n_modes):
        for i in range(n_nodes):
            s_e = x[0, i, m]
            s_i = x[1, i, m]
            c_val = c[0, i, m]

            # Support both scalar and node-heterogeneous 1D parameter arrays
            _ae = ae[i] if ae.size > 1 else ae[0]
            _be = be[i] if be.size > 1 else be[0]
            _de = de[i] if de.size > 1 else de[0]
            _ge = ge[i] if ge.size > 1 else ge[0]
            _te = te[i] if te.size > 1 else te[0]
            _wp = wp[i] if wp.size > 1 else wp[0]
            _we = we[i] if we.size > 1 else we[0]
            _jn = jn[i] if jn.size > 1 else jn[0]

            _ai = ai[i] if ai.size > 1 else ai[0]
            _bi = bi[i] if bi.size > 1 else bi[0]
            _di = di[i] if di.size > 1 else di[0]
            _gi = gi[i] if gi.size > 1 else gi[0]
            _ti = ti[i] if ti.size > 1 else ti[0]
            _wi = wi[i] if wi.size > 1 else wi[0]
            _ji = ji[i] if ji.size > 1 else ji[0]

            _g = g[i] if g.size > 1 else g[0]
            _l = l[i] if l.size > 1 else l[0]
            _io = io[i] if io.size > 1 else io[0]
            _ie = ie[i] if ie.size > 1 else ie[0]

            cc = _g * _jn * (c_val + local_coupling * s_e)
            jn_se = _jn * s_e

            # Excitatory population
            x_e = _wp * jn_se - _ji * s_i + _we * _io + cc + _ie
            x_e = _ae * x_e - _be
            h_e = x_e / (1.0 - numpy.exp(-_de * x_e))
            deriv[0, i, m] = - (s_e / _te) + (1.0 - s_e) * h_e * _ge

            # Inhibitory population
            x_i = jn_se - s_i + _wi * _io + _l * cc
            x_i = _ai * x_i - _bi
            h_i = x_i / (1.0 - numpy.exp(-_di * x_i))
            deriv[1, i, m] = - (s_i / _ti) + h_i * _gi

    return deriv

class ReducedWongWangExcInh(ModelNumbaDfun):
    r"""
    .. [WW_2006] Kong-Fatt Wong and Xiao-Jing Wang,  *A Recurrent Network
                Mechanism of Time Integration in Perceptual Decisions*.
                Journal of Neuroscience 26(4), 1314-1328, 2006.

    .. [DPA_2014] Deco Gustavo, Ponce Alvarez Adrian, Patric Hagmann,
                  Gian Luca Romani, Dante Mantini, and Maurizio Corbetta. *How Local
                  Excitation–Inhibition Ratio Impacts the Whole Brain Dynamics*.
                  The Journal of Neuroscience 34(23), 7886 –7898, 2014.


    Equations taken from [DPA_2013]_ , page 11242

    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{a_ex_{ek}- b_e}{1 - \exp(-d_e(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{a_ix_{ik} - b_i}{1 - \exp(-d_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \

    """

    # Define traited attributes for this model, these represent possible kwargs.

    a_e = NArray(
        label=":math:`a_e`",
        default=numpy.array(
            [
                310.0,
            ]
        ),
        domain=Range(lo=0.0, hi=500.0, step=1.0),
        doc="[n/C]. Excitatory population input gain parameter, chosen to fit numerical solutions.",
    )

    b_e = NArray(
        label=":math:`b_e`",
        default=numpy.array(
            [
                125.0,
            ]
        ),
        domain=Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Excitatory population input shift parameter chosen to fit numerical solutions.",
    )

    d_e = NArray(
        label=":math:`d_e`",
        default=numpy.array(
            [
                0.160,
            ]
        ),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Excitatory population input scaling parameter chosen to fit numerical solutions.""",
    )

    gamma_e = NArray(
        label=r":math:`\gamma_e`",
        default=numpy.array(
            [
                0.641 / 1000,
            ]
        ),
        domain=Range(lo=0.0, hi=1.0 / 1000, step=0.01 / 1000),
        doc="""Excitatory population kinetic parameter""",
    )

    tau_e = NArray(
        label=r":math:`\tau_e`",
        default=numpy.array(
            [
                100.0,
            ]
        ),
        domain=Range(lo=50.0, hi=150.0, step=1.0),
        doc="""[ms]. Excitatory population NMDA decay time constant.""",
    )

    w_p = NArray(
        label=r":math:`w_p`",
        default=numpy.array(
            [
                1.4,
            ]
        ),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population recurrence weight""",
    )

    J_N = NArray(
        label=r":math:`J_N`",
        default=numpy.array(
            [
                0.15,
            ]
        ),
        domain=Range(lo=0.001, hi=0.5, step=0.001),
        doc="""[nA] NMDA current""",
    )

    W_e = NArray(
        label=r":math:`W_e`",
        default=numpy.array(
            [
                1.0,
            ]
        ),
        domain=Range(lo=0.0, hi=2.0, step=0.01),
        doc="""Excitatory population external input scaling weight""",
    )

    a_i = NArray(
        label=":math:`a_i`",
        default=numpy.array(
            [
                615.0,
            ]
        ),
        domain=Range(lo=0.0, hi=1000.0, step=1.0),
        doc="[n/C]. Inhibitory population input gain parameter, chosen to fit numerical solutions.",
    )

    b_i = NArray(
        label=":math:`b_i`",
        default=numpy.array(
            [
                177.0,
            ]
        ),
        domain=Range(lo=0.0, hi=200.0, step=1.0),
        doc="[Hz]. Inhibitory population input shift parameter chosen to fit numerical solutions.",
    )

    d_i = NArray(
        label=":math:`d_i`",
        default=numpy.array(
            [
                0.087,
            ]
        ),
        domain=Range(lo=0.0, hi=0.2, step=0.001),
        doc="""[s]. Inhibitory population input scaling parameter chosen to fit numerical solutions.""",
    )

    gamma_i = NArray(
        label=r":math:`\gamma_i`",
        default=numpy.array(
            [
                1.0 / 1000,
            ]
        ),
        domain=Range(lo=0.0, hi=2.0 / 1000, step=0.01 / 1000),
        doc="""Inhibitory population kinetic parameter""",
    )

    tau_i = NArray(
        label=r":math:`\tau_i`",
        default=numpy.array(
            [
                10.0,
            ]
        ),
        domain=Range(lo=5.0, hi=100.0, step=1.0),
        doc="""[ms]. Inhibitory population NMDA decay time constant.""",
    )

    J_i = NArray(
        label=r":math:`J_{i}`",
        default=numpy.array(
            [
                1.0,
            ]
        ),
        domain=Range(lo=0.001, hi=2.0, step=0.001),
        doc="""[nA] Local inhibitory current""",
    )

    W_i = NArray(
        label=r":math:`W_i`",
        default=numpy.array(
            [
                0.7,
            ]
        ),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory population external input scaling weight""",
    )

    I_o = NArray(
        label=":math:`I_{o}`",
        default=numpy.array(
            [
                0.382,
            ]
        ),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external input""",
    )

    I_ext = NArray(
        label=":math:`I_{ext}`",
        default=numpy.array(
            [
                0.0,
            ]
        ),
        domain=Range(lo=0.0, hi=1.0, step=0.001),
        doc="""[nA]. Effective external stimulus input""",
    )

    G = NArray(
        label=":math:`G`",
        default=numpy.array(
            [
                2.0,
            ]
        ),
        domain=Range(lo=0.0, hi=10.0, step=0.01),
        doc="""Global coupling scaling""",
    )

    lamda = NArray(
        label=r":math:`\lambda`",
        default=numpy.array(
            [
                0.0,
            ]
        ),
        domain=Range(lo=0.0, hi=1.0, step=0.01),
        doc="""Inhibitory global coupling scaling""",
    )

    coupling_terms = Final(label="Coupling terms", default=["Coupling_Term"])

    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default="a_e b_e d_e gamma_e tau_e w_p W_e J_N a_i b_i d_i gamma_i tau_i J_i W_i I_o I_ext G lamda".split(),
    )

    dfun_intermediates = Final(
        label="Shared intermediates for dfun code generation",
        default=[
            ("cc", "G * J_N * Coupling_Term"),
            ("jnSe", "J_N * S_e"),
            ("x_e", "a_e * (w_p * jnSe - J_i * S_i + W_e * I_o + cc + I_ext) - b_e"),
            ("H_e", "x_e / (1.0 - math.exp(-d_e * x_e))"),
            ("x_i", "a_i * (jnSe - S_i + W_i * I_o + lamda * cc) - b_i"),
            ("H_i", "x_i / (1.0 - math.exp(-d_i * x_i))"),
        ],
    )

    state_variable_dfuns = Final(
        label="Drift functions for numba codegen",
        default={
            "S_e": "-(S_e / tau_e) + (1.0 - S_e) * H_e * gamma_e",
            "S_i": "-(S_i / tau_i) + H_i * gamma_i",
        },
    )

    state_variable_range = Final(
        default={"S_e": numpy.array([0.0, 1.0]), "S_i": numpy.array([0.0, 1.0])},
        label="State variable ranges [lo, hi]",
        doc="Population firing rate",
    )

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"S_e": numpy.array([0.0, 1.0]), "S_i": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
            the boundaries of the dynamic range of that state-variable. Set None for one-sided boundaries""",
    )

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("S_e", "S_i"),
        default=("S_e", "S_i"),
        doc="""default state variables to be monitored""",
    )

    state_variables = ["S_e", "S_i"]
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)


    def configure(self):
        """ """
        super(ReducedWongWangExcInh, self).configure()
        self.update_derived_parameters()


    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        S = state_variables[:, :]

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S[0]

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S[0]

        x_e = (
            self.w_p * J_N_S_e
            - self.J_i * S[1]
            + self.W_e * self.I_o
            + coupling
            + self.I_ext
        )

        x_e = self.a_e * x_e - self.b_e
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = -(S[0] / self.tau_e) + (1 - S[0]) * H_e * self.gamma_e

        x_i = J_N_S_e - S[1] + self.W_i * self.I_o + self.lamda * coupling

        x_i = self.a_i * x_i - self.b_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = -(S[1] / self.tau_i) + H_i * self.gamma_i

        derivative = numpy.array([dS_e, dS_i])

        return derivative



    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        # Ensure contiguous memory layout and allocate a clean derivative array per step
        x_arr = numpy.ascontiguousarray(x, dtype=numpy.float64)
        c_arr = numpy.ascontiguousarray(c, dtype=numpy.float64)
        derivative = numpy.empty_like(x_arr)

        return _fast_ww_dfun(
            x_arr, c_arr, float(local_coupling),
            numpy.ascontiguousarray(self.a_e, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.b_e, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.d_e, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.gamma_e, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.tau_e, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.w_p, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.W_e, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.J_N, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.a_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.b_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.d_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.gamma_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.tau_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.W_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.J_i, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.G, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.lamda, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.I_o, dtype=numpy.float64).ravel(),
            numpy.ascontiguousarray(self.I_ext, dtype=numpy.float64).ravel(),
            derivative
        )
@guvectorize([(float64[:],)*23], '(n),(m)' + ',()'*20 + '->(n)', nopython=True)
def _numba_dfun_bei(S, c, mi, ae, be, de, ge, te, wp, we, jn, ai, bi, di, gi, ti, wi, ji, g, l, io, ie, dx):
    """Gufunc for transcriptional model presented in Deco et Al 2020, Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape"""

    cc = g[0] * jn[0] * c[0]

    jnSe = jn[0] * S[0]

    x = wp[0] * jnSe - ji[0] * S[1] + we[0] * io[0] + cc + ie[0]
    x = (ae[0] * x - be[0]) * mi[0]
    h = x / (1 - numpy.exp(-de[0] * x))
    dx[0] = -(S[0] / te[0]) + (1.0 - S[0]) * h * ge[0]

    x = jnSe - S[1] + wi[0] * io[0] + l[0] * cc
    x = (ai[0] * x - bi[0]) * mi[0]
    h = x / (1 - numpy.exp(-di[0] * x))
    dx[1] = -(S[1] / ti[0]) + h * gi[0]


class DecoBalancedExcInh(ReducedWongWangExcInh):
    r"""
    .. [Deco_2021] Deco, Gustavo, Morten L. Kringelbach, Aurina Arnatkeviciute,
    Stuart Oldham, Kristina Sabaroedin, Nigel C. Rogasch, Kevin M. Aquino, and
    Alex Fornito. "Dynamical consequences of regional heterogeneity in the
    brain’s transcriptional landscape." Science Advances 7, no. 29 (2021):
    eabf4752.

    Equations extend the [DPA_2013] with effective gain parameter M_i to


    .. math::
                 x_{ek}       &=   w_p\,J_N \, S_{ek} - J_iS_{ik} + W_eI_o + GJ_N \mathbf\Gamma(S_{ek}, S_{ej}, u_{kj}) \\
                 H(x_{ek})    &=  \dfrac{M_i(a_ex_{ek}- b_e)}{1 - \exp(-d_e M_i(a_ex_{ek} -b_e))} \\
                 \dot{S}_{ek} &= -\dfrac{S_{ek}}{\tau_e} + (1 - S_{ek}) \, {\gamma}H(x_{ek}) \\

                 x_{ik}       &=   J_N \, S_{ek} - S_{ik} + W_iI_o + {\lambda}GJ_N \mathbf\Gamma(S_{ik}, S_{ej}, u_{kj}) \\
                 H(x_{ik})    &=  \dfrac{M_i(a_ix_{ik} - b_i)}{1 - \exp(-d_i M_i(a_ix_{ik} -b_i))} \\
                 \dot{S}_{ik} &= -\dfrac{S_{ik}}{\tau_i} + \gamma_iH(x_{ik}) \
    """

    # Define traited attributes for this model, these represent possible kwargs.

    M_i = NArray(
        label=":math:`ratio`",
        default=numpy.array(
            [
                1.0,
            ]
        ),
        domain=Range(lo=1.0, hi=10.0, step=0.01),
        doc="""Effective gain within a region.""",
    )

    coupling_terms = Final(label="Coupling terms", default=["Coupling_Term"])

    # Override parameter_names to include M_i from this subclass
    parameter_names = List(
        of=str,
        label="List of parameters for this model",
        default="a_e b_e d_e gamma_e tau_e w_p W_e J_N a_i b_i d_i gamma_i tau_i J_i W_i I_o I_ext G lamda M_i".split(),
    )

    # Override dfun_intermediates to include M_i scaling
    dfun_intermediates = Final(
        label="Shared intermediates for dfun code generation",
        default=[
            ("cc", "G * J_N * Coupling_Term"),
            ("jnSe", "J_N * S_e"),
            ("x_e", "M_i * (a_e * (w_p * jnSe - J_i * S_i + W_e * I_o + cc + I_ext) - b_e)"),
            ("H_e", "x_e / (1.0 - math.exp(-d_e * x_e))"),
            ("x_i", "M_i * (a_i * (jnSe - S_i + W_i * I_o + lamda * cc) - b_i)"),
            ("H_i", "x_i / (1.0 - math.exp(-d_i * x_i))"),
        ],
    )

    # Override state_variable_dfuns with M_i-scaled transfer functions
    state_variable_dfuns = Final(
        label="Drift functions for numba codegen",
        default={
            "S_e": "-(S_e / tau_e) + (1.0 - S_e) * H_e * gamma_e",
            "S_i": "-(S_i / tau_i) + H_i * gamma_i",
        },
    )

    def _numpy_dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        Numpy dfun for transcriptional model presented in [Deco_2020],
        Dynamical consequences of regional heterogeneity in the brain’s
        transcriptional landscape
        """

        S = state_variables[:, :]

        S_e = S[0, :]
        S_i = S[1, :]

        c_0 = coupling[0, :]

        # if applicable
        lc_0 = local_coupling * S_e

        coupling = self.G * self.J_N * (c_0 + lc_0)

        J_N_S_e = self.J_N * S_e

        inh = self.J_i * S_i

        I_e = self.W_e * self.I_o + self.w_p * J_N_S_e + coupling - inh + self.I_ext

        x_e = (self.a_e * I_e - self.b_e) * self.M_i
        H_e = x_e / (1 - numpy.exp(-self.d_e * x_e))

        dS_e = -(S_e / self.tau_e) + (1.0 - S_e) * H_e * self.gamma_e

        I_i = self.W_i * self.I_o + J_N_S_e - S_i + self.lamda * coupling

        x_i = (self.a_i * I_i - self.b_i) * self.M_i
        H_i = x_i / (1 - numpy.exp(-self.d_i * x_i))

        dS_i = -(S_i / self.tau_i) + H_i * self.gamma_i

        derivative = numpy.array([dS_e, dS_i])

        return derivative

    def dfun(self, x, c, local_coupling=0.0, **kwargs):
        r"""
        Numpy dfun for transcriptional model presented in [Deco_2020],
        Dynamical consequences of regional heterogeneity in the brain’s
        transcriptional landscape
        """
        x_ = x.reshape(x.shape[:-1]).T
        c_ = c.reshape(c.shape[:-1]).T + local_coupling * x[0]
        deriv = _numba_dfun_bei(
            x_,
            c_,
            self.M_i,
            self.a_e,
            self.b_e,
            self.d_e,
            self.gamma_e,
            self.tau_e,
            self.w_p,
            self.W_e,
            self.J_N,
            self.a_i,
            self.b_i,
            self.d_i,
            self.gamma_i,
            self.tau_i,
            self.W_i,
            self.J_i,
            self.G,
            self.lamda,
            self.I_o,
            self.I_ext,
        )
        return deriv.T[..., numpy.newaxis]
        # Numpy version, for debugging purposes
        # return self._numpy_dfun(x, c, local_coupling)
