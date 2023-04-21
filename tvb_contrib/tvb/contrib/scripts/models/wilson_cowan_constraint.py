# -*- coding: utf-8 -*-
#
#
#  TheVirtualBrain-Contributors Package. This package holds simulator extensions.
#  See also http://www.thevirtualbrain.org
#
# (c) 2012-2023, Baycrest Centre for Geriatric Care ("Baycrest") and others
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

"""
Wilson-Cowan equations based model definition.
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>
"""
import numpy
from tvb.basic.neotraits.api import Final, NArray, Range, List
from tvb.simulator.models.wilson_cowan import WilsonCowan as TVBWilsonCowan


class WilsonCowan(TVBWilsonCowan):

    r"""
    **References**:

    .. [WC_1972] Wilson, H.R. and Cowan, J.D. *Excitatory and inhibitory
        interactions in localized populations of model neurons*, Biophysical
        journal, 12: 1-24, 1972.
    .. [WC_1973] Wilson, H.R. and Cowan, J.D  *A Mathematical Theory of the
        Functional Dynamics of Cortical and Thalamic Nervous Tissue*

    .. [D_2011] Daffertshofer, A. and van Wijk, B. *On the influence of
        amplitude on the connectivity between phases*
        Frontiers in Neuroinformatics, July, 2011

    Used Eqns 11 and 12 from [WC_1972]_ in ``dfun``.  P and Q represent external
    inputs, which when exploring the phase portrait of the local model are set
    to constant values. However in the case of a full network, P and Q are the
    entry point to our long range and local couplings, that is, the  activity
    from all other nodes is the external input to the local population.

    The default parameters are taken from figure 4 of [WC_1972]_, pag. 10

    In [WC_1973]_ they present a model of neural tissue on the pial surface is.
    See Fig. 1 in page 58. The following local couplings (lateral interactions)
    occur given a region i and a region j:

      E_i-> E_j
      E_i-> I_j
      I_i-> I_j
      I_i-> E_j


    +---------------------------+
    |          Table 1          |
    +--------------+------------+
    |                           |
    |  SanzLeonetAl,   2014     |
    +--------------+------------+
    |Parameter     |  Value     |
    +==============+============+
    | k_e, k_i     |    1.00    |
    +--------------+------------+
    | r_e, r_i     |    0.00    |
    +--------------+------------+
    | tau_e, tau_i |    10.0    |
    +--------------+------------+
    | c_1          |    10.0    |
    +--------------+------------+
    | c_2          |    6.0     |
    +--------------+------------+
    | c_3          |    1.0     |
    +--------------+------------+
    | c_4          |    1.0     |
    +--------------+------------+
    | a_e, a_i     |    1.0     |
    +--------------+------------+
    | b_e, b_i     |    0.0     |
    +--------------+------------+
    | theta_e      |    2.0     |
    +--------------+------------+
    | theta_i      |    3.5     |
    +--------------+------------+
    | alpha_e      |    1.2     |
    +--------------+------------+
    | alpha_i      |    2.0     |
    +--------------+------------+
    | P            |    0.5     |
    +--------------+------------+
    | Q            |    0       |
    +--------------+------------+
    | c_e, c_i     |    1.0     |
    +--------------+------------+
    | alpha_e      |    1.2     |
    +--------------+------------+
    | alpha_i      |    2.0     |
    +--------------+------------+
    |                           |
    |  frequency peak at 20  Hz |
    |                           |
    +---------------------------+


    The parameters in Table 1 reproduce Figure A1 in  [D_2011]_
    but set the limit cycle frequency to a sensible value (eg, 20Hz).

    Model bifurcation parameters:
        * :math:`c_1`
        * :math:`P`



    The builders (:math:`E`, :math:`I`) phase-plane, including a representation of
    the vector field as well as its nullclines, using default parameters, can be
    seen below:

        .. _phase-plane-WC:
        .. figure :: img/WilsonCowan_01_mode_0_pplane.svg
            :alt: Wilson-Cowan phase plane (E, I)

            The (:math:`E`, :math:`I`) phase-plane for the Wilson-Cowan model.


    The general formulation for the \textit{\textbf{Wilson-Cowan}} model as a
    dynamical unit at a node $k$ in a BNM with $l$ nodes reads:

    .. math::
            \dot{E}_k &= \dfrac{1}{\tau_e} (-E_k  + (k_e - r_e E_k) \mathcal{S}_e (\alpha_e \left( c_{ee} E_k - c_{ei} I_k  + P_k - \theta_e + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) ))\\
            \dot{I}_k &= \dfrac{1}{\tau_i} (-I_k  + (k_i - r_i I_k) \mathcal{S}_i (\alpha_i \left( c_{ie} E_k - c_{ee} I_k  + Q_k - \theta_i + \mathbf{\Gamma}(E_k, E_j, u_{kj}) + W_{\zeta}\cdot E_j + W_{\zeta}\cdot I_j\right) )),

    """

    # Define traited attributes for this model, these represent possible kwargs.

    tau_Ein = NArray(
        label=r":math:`\tau_Ein`",
        default=numpy.array([50., ]),
        domain=Range(lo=1., hi=100., step=1.0),
        doc="""[ms]. Excitatory population instant spiking rate time constant.""")

    tau_Iin = NArray(
        label=r":math:`\tau_Iin`",
        default=numpy.array([50., ]),
        domain=Range(lo=1., hi=100., step=1.0),
        doc="""[ms]. Inhibitory population instant spiking rate time constant.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        label="State Variable boundaries [lo, hi]",
        default={"E": numpy.array([0.0, 1.0]),
                 "I": numpy.array([0.0, 1.0]),
                 "Ein": numpy.array([0.0, 1.0]),
                 "Iin": numpy.array([0.0, 1.0])},
        doc="""The values for each state-variable should be set to encompass
                the boundaries of the dynamic range of that state-variable. 
                Set None for one-sided boundaries.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"E": numpy.array([0.0, 0.5]),
                 "I": numpy.array([0.0, 0.5]),
                 "Ein": numpy.array([0.0, 0.5]),
                 "Iin": numpy.array([0.0, 0.5])},
        doc="""The values for each state-variable should be set to encompass
        the expected dynamic range of that state-variable for the current
        parameters, it is used as a mechanism for bounding random inital
        conditions when the simulation isn't started from an explicit history,
        it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=('E', 'I', 'Ein', 'Iin'),
        default=('E', 'I', 'Ein', 'Iin'),
        doc="""default state variables to be monitored""")

    state_variables = ['E', 'I', 'Ein', 'Iin']
    _nvar = 4

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        for var in ["Ein", "Iin"]:
            if hasattr(self, var):
                setattr(self, "_" + var, getattr(self, var) > 0)
            else:
                setattr(self, "_" + var, numpy.array([False, ]))

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""

        .. math::
            \tau \dot{x}(t) &= -z(t) + \phi(z(t)) \\
            \phi(x) &= \frac{c}{1-exp(-a (x-b))}

        """

        E = state_variables[0, :]
        I = state_variables[1, :]
        Ein = state_variables[2, :]  # Input from Spiking Network
        Iin = state_variables[3, :]  # Input from Spiking Network
        derivative = numpy.zeros(state_variables.shape)

        # long-range coupling
        c_0 = coupling[0, :]

        # short-range (local) coupling
        lc_0 = local_coupling * E
        lc_1 = local_coupling * I

        x_e = self.alpha_e * (self.c_ee * E - self.c_ei * I + self.P  - self.theta_e +  c_0 + lc_0 + lc_1)
        x_i = self.alpha_i * (self.c_ie * E - self.c_ii * I + self.Q  - self.theta_i + lc_0 + lc_1)

        s_e = self.c_e / (1.0 + numpy.exp(-self.a_e * (x_e - self.b_e)))
        s_i = self.c_i / (1.0 + numpy.exp(-self.a_i * (x_i - self.b_i)))

        derivative[0] = numpy.where(self._Ein,
                                    (-E + Ein) / self.tau_Ein,                            # Update from Spiking Network
                                    (-E + (self.k_e - self.r_e * E) * s_e) / self.tau_e)
        derivative[1] = numpy.where(self._Iin,
                                    (-I + Iin) / self.tau_Iin,                            # Update from Spiking Network
                                    (-I + (self.k_i - self.r_i * I) * s_i) / self.tau_i)

        return derivative
