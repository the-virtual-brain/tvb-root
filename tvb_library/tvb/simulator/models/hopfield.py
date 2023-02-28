# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Scientific Package. This package holds all simulators, and
# analysers necessary to run brain-simulations. You can use it stand alone or
# in conjunction with TheVirtualBrain-Framework Package. See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
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
#
#

"""
Hopfield model with modifications following Golos & Daucé.

"""
import numpy
from .base import Model
from tvb.basic.neotraits.api import NArray, List, Range, Final


class Hopfield(Model):
    r"""

    The Hopfield neural network is a discrete time dynamical system composed
    of multiple binary nodes, with a connectivity matrix built from a
    predetermined set of patterns. The update, inspired from the spin-glass
    model (used to describe magnetic properties of dilute alloys), is based on
    a random scanning of every node. The existence of a fixed point dynamics
    is guaranteed by a Lyapunov function. The Hopfield network is expected to
    have those multiple patterns as attractors (multistable dynamical system).
    When the initial conditions are close to one of the 'learned' patterns,
    the dynamical system is expected to relax on the corresponding attractor.
    A possible output of the system is the final attractive state (interpreted
    as an associative memory).

    Various extensions of the initial model have been proposed, among which a
    noiseless and continuous version [Hopfield 1984] having a slightly
    different Lyapunov function, but essentially the same dynamical
    properties, with more straightforward physiological interpretation. A
    continuous Hopfield neural network (with a sigmoid transfer function) can
    indeed be interpreted as a network of neural masses with every node
    corresponding to the mean field activity of a local brain region, with
    many bridges with the Wilson Cowan model [WC_1972].

    **References**:

        .. [Hopfield1982] Hopfield, J. J., *Neural networks and physical systems with emergent collective
                        computational abilities*, Proc. Nat. Acad. Sci. (USA) 79, 2554-2558, 1982.

        .. [Hopfield1984] Hopfield, J. J., *Neurons with graded response have collective computational
                        properties like those of two-sate neurons*, Proc. Nat. Acad. Sci. (USA) 81, 3088-3092, 1984.

    See also, http://www.scholarpedia.org/article/Hopfield_network

    .. #This model can use a global threshold permitting multistable dynamic for
    .. #a positive structural connectivity matrix.

    .. automethod:: Hopfield.configure

    Dynamic equations:

    dfun equation
        .. math::
                \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0)
    dfun dynamic equation
        .. math::
            \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0(i)) \\
            \dot{\theta_{i}} &= 1 / \tau_{\theta_{i}} (-\theta + c_1(i))


    .. figure :: img/Hopfield_01_mode_0_pplane.svg

    The phase-plane for the Hopfield model.

    """

    # Define traited attributes for this model, these represent possible kwargs.
    taux = NArray(
        label=r":math:`\tau_{x}`",
        default=numpy.array([1.]),
        domain=Range(lo=0.01, hi=100., step=0.01),
        doc="""The fast time-scale for potential calculus :math:`x`, state-variable of the model.""")

    tauT = NArray(
        label=r":math:`\tau_{\theta}`",
        default=numpy.array([5.]),
        domain=Range(lo = 0.01, hi = 100., step = 0.01),
        doc="""The slow time-scale for threshold calculus :math:`\\theta`, state-variable of the model.""")

    dynamic = NArray(
        dtype=int,
        label="Dynamic",
        default=numpy.array([0]),
        domain=Range(lo=0, hi=1., step=1),
        doc="""Boolean value for static/dynamic threshold theta for (0/1).""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"x": numpy.array([-1., 2.]), "theta": numpy.array([0., 1.])},
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random inital
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("x", "theta"),
        default=("x",),
        doc="""The values for each state-variable should be set to encompass
            the expected dynamic range of that state-variable for the current
            parameters, it is used as a mechanism for bounding random initial
            conditions when the simulation isn't started from an explicit
            history, it is also provides the default range of phase-plane plots.""")

    state_variables = ('x', 'theta')

    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def configure(self):
        """Set the threshold as a state variable for a dynamical threshold."""
        super(Hopfield, self).configure()
        if self.dynamic:
            self.dfun = self.dfunDyn
            self._nvar = 2
            self.cvar = numpy.array([0, 1], dtype=numpy.int32)
            # self.variables_of_interest = ["x", "theta"]

    def dfun(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The fast, :math:`x`, and slow, :math:`\theta`, state variables are typically
        considered to represent a membrane potentials of nodes and the global inhibition term,
        respectively:

            .. math::

                \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0) \\
        """
        x = state_variables[0, :]
        dx = (- x + coupling[0]) / self.taux

        # todo: display dependent hack. It returns dx twice to be compatible with dfunDyn
        # We return 2 arrays here, because we have 2 possible state Variable, even if not dynamic
        # Otherwise the phase-plane display will fail.
        derivative = numpy.array([dx, dx])
        return derivative

    def dfunDyn(self, state_variables, coupling, local_coupling=0.0):
        r"""
        The fast, :math:`x`, and slow, :math:`\theta`, state variables are typically
        considered to represent a membrane potentials of nodes and the inhibition term(s),
        respectively:

            .. math::
                \dot{x_{i}} &= 1 / \tau_{x} (-x_{i} + c_0(i)) \\
                \dot{\theta_{i}} &= 1 / \tau_{\theta_{i}} (-\theta + c_1(i))

        where c_0 is the coupling term and c_1 should be the direct output.

        """

        x = state_variables[0, :]
        theta = state_variables[1, :]
        dx = (- x + coupling[0]) / self.taux
        dtheta = (- theta + coupling[1]) / self.tauT

        derivative = numpy.array([dx, dtheta])
        return derivative
