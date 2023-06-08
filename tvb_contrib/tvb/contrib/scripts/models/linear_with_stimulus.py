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
Generic linear model.
.. moduleauthor:: Dionysios Perdikis <dionysios.perdikis@charite.de>
"""

import numpy
from tvb.simulator.models.base import Model
from tvb.basic.neotraits.api import NArray, Final, List, Range
from tvb.simulator.models.linear import Linear

class Linear(Linear):
    I_o = NArray(
        label=r":math:`I_o`",
        default=numpy.array([0.0]),
        domain=Range(lo=-100.0, hi=100.0, step=1.0),
        doc="External stimulus")

    G = NArray(
        label=r":math:`G`",
        default=numpy.array([0.0]),
        domain=Range(lo=-0.0, hi=100.0, step=1.0),
        doc="Global coupling scaling")

    tau = NArray(
        label=r":math:`\tau`",
        default=numpy.array([1.0]),
        domain=Range(lo=-0.1, hi=100.0, step=0.1),
        doc="Time constant")

    tau_rin = NArray(
        label=r":math:`\tau_rin_e`",
        default=numpy.array([10., ]),
        domain=Range(lo=1., hi=100., step=1.0),
        doc="""[ms]. Excitatory population instant spiking rate time constant.""")

    # Used for phase-plane axis ranges and to bound random initial() conditions.
    state_variable_boundaries = Final(
        default={"R": numpy.array([0.0, None]),
                 "Rin": numpy.array([0.0, None])},
        label="State Variable boundaries [lo, hi]",
        doc="""The values for each state-variable should be set to encompass
                the boundaries of the dynamic range of that state-variable. 
                Set None for one-sided boundaries""")

    state_variable_range = Final(
        label="State Variable ranges [lo, hi]",
        default={"R": numpy.array([0, 100]),
                 "Rin": numpy.array([0, 100])},
        doc="Range used for state variable initialization and visualization.")

    variables_of_interest = List(
        of=str,
        label="Variables watched by Monitors",
        choices=("R", 'Rin'),
        default=("R", 'Rin'), )

    state_variables = ('R', 'Rin')
    integration_variables = ('R',)
    _nvar = 2
    cvar = numpy.array([0], dtype=numpy.int32)

    def update_derived_parameters(self):
        """
        When needed, this should be a method for calculating parameters that are
        calculated based on paramaters directly set by the caller. For example,
        see, ReducedSetFitzHughNagumo. When not needed, this pass simplifies
        code that updates an arbitrary models parameters -- ie, this can be
        safely called on any model, whether it's used or not.
        """
        if hasattr(self, "Rin"):
            setattr(self, "_Rin", getattr(self, "Rin") > 0)
        else:
            setattr(self, "Rin", numpy.array([0.0, ]))
            setattr(self, "_Rin", numpy.array([False, ]))

    def update_non_state_variables_after_integration(self, state_variables):
        # Reset to 0 the Rin for nodes not updated by Spiking Network
        state_variables[1] = numpy.where(self._Rin, state_variables[1], 0.0)
        return state_variables

    def dfun(self, state, coupling, local_coupling=0.0):
        """
        .. math::
            dR/dt = (-R + G * coupling) / {\tau} + I_o
        """
        dR = numpy.where(self._Rin,
                         (- state[0] + state[1]) / self.tau_rin,
                         (-state[0] + self.G * coupling[0] + local_coupling * state[0] ) / self.tau + self.I_o)
        return numpy.array([dR, 0.0*dR])
