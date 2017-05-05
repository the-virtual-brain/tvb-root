# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2017, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Reloadable module of functions that implements burst API.

This is separated in order to speed development 

"""

import json 

import tvb.interfaces.web.controllers.base_controller as base

"""
model_parameters_option_Generic2dOscillator_state_variable_range_parameters_W
integrator
surface
integrator_parameters_option_HeunDeterministic_dt
simulation_length
monitors_parameters_option_TemporalAverage_period
monitors
conduction_speed
model_parameters_option_Generic2dOscillator_a
model_parameters_option_Generic2dOscillator_b
model_parameters_option_Generic2dOscillator_c
model_parameters_option_Generic2dOscillator_tau
model_parameters_option_Generic2dOscillator_noise
model_parameters_option_Generic2dOscillator_noise_parameters_option_Noise_random_stream_parameters_option_RandomStream_init_seed
connectivity
model_parameters_option_Generic2dOscillator_noise_parameters_option_Noise_random_stream
range_1
range_2
coupling_parameters_option_Linear_b
model_parameters_option_Generic2dOscillator_noise_parameters_option_Noise_ntau
coupling_parameters_option_Linear_a
model_parameters_option_Generic2dOscillator_state_variable_range_parameters_V
coupling
model_parameters_option_Generic2dOscillator_I
stimulus
currentAlgoId
model_parameters_option_Generic2dOscillator_variables_of_interest
model
"""


class HierStruct(object):
    """
    Class to handle by flat and nested indexing of simulator configuration


    """

    pass


def index(self):
    return 'Burst API'


def read(self, pid):
    pid = int(pid)
    info = {}
    bursts = self.burst_service.get_available_bursts(pid)
    import pdb; pdb.set_trace()
    for burst in bursts:
        info[burst.name] = {k: v for k, v in burst.simulator_configuration.iteritems() if len(k) > 0}
    return json.dumps(info)


def create(self, opt):
    # NotImplemented
    return 'NotImplemented'

