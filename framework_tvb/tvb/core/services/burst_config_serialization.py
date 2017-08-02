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
# CITATION:
# When using The Virtual Brain for scientific publications, please cite it as follows:
#
#   Paula Sanz Leon, Stuart A. Knock, M. Marmaduke Woodman, Lia Domide,
#   Jochen Mersmann, Anthony R. McIntosh, Viktor Jirsa (2013)
#       The Virtual Brain: a simulator of primate brain network dynamics.
#   Frontiers in Neuroinformatics (7:10. doi: 10.3389/fninf.2013.00010)
#
#

"""
Service for serianlizing a Burst (Simulator) configuration.

.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""

import six
from tvb.basic.logger.builder import get_logger
from tvb.basic.traits.parameters_factory import get_traited_instance_for_name
from tvb.config import SIMULATOR_MODULE, SIMULATOR_CLASS
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model import RANGE_PARAMETER_1, RANGE_PARAMETER_2, PARAMS_MODEL_PATTERN
from tvb.core.entities.model import PARAM_MODEL, PARAM_INTEGRATOR, PARAM_CONNECTIVITY, PARAM_SURFACE
from tvb.core.services.flow_service import FlowService
from tvb.datatypes import noise_framework
from tvb.simulator.integrators import Integrator
from tvb.simulator.models import Model

MODEL_PARAMETERS = 'model_parameters'
INTEGRATOR_PARAMETERS = 'integrator_parameters'


class SerializationManager(object):
    """
    Constructs data types based on a burst configuration.
    Updates the burst configuration.
    """

    def __init__(self, conf):
        """
        :param conf: burst configuration entity
        """
        self.logger = get_logger(__name__)
        self.flow_service = FlowService()
        self.conf = conf


    def _build_simulator_adapter(self):
        stored_adapter = self.flow_service.get_algorithm_by_module_and_class(SIMULATOR_MODULE, SIMULATOR_CLASS)
        return ABCAdapter.build_adapter(stored_adapter)


    def has_model_pse_ranges(self):
        """ Returns True if the burst configuration describes a range on a model parameter """
        first_range = self.conf.get_simulation_parameter_value(RANGE_PARAMETER_1)
        second_range = self.conf.get_simulation_parameter_value(RANGE_PARAMETER_2)
        first_range_on = first_range is not None and str(first_range).startswith(MODEL_PARAMETERS)
        second_range_on = second_range is not None and str(second_range).startswith(MODEL_PARAMETERS)
        return first_range_on or second_range_on


    def _get_params_dict(self):
        """ Convert ui inputs from the configuration to python types """
        simulator_adapter = self._build_simulator_adapter()
        return simulator_adapter.convert_ui_inputs(self.conf.get_all_simulator_values()[0], False)


    def __make_instance_from_burst_config(self, params_dict, parent_class, class_name_key, params_key):
        """ This is used internally to create a model or an integrator based on the burst config """
        class_name = self.conf.get_simulation_parameter_value(class_name_key)
        parameters = params_dict[params_key]
        noise_framework.build_noise(parameters)
        try:
            return get_traited_instance_for_name(class_name, parent_class, parameters)
        except Exception:
            self.logger.exception("Could not create an instance of %s with the given parameters. "
                                  "A new instance will be created with the default values." % class_name)
            return get_traited_instance_for_name(class_name, parent_class, {})


    def __make_shallow_model(self):
        """ Creates a model of the type present in the config without setting any parameters on it """
        class_name = self.conf.get_simulation_parameter_value(PARAM_MODEL)
        return get_traited_instance_for_name(class_name, Model, {})


    def make_model_and_integrator(self):
        """
        :return: A model and an integrator.
        :rtype: Model, Integrator
        """
        params_dict = self._get_params_dict()
        model = self.__make_instance_from_burst_config(params_dict, Model, PARAM_MODEL, MODEL_PARAMETERS)
        integrator = self.__make_instance_from_burst_config(params_dict, Integrator,
                                                            PARAM_INTEGRATOR, INTEGRATOR_PARAMETERS)
        return model, integrator


    def get_connectivity(self):
        """ Prepare Connectivity """
        connectivity_gid = self.conf.get_simulation_parameter_value(PARAM_CONNECTIVITY)
        return ABCAdapter.load_entity_by_gid(connectivity_gid)


    def get_surface(self):
        """ Prepare Surface """
        surface_gid = self.conf.get_simulation_parameter_value(PARAM_SURFACE)
        if surface_gid:
            return ABCAdapter.load_entity_by_gid(surface_gid)
        return None

    @staticmethod
    def group_parameter_values_by_name(model_parameters_list):
        """
        @:param model_parameters_list: Given a list of model parameters like this:
                [{"a": 2.0, 'b': 1.0},
                 {"a": 3.0, 'b': 7.0}])

        @:return: This method will group them by param name to get:
                {'a': [2.0, 3.0], 'b': [1.0, 7.0]}
        """
        ret = {}
        for model_parameters in model_parameters_list:
            for param_name, param_val in six.iteritems(model_parameters):
                if param_name not in ret:
                    ret[param_name] = []
                ret[param_name].append(param_val)
        return ret


    def write_model_parameters(self, model_name, model_parameters_list):
        """
        Update model parameters in burst config.

        :param model_name: This model will be selected in burst
        :param model_parameters_list: A list of model parameter configurations. One for each connectivity node.
                Ex. [{'a': 1, 'b': 2}, ...]
        """


        def format_param_vals(vals):
            # contract constant array
            if len(set(vals)) == 1:
                vals = [vals[0]]
            return str(vals)

        model_parameters = self.group_parameter_values_by_name(model_parameters_list)
        # change selected model in burst config
        self.conf.update_simulation_parameter(PARAM_MODEL, model_name)

        for param_name, param_vals in six.iteritems(model_parameters):
            full_name = PARAMS_MODEL_PATTERN % (model_name, param_name)
            self.conf.update_simulation_parameter(full_name, format_param_vals(param_vals))


    def write_noise_parameters(self, noise_dispersions):
        """
        Set noise dispersions in burst config.
        It will set all nsig fields it can find in the config (at least 1 per stochastic integrator).
        :param noise_dispersions: A list of noise dispersions. One for each connectivity node. Ex [{'V': 1, 'W':2}, ...]
        """
        noise_dispersions = self.group_parameter_values_by_name(noise_dispersions)
        # Flatten the dict to an array of shape (state_vars, nodes)
        state_vars = self.__make_shallow_model().state_variables
        noise_arr = [noise_dispersions[sv] for sv in state_vars]

        simulator_adapter = self._build_simulator_adapter()
        for param_name in simulator_adapter.noise_configurable_parameters():
            self.conf.update_simulation_parameter(param_name, str(noise_arr))
