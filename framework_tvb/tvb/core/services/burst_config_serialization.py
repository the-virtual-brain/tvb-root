# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2020, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
import numpy
from tvb.basic.logger.builder import get_logger
from tvb.simulator import models

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
        self.conf = conf

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
            for param_name, param_val in model_parameters.items():
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
            return numpy.array(vals)

        model_parameters = self.group_parameter_values_by_name(model_parameters_list)
        # change selected model in burst config
        model_class = getattr(models, model_name)
        self.conf.model = model_class()

        for param_name, param_vals in model_parameters.items():
            setattr(self.conf.model, param_name, format_param_vals(param_vals))

    def write_noise_parameters(self, noise_dispersions):
        """
        Set noise dispersions in burst config.
        It will set all nsig fields it can find in the config (at least 1 per stochastic integrator).
        :param noise_dispersions: A list of noise dispersions. One for each connectivity node. Ex [{'V': 1, 'W':2}, ...]
        """
        noise_dispersions = self.group_parameter_values_by_name(noise_dispersions)
        # Flatten the dict to an array of shape (state_vars, nodes)
        state_vars = self.conf.model.state_variables
        noise_arr = [noise_dispersions[sv] for sv in state_vars]

        self.conf.integrator.noise.nsig = numpy.array(noise_arr)
