# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
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
.. moduleauthor:: Robert Vincze <robert.vincze@codemart.ro>
"""

from tvb.basic.logger.builder import get_logger
from tvb.core.neocom import h5
from tvb.core.neotraits.spatial_model import SpatialModel
from tvb.interfaces.web.controllers.burst.transfer_vector_controller import TransferVectorForm

KEY_EQUATION = "equation"
KEY_RESULT = "result"


class TransferVectorContext(SpatialModel):
    def __init__(self, current_transfer_function, current_model_param, current_connectivity_measure):
        self.logger = get_logger(self.__class__.__module__)

        self.applied_transfer_functions = {}
        self.current_transfer_function = current_transfer_function
        self.current_model_param = current_model_param
        self.current_connectivity_measure = current_connectivity_measure

    @staticmethod
    def get_equation_information():
        return {
            TransferVectorForm.transfer_function_label: 'current_transfer_function'
        }

    def apply_transfer_function(self):
        """
        Applies a transfer function on the given model parameter.
        """

        cm = h5.load_from_gid(self.current_connectivity_measure)
        result = self.current_transfer_function.evaluate(cm.array_data)

        self.applied_transfer_functions[self.current_model_param] = {
            KEY_EQUATION: self.current_transfer_function,
            KEY_RESULT: result
        }
        return result

    def get_applied_transfer_functions(self):
        """
        :returns: a dictionary which contains information about the applied transfer functions on the model parameters.
        """
        result = {}
        for param in self.applied_transfer_functions:
            transfer_function = self.applied_transfer_functions[param][KEY_EQUATION]
            keys = sorted(list(transfer_function.parameters), key=lambda x: len(x))
            keys.reverse()
            base_tf = transfer_function.equation
            base_tf = base_tf.replace('var', 'x')
            for tf_param in keys:
                while True:
                    stripped_tf = "".join(base_tf.split())
                    param_idx = stripped_tf.find('\\' + tf_param)
                    if param_idx < 0:
                        break
                    if param_idx > 0 and stripped_tf[param_idx - 1].isalnum():
                        base_tf = transfer_function.replace('\\' + tf_param,
                                                            '*' + str(transfer_function.parameters[tf_param]))
                    else:
                        base_tf = base_tf.replace('\\' + tf_param, str(transfer_function.parameters[tf_param]))
                base_tf = base_tf.replace(tf_param, str(transfer_function.parameters[tf_param]))
            result[param] = base_tf
        return result
