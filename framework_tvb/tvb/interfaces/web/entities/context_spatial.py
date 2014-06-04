# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need do download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2013, Baycrest Centre for Geriatric Care ("Baycrest")
#
# This program is free software; you can redistribute it and/or modify it under 
# the terms of the GNU General Public License version 2 as published by the Free
# Software Foundation. This program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
# License for more details. You should have received a copy of the GNU General 
# Public License along with this program; if not, you can download it here
# http://www.gnu.org/licenses/old-licenses/gpl-2.0
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
"""

import numpy
from copy import deepcopy
import tvb.simulator.models as models_module
import tvb.simulator.integrators as integrators_module
from tvb.basic.logger.builder import get_logger


class BaseSpatialContext():
    
    def __init__(self, connectivity, default_model=None, default_integrator=None):
        self.logger = get_logger(self.__class__.__module__)
        self.default_model = default_model
        self.default_integrator = default_integrator
        self.connectivity = connectivity
        self.connectivity_models = dict()

        if self.default_model is None:
            self.default_model = models_module.Generic2dOscillator()
        if self.default_integrator is None:
            self.default_integrator = integrators_module.RungeKutta4thOrderDeterministic()

        self.model_parameter_names = deepcopy(self.default_model.ui_configurable_parameters)
        if not len(self.model_parameter_names):
            self.logger.warning("The 'ui_configurable_parameters' list of the current model is empty!")
            
    
    def _get_model_for_region(self, connectivity_node_index):
        """
        Returns the model instance corresponding to the connectivity node found at the specified index.
        """
        connectivity_node_index = int(connectivity_node_index)
        if connectivity_node_index not in self.connectivity_models:
            model = self._compute_default_model(connectivity_node_index)
            self.connectivity_models[connectivity_node_index] = deepcopy(model)
        return self.connectivity_models[connectivity_node_index]
    
    
    def _compute_default_model(self, connectivity_node_index):
        """
        Computes the default model for the given connectivity node.

        If the parameters of the default model contain values for all the nodes
        of the connectivity, than this method will build a new model which will
        have the parameters set to a numpy array with a single value. The value
        is computed from the default model parameter values.
        """
        model = deepcopy(self.default_model)
        for parameter_name in self.model_parameter_names:
            default_attr = getattr(model, parameter_name)
            if isinstance(default_attr, numpy.ndarray):
                default_attr = default_attr.tolist()
                if len(default_attr) > 1:
                    if len(default_attr) <= connectivity_node_index:
                        setattr(model, parameter_name, numpy.array([default_attr[0]]))
                    else:
                        setattr(model, parameter_name, numpy.array([default_attr[connectivity_node_index]]))
        return model

