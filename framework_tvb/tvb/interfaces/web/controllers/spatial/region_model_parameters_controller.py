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
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
"""

import json
import cherrypy
from tvb.basic.traits.util import multiline_math_directives_to_matjax
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.base_controller import BaseController
from tvb.interfaces.web.controllers.decorators import expose_page, expose_fragment, expose_json, handle_error, check_user
from tvb.interfaces.web.entities.context_model_parameters import ContextModelParameters
from tvb.interfaces.web.controllers.spatial.base_spatio_temporal_controller import SpatioTemporalController, PARAMS_MODEL_PATTERN


### SESSION KEY for ContextModelParameter entity.
KEY_CONTEXT_MPR = "ContextForModelParametersOnRegion"


class RegionsModelParametersController(SpatioTemporalController):
    """
    Controller class for editing Model Parameters on regions in a visual manner.
    """
    
    def __init__(self):
        SpatioTemporalController.__init__(self)


    def _dfun_math_directives_to_matjax(self, model):
        """
        Looks for sphinx math directives if the docstring of the dfun function of a model.
        It converts them in html text that will be interpreted by mathjax
        The parsing is simplistic, not a full rst parser.
        """
        dfun = getattr(model, 'dfun', None)

        if dfun:
            return multiline_math_directives_to_matjax(dfun.__doc__).replace('&', '&amp;').replace('.. math::','')
        else:
            return ''


    @expose_page
    def edit_model_parameters(self):
        """
        Main method, to initialize Model-Parameter visual-set.
        """
        model, integrator, connectivity, _ = self.get_data_from_burst_configuration()

        connectivity_viewer_params = self.get_connectivity_parameters(connectivity)
        context_model_parameters = ContextModelParameters(connectivity, model, integrator)
        data_for_param_sliders = self.get_data_for_param_sliders('0', context_model_parameters)
        common.add2session(KEY_CONTEXT_MPR, context_model_parameters)

        template_specification = dict(title="Spatio temporal - Model parameters")
        template_specification['submit_parameters_url'] = '/spatial/modelparameters/regions/submit_model_parameters'
        template_specification['parametersNames'] = context_model_parameters.model_parameter_names
        template_specification['isSingleMode'] = True
        template_specification['paramSlidersData'] = json.dumps(data_for_param_sliders)
        template_specification.update(connectivity_viewer_params)
        template_specification['mainContent'] = 'spatial/model_param_region_main'
        template_specification['displayDefaultSubmitBtn'] = True
        template_specification.update(context_model_parameters.phase_plane_params)
        template_specification['modelEquations'] = self._dfun_math_directives_to_matjax(model)
        return self.fill_default_attributes(template_specification)


    @expose_fragment('spatial/model_param_region_param_sliders')
    def load_model_for_connectivity_node(self, connectivity_node_index):
        """
        Loads the model of the given connectivity node into the phase plane.
        """
        if int(connectivity_node_index) < 0:
            return
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPR)
        context_model_parameters.load_model_for_connectivity_node(connectivity_node_index)

        data_for_param_sliders = self.get_data_for_param_sliders(connectivity_node_index, context_model_parameters)
        template_specification = dict()
        template_specification['paramSlidersData'] = json.dumps(data_for_param_sliders)
        template_specification['parametersNames'] = data_for_param_sliders['all_param_names']
        return template_specification


    @expose_json
    def update_model_parameter_for_nodes(self, param_name, new_param_value, connectivity_node_indexes):
        """
        Updates the specified model parameter for the first node from the 'connectivity_node_indexes'
        list and after that replace the model of each node from the 'connectivity_node_indexes' list
        with the model of the first node from the list.
        """
        connectivity_node_indexes = json.loads(connectivity_node_indexes)
        if not len(connectivity_node_indexes):
            return
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPR)
        first_node_index = connectivity_node_indexes[0]
        context_model_parameters.update_model_parameter(first_node_index, param_name, new_param_value)
        if len(connectivity_node_indexes) > 1:
            #eliminate the first node
            connectivity_node_indexes = connectivity_node_indexes[1: len(connectivity_node_indexes)]
            context_model_parameters.set_model_for_connectivity_nodes(first_node_index, connectivity_node_indexes)
        common.add2session(KEY_CONTEXT_MPR, context_model_parameters)


    @expose_json
    def copy_model(self, from_node, to_nodes):
        """
        Replace the model of the nodes 'to_nodes' with the model of the node 'from_node'.

        ``from_node``: the index of the node from where will be copied the model
        ``to_nodes``: a list with the nodes indexes for which will be replaced the model
        """
        from_node = int(from_node)
        to_nodes = json.loads(to_nodes)
        if from_node < 0 or not len(to_nodes):
            return
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPR)
        context_model_parameters.set_model_for_connectivity_nodes(from_node, to_nodes)
        common.add2session(KEY_CONTEXT_MPR, context_model_parameters)


    @expose_fragment('spatial/model_param_region_param_sliders')
    def reset_model_parameters_for_nodes(self, connectivity_node_indexes):
        """
        Resets the model parameters, of the specified connectivity nodes, to their default values.
        """
        connectivity_node_indexes = json.loads(connectivity_node_indexes)
        if not len(connectivity_node_indexes):
            return

        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPR)
        context_model_parameters.reset_model_parameters_for_nodes(connectivity_node_indexes)
        context_model_parameters.load_model_for_connectivity_node(connectivity_node_indexes[0])
        data_for_param_sliders = self.get_data_for_param_sliders(connectivity_node_indexes[0], context_model_parameters)
        common.add2session(KEY_CONTEXT_MPR, context_model_parameters)

        template_specification = dict()
        template_specification['paramSlidersData'] = json.dumps(data_for_param_sliders)
        template_specification['parametersNames'] = data_for_param_sliders['all_param_names']
        return template_specification


    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit_model_parameters(self):
        """
        Collects the model parameters values from all the models used for the connectivity nodes.
        """
        context_model_parameters = common.get_from_session(KEY_CONTEXT_MPR)
        burst_configuration = common.get_from_session(common.KEY_BURST_CONFIG)
        for param_name in context_model_parameters.model_parameter_names:
            full_name = PARAMS_MODEL_PATTERN % (context_model_parameters.model_name, param_name)
            full_values = context_model_parameters.get_values_for_parameter(param_name)
            burst_configuration.update_simulation_parameter(full_name, full_values) 
        ### Clean from session drawing context
        common.remove_from_session(KEY_CONTEXT_MPR)
        ### Update in session BURST configuration for burst-page. 
        common.add2session(common.KEY_BURST_CONFIG, burst_configuration.clone())
        raise cherrypy.HTTPRedirect("/burst/")

        
    def fill_default_attributes(self, template_dictionary):
        """
        Overwrite base controller to add required parameters for adapter templates.
        """
        template_dictionary[common.KEY_SECTION] = 'burst'
        template_dictionary[common.KEY_SUB_SECTION] = 'regionmodel'
        template_dictionary[common.KEY_INCLUDE_RESOURCES] = 'spatial/included_resources'
        BaseController.fill_default_attributes(self, template_dictionary)
        return template_dictionary
    
