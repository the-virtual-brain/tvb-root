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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>

"""

import json
import cherrypy
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.model import PARAM_CONNECTIVITY, PARAMS_MODEL_PATTERN, PARAM_MODEL
from tvb.core.entities.storage import dao
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_page, handle_error, check_user


class RegionsModelParametersController(BurstBaseController):
    """
    Controller class for placing model parameters in nodes.
    """

    @staticmethod
    def _dynamics_json(dynamics):
        """ Construct a json representation of a list of dynamics as used by the js client """
        ret = {}
        for d in dynamics:
            ret[d.id] = {
                'id': d.id,
                'name': d.name,
                'model_class' : d.model_class
            }
        return json.dumps(ret)

    @staticmethod
    def get_connectivity_from_burst_configuration():
        burst_configuration = common.get_from_session(common.KEY_BURST_CONFIG)
        connectivity_gid = burst_configuration.get_simulation_parameter_value(PARAM_CONNECTIVITY)
        return ABCAdapter.load_entity_by_gid(connectivity_gid)

    @expose_page
    def index(self):
        current_user_id = common.get_logged_user().id
        dynamics = dao.get_dynamics_for_user(current_user_id)

        connectivity = self.get_connectivity_from_burst_configuration()
        params = ConnectivityViewer.get_connectivity_parameters(connectivity)

        params.update({
            'title': 'Model parameters',
            'mainContent': 'burst/model_param_region',
            'isSingleMode': True,
            'submit_parameters_url': '/burst/modelparameters/regions/submit_model_parameters',
            'dynamics': dynamics,
            'dynamics_json': self._dynamics_json(dynamics)
        })

        return self.fill_default_attributes(params)


    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit_model_parameters(self, dynamic_ids):
        """
        Collects the model parameters values from all the models used for the connectivity nodes.
        Assumes that the array indices are consistent with the node order.
        """
        dynamics = []

        for dynamic_id in json.loads(dynamic_ids):
            dynamics.append(dao.get_dynamic(dynamic_id))

        for dynamic in dynamics[1:]:
            if dynamic.model_class != dynamics[0].model_class:
                raise Exception("All dynamics must have the same model type")

        model_name = dynamics[0].model_class
        burst_configuration = common.get_from_session(common.KEY_BURST_CONFIG)
        model_parameters = self.group_parameter_values_by_name(json.loads(d.model_parameters) for d in dynamics)

        # change selected model in burst config
        burst_configuration.update_simulation_parameter(PARAM_MODEL, model_name)

        # update model parameters in burst config
        for param_name, param_vals in model_parameters.iteritems():
            full_name = PARAMS_MODEL_PATTERN % (model_name, param_name)
            burst_configuration.update_simulation_parameter(full_name, str(param_vals))

        ### Update in session BURST configuration for burst-page.
        common.add2session(common.KEY_BURST_CONFIG, burst_configuration.clone())
        raise cherrypy.HTTPRedirect("/burst/")
