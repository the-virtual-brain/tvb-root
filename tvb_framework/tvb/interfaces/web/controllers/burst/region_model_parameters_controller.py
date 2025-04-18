# -*- coding: utf-8 -*-
#
#
# TheVirtualBrain-Framework Package. This package holds all Data Management, and 
# Web-UI helpful to run brain-simulations. To use it, you also need to download
# TheVirtualBrain-Scientific Package (for simulators). See content of the
# documentation-folder for more details. See also http://www.thevirtualbrain.org
#
# (c) 2012-2025, Baycrest Centre for Geriatric Care ("Baycrest") and others
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
.. moduleauthor:: Bogdan Neacsa <bogdan.neacsa@codemart.ro>
.. moduleauthor:: Ionel Ortelecan <ionel.ortelecan@codemart.ro>
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>

"""

import json

import cherrypy
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.core.entities import load
from tvb.core.entities.storage import dao
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_page, handle_error, check_user
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorWizzardURLs
from tvb.interfaces.web.entities.context_simulator import SimulatorContext


@traced
class RegionsModelParametersController(BurstBaseController):
    """
    Controller class for placing model parameters in nodes.
    """

    def __init__(self):
        super(RegionsModelParametersController, self).__init__()
        self.simulator_context = SimulatorContext()

    @staticmethod
    def _dynamics_json(dynamics):
        """ Construct a json representation of a list of dynamics as used by the js client """
        ret = {}
        for d in dynamics:
            ret[d.id] = {
                'id': d.id,
                'name': d.name,
                'model_class': d.model_class
            }
        return json.dumps(ret)

    def no_dynamics_page(self):
        params = ({
            'title': 'Model parameters',
            'mainContent': 'burst/model_param_region_empty',
        })
        return self.fill_default_attributes(params)

    @expose_page
    def index(self):
        current_user_id = self.simulator_context.logged_user.id
        # In case the number of dynamics gets big we should add a filter in the ui.
        dynamics = dao.get_dynamics_for_user(current_user_id)

        if not dynamics:
            return self.no_dynamics_page()

        sim_config = self.simulator_context.simulator
        connectivity = sim_config.connectivity

        if connectivity is None:
            msg = 'You have to select a connectivity before setting up the region Model. '
            common.set_error_message(msg)
            raise ValueError(msg)

        current_project = common.get_current_project()
        conn_idx = load.load_entity_by_gid(connectivity)
        params = ConnectivityViewer.get_connectivity_parameters(conn_idx, current_project.name,
                                                                str(conn_idx.fk_from_operation))
        burst_config = self.simulator_context.burst_config

        params.update({
            'title': 'Model parameters',
            'mainContent': 'burst/model_param_region',
            'isSingleMode': True,
            'submit_parameters_url': self.build_path('/burst/modelparameters/regions/submit_model_parameters'),
            'dynamics': dynamics,
            'dynamics_json': self._dynamics_json(dynamics),
            'initial_dynamic_ids': burst_config.dynamic_ids
        })

        return self.fill_default_attributes(params, 'regionmodel')

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit_model_parameters(self, node_values):
        """
        Collects the model parameters values from all the models used for the connectivity nodes.
        Assumes that the array indices are consistent with the node order.
        """
        dynamic_ids = json.loads(node_values)
        dynamics = [dao.get_dynamic(did) for did in dynamic_ids]

        for dynamic in dynamics[1:]:
            if dynamic.model_class != dynamics[0].model_class:
                raise Exception("All dynamics must have the same model type")

        model_name = dynamics[0].model_class
        burst_config = self.simulator_context.burst_config
        simulator_config = self.simulator_context.simulator

        # update model parameters in burst config
        des = SerializationManager(simulator_config)
        model_parameters = [dict(json.loads(d.model_parameters)) for d in dynamics]
        des.write_model_parameters(model_name, model_parameters)

        # update dynamic ids in burst config
        burst_config.dynamic_ids = json.dumps(dynamic_ids)

        # Update in session the simulator configuration and the current form URL in wizzard for burst-page.
        self.simulator_context.set_burst_config(burst_config)
        self.simulator_context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_INTEGRATOR_URL)
        self.redirect("/burst/")
