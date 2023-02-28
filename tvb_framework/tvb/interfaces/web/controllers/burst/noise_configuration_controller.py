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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json
import cherrypy

from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.core.entities import load
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_page, handle_error, check_user
from tvb.interfaces.web.controllers.simulator.simulator_wizzard_urls import SimulatorWizzardURLs
from tvb.interfaces.web.entities.context_simulator import SimulatorContext


@traced
class NoiseConfigurationController(BurstBaseController):
    """
    Controller class for placing noise parameters in nodes.
    """

    def __init__(self):
        super(NoiseConfigurationController, self).__init__()
        self.simulator_context = SimulatorContext()

    @expose_page
    def index(self):
        des = SerializationManager(self.simulator_context.simulator)
        conn_idx = load.load_entity_by_gid(des.conf.connectivity)
        model = des.conf.model
        integrator = des.conf.integrator

        state_vars = model.state_variables
        noise_values = self.init_noise_config_values(model, integrator, conn_idx)
        initial_noise = self.group_noise_array_by_state_var(noise_values, state_vars, conn_idx.number_of_regions)

        current_project = common.get_current_project()

        params = ConnectivityViewer.get_connectivity_parameters(conn_idx, current_project.name,
                                                                str(conn_idx.fk_from_operation))
        params.update({
            'title': 'Noise configuration',
            'mainContent': 'burst/noise',
            'isSingleMode': True,
            'submit_parameters_url': self.build_path('/burst/noise/submit'),
            'stateVars': state_vars,
            'stateVarsJson': json.dumps(state_vars),
            'noiseInputValues': initial_noise[0],
            'initialNoiseValues': json.dumps(initial_noise)
        })
        return self.fill_default_attributes(params, 'regionmodel')

    @cherrypy.expose
    @handle_error(redirect=True)
    @check_user
    def submit(self, node_values):
        """
        Submit noise dispersions
        :param node_values: A map from state variable names to noise dispersion arrays. Ex {'V': [1,2...74]}
        """
        des = SerializationManager(self.simulator_context.simulator)
        des.write_noise_parameters(json.loads(node_values))
        self.simulator_context.add_last_loaded_form_url_to_session(SimulatorWizzardURLs.SET_NOISE_PARAMS_URL)
        self.redirect("/burst/")

    @staticmethod
    def group_noise_array_by_state_var(noise_values, state_vars, number_of_regions):
        initial_noise = []
        for i in range(number_of_regions):
            node_noise = {}
            for sv_idx, sv in enumerate(state_vars):
                node_noise[sv] = noise_values[sv_idx][i]
            initial_noise.append(node_noise)
        return initial_noise

    @staticmethod
    def init_noise_config_values(model, integrator, connectivity):
        """
        Initialize a state var x number of nodes array with noise values.
        """
        state_variables = model.state_variables
        nr_nodes = connectivity.number_of_regions
        nr_state_vars = len(state_variables)

        try:
            nsig = integrator.noise.nsig
            noise_values = nsig.tolist()
        except AttributeError:
            # Just fallback to default
            return [[1 for _ in range(nr_nodes)] for _ in state_variables]

        if nsig.shape == (1,):
            # Only one number for noise
            return [noise_values * nr_nodes for _ in state_variables]
        elif nsig.shape == (nr_state_vars, 1) or nsig.shape == (nr_state_vars,):
            # Only one number per state variable
            return [[noise_values[idx]] * nr_nodes for idx in range(nr_state_vars)]
        elif nsig.shape == (nr_state_vars, nr_nodes):
            return noise_values
        else:
            raise ValueError("Got unexpected noise shape %s." % (nsig.shape,))
