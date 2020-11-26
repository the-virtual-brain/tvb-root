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
.. moduleauthor:: Mihai Andrei <mihai.andrei@codemart.ro>
"""
import json

import cherrypy
from tvb.adapters.visualizers.connectivity import ConnectivityViewer
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.storage import dao
from tvb.core.services.burst_config_serialization import SerializationManager
from tvb.interfaces.web.controllers import common
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.burst.base_controller import BurstBaseController
from tvb.interfaces.web.controllers.decorators import expose_page, handle_error, check_user
from tvb.interfaces.web.controllers.simulator.simulator_controller import SimulatorWizzardURLs


@traced
class NoiseConfigurationController(BurstBaseController):
    """
    Controller class for placing noise parameters in nodes.
    """
    @expose_page
    def index(self):
        des = SerializationManager(common.get_from_session(common.KEY_SIMULATOR_CONFIG))
        connectivity = des.conf.connectivity
        conn_idx = dao.get_datatype_by_gid(connectivity.hex)
        model = des.conf.model
        integrator = des.conf.integrator

        state_vars = model.state_variables
        noise_values = self.init_noise_config_values(model, integrator, conn_idx)
        initial_noise = self.group_noise_array_by_state_var(noise_values, state_vars, conn_idx.number_of_regions)

        current_project = common.get_current_project()
        file_handler = FilesHelper()
        conn_path = file_handler.get_project_folder(current_project, str(conn_idx.fk_from_operation))

        params = ConnectivityViewer.get_connectivity_parameters(conn_idx, conn_path)
        params.update({
            'title': 'Noise configuration',
            'mainContent': 'burst/noise',
            'isSingleMode': True,
            'submit_parameters_url': '/burst/noise/submit',
            'stateVars': state_vars,
            'stateVarsJson' : json.dumps(state_vars),
            'noiseInputValues' : initial_noise[0],
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
        des = SerializationManager(common.get_from_session(common.KEY_SIMULATOR_CONFIG))
        des.write_noise_parameters(json.loads(node_values))
        common.add2session(common.KEY_LAST_LOADED_FORM_URL, SimulatorWizzardURLs.SET_NOISE_PARAMS_URL)
        raise cherrypy.HTTPRedirect("/burst/")

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
            raise ValueError("Got unexpected noise shape %s." % (nsig.shape, ))
