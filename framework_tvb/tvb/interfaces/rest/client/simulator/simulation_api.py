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

import os
import tempfile

import requests
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.entities.model.simulator.simulator import SimulatorIndex
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons import RestLink, LinkPlaceholder


class SimulationApi(MainApi):

    @handle_response
    def fire_simulation(self, project_gid, session_stored_simulator, temp_folder):
        simulator_index = SimulatorIndex()
        temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
        destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)
        simulation_state_gid = None

        SimulatorSerializer().serialize_simulator(session_stored_simulator, simulator_index.gid,
                                                  simulation_state_gid, destination_folder)
        zip_folder_path = temp_folder + '/SimulationData.zip'
        FilesHelper().zip_folder(zip_folder_path, destination_folder)

        file_obj = open(zip_folder_path, 'rb')
        return requests.post(self.build_request_url(RestLink.FIRE_SIMULATION.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid
        })), files={"file": ("SimulationData.zip", file_obj)})
