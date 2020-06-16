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
import shutil
import tempfile
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.services.simulator_serializer import SimulatorSerializer
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.commons.strings import RestLink, LinkPlaceholder


class SimulationApi(MainApi):

    @handle_response
    def fire_simulation(self, project_gid, session_stored_simulator, temp_folder):
        temporary_folder = FilesHelper.create_temp_folder()
        simulation_state_gid = None

        SimulatorSerializer().serialize_simulator(session_stored_simulator, simulation_state_gid, temporary_folder)
        zip_folder_path = os.path.join(temp_folder, RequestFileKey.SIMULATION_FILE_NAME.value)
        FilesHelper().zip_folder(zip_folder_path, temporary_folder)
        shutil.rmtree(temporary_folder)

        file_obj = open(zip_folder_path, 'rb')
        return self.secured_request().post(self.build_request_url(RestLink.FIRE_SIMULATION.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid
        })), files={RequestFileKey.SIMULATION_FILE_KEY.value: (RequestFileKey.SIMULATION_FILE_NAME.value, file_obj)})
