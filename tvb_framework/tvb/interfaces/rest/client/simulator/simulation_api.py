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

import os

from tvb.core.neocom import h5
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.files_helper import create_temp_folder
from tvb.interfaces.rest.commons.strings import RequestFileKey
from tvb.interfaces.rest.commons.strings import RestLink, LinkPlaceholder
from tvb.storage.storage_interface import StorageInterface


class SimulationApi(MainApi):

    @handle_response
    def fire_simulation(self, project_gid, session_stored_simulator, temp_folder):
        temporary_folder = create_temp_folder()

        h5.store_view_model(session_stored_simulator, temporary_folder)
        zip_folder_path = os.path.join(temp_folder, RequestFileKey.SIMULATION_FILE_NAME.value)
        StorageInterface().write_zip_folder(zip_folder_path, temporary_folder)
        StorageInterface.remove_folder(temporary_folder)

        file_obj = open(zip_folder_path, 'rb')
        return self.secured_request().post(self.build_request_url(RestLink.FIRE_SIMULATION.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid
        })), files={RequestFileKey.SIMULATION_FILE_KEY.value: (RequestFileKey.SIMULATION_FILE_NAME.value, file_obj)})
