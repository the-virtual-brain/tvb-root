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

from tvb.basic.profile import TvbProfile
from tvb.core.entities.load import load_entity_by_gid
from tvb.core.entities.storage import dao
from tvb.core.neocom.h5 import h5_file_for_index
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.interfaces.rest.commons.dtos import AlgorithmDto
from tvb.interfaces.rest.commons.exceptions import ServiceException
from tvb.storage.storage_interface import StorageInterface


class DatatypeFacade:
    def __init__(self):
        self.algorithm_service = AlgorithmService()
        self.storage_interface = StorageInterface()

    @staticmethod
    def is_data_encrypted():
        return StorageInterface.encryption_enabled()

    def get_dt_h5_path(self, datatype_gid, public_key_path=None):
        index = load_entity_by_gid(datatype_gid)
        h5_path = h5_file_for_index(index).path
        file_name = os.path.basename(h5_path)

        if StorageInterface.encryption_enabled():
            if public_key_path and os.path.exists(public_key_path):
                operation = dao.get_operation_by_id(index.fk_from_operation)
                project = dao.get_project_by_id(operation.fk_launched_in)
                folder_path = self.storage_interface.get_project_folder(project.name)
                self.storage_interface.inc_running_op_count(folder_path)
                self.storage_interface.sync_folders(folder_path)
                encrypted_h5_path = self.storage_interface.export_datatype_from_rest_server(
                    h5_path, index, file_name, public_key_path)
                file_name = file_name.replace(StorageInterface.TVB_STORAGE_FILE_EXTENSION,
                                              StorageInterface.TVB_ZIP_FILE_EXTENSION)
                self.storage_interface.dec_running_op_count(folder_path)
                self.storage_interface.check_and_delete(folder_path)
                return encrypted_h5_path, file_name
            else:
                raise ServiceException('Client requested encrypted retrieval of data but the server has no'
                                       ' valid private key!')
        return h5_path, file_name

    def get_datatype_operations(self, datatype_gid):
        categories = dao.get_launchable_categories(elimin_viewers=True)
        datatype = dao.get_datatype_by_gid(datatype_gid)
        _, filtered_adapters, _ = self.algorithm_service.get_launchable_algorithms_for_datatype(datatype, categories)
        return [AlgorithmDto(algorithm) for algorithm in filtered_adapters]

    @staticmethod
    def get_extra_info(datatype_gid):
        extra_info = dao.get_datatype_extra_info(datatype_gid)
        if extra_info is None:
            return None

        return extra_info
