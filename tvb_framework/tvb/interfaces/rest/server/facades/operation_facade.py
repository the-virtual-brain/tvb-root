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

from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcuploader import ABCUploader
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons.dtos import DataTypeDto
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, ServiceException
from tvb.interfaces.rest.commons.files_helper import create_temp_folder, save_temporary_file


class OperationFacade:
    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.operation_service = OperationService()
        self.project_service = ProjectService()
        self.user_service = UserService()

    @staticmethod
    def get_operation_status(operation_gid):
        operation = ProjectService.load_operation_by_gid(operation_gid)
        if operation is None:
            get_logger().warning("Invalid operation GID: {}".format(operation_gid))
            raise InvalidIdentifierException()

        return operation.status

    @staticmethod
    def get_operations_results(operation_gid):
        operation = ProjectService.load_operation_lazy_by_gid(operation_gid)
        if operation is None:
            get_logger().warning("Invalid operation GID: {}".format(operation_gid))
            raise InvalidIdentifierException()

        data_types = ProjectService.get_results_for_operation(operation.id)
        if data_types is None:
            return []

        return [DataTypeDto(datatype) for datatype in data_types]

    def launch_operation(self, current_user_id, model_file, project_gid, algorithm_module, algorithm_classname,
                         fetch_file):
        temp_folder = create_temp_folder()
        model_h5_path = save_temporary_file(model_file, temp_folder)

        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException()

        try:
            algorithm = AlgorithmService.get_algorithm_by_module_and_class(algorithm_module, algorithm_classname)
            if algorithm is None:
                raise InvalidIdentifierException(
                    'No algorithm found for: %s.%s' % (algorithm_module, algorithm_classname))

            adapter_instance = ABCAdapter.build_adapter(algorithm)
            view_model = h5.load_view_model_from_file(model_h5_path)
            if isinstance(adapter_instance, ABCUploader):
                with ViewModelH5(model_h5_path, view_model) as view_model_h5:
                    for key, value in adapter_instance.get_form_class().get_upload_information().items():
                        data_file = fetch_file(request_file_key=key, file_extension=value)
                        data_file_path = save_temporary_file(data_file, temp_folder)
                        view_model_h5.store_metadata_param(key, data_file_path)
            view_model = h5.load_view_model_from_file(model_h5_path)

            operation = self.operation_service.prepare_operation(current_user_id, project, algorithm,
                                                                 view_model=view_model)
            if os.path.exists(model_h5_path):
                os.remove(model_h5_path)

            OperationService().launch_operation(operation.id, True)
            return operation.gid
        except Exception as excep:
            self.logger.error(excep, exc_info=True)
            raise ServiceException(str(excep))
