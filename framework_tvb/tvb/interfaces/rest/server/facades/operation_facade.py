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
from tvb.basic.logger.builder import get_logger
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcuploader import ABCUploader
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.core.services.exceptions import ProjectServiceException
from tvb.core.services.algorithm_service import AlgorithmService
from tvb.core.services.operation_service import OperationService
from tvb.core.services.project_service import ProjectService
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons.dtos import DataTypeDto
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException, ServiceException


class OperationFacade:
    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)
        self.operation_service = OperationService()
        self.project_service = ProjectService()
        self.user_service = UserService()
        self.files_helper = FilesHelper()

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
        temp_folder = FilesHelper.create_temp_folder()
        model_h5_path = FilesHelper.save_temporary_file(model_file, temp_folder)

        try:
            project = self.project_service.find_project_lazy_by_gid(project_gid)
        except ProjectServiceException:
            raise InvalidIdentifierException()

        algorithm = AlgorithmService.get_algorithm_by_module_and_class(algorithm_module, algorithm_classname)
        if algorithm is None:
            raise InvalidIdentifierException('No algorithm found for: %s.%s' % (algorithm_module, algorithm_classname))

        try:
            adapter_instance = ABCAdapter.build_adapter(algorithm)
            view_model = adapter_instance.get_view_model_class()()

            view_model_h5 = ViewModelH5(model_h5_path, view_model)
            view_model_gid = view_model_h5.gid.load()

            operation = self.operation_service.prepare_operation(current_user_id, project.id, algorithm.id,
                                                                 algorithm.algorithm_category, view_model_gid.hex, None,
                                                                 {})
            storage_path = self.files_helper.get_project_folder(project, str(operation.id))

            if isinstance(adapter_instance, ABCUploader):
                for key, value in adapter_instance.get_form_class().get_upload_information().items():
                    data_file = fetch_file(request_file_key=key, file_extension=value)
                    data_file_path = FilesHelper.save_temporary_file(data_file, temp_folder)
                    file_name = os.path.basename(data_file_path)
                    upload_field = getattr(view_model_h5, key)
                    upload_field.store(os.path.join(storage_path, file_name))
                    shutil.move(data_file_path, storage_path)

            shutil.move(model_h5_path, storage_path)
            os.rmdir(temp_folder)
            view_model_h5.close()
            OperationService().launch_operation(operation.id, True)
            return operation.gid
        except Exception as excep:
            self.logger.error(excep, exc_info=True)
            raise ServiceException(str(excep))
