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

from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.adapters.abcuploader import ABCUploader
from tvb.core.neocom import h5
from tvb.core.neotraits.h5 import ViewModelH5
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.strings import RestLink, LinkPlaceholder
from tvb.interfaces.rest.commons.dtos import DataTypeDto
from tvb.interfaces.rest.commons.strings import RequestFileKey


class OperationApi(MainApi):
    @handle_response
    def get_operation_status(self, operation_gid):
        return self.secured_request().get(self.build_request_url(RestLink.OPERATION_STATUS.compute_url(True, {
            LinkPlaceholder.OPERATION_GID.value: operation_gid
        })))

    @handle_response
    def get_operations_results(self, operation_gid):
        response = self.secured_request().get(
            self.build_request_url(RestLink.OPERATION_RESULTS.compute_url(True, {
                LinkPlaceholder.OPERATION_GID.value: operation_gid
            })))
        return response, DataTypeDto

    @handle_response
    def launch_operation(self, project_gid, algorithm_class, view_model, temp_folder):
        h5_file_path = h5.store_view_model(view_model, temp_folder)

        model_file_obj = open(h5_file_path, 'rb')
        files = {RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value: (os.path.basename(h5_file_path), model_file_obj)}

        if issubclass(algorithm_class, ABCUploader):
            for key in algorithm_class().get_form_class().get_upload_information().keys():
                path = getattr(view_model, key)
                data_file_obj = open(path, 'rb')
                files[key] = (os.path.basename(path), data_file_obj)

        return self.secured_request().post(self.build_request_url(RestLink.LAUNCH_OPERATION.compute_url(True, {
            LinkPlaceholder.PROJECT_GID.value: project_gid,
            LinkPlaceholder.ALG_MODULE.value: algorithm_class.__module__,
            LinkPlaceholder.ALG_CLASSNAME.value: algorithm_class.__name__
        })), files=files)

    def quick_launch_operation(self, project_gid, algorithm_dto, datatype_gid, temp_folder):
        adapter_class = ABCAdapter.determine_adapter_class(algorithm_dto)
        form = adapter_class().get_form()()

        post_data = self._prepare_post_data(datatype_gid, form)
        form.fill_from_post_plus_defaults(post_data)

        view_model = form.get_view_model()()
        form.fill_trait(view_model)

        operation_gid = self.launch_operation(project_gid, adapter_class, view_model, temp_folder)
        return operation_gid

    def _prepare_post_data(self, datatype_gid, form):
        post_data = {form.get_input_name(): datatype_gid,
                     'fill_defaults': 'true'}
        return post_data
