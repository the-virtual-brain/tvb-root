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

import cgi
import json
import os

from tvb.basic.neotraits.api import HasTraits
from tvb.core.neocom import h5
from tvb.core.neocom.h5 import REGISTRY, TVBLoader
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons.dtos import AlgorithmDto
from tvb.interfaces.rest.commons.exceptions import ClientException
from tvb.interfaces.rest.commons.files_helper import save_file
from tvb.interfaces.rest.commons.strings import RestLink, LinkPlaceholder
from tvb.storage.storage_interface import StorageInterface


class DataTypeApi(MainApi):

    @handle_response
    def is_data_encrypted(self):
        response = self.secured_request().get(self.build_request_url(RestLink.IS_DATA_ENCRYPTED.compute_url(True)))
        return response

    def retrieve_datatype(self, datatype_gid, download_folder, is_data_encrypted):
        storage_interface = StorageInterface()
        import_export_encryption_handler = storage_interface.get_import_export_encryption_handler()
        public_key = None

        if is_data_encrypted:
            import_export_encryption_handler.generate_public_private_key_pair(download_folder)
            public_key_path = os.path.join(download_folder, import_export_encryption_handler.PUBLIC_KEY_NAME)
            with open(public_key_path) as f:
                public_key = f.read()

        response = self.secured_request().get(self.build_request_url(
            RestLink.GET_DATATYPE.compute_url(
                True, {LinkPlaceholder.DATATYPE_GID.value: datatype_gid})),
            files={import_export_encryption_handler.PUBLIC_KEY_NAME: public_key}, stream=True)

        if response.ok:
            content_disposition = response.headers['Content-Disposition']
            value, params = cgi.parse_header(content_disposition)
            file_name = params['filename']
            file_path = os.path.join(download_folder, os.path.basename(file_name))
            file_path = save_file(file_path, response)
            if is_data_encrypted:
                private_kay_path = os.path.join(download_folder, import_export_encryption_handler.PRIVATE_KEY_NAME)
                file_path = storage_interface.import_datatype_to_rest_client(file_path, download_folder,
                                                                             private_kay_path)

            return file_path

        error_response = json.loads(response.content.decode('utf-8'))
        raise ClientException(error_response['message'], error_response['code'])

    @handle_response
    def get_operations_for_datatype(self, datatype_gid):
        response = self.secured_request().get(
            self.build_request_url(RestLink.DATATYPE_OPERATIONS.compute_url(True, {
                LinkPlaceholder.DATATYPE_GID.value: datatype_gid
            })))
        return response, AlgorithmDto

    @handle_response
    def get_extra_info(self, datatype_gid):
        response = self.secured_request().get(
            self.build_request_url(RestLink.DATATYPE_EXTRA_INFO.compute_url(True, {
                LinkPlaceholder.DATATYPE_GID.value: datatype_gid
            })))
        return response, dict

    @staticmethod
    def load_datatype_from_file(datatype_path):
        datatype, _ = h5.load_with_links(datatype_path)
        return datatype

    def _load_with_full_references(self, file_path, download_folder, is_data_encrypted):
        # type: (str, str, bool) -> HasTraits
        def load_ht_function(sub_gid, traited_attr):
            ref_ht_path = self.retrieve_datatype(sub_gid.hex, download_folder, is_data_encrypted)
            ref_ht, _ = h5.load_with_links(ref_ht_path)
            return ref_ht

        loader = TVBLoader(REGISTRY)
        return loader.load_complete_by_function(file_path, load_ht_function)

    def load_datatype_with_full_references(self, datatype_gid, download_folder, is_data_encrypted):
        datatype_path = self.retrieve_datatype(datatype_gid, download_folder, is_data_encrypted)
        datatype, _ = self._load_with_full_references(datatype_path, download_folder, is_data_encrypted)
        return datatype

    def load_datatype_with_links(self, datatype_gid, download_folder, is_data_encrypted):
        datatype_path = self.retrieve_datatype(datatype_gid, download_folder, is_data_encrypted)
        datatype, _ = h5.load_with_links(datatype_path)

        return datatype
