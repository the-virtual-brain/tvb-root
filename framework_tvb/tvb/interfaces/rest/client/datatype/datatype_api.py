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

import cgi
import json
import os
import requests
from tvb.interfaces.rest.client.client_decorators import handle_response
from tvb.interfaces.rest.client.main_api import MainApi
from tvb.interfaces.rest.commons import RestLink, LinkPlaceholder
from tvb.interfaces.rest.commons.dtos import AlgorithmDto
from tvb.interfaces.rest.commons.exceptions import ClientException


class DataTypeApi(MainApi):

    def retrieve_datatype(self, datatype_gid, download_folder):
        response = requests.get(self.build_request_url(
            RestLink.GET_DATATYPE.compute_url(True,
                                              {LinkPlaceholder.DATATYPE_GID.value: datatype_gid})))
        content_disposition = response.headers['Content-Disposition']
        value, params = cgi.parse_header(content_disposition)
        file_name = params['filename']
        file_path = os.path.join(download_folder, os.path.basename(file_name))

        if response.ok:
            with open(file_path, 'wb') as local_file:
                for chunk in response.iter_content(chunk_size=128):
                    local_file.write(chunk)
            return file_path

        error_response = json.loads(response.content.decode('utf-8'))
        raise ClientException(error_response['message'], error_response['code'])

    @handle_response
    def get_operations_for_datatype(self, datatype_gid):
        response = requests.get(
            self.build_request_url(RestLink.DATATYPE_OPERATIONS.compute_url(True, {
                LinkPlaceholder.DATATYPE_GID.value: datatype_gid
            })))
        return response, AlgorithmDto

    def load_datatype(self, datatype_path):
        # TODO: TO BE IMPLEMENTED
        pass
