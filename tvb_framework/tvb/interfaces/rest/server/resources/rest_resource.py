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

import flask
from flask_restx import Resource

from tvb.interfaces.rest.commons.exceptions import BadRequestException, InvalidInputException
from tvb.interfaces.rest.commons.strings import RequestFileKey, Strings
from tvb.interfaces.rest.server.decorators.rest_decorators import rest_jsonify, secured
from tvb.storage.storage_interface import StorageInterface


class SecuredResource(Resource):
    method_decorators = [secured]


class RestResource(SecuredResource):
    method_decorators = [rest_jsonify]

    def __init__(self, *args, **kwargs):
        super(RestResource, self).__init__(args, kwargs)
        if not all(decorator in self.method_decorators for decorator in super().method_decorators):
            self.method_decorators.extend(super().method_decorators)

    @staticmethod
    def extract_file_from_request(request_file_key=RequestFileKey.LAUNCH_ANALYZERS_MODEL_FILE.value,
                                  file_extension=StorageInterface.TVB_STORAGE_FILE_EXTENSION):
        if not RestResource.is_key_in_request_files(request_file_key):
            raise BadRequestException("No file '%s' in the request!" % request_file_key)
        file = flask.request.files[request_file_key]
        if not file.filename.endswith(file_extension):
            raise BadRequestException("Only %s files are allowed!" % file_extension)

        return file

    def extract_page_number(self):
        page_number = flask.request.args.get(Strings.PAGE_NUMBER.value)
        if page_number is None:
            page_number = 1
        try:
            page_number = int(page_number)
        except ValueError:
            raise InvalidInputException(message="Invalid page number")
        return page_number

    @staticmethod
    def is_key_in_request_files(key):
        return key in flask.request.files
