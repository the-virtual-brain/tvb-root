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

import flask
from flask_restplus import Resource
from tvb.basic.profile import TvbProfile
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.interfaces.rest.commons.exceptions import BadRequestException
from tvb.interfaces.rest.server.decorators.rest_decorators import rest_jsonify, secured
from werkzeug.utils import secure_filename

USERS_PAGE_SIZE = 1000


class SecuredResource(Resource):
    method_decorators = [secured]


class RestResource(SecuredResource):
    method_decorators = [rest_jsonify]

    def __init__(self, *args, **kwargs):
        super(RestResource, self).__init__(args, kwargs)
        if not all(decorator in self.method_decorators for decorator in super().method_decorators):
            self.method_decorators.extend(super().method_decorators)

    @staticmethod
    def extract_file_from_request(file_name='model_file', file_extension=FilesHelper.TVB_STORAGE_FILE_EXTENSION):
        if RestResource.is_path_in_files(file_name):
            raise BadRequestException("No file '%s' in the request!" % file_name)
        file = flask.request.files[file_name]
        if not file.filename.endswith(file_extension):
            raise BadRequestException("Only %s files are allowed!" % file_extension)

        return file

    @staticmethod
    def save_temporary_file(file, destination_folder):
        filename = secure_filename(file.filename)
        full_path = os.path.join(destination_folder, filename)
        file.save(full_path)

        return full_path

    @staticmethod
    def get_destination_folder():
        temp_name = tempfile.mkdtemp(dir=TvbProfile.current.TVB_TEMP_FOLDER)
        destination_folder = os.path.join(TvbProfile.current.TVB_TEMP_FOLDER, temp_name)

        return destination_folder

    @staticmethod
    def is_path_in_files(file_name):
        return file_name not in flask.request.files
