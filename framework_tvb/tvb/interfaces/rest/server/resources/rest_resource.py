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
import flask
from flask_restplus import Resource
from tvb.core.entities.file.files_helper import FilesHelper
from tvb.interfaces.rest.commons.exceptions import BadRequestException
from tvb.interfaces.rest.server.decorators.rest_decorators import rest_jsonify


class RestResource(Resource):
    method_decorators = [rest_jsonify]

    @staticmethod
    def extract_file_from_request(file_extension=FilesHelper.TVB_STORAGE_FILE_EXTENSION):
        if 'file' not in flask.request.files:
            raise BadRequestException('No file part in the request!')
        file = flask.request.files['file']
        if not file.filename.endswith(file_extension):
            raise BadRequestException('Only %s files are allowed!' % file_extension)

        return file
