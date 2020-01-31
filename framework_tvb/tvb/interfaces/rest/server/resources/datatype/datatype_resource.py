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
from tvb.core.adapters.abcadapter import ABCAdapter
from tvb.core.entities.storage import dao
from tvb.core.neocom.h5 import h5_file_for_index
from tvb.core.services.flow_service import FlowService
from tvb.interfaces.rest.commons.dtos import AlgorithmDto
from tvb.interfaces.rest.commons.exceptions import InvalidIdentifierException
from tvb.interfaces.rest.server.resources.rest_resource import RestResource

INVALID_DATATYPE_GID_MESSAGE = 'No datatype found for GID: {}'


class RetrieveDatatypeResource(Resource):

    def get(self, datatype_gid):
        """
        :given a guid, this function will download the H5 full data
        """
        index = ABCAdapter.load_entity_by_gid(datatype_gid)
        if index is None:
            raise InvalidIdentifierException(INVALID_DATATYPE_GID_MESSAGE.format(datatype_gid))
        h5_file = h5_file_for_index(index)
        last_index = h5_file.path.rfind('\\')
        file_name = h5_file.path[last_index + 1:]
        return flask.send_file(h5_file.path, as_attachment=True, attachment_filename=file_name)


class GetOperationsForDatatypeResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_service = FlowService()

    def get(self, datatype_gid):
        """
        :return the available operations for that datatype, as a list of Algorithm instances
        """
        categories = dao.get_launchable_categories()
        datatype = dao.get_datatype_by_gid(datatype_gid)
        if datatype is None:
            raise InvalidIdentifierException(INVALID_DATATYPE_GID_MESSAGE.format(datatype_gid))
        _, filtered_adapters = self.flow_service.get_launchable_algorithms_for_datatype(datatype, categories)
        return [AlgorithmDto(algorithm) for algorithm in filtered_adapters]
