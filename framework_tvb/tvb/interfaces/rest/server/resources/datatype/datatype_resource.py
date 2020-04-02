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

import flask
from tvb.interfaces.rest.server.access_permissions.permissions import DataTypeAccessPermission
from tvb.interfaces.rest.server.decorators.rest_decorators import check_permission
from tvb.interfaces.rest.server.facades.datatype_facade import DatatypeFacade
from tvb.interfaces.rest.server.resources.rest_resource import RestResource, SecuredResource


class RetrieveDatatypeResource(SecuredResource):

    @check_permission(DataTypeAccessPermission, 'datatype_gid')
    def get(self, datatype_gid):
        """
        :given a guid, this function will download the H5 full data
        """
        h5_file_path = DatatypeFacade.get_dt_h5_path(datatype_gid)
        file_name = os.path.basename(h5_file_path)
        return flask.send_file(h5_file_path, as_attachment=True, attachment_filename=file_name)


class GetOperationsForDatatypeResource(RestResource):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datatypes_facade = DatatypeFacade()

    @check_permission(DataTypeAccessPermission, 'datatype_gid')
    def get(self, datatype_gid):
        """
        :return the available operations for that datatype, as a list of Algorithm instances
        """
        return self.datatypes_facade.get_datatype_operations(datatype_gid)
