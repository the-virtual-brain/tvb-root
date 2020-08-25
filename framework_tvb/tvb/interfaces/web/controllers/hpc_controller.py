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

"""
REST endpoints that we use while running simulations on HPC nodes.

.. moduleauthor:: Paula Popa <paula.popa@codemart.ro>
.. moduleauthor:: Bogdan Valean <bogdan.valean@codemart.ro>
"""
import json
import os
from http import HTTPStatus

import cherrypy
from cherrypy.lib.static import serve_file
from tvb.basic.logger.builder import get_logger
from tvb.core.entities.model.model_operation import OperationPossibleStatus
from tvb.core.entities.storage import dao
from tvb.core.operation_hpc_launcher import UPDATE_STATUS_KEY
from tvb.core.services.encryption_handler import EncryptionHandler
from tvb.core.services.hpc_operation_service import HPCOperationService
from tvb.interfaces.web.controllers.autologging import traced
from tvb.interfaces.web.controllers.decorators import expose_endpoint


@traced
class HPCController(object):
    """
    Receive requests from simulation jobs that run on HPC nodes.
    """

    def __init__(self):
        self.logger = get_logger(self.__class__.__module__)

    def _validate_request_params(self, simulator_gid, operation_id):
        operation = dao.get_operation_by_id(operation_id)
        if not operation:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST, "Invalid operation id.")

        if not operation.view_model_gid:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST, "Invalid operation id.")

        if operation.view_model_gid != simulator_gid:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST, "Invalid simulator gid")

        return operation

    @expose_endpoint
    def update_status(self, simulator_gid, operation_id, **data):
        if cherrypy.request.method != 'PUT':
            raise cherrypy.HTTPError(HTTPStatus.METHOD_NOT_ALLOWED)

        if UPDATE_STATUS_KEY not in data:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST,
                                     "Invalid request. {} body param is missing.".format(UPDATE_STATUS_KEY))

        new_status = data[UPDATE_STATUS_KEY]
        if new_status not in OperationPossibleStatus:
            raise cherrypy.HTTPError(HTTPStatus.BAD_REQUEST,
                                     "Invalid status.")

        operation = self._validate_request_params(simulator_gid, operation_id)

        HPCOperationService.handle_hpc_status_changed(operation, simulator_gid, new_status)

    @expose_endpoint
    def encryption_config(self, simulator_gid, operation_id):
        self.logger.info("Received a request for passfile with gid: {}".format(simulator_gid))
        if cherrypy.request.method != 'GET':
            raise cherrypy.HTTPError(HTTPStatus.METHOD_NOT_ALLOWED)

        self._validate_request_params(simulator_gid, operation_id)

        encryption_handler = EncryptionHandler(simulator_gid)
        file_path = encryption_handler.get_password_file()

        return serve_file(file_path, "application/x-download", "attachment", os.path.basename(file_path))
