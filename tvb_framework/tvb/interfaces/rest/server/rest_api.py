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
import json

from flask_restx import Api
from keycloak.exceptions import KeycloakError
from tvb.basic.exceptions import TVBException
from tvb.interfaces.rest.commons.status_codes import HTTP_STATUS_SERVER_ERROR


class RestApi(Api):
    def handle_error(self, e):
        if not isinstance(e, (TVBException, KeycloakError)):
            super().handle_error(e)

        # TVBException handling
        code = getattr(e, 'code', HTTP_STATUS_SERVER_ERROR)
        message = getattr(e, 'message', 'Internal Server Error')
        to_dict = getattr(e, 'to_dict', None)

        # KeycloakError handling
        code = getattr(e, 'response_code', code)
        error_message = getattr(e, 'error_message', None)

        if to_dict:
            data = to_dict()
        else:
            data = {'message': message}
        if error_message is not None:
            data['message'] = json.loads(error_message.decode())['error_description']
        return self.make_response(data, code)
