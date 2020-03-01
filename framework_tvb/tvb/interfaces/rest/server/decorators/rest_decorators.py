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
import json
from functools import wraps

from flask import current_app, request
from flask.json import dumps
from keycloak.exceptions import KeycloakError
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons import Strings
from tvb.interfaces.rest.commons.exceptions import AuthorizationRequestException
from tvb.interfaces.rest.server.security.authorization import AuthorizationManager, set_current_user


def _convert(obj):
    try:
        return obj.__dict__
    except AttributeError:
        return current_app.json_encoder().default(obj)


def rest_jsonify(func):
    @wraps(func)
    def deco(*a, **b):
        result = func(*a, **b)
        data = result
        status = 200
        if isinstance(result, tuple):
            data = result[0]
            status = result[1]
        if data is None:
            data = {}
        return current_app.response_class(dumps(data, default=lambda o: _convert(o), sort_keys=False),
                                          mimetype=current_app.config['JSONIFY_MIMETYPE'],
                                          status=status)

    return deco


def secured(func):
    @wraps(func)
    def deco(*a, **b):
        authorization = request.headers[Strings.AUTH_HEADER.value] if Strings.AUTH_HEADER.value in request.headers \
            else None
        if not authorization:
            raise AuthorizationRequestException()

        token = authorization.replace(Strings.BEARER.value, "")
        try:
            # Load user details
            kc_user_info = AuthorizationManager.get_keycloak_instance().userinfo(token)
            external_id = kc_user_info['sub']
            db_user = UserService.get_user_by_external_id(external_id)
            if db_user is None:
                db_user = UserService().create_external_service_user(kc_user_info)
            set_current_user(db_user)

        except KeycloakError as kc_error:
            try:
                error_message = json.loads(kc_error.error_message.decode())['error_description']
            except (KeyError, TypeError):
                error_message = kc_error.error_message.decode()
            raise AuthorizationRequestException(message=error_message, code=kc_error.response_code)

        return func(*a, **b)

    return deco
