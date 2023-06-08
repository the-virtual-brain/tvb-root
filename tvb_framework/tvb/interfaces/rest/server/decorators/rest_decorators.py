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
from functools import wraps
from typing import Any

from flask import current_app, request
from flask.json import dumps
from keycloak.exceptions import KeycloakError
from tvb.basic.logger.builder import get_logger
from tvb.core.services.authorization import AuthorizationManager
from tvb.core.services.user_service import UserService
from tvb.interfaces.rest.commons.exceptions import AuthorizationRequestException
from tvb.interfaces.rest.commons.strings import Strings
from tvb.interfaces.rest.server.access_permissions.permissions import ResourceAccessPermission
from tvb.interfaces.rest.server.request_helper import set_current_user


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
            db_user = UserService().get_external_db_user(kc_user_info)
            set_current_user(db_user)

        except KeycloakError as kc_error:
            try:
                error_message = json.loads(kc_error.error_message.decode())['error_description']
            except (KeyError, TypeError):
                error_message = kc_error.error_message.decode()
            raise AuthorizationRequestException(message=error_message, code=kc_error.response_code)

        return func(*a, **b)

    return deco


def check_permission(resource_access_permission_class, requested_resource_identifier_name):
    # type: (ResourceAccessPermission.__subclasses__(), str) -> Any
    """
    Decorator which is used to check if the logged user has access to the requested resource
    :param resource_access_permission_class: ResourceAccessPermission subclass
    :param requested_resource_identifier_name: resource identifier parameter name
    :return: continue execution if user has access, raise an exception otherwise
    """

    def dec(func):
        @wraps(func)
        def deco(*a, **b):
            try:
                identifier = b[requested_resource_identifier_name]
                access_permission_instance = resource_access_permission_class(identifier)
                if access_permission_instance.has_access():
                    return func(*a, **b)
            except KeyError:
                get_logger().warning("Invalid identifier name")
            raise AuthorizationRequestException("You cannot access this resource")

        return deco

    return dec
